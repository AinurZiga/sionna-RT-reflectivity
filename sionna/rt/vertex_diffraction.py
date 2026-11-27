
from __future__ import annotations

import numpy as np
import mitsuba as mi
import drjit as dr
import tensorflow as tf
import trimesh
import time
from typing import Literal, Tuple, TYPE_CHECKING

from sionna.constants import SPEED_OF_LIGHT, PI
from sionna.utils.tensors import expand_to_rank, insert_dims, flatten_dims,\
    split_dim, insert_dim_num
from .paths import Paths, PathsTmpData
from .diffraction_funcs import calc_angles, calc_angles_concave, func_G_approximated_capolino, \
    t_gfi_full_approximated_capolino, fresnel_integral_capolino
from .utils import dot, phi_hat, theta_hat, theta_phi_from_unit_vec,\
    normalize, moller_trumbore, component_transform, mi_to_tf_tensor,\
        compute_field_unit_vectors, reflection_coefficient, slab_coefficients,\
        cot, cross, acos_diff, material_reflection_coefficients

if TYPE_CHECKING:
    from .solver_paths import SolverPaths
    from .paths import PathsTmpData
    from .double_diffraction import ObjectsGeometry


class VertexDiffraction:
    def __init__(self, solver_paths: SolverPaths):
        self.solver_paths = solver_paths
        self._dtype = self.solver_paths._dtype
        self._rdtype = self.solver_paths._rdtype

        self.is_setup = False
        self.use_stable_sb = False
        self.is_refl_vd = False

        # [num_primitives, 3]
        self._primitives_2_wedges = self.solver_paths._primitives_2_wedges
    
    def set_objects_geometry(self, objects_geometry: ObjectsGeometry):
        self.obj_geom = objects_geometry
        self.solver_paths.obj_geom = objects_geometry
        self.is_setup = True

        solver = self.solver_paths
        self._wedges_origin = solver._wedges_origin
        self._wedges_e_hat = solver._wedges_e_hat
        self._wedges_length = solver._wedges_length
        self._wedges_normals = solver._wedges_normals
        self._is_concave_wedge = solver._is_concave_wedge
        
        self._vertices = self.obj_geom._vertices
        self._num_vertices = self._vertices.shape[0]
        self.vertices_2_wedges = self.obj_geom.vertex_2_wedges
        self._objects_2_primitives = self.obj_geom._objects_2_primitives
        self.vertices_2_objects = self.obj_geom.vertices_2_objects
        self.vertices_2_objects_sionna = self.obj_geom.vertices_2_objects_sionna
        
        if self._num_vertices == 0:
            print("No objects!!")
            return 
        
        res = self._process_map()

        # [num_edges, 3]
        self._vertex_edges_origin = res[0]
        self._vertex_edges_e_hat = res[1]
        # [num_edges]
        self._vertex_edges_length = res[2]
        # [num_edges, 2, 3]
        self._vertex_edge_normals = res[3]
        # [num_edges]
        self._vertex_edges = res[4]   # wedge_2_vertices
        # [num_vertices]
        self._vertices_2_edges_row_lengths = res[5]
        self._num_vertex_edges = res[6]
        # [num_edges]
        self.vertex_wedges_2_objects = res[7]

    def _process_map(self):
        """
        Get edges related to the vertices.
        """

        vertex_wedges = self.vertices_2_wedges.values
        vertices_2_objects = self.vertices_2_objects_sionna
        vertices_2_wedges_row_lengths = self.vertices_2_wedges.row_lengths()
        num_vertex_wedges = vertex_wedges.shape[0]

        vertex_wedges_origin_tmp = tf.gather(self.solver_paths._wedges_origin, vertex_wedges)
        vertex_wedges_e_hat_tmp = tf.gather(self.solver_paths._wedges_e_hat, vertex_wedges)

        vertex_wedges_length = tf.gather(self.solver_paths._wedges_length, vertex_wedges)
        vertex_wedges_origin = tf.repeat(self._vertices, vertices_2_wedges_row_lengths, axis=0)
        vertex_wedge_normals = tf.gather(self.solver_paths._wedges_normals, vertex_wedges)
        vertex_wedges_2_objects = tf.repeat(vertices_2_objects, vertices_2_wedges_row_lengths)

        eps = 1e-6
        needs_swap = tf.reduce_all(
            tf.abs(vertex_wedges_origin_tmp - vertex_wedges_origin) < eps,
            axis=1
        )

        needs_swap = tf.expand_dims(needs_swap, axis=1)
        vertex_wedges_e_hat = tf.where(needs_swap, vertex_wedges_origin_tmp, vertex_wedges_e_hat_tmp)

        output = (
            vertex_wedges_origin,
            vertex_wedges_e_hat,
            vertex_wedges_length,
            vertex_wedge_normals,
            vertex_wedges,
            vertices_2_wedges_row_lengths,
            num_vertex_wedges,
            vertex_wedges_2_objects,
        )

        return output

    def tracing(self, sources, targets, case="both", is_dd=False, edge_diffraction=True,
                transmission=False, link_candidates=None, device_candidates=None, flat=True):
        """Trace ordinary VD using flat vertex paths and flat constituent edges.

        Path rows: [N, 3], columns [rx, tx, ObjectsGeometry vertex ID].
        Edge rows: [M], path-row IDs, wedge IDs and local edge slots.
        Only final Paths objects carry padded path and constituent-edge axes.
        Mixed reflection-VD tracing retains its virtual-source convention.
        """
        if is_dd or not flat:
            return self._tracing_legacy(
                sources, targets, case, is_dd, edge_diffraction, transmission, link_candidates
            )
        if case not in ("both", "tx", "rx"):
            raise ValueError(f"Unknown vertex diffraction case: {case!r}")
        solver = self.solver_paths
        paths = Paths(sources, targets, solver._scene, types=Paths.VERTEX_DIFFRACTED)
        tmp = PathsTmpData(sources, targets, self._dtype)
        tmp.junction_transmission = transmission
        rows = self._flat_vertex_candidates(sources, targets, link_candidates, device_candidates)

        # Expand the ragged incidence rows directly, without a max_vertex_edges axis.
        incidence = tf.gather(self.vertices_2_wedges, rows[:, 2])
        edge_paths = tf.cast(incidence.value_rowids(), tf.int32)
        wedges = tf.cast(incidence.values, tf.int32)
        lengths = tf.cast(incidence.row_lengths(), tf.int32)
        edge_slots = tf.range(tf.size(wedges)) - tf.repeat(
            tf.cast(incidence.row_splits[:-1], tf.int32), lengths
        )
        real_edges = wedges >= 0
        edge_paths, wedges, edge_slots = [
            tf.boolean_mask(x, real_edges) for x in (edge_paths, wedges, edge_slots)
        ]

        enabled = None
        if transmission and case == "both" and tf.size(solver._primitives_2_objects) > 0:
            size = tf.reduce_max(solver._primitives_2_objects) + 1
            _, enabled = solver._build_slab_properties_tensors(size)
        allowed = self._flat_vertex_edge_visibility(
            rows, edge_paths, wedges, sources, targets, case, edge_diffraction, enabled
        )
        edge_paths, wedges, edge_slots = [
            tf.boolean_mask(x, allowed) for x in (edge_paths, wedges, edge_slots)
        ]
        counts = tf.math.unsorted_segment_sum(
            tf.ones_like(edge_paths), edge_paths, tf.shape(rows)[0]
        )
        has_edge = counts > 0
        # Remove paths with no allowed constituent edge before tracing any rays.
        compact_ids = tf.cumsum(tf.cast(has_edge, tf.int32)) - 1
        edge_paths = tf.gather(compact_ids, edge_paths)
        rows = tf.boolean_mask(rows, has_edge)
        points = tf.gather(self._vertices, rows[:, 2])
        tx = tf.gather(sources, rows[:, 1])
        rx = tf.gather(targets, rows[:, 0])
        records = None
        if enabled is not None:
            ids = tf.range(tf.shape(rows)[0], dtype=tf.int32)[:, None]
            tracer = solver.solver_reflection._trace_transmission_segments
            blocked_in, incoming = tracer(tx, points, ids, enabled)
            blocked_out, outgoing = tracer(points, rx, ids, enabled)
            visible = ~(blocked_in | blocked_out)
            records = (
                tf.concat([incoming[0][:, 0], outgoing[0][:, 0]], axis=0),
                tf.concat([incoming[1], outgoing[1]], axis=0),
                tf.concat([incoming[2], outgoing[2]], axis=0),
                tf.concat([incoming[3], outgoing[3]], axis=0),
                tf.concat([tf.zeros_like(incoming[1]), tf.ones_like(outgoing[1])], axis=0),
            )
        else:
            visible = tf.ones([tf.shape(rows)[0]], tf.bool)
            if tf.shape(rows)[0] > 0:
                if case in ("both", "tx"):
                    direction, distance = normalize(points - tx)
                    visible &= ~solver._test_obstruction(tx, direction, distance)
                if case in ("both", "rx"):
                    direction, distance = normalize(points - rx)
                    visible &= ~solver._test_obstruction(rx, direction, distance)

        # VD keeps distinct vertices even when their departure/arrival angles match.
        paths, tmp, kept_ids, indices = solver.solver_wedge_diffraction._pack_flat_paths(
            paths, tmp, rows, points, visible, records,
            deduplicate=False, path_type=Paths.VERTEX_DIFFRACTED, return_indices=True,
        )
        mapping = tf.tensor_scatter_nd_update(
            tf.fill([tf.shape(rows)[0], 3], -1), kept_ids[:, None], indices
        )
        keep_edges = tf.gather(visible, edge_paths)
        edge_paths = tf.boolean_mask(edge_paths, keep_edges)
        edge_slots = tf.boolean_mask(edge_slots, keep_edges)
        edge_indices = tf.concat([tf.gather(mapping, edge_paths), edge_slots[:, None]], axis=1)
        max_edges = tf.reduce_max(tf.concat([
            tf.cast(self.vertices_2_wedges.row_lengths(), tf.int32), tf.zeros([1], tf.int32)
        ], axis=0))
        paths.vertex_wedge_mask = tf.scatter_nd(
            edge_indices, tf.ones_like(edge_paths, tf.bool),
            tf.concat([tf.shape(paths.mask), [max_edges]], axis=0),
        )
        return paths, tmp

    def _flat_vertex_candidates(self, sources, targets, link_candidates, device_candidates):
        """Return only actual [rx, tx, vertex] rows in per-link candidate order."""
        if device_candidates is not None:
            tx_rows, rx_rows = device_candidates
            vertices, slots = tf.unique(tf.concat([tx_rows[:, 1], rx_rows[:, 1]], axis=0))
            n_tx = tf.shape(tx_rows)[0]
            tx_rows = tf.stack([tx_rows[:, 0], slots[:n_tx]], axis=1)
            rx_rows = tf.stack([rx_rows[:, 0], slots[n_tx:]], axis=1)
            # The join is index-based and works for vertex IDs as well as wedge IDs.
            return self.solver_paths.solver_wedge_diffraction._join_device_wedges(
                tx_rows, rx_rows, vertices, tf.shape(sources)[0]
            )
        if link_candidates is not None:
            vertices, mask = link_candidates
            hits = tf.cast(tf.where(mask), tf.int32)
            return tf.stack([hits[:, 0], hits[:, 1], tf.gather(vertices, hits[:, 2])], axis=1)
        vertices = tf.cast(tf.where(self.obj_geom.active_vertices)[:, 0], tf.int32)
        n_vertices = tf.size(vertices)
        num_tx = tf.shape(sources)[0]
        count = tf.shape(targets)[0] * num_tx * n_vertices
        ids = tf.range(count, dtype=tf.int32)
        # Safe divisors also cover empty scenes and zero-device batches.
        vertex_slots = ids % tf.maximum(n_vertices, 1)
        links = ids // tf.maximum(n_vertices, 1)
        return tf.stack([
            links // tf.maximum(num_tx, 1), links % tf.maximum(num_tx, 1),
            tf.gather(vertices, vertex_slots),
        ], axis=1)

    def _flat_vertex_edge_visibility(self, rows, edge_paths, wedges, sources, targets,
                                     case, edge_diffraction, enabled):
        """Apply the existing sector, planar-edge and junction rules to [M] edges."""
        path_rows = tf.gather(rows, edge_paths)
        tx = tf.gather(sources, path_rows[:, 1])
        rx = tf.gather(targets, path_rows[:, 0])
        return self._vertex_edge_sector_mask(wedges, tx, rx, case, edge_diffraction, enabled)

    def _vertex_edge_sector_mask(self, wedges, tx, rx, case, edge_diffraction, enabled):
        """Test each edge against the endpoints of its incident and outgoing rays."""
        solver = self.solver_paths
        origins = tf.gather(solver._wedges_origin, wedges)
        normals = tf.gather(solver._wedges_normals, wedges)
        d_tx = dot((tx - origins)[:, None, :], normals)
        d_rx = dot((rx - origins)[:, None, :], normals)
        eps = tf.cast(solver.EPSILON, d_tx.dtype)
        allowed = tf.ones(tf.shape(wedges), tf.bool)
        if case in ("both", "tx"):
            allowed &= tf.reduce_any(d_tx > eps, axis=1)
        if case in ("both", "rx"):
            allowed &= tf.reduce_any(d_rx > eps, axis=1)
        wd = solver.solver_wedge_diffraction
        flat = wd._is_flat_shared_wedge(wedges)
        # Keep the real n=1 junction on both reflecting sides. Only the
        # existing crossed-sheet branch may introduce knife-edge transmission.
        if case == "both":
            allowed = tf.where(flat, wd._same_sides(d_tx[:, 0], d_rx[:, 0]), allowed)
        elif case == "tx":
            allowed = tf.where(flat, tf.abs(d_tx[:, 0]) > eps, allowed)
        else:
            allowed = tf.where(flat, tf.abs(d_rx[:, 0]) > eps, allowed)
        if enabled is not None:
            crossed = flat & wd._opposite_sides(d_tx[:, 0], d_rx[:, 0])
            face_flags = tf.gather(enabled, tf.gather(solver._wedges_objects, wedges))
            allowed = tf.where(crossed, face_flags[:, 0] != face_flags[:, 1], allowed)
        is_edge = tf.gather(solver._is_edge, wedges)
        return (allowed | is_edge) if edge_diffraction else (allowed & ~is_edge)

    def _mixed_vertex_edge_rows(self, vertex_ids, sources, targets, edge_diffraction):
        """Return [M] path row IDs and [M] wedge IDs for N physical vertex interactions."""
        incidence = tf.gather(self.vertices_2_wedges, vertex_ids)
        path_rows = tf.cast(incidence.value_rowids(), tf.int32)
        wedges = tf.cast(incidence.values, tf.int32)
        valid = wedges >= 0
        path_rows, wedges = [tf.boolean_mask(x, valid) for x in (path_rows, wedges)]
        allowed = self._vertex_edge_sector_mask(
            wedges, tf.gather(sources, path_rows), tf.gather(targets, path_rows),
            "both", edge_diffraction, enabled=None,
        )
        return tf.boolean_mask(path_rows, allowed), tf.boolean_mask(wedges, allowed)

    def _mixed_vertex_matrices(self, relative_permittivity, scattering_coefficient,
                               vertex_ids, points, sources, targets, s_in, s_out,
                               theta_t, phi_t, theta_r, phi_r, edge_diffraction):
        """Sum the adjacent-edge VD matrices for N valid mixed paths."""
        path_rows, wedges = self._mixed_vertex_edge_rows(
            vertex_ids, sources, targets, edge_diffraction
        )
        num_paths = tf.shape(vertex_ids)[0]
        if tf.size(wedges) == 0:
            return tf.zeros([num_paths, 2, 2], self._dtype)

        edge_points = tf.gather(points, path_rows)
        etas, scattering = self._vertex_edge_materials(
            relative_permittivity, scattering_coefficient, wedges, edge_points
        )
        angles = [tf.gather(x, path_rows) for x in (theta_t, phi_t, theta_r, phi_r)]
        distances = tuple(tf.gather(x, path_rows) for x in (s_in, s_out))
        matrices = self._vertex_edge_matrices(
            wedges, edge_points, tf.gather(sources, path_rows), tf.gather(targets, path_rows),
            etas, scattering, *angles, propagation_distances=distances,
        )
        return tf.math.unsorted_segment_sum(matrices, path_rows, num_paths)

    def _tracing_legacy(self, sources, targets, case="both", is_dd=False, edge_diffraction=True,
                transmission=False, link_candidates=None):
        """
        Trace vertex diffracted paths between sources and targets.
        is_dd (double-diffraction) to get vertices for double-diffraction
        """

        vd_paths = Paths(sources=sources, targets=targets, scene=self.solver_paths._scene, types=Paths.VERTEX_DIFFRACTED)
        vd_paths_tmp = PathsTmpData(sources, targets, self.solver_paths._dtype)

        num_sources = tf.shape(sources)[0]
        num_targets = tf.shape(targets)[0]

        candidate_mask = None
        if link_candidates is None:
            active = self.obj_geom.active_vertices_dd if is_dd else self.obj_geom.active_vertices
            vertex_ids = tf.cast(tf.where(active)[:, 0], tf.int32)
        else:
            vertex_ids, candidate_mask = link_candidates

        # vertex_ids are rows of ObjectsGeometry, not global scene vertex IDs.
        vertices = tf.gather(self._vertices, vertex_ids)
        vertex_shape = [num_targets, num_sources, tf.shape(vertex_ids)[0]]
        diff_vertex_points = tf.broadcast_to(
            vertices[None, None, :, :], vertex_shape + [3]
        )
        vertex_indices = tf.broadcast_to(vertex_ids[None, None, :], vertex_shape)
        if candidate_mask is not None:
            vertex_indices = tf.where(candidate_mask, vertex_indices, -1)

        # Discard obstructed diffracted paths
        # Only check for vertex visibility if there is at least one vertex-candidate
        if vertex_indices.shape[2] > 0: # Number of diff. paths > 0
            # Discard obstructed paths
            if transmission and not is_dd and case == "both":
                vertex_indices = self.solver_paths.solver_reflection._trace_diffraction_transmissions(
                    vertex_indices, diff_vertex_points, sources, targets, vd_paths_tmp
                )
            elif link_candidates is not None:
                vertex_indices = self.solver_paths._check_env_first_order_visibility(
                    sources, targets, vertex_indices, diff_vertex_points, case
                )
            else:
                vertex_wedges = tf.gather(self.obj_geom.vertex_2_wedges, vertex_ids)
                vertex_wedges_indices = vertex_wedges.values
                vertices_2_edges_row_lengths = vertex_wedges.row_lengths()
                vertex_indices, vertex_wedges_indices, diff_vertex_points, valid_vertex_idxs = (
                    self._check_vertices_visibility(
                        targets, sources, vertex_indices, vertex_wedges_indices,
                        diff_vertex_points, vertices_2_edges_row_lengths, case,
                    )
                )
        
            vd_paths, vd_paths_tmp = \
                self._vertex_diffraction_create_paths(vd_paths, vd_paths_tmp, vertex_indices, \
                                                       diff_vertex_points)

            vd_paths.vertex_wedge_mask = self._visible_vertex_wedges(
                vd_paths, case, edge_diffraction=edge_diffraction,
                transmission=transmission, check_half_spaces=not is_dd,
            )
            vd_paths.mask &= tf.reduce_any(vd_paths.vertex_wedge_mask, axis=-1)
            vd_paths.objects = tf.where(vd_paths.mask[None, ...], vd_paths.objects, -1)

        return vd_paths, vd_paths_tmp

    def _visible_vertex_wedges(
        self, paths, case, edge_diffraction=True, transmission=False,
        check_half_spaces=True,
    ):
        """Return the constituent-edge mask [targets, sources, paths, edges]."""
        solver = self.solver_paths
        vertex_idxs = paths.objects[0]
        safe_vertices = tf.maximum(vertex_idxs, 0)
        wedge_table = self.vertices_2_wedges.to_tensor(default_value=-1)
        wedges = tf.gather(wedge_table, safe_vertices)
        valid = (wedges >= 0) & (vertex_idxs[..., None] >= 0) & paths.mask[..., None]

        # Keep the existing policy for mixed paths traced with virtual sources.
        if not check_half_spaces and edge_diffraction:
            return valid

        # Only check real constituent edges of surviving vertex paths.
        rows = tf.cast(tf.where(valid), tf.int32)  # [N, 4]: [rx, tx, path, edge slot]
        if tf.shape(rows)[0] == 0:
            return valid

        wedge_ids = tf.gather_nd(wedges, rows)
        is_edge = tf.gather(solver._is_edge, wedge_ids)
        if not check_half_spaces:
            return tf.scatter_nd(rows, ~is_edge, tf.shape(valid))

        origins = tf.gather(solver._wedges_origin, wedge_ids)
        normals = tf.gather(solver._wedges_normals, wedge_ids)
        tx = tf.gather(paths.sources, rows[:, 1])
        rx = tf.gather(paths.targets, rows[:, 0])
        d_tx = dot((tx - origins)[:, None, :], normals)  # [N, 2]
        d_rx = dot((rx - origins)[:, None, :], normals)  # [N, 2]
        eps = tf.cast(solver.EPSILON, d_tx.dtype)

        allowed = tf.ones_like(is_edge)
        if case in ("both", "tx"):
            allowed &= tf.reduce_any(d_tx > eps, axis=1)
        if case in ("both", "rx"):
            allowed &= tf.reduce_any(d_rx > eps, axis=1)
        if case not in ("both", "tx", "rx"):
            raise ValueError(f"Unknown vertex diffraction case: {case!r}")

        if transmission and case == "both":
            # Apply the same flat-junction rule as ordinary wedge diffraction.
            wd = solver.solver_wedge_diffraction
            crossed_flat = wd._is_flat_shared_wedge(wedge_ids) & wd._opposite_sides(
                d_tx[:, 0], d_rx[:, 0]
            )
            array_size = tf.reduce_max(solver._primitives_2_objects) + 1
            _, enabled = solver._build_slab_properties_tensors(array_size)
            face_objects = tf.gather(solver._wedges_objects, wedge_ids)
            face_enabled = tf.gather(enabled, face_objects)
            mixed = face_enabled[:, 0] != face_enabled[:, 1]
            allowed = tf.where(crossed_flat, mixed, allowed)

        allowed = allowed | is_edge if edge_diffraction else allowed & ~is_edge
        return tf.scatter_nd(rows, allowed, tf.shape(valid))

    def _check_vertices_visibility(self, targets, sources, vertices_indices,
                                vertex_wedges_indices, vertices, vertices_2_edges_row_lengths, case='both'):
        r"""
        Discard the wedges that are not valid due to obstruction by updating the
        mask and removing the wedges related to no valid links.

        Input
        ------
        targets : [num_targets, 3], tf.float
            Coordinates of the targets.

        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        vertices_indices : [num_targets, num_sources, max_num_paths], tf.int
            Indices of the vertices that interacted with the diffracted paths

        vertex_wedges_indices : [num_wegdes_in_vertices], tf.int

        vertices : [num_targets, num_sources, max_num_paths, 3], tf.float
            Coordinates of the interaction points on the intersected wedges

        case : 'tx', 'rx', 'both'
            Case to consider:
            - 'tx' : Check visibility between TX and diffraction point
            - 'rx' : Check visibility between RX and diffraction point
            - 'both' : Check visibility between TX and diffraction point and
                        between RX and diffraction point

        Output
        -------
        vertices_indices : [num_targets, num_sources, num_vertices], tf.int
            Indices of the wedges that interacted with the diffracted paths

        vertex_wedges_indices : [num_targets, num_sources, max_num_paths], tf.int

        vertices : [num_targets, num_sources, num_vertices, 3], tf.float
            Coordinates of the interaction points on the intersected wedges

        valid : [num_targets, num_sources, max_num_paths], tf.bool
            Mask of valid paths 
        """

        max_num_paths = vertices.shape[2]  # number of vertices
        num_sources = sources.shape[0]
        num_targets = targets.shape[0]

        # Broadcast sources and targets with wedges diffraction point
        # [1, num_sources, 1, 3]
        sources = tf.expand_dims(sources, axis=0)
        sources = tf.expand_dims(sources, axis=2)
        # [num_targets, num_sources, max_num_paths, 3]
        sources = tf.broadcast_to(sources, vertices.shape)
        # Flatten
        # batch_size = num_targets*num_sources*max_num_paths
        # [batch_size, 3]
        sources = tf.reshape(sources, [-1, 3])
        # [num_targets, 1, 1, 3]
        targets = expand_to_rank(targets, tf.rank(vertices), 1)
        # [num_targets, num_sources, max_num_paths, 3]
        targets = tf.broadcast_to(targets, vertices.shape)
        # Flatten
        # [batch_size, 3]
        targets = tf.reshape(targets, [-1, 3])

        # Flatten interaction points
        # [batch_size, 3]
        vertex_points = tf.reshape(vertices, [-1, 3])

        if case == 'both' or case == 'tx':
            # Check visibility between transmitter and wedge
            # Ray origin
            # d : [batch_size, 3]
            # maxt : [batch_size]
            d,maxt = tf.linalg.normalize(vertex_points - sources, axis=1)
            maxt = tf.squeeze(maxt, axis=1)
            # [batch_size]
            valid_t2w = tf.logical_not(self.solver_paths._test_obstruction(sources, d, maxt))
            valid = valid_t2w

        if case == 'both' or case == 'rx':
            # Check visibility between wedge and receiver
            # Ray origin
            # d : [batch_size, 3]
            # maxt : [batch_size]
            d,maxt = tf.linalg.normalize(vertex_points - targets, axis=1)
            maxt = tf.squeeze(maxt, axis=1)
            # [batch_size]
            valid_w2r = tf.logical_not(self.solver_paths._test_obstruction(targets, d, maxt))
            valid = valid_w2r

        if case == 'both':
            # [batch_size]
            valid = tf.logical_and(valid_t2w, valid_w2r)

        # [num_targets, num_sources, max_num_paths]
        valid = tf.reshape(valid, [num_targets, num_sources, max_num_paths])
        # Set wedge indices of blocked paths to -1
        vertices_indices = tf.where(valid, vertices_indices, -1)

        # Discard wedges not involved in any link
        # [max_num_paths]
        used_vertices = tf.where(tf.reduce_any(tf.not_equal(vertices_indices, -1),
                                             axis=(0,1)))[:,0]
        # [num_targets, num_sources, max_num_paths, 3]
        vertices = tf.gather(vertices, used_vertices, axis=2)
        # [num_targets, num_sources, max_num_paths]
        vertices_indices = tf.gather(vertices_indices, used_vertices, axis=2)     

        valid_vertex_wedges = tf.repeat(valid, vertices_2_edges_row_lengths, axis=2)
        vertex_wedges_indices = tf.where(valid_vertex_wedges, vertex_wedges_indices, -1)
        used_vertex_wedges = tf.where(tf.reduce_any(tf.not_equal(vertex_wedges_indices, -1),
                                             axis=(0,1)))[:,0]
        vertex_wedges_indices = tf.gather(vertex_wedges_indices, used_vertex_wedges, axis=2) 

        return vertices_indices, vertex_wedges_indices, vertices, valid
    
    def _check_visibility(self, targets, sources, vertices, case='both'):
        r"""
        Discard the wedges that are not valid due to obstruction by updating the
        mask and removing the wedges related to no valid links.

        Input
        ------
        targets : [num_targets, 3], tf.float
            Coordinates of the targets.

        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        vertices : [num_targets, num_sources, max_num_paths, 3], tf.float
            Coordinates of the interaction points on the intersected wedges

        case : 'tx', 'rx', 'both'
            Case to consider:
            - 'tx' : Check visibility between TX and diffraction point
            - 'rx' : Check visibility between RX and diffraction point
            - 'both' : Check visibility between TX and diffraction point and
                        between RX and diffraction point

        Output
        -------

        valid : [num_targets, num_sources, max_num_paths], tf.bool
            Mask of valid paths 
        """

        max_num_paths = vertices.shape[2]  # number of vertices
        num_sources = sources.shape[0]
        num_targets = targets.shape[0]

        # Broadcast sources and targets with wedges diffraction point
        # [1, num_sources, 1, 3]
        sources = tf.expand_dims(sources, axis=0)
        sources = tf.expand_dims(sources, axis=2)
        # [num_targets, num_sources, max_num_paths, 3]
        sources = tf.broadcast_to(sources, vertices.shape)
        # Flatten
        # batch_size = num_targets*num_sources*max_num_paths
        # [batch_size, 3]
        sources = tf.reshape(sources, [-1, 3])
        # [num_targets, 1, 1, 3]
        targets = expand_to_rank(targets, tf.rank(vertices), 1)
        # [num_targets, num_sources, max_num_paths, 3]
        targets = tf.broadcast_to(targets, vertices.shape)
        # Flatten
        # [batch_size, 3]
        targets = tf.reshape(targets, [-1, 3])

        # Flatten interaction points
        # [batch_size, 3]
        vertex_points = tf.reshape(vertices, [-1, 3])

        if case == 'both' or case == 'tx':
            # Check visibility between transmitter and wedge
            # Ray origin
            # d : [batch_size, 3]
            # maxt : [batch_size]
            d,maxt = tf.linalg.normalize(vertex_points - sources, axis=1)
            maxt = tf.squeeze(maxt, axis=1)
            # [batch_size]
            valid_t2w = tf.logical_not(self.solver_paths._test_obstruction(sources, d, maxt))
            valid = valid_t2w

        if case == 'both' or case == 'rx':
            # Check visibility between wedge and receiver
            # Ray origin
            # d : [batch_size, 3]
            # maxt : [batch_size]
            d,maxt = tf.linalg.normalize(vertex_points - targets, axis=1)
            maxt = tf.squeeze(maxt, axis=1)
            # [batch_size]
            valid_w2r = tf.logical_not(self.solver_paths._test_obstruction(targets, d, maxt))
            valid = valid_w2r

        if case == 'both':
            # [batch_size]
            valid = tf.logical_and(valid_t2w, valid_w2r)

        # [num_targets, num_sources, max_num_paths]
        valid = tf.reshape(valid, [num_targets, num_sources, max_num_paths])

        return valid
    
    def _discard_nonvisible(self, vertex_indices, targets):
        r"""
        Discard the wedges that are not valid due to obstruction by updating the
        mask and removing the wedges related to no valid links.

        Input
        ------
        targets : [num_targets, 3], tf.float
            Coordinates of the targets or any points.

        vertex_indices : [num_candidate_vertices], tf.int
            Indices of the vertices that interacted with the diffracted paths

        Output
        -------

        valid : [num_targets, num_candidate_vertices, 3], tf.bool
            Mask of valid paths
            
        """

        num_candidate_vertices = vertex_indices.shape[0] 
        num_targets = targets.shape[0]

        # [num_targets, num_candidate_vertices, 3]
        targets = tf.expand_dims(targets, axis=1)
        targets = tf.broadcast_to(targets, [num_targets, num_candidate_vertices, 3])
        # [batch_size, 3]
        targets = tf.reshape(targets, [-1, 3])

        # [num_candidate_vertices, 3]
        vertex_points = tf.gather(self._vertices, vertex_indices)
        # [num_targets, num_candidate_vertices, 3]
        vertex_points = vertex_points[None, ...]
        vertex_points = tf.broadcast_to(vertex_points, [num_targets, num_candidate_vertices, 3])
        # [batch_size, 3]
        vertex_points = tf.reshape(vertex_points, [-1, 3])

        # Check visibility between vertices and targets
        # Ray origin
        # d : [batch_size, 3]
        # maxt : [batch_size]
        d, maxt = tf.linalg.normalize(vertex_points - targets, axis=1)
        maxt = tf.squeeze(maxt, axis=1)
        # [batch_size]
        valid = tf.logical_not(self.solver_paths._test_obstruction(targets, d, maxt))
        # [num_targets, num_candidate_vertices]
        valid = tf.reshape(valid, [num_targets, num_candidate_vertices])

        return valid
    
    def _vertex_diffraction_create_paths(self, diff_paths, diff_paths_tmp, vertex_indices, diff_vertex_points):

        diff_paths.objects = tf.expand_dims(vertex_indices, axis=0)
        diff_paths.vertices = tf.expand_dims(diff_vertex_points, axis=0)

        diff_paths = self._gather_valid_diff_paths(diff_paths)

        diff_paths, diff_paths_tmp =\
                self.solver_paths._compute_directions_distances_delays_angles(diff_paths,
                                                        diff_paths_tmp, False, True)
        
        return diff_paths, diff_paths_tmp
    
    
    def _gather_valid_diff_paths(self, paths: Paths):
        r"""
        Extracts only valid diffracted paths to reduce memory consumption when
        having multiple links with different number of valid paths.

        Input
        ------
        paths : :class:`~sionna.rt.Paths`
            Paths to update

        Output
        ------
        paths : :class:`~sionna.rt.Paths`
            Updated paths
        """

        # [num_targets, num_sources, max_num_candidates]
        vertex_indices = paths.objects[0]
        # [num_targets, num_sources, max_num_candidates, 3]
        vertex_points = paths.vertices[0]
        # [num_sources, 3]
        sources = paths.sources
        # [num_targets, 3]
        targets = paths.targets

        num_sources = tf.shape(sources)[0]
        num_targets = tf.shape(targets)[0]

        # [num_targets, num_sources, max_num_candidates]
        valid = tf.not_equal(vertex_indices, -1)

        # [num_targets, num_sources]
        num_paths = tf.reduce_sum(tf.cast(valid, tf.int32), axis=-1)
        max_num_paths = tf.reduce_max(num_paths)

        # Build indices for keeping only valid path
        # [num_valid_paths, 3]
        gather_indices = tf.where(valid)
        # [num_targets, num_sources, max_num_candidates]
        path_indices = tf.cumsum(tf.cast(valid, tf.int32), axis=-1)
        # [num_valid_paths]
        path_indices = tf.gather_nd(path_indices, gather_indices) - 1
        scatter_indices = tf.transpose(gather_indices, [1,0])
        if not tf.size(scatter_indices) == 0:
            scatter_indices = tf.tensor_scatter_nd_update(scatter_indices,
                                [[2]], [path_indices])
        # [num_valid_paths, 3]
        scatter_indices = tf.transpose(scatter_indices, [1,0])

        # Mask of valid paths
        # [num_targets, num_sources, max_num_paths]
        mask = tf.fill([num_targets, num_sources, max_num_paths], False)
        mask = tf.tensor_scatter_nd_update(mask, scatter_indices,
                tf.fill([tf.shape(scatter_indices)[0]], True))

        # Locations of the interactions
        # [num_targets, num_sources, max_num_paths, 3]
        valid_vertex_points = tf.zeros([num_targets, num_sources, max_num_paths, 3],
                                  dtype=self._rdtype)
        # [total_num_valid_paths, 3]
        vertex_points = tf.gather_nd(vertex_points, gather_indices)
        valid_vertex_points = tf.tensor_scatter_nd_update(valid_vertex_points,
                                                     scatter_indices, vertex_points)

        # Intersected wedges
        # [num_targets, num_sources, max_num_paths]
        valid_vertex_indices = tf.fill([num_targets, num_sources,
                                        max_num_paths], -1)
        # [total_num_valid_paths]
        vertex_indices = tf.gather_nd(vertex_indices, gather_indices)
        valid_vertex_indices = tf.tensor_scatter_nd_update(valid_vertex_indices,
                                            scatter_indices, vertex_indices)

        # [1, num_targets, num_sources, max_num_candidates]
        paths.objects = tf.expand_dims(valid_vertex_indices, axis=0)
        # [1, num_targets, num_sources, max_num_candidates, 3]
        paths.vertices = tf.expand_dims(valid_vertex_points, axis=0)
        # [num_targets, num_sources, max_num_candidates]
        paths.mask = mask

        return paths

    def compute_fields2(self, relative_permittivity, scattering_coefficient, paths, paths_tmp,
                        diff_paths=None, diff_paths_tmp=None, debug_resolve_edges=False):
        shape = tf.shape(paths.objects[0])  # [targets, sources, vertex_paths]
        indices = tf.cast(tf.where(paths.vertex_wedge_mask & paths.mask[..., None]), tf.int32)
        path_indices = indices[:, :3]

        vertex_idxs = tf.cast(tf.gather_nd(paths.objects[0], path_indices), tf.int32)
        offsets = tf.gather(tf.cast(self.vertices_2_wedges.row_splits, tf.int32), vertex_idxs)
        wedge_idxs = tf.gather(self.vertices_2_wedges.values, offsets + indices[:, 3])

        valid = wedge_idxs >= 0
        indices = tf.boolean_mask(indices, valid)
        wedge_idxs = tf.boolean_mask(wedge_idxs, valid)
        path_indices = indices[:, :3]

        if tf.shape(indices)[0] == 0:
            if debug_resolve_edges:
                return path_indices, tf.cast(wedge_idxs, tf.int32), tf.zeros([0, 2, 2], self._dtype)
            return tf.zeros(tf.concat([shape, [2, 2]], axis=0), dtype=self._dtype)

        ### flatten everything
        points = tf.gather_nd(paths.vertices[0], path_indices)
        sources = tf.gather(paths.sources, indices[:, 1])
        targets = tf.gather(paths.targets, indices[:, 0])
        theta_t = tf.gather_nd(paths.theta_t, path_indices)
        phi_t = tf.gather_nd(paths.phi_t, path_indices)
        theta_r = tf.gather_nd(paths.theta_r, path_indices)
        phi_r = tf.gather_nd(paths.phi_r, path_indices)

        wd = self.solver_paths.solver_wedge_diffraction
        crossed_flat = tf.zeros_like(wedge_idxs, dtype=tf.bool)
        transmitted = tf.zeros_like(wedge_idxs, dtype=tf.bool)

        if getattr(paths_tmp, "junction_transmission", False):
            origins = tf.gather(self.solver_paths._wedges_origin, wedge_idxs)
            normals = tf.gather(self.solver_paths._wedges_normals[:, 0], wedge_idxs)
            d_tx = dot(sources - origins, normals)
            d_rx = dot(targets - origins, normals)

            crossed_flat = wd._is_flat_shared_wedge(wedge_idxs) & wd._opposite_sides(d_tx, d_rx)

            face_objects = tf.gather(self.solver_paths._wedges_objects, wedge_idxs)
            enabled = tf.gather(self.solver_paths._object_transmission_enabled, face_objects)
            mixed = tf.math.logical_xor(enabled[:, 0], enabled[:, 1])
            transmitted = crossed_flat & mixed

        ordinary_rows = tf.where(~crossed_flat)
        transmitted_rows = tf.where(transmitted)
        edge_matrices = tf.zeros([tf.shape(wedge_idxs)[0], 2, 2], dtype=self._dtype)

        def compute_rows(rows, metal_knife_edge):
            selected_wedges = tf.gather_nd(wedge_idxs, rows)
            selected_points = tf.gather_nd(points, rows)
            sb_check = None
            if (getattr(self.solver_paths, "warn_sb_side_mismatch", False)
                    and diff_paths is not None
                    and getattr(diff_paths_tmp, "sb_wd_rows", None) is not None):
                selected_paths = tf.gather_nd(path_indices, rows)
                selected_vertices = tf.gather_nd(paths.objects[0], selected_paths)
                sb_check = (selected_paths[:, :2], selected_vertices, diff_paths, diff_paths_tmp)
            etas, scattering = self._vertex_edge_materials(
                relative_permittivity, scattering_coefficient, selected_wedges, selected_points
            )
            return self._vertex_edge_matrices(
                selected_wedges, selected_points,
                tf.gather_nd(sources, rows), tf.gather_nd(targets, rows),
                etas, scattering,
                tf.gather_nd(theta_t, rows), tf.gather_nd(phi_t, rows),
                tf.gather_nd(theta_r, rows), tf.gather_nd(phi_r, rows),
                metal_knife_edge=metal_knife_edge, sb_check=sb_check,
            )

        if tf.shape(ordinary_rows)[0] > 0:
            edge_matrices = tf.tensor_scatter_nd_update(
                edge_matrices, ordinary_rows, compute_rows(ordinary_rows, False)
            )

        if tf.shape(transmitted_rows)[0] > 0:
            edge_matrices = tf.tensor_scatter_nd_update(
                edge_matrices, transmitted_rows, compute_rows(transmitted_rows, True)
            )

        if debug_resolve_edges:
            # Each row keeps the incident spreading of its parent VD path.
            _, s_prime = normalize(points - sources)
            edge_matrices /= tf.cast(s_prime[:, None, None], self._dtype)
            return path_indices, tf.cast(wedge_idxs, tf.int32), edge_matrices

        path_ids = (path_indices[:, 0] * shape[1] + path_indices[:, 1]) * shape[2] + path_indices[:, 2]
        matrices = tf.math.unsorted_segment_sum(edge_matrices, path_ids, tf.reduce_prod(shape))
        matrices = tf.reshape(matrices, tf.concat([shape, [2, 2]], axis=0))

        _, s_prime = normalize(paths.vertices[0] - paths.sources[None, :, None, :])
        safe_s_prime = tf.where(paths.mask, s_prime, tf.ones_like(s_prime))
        matrices /= tf.cast(safe_s_prime[..., None, None], self._dtype)
        return tf.where(paths.mask[..., None, None], matrices, tf.zeros_like(matrices))


    def _vertex_edge_materials(self, relative_permittivity, scattering_coefficient, wedge_idxs, points):
        object_idxs = tf.gather(self.solver_paths._wedges_objects, wedge_idxs)
        rm_callable = self.solver_paths._scene.radio_material_callable

        if rm_callable is None:
            return tf.gather(relative_permittivity, object_idxs), tf.gather(scattering_coefficient, object_idxs)

        material_points = tf.repeat(points[:, None, :], 2, axis=1)
        etas, scattering, _ = rm_callable(object_idxs, material_points)
        return etas, scattering

    def _vertex_edge_matrices(self, wedge_idxs, points, sources, targets, etas, scattering,
                            theta_t, phi_t, theta_r, phi_r, metal_knife_edge=False,
                            sb_check=None, propagation_distances=None):
        wavelength = self.solver_paths._scene.wavelength
        k = 2.0 * PI / wavelength

        swap = self._vertex_edge_swap(wedge_idxs, points)
        e_hat, n_0_hat, n_n_hat, wedge_n = self._vertex_edge_geometry(
            wedge_idxs, points, swap=swap
        )

        object_idxs = tf.gather(self.solver_paths._wedges_objects, wedge_idxs)

        # Match material ordering to the endpoint orientation exactly once.
        etas = tf.where(swap[:, None], tf.reverse(etas, axis=[1]), etas)
        scattering = tf.where(swap[:, None], tf.reverse(scattering, axis=[1]), scattering)
        object_idxs = tf.where(swap[:, None], tf.reverse(object_idxs, axis=[1]), object_idxs)

        s_prime_hat, s_prime = normalize(points - sources)
        s_hat, s = normalize(targets - points)
        if propagation_distances is not None:
            # Reflection unfolds a leg's length but does not change the local ray directions.
            s_prime, s = propagation_distances

        incident_slab_coeffs = None

        if metal_knife_edge:
            enabled = tf.gather(self.solver_paths._object_transmission_enabled, object_idxs)
            glass_face = tf.cast(~enabled[:, 0], tf.int32)
            metal_face = 1 - glass_face

            glass_objects = tf.gather(object_idxs, glass_face, axis=1, batch_dims=1)
            metal_objects = tf.gather(object_idxs, metal_face, axis=1, batch_dims=1)
            glass_eta = tf.gather(etas, glass_face, axis=1, batch_dims=1)
            metal_eta = tf.gather(etas, metal_face, axis=1, batch_dims=1)
            metal_scattering = tf.gather(scattering, metal_face, axis=1, batch_dims=1)

            # Keep the outward vertex tangent and point cross(n_0_hat, e_hat)
            # into the metal half-plane.
            n_0_hat = tf.where((metal_face == 0)[:, None], n_0_hat, -n_0_hat)
            n_n_hat = -n_0_hat
            wedge_n = tf.fill(tf.shape(wedge_n), tf.cast(2.0, wedge_n.dtype))
            etas = tf.stack([metal_eta, metal_eta], axis=1)
            scattering = tf.stack([metal_scattering, metal_scattering], axis=1)
            object_idxs = tf.stack([metal_objects, metal_objects], axis=1)

            _, _, t_s, t_p = slab_coefficients(
                glass_eta, tf.abs(dot(s_prime_hat, n_0_hat, clip=True)),
                tf.gather(self.solver_paths._object_thickness, glass_objects),
                wavelength, dtype=self._dtype,
            )
            incident_slab_coeffs = (t_s, t_p)
        else:
            # Locate the source in the original oriented cross-sectional plane.
            t_0_hat = cross(n_0_hat, e_hat)
            source_phi = tf.math.floormod(
                tf.atan2(-dot(s_prime_hat, n_0_hat), -dot(s_prime_hat, t_0_hat)),
                2.0 * PI,
            )
            use_other_sector = source_phi > wedge_n * PI

            # Preserve the tangent pointing away from the vertex.
            # The complementary sector starts at the original n-face.
            original_n_0, original_n_n = n_0_hat, n_n_hat
            n_0_hat = tf.where(use_other_sector[:, None], -original_n_n, original_n_0)
            n_n_hat = tf.where(use_other_sector[:, None], -original_n_0, original_n_n)
            wedge_n = tf.where(use_other_sector, 2.0 - wedge_n, wedge_n)

            etas = tf.where(use_other_sector[:, None], tf.reverse(etas, axis=[1]), etas)
            scattering = tf.where(use_other_sector[:, None], tf.reverse(scattering, axis=[1]), scattering)
            object_idxs = tf.where(
                use_other_sector[:, None], tf.reverse(object_idxs, axis=[1]), object_idxs
            )

        phi_prime_hat, _ = normalize(cross(s_prime_hat, e_hat))
        beta_0_prime_hat = cross(phi_prime_hat, s_prime_hat)
        phi_hat_, _ = normalize(-cross(s_hat, e_hat))
        beta_0_hat = cross(phi_hat_, s_hat)

        phi_prime, phi, beta_prime, beta = calc_angles(n_0_hat, e_hat, s_prime_hat, s_hat)
        phi_prime_c, phi_c, _, _ = calc_angles_concave(
            n_0_hat, e_hat, s_prime_hat, s_hat, wedge_n
        )
        phi_prime = tf.where(wedge_n < 1.0, phi_prime_c, phi_prime)
        phi = tf.where(wedge_n < 1.0, phi_c, phi)

        if sb_check is not None:
            self._warn_wedge_sb_side(wedge_idxs, swap, beta_prime, beta, *sb_check)

        e_i_s_0, e_i_p_0, e_r_s_0, e_r_p_0 = compute_field_unit_vectors(
            s_prime_hat, s_hat, n_0_hat, self.solver_paths.EPSILON
        )
        e_i_s_n, e_i_p_n, e_r_s_n, e_r_p_n = compute_field_unit_vectors(
            s_prime_hat, s_hat, n_n_hat, self.solver_paths.EPSILON
        )

        # Use the full 3D incidence angle, matching ordinary WD.
        cos_theta_0 = tf.abs(dot(s_prime_hat, n_0_hat, clip=True))
        cos_theta_n = tf.abs(dot(s_prime_hat, n_n_hat, clip=True))
        face_cos_theta = tf.stack([cos_theta_0, cos_theta_n], axis=-1)

        # Object indices already follow both face permutations.
        face_thickness = tf.gather(self.solver_paths._object_thickness, object_idxs)
        face_slab_enabled = tf.gather(self.solver_paths._object_transmission_enabled, object_idxs)

        face_r_s, face_r_p = material_reflection_coefficients(
            etas, face_cos_theta, face_thickness, face_slab_enabled, wavelength
        )
        r_s_0, r_s_n = face_r_s[:, 0], face_r_s[:, 1]
        r_p_0, r_p_n = face_r_p[:, 0], face_r_p[:, 1]

        reduction_0 = tf.cast(tf.sqrt(1.0 - scattering[:, 0]**2), self._dtype)
        reduction_n = tf.cast(tf.sqrt(1.0 - scattering[:, 1]**2), self._dtype)
        r_s_0 *= reduction_0
        r_p_0 *= reduction_0
        r_s_n *= reduction_n
        r_p_n *= reduction_n

        w_i_0 = tf.cast(component_transform(phi_prime_hat, beta_0_prime_hat, e_i_s_0, e_i_p_0), self._dtype)
        w_r_0 = tf.cast(component_transform(e_r_s_0, e_r_p_0, phi_hat_, beta_0_hat), self._dtype)
        w_i_n = tf.cast(component_transform(phi_prime_hat, beta_0_prime_hat, e_i_s_n, e_i_p_n), self._dtype)
        w_r_n = tf.cast(component_transform(e_r_s_n, e_r_p_n, phi_hat_, beta_0_hat), self._dtype)
        r_0 = -tf.matmul(w_r_0, tf.stack([r_s_0, r_p_0], axis=-1)[..., None] * w_i_0)
        r_n = -tf.matmul(w_r_n, tf.stack([r_s_n, r_p_n], axis=-1)[..., None] * w_i_n)

        # cos_difference = -2.0 * tf.sin(0.5 * (beta_prime + beta)) * tf.sin(0.5 * (beta_prime - beta))
        # coef = -1.0 / tf.complex(tf.cast(0.0, self._rdtype), 2.0 * k * PI * cos_difference)
        
        step = (wedge_n * PI - phi + 1e-5 > 0.0) & (wedge_n * PI - phi_prime + 1e-5 > 0.0)
        # coef *= tf.cast(step, self._dtype)

        param_u = self._get_param_u(beta_prime, beta)
        param_b = tf.repeat(self._get_param_b(k, s_prime, s, beta_prime, beta)[:, None], 4, axis=-1)

        ell = s_prime * s / (s_prime + s) * tf.sin(beta_prime) * tf.sin(beta)
        ell = tf.maximum(ell, tf.zeros_like(ell))

        phi_m, phi_p = phi - phi_prime, phi + phi_prime
        param_a = self._get_tip_a_all(phi_m, phi_p, wedge_n, k, ell)

        # coefficients, func_B, (_, gfi) = self._get_coefs_D2(wedge_n, param_a, param_b, param_u, phi_prime, phi)
        if self.use_stable_sb:
            terms = self._vertex_sb_terms(
                k, s_prime, s, beta_prime, beta, wedge_n,
                param_a, param_b, param_u, phi_m, phi_p,
            )
            terms *= tf.cast(step[:, None], self._dtype)
            coefficients = tf.unstack(terms, num=4, axis=-1)
        else:
            cos_difference = (
                -2.0 * tf.sin(0.5 * (beta_prime + beta))
                * tf.sin(0.5 * (beta_prime - beta))
            )
            denominator = 2.0 * k * PI * cos_difference
            coef = -1.0 / tf.complex(tf.zeros_like(denominator), denominator)
            coef *= tf.cast(step, self._dtype)

            d, _, _ = self._get_coefs_D2(
                wedge_n, param_a, param_b, param_u, phi_prime, phi
            )
            coefficients = tuple(coef * value for value in d)

        d1, d2, d3, d4 = [self._clear_vertex_edge_nan(d) for d in coefficients]

        # d1, d2, d3, d4 = [self._clear_vertex_edge_nan(coef * d) for d in coefficients]

        eye = tf.eye(2, batch_shape=tf.shape(wedge_n), dtype=self._dtype)
        mat_t = (d1 + d2)[:, None, None] * eye + d3[:, None, None] * r_n + d4[:, None, None] * r_0
        mat_t *= -tf.cast((1.0 / s)[:, None, None], self._dtype)

        if incident_slab_coeffs is not None:
            t_s, t_p = incident_slab_coeffs
            mat_to_sp = tf.cast(component_transform(
                phi_prime_hat, beta_0_prime_hat, e_i_s_0, e_i_p_0
            ), self._dtype)
            mat_from_sp = tf.cast(component_transform(
                e_i_s_0, e_i_p_0, phi_prime_hat, beta_0_prime_hat
            ), self._dtype)
            transmission = tf.stack([t_s, t_p], axis=-1)
            glass = tf.matmul(mat_from_sp, transmission[..., None] * mat_to_sp)
            mat_t = tf.matmul(mat_t, glass)
        elif self.solver_paths.obj_geom is not None and self.solver_paths.obj_geom.is_glass_transmission:
            glass = self.solver_paths.obj_geom.diffraction_glass_transmission(
                valid_wedges_idx=wedge_idxs, n=wedge_n, mask=tf.ones_like(wedge_n, dtype=tf.bool),
                phi_prime_hat=phi_prime_hat, beta_0_prime_hat=beta_0_prime_hat,
                e_i_s_0=e_i_s_0, e_i_p_0=e_i_p_0, s_prime_hat=s_prime_hat,
                n_0_hat=n_0_hat, wavelength=wavelength, dtype=self._dtype,
            )
            if self.solver_paths.obj_geom.los_transmissions == "two":
                glass = tf.linalg.matmul(glass, glass)
            mat_t = tf.linalg.matmul(mat_t, glass)

        mat_from_gcs = tf.cast(component_transform(
            theta_hat(theta_t, phi_t), phi_hat(phi_t), phi_prime_hat, beta_0_prime_hat
        ), self._dtype)
        mat_to_gcs = tf.cast(component_transform(
            phi_hat_, beta_0_hat, theta_hat(theta_r, phi_r), phi_hat(phi_r)
        ), self._dtype)
        mat_t = tf.matmul(mat_to_gcs, tf.matmul(mat_t, mat_from_gcs))
        return self._clear_vertex_edge_nan(mat_t)

    @staticmethod
    def _lookup_sb_wedge_rows(query_rows, candidate_rows, num_sources, num_wedges):
        """Match [rx, tx, wedge] rows without a pairwise comparison tensor.

        Return one candidate-row index per query, or -1 if the candidate
        was not traced. Duplicate candidate keys have identical endpoint
        geometry, so the first matching row suffices for this diagnostic.
        """
        if tf.shape(candidate_rows)[0] == 0:
            return tf.fill([tf.shape(query_rows)[0]], tf.cast(-1, tf.int32))

        def keys(rows):
            rows = tf.cast(rows, tf.int64)
            return ((rows[:, 0] * tf.cast(num_sources, tf.int64) + rows[:, 1])
                    * tf.cast(num_wedges, tf.int64) + rows[:, 2])

        candidate_keys = keys(candidate_rows)
        query_keys = keys(query_rows)
        order = tf.argsort(candidate_keys, stable=True)
        sorted_keys = tf.gather(candidate_keys, order)
        positions = tf.searchsorted(sorted_keys, query_keys, side="left")
        safe_positions = tf.minimum(positions, tf.shape(sorted_keys)[0] - 1)
        found = ((positions < tf.shape(sorted_keys)[0])
                 & (tf.gather(sorted_keys, safe_positions) == query_keys))
        return tf.where(found, tf.gather(order, safe_positions), -tf.ones_like(positions))

    def _warn_wedge_sb_side(self, wedge_idxs, swap, beta_prime, beta,
                           link_indices, vertex_idxs, diff_paths, diff_paths_tmp):
        """Compare the VD cone side with WD's recorded finite-edge decision.

        Tangents used for beta point away from the current vertex. Hence
        beta_prime - beta > 0 predicts a stationary point on that half-line.
        Recorded WD coordinates use the original origin-to-end tangent.
        Only candidates actually evaluated by WD are compared. Occlusion,
        candidate omission and path deduplication are not side mismatches.
        The joint GO/WD/VD limit is not diagnosed by this endpoint check.
        """
        solver = self.solver_paths
        if not getattr(solver, "warn_sb_side_mismatch", False):
            return
        recorded_rows = getattr(diff_paths_tmp, "sb_wd_rows", None)
        if recorded_rows is None or tf.shape(recorded_rows)[0] == 0:
            return

        delta = beta_prime - beta
        # The axial case has no unique isolated WD stationary point.
        axial = ((tf.abs(tf.sin(beta)) <= tf.cast(1e-6, beta.dtype))
                 | (tf.abs(tf.sin(beta_prime)) <= tf.cast(1e-6, beta_prime.dtype)))
        near = tf.cast(tf.where(
            tf.math.is_finite(delta) & (tf.abs(delta) < solver.sb_warning_tol)
            & (delta != 0.0) & ~axial
        )[:, 0], tf.int32)
        if tf.size(near) == 0:
            return

        queries = tf.concat([
            tf.gather(link_indices, near),
            tf.cast(tf.gather(wedge_idxs, near)[:, None], link_indices.dtype),
        ], axis=-1)
        num_sources = tf.shape(diff_paths.sources)[0]
        num_wedges = tf.shape(solver._wedges_origin)[0]
        matched = self._lookup_sb_wedge_rows(queries, recorded_rows, num_sources, num_wedges)
        found = matched >= 0
        near = tf.boolean_mask(near, found)
        queries = tf.boolean_mask(queries, found)
        matched = tf.boolean_mask(matched, found)
        if tf.size(near) == 0:
            return

        t = tf.gather(diff_paths_tmp.sb_wd_stationary_t, matched)
        length = tf.cast(tf.gather(solver._wedges_length, queries[:, 2]), t.dtype)
        at_end = tf.gather(swap, near)
        from_vertex = tf.where(at_end, length - t, t)
        other_endpoint_margin = tf.where(at_end, t, length - t)
        expected = tf.gather(delta, near) > 0.0
        traced = from_vertex >= 0.0
        # Do not compare with a rejection caused by the OTHER endpoint.
        # Exact boundary values are left to a future one-sided convention.
        comparable = (tf.math.is_finite(from_vertex) & (from_vertex != 0.0)
                      & (length > 0.0) & (other_endpoint_margin >= 0.0))
        bad = tf.cast(tf.where(comparable & (expected != traced))[:, 0], tf.int32)
        if tf.size(bad) == 0:
            return

        sample = bad[:8]
        selected = tf.gather(near, sample)
        sample_queries = tf.gather(queries, sample)
        # Final presence is context only, not the criterion for a mismatch.
        final_indices = tf.cast(tf.where(
            diff_paths.mask & (diff_paths.objects[0] >= 0)
        ), tf.int32)
        final_rows = tf.concat([
            final_indices[:, :2],
            tf.cast(tf.gather_nd(diff_paths.objects[0], final_indices)[:, None], tf.int32),
        ], axis=-1)
        has_final_wd = self._lookup_sb_wedge_rows(
            sample_queries, final_rows, num_sources, num_wedges
        ) >= 0
        ids = tf.concat([
            tf.gather(link_indices, selected),
            tf.cast(tf.gather(vertex_idxs, selected)[:, None], link_indices.dtype),
            tf.cast(tf.gather(wedge_idxs, selected)[:, None], link_indices.dtype),
            tf.cast(tf.gather(swap, selected)[:, None], link_indices.dtype),
        ], axis=-1)
        tf.print(
            "WARNING: VD/WD shadow-boundary side mismatch; count =", tf.size(bad),
            "\n[RX, TX, vertex, wedge, endpoint (0=origin, 1=end)] =", ids,
            "\nbeta_prime - beta [rad] =", tf.gather(delta, selected),
            "\nWD stationary distance from vertex [m] =", tf.gather(from_vertex, sample),
            "\nVD expects inside =", tf.gather(expected, sample),
            "\nWD endpoint inside =", tf.gather(traced, sample),
            "\nWD finite-edge valid =", tf.gather(
                diff_paths_tmp.sb_wd_finite, tf.gather(matched, sample)
            ),
            "\nhas_final_wd =", has_final_wd,
        )

    def _vertex_sb_terms(self, k, s_prime, s, beta_prime, beta, n,
                         a, b, u, phi_m, phi_p):
        """Evaluate the four VD terms, including the outer prefactor."""
        joint_origin = (a == 0.0) & (b == 0.0)
        tf.debugging.assert_equal(
            tf.reduce_any(joint_origin), False,
            message="VD: a=b=0 requires a joint GO/WD/VD limit.",
        )

        h = tf.sin(0.5 * (beta - beta_prime))
        sin_mean = tf.sin(0.5 * (beta + beta_prime))
        K = tf.sqrt(2.0 * k * s_prime * s / (s_prime + s))
        t = K * h

        Phi = tf.stack(
            [PI + phi_m, PI - phi_m, PI + phi_p, PI - phi_p], axis=-1
        )
        B = self._get_func_B(Phi, u[:, None], n[:, None])

        small_a = (b > 0.0) & (a <= 1e-4 * tf.minimum(b, 1.0))
        stable = (b <= 1e-2) & (a > 0.0) & ~small_a
        stable_idx = tf.where(stable)
        regular_idx = tf.where(~stable)

        # Evaluate the legacy GFI only where b is strictly positive.
        regular_rows = regular_idx[:, 0]
        cos_diff = tf.gather(2.0 * sin_mean * h, regular_rows)
        denominator = 2.0 * k * PI * cos_diff
        coef = -1.0 / tf.complex(tf.zeros_like(denominator), denominator)
        T = t_gfi_full_approximated_capolino(
            tf.gather_nd(b, regular_idx),
            tf.gather_nd(a, regular_idx),
            self.obj_geom.diffraction_tables,
        )
        regular_values = coef * tf.cast(tf.gather_nd(B, regular_idx), self._dtype) * T

        stable_rows = stable_idx[:, 0]
        a_sb = tf.gather_nd(a, stable_idx)
        y = tf.sqrt(a_sb)
        t_sb = tf.gather(t, stable_rows)
        b_sb = tf.square(t_sb)

        # At t=0, select the positive one-sided value for now.
        side = tf.where(t_sb < 0.0, -tf.ones_like(t_sb), tf.ones_like(t_sb))
        alpha = tf.atan2(t_sb, y)
        Q = fresnel_integral_capolino(tf.complex(y, tf.zeros_like(y)), y)
        phase = tf.exp(tf.complex(tf.zeros_like(y), tf.fill(tf.shape(y), tf.cast(PI / 4.0, y.dtype))))
        G0 = phase * Q / tf.cast(2.0 * tf.sqrt(tf.cast(PI, y.dtype)), self._dtype)
        G_signed = tf.exp(tf.complex(tf.zeros_like(b_sb), b_sb)) * (
            tf.cast(side, self._dtype) * G0
            - tf.cast(alpha / (2.0 * PI), self._dtype)
            + tf.complex(tf.zeros_like(y), y * (t_sb - y * alpha) / (2.0 * PI))
        )

        W = (a_sb + b_sb) * tf.gather_nd(B, stable_idx) / y
        prefactor = -tf.gather(K / (k * sin_mean), stable_rows)
        stable_values = tf.cast(prefactor * W, self._dtype) * G_signed

        return tf.scatter_nd(
            tf.concat([regular_idx, stable_idx], axis=0),
            tf.concat([regular_values, stable_values], axis=0),
            tf.shape(a, out_type=tf.int64),
        )

    def _clear_vertex_edge_nan(self, value):
        return tf.where(tf.math.is_nan(tf.math.real(value)), tf.zeros_like(value), value)

    def simple_compute_fields(self, phi_prime, phi, beta_prime, beta, s_prime, s, wedge_n, mask_wedges_in_vertex_, 
                              add_to_r1=None, skip_sf=False, skip_F=False):
        r"""
        Simple function, doesn't involve materials (PEC assumed)
        """
        eps_phi = 1e-2
        wavelength = self.solver_paths._scene.wavelength
        k = 2.*PI/wavelength
        num_targets = phi_prime.shape[0]
        num_sources = phi_prime.shape[1]
        num_vertices = mask_wedges_in_vertex_.shape[2]

        # [num_targets, num_sources, max_num_paths]
        coef_ = -1 / tf.complex(tf.cast(0.0, dtype=self._rdtype), 
                               (2.0 * k * PI * (tf.math.cos(beta_prime) - tf.math.cos(beta))))

        step_func1 = tf.math.greater(wedge_n * PI - phi + eps_phi, 0.0)
        step_func2 = tf.math.greater(wedge_n * PI - phi_prime + eps_phi, 0.0)
        
        step_func = tf.logical_and(step_func1, step_func2)

        step_func = tf.cast(step_func, self._dtype)

        coef = coef_ * step_func

        # coef = coef_

        # if ell_edge is None:
        ell_edge = s_prime * s / (s_prime + s) * tf.math.sin(beta_prime) * tf.math.sin(beta)

        phi_m = phi - phi_prime
        phi_p = phi + phi_prime

        a_d1 = self._get_tip_a_plus(phi_m, wedge_n, k, ell_edge)
        a_d2 = self._get_tip_a_minus(phi_m, wedge_n, k, ell_edge)
        a_d3 = self._get_tip_a_plus(phi_p, wedge_n, k, ell_edge)
        a_d4 = self._get_tip_a_minus(phi_p, wedge_n, k, ell_edge)

        if add_to_r1 is not None:
            ell_vertex = (s_prime + add_to_r1) * s / (s_prime + s + add_to_r1)
        else:
            ell_vertex = s_prime * s / (s_prime + s)

        param_b = self._get_param_b_using_ell(k, ell_vertex, beta_prime, beta)
        param_b = tf.repeat(param_b[..., None], 4, axis=-1)

        # [num_targets, num_sources, max_num_paths, 4]
        param_a = tf.stack([a_d1, a_d2, a_d3, a_d4], axis=-1)

        ## due to numerical instability, a==0.0 may become negative, so clip it to 0.0
        param_a = tf.where(param_a < 0.0, tf.zeros_like(param_a), param_a)

        # [num_targets, num_sources, max_num_paths]
        param_u = self._get_param_u(beta_prime, beta)

        res = self._get_coefs_D_new(wedge_n, param_a, param_b, param_u, phi_prime, phi, coef, skip_F)
        d1_, d2_, d3_, d4_ = res

        # [num_targets, num_sources, max_num_paths, ...]
        d1 = _clear_diff_coef2(d1_, mask_wedges_in_vertex_, num_targets, num_sources, num_vertices, self._dtype)
        d2 = _clear_diff_coef2(d2_, mask_wedges_in_vertex_, num_targets, num_sources, num_vertices, self._dtype)
        d3 = _clear_diff_coef2(d3_, mask_wedges_in_vertex_, num_targets, num_sources, num_vertices, self._dtype)
        d4 = _clear_diff_coef2(d4_, mask_wedges_in_vertex_, num_targets, num_sources, num_vertices, self._dtype)

        if skip_sf:
            return (d1, d2, d3, d4), param_a, param_b
        
        spreading_factor = 1.0 / s
        spreading_factor = tf.complex(spreading_factor, tf.zeros_like(spreading_factor))

        d1 *= spreading_factor
        d2 *= spreading_factor
        d3 *= spreading_factor
        d4 *= spreading_factor

        return (d1, d2, d3, d4), param_a, param_b
        
    def compute_refl_vd(self,
                            relative_permittivity,
                            scattering_coefficient,
                            paths: Paths,
                            paths_tmp: PathsTmpData):
        # pylint: disable=line-too-long
        """
        Compute the transition matrices for diffracted rays.

        Input
        ------
        relative_permittivity : [num_shape], tf.complex
            Tensor containing the complex relative permittivity of all shape
            in the scene

        scattering_coefficient : [num_shape], tf.float
            Tensor containing the scattering coefficients of all shapes

        paths : :class:`~sionna.rt.Paths`
            Paths to update

        paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Addtional quantities required for paths computation

        Output
        ------
        paths : :class:`~sionna.rt.Paths`
            Updated paths
        """
        mask_21 = tf.reduce_all(tf.reduce_all(paths.dd_type == 21, axis=1), axis=0)
        mask_22 = tf.reduce_all(tf.reduce_all(paths.dd_type == 22, axis=1), axis=0)

        # [max_num_paths, num_sources, num_targets, 2, 3]
        mat_t = tf.zeros((paths.vertices[0].shape[2], paths.targets.shape[0], paths.sources.shape[0], 2, 2), dtype=self._dtype)

        ### 1) refl-vd
        paths_21 = paths.gather_paths(tf.where(mask_21)[:, 0], is_finalized=False)
        paths_tmp_21 = paths_tmp.gather_paths(tf.where(mask_21)[:, 0])
        if tf.reduce_any(mask_21):
            mat_t_21 = self._compute_refl_vd_21(relative_permittivity, scattering_coefficient, paths_21, paths_tmp_21)

        ### 2) vd-refl
        paths_22 = paths.gather_paths(tf.where(mask_22)[:, 0], is_finalized=False)
        paths_tmp_22 = paths_tmp.gather_paths(tf.where(mask_22)[:, 0])
        if tf.reduce_any(mask_22):
            mat_t_22 = self._compute_refl_vd_22(relative_permittivity, scattering_coefficient, paths_22, paths_tmp_22)

        if tf.reduce_any(mask_21):
            mat_t = tf.tensor_scatter_nd_update(mat_t, tf.where(mask_21), tf.transpose(mat_t_21, perm=[2, 0, 1, 3, 4]))
        if tf.reduce_any(mask_22):
            mat_t = tf.tensor_scatter_nd_update(mat_t, tf.where(mask_22), tf.transpose(mat_t_22, perm=[2, 0, 1, 3, 4]))

        mat_t = tf.transpose(mat_t, perm=[1, 2, 0, 3, 4])

        return mat_t
    
    def _compute_refl_vd_21(self, relative_permittivity, scattering_coefficient, paths: Paths, paths_tmp: PathsTmpData):
        mask = paths.mask
        objects = paths.objects
        targets = paths.targets
        sources = paths.sources
        theta_t = paths.theta_t
        phi_t = paths.phi_t
        theta_r = paths.theta_r
        phi_r = paths.phi_r

        num_targets = targets.shape[0]
        num_sources = sources.shape[0]
        num_vertices = paths.vertices[0].shape[2]

        _normals = paths_tmp.normals
        k_i = paths_tmp.k_i
        k_r = paths_tmp.k_r

        # [1, num_sources, 1, 3]
        sources = sources[None, :, None, :] 
        # [num_targets, 1, 1, 3]
        targets = targets[:, None, None, :]

        #normals = paths_tmp.normals

        wavelength = self.solver_paths._scene.wavelength
        k = 2.0*PI/wavelength

        #### TX - Refl point - Diff point - RX
        # [num_targets, num_sources, max_num_paths, 3]
        refl_points = tf.gather(paths.vertices, 0, axis=0)
        vd_points = tf.gather(paths.vertices, 1, axis=0)

        # [num_targets, num_sources, max_num_paths]
        refl_idxs = tf.gather(objects, 0, axis=0)
        vd_idxs = tf.gather(objects, 1, axis=0)
        valid_vd_idx = tf.where(vd_idxs == -1, 0, vd_idxs)
        
        #[max_num_paths, max_num_wedges_per_vertex]
        tensor_vertices_2_wedges = self.vertices_2_wedges.to_tensor(default_value=0)
        #[num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex]
        tensor_valid_vertices_2_wedges = tf.gather(tensor_vertices_2_wedges, valid_vd_idx)
        #[num_targets, num_sources, max_num_paths_extended]
        valid_vertex_wedges_idx = tf.reshape(tensor_valid_vertices_2_wedges, [num_targets, num_sources, -1])
        object_idxs_vd = tf.gather(self.solver_paths._wedges_objects, valid_vertex_wedges_idx, axis=0)

        r1_hat, r1 = normalize(refl_points - sources)
        l_hat, l = normalize(vd_points - refl_points)
        r2_hat, r2 = normalize(targets - vd_points)

        # Relative perimittivities and scattering coefficients.
        # If a callable is defined to compute the radio material properties,
        # it is invoked. Otherwise, the radio materials of objects are used.
        rm_callable = self.solver_paths._scene.radio_material_callable
        if rm_callable is None:
            # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
            # This makes no difference on the resulting paths as such paths
            # are not flagged as active.
            # [max_depth, num_targets, num_sources, max_num_paths]
            valid_refl_object_idx = tf.where(refl_idxs == -1, 0, refl_idxs)

            # [max_depth, num_targets, num_sources, max_num_paths]
            etas_refl = tf.gather(relative_permittivity, valid_refl_object_idx)
            scattering_coefficient_refl = tf.gather(scattering_coefficient, valid_refl_object_idx)
            etas_vd = tf.gather(relative_permittivity, object_idxs_vd)
            scattering_coefficient_vd = tf.gather(scattering_coefficient, object_idxs_vd)
        else:
            # [max_depth, num_targets, num_sources, max_num_paths]
            etas, scattering_coefficient, _  = rm_callable(objects, paths.vertices)

        #### 1) Reflection
        # [max_depth, num_targets, num_sources, max_num_paths]
        reduction_factor_refl = tf.sqrt(1 - scattering_coefficient_refl**2)
        reduction_factor_refl = tf.complex(reduction_factor_refl, tf.zeros_like(reduction_factor_refl))
        #_objects = objects[0]
        _k_t = k_i[0] # r1_hat
        _k_r = k_r[0] # l_hat
        #_reduction_factor = reduction_factor[0]
        #_etas = etas[0]
        mat_t_refl = self.solver_paths.solver_reflection._foo(refl_idxs, _k_t, _k_r, _normals, reduction_factor_refl, etas_refl)

        #### 2) VD
        ## TODO: Can be simplified?
        vertex_indices_mask = vd_idxs != -1

        #[max_num_paths, max_num_wedges_per_vertex]
        non_valid_tensor_vertices_2_wedges = self.vertices_2_wedges.to_tensor(default_value=-1)
        tensor_vertices_2_wedges = self.vertices_2_wedges.to_tensor(default_value=0)

        # [num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex]
        mask_wedges_in_vertex_ = tf.gather(non_valid_tensor_vertices_2_wedges, valid_vd_idx)
        mask_wedges_in_vertex_ = tf.not_equal(mask_wedges_in_vertex_, -1)
        mask_wedges_in_vertex_ = tf.logical_and(mask_wedges_in_vertex_, tf.expand_dims(vertex_indices_mask, -1))

        #[num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex, 2, 2]
        mask_wedges_in_vertex = tf.stack([mask_wedges_in_vertex_, mask_wedges_in_vertex_], axis=-1)
        mask_wedges_in_vertex = tf.stack([mask_wedges_in_vertex, mask_wedges_in_vertex], axis=-1)

        #[num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex]
        tensor_valid_vertices_2_wedges = tf.gather(tensor_vertices_2_wedges, valid_vd_idx)

        # [num_targets, num_sources, max_num_paths_extended, 3]
        vd_points_ext = tf.repeat(vd_points, tensor_valid_vertices_2_wedges.shape[3], axis=2)

        #[num_targets, num_sources, max_num_paths_extended]
        valid_vertex_wedges_idx = tf.reshape(tensor_valid_vertices_2_wedges, [num_targets, num_sources, -1])

        normals = tf.gather(self._wedges_normals, valid_vertex_wedges_idx, axis=0)

        # [num_targets, num_sources, max_num_paths]
        cos_wedges_angle = dot(normals[...,0,:], normals[...,1,:], clip=True)
        wedges_angle = PI - tf.math.acos(cos_wedges_angle)
        wedge_n = (2.*PI-wedges_angle)/PI

        is_concave_wedge = tf.gather(self._is_concave_wedge, valid_vertex_wedges_idx)
        wedge_n = tf.where(is_concave_wedge, wedges_angle / PI, wedge_n)

        # [num_targets, num_sources, max_num_paths_extended, 3]
        e_hat = tf.gather(self._wedges_e_hat, valid_vertex_wedges_idx)
        wedge_origins_tmp = tf.gather(self._wedges_origin, valid_vertex_wedges_idx)

        # For 1)
        # [num_targets, num_sources, max_num_paths_extended]
        needs_swap = tf.logical_not(tf.reduce_all(tf.math.equal(wedge_origins_tmp, vd_points_ext), axis=3))
        # [num_targets, num_sources, max_num_paths_extended, 3]
        needs_swap = tf.expand_dims(needs_swap, axis=3)
        e_hat = tf.where(needs_swap, -e_hat, e_hat)
        # [num_targets, num_sources, max_num_paths_extended, 2, 3]
        #normals = tf.where(tf.expand_dims(needs_swap, -1), -normals, normals) # (for 1)  # TODO

        # Reshape sources and targets
        # [1, num_sources, 1, 3]
        sources = tf.reshape(sources, [1, -1, 1, 3])
        # [num_targets, 1, 1, 3]
        targets = tf.reshape(targets, [-1, 1, 1, 3])

        n_0_hat = tf.where(needs_swap, normals[..., 1,:], normals[..., 0,:])  # TODO
        n_n_hat = tf.where(needs_swap, normals[..., 0,:], normals[..., 1,:])

        # s_prime_hat : [num_targets, num_sources, max_num_paths, 3]
        # s_prime : [num_targets, num_sources, max_num_paths]
        s_prime_hat, s_prime = l_hat, l + r1 #normalize(vd_points_ext - sources)
        # s_hat : [num_targets, num_sources, max_num_paths, 3]
        # s : [num_targets, num_sources, max_num_paths]
        s_hat, s = normalize(targets - vd_points_ext)

        _theta_t, _phi_t = theta_phi_from_unit_vec(k_i[1])

        s_prime_hat = tf.repeat(s_prime_hat, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        s_prime = tf.repeat(s_prime, tensor_valid_vertices_2_wedges.shape[3], axis=2)

        # v_phi_prime, v_phi, v_beta_prime, v_beta = calc_angles(n_0_hat, e_hat, s_prime_hat, s_hat)
        # v_phi_prime_, v_phi_, v_beta_prime, v_beta = calc_angles_(n_0_hat, e_hat, s_prime_hat, s_hat, wedge_n)
        # is_concave = wedge_n < 1.0
        # v_phi_prime = tf.where(is_concave, v_phi_prime_, v_phi_prime)
        # v_phi = tf.where(is_concave, v_phi_, v_phi)
        # mat_t_vd = self.simple_compute_fields(v_phi_prime, v_phi, v_beta_prime, v_beta, 
        #                         s_prime, s, wedge_n, tensor_valid_vertices_2_wedges, skip_sf=True,
        #                         theta_t=_theta_t, phi_t=_phi_t, theta_r=theta_r, phi_r=phi_r)

        mat_t_vd = self._compute_fields(s_prime_hat, s_hat, n_0_hat, n_n_hat, e_hat, etas_vd,
                                    scattering_coefficient_vd, s_prime, s, wedge_n, mask,
                                    mask_wedges_in_vertex_, mask_wedges_in_vertex,
                                    _theta_t, _phi_t, theta_r, phi_r, tensor_valid_vertices_2_wedges,
                                    vd_points_ext, valid_vertex_wedges_idx, skip_sf=True)
        
        ### 3) total field
        sf = 1 / ((r1+l) * r2)
        mat_t = tf.multiply(mat_t_refl, mat_t_vd) * tf.cast(sf, self.solver_paths._dtype)[..., None, None]
        #mat_t = mat_t_vd * tf.cast(sf, self.solver_paths._dtype)[..., None, None] # TODO

        return mat_t
    
    def _compute_refl_vd_22(self, relative_permittivity, scattering_coefficient, paths: Paths, paths_tmp: PathsTmpData):
        mask = paths.mask
        objects = paths.objects
        targets = paths.targets
        sources = paths.sources
        theta_t = paths.theta_t
        phi_t = paths.phi_t
        theta_r = paths.theta_r
        phi_r = paths.phi_r

        num_targets = targets.shape[0]
        num_sources = sources.shape[0]
        num_vertices = paths.vertices[0].shape[2]

        _normals = paths_tmp.normals
        k_i = paths_tmp.k_i
        k_r = paths_tmp.k_r

        # [1, num_sources, 1, 3]
        sources = sources[None, :, None, :] 
        # [num_targets, 1, 1, 3]
        targets = targets[:, None, None, :]

        #normals = paths_tmp.normals

        wavelength = self.solver_paths._scene.wavelength
        k = 2.0*PI/wavelength

        #### TX - VD_point - Refl_point - RX
        # [num_targets, num_sources, max_num_paths, 3]
        vd_points = tf.gather(paths.vertices, 0, axis=0)
        refl_points = tf.gather(paths.vertices, 1, axis=0)

        # [num_targets, num_sources, max_num_paths]
        vd_idxs = tf.gather(objects, 0, axis=0)
        refl_idxs = tf.gather(objects, 1, axis=0)
        
        valid_vd_idx = tf.where(vd_idxs == -1, 0, vd_idxs)

        #[max_num_paths, max_num_wedges_per_vertex]
        tensor_vertices_2_wedges = self.vertices_2_wedges.to_tensor(default_value=0)
        #[num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex]
        tensor_valid_vertices_2_wedges = tf.gather(tensor_vertices_2_wedges, valid_vd_idx)
        #[num_targets, num_sources, max_num_paths_extended]
        valid_vertex_wedges_idx = tf.reshape(tensor_valid_vertices_2_wedges, [num_targets, num_sources, -1])
        object_idxs_vd = tf.gather(self.solver_paths._wedges_objects, valid_vertex_wedges_idx, axis=0)

        # r1_hat, r1 = normalize(refl_points - sources)
        # l_hat, l = normalize(vd_points - refl_points)
        # r2_hat, r2 = normalize(targets - vd_points)

        r1_hat, r1 = normalize(vd_points - sources)
        l_hat, l = normalize(refl_points - vd_points)
        r2_hat, r2 = normalize(targets - refl_points)

        # Relative perimittivities and scattering coefficients.
        # If a callable is defined to compute the radio material properties,
        # it is invoked. Otherwise, the radio materials of objects are used.
        rm_callable = self.solver_paths._scene.radio_material_callable
        if rm_callable is None:
            # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
            # This makes no difference on the resulting paths as such paths
            # are not flagged as active.
            # [max_depth, num_targets, num_sources, max_num_paths]
            valid_refl_object_idx = tf.where(refl_idxs == -1, 0, refl_idxs)

            # [max_depth, num_targets, num_sources, max_num_paths]
            etas_refl = tf.gather(relative_permittivity, valid_refl_object_idx)
            scattering_coefficient_refl = tf.gather(scattering_coefficient, valid_refl_object_idx)
            etas_vd = tf.gather(relative_permittivity, object_idxs_vd)
            scattering_coefficient_vd = tf.gather(scattering_coefficient, object_idxs_vd)
        else:
            # [max_depth, num_targets, num_sources, max_num_paths]
            etas, scattering_coefficient, _  = rm_callable(objects, paths.vertices)

        #### 2) VD
        vertex_indices_mask = vd_idxs != -1

        #[max_num_paths, max_num_wedges_per_vertex]
        non_valid_tensor_vertices_2_wedges = self.vertices_2_wedges.to_tensor(default_value=-1)
        tensor_vertices_2_wedges = self.vertices_2_wedges.to_tensor(default_value=0)

        # [num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex]
        mask_wedges_in_vertex_ = tf.gather(non_valid_tensor_vertices_2_wedges, valid_vd_idx)
        mask_wedges_in_vertex_ = tf.not_equal(mask_wedges_in_vertex_, -1)
        mask_wedges_in_vertex_ = tf.logical_and(mask_wedges_in_vertex_, tf.expand_dims(vertex_indices_mask, -1))

        #[num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex, 2, 2]
        mask_wedges_in_vertex = tf.stack([mask_wedges_in_vertex_, mask_wedges_in_vertex_], axis=-1)
        mask_wedges_in_vertex = tf.stack([mask_wedges_in_vertex, mask_wedges_in_vertex], axis=-1)

        #[num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex]
        tensor_valid_vertices_2_wedges = tf.gather(tensor_vertices_2_wedges, valid_vd_idx)

        # [num_targets, num_sources, max_num_paths_extended, 3]
        vd_points_ext = tf.repeat(vd_points, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        refl_points_ext = tf.repeat(refl_points, tensor_valid_vertices_2_wedges.shape[3], axis=2)

        #[num_targets, num_sources, max_num_paths_extended]
        valid_vertex_wedges_idx = tf.reshape(tensor_valid_vertices_2_wedges, [num_targets, num_sources, -1])

        normals = tf.gather(self._wedges_normals, valid_vertex_wedges_idx, axis=0)

        # [num_targets, num_sources, max_num_paths]
        cos_wedges_angle = dot(normals[...,0,:], normals[...,1,:], clip=True)
        wedges_angle = PI - tf.math.acos(cos_wedges_angle)
        wedge_n = (2.*PI-wedges_angle)/PI

        is_concave_wedge = tf.gather(self._is_concave_wedge, valid_vertex_wedges_idx)
        wedge_n = tf.where(is_concave_wedge, wedges_angle / PI, wedge_n)

        # [num_targets, num_sources, max_num_paths_extended, 3]
        e_hat = tf.gather(self._wedges_e_hat, valid_vertex_wedges_idx)
        wedge_origins_tmp = tf.gather(self._wedges_origin, valid_vertex_wedges_idx)

        # For 1)
        # [num_targets, num_sources, max_num_paths_extended]
        needs_swap = tf.logical_not(tf.reduce_all(tf.math.equal(wedge_origins_tmp, vd_points_ext), axis=3))
        # [num_targets, num_sources, max_num_paths_extended, 3]
        needs_swap = tf.expand_dims(needs_swap, axis=3)
        e_hat = tf.where(needs_swap, -e_hat, e_hat)
        # [num_targets, num_sources, max_num_paths_extended, 2, 3]
        #normals = tf.where(tf.expand_dims(needs_swap, -1), -normals, normals) # (for 1)  # TODO

        # Reshape sources and targets
        # [1, num_sources, 1, 3]
        sources = tf.reshape(sources, [1, -1, 1, 3])
        # [num_targets, 1, 1, 3]
        targets = tf.reshape(targets, [-1, 1, 1, 3])

        n_0_hat = tf.where(needs_swap, normals[..., 1,:], normals[..., 0,:])  # TODO
        n_n_hat = tf.where(needs_swap, normals[..., 0,:], normals[..., 1,:])

        s_prime_hat, s_prime = r1_hat, r1
        s_hat, s = normalize(refl_points_ext - vd_points_ext)
        r2_ext = tf.repeat(r2, tensor_valid_vertices_2_wedges.shape[3], axis=2)

        # _theta_t, _phi_t = theta_phi_from_unit_vec(k_i[1])
        _theta_r, _phi_r = theta_phi_from_unit_vec(-l_hat)

        s_prime_hat = tf.repeat(s_prime_hat, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        s_prime = tf.repeat(s_prime, tensor_valid_vertices_2_wedges.shape[3], axis=2)

        mat_t_vd = self._compute_fields(s_prime_hat, s_hat, n_0_hat, n_n_hat, e_hat, etas_vd,
                                    scattering_coefficient_vd, s_prime, s+r2_ext, wedge_n, mask,
                                    mask_wedges_in_vertex_, mask_wedges_in_vertex,
                                    theta_t, phi_t, _theta_r, _phi_r, tensor_valid_vertices_2_wedges,
                                    vd_points_ext, valid_vertex_wedges_idx, skip_sf=True)
        
        #### 2) Reflection
        # [max_depth, num_targets, num_sources, max_num_paths]
        reduction_factor_refl = tf.sqrt(1 - scattering_coefficient_refl**2)
        reduction_factor_refl = tf.complex(reduction_factor_refl, tf.zeros_like(reduction_factor_refl))
        #_objects = objects[0]
        _k_t = k_i[1] # l_hat
        _k_r = k_r[1] # r2_hat
        #_reduction_factor = reduction_factor[0]
        #_etas = etas[0]
        mat_t_refl = self.solver_paths.solver_reflection._foo(refl_idxs, _k_t, _k_r, _normals, reduction_factor_refl, etas_refl)
        
        ### 3) total field
        sf = 1 / ((r2+l) * r1)
        mat_t = tf.multiply(mat_t_vd, mat_t_refl) * tf.cast(sf, self.solver_paths._dtype)[..., None, None]

        return mat_t

    def _get_param_u(self, beta1, beta2):  # Rubinowicz parameter
        r"""beta1, beta2 - elevation of incidence and diffraction"""

        eps = tf.cast(1e-6, beta1.dtype)
        # pi = tf.cast(PI, beta1.dtype)
        beta1 = tf.clip_by_value(beta1, eps, PI - eps)
        beta2 = tf.clip_by_value(beta2, eps, PI - eps)

        tmp1 = tf.math.log(tf.math.tan(beta2 / 2))
        tmp2 = tf.math.log(tf.math.tan(beta1 / 2))

        return tmp1 - tmp2
    
    def _get_func_B(self, Phi, u, n):
        """Evaluate the angular factor using a cancellation-free denominator."""

        angle = Phi / n
        half_u = u / (2.0 * n)
        denominator = tf.square(tf.sin(0.5 * angle)) + tf.square(tf.sinh(half_u))

        return tf.sin(angle) / (4.0 * n * denominator)

    def _get_param_b_using_ell(self, k, ell, beta1, beta2):
        tmp2 = 2.0 * tf.sin(0.5 * (beta2 - beta1))**2

        bad = tf.where(tmp2 == 0)
        if tf.shape(bad)[0] > 0:
            print("warning: division by zero in _get_param_b")

        return k * ell * tmp2
    
    def _get_param_b(self, k, r1, r2, beta1, beta2, coef_ell=None):
        ell = r1 * r2 / (r1 + r2)

        if coef_ell is not None:
            ell = coef_ell * ell

        # tmp2 = 1 - tf.math.cos(beta2 - beta1)
        tmp2 = 2.0 * tf.sin(0.5 * (beta2 - beta1))**2

        return k * ell * tmp2

    def _get_coefs_D_new(self, wedge_n, param_a, param_b, param_u, phi1_rad, phi2_rad, coef, skip_F=False):
        # param_a: [..., 4]

        phi_m = phi2_rad - phi1_rad
        phi_p = phi2_rad + phi1_rad

        B1 = tf.cast(self._get_func_B(PI + phi_m, param_u, wedge_n), self._dtype) * coef
        B2 = tf.cast(self._get_func_B(PI - phi_m, param_u, wedge_n), self._dtype) * coef
        B3 = tf.cast(self._get_func_B(PI + phi_p, param_u, wedge_n), self._dtype) * coef
        B4 = tf.cast(self._get_func_B(PI - phi_p, param_u, wedge_n), self._dtype) * coef

        if skip_F:
            return B1, B2, B3, B4

        t_gfi = t_gfi_full_approximated_capolino(param_b, param_a, self.obj_geom.diffraction_tables)

        d1 = B1 * t_gfi[..., 0]
        d2 = B2 * t_gfi[..., 1]
        d3 = B3 * t_gfi[..., 2]
        d4 = B4 * t_gfi[..., 3]

        return d1, d2, d3, d4

    def _get_coefs_D2(self, wedge_n, param_a, param_b, param_u, phi1_rad, phi2_rad, skip_F=False):
        """Evaluate all four VD coefficients on the trailing axis.

        ``param_a`` and ``param_b`` have shape [..., 4]; angular inputs
        have shape [...]. Preserve the legacy tuple return structure.
        """

        phi_m = phi2_rad - phi1_rad
        phi_p = phi2_rad + phi1_rad
        Phi = tf.stack([PI + phi_m, PI - phi_m, PI + phi_p, PI - phi_p], axis=-1)
        _t_gfi = t_gfi_full_approximated_capolino(param_b, param_a, self.obj_geom.diffraction_tables)

        # Broadcast u and n: the shared hyperbolic factor is computed once.
        func_B = tf.cast(self._get_func_B(
            Phi, param_u[..., None], wedge_n[..., None]
        ), self._dtype)
        coefficients = _t_gfi * func_B

        return (
            tuple(tf.unstack(coefficients, num=4, axis=-1)),
            tuple(tf.unstack(func_B, num=4, axis=-1)),
            (param_b, _t_gfi),
        )

    def _coef_D_plus2(self, wedge_n, Phi, param_u, t_gfi):
        func_B = tf.complex(self._get_func_B(PI + Phi, param_u, wedge_n), tf.cast(0.0, self._rdtype))

        return t_gfi * func_B, func_B
    
    def _coef_D_minus2(self, wedge_n, Phi, param_u, t_gfi):
        func_B = tf.complex(self._get_func_B(PI - Phi, param_u, wedge_n), tf.cast(0.0, self._rdtype))

        return t_gfi * func_B, func_B
    
    # def _get_tip_a_plus(self, Phi, n, k, ell):
    #     tmp = 1 + tf.math.cos(Phi - 2 * PI * n * self._get_N_plus(Phi, n))

    #     return k * ell * tmp

    # def _get_tip_a_minus(self, Phi, n, k, ell):
    #     tmp = 1 + tf.math.cos(Phi - 2 * PI * n * self._get_N_minus(Phi, n))

    #     return k * ell * tmp

    def _get_tip_a_all(self, phi_m, phi_p, n, k, ell):
        """Return transition parameters [..., 4] in D1, D2, D3, D4 order.

        Use the same squared-sine expression as the scalar helpers near SB.
        Round is odd, so the minus terms use the same formula as the plus
        terms after reversing their angular numerator.
        """
        Phi = tf.stack([PI + phi_m, PI - phi_m, PI + phi_p, PI - phi_p], axis=-1)
        n = n[..., None]
        # Preserve the multiplication order used by the existing helpers.
        turns = tf.math.round(Phi / (2.0 * n * PI))
        delta = Phi - 2.0 * PI * n * turns
        return (2.0 * k * ell)[..., None] * tf.square(tf.sin(0.5 * delta))

    def _get_tip_a_plus(self, Phi, n, k, ell):
        """Compute the plus transition parameter without cancellation near SB."""

        delta = PI + Phi - 2.0 * PI * n * self._get_N_plus(Phi, n)
        return 2.0 * k * ell * tf.square(tf.sin(0.5 * delta))


    def _get_tip_a_minus(self, Phi, n, k, ell):
        """Compute the minus transition parameter without cancellation near SB."""

        delta = PI - Phi + 2.0 * PI * n * self._get_N_minus(Phi, n)
        return 2.0 * k * ell * tf.square(tf.sin(0.5 * delta))

    def _get_N_plus(self, Phi, n):
        return tf.math.round((Phi + PI) / (2*n*PI))

    def _get_N_minus(self, Phi, n):
        return tf.math.round((Phi - PI) / (2*n*PI))

    def _vertex_edge_geometry(self, wedge_idxs, points, use_isclose=False, swap=None):
        """Orient selected wedge tangents away from their vertex.

        ``swap`` optionally supplies the endpoint-orientation mask [N],
        allowing geometry and materials to reuse one computation. Existing
        callers can omit it to compute the mask here as before.
        """
        solver = self.solver_paths
        normals = tf.gather(solver._wedges_normals, wedge_idxs)
        wedge_angle = PI - acos_diff(dot(normals[:, 0], normals[:, 1], clip=True))
        is_concave = tf.gather(solver._is_concave_wedge, wedge_idxs)
        wedge_n = tf.where(is_concave, wedge_angle / PI, (2.0 * PI - wedge_angle) / PI)

        e_hat = tf.gather(solver._wedges_e_hat, wedge_idxs)
        if swap is None:
            swap = self._vertex_edge_swap(wedge_idxs, points)

        e_hat = tf.where(swap[:, None], -e_hat, e_hat)
        n_0_hat = tf.where(swap[:, None], normals[:, 1], normals[:, 0])
        n_n_hat = tf.where(swap[:, None], normals[:, 0], normals[:, 1])
        return e_hat, n_0_hat, n_n_hat, wedge_n

    def _vertex_edge_swap(self, wedge_idxs, points):
        """Return whether the vertex is closer to the end than to the origin.

        Parameters
        ----------
        wedge_idxs : tf.Tensor, integer, shape [N]
            Constituent wedge indices.
        points : tf.Tensor, real, shape [N, 3]
            Vertex positions associated with those wedges.

        Returns
        -------
        tf.Tensor, bool, shape [N]
            True when the edge orientation must be reversed.
        """

        solver = self.solver_paths
        origins = tf.gather(solver._wedges_origin, wedge_idxs)
        tangents = tf.gather(solver._wedges_e_hat, wedge_idxs)
        lengths = tf.gather(solver._wedges_length, wedge_idxs)
        ends = origins + lengths[:, None] * tangents

        distance_origin = tf.reduce_sum(tf.square(points - origins), axis=-1)
        distance_end = tf.reduce_sum(tf.square(points - ends), axis=-1)
        return distance_end < distance_origin

    def _vertex_edge_coefficients_flat(self, wedge_idxs, points, s_prime_hat, s_hat,
                                    s_prime, s, add_to_r1=None, geometry_override=None):
        """Compute selected VD coefficients without their transition function."""
        e_hat, n_0_hat, _, wedge_n = self._vertex_edge_geometry(wedge_idxs, points, use_isclose=True)

        if geometry_override is not None:
            n_0_hat, e_hat, wedge_n = geometry_override

        phi_prime, phi, beta_prime, beta = calc_angles(n_0_hat, e_hat, s_prime_hat, s_hat)
        phi_prime_c, phi_c, _, _ = calc_angles_concave(n_0_hat, e_hat, s_prime_hat, s_hat, wedge_n)
        phi_prime = tf.where(wedge_n < 1.0, phi_prime_c, phi_prime)
        phi = tf.where(wedge_n < 1.0, phi_c, phi)

        # Preserve the angular boundary convention of simple_compute_fields_main().
        phi_prime = tf.where(tf.experimental.numpy.isclose(phi_prime, 2 * PI, atol=1e-2),
                            tf.ones_like(phi_prime) * 1e-3, phi_prime)

        values = (phi_prime, phi, beta_prime, beta, s_prime, s, wedge_n)
        inputs = [value[None, None, :] for value in values]
        mask = tf.ones([1, 1, tf.shape(wedge_idxs)[0], 1], dtype=tf.bool)
        extra_distance = None if add_to_r1 is None else add_to_r1[None, None, :]

        coefficients, param_b, param_c = self.simple_compute_fields(
            *inputs, mask_wedges_in_vertex_=mask, add_to_r1=extra_distance, skip_F=True
        )

        coefficients = tuple(value[0, 0] for value in coefficients)
        geometry = (e_hat, n_0_hat, wedge_n, phi_prime, phi, beta_prime, beta)
        return coefficients, param_b[0, 0], param_c[0, 0], geometry


def _clear_diff_coef2(d, mask_wedges_in_vertex_, num_targets, num_sources, num_vertices, dtype):
    r""""
    d: [num_targets, num_sources, num_paths, ...]
    mask_wedges_in_vertex_: [num_targets, num_sources, num_vertices, num_wedges_per_vertex, ...]
    """
    res = np.reshape(d, mask_wedges_in_vertex_.shape)
    res = np.where(mask_wedges_in_vertex_, res, np.zeros_like(res))
    nans_bool = tf.math.is_nan(tf.math.real(res))
    res = tf.where(nans_bool, tf.zeros_like(res, dtype=dtype), res)
    res = tf.reshape(res, d.shape)

    return res
