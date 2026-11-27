
from __future__ import annotations

import numpy as np
import mitsuba as mi
import drjit as dr
import tensorflow as tf
import trimesh
import pickle
from typing import Tuple, TYPE_CHECKING

from sionna.constants import SPEED_OF_LIGHT, PI
from sionna.utils.tensors import expand_to_rank, insert_dims, flatten_dims,\
    split_dim, insert_dim_num
from .paths import Paths, PathsTmpData, invert_s_t
from .objects_geometry import ObjectsGeometry
from .diffraction_funcs import func_G_approximated_capolino, calc_angles, invert_angles,\
    my_wd_compute_fields, transition_func, slope_diffr_coef, calc_angles_concave,\
    get_wd_points, my_wd_compute_fields2,\
    slope_diffr_coef_2, slope_vertex_diffr_coef, wd_numerical_derivative, vd_numerical_derivative,\
    slope_transition_func, t_gfi_full_approximated_capolino, vertex_B_odd
from .utils import dot, phi_hat, theta_hat, theta_phi_from_unit_vec,\
    normalize, moller_trumbore, component_transform, mi_to_tf_tensor,\
        compute_field_unit_vectors, reflection_coefficient, fibonacci_lattice,\
            cot, cross, sign, rotation_matrix, acos_diff, rot_mat_from_unit_vecs, \
            r_hat, slab_coefficients

if TYPE_CHECKING:
    from .solver_paths import SolverPaths
    from .solver_paths import PathsTmpData


class DoubleDiffraction:
    def __init__(self, solver_paths: SolverPaths):
        self.solver_paths = solver_paths
        self._dtype = self.solver_paths._dtype   # complex
        self._rdtype = self.solver_paths._rdtype  # float

        self.coplanar_eps = 0.03

        self.is_wedge_wedge = True
        self.is_ww_analytical = True
        self.is_ww_analytical_with_double_tilda = True
        self.is_only_coplanar = True
        self.is_wedge_wedge_sv = False  # shared vertex

        self.is_wedge_vertex = False
        self.is_vertex_wedge = False
        self.is_eee = False
        self.is_vertex_vertex = False
        self.is_ev_method = "T_ev" # T_ev, R_correction

        self.is_only_in_nlos = True  # for VE, EV combinations 

        self.is_vw_reversed_compute_fields = True
        self.is_add_slope = False
        self.dd_mode = "all"  # 'all', "main", "slope"
        self.sv_add_delta = True  # Local even/odd SV pole corrections.
        self.is_only_close_2_transition = True
        self.is_coef_ell_separate_in_vd = True

        self.dd_pairs_coplanar = None

        self.close_2_transition_eps = 0.5
        r"""More value - less paths"""

        # [num_wedges, 3]
        self._wedges_origin = self.solver_paths.wedges_origin
        self._wedges_normals = self.solver_paths.wedges_normals
        self._wedges_e_hat = self.solver_paths.wedges_e_hat
        # [num_wedges, 2]
        self._wedges_objects = self.solver_paths.wedges_objects
        # [num_wedges]
        self._is_edge = self.solver_paths.is_edge
        self._wedges_length = self.solver_paths.wedges_length

        # [num_objects+1]
        self._objects_2_primitives = [shape.effective_primitive_count() 
                    for shape in (self.solver_paths._mi_scene.shapes())]
        self._objects_2_primitives.insert(0, 0)
        self._objects_2_primitives = np.cumsum(self._objects_2_primitives)

        # [num_primitives, 3]
        self._primitives_2_wedges = self.solver_paths._primitives_2_wedges
        self._facet_normals = self.solver_paths._normals

    def set_objects_geometry(self, objects_geometry: ObjectsGeometry):
        self.obj_geom = objects_geometry
        self.vertices = self.obj_geom._vertices
        self.vertices_2_wedges = self.obj_geom.vertex_2_wedges

    def get_coplanar_dd_wedges(self, facet_2_wedges, facet_2_adj_facets):
        coplanar_double_wedges = []
        slope_normals = []
        primitives_2_wedges = self._primitives_2_wedges.numpy()  # facet_2_wedges

        for facet_idx in range(len(primitives_2_wedges)):
            facet_wedges = facet_2_wedges[facet_idx]
            for i in range(len(facet_wedges)):
                for j in range(len(facet_wedges)):
                    if i == j:
                        continue

                    wedge1_idx, wedge2_idx = facet_wedges[i], facet_wedges[j]
                    if wedge1_idx == -1:
                        continue

                    wedge2_candidate_idxs = []
                    if wedge2_idx != -1:
                        wedge2_candidate_idxs = [wedge2_idx]
                    else:
                        adj_facets = facet_2_adj_facets[facet_idx]   # [(facet_idx, wedge_idx), ...]
                        for adj_facet_idx, adj_wedge_idx in adj_facets:
                            if adj_wedge_idx == -1:     # wedge2_idx:
                                wedge2_candidate_idxs = facet_2_wedges[adj_facet_idx]
                    
                    for wedge2_candidate_idx in wedge2_candidate_idxs:
                        if wedge2_candidate_idx != -1:
                            coplanar_double_wedges.append([wedge1_idx, wedge2_candidate_idx])
                            slope_normals.append(self._facet_normals[facet_idx])
                    
        # [num_coplanar_double_wedges, 2]
        coplanar_double_wedges = tf.convert_to_tensor(coplanar_double_wedges, dtype=tf.int32)
        # [num_coplanar_double_wedges, 3]
        slope_normals = tf.convert_to_tensor(slope_normals, dtype=self._rdtype)

        return coplanar_double_wedges, slope_normals
    
    def filter_slope_diffraction_pairs(self, pairs):
        """
        pairs : [num_pairs, 2]

        Finds intersection of `pairs` with `self._coplanar_dd_wedges`
        """

        a = self._coplanar_dd_wedges
        find_match = tf.reduce_sum(tf.abs(tf.expand_dims(pairs, 0) - tf.expand_dims(a, 1)), 2)
        indices = tf.transpose(tf.where(tf.equal(find_match, tf.zeros_like(find_match))))[0]
        res = tf.gather(self._coplanar_dd_wedges, indices, axis=0)

        return res, indices
        
    
    def _discard_obstructing_wedges(self, candidate_wedges, points):
        r"""
        Discard wedges for which at least one of the source or target are
        "inside" the wedge

        Inputs
        ------
        candidate_wedges : [num_candidate_wedges], int
            Candidate wedges.
            Entries correspond to wedges indices.

        points : [num_points, 3], tf.float
            Coordinates of the targets / sources.

        Output
        -------
        wedges_indices : [num_points, max_num_paths], tf.int
            Indices of the wedges that interacted with the diffracted paths
        """

        epsilon = tf.cast(self.solver_paths.EPSILON, self._rdtype)

        # [num_candidate_wedges, 3]
        origins = tf.gather(self._wedges_origin, candidate_wedges)

        # Expand to broadcast with sources/targets and 0/n faces
        # [1, num_candidate_wedges, 1, 3]
        origins = tf.expand_dims(origins, axis=0)
        origins = tf.expand_dims(origins, axis=2)

        # Normals
        # [num_candidate_wedges, 2, 3]
        # [:,0,:] : 0-face
        # [:,1,:] : n-face
        normals = tf.gather(self._wedges_normals, candidate_wedges)
        # Expand to broadcast with the sources or targets
        # [1, num_candidate_wedges, 2, 3]
        normals = tf.expand_dims(normals, axis=0)

        # Expand to broadcast with candidate and 0/n faces wedges
        # [num_points, 1, 1, 3]
        points = expand_to_rank(points, 4, 1)
        # [num_points, num_candidate_wedges, 1, 3]
        u_t = points - origins

        # [num_points, num_candidate_wedges, 2]
        sources_valid_half_space = dot(u_t, normals)
        sources_valid_half_space = tf.greater(sources_valid_half_space,
                    tf.fill(tf.shape(sources_valid_half_space), epsilon))
        # [num_points, num_candidate_wedges]
        sources_valid_half_space = tf.reduce_any(sources_valid_half_space,
                                                 axis=2)

        mask_full = sources_valid_half_space

        # Discard paths with no valid link
        # [max_num_paths]
        valid_paths = tf.where(tf.reduce_any(mask_full, axis=(0)))[:,0]
        # [num_points, max_num_paths]
        mask = tf.gather(mask_full, valid_paths, axis=1)
        # [max_num_paths]
        wedges_indices = tf.gather(candidate_wedges, valid_paths, axis=0)
        # Set invalid wedges to -1
        # [num_points, max_num_paths]
        wedges_indices = tf.where(mask, wedges_indices, -1)

        return wedges_indices, mask_full

    def _discard_obstructing_wedges2(self, candidate_wedges, points):
            r"""
            Discard wedges for which at least one of the source or target are
            "inside" the wedge.
            Shapes are identical.
    
            Inputs
            ------
            candidate_wedges : [num_candidate_wedges], int
                Candidate wedges.
                Entries correspond to wedges indices.
    
            points : [num_candidate_wedges, 3], tf.float
                Coordinates of the targets / sources.
    
            Output
            -------
            wedges_indices : [num_points, max_num_paths], tf.int
                Indices of the wedges that interacted with the diffracted paths
            """
    
            epsilon = -tf.cast(self.solver_paths.EPSILON, self._rdtype)
    
            # [num_candidate_wedges, 3]
            origins = tf.gather(self._wedges_origin, candidate_wedges)
    
            # Normals
            # [num_candidate_wedges, 2, 3]
            # [:,0,:] : 0-face
            # [:,1,:] : n-face
            normals = tf.gather(self._wedges_normals, candidate_wedges)

            # [num_candidate_wedges, 3]
            u_t = points - origins

            u_t = u_t[..., None, :]  # [num_candidate_wedges, 1, 3]
    
            # [num_candidate_wedges, 2]
            sources_valid_half_space = dot(u_t, normals)
            sources_valid_half_space = tf.greater(sources_valid_half_space,
                        tf.fill(tf.shape(sources_valid_half_space), epsilon))
            # [num_points, num_candidate_wedges]
            sources_valid_half_space = tf.reduce_any(sources_valid_half_space,
                                                     axis=-1)
    
            mask_full = sources_valid_half_space
    
            return mask_full
    
    def get_coplanar(self, dd_pair_idxs):
        r"""

        Inputs
        ------
        dd_pair_idxs : [..., num_pairs, 2], int
            Indices of the wedge pairs.
        """
        #eps = 0.02 # 1e-2

        # [..., num_pairs, 3]
        e_hat_1 = tf.gather(self._wedges_e_hat, dd_pair_idxs[..., 0], axis=0)
        e_hat_2 = tf.gather(self._wedges_e_hat, dd_pair_idxs[..., 1], axis=0)
        w1_point = tf.gather(self._wedges_origin, dd_pair_idxs[..., 0], axis=0)
        w2_point = tf.gather(self._wedges_origin, dd_pair_idxs[..., 1], axis=0)
        l_hat, _ = normalize(w2_point - w1_point)

        mask_coplanar = tf.abs(dot(cross(e_hat_1, e_hat_2), l_hat)) < self.coplanar_eps

        return mask_coplanar
    
    def _junction_candidates(self, wedge_idxs, enabled):
        """Identify flat shared edges with exactly one transmissive face."""
        wd = self.solver_paths.solver_wedge_diffraction
        faces = tf.gather(self._wedges_objects, wedge_idxs)
        flags = tf.gather(enabled, faces)
        return wd._is_flat_shared_wedge(wedge_idxs) & (flags[..., 0] != flags[..., 1])

    def _vertex_has_junction(self, vertex_ids, enabled):
        """Test whether a vertex has any eligible constituent edge."""
        table = self.solver_paths.vertex_diffraction.vertices_2_wedges.to_tensor(default_value=-1)
        wedges = tf.gather(table, vertex_ids)
        safe = tf.maximum(wedges, 0)
        return tf.reduce_any((wedges >= 0) & self._junction_candidates(safe, enabled), axis=-1)

    def _junction_geometry(self, wedge_idxs, normal, tangent, n, incoming, outgoing,
                           enabled, endpoint_swap=None):
        """Select a virtual metal half-plane using the actual adjacent segment offsets."""
        eligible = self._junction_candidates(wedge_idxs, enabled)
        wd = self.solver_paths.solver_wedge_diffraction
        plane_normal = tf.gather(self._wedges_normals[:, 0], wedge_idxs)
        crossed = wd._opposite_sides(-dot(incoming, plane_normal), dot(outgoing, plane_normal))
        junction = eligible & crossed

        faces = tf.gather(self._wedges_objects, wedge_idxs)
        glass_zero = tf.gather(enabled, faces[:, 0])
        glass_object = tf.where(glass_zero, faces[:, 0], faces[:, 1])
        if endpoint_swap is None:
            # The edge tangent can be reversed while retaining the physical metal face.
            flip_tangent = junction & glass_zero
            tangent = tf.where(flip_tangent[:, None], -tangent, tangent)
            back = eligible & (dot(incoming, normal) > 0.0)
            normal = tf.where(back[:, None], -normal, normal)
            tangent = tf.where(back[:, None], -tangent, tangent)
        else:
            # The vertex tangent must keep pointing away from the endpoint.
            metal_is_face_one = glass_zero != endpoint_swap
            back = eligible & ~junction & (dot(incoming, normal) > 0.0)
            flip_normal = (junction & metal_is_face_one) | back
            normal = tf.where(flip_normal[:, None], -normal, normal)

        n = tf.where(junction, tf.cast(2.0, n.dtype), n)
        return normal, tangent, n, junction, glass_object

    def _junction_slab_matrix(self, junction, glass_objects, points, direction, normal,
                              basis_s, basis_p, relative_permittivity):
        """Return slab transmission in the corresponding outer-ray basis."""
        indices = tf.cast(tf.where(junction), tf.int32)
        transfer = tf.eye(2, batch_shape=tf.shape(junction), dtype=self._dtype)
        if tf.shape(indices)[0] == 0:
            return transfer

        objects = tf.gather_nd(glass_objects, indices)
        ray = tf.gather_nd(direction, indices)
        normal = tf.gather_nd(normal, indices)
        basis_s = tf.gather_nd(basis_s, indices)
        basis_p = tf.gather_nd(basis_p, indices)
        solver = self.solver_paths
        rm_callable = solver._scene.radio_material_callable
        if rm_callable is None:
            eta = tf.gather(relative_permittivity, objects)
        else:
            eta, _, _ = rm_callable(objects, tf.gather_nd(points, indices))

        _, _, t_s, t_p = slab_coefficients(
            eta, tf.abs(dot(ray, normal, clip=True)),
            tf.gather(solver._object_thickness, objects),
            solver._scene.wavelength, dtype=self._dtype,
        )
        te, te_norm = normalize(cross(ray, normal))
        te = tf.where((te_norm < 1e-7)[:, None], basis_s, te)
        tm = cross(te, ray)
        to_slab = tf.cast(component_transform(basis_s, basis_p, te, tm), self._dtype)
        from_slab = tf.cast(component_transform(te, tm, basis_s, basis_p), self._dtype)
        matrix = tf.matmul(from_slab, tf.stack([t_s, t_p], axis=-1)[..., None] * to_slab)
        return tf.tensor_scatter_nd_update(transfer, indices, matrix)

    def _unshifted_middle_visibility(self, point1, point2):
        """Check the actual open segment between two diffraction points."""
        direction, distance = normalize(point2 - point1)
        return ~self.solver_paths._test_obstruction(point1, direction, distance)

    def discard_obstructing_flat(
        self, candidate_pairs, sources, targets,
        junction_enabled=None, pair_block_size=4096,
    ):
        """Return [N, 3] rows: [tx, rx, index in candidate_pairs]."""
        if pair_block_size <= 0:
            raise ValueError("pair_block_size must be positive")

        def build_device_lists(wedges, devices):
            # unique_wedges: [W], pair_to_wedge: [num_pairs].
            unique_wedges, pair_to_wedge = tf.unique(wedges)
            _, allowed = self._discard_obstructing_wedges(unique_wedges, devices)

            # allowed: [num_devices, W].
            exception = tf.gather(self.solver_paths._is_edge, unique_wedges)
            if junction_enabled is not None:
                exception |= self._junction_candidates(unique_wedges, junction_enabled)
            allowed |= exception[None, :]

            # Group allowed device IDs by wedge.
            device_ids = tf.cast(tf.where(tf.transpose(allowed))[:, 1], tf.int32)
            counts = tf.reduce_sum(tf.cast(allowed, tf.int32), axis=0)
            offsets = tf.cumsum(counts, exclusive=True)
            return device_ids, counts, offsets, pair_to_wedge

        tx_ids, tx_counts, tx_offsets, tx_map = build_device_lists(
            candidate_pairs[:, 0], sources
        )
        rx_ids, rx_counts, rx_offsets, rx_map = build_device_lists(
            candidate_pairs[:, 1], targets
        )

        num_pairs = tf.shape(candidate_pairs)[0]
        chunks = tf.TensorArray(
            tf.int32,
            size=1,
            dynamic_size=True,
            infer_shape=False,
            element_shape=(None, 3),
        ).write(0, tf.zeros([0, 3], tf.int32))

        def cond(start, chunk_idx, chunks):
            return start < num_pairs

        def body(start, chunk_idx, chunks):
            stop = tf.minimum(start + pair_block_size, num_pairs)
            tx_wedges = tx_map[start:stop]
            rx_wedges = rx_map[start:stop]

            n_tx = tf.gather(tx_counts, tx_wedges)
            n_rx = tf.gather(rx_counts, rx_wedges)
            counts = n_tx * n_rx

            # Enumerate only the allowed TX/RX products for each pair.
            pair_local = tf.repeat(tf.range(stop - start), counts)
            local = tf.range(tf.reduce_sum(counts)) - tf.repeat(
                tf.cumsum(counts, exclusive=True), counts
            )
            n_rx_row = tf.gather(n_rx, pair_local)
            tx_wedge = tf.gather(tx_wedges, pair_local)
            rx_wedge = tf.gather(rx_wedges, pair_local)

            tx = tf.gather(tx_ids, tf.gather(tx_offsets, tx_wedge) + local // n_rx_row)
            rx = tf.gather(rx_ids, tf.gather(rx_offsets, rx_wedge) + local % n_rx_row)
            rows = tf.stack([tx, rx, start + pair_local], axis=-1)
            return stop, chunk_idx + 1, chunks.write(chunk_idx, rows)

        _, _, chunks = tf.while_loop(cond, body, (0, 1, chunks))
        return chunks.concat()

    def pack_ee_candidates(self, candidate_pairs, candidate_rows, num_tx, num_rx):
        """Pack flat EE candidates into the existing [TX, RX, slot, ...] layout."""
        # Sort by link and retain candidate order within each link.
        link_ids = candidate_rows[:, 0] * num_rx + candidate_rows[:, 1]
        order = tf.argsort(link_ids, stable=True)
        candidate_rows = tf.gather(candidate_rows, order)
        link_ids = tf.gather(link_ids, order)

        counts = tf.math.unsorted_segment_sum(
            tf.ones_like(link_ids), link_ids, num_tx * num_rx
        )
        offsets = tf.cumsum(counts, exclusive=True)
        slots = tf.range(tf.shape(link_ids)[0]) - tf.gather(offsets, link_ids)
        ee_slots = tf.concat([candidate_rows[:, :2], slots[:, None]], axis=1)

        max_slots = tf.reduce_max(tf.concat([counts, tf.zeros([1], tf.int32)], axis=0))
        shape = tf.stack([num_tx, num_rx, max_slots])

        pairs = tf.gather(candidate_pairs, candidate_rows[:, 2])
        dd_pair_idxs = tf.tensor_scatter_nd_update(
            tf.fill(tf.concat([shape, [2]], axis=0), tf.cast(-1, pairs.dtype)),
            ee_slots,
            pairs,
        )
        active = tf.scatter_nd(ee_slots, tf.ones_like(slots, dtype=tf.bool), shape)
        return dd_pair_idxs, active, (candidate_rows, ee_slots)

    def build_env_ee_rows(self, sources, targets, tx_edges, rx_edges,
                          junction_enabled=None):
        """Build [N, 4] rows [tx, rx, w1, w2] from per-device edge lists."""
        solver = self.solver_paths
        num_sources = tf.shape(sources)[0]
        num_targets = tf.shape(targets)[0]

        def prepare(devices, edge_rows):
            # edge_rows: [M, 2], columns [device_id, wedge_id].
            device_ids, wedges = tf.unstack(edge_rows, axis=1)
            points = tf.gather(devices, device_ids)
            origins = tf.gather(self._wedges_origin, wedges)
            normals = tf.gather(self._wedges_normals, wedges)

            # Check one device against its candidate edge per row.
            offsets = points - origins
            signed = dot(offsets[:, None, :], normals)  # [M, 2]
            allowed = tf.reduce_any(signed > tf.cast(solver.EPSILON, signed.dtype), axis=1)
            allowed |= tf.gather(solver._is_edge, wedges)
            if junction_enabled is not None:
                allowed |= self._junction_candidates(wedges, junction_enabled)

            rows = tf.boolean_mask(edge_rows, allowed)
            rows = tf.gather(rows, tf.argsort(rows[:, 0], stable=True))
            counts = tf.math.unsorted_segment_sum(
                tf.ones_like(rows[:, 0]), rows[:, 0], tf.shape(devices)[0]
            )
            return rows[:, 1], counts, tf.cumsum(counts, exclusive=True)

        tx_wedges, tx_counts, tx_offsets = prepare(sources, tx_edges)
        rx_wedges, rx_counts, rx_offsets = prepare(targets, rx_edges)

        # One count per link: [num_sources, num_targets].
        counts = tf.reshape(tx_counts[:, None] * rx_counts[None, :], [-1])
        link = tf.repeat(tf.range(num_sources * num_targets), counts)
        local = tf.range(tf.reduce_sum(counts)) - tf.repeat(
            tf.cumsum(counts, exclusive=True), counts
        )

        tx = link // num_targets
        rx = link % num_targets
        n_rx = tf.gather(rx_counts, rx)
        w1 = tf.gather(tx_wedges, tf.gather(tx_offsets, tx) + local // n_rx)
        w2 = tf.gather(rx_wedges, tf.gather(rx_offsets, rx) + local % n_rx)

        rows = tf.stack([tx, rx, w1, w2], axis=1)
        return tf.boolean_mask(rows, w1 != w2)

    def discard_obstructing(self, candidate_pairs, sources, targets, return_indices=False, junction_enabled=None, 
                            check_neighbor_origins=True):
        r"""
        Discard wedges for which at least one of the source or target are
        "inside" the wedge

        Inputs
        ------
        candidate_pairs : [num_dd_candidates, 2], int
            wedge1, wedge2 index pairs.

        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        targets : [num_targets, 3], tf.float
            Coordinates of the targets.   

        Output
        -------
        pairs : [num_targets, num_sources, max_num_paths], tf.int
            Indices of the wedges that interacted with the diffracted paths
        """

        num_targets = tf.shape(targets)[0]
        num_sources = tf.shape(sources)[0]

        w1, w2 = candidate_pairs[:, 0], candidate_pairs[:, 1]
        _, mask1 = self._discard_obstructing_wedges(w1, sources)  # [TX, pairs]
        _, mask2 = self._discard_obstructing_wedges(w2, targets)  # [RX, pairs]

        if check_neighbor_origins:
            # Preserve the legacy filter for shared-vertex callers.
            o1 = tf.gather(self._wedges_origin, w1)
            o2 = tf.gather(self._wedges_origin, w2)
            mask1 &= self._discard_obstructing_wedges2(w1, o2)[None, :]
            mask2 &= self._discard_obstructing_wedges2(w2, o1)[None, :]
        else:
            # Open edges have no forbidden interior sector.
            mask1 |= tf.gather(self.solver_paths._is_edge, w1)[None, :]
            mask2 |= tf.gather(self.solver_paths._is_edge, w2)[None, :]

        if junction_enabled is not None:
            mask1 |= self._junction_candidates(candidate_pairs[:, 0], junction_enabled)[None, :]
            mask2 |= self._junction_candidates(candidate_pairs[:, 1], junction_enabled)[None, :]

        # [num_sources, num_targets, num_dd_candidates]
        mask1 = tf.expand_dims(mask1, axis=1)
        mask1 = tf.repeat(mask1, tf.shape(targets)[0], axis=1)

        # [num_sources, num_targets, num_dd_candidates]
        mask2 = tf.expand_dims(mask2, axis=0)
        mask2 = tf.repeat(mask2, tf.shape(sources)[0], axis=0)

        # [num_sources, num_targets, num_dd_candidates]
        pairs = candidate_pairs[None, None, ...]
        pairs = tf.repeat(pairs, tf.shape(sources)[0], axis=0)
        pairs = tf.repeat(pairs, tf.shape(targets)[0], axis=1)

        mask = tf.logical_and(mask1, mask2)
        #mask = mask[..., None]
        pairs = tf.where(mask[..., None], candidate_pairs, -1)

        gather_indices = tf.cast(tf.where(mask), tf.int32)
        path_indices = tf.cumsum(tf.cast(mask, tf.int32), axis=-1)
        packed_slots = tf.gather_nd(path_indices, gather_indices) - 1
        new_scatter_indices = tf.concat([gather_indices[:, :2], packed_slots[:, None]], axis=1)

        # [num_sources, num_targets]
        num_paths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
        # Maximum number of valid paths
        # ()
        max_num_paths = tf.reduce_max(num_paths)

        # [num_sources, num_targets, max_num_paths]
        new_mask = tf.fill([num_sources, num_targets, max_num_paths], False)
        # [num_keep_paths]
        mask_ = tf.gather_nd(mask, gather_indices)
        # [num_sources, num_targets, max_num_paths]
        new_mask = tf.tensor_scatter_nd_update(new_mask, new_scatter_indices, mask_)
        valid_pairs = -1 * tf.ones([num_sources, num_targets, max_num_paths, 2], dtype=tf.int32)

        pairs_ = tf.gather_nd(pairs, gather_indices)
        new_scatter_indices_0 = tf.pad(new_scatter_indices, [[0, 0], [0, 1]], constant_values=0)
        new_scatter_indices_1 = tf.pad(new_scatter_indices, [[0, 0], [0, 1]], constant_values=1)
        # [max_depth, num_sources, num_targets, max_num_paths]
        valid_pairs = tf.tensor_scatter_nd_update(valid_pairs,
                                new_scatter_indices_0, pairs_[:, 0])
        valid_pairs = tf.tensor_scatter_nd_update(valid_pairs,
                                new_scatter_indices_1, pairs_[:, 1])
        
        if return_indices:
            return valid_pairs, new_mask, (gather_indices, new_scatter_indices)

        return valid_pairs, new_mask

    def trace_sv_flat(self, sources, targets, rows):
        """Trace shared-vertex candidates [N, 5]: [tx, rx, w1, w2, vertex].

        Keep different wedge pairs as separate paths. Only device-to-vertex
        visibility queries are deduplicated, not the resulting SV contributions.
        """
        vertices = self.obj_geom._vertices
        points = tf.gather(vertices, rows[:, 4])
        if self.is_only_close_2_transition:
            tx = tf.gather(sources, rows[:, 0])
            rx = tf.gather(targets, rows[:, 1])
            direct = tf.norm(rx - tx, axis=-1)
            via_vertex = tf.norm(points - tx, axis=-1) + tf.norm(rx - points, axis=-1)
            # Preserve the existing SV excess-length criterion (not the EE angle test).
            close = tf.abs(direct - via_vertex) < 10 * self.solver_paths._scene._wavelength
            rows = tf.boolean_mask(rows, close)
        tx_visible = self._sv_device_visibility_flat(sources, rows[:, 0], rows[:, 4])
        rx_visible = self._sv_device_visibility_flat(targets, rows[:, 1], rows[:, 4])
        rows = tf.boolean_mask(rows, tx_visible & rx_visible)
        return self.pack_sv_paths_flat(sources, targets, rows)

    def _sv_device_visibility_flat(self, devices, device_ids, vertex_ids):
        """Check each distinct [device, vertex] segment once; return [N] validity."""
        if tf.size(vertex_ids) == 0:
            return tf.zeros([0], tf.bool)
        vertices = self.obj_geom._vertices
        stride = tf.cast(tf.maximum(tf.shape(vertices)[0], 1), tf.int64)
        keys = tf.cast(device_ids, tf.int64) * stride + tf.cast(vertex_ids, tf.int64)
        unique, inverse = tf.unique(keys)
        origins = tf.gather(devices, tf.cast(unique // stride, tf.int32))
        endpoints = tf.gather(vertices, tf.cast(unique % stride, tf.int32))
        direction, distance = normalize(endpoints - origins)
        visible = ~self.solver_paths._test_obstruction(origins, direction, distance)
        return tf.gather(visible, inverse)

    def pack_sv_paths_flat(self, sources, targets, rows):
        """Assemble Paths only after SV filtering; preserve all distinct edge pairs.

        The two interaction slots share V. The middle direction is explicitly
        zero: do not differentiate a norm of V-V just to construct metadata.
        """
        solver = self.solver_paths
        paths = Paths(sources, targets, solver._scene, types=Paths.DOUBLE_DIFF)
        tmp = PathsTmpData(sources, targets, self._dtype)
        num_rx, num_tx = tf.shape(targets)[0], tf.shape(sources)[0]
        links = rows[:, 1] * num_tx + rows[:, 0]
        order = tf.argsort(links, stable=True)
        rows, links = tf.gather(rows, order), tf.gather(links, order)
        points = tf.gather(self.obj_geom._vertices, rows[:, 4])
        incident, r1 = normalize(points - tf.gather(sources, rows[:, 0]))
        outgoing, r2 = normalize(tf.gather(targets, rows[:, 1]) - points)
        theta_t, phi_t = theta_phi_from_unit_vec(incident)
        theta_r, phi_r = theta_phi_from_unit_vec(-outgoing)
        counts = tf.math.unsorted_segment_sum(tf.ones_like(links), links, num_rx * num_tx)
        offsets = tf.cumsum(counts, exclusive=True)
        slots = tf.range(tf.size(links)) - tf.gather(offsets, links)
        indices = tf.stack([rows[:, 1], rows[:, 0], slots], axis=1)
        max_paths = tf.reduce_max(tf.concat([counts, tf.zeros([1], tf.int32)], axis=0))
        shape = tf.stack([num_rx, num_tx, max_paths])

        def packed(values):
            return tf.scatter_nd(indices, values, tf.concat([shape, tf.shape(values)[1:]], axis=0))

        paths.objects = tf.stack([
            tf.tensor_scatter_nd_update(tf.fill(shape, -1), indices, rows[:, 2]),
            tf.tensor_scatter_nd_update(tf.fill(shape, -1), indices, rows[:, 3]),
        ], axis=0)
        vertex_points = packed(points)
        paths.vertices = tf.stack([vertex_points, vertex_points], axis=0)
        paths.mask = packed(tf.ones_like(links, tf.bool))
        paths.depth_mask = paths.objects >= 0
        paths.dd_type = tf.fill(shape, 4)
        paths.my_types = tf.fill(tf.shape(paths.objects), Paths.VERTEX_DIFFRACTED)
        paths.path_types = paths.my_types
        paths.true_objects = tf.gather(solver._wedges_objects[:, 0], tf.maximum(paths.objects, 0))
        paths.theta_t, paths.phi_t = packed(theta_t), packed(phi_t)
        paths.theta_r, paths.phi_r = packed(theta_r), packed(phi_r)
        total_distance = r1 + r2
        paths.tau = tf.tensor_scatter_nd_update(
            -tf.ones(shape, self._rdtype), indices, total_distance / SPEED_OF_LIGHT
        )
        incoming, leaving = packed(incident), packed(outgoing)
        zero_middle = tf.zeros_like(incoming)
        tmp.k_i = tf.stack([incoming, zero_middle, leaving], axis=0)
        tmp.k_r = tf.stack([zero_middle, leaving], axis=0)
        tmp.k_tx, tmp.k_rx = incoming, -leaving
        tmp.total_distance = packed(total_distance)
        # Retain the legacy diagnostic attribute in [TX, RX, slot] order.
        vertex_ids = tf.tensor_scatter_nd_update(tf.fill(shape, -1), indices, rows[:, 4])
        self.valid_vertex_idxs = tf.transpose(vertex_ids, [1, 0, 2])
        return paths, tmp

    def trace_mixed_flat(self, sources, targets, rows, vertex_depth, junction_enabled=None):
        """Trace EV/VE rows [N, 4] in propagation order [tx, rx, object1, object2].

        vertex_depth=1 denotes EV; vertex_depth=0 denotes VE. Constituent
        edges remain [M] rows until the final public mask is assembled.
        """
        if vertex_depth not in (0, 1):
            raise ValueError("vertex_depth must be 0 for VE or 1 for EV.")
        solver = self.solver_paths
        if self.is_only_in_nlos:
            links = tf.stack([rows[:, 1], rows[:, 0]], axis=1)
            rows = tf.boolean_mask(rows, ~tf.gather_nd(solver.is_los[..., 0], links))
        vertices = rows[:, 2 + vertex_depth]
        wedges = rows[:, 3 - vertex_depth]
        vp = tf.gather(self.vertices, vertices)
        tx, rx = tf.gather(sources, rows[:, 0]), tf.gather(targets, rows[:, 1])
        # Both mechanisms reduce to the same finite-edge stationary-point problem.
        before, after = (tx, vp) if vertex_depth == 1 else (vp, rx)
        qp, valid = solver.solver_wedge_diffraction._flat_diffraction_points(before, after, wedges)
        points = tf.stack([qp, vp] if vertex_depth == 1 else [vp, qp], axis=1)
        # Preserve omit_identical_points() and close_2_transition() thresholds.
        valid &= tf.norm(points[:, 1] - points[:, 0], axis=-1) > 2e-2
        rows, points, tx, rx = [tf.boolean_mask(x, valid) for x in (rows, points, tx, rx)]
        if self.is_only_close_2_transition:
            incoming, _ = normalize(points[:, 0] - tx)
            middle, distance = normalize(points[:, 1] - points[:, 0])
            outgoing, _ = normalize(rx - points[:, 1])
            valid = ((dot(incoming, middle) > self.close_2_transition_eps)
                     & (dot(middle, outgoing) > self.close_2_transition_eps)) | (distance < 1e-3)
            rows, points, tx, rx = [tf.boolean_mask(x, valid) for x in (rows, points, tx, rx)]

        edge_paths, _, edge_slots = self.mixed_vertex_edges_flat(
            tx, rx, rows[:, 2:4], points, vertex_depth, junction_enabled
        )
        counts = tf.math.unsorted_segment_sum(tf.ones_like(edge_paths), edge_paths, tf.shape(rows)[0])
        has_edges = counts > 0
        edge_paths = tf.gather(tf.cumsum(tf.cast(has_edges, tf.int32)) - 1, edge_paths)
        rows, points, tx, rx = [tf.boolean_mask(x, has_edges) for x in (rows, points, tx, rx)]
        if vertex_depth == 1:
            valid = self.check_visibility_mixed_flat(
                tx, rx, points, rows[:, 2], rows[:, 3], junction_enabled
            )
        else:
            # Reuse the EV ray conventions with reversed endpoints and interactions.
            valid = self.check_visibility_mixed_flat(
                rx, tx, points[:, ::-1], rows[:, 3], rows[:, 2], junction_enabled
            )
        keep_edges = tf.gather(valid, edge_paths)
        edge_paths, edge_slots = [tf.boolean_mask(x, keep_edges) for x in (edge_paths, edge_slots)]
        edge_paths = tf.gather(tf.cumsum(tf.cast(valid, tf.int32)) - 1, edge_paths)
        rows, points = tf.boolean_mask(rows, valid), tf.boolean_mask(points, valid)
        return self.pack_ee_paths_flat(
            sources, targets, rows, points, vertex_depth=vertex_depth,
            edge_paths=edge_paths, edge_slots=edge_slots,
        )

    def mixed_vertex_edges_flat(self, sources, targets, pairs, points, vertex_depth,
                                junction_enabled=None):
        """Return allowed [M] path IDs, constituent wedge IDs and local slots.

        sources/targets: [N, 3]; pairs: [N, 2] in propagation order;
        points: [N, 2, 3]. These are the existing _vertex_wedge_mask rules.
        """
        solver = self.solver_paths
        vertex_ids = pairs[:, vertex_depth]
        edge_ids = pairs[:, 1 - vertex_depth]
        incidence = tf.gather(solver.vertex_diffraction.vertices_2_wedges, vertex_ids)
        path_ids = tf.cast(incidence.value_rowids(), tf.int32)
        wedges = tf.cast(incidence.values, tf.int32)
        slots = tf.range(tf.size(wedges)) - tf.repeat(
            tf.cast(incidence.row_splits[:-1], tf.int32), tf.cast(incidence.row_lengths(), tf.int32)
        )
        real = wedges >= 0
        path_ids, wedges, slots = [tf.boolean_mask(x, real) for x in (path_ids, wedges, slots)]
        origins = tf.gather(solver._wedges_origin, wedges)
        normals = tf.gather(solver._wedges_normals, wedges)
        concave = tf.gather(solver._is_concave_wedge, wedges)
        is_edge = tf.gather(solver._is_edge, wedges)
        eps = tf.cast(solver.EPSILON, origins.dtype)
        if vertex_depth == 1:
            before, after = points[:, 0], targets
        else:
            before, after = sources, points[:, 1]
        d_before = dot((tf.gather(before, path_ids) - origins)[:, None, :], normals)
        d_after = dot((tf.gather(after, path_ids) - origins)[:, None, :], normals)

        def outside(signed):
            half_spaces = signed >= -eps
            return tf.where(concave, tf.reduce_all(half_spaces, axis=1),
                            tf.reduce_any(half_spaces, axis=1)) | is_edge

        allowed = outside(d_before) & outside(d_after)
        if junction_enabled is not None:
            crossed = solver.solver_wedge_diffraction._opposite_sides(d_before[:, 0], d_after[:, 0])
            allowed |= self._junction_candidates(wedges, junction_enabled) & crossed
        ordinary_edges = tf.gather(edge_ids, path_ids)
        edge_tangent = tf.gather(solver._wedges_e_hat, ordinary_edges)
        vertex_tangent = tf.gather(solver._wedges_e_hat, wedges)
        direction, _ = normalize(points[:, 1] - points[:, 0])
        triple = dot(cross(edge_tangent, vertex_tangent), tf.gather(direction, path_ids))
        allowed &= (tf.abs(triple) < self.coplanar_eps) & (ordinary_edges != wedges)
        return tuple(tf.boolean_mask(x, allowed) for x in (path_ids, wedges, slots))

    def _mixed_vertex_visibility_data(self, vertex_ids, junction_enabled=None):
        """Compute the legacy mean-normal shift once per distinct used vertex."""
        unique_vertices, inverse = tf.unique(vertex_ids)
        incidence = tf.gather(self.vertices_2_wedges, unique_vertices)
        row_ids = tf.cast(incidence.value_rowids(), tf.int32)
        wedges = tf.cast(incidence.values, tf.int32)
        real = wedges >= 0
        row_ids, wedges = tf.boolean_mask(row_ids, real), tf.boolean_mask(wedges, real)
        normals = tf.gather(self._wedges_normals, wedges)
        concave = tf.gather(self.solver_paths._is_concave_wedge, wedges)
        normals = tf.where(concave[:, None, None], -normals, normals)
        n = tf.size(unique_vertices)
        sums = tf.math.unsorted_segment_sum(normals, row_ids, n)
        counts = tf.math.unsorted_segment_sum(tf.ones_like(row_ids), row_ids, n)
        mean = tf.math.divide_no_nan(sums, tf.cast(counts, sums.dtype)[:, None, None])
        shifts = self.solver_paths.DD_EPSILON_OBSTRUCTION * (mean[:, 0] + mean[:, 1]) / 3.0
        has_junction = tf.zeros([n], tf.bool)
        if junction_enabled is not None:
            flags = tf.cast(self._junction_candidates(wedges, junction_enabled), tf.int32)
            has_junction = tf.math.unsorted_segment_sum(flags, row_ids, n) > 0
        return tf.gather(shifts, inverse), tf.gather(has_junction, inverse)

    def check_visibility_mixed_flat(self, sources, targets, points, wedge_ids, vertex_ids,
                                    junction_enabled=None):
        """Check [N] EV paths; VE calls this on the reversed flat data.

        Keep the old shifted middle ray, fallback ray and junction override.
        Vertex shifts use all incident edges, as in the previous implementation.
        """
        count = tf.shape(points)[0]
        if count == 0:
            return tf.zeros([0], tf.bool)
        solver = self.solver_paths
        direction, distance = normalize(points[:, 0] - sources)
        valid = ~solver._test_obstruction(sources, direction, distance)
        direction, distance = normalize(targets - points[:, 1])
        valid &= ~solver._test_obstruction(points[:, 1], direction, distance, True)
        selected = tf.cast(tf.where(valid)[:, 0], tf.int32)
        if tf.size(selected) == 0:
            return valid
        q = tf.gather(points, selected)
        ids = tf.gather(wedge_ids, selected)
        vertex_shift, special = self._mixed_vertex_visibility_data(
            tf.gather(vertex_ids, selected), junction_enabled
        )
        if junction_enabled is not None:
            special |= self._junction_candidates(ids, junction_enabled)
        middle_valid = tf.zeros([tf.size(selected)], tf.bool)
        ordinary = tf.cast(tf.where(~special)[:, 0], tf.int32)
        if tf.size(ordinary) > 0:
            normals = tf.gather(self._wedges_normals, tf.gather(ids, ordinary))
            concave = tf.gather(solver._is_concave_wedge, tf.gather(ids, ordinary))
            normals = tf.where(concave[:, None, None], -normals, normals)
            shift = solver.DD_EPSILON_OBSTRUCTION * (normals[:, 0] + normals[:, 1]) / 3.0
            qo = tf.gather(q, ordinary)
            vs = tf.gather(vertex_shift, ordinary)
            direction, distance = normalize((qo[:, 1] + vs) - (qo[:, 0] + shift))
            ok = ~solver._test_obstruction_2(qo[:, 0] + shift, direction, distance)
            fallback = tf.cast(tf.where(~ok)[:, 0], tf.int32)
            if tf.size(fallback) > 0:
                extra = ~solver._test_obstruction(
                    tf.gather(qo[:, 0], fallback), tf.gather(direction, fallback), tf.gather(distance, fallback)
                )
                ok = tf.tensor_scatter_nd_update(ok, fallback[:, None], extra)
            middle_valid = tf.tensor_scatter_nd_update(middle_valid, ordinary[:, None], ok)
        special_rows = tf.cast(tf.where(special)[:, 0], tf.int32)
        if tf.size(special_rows) > 0:
            qs = tf.gather(q, special_rows)
            exact = self._unshifted_middle_visibility(qs[:, 0], qs[:, 1])
            middle_valid = tf.tensor_scatter_nd_update(middle_valid, special_rows[:, None], exact)
        return tf.tensor_scatter_nd_update(valid, selected[:, None], middle_valid)

    def filter_ee_paths_flat(self, sources, targets, rows, points, junction_enabled=None):
        """Filter solved EE rows without packing them into per-link slots.

        rows: [N, 4], columns [tx, rx, wedge1, wedge2].
        points: [N, 2, 3], ordered stationary points [Q1, Q2].
        Returns surviving rows and points in their original relative order.
        """
        if tf.shape(rows)[0] == 0:
            return rows, points
        wedges = rows[:, 2:4]
        origins = tf.gather(self._wedges_origin, wedges)
        tangents = tf.gather(self._wedges_e_hat, wedges)
        lengths = tf.gather(self._wedges_length, wedges)
        ends = origins + tangents * lengths[..., None]
        # Retain the previous finite-edge test, including its endpoint tolerance.
        valid = tf.reduce_all(dot(origins - points, ends - points) <= 0.0, axis=1)
        rows, points = tf.boolean_mask(rows, valid), tf.boolean_mask(points, valid)
        if tf.shape(rows)[0] == 0:
            return rows, points
        tx, rx = tf.gather(sources, rows[:, 0]), tf.gather(targets, rows[:, 1])

        if self.is_only_close_2_transition:
            incoming, _ = normalize(points[:, 0] - tx)
            middle, distance = normalize(points[:, 1] - points[:, 0])
            outgoing, _ = normalize(rx - points[:, 1])
            valid = ((dot(incoming, middle) > self.close_2_transition_eps)
                     & (dot(middle, outgoing) > self.close_2_transition_eps)) | (distance < 1e-3)
            rows, points, tx, rx = [tf.boolean_mask(x, valid) for x in (rows, points, tx, rx)]
            if tf.shape(rows)[0] == 0:
                return rows, points

        valid = self.check_ee_half_spaces_flat(tx, rx, points, rows[:, 2:4], junction_enabled)
        rows, points, tx, rx = [tf.boolean_mask(x, valid) for x in (rows, points, tx, rx)]
        if tf.shape(rows)[0] == 0:
            return rows, points
        valid = self.check_ee_visibility_flat(tx, rx, points, rows[:, 2:4], junction_enabled)
        return tf.boolean_mask(rows, valid), tf.boolean_mask(points, valid)

    def check_ee_half_spaces_flat(self, sources, targets, points, wedges, junction_enabled=None):
        """Return [N] sector validity using the actual adjacent 3D segments.

        sources and targets: [N, 3], one pair of devices per path.
        points: [N, 2, 3]; wedges: [N, 2], without padding.
        """
        eps = tf.cast(self.solver_paths.EPSILON, points.dtype)
        middle = points[:, 1] - points[:, 0]

        def allowed(ids, outer, inner):
            normals = tf.gather(self._wedges_normals, ids)
            d_outer = dot(outer[:, None, :], normals)
            d_inner = dot(inner[:, None, :], normals)
            valid = tf.reduce_any(d_outer > eps, axis=-1) & tf.reduce_any(d_inner > -eps, axis=-1)
            if junction_enabled is not None:
                wd = self.solver_paths.solver_wedge_diffraction
                crossed = wd._is_flat_shared_wedge(ids) & wd._opposite_sides(d_outer[:, 0], d_inner[:, 0])
                valid = tf.where(crossed, self._junction_candidates(ids, junction_enabled), valid)
            return valid | tf.gather(self.solver_paths._is_edge, ids)

        return (allowed(wedges[:, 0], sources - points[:, 0], middle)
                & allowed(wedges[:, 1], targets - points[:, 1], -middle))

    def check_ee_visibility_flat(self, sources, targets, points, wedges, junction_enabled=None):
        """Check TX-Q1-Q2-RX only for actual candidates [N].

        Preserve both existing middle-segment ray conventions and the exact
        unshifted override for supported metal-glass junctions. The fallback
        middle ray is needed only when the first ray failed.
        """
        count = tf.shape(points)[0]
        if count == 0:
            return tf.zeros([0], tf.bool)
        solver = self.solver_paths
        incoming, r1 = normalize(points[:, 0] - sources)
        outgoing, r2 = normalize(targets - points[:, 1])
        valid = ~solver._test_obstruction(sources, incoming, r1)
        valid &= ~solver._test_obstruction(points[:, 1], outgoing, r2, True)
        selected = tf.cast(tf.where(valid)[:, 0], tf.int32)
        if tf.size(selected) == 0:
            return valid

        selected_points = tf.gather(points, selected)
        selected_wedges = tf.gather(wedges, selected)
        special = tf.zeros([tf.size(selected)], tf.bool)
        if junction_enabled is not None:
            special = tf.reduce_any(self._junction_candidates(selected_wedges, junction_enabled), axis=-1)
        middle_valid = tf.zeros([tf.size(selected)], tf.bool)
        ordinary_ids = tf.cast(tf.where(~special)[:, 0], tf.int32)
        if tf.size(ordinary_ids) > 0:
            q = tf.gather(selected_points, ordinary_ids)
            ids = tf.gather(selected_wedges, ordinary_ids)
            normals = tf.gather(self._wedges_normals, ids)
            concave = tf.gather(solver._is_concave_wedge, ids)
            normals = tf.where(concave[..., None, None], -normals, normals)
            shifts = solver.DD_EPSILON_OBSTRUCTION * (normals[:, :, 0] + normals[:, :, 1]) / 3.0
            direction, distance = normalize((q[:, 1] + shifts[:, 1]) - (q[:, 0] + shifts[:, 0]))
            ordinary_valid = ~solver._test_obstruction_2(q[:, 0] + shifts[:, 0], direction, distance)
            fallback = tf.cast(tf.where(~ordinary_valid)[:, 0], tf.int32)
            if tf.size(fallback) > 0:
                # Keep the original unshifted origin and shifted direction/length.
                extra = ~solver._test_obstruction(
                    tf.gather(q[:, 0], fallback), tf.gather(direction, fallback), tf.gather(distance, fallback)
                )
                ordinary_valid = tf.tensor_scatter_nd_update(ordinary_valid, fallback[:, None], extra)
            middle_valid = tf.tensor_scatter_nd_update(middle_valid, ordinary_ids[:, None], ordinary_valid)

        special_ids = tf.cast(tf.where(special)[:, 0], tf.int32)
        if tf.size(special_ids) > 0:
            q = tf.gather(selected_points, special_ids)
            exact = self._unshifted_middle_visibility(q[:, 0], q[:, 1])
            middle_valid = tf.tensor_scatter_nd_update(middle_valid, special_ids[:, None], exact)
        return tf.tensor_scatter_nd_update(valid, selected[:, None], middle_valid)

    def pack_ee_paths_flat(self, sources, targets, rows, points, vertex_depth=None,
                           edge_paths=None, edge_slots=None):
        """Compute EE geometry on [N] paths, then pack the surviving results.

        The public layout remains [RX, TX, slot]. Duplicate detection keeps
        the last same-angle path within a link, as in the previous tracer.
        No padded candidate tensors are created before this final assembly.
        Optional vertex_depth packs EV/VE and their constituent-edge mask;
        those paths retain the previous no-deduplication convention.
        """
        solver = self.solver_paths
        paths = Paths(sources, targets, solver._scene, types=Paths.DOUBLE_DIFF)
        tmp = PathsTmpData(sources, targets, self._dtype)
        num_rx, num_tx = tf.shape(targets)[0], tf.shape(sources)[0]
        link_ids = rows[:, 1] * num_tx + rows[:, 0]
        order = tf.argsort(link_ids, stable=True)
        input_ids = order
        rows, points = tf.gather(rows, order), tf.gather(points, order)
        tx, rx = tf.gather(sources, rows[:, 0]), tf.gather(targets, rows[:, 1])
        incoming, r1 = normalize(points[:, 0] - tx)
        middle, distance = normalize(points[:, 1] - points[:, 0])
        outgoing, r2 = normalize(rx - points[:, 1])
        theta_t, phi_t = theta_phi_from_unit_vec(incoming)
        theta_r, phi_r = theta_phi_from_unit_vec(-outgoing)
        link_ids = rows[:, 1] * num_tx + rows[:, 0]
        keep = (solver.solver_wedge_diffraction._flat_unique_paths(
            link_ids, tf.stack([theta_t, phi_t, theta_r, phi_r], axis=1)
        ) if vertex_depth is None else tf.ones_like(link_ids, tf.bool))
        input_ids = tf.boolean_mask(input_ids, keep)
        rows, points, incoming, middle, outgoing, r1, distance, r2, theta_t, phi_t, theta_r, phi_r = [
            tf.boolean_mask(x, keep) for x in (
                rows, points, incoming, middle, outgoing, r1, distance, r2, theta_t, phi_t, theta_r, phi_r
            )
        ]
        link_ids = rows[:, 1] * num_tx + rows[:, 0]
        counts = tf.math.unsorted_segment_sum(tf.ones_like(link_ids), link_ids, num_rx * num_tx)
        offsets = tf.cumsum(counts, exclusive=True)
        slots = tf.range(tf.size(link_ids)) - tf.gather(offsets, link_ids)
        indices = tf.stack([rows[:, 1], rows[:, 0], slots], axis=1)
        max_paths = tf.reduce_max(tf.concat([counts, tf.zeros([1], tf.int32)], axis=0))
        shape = tf.stack([num_rx, num_tx, max_paths])

        def packed(values):
            return tf.scatter_nd(indices, values, tf.concat([shape, tf.shape(values)[1:]], axis=0))

        paths.objects = tf.stack([
            tf.tensor_scatter_nd_update(tf.fill(shape, -1), indices, rows[:, 2]),
            tf.tensor_scatter_nd_update(tf.fill(shape, -1), indices, rows[:, 3]),
        ], axis=0)
        paths.vertices = tf.stack([packed(points[:, 0]), packed(points[:, 1])], axis=0)
        paths.mask = packed(tf.ones_like(link_ids, tf.bool))
        paths.depth_mask = paths.objects >= 0
        paths.dd_type = tf.zeros(shape, tf.int32)
        paths.my_types = tf.fill(tf.shape(paths.objects), Paths.DIFFRACTED)
        paths.path_types = tf.fill(tf.shape(paths.objects), Paths.DIFFRACTED)
        if vertex_depth is None:
            paths.true_objects = tf.gather(solver._wedges_objects[:, 0], tf.maximum(paths.objects, 0))
        else:
            vertex_objects = tf.gather(solver.vertex_diffraction.vertices_2_objects_sionna,
                                       tf.maximum(paths.objects[vertex_depth], 0))
            edge_objects = tf.gather(solver._wedges_objects[:, 0],
                                     tf.maximum(paths.objects[1 - vertex_depth], 0))
            paths.true_objects = tf.stack(
                [edge_objects, vertex_objects] if vertex_depth == 1 else [vertex_objects, edge_objects], axis=0
            )
            paths.dd_type = tf.fill(shape, 1 if vertex_depth == 1 else 2)
            paths.my_types = tf.stack([
                tf.fill(shape, Paths.VERTEX_DIFFRACTED if depth == vertex_depth else Paths.DIFFRACTED)
                for depth in (0, 1)
            ], axis=0)
            paths.path_types = paths.my_types
            mapping = tf.scatter_nd(input_ids[:, None], indices, [tf.shape(rows)[0], 3])
            mask_indices = tf.concat([tf.gather(mapping, edge_paths), edge_slots[:, None]], axis=1)
            max_edges = tf.reduce_max(tf.concat([
                tf.cast(solver.vertex_diffraction.vertices_2_wedges.row_lengths(), tf.int32),
                tf.zeros([1], tf.int32),
            ], axis=0))
            paths.vertex_wedge_mask = tf.scatter_nd(
                mask_indices, tf.ones_like(edge_paths, tf.bool),
                tf.concat([shape, [max_edges]], axis=0),
            )
        paths.theta_t, paths.phi_t = packed(theta_t), packed(phi_t)
        paths.theta_r, paths.phi_r = packed(theta_r), packed(phi_r)
        total_distance = tf.reduce_sum(tf.stack([r1, distance, r2], axis=0), axis=0)
        paths.tau = tf.tensor_scatter_nd_update(
            -tf.ones(shape, self._rdtype), indices, total_distance / SPEED_OF_LIGHT
        )
        tmp.k_i = tf.stack([packed(incoming), packed(middle), packed(outgoing)], axis=0)
        tmp.k_r = tf.stack([packed(middle), packed(outgoing)], axis=0)
        tmp.k_tx, tmp.k_rx = packed(incoming), packed(-outgoing)
        tmp.total_distance = packed(total_distance)
        return paths, tmp

    def check_ee_half_spaces(self, sources, targets, diff_points, wedge_pairs,
                             active, junction_enabled=None):
        """Validate both EE wedges using the actual adjacent ray segments.

        diff_points: [TX, RX, slot, 2, 3].
        wedge_pairs: [TX, RX, slot, 2].
        active and returned mask: [TX, RX, slot].
        """
        rows = tf.cast(tf.where(active), tf.int32)
        w1, w2 = tf.unstack(tf.gather_nd(wedge_pairs, rows), axis=-1)
        q1, q2 = tf.unstack(tf.gather_nd(diff_points, rows), axis=1)
        tx = tf.gather(sources, rows[:, 0])
        rx = tf.gather(targets, rows[:, 1])
        middle = q2 - q1

        solver = self.solver_paths
        eps = tf.cast(solver.EPSILON, diff_points.dtype)

        def allowed(wedges, outer, inner):
            normals = tf.gather(self._wedges_normals, wedges)
            d_outer = dot(outer[:, None, :], normals)  # [N, 2]
            d_inner = dot(inner[:, None, :], normals)  # [N, 2]

            # Retain the existing tolerances for outer and middle segments.
            ok = tf.reduce_any(d_outer > eps, axis=-1)
            ok &= tf.reduce_any(d_inner > -eps, axis=-1)

            if junction_enabled is not None:
                wd = solver.solver_wedge_diffraction
                crossed_flat = wd._is_flat_shared_wedge(wedges) & wd._opposite_sides(
                    d_outer[:, 0], d_inner[:, 0]
                )
                eligible = self._junction_candidates(wedges, junction_enabled)
                ok = tf.where(crossed_flat, eligible, ok)

            return ok | tf.gather(solver._is_edge, wedges)

        valid = allowed(w1, tx - q1, middle) & allowed(w2, rx - q2, -middle)
        return tf.scatter_nd(rows, valid, tf.shape(active))

    def _vertex_wedge_mask(self, paths, vertex_depth, junction_enabled=None):
        """Return [targets, sources, paths, vertex_wedges] validity."""
        if vertex_depth not in (0, 1):
            raise ValueError("vertex_depth must be 0 for VE or 1 for EV.")

        solver = self.solver_paths
        wedge_table = solver.vertex_diffraction.vertices_2_wedges.to_tensor(default_value=-1)
        num_slots = tf.shape(wedge_table)[1]
        output_shape = tf.concat([tf.shape(paths.mask), [num_slots]], axis=0)

        vertex_ids = paths.objects[vertex_depth]
        edge_ids = paths.objects[1 - vertex_depth]
        valid_paths = paths.mask & (vertex_ids >= 0) & (edge_ids >= 0)
        path_indices = tf.cast(tf.where(valid_paths), tf.int32)

        if tf.shape(path_indices)[0] == 0 or num_slots == 0:
            return tf.zeros(output_shape, dtype=tf.bool)

        vertex_ids = tf.gather_nd(vertex_ids, path_indices)
        edge_ids = tf.gather_nd(edge_ids, path_indices)
        vertex_edge_ids = tf.gather(wedge_table, vertex_ids)

        valid_wedges = vertex_edge_ids >= 0
        safe_ids = tf.where(valid_wedges, vertex_edge_ids, tf.zeros_like(vertex_edge_ids))

        q0 = tf.gather_nd(paths.vertices[0], path_indices)
        q1 = tf.gather_nd(paths.vertices[1], path_indices)

        if vertex_depth == 1:
            # EV: the incident segment at the vertex starts at the edge point.
            before = q0
            after = tf.gather(paths.targets, path_indices[:, 0])
        else:
            # VE: the outgoing segment at the vertex ends at the edge point.
            before = tf.gather(paths.sources, path_indices[:, 1])
            after = q1

        origins = tf.gather(solver._wedges_origin, safe_ids)
        normals = tf.gather(solver._wedges_normals, safe_ids)
        is_concave = tf.gather(solver._is_concave_wedge, safe_ids)
        is_edge = tf.gather(solver._is_edge, safe_ids)
        eps = tf.cast(solver.EPSILON, origins.dtype)

        def outside_wedge(points):
            delta = tf.cast(points, origins.dtype)[:, None, :] - origins
            half_spaces = tf.reduce_sum(delta[..., None, :] * normals, axis=-1) >= -eps
            outside = tf.where(is_concave, tf.reduce_all(half_spaces, axis=-1),
                            tf.reduce_any(half_spaces, axis=-1))
            return outside | is_edge

        mask = valid_wedges & outside_wedge(before) & outside_wedge(after)

        if junction_enabled is not None:
            plane_normal = normals[..., 0, :]
            d_before = dot(before[:, None, :] - origins, plane_normal)
            d_after = dot(after[:, None, :] - origins, plane_normal)
            wd = solver.solver_wedge_diffraction
            crossed = wd._opposite_sides(d_before, d_after)
            eligible = self._junction_candidates(safe_ids, junction_enabled)
            mask |= valid_wedges & eligible & crossed

        # Reversing an edge tangent does not change this coplanarity test.
        edge_tangent = tf.gather(solver._wedges_e_hat, edge_ids)
        vertex_edge_tangent = tf.gather(solver._wedges_e_hat, safe_ids)
        l_hat, _ = normalize(tf.cast(q1 - q0, origins.dtype))

        triple_product = tf.reduce_sum(
            cross(edge_tangent[:, None, :], vertex_edge_tangent) * l_hat[:, None, :], axis=-1
        )
        coplanar = tf.abs(triple_product) < self.coplanar_eps
        different_edges = edge_ids[:, None] != vertex_edge_ids
        mask &= coplanar & different_edges

        return tf.scatter_nd(path_indices, mask, output_shape)

    def discard_obstructing_dd_wedge_vertex(self, candidate_pairs, sources, targets, junction_enabled=None):
        r"""
        Discard wedges for which at least one of the source or target are
        "inside" the wedge

        Inputs
        ------
        candidate_pairs : [num_dd_candidates, 2], int
            wedge, vertex index pairs.

        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        targets : [num_targets, 3], tf.float
            Coordinates of the targets.   

        Output
        -------
        pairs : [num_targets, num_sources, max_num_paths], tf.int
            Indices of the wedges that interacted with the diffracted paths
        """

        num_sources = tf.shape(sources)[0]
        num_targets = tf.shape(targets)[0]

        ## find edges midpoints and use them for obtruction as well
        # w1_origin = tf.gather(self._wedges_origin, candidate_pairs[:, 0], axis=0)
        v_origin = tf.gather(self.vertices, candidate_pairs[:, 1], axis=0)
        
        ## 1) prepare check points and wedges
        # [num_sources, num_dd_candidates]
        # _, mask1 = self._discard_obstructing_wedges(candidate_pairs[:, 0], sources)
        _, mask11 = self._discard_obstructing_wedges(candidate_pairs[:, 0], sources)
        mask12 = self._discard_obstructing_wedges2(candidate_pairs[:, 0], v_origin)
        mask1 = tf.logical_and(mask11, mask12[None, :])
        if junction_enabled is not None:
            mask1 |= self._junction_candidates(candidate_pairs[:, 0], junction_enabled)[None, :]


        # [num_sources, num_targets, num_candidates]
        mask1 = tf.expand_dims(mask1, axis=1)
        mask1 = tf.repeat(mask1, num_targets, axis=1)

        # [num_targets, num_dd_candidates]
        mask2 = self.solver_paths.vertex_diffraction._discard_nonvisible(candidate_pairs[:, 1], targets)
        # [num_sources, num_targets, num_candidates]
        mask2 = tf.expand_dims(mask2, axis=0)
        mask2 = tf.repeat(mask2, num_sources, axis=0)

        # [num_sources, num_targets, num_candidates]
        pairs = candidate_pairs[None, None, ...]
        pairs = tf.repeat(pairs, num_sources, axis=0)
        pairs = tf.repeat(pairs, num_targets, axis=1)

        # [num_sources, num_targets, num_candidates]
        mask = tf.logical_and(mask1, mask2)
        pairs = tf.where(mask[..., None], candidate_pairs, -1)

        gather_indices = tf.where(mask)
        # [num_sources, num_targets, max_num_candidates]
        path_indices = tf.cumsum(tf.cast(mask, tf.int32), axis=-1)
        # [num_valid_paths]
        path_indices2 = tf.gather_nd(path_indices, gather_indices) - 1
        scatter_indices = tf.transpose(gather_indices, [1,0])
        if not tf.size(scatter_indices) == 0:
            new_scatter_indices = tf.tensor_scatter_nd_update(scatter_indices,
                                [[2]], [path_indices2])
        # [num_valid_paths, 3]
        new_scatter_indices = tf.transpose(new_scatter_indices, [1,0])

        # [num_sources, num_targets]
        num_paths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
        # Maximum number of valid paths
        # ()
        max_num_paths = tf.reduce_max(num_paths)

        # [num_sources, num_targets, max_num_paths]
        new_mask = tf.fill([num_sources, num_targets, max_num_paths], False)
        # [num_keep_paths]
        mask_ = tf.gather_nd(mask, gather_indices)
        # [num_sources, num_targets, max_num_paths]
        new_mask = tf.tensor_scatter_nd_update(new_mask, new_scatter_indices, mask_)
        valid_pairs = -1 * tf.ones([num_sources, num_targets, max_num_paths, 2], dtype=tf.int32)

        pairs_ = tf.gather_nd(pairs, gather_indices)
        new_scatter_indices_0 = tf.pad(new_scatter_indices, [[0, 0], [0, 1]], constant_values=0)
        new_scatter_indices_1 = tf.pad(new_scatter_indices, [[0, 0], [0, 1]], constant_values=1)
        # [max_depth, num_sources, num_targets, max_num_paths]
        valid_pairs = tf.tensor_scatter_nd_update(valid_pairs,
                                new_scatter_indices_0, pairs_[:, 0])
        valid_pairs = tf.tensor_scatter_nd_update(valid_pairs,
                                new_scatter_indices_1, pairs_[:, 1])

        return valid_pairs, new_mask

    def discard_obstructing_dd_vertex_wedge(self, candidate_pairs, sources, targets, junction_enabled=None):
        # TODO: make function from above general ?
        r"""

        Inputs
        ------
        candidate_pairs : [num_dd_candidates, 2], int
            wedge, vertex index pairs.

        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        targets : [num_targets, 3], tf.float
            Coordinates of the targets.   

        Output
        -------
        pairs : [num_targets, num_sources, max_num_paths], tf.int
            Indices of the wedges that interacted with the diffracted paths
        """

        num_sources = tf.shape(sources)[0]
        num_targets = tf.shape(targets)[0]

        ## find edges midpoints and use them for obtruction as well
        v1_origin = tf.gather(self.vertices, candidate_pairs[:, 0], axis=0)
        # w2_origin = tf.gather(self.vertices, candidate_pairs[:, 1], axis=0)
        
        ## 1) prepare check points and wedges
        # [num_targets, num_dd_candidates]
        _, mask11 = self._discard_obstructing_wedges(candidate_pairs[:, 1], targets)
        mask12 = self._discard_obstructing_wedges2(candidate_pairs[:, 1], v1_origin)
        mask1 = tf.logical_and(mask11, mask12[None, :])
        if junction_enabled is not None:
            mask1 |= self._junction_candidates(candidate_pairs[:, 1], junction_enabled)[None, :]


        # [num_sources, num_targets, num_candidates]
        mask1 = tf.expand_dims(mask1, axis=0)
        mask1 = tf.repeat(mask1, num_sources, axis=0)

        # [num_sources, num_dd_candidates]
        mask2 = self.solver_paths.vertex_diffraction._discard_nonvisible(candidate_pairs[:, 0], sources)
        # [num_sources, num_targets, num_candidates]
        mask2 = tf.expand_dims(mask2, axis=1)
        mask2 = tf.repeat(mask2, num_targets, axis=1)

        # [num_sources, num_targets, num_candidates]
        pairs = candidate_pairs[None, None, ...]
        pairs = tf.repeat(pairs, num_sources, axis=0)
        pairs = tf.repeat(pairs, num_targets, axis=1)

        # [num_sources, num_targets, num_candidates]
        mask = tf.logical_and(mask1, mask2)
        pairs = tf.where(mask[..., None], candidate_pairs, -1)

        gather_indices = tf.where(mask)
        # [num_sources, num_targets, max_num_candidates]
        path_indices = tf.cumsum(tf.cast(mask, tf.int32), axis=-1)
        # [num_valid_paths]
        path_indices2 = tf.gather_nd(path_indices, gather_indices) - 1
        scatter_indices = tf.transpose(gather_indices, [1,0])
        if not tf.size(scatter_indices) == 0:
            new_scatter_indices = tf.tensor_scatter_nd_update(scatter_indices,
                                [[2]], [path_indices2])
        # [num_valid_paths, 3]
        new_scatter_indices = tf.transpose(new_scatter_indices, [1,0])

        # [num_sources, num_targets]
        num_paths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
        # Maximum number of valid paths
        # ()
        max_num_paths = tf.reduce_max(num_paths)

        # [num_sources, num_targets, max_num_paths]
        new_mask = tf.fill([num_sources, num_targets, max_num_paths], False)
        # [num_keep_paths]
        mask_ = tf.gather_nd(mask, gather_indices)
        # [num_sources, num_targets, max_num_paths]
        new_mask = tf.tensor_scatter_nd_update(new_mask, new_scatter_indices, mask_)
        valid_pairs = -1 * tf.ones([num_sources, num_targets, max_num_paths, 2], dtype=tf.int32)

        pairs_ = tf.gather_nd(pairs, gather_indices)
        new_scatter_indices_0 = tf.pad(new_scatter_indices, [[0, 0], [0, 1]], constant_values=0)
        new_scatter_indices_1 = tf.pad(new_scatter_indices, [[0, 0], [0, 1]], constant_values=1)
        # [max_depth, num_sources, num_targets, max_num_paths]
        valid_pairs = tf.tensor_scatter_nd_update(valid_pairs,
                                new_scatter_indices_0, pairs_[:, 0])
        valid_pairs = tf.tensor_scatter_nd_update(valid_pairs,
                                new_scatter_indices_1, pairs_[:, 1])

        return valid_pairs, new_mask

    def discard_obstructing_dd_vertex_vertex(self, candidate_pairs, sources, targets):
        r"""
        Discard wedges for which at least one of the source or target are
        "inside" the wedge

        Inputs
        ------
        candidate_pairs : [num_dd_candidates, 2], int
            wedge, vertex index pairs.

        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        targets : [num_targets, 3], tf.float
            Coordinates of the targets.   

        Output
        -------
        pairs : [num_targets, num_sources, max_num_paths], tf.int
            Indices of the wedges that interacted with the diffracted paths
        """

        num_sources = tf.shape(sources)[0]
        num_targets = tf.shape(targets)[0]
        
        # [num_sources, num_dd_candidates]
        mask1 = self.solver_paths.vertex_diffraction._discard_nonvisible(candidate_pairs[:, 0], sources)
        # [num_sources, num_targets, num_candidates]
        mask1 = tf.expand_dims(mask1, axis=1)
        mask1 = tf.repeat(mask1, num_targets, axis=1)

        ## 2) vertex2
        # [num_targets, num_dd_candidates]
        mask2 = self.solver_paths.vertex_diffraction._discard_nonvisible(candidate_pairs[:, 1], targets)
        # [num_sources, num_targets, num_candidates]
        mask2 = tf.expand_dims(mask2, axis=0)
        mask2 = tf.repeat(mask2, num_sources, axis=0)

        # [num_sources, num_targets, num_candidates]
        pairs = candidate_pairs[None, None, ...]
        pairs = tf.repeat(pairs, num_sources, axis=0)
        pairs = tf.repeat(pairs, num_targets, axis=1)

        # [num_sources, num_targets, num_candidates]
        mask = tf.logical_and(mask1, mask2)
        pairs = tf.where(mask[..., None], candidate_pairs, -1)

        gather_indices = tf.where(mask)
        # [num_sources, num_targets, max_num_candidates]
        path_indices = tf.cumsum(tf.cast(mask, tf.int32), axis=-1)
        # [num_valid_paths]
        path_indices2 = tf.gather_nd(path_indices, gather_indices) - 1
        scatter_indices = tf.transpose(gather_indices, [1,0])
        if not tf.size(scatter_indices) == 0:
            new_scatter_indices = tf.tensor_scatter_nd_update(scatter_indices,
                                [[2]], [path_indices2])
        # [num_valid_paths, 3]
        new_scatter_indices = tf.transpose(new_scatter_indices, [1,0])

        # [num_sources, num_targets]
        num_paths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
        # Maximum number of valid paths
        # ()
        max_num_paths = tf.reduce_max(num_paths)

        # [num_sources, num_targets, max_num_paths]
        new_mask = tf.fill([num_sources, num_targets, max_num_paths], False)
        # [num_keep_paths]
        mask_ = tf.gather_nd(mask, gather_indices)
        # [num_sources, num_targets, max_num_paths]
        new_mask = tf.tensor_scatter_nd_update(new_mask, new_scatter_indices, mask_)
        valid_pairs = -1 * tf.ones([num_sources, num_targets, max_num_paths, 2], dtype=tf.int32)

        pairs_ = tf.gather_nd(pairs, gather_indices)
        new_scatter_indices_0 = tf.pad(new_scatter_indices, [[0, 0], [0, 1]], constant_values=0)
        new_scatter_indices_1 = tf.pad(new_scatter_indices, [[0, 0], [0, 1]], constant_values=1)
        # [max_depth, num_sources, num_targets, max_num_paths]
        valid_pairs = tf.tensor_scatter_nd_update(valid_pairs,
                                new_scatter_indices_0, pairs_[:, 0])
        valid_pairs = tf.tensor_scatter_nd_update(valid_pairs,
                                new_scatter_indices_1, pairs_[:, 1])

        return valid_pairs, new_mask
    
    def get_pairs(self, wedge_idxs_s, wedge_idxs_t):
        r"""
        Get the pairs of wedges that are active

        Inputs
        ------
        wedge_idxs_s: [num_wedges], int
            The indices of the target wedges

        wedge_idxs_t: [num_wedges], int
            The indices of the source wedges

        Output
        ------
        pairs: [num_pairs, 2], int
            The pairs of wedges
        """

        # [num_wedges1, num_wedges2, 2]
        pairs = tf.stack(tf.meshgrid(wedge_idxs_s, wedge_idxs_t), axis=-1)

        # [num_pairs, 2]
        pairs = tf.reshape(pairs, (-1, 2))
              
        return pairs

    def get_pairs2(self, wedge_idxs_t, wedge_idxs_s):
        r"""
        Get the pairs of wedges that are active

        Inputs
        ------
        wedge_idxs_t: [num_targets, num_wedges], int
            The indices of the target wedges

        wedge_idxs_s: [num_sources, num_wedges], int
            The indices of the source wedges

        Output
        ------
        pairs: [num_targets, num_sources, num_pairs], bool
            The pairs that are active
        """

        num_targets, num_sources = tf.shape(wedge_idxs_t)[0], tf.shape(wedge_idxs_s)[0]
        num_wedges1, num_wedges2 = tf.shape(wedge_idxs_t)[1], tf.shape(wedge_idxs_s)[1]

        wedge_idxs_t = tf.reshape(wedge_idxs_t, (num_targets*num_wedges1, -1))
        wedge_idxs_s = tf.reshape(wedge_idxs_s, (num_sources*num_wedges2, -1))

        # [num_targets*num_wedges1, num_sources*num_wedges2, 2]
        pairs = tf.stack(tf.meshgrid(wedge_idxs_t, wedge_idxs_s), axis=-1)

        # [num_pairs, 2]
        pairs = tf.reshape(pairs, (-1, 2))

    def compute_diffraction_points_coplanar_flat(
        self, sources, targets, wedge_pairs, parallel_tol=1e-7, keller_tol=1e-6
    ):
        """Solve flat coplanar EE candidates.

        sources, targets: [N, 3], one TX and RX position per candidate.
        wedge_pairs: [N, 2], ordered wedge IDs without padding.
        Returns points [N, 2, 3] and a geometry validity mask [N].
        """
        sources = tf.cast(sources, tf.float64)
        targets = tf.cast(targets, tf.float64)
        origins = tf.cast(tf.gather(self._wedges_origin, wedge_pairs), tf.float64)
        tangents = tf.cast(tf.gather(self._wedges_e_hat, wedge_pairs), tf.float64)
        tangents, _ = normalize(tangents)

        o1, o2 = tf.unstack(origins, axis=1)
        e1, e2 = tf.unstack(tangents, axis=1)
        parallel = tf.linalg.norm(cross(e1, e2), axis=-1) <= parallel_tol

        def select(mask):
            rows = tf.cast(tf.where(mask)[:, 0], tf.int32)
            return rows, [tf.gather(v, rows) for v in (sources, targets, o1, o2, e1, e2)]

        def project(v, e):
            z = dot(v, e)
            h = tf.linalg.norm(v - z[:, None] * e, axis=-1)
            return z, h

        # Parallel lines: unfold the three transverse distances.
        i_par, (tx, rx, p1, p2, v1, v2) = select(parallel)
        z_tx, h_tx = project(tx - p1, v1)
        z_rx_local, h_rx = project(rx - p2, v1)
        z_offset, separation = project(p2 - p1, v1)
        dz = z_offset + z_rx_local - z_tx
        slope = tf.math.divide_no_nan(dz, h_tx + separation + h_rx)

        z1 = z_tx + slope * h_tx
        z2 = z_tx + slope * (h_tx + separation)
        q1_par = p1 + z1[:, None] * v1

        # Place the second point on the actual second line.
        q2_offset = (z2 - z_offset)[:, None] * v1
        q2_par = p2 + dot(q2_offset, v2)[:, None] * v2
        valid_par = separation > 0.0

        # Nonparallel coplanar lines: use their intersection as the origin.
        i_cross, (tx, rx, p1, p2, v1, v2) = select(~parallel)
        axis = cross(v1, v2)
        sin_sq = dot(axis, axis)
        offset = p2 - p1
        t_origin = tf.math.divide_no_nan(dot(cross(offset, v2), axis), sin_sq)
        r_origin = tf.math.divide_no_nan(dot(cross(offset, v1), axis), sin_sq)
        c1 = p1 + t_origin[:, None] * v1
        c2 = p2 + r_origin[:, None] * v2

        z_tx, h_tx = project(tx - c1, v1)
        z_rx, h_rx = project(rx - c2, v2)
        c = dot(v1, v2)[:, None]
        s = tf.sqrt(sin_sq)[:, None]
        z_tx, h_tx = z_tx[:, None], h_tx[:, None]
        z_rx, h_rx = z_rx[:, None], h_rx[:, None]

        # Test the four sign combinations for (r, t).
        sign_r = tf.constant([1.0, 1.0, -1.0, -1.0], tf.float64)
        sign_t = tf.constant([1.0, -1.0, 1.0, -1.0], tf.float64)
        a = h_rx * c + z_rx * s * sign_t
        b = h_tx * c + z_tx * s * sign_r
        denominator = s * (h_rx * sign_r + b * sign_t)
        r = tf.math.divide_no_nan(a * b - h_tx * h_rx, denominator)
        t = tf.math.divide_no_nan(
            h_tx * c * r + z_tx * s * tf.abs(r), h_tx + s * tf.abs(r)
        )

        quadrant_ok = (denominator != 0.0) & (r * sign_r > 0.0) & (t * sign_t > 0.0)
        best = tf.argmax(tf.cast(quadrant_ok, tf.int32), axis=1, output_type=tf.int32)
        t = tf.gather(t, best, axis=1, batch_dims=1)
        r = tf.gather(r, best, axis=1, batch_dims=1)
        q1_cross = c1 + t[:, None] * v1
        q2_cross = c2 + r[:, None] * v2
        valid_cross = tf.reduce_any(quadrant_ok, axis=1)

        rows = tf.concat([i_par, i_cross], axis=0)
        q1 = tf.concat([q1_par, q1_cross], axis=0)
        q2 = tf.concat([q2_par, q2_cross], axis=0)
        valid = tf.concat([valid_par, valid_cross], axis=0)

        # Check both Keller conditions on the original 3D geometry.
        u01, l01 = normalize(q1 - tf.gather(sources, rows))
        u12, l12 = normalize(q2 - q1)
        u23, l23 = normalize(tf.gather(targets, rows) - q2)
        r1 = dot(tf.gather(e1, rows), u01 - u12)
        r2 = dot(tf.gather(e2, rows), u12 - u23)
        valid &= (l01 > 0.0) & (l12 > 0.0) & (l23 > 0.0)
        valid &= (tf.abs(r1) <= keller_tol) & (tf.abs(r2) <= keller_tol)

        points = tf.stack([q1, q2], axis=1)
        points = tf.where(valid[:, None, None], points, tf.zeros_like(points))
        n = tf.shape(wedge_pairs)[0]
        return (
            tf.scatter_nd(rows[:, None], tf.cast(points, self._rdtype), [n, 2, 3]),
            tf.scatter_nd(rows[:, None], valid, [n]),
        )
        
    def compute_diffraction_points_analytical(self, sources, targets, wedge_pair_idxs):
        r"""
        Only for double diffraction with coplanar edges (i.e. slope diffraction)

        Input
        ------
        wedge_pair_idxs: [num_sources, num_targets, num_pairs, 2], int

        Output
        ------
        diff_points: [num_sources, num_targets, num_pairs, 2, 3], tf.float
            The diffraction points

        active: [num_sources, num_targets, num_pairs], bool
            The active pairs
        """

        num_sources, num_targets = tf.shape(wedge_pair_idxs)[0], tf.shape(wedge_pair_idxs)[0]
        num_pairs = tf.shape(wedge_pair_idxs)[2]

        # Change -1 by 0
        wedge_pair_idxs = tf.where(wedge_pair_idxs == -1, 0, wedge_pair_idxs)

        # [num_sources, num_targets, num_pairs, 3]
        w1_start_points = tf.gather(self._wedges_origin, wedge_pair_idxs[:, :, :, 0], axis=0)
        w2_start_points = tf.gather(self._wedges_origin, wedge_pair_idxs[:, :, :, 1], axis=0)
        edge1_hat = tf.gather(self._wedges_e_hat, wedge_pair_idxs[:, :, :, 0], axis=0)
        edge2_hat = tf.gather(self._wedges_e_hat, wedge_pair_idxs[:, :, :, 1], axis=0)
        vec_w1_w2 = w2_start_points - w1_start_points

        # [num_sources, num_targets, num_pairs, 3]
        if tf.rank(sources) == 2:
            sources_ext = insert_dim_num(sources, 1, num_targets)
            sources_ext = insert_dim_num(sources_ext, 2, num_pairs)
        else:
            sources_ext = sources

        if tf.rank(targets) == 2:
            targets_ext = insert_dim_num(targets, 0, num_sources)
            targets_ext = insert_dim_num(targets_ext, 2, num_pairs)
        else:
            targets_ext = targets

        cond_is_parallel = tf.abs(dot(edge1_hat, edge2_hat)) > 0.999
        cond_is_parallel_ext = cond_is_parallel[..., None]

        # parallel
        origin_par = w1_start_points
        e1_par = edge1_hat
        e2_par = vec_w1_w2 - tf.expand_dims(dot(vec_w1_w2, edge1_hat), axis=-1) * edge1_hat
        e2_par, _ = normalize(e2_par)
        # e2_par = cross(e1_par, edge2_hat)
        e3_par = cross(e1_par, e2_par)
        dist_par = dot(vec_w1_w2, e2_par)

        # non-parallel
        s = ((dot(vec_w1_w2, edge1_hat) * dot(edge1_hat, edge2_hat)) - dot(vec_w1_w2, edge2_hat)) / (1 - (dot(edge1_hat, edge2_hat))**2)
        t = dot(vec_w1_w2, edge1_hat) + s * dot(edge1_hat, edge2_hat)
        origin_ = w1_start_points + tf.expand_dims(t, axis=-1) * edge1_hat
        #assert tf.reduce_all(tf.abs(origin - (w2_start_points + tf.expand_dims(s, axis=-1) * edge2_hat)) < 1e-4)
        e1_ = edge1_hat   
        e2_ = cross(e1_, edge2_hat)
        e2_, _ = normalize(e2_)
        e3_ = cross(e1_, e2_)
        dist_ = dot(vec_w1_w2, e2_)

        # combine
        origin = tf.where(cond_is_parallel_ext, origin_par, origin_)
        e1 = tf.where(cond_is_parallel_ext, e1_par, e1_)
        e2 = tf.where(cond_is_parallel_ext, e2_par, e2_)
        e3 = tf.where(cond_is_parallel_ext, e3_par, e3_)
        dist = tf.where(cond_is_parallel, dist_par, dist_)

        # [num_sources, num_targets, num_pairs, 3]
        n_dir = dot(edge2_hat, e3)
        l_dir = dot(edge2_hat, e1) 
        _p1 = sources_ext - origin
        _p2 = targets_ext - origin
        p1_local = tf.stack([dot(_p1, e1), dot(_p1, e2), dot(_p1, e3)], axis=-1)
        p2_local = tf.stack([dot(_p2, e1), dot(_p2, e2), dot(_p2, e3)], axis=-1)
        local_T1 = p1_local[..., 0]
        local_T2 = - tf.sqrt(p1_local[..., 1]**2 + p1_local[..., 2]**2)
        local_R1 = l_dir*p2_local[..., 0] + n_dir*p2_local[..., 2]
        tmp_vec0 = p2_local[..., 0] - l_dir*local_R1
        tmp_vec1 = p2_local[..., 1] - dist
        tmp_vec2 = p2_local[..., 2] - n_dir*local_R1
        local_R2 = - tf.sqrt(tmp_vec0*tmp_vec0 + tmp_vec1*tmp_vec1 + tmp_vec2*tmp_vec2)

        def get_solution_parralel():
            """a1 = local_R2*l_dir / (local_R2 - dist)
            a2 = local_T2*l_dir / (local_T2 - dist)
            b1 = local_R1 * dist / (local_R2 - dist)
            b2 = local_T1 * dist / (local_T2 - dist)
            t_sol = (a2*b1 + b2) / (a1*a2 - 1)
            r_sol = a1 * t_sol - b1"""
            # convert from numpy to tf
            a1 = local_R2*l_dir / (local_R2 - dist)
            a2 = local_T2*l_dir / (local_T2 - dist)
            b1 = local_R1 * dist / (local_R2 - dist)
            b2 = local_T1 * dist / (local_T2 - dist)
            t_sol = (a2*b1 + b2) / (a1*a2 - 1)
            r_sol = a1 * t_sol - b1

            return r_sol, t_sol
        
        def get_solution_(sign_r, sign_t):
            # convert from numpy to tf
            abs_n = tf.abs(n_dir)

            tmp11 = local_R2 * local_T2
            tmp12 = local_R2 * l_dir
            tmp13 = local_R1 * abs_n * sign_t
            tmp14 = local_T2 * l_dir
            tmp15 = local_T1 * abs_n * sign_r
            tmp16 = local_R2 * abs_n * sign_r
            tmp17 = abs_n * sign_t

            tmp1 = tmp11 - (tmp12 - tmp13) * (tmp14 - tmp15)
            tmp2 = tmp16 + tmp17 * (tmp14 - tmp15)
            r_sol = tmp1 / tmp2

            tmp21 = local_T2*l_dir*r_sol - local_T1*abs_n*tf.abs(r_sol)
            tmp22 = local_T2 - abs_n*tf.abs(r_sol)
            t_sol = tmp21 / tmp22

            return r_sol, t_sol
        
        r_sol_par, t_sol_par = get_solution_parralel()

        plus_arr = tf.ones_like(n_dir)
        minus_arr = -tf.ones_like(n_dir)

        r_sol_1, t_sol_1 = get_solution_(plus_arr, plus_arr)
        r_sol_2, t_sol_2 = get_solution_(plus_arr, minus_arr)
        r_sol_3, t_sol_3 = get_solution_(minus_arr, plus_arr)
        r_sol_4, t_sol_4 = get_solution_(minus_arr, minus_arr)
        cond_1 = tf.logical_and(r_sol_1 > 1e-4, t_sol_1 > 1e-4)
        cond_2 = tf.logical_and(r_sol_2 > 1e-4, t_sol_2 < -1e-4)
        cond_3 = tf.logical_and(r_sol_3 < -1e-4, t_sol_3 > 1e-4)
        cond_4 = tf.logical_and(r_sol_4 < -1e-4, t_sol_4 < -1e-4)
        # if non of the cond_1 .. _4
        cond_5 = tf.logical_not(tf.logical_or(tf.logical_or(cond_1, cond_2), tf.logical_or(cond_3, cond_4)))

        r_sol_ = tf.where(cond_1, r_sol_1, 
                        tf.where(cond_2, r_sol_2, 
                        tf.where(cond_3, r_sol_3, 
                        tf.where(cond_4, r_sol_4, 
                        tf.where(cond_5, tf.zeros_like(n_dir), tf.zeros_like(n_dir))))))
        
        t_sol_ = tf.where(cond_1, t_sol_1,
                        tf.where(cond_2, t_sol_2,
                        tf.where(cond_3, t_sol_3,
                        tf.where(cond_4, t_sol_4, tf.zeros_like(n_dir)))))
        
        r_sol = tf.where(cond_is_parallel, r_sol_par, r_sol_)
        t_sol = tf.where(cond_is_parallel, t_sol_par, t_sol_)
        new_active = tf.where(cond_is_parallel, tf.ones_like(cond_is_parallel), tf.logical_not(cond_5))

        w1_point_local = tf.stack([t_sol, tf.zeros_like(t_sol), tf.zeros_like(t_sol)], axis=-1)
        w2_point_local = tf.stack([l_dir*r_sol, dist, n_dir*r_sol], axis=-1)

        w1_point = origin + w1_point_local[..., 0:1] * e1 + w1_point_local[..., 1:2] * e2 + w1_point_local[..., 2:3] * e3
        w2_point = origin + w2_point_local[..., 0:1] * e1 + w2_point_local[..., 1:2] * e2 + w2_point_local[..., 2:3] * e3

        return tf.stack([w1_point, w2_point], axis=-2), new_active

    def compute_diffraction_points_noncoplanar_flat(
        self, sources, targets, wedge_pairs, keller_tol=1e-6,
        max_newton_steps=8, max_bisection_steps=40,
    ):
        """Solve flat EE candidates on nondegenerate skew lines.

        sources, targets: [N, 3], one TX and RX position per candidate.
        wedge_pairs: [N, 2], ordered wedge IDs without padding.
        Returns points [N, 2, 3] and an interior-root validity mask [N].
        """
        sources = tf.cast(sources, tf.float64)
        targets = tf.cast(targets, tf.float64)
        origins = tf.cast(tf.gather(self._wedges_origin, wedge_pairs), tf.float64)
        tangents = tf.cast(tf.gather(self._wedges_e_hat, wedge_pairs), tf.float64)
        lengths = tf.cast(tf.gather(self._wedges_length, wedge_pairs), tf.float64)
        tangents, _ = normalize(tangents)

        o1, o2 = tf.unstack(origins, axis=1)
        e1, e2 = tf.unstack(tangents, axis=1)
        length1, length2 = tf.unstack(lengths, axis=1)

        tx_offset = sources - o1
        z_tx = dot(tx_offset, e1)
        h_tx = tf.linalg.norm(tx_offset - z_tx[:, None] * e1, axis=-1)
        geometry = (sources, targets, o1, o2, e1, e2, z_tx, h_tx)

        def take(values, rows):
            return tuple(tf.gather(v, rows) for v in values)

        def evaluate(geom, t2):
            """Eliminate t1 analytically and evaluate the reduced derivative."""
            tx, rx, p1, p2, v1, v2, z_tx, h_tx = geom

            q2 = p2 + t2[:, None] * v2
            offset = q2 - p1
            z2 = dot(offset, v1)
            h2 = tf.linalg.norm(offset - z2[:, None] * v1, axis=-1)
            t1 = tf.math.divide_no_nan(h2 * z_tx + h_tx * z2, h_tx + h2)
            q1 = p1 + t1[:, None] * v1

            u01, l01 = normalize(q1 - tx)
            u12, l12 = normalize(q2 - q1)
            u23, l23 = normalize(rx - q2)

            c01 = dot(v1, u01, clip=True)
            c12_1 = dot(v1, u12, clip=True)
            c12_2 = dot(v2, u12, clip=True)
            c23 = dot(v2, u23, clip=True)

            r1 = c01 - c12_1
            g = c12_2 - c23

            h11 = (
                tf.math.divide_no_nan(1.0 - c01**2, l01)
                + tf.math.divide_no_nan(1.0 - c12_1**2, l12)
            )
            h22 = (
                tf.math.divide_no_nan(1.0 - c12_2**2, l12)
                + tf.math.divide_no_nan(1.0 - c23**2, l23)
            )
            h12 = tf.math.divide_no_nan(c12_1 * c12_2 - dot(v1, v2), l12)

            # Schur complement: derivative of g after eliminating t1.
            curvature = h22 - tf.math.divide_no_nan(h12**2, h11)
            usable = (h_tx > 0.0) & (h2 > 0.0)
            usable &= (l01 > 0.0) & (l12 > 0.0) & (l23 > 0.0)
            usable &= tf.math.is_finite(g)
            return g, curvature, r1, usable, t1, q1, q2

        def solve(geom, lo, hi, t, steps, use_newton):
            def converged(g, usable):
                return usable & (tf.abs(g) <= keller_tol)

            def cond(lo, hi, t, g, curvature, usable):
                return tf.reduce_any(usable & ~converged(g, usable))

            def body(lo, hi, t, g, curvature, usable):
                pending = usable & ~converged(g, usable)
                lo = tf.where(pending & (g < 0.0), t, lo)
                hi = tf.where(pending & (g > 0.0), t, hi)
                trial = lo + 0.5 * (hi - lo)

                if use_newton:
                    newton = t - tf.math.divide_no_nan(g, curvature)
                    inside = (curvature > 0.0) & (newton > lo) & (newton < hi)
                    trial = tf.where(inside, newton, trial)

                t = tf.where(pending, trial, t)
                g, curvature, _, usable, _, _, _ = evaluate(geom, t)
                return lo, hi, t, g, curvature, usable

            g, curvature, _, usable, _, _, _ = evaluate(geom, t)
            lo, hi, t, g, _, usable = tf.while_loop(
                cond, body, (lo, hi, t, g, curvature, usable),
                maximum_iterations=steps,
            )
            return lo, hi, t, converged(g, usable)

        # Keep only candidates with a root strictly inside the second edge.
        g_lo, _, _, ok_lo, _, _, _ = evaluate(geometry, tf.zeros_like(length2))
        g_hi, _, _, ok_hi, _, _, _ = evaluate(geometry, length2)
        bracketed = ok_lo & ok_hi & (g_lo < 0.0) & (g_hi > 0.0)
        rows = tf.cast(tf.where(bracketed)[:, 0], tf.int32)

        geom = take(geometry, rows)
        hi = tf.gather(length2, rows)
        lo = tf.zeros_like(hi)
        lo, hi, t2, done = solve(geom, lo, hi, 0.5 * hi, max_newton_steps, True)

        # Run bisection only on candidates that still need it.
        pending = tf.cast(tf.where(~done)[:, 0], tf.int32)
        lo_pending, hi_pending = tf.gather(lo, pending), tf.gather(hi, pending)
        midpoint = lo_pending + 0.5 * (hi_pending - lo_pending)
        _, _, t2_pending, _ = solve(
            take(geom, pending), lo_pending, hi_pending, midpoint,
            max_bisection_steps, False,
        )
        t2 = tf.tensor_scatter_nd_update(t2, pending[:, None], t2_pending)

        g, _, r1, usable, t1, q1, q2 = evaluate(geom, t2)
        valid = usable & (tf.abs(r1) <= keller_tol) & (tf.abs(g) <= keller_tol)
        valid &= (t1 > 0.0) & (t1 < tf.gather(length1, rows))
        valid &= (t2 > 0.0) & (t2 < tf.gather(length2, rows))

        points = tf.stack([q1, q2], axis=1)
        points = tf.where(valid[:, None, None], points, tf.zeros_like(points))
        n = tf.shape(wedge_pairs)[0]
        return (
            tf.scatter_nd(rows[:, None], tf.cast(points, self._rdtype), [n, 2, 3]),
            tf.scatter_nd(rows[:, None], valid, [n]),
        )

    def compute_diffraction_points(self, sources, targets, wedge_pair_idxs):
        r"""
        Computes double diffraction

        Input
        ------
        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        targets : [num_targets, 3], tf.float
            Coordinates of the targets.

        wedge_pair_idxs: [num_sources, num_targets, num_pairs, 2], int
            The indices of the wedges

        Output
        ------
        diff_points: [num_sources, num_targets, num_pairs, 2, 3], tf.float
            The diffraction points

        active: [num_sources, num_targets, num_pairs], bool
            The active pairs
        
        """

        EPSILON = 1e-4
        max_opt_iterations = 10  #10
        max_wolfe_iterations = 8  #8

        num_sources, num_targets = tf.shape(sources)[0], tf.shape(targets)[0]
        num_pairs = tf.shape(wedge_pair_idxs)[2]

        # Change -1 by 0
        wedge_pair_idxs = tf.where(wedge_pair_idxs == -1, 0, wedge_pair_idxs)

        # [num_sources, num_targets, num_pairs, 3]
        w1_start_points = tf.gather(self._wedges_origin, wedge_pair_idxs[:, :, :, 0], axis=0)
        w2_start_points = tf.gather(self._wedges_origin, wedge_pair_idxs[:, :, :, 1], axis=0)

        # [num_sources, num_targets, num_pairs, 3]
        sources_ext = insert_dim_num(sources, 1, num_targets)
        sources_ext = insert_dim_num(sources_ext, 2, num_pairs)
        targets_ext = insert_dim_num(targets, 0, num_sources)
        targets_ext = insert_dim_num(targets_ext, 2, num_pairs)

        def total_len2(diff_points):
             r"""
             Compute the total length of the diffraction path
             targets -> diff_point1 -> diff_point2 -> sources

             sources_ext: [num_sources, num_targets, num_pairs, 3] 
             diff_points: [num_sources, num_targets, num_pairs, 2, 3]
             targets_ext: [num_sources, num_targets, num_pairs, 3]
             """

             return tf.norm(diff_points[:, :, :, 0] - sources_ext, axis=-1) +\
                    tf.norm(diff_points[:, :, :, 1] - diff_points[:, :, :, 0], axis=-1) +\
                    tf.norm(targets_ext - diff_points[:, :, :, 1], axis=-1)

        def total_len(diff_points1, diff_points2):
             r"""
             Compute the total length of the diffraction path
             targets -> diff_point1 -> diff_point2 -> sources

             sources_ext: [num_sources, num_targets, num_pairs, 3] 
             diff_points: [num_sources, num_targets, num_pairs, 2, 3]
             targets_ext: [num_sources, num_targets, num_pairs, 3]
             """

             return tf.norm(diff_points1 - sources_ext, axis=-1) +\
                    tf.norm(diff_points2 - diff_points1, axis=-1) +\
                    tf.norm(targets_ext - diff_points2, axis=-1)
        
        def _get_inv_hessian(hess):
            r"""
            hess: [num_sources, num_targets, num_pairs, 2, 2]

            return: [num_sources, num_targets, num_pairs, 2, 2]
            """
            res_tmp = tf.stack([
                tf.stack([hess[..., 1, 1], -hess[..., 0, 1]], axis=-1),
                tf.stack([-hess[..., 1, 0], hess[..., 0, 0]], axis=-1)], 
                axis=-2)
            
            div = (hess[..., 0, 0] * hess[..., 1, 1] - hess[..., 0, 1] * hess[..., 1, 0])

            return res_tmp / div[..., None, None]
        
        # [num_sources, num_targets, num_pairs]
        is_done = tf.zeros((num_sources, num_targets, num_pairs), dtype=tf.bool)

        # max_opt_iterations = 10  #20
        # max_wolfe_iterations = 8  #20

        # [num_sources, num_targets, num_pairs, 3]
        ksi1_hat = tf.gather(self._wedges_e_hat, wedge_pair_idxs[:, :, :, 0], axis=0) 
        ksi2_hat = tf.gather(self._wedges_e_hat, wedge_pair_idxs[:, :, :, 1], axis=0)

        # [num_sources, num_targets, num_pairs]
        h1_init = -2.0 * (dot(ksi1_hat, ksi2_hat))

        # [num_sources, num_targets, num_pairs, 2, 2]
        hessian_init = tf.stack([
                        tf.stack([4.0*tf.ones_like(h1_init), h1_init], axis=-1),
                        tf.stack([h1_init, 4.0*tf.ones_like(h1_init)], axis=-1)], 
                        axis=-2)
        
        # [num_sources, num_targets, num_pairs, 2, 2]
        inv_hessian_init = _get_inv_hessian(hessian_init)
        
        # [num_sources, num_targets, num_pairs]
        l1_init = dot((w2_start_points - 2.0 * w1_start_points + sources_ext), ksi1_hat)
        l2_init = dot((targets_ext - 2.0*w2_start_points + w1_start_points), ksi2_hat)
        # [num_sources, num_targets, num_pairs, 2]
        loss_init = -2.0 * tf.stack([l1_init, l2_init], axis=-1)

        # [num_sources, num_targets, num_pairs, 2]
        #ksi_0 = -1.0 * dot(inv_hessian_init, loss_init)
        #ksi_0 = -1.0 * tf.linalg.matmul(inv_hessian_init, loss_init)
        ksi_0 = -1.0 * tf.linalg.matmul(inv_hessian_init, tf.expand_dims(loss_init, axis=-1))[:, :, :, :, 0]

        # [num_sources, num_targets, num_pairs, 3]
        Q1 = w1_start_points + tf.expand_dims(ksi_0[..., 0], axis=-1) * ksi1_hat
        Q2 = w2_start_points + tf.expand_dims(ksi_0[..., 1], axis=-1) * ksi2_hat

        # [num_sources, num_targets, num_pairs, 2]
        ksi = ksi_0

        def _get_cost_vector(Q1, Q2):
            r"""
            Q1: [num_sources, num_targets, num_pairs, 3]
            Q2: [num_sources, num_targets, num_pairs, 3]

            return: [num_sources, num_targets, num_pairs, 2]
            """

            # [num_sources, num_targets, num_pairs]
            l11 = dot(Q1 - sources_ext, ksi1_hat)
            l12 = dot(Q2 - Q1, ksi1_hat)
            l21 = dot(Q2 - Q1, ksi2_hat)
            l22 = dot(targets_ext - Q2, ksi2_hat)

            l1 = l11 / tf.norm(Q1 - sources_ext, axis=-1) - l12 / tf.norm(Q2 - Q1, axis=-1)
            l2 = l21 / tf.norm(Q2 - Q1, axis=-1) - l22 / tf.norm(targets_ext - Q2, axis=-1)

            return tf.stack([l1, l2], axis=-1)
        
        def _get_hessian(Q1, Q2):
            r"""
            Q1: [num_sources, num_targets, num_pairs, 3]
            Q2: [num_sources, num_targets, num_pairs, 3]

            return: [num_sources, num_targets, num_pairs, 2, 2]
            """

            cos_beta1_prime = dot(ksi1_hat, Q1 - sources_ext) / tf.norm(Q1 - sources_ext, axis=-1)
            cos_beta1 = dot(ksi1_hat, Q2 - Q1) / tf.norm(Q2 - Q1, axis=-1)
            cos_beta2_prime = dot(ksi2_hat, Q2 - Q1) / tf.norm(Q2 - Q1, axis=-1)
            cos_beta2 = dot(ksi2_hat, targets_ext - Q2) / tf.norm(targets_ext - Q2, axis=-1)

            h11 = (1 - cos_beta1_prime**2) / tf.norm(Q1 - sources_ext, axis=-1) \
                    + (1 - cos_beta1**2) / tf.norm(Q2 - Q1, axis=-1)
            
            h12 = (cos_beta1 * cos_beta2_prime - dot(ksi1_hat, ksi2_hat)) / tf.norm(Q2 - Q1, axis=-1)
            #h21 = h12
            h22 = (1 - cos_beta2_prime**2) / tf.norm(Q2 - Q1, axis=-1) \
                    + (1 - cos_beta2**2) / tf.norm(targets_ext - Q2, axis=-1)
            
            return tf.stack([
                    tf.stack([h11, h12], axis=-1),
                    tf.stack([h12, h22], axis=-1)], axis=-2)
        
        def _wolfe_loop(active_wolfe, alpha, d_k, ksi, cost_vector, Q1, Q2):
            r"""
            active: [num_sources, num_targets, num_pairs]
            alpha: [num_sources, num_targets, num_pairs]
            """

            COEF = 1e-3

            # [num_sources, num_targets, num_pairs, 2]
            #ksi_new = ksi + alpha * d_k
            ksi_new = ksi + tf.expand_dims(alpha, axis=-1) * d_k
            # [num_sources, num_targets, num_pairs, 3]
            Q1_new = w1_start_points + tf.expand_dims(ksi_new[..., 0], axis=-1) * ksi1_hat
            Q2_new = w2_start_points + tf.expand_dims(ksi_new[..., 1], axis=-1) * ksi2_hat

            # [num_sources, num_targets, num_pairs]
            wolfe1 = total_len(Q1_new, Q2_new) < total_len(Q1, Q2) + COEF * alpha * dot(cost_vector, d_k)

            cond1 = tf.logical_and(active_wolfe, wolfe1)
            cond2 = tf.logical_and(active_wolfe, tf.logical_not(wolfe1))

            #active_wolfe = tf.where(cond1, active_wolfe, False)
            active_wolfe = tf.where(cond1, False, active_wolfe)
            alpha = tf.where(cond2, alpha * 0.5, alpha)

            return active_wolfe, alpha

        
        def _opt_loop(Q1, Q2, ksi, is_done):
            # [num_sources, num_targets, num_pairs, 2]
            cost_vector = _get_cost_vector(Q1, Q2)
            # [num_sources, num_targets, num_pairs, 2, 2]
            hessian = _get_hessian(Q1, Q2)
            # [num_sources, num_targets, num_pairs, 2, 2]
            inv_hessian = _get_inv_hessian(hessian)
            # [num_sources, num_targets, num_pairs, 2]
            #d_k = -1.0 * tf.linalg.matmul(inv_hessian, cost_vector)
            d_k = -1.0 * tf.linalg.matmul(inv_hessian, tf.expand_dims(cost_vector, axis=-1))[:, :, :, :, 0]

            # [num_sources, num_targets, num_pairs]
            alpha_k = tf.ones((num_sources, num_targets, num_pairs), dtype=self._rdtype)
            active_wolfe = tf.ones((num_sources, num_targets, num_pairs), dtype=tf.bool)

            for j in range(max_wolfe_iterations):
                active_wolfe, alpha_k = _wolfe_loop(active_wolfe, alpha_k, d_k, ksi, cost_vector, Q1, Q2)

            is_done = tf.logical_or(is_done, active_wolfe)  # break if Wolfe overflow

            # [num_sources, num_targets, num_pairs, 2]
            #ksi_new = ksi + alpha_k * d_k
            ksi_new = ksi + tf.expand_dims(alpha_k, axis=-1) * d_k
            # [num_sources, num_targets, num_pairs, 3]
            # Q1 = w1_start_points + ksi_new[..., 0] * ksi1_hat
            # Q2 = w2_start_points + ksi_new[..., 1] * ksi2_hat
            Q1 = tf.where(tf.expand_dims(is_done, axis=-1), Q1, w1_start_points + tf.expand_dims(ksi_new[..., 0], axis=-1) * ksi1_hat)
            Q2 = tf.where(tf.expand_dims(is_done, axis=-1), Q2, w2_start_points + tf.expand_dims(ksi_new[..., 1], axis=-1) * ksi2_hat)
            tl = total_len(Q1, Q2) 

            # is_done = tf.logical_or(is_done, active_wolfe)  # break if Wolfe overflow
            is_done = tf.logical_or(is_done, tf.norm(ksi_new - ksi, axis=-1) < EPSILON)  # break if step is too small

            return Q1, Q2, ksi_new, is_done

        for i in range(max_opt_iterations):
            if tf.reduce_all(is_done):
                break

            Q1, Q2, ksi, is_done = _opt_loop(Q1, Q2, ksi, is_done)

        # [num_sources, num_targets, num_pairs]
        cos_beta1_prime = dot(ksi1_hat, Q1 - sources_ext) / tf.norm(Q1 - sources_ext, axis=-1)
        cos_beta1 = dot(ksi1_hat, Q2 - Q1) / tf.norm(Q2 - Q1, axis=-1)
        cos_beta2_prime = dot(ksi2_hat, Q2 - Q1) / tf.norm(Q2 - Q1, axis=-1)
        cos_beta2 = dot(ksi2_hat, targets_ext - Q2) / tf.norm(targets_ext - Q2, axis=-1)
        # [np.arccos(cos_beta1_prime)*57.3, np.arccos(cos_beta1)*57.3]  # beta1
        # [np.arccos(cos_beta2_prime)*57.3, np.arccos(cos_beta2)*57.3]  # beta2

        # [num_sources, num_targets, num_pairs]
        active1 = tf.experimental.numpy.isclose(cos_beta1_prime, cos_beta1, rtol=1e-2)
        active2 = tf.experimental.numpy.isclose(cos_beta2_prime, cos_beta2, rtol=1e-2)
        active = tf.logical_and(active1, active2)

        return tf.stack([Q1, Q2], axis=-2), active

    def omit_identical_points(self, diff_points):
        r"""
        Omit the identical points in the diffraction path

        Input
        ------
        diff_points: [2, num_sources, num_targets, num_pairs, 3], tf.float
            The diffraction points

        Output
        ------

        active: [num_sources, num_targets, num_pairs], bool
            The active pairs
        """

        # [num_sources, num_targets, num_pairs]
        cond = tf.norm(diff_points[0] - diff_points[1], axis=-1) > 2e-2  # 1e-4

        return diff_points, cond
    
    def compute_diffraction_points_wedge_vertex(self, sources, targets, pair_idxs):
        r"""
        Computes double diffraction

        Input
        ------
        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        targets : [num_targets, 3], tf.float
            Coordinates of the targets.

        pair_idxs: [num_sources, num_targets, num_pairs, 2], int
            The indices [wedge, vertex]
        """
        # num_sources, num_targets = tf.shape(sources)[0], tf.shape(targets)[0]
        # num_pairs = tf.shape(pair_idxs)[2]
        # [num_sources, num_targets, num_pairs]
    
        # Change -1 by 0
        pair_idxs = tf.where(pair_idxs == -1, 0, pair_idxs)
        wedge_idxs = pair_idxs[:, :, :, 0]
        vertex_idxs = pair_idxs[:, :, :, 1]

        # [num_sources, num_targets, num_pairs, 3]
        vertices = tf.gather(self.vertices, vertex_idxs, axis=0)

        # 
        #vertex_targets = tf.reshape([-1, 3])
        # [num_sources, num_targets, num_pairs, 3]
        wedge_points, new_wedge_idxs, mask = self._wedge_diff_point_to_vertex(sources, wedge_idxs, vertices)

        # [num_sources, num_targets, num_pairs, 3]
        #valid = new_wedge_idxs != -1

        # [2, num_sources, num_targets, num_pairs, 3]
        path_vertices = tf.stack([wedge_points, vertices], axis=0)

        return path_vertices, mask
    
    def compute_diffraction_points_vertex_wedge(self, sources, targets, pair_idxs):
        r"""
        Computes double diffraction

        Input
        ------
        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        targets : [num_targets, 3], tf.float
            Coordinates of the targets.

        pair_idxs: [num_sources, num_targets, num_pairs, 2], int
            The indices [vertex, wedge]
        """
        # num_sources, num_targets = tf.shape(sources)[0], tf.shape(targets)[0]
        # num_pairs = tf.shape(pair_idxs)[2]
        # [num_sources, num_targets, num_pairs]
    
        # Change -1 by 0
        pair_idxs = tf.where(pair_idxs == -1, 0, pair_idxs)
        vertex_idxs = pair_idxs[:, :, :, 0]
        wedge_idxs = pair_idxs[:, :, :, 1]

        # [num_sources, num_targets, num_pairs, 3]
        vertices = tf.gather(self.vertices, vertex_idxs, axis=0)

        # [num_sources, num_targets, num_pairs, 3]
        wedge_points, new_wedge_idxs, mask = self._vertex_to_wedge_diff_point(vertices, wedge_idxs, targets)

        # [num_sources, num_targets, num_pairs, 3]
        #valid = new_wedge_idxs != -1

        # [2, num_sources, num_targets, num_pairs, 3]
        path_vertices = tf.stack([vertices, wedge_points], axis=0)

        return path_vertices, mask
    
    def compute_diffraction_points_vertex_vertex(self, sources, targets, pair_idxs):
        r"""
        Computes double diffraction

        Input
        ------
        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        targets : [num_targets, 3], tf.float
            Coordinates of the targets.

        pair_idxs: [num_sources, num_targets, num_pairs, 2], int
            The indices [vertex, vertex]
        """
        num_sources, num_targets = tf.shape(sources)[0], tf.shape(targets)[0]
        num_pairs = tf.shape(pair_idxs)[2]
        # [num_sources, num_targets, num_pairs]
    
        # Change -1 by 0
        pair_idxs = tf.where(pair_idxs == -1, 0, pair_idxs)
        v1_idxs = pair_idxs[:, :, :, 0]
        v2_idxs = pair_idxs[:, :, :, 1]

        # [num_sources, num_targets, num_pairs, 3]
        vertices1 = tf.gather(self.vertices, v1_idxs, axis=0)
        vertices2 = tf.gather(self.vertices, v2_idxs, axis=0)

        # [2, num_sources, num_targets, num_pairs, 3]
        path_vertices = tf.stack([vertices1, vertices2], axis=0)
        mask = tf.ones((num_sources, num_targets, num_pairs), dtype=tf.bool)

        return path_vertices, mask

    
    def _wedge_diff_point_to_vertex(self, sources, wedge_idxs, vertices):
        r"""
        sources : [num_sources, 3], tf.float
        
        wedge_idxs: [num_sources, num_targets, num_pairs], int

        vertices: [num_sources, num_targets, num_pairs, 3], tf.float
        """
        # [num_sources, num_targets, num_pairs]
        valid_wedges_idx = tf.where(wedge_idxs == -1, 0, wedge_idxs)
        # [num_sources, num_targets, num_pairs, 3]
        origins = tf.gather(self._wedges_origin, valid_wedges_idx)
        e_hat = tf.gather(self._wedges_e_hat, valid_wedges_idx)
        wedges_length = tf.gather(self._wedges_length, valid_wedges_idx)

        # [num_sources, num_targets, num_pairs, 3]
        u_t = origins - sources[:, None, None, :]
        # [num_sources, num_targets, num_pairs, 3]
        u_r = origins - vertices

        # Quantites required for the computation of the interaction points
        # [num_sources, num_targets, num_pairs]
        a = dot(u_t, e_hat)
        b = dot(u_r, e_hat)
        c = dot(u_t, u_t)
        d = dot(u_r, u_r)

        # Quantites required for the computation of the interaction points
        # [num_sources, num_targets, num_pairs]
        alpha = -tf.square(a) + tf.square(b) + c - d
        beta = 2.*(a*tf.square(b) - b*tf.square(a) + b*c - a*d)
        gamma = tf.square(b)*c - tf.square(a)*d

        # Normalized quantites to improve numerical preicion, only valid if
        # alpha != 0
        # [num_sources, num_targets, num_pairs]
        beta_norm = tf.math.divide_no_nan(beta, alpha)
        gamma_norm = tf.math.divide_no_nan(gamma, alpha)
        delta = tf.square(beta_norm) - 4.0*gamma_norm

        # Because of numerical imprecision, delta could be slighlty smaller than 0
        # [num_sources, num_targets, max_num_paths]
        delta = tf.where(tf.less(delta, tf.zeros_like(delta)), tf.zeros_like(delta), delta)
        
        # Four possible outcomes depending on the value of the previous
        # quantities.
        # Values of t that minimizes the path length for each outcome.
        # [num_sources, num_targets, num_pairs]
        t_min_1 = -a
        t_min_2 = -tf.math.divide_no_nan(gamma, beta)
        t_min_3 = (-beta_norm + tf.sqrt(delta))*0.5
        t_min_4 = (-beta_norm - tf.sqrt(delta))*0.5
        # Condition for each outcome to be selected
        # If a == b and c == d, then set to t_min_1
        # [num_sources, num_targets, num_pairs]
        cond_1 = tf.logical_and(tf.experimental.numpy.isclose(a, b),
                                tf.experimental.numpy.isclose(c, d))
        # If cond_1 does not hold and alpha == 0, then set to t_min_2
        # [num_sources, num_targets, num_pairs]
        cond_2 = tf.logical_and(tf.logical_not(cond_1),
                                tf.experimental.numpy.isclose(alpha,
                                tf.zeros_like(alpha)))
        # If neither cond_1 nor cond_2 holds, then set to t_min_3 or t_min_4
        # depending on the signs of t+a and t+b
        # [num_sources, num_targets, num_pairs]
        not_cond_12 = tf.logical_and(tf.logical_not(cond_1),
                                     tf.logical_not(cond_2))
        # [num_sources, num_targets, num_pairs]
        t_min_3a = t_min_3 + a
        t_min_3b = t_min_3 + b
        # [num_sources, num_targets, num_pairs]
        cond_3 = tf.logical_and(not_cond_12,
                tf.less_equal(tf.sign(t_min_3a)*tf.sign(t_min_3b), 0.0))
        # If none of conditions 1, 2, or 3 are satisfied, then all is left is
        # t_min_4
        # [num_sources, num_targets, num_pairs]
        cond_4 = tf.logical_and(not_cond_12, tf.logical_not(cond_3))
        # Assign t_min according to the previously computed conditions
        # [num_sources, num_targets, num_pairs]
        # t_min = tf.zeros_like(cond_1, tf.float64)
        t_min = tf.zeros_like(cond_1, self._rdtype)
        t_min = tf.where(cond_1, t_min_1, t_min)
        t_min = tf.where(cond_2, t_min_2, t_min)
        t_min = tf.where(cond_3, t_min_3, t_min)
        t_min = tf.where(cond_4, t_min_4, t_min)

        # Mask paths for which the interaction point is not on the finite
        # wedge
        # [num_sources, num_targets, num_pairs]
        mask_ = tf.logical_and(
            tf.greater_equal(t_min, tf.zeros_like(t_min)),
            tf.less_equal(t_min, wedges_length))
        # [num_sources, num_targets, num_pairs]
        new_wedge_idxs = tf.where(mask_, wedge_idxs, -1)

        # Interaction points
        # Expand to broadcast with coordinates
        # [num_sources, num_targets, max_num_pairs, 1]
        t_min = tf.expand_dims(t_min, axis=3)
        # [num_sources, num_targets, max_num_pairs, 3]
        inter_point = origins + t_min*e_hat

        return inter_point, new_wedge_idxs, mask_
    
    def _vertex_to_wedge_diff_point(self, vertices, wedge_idxs, targets):
        r"""
        
        vertices: [num_sources, num_targets, num_pairs, 3], tf.float
        
        wedge_idxs: [num_sources, num_targets, num_pairs], int

        targets : [num_targets, 3], tf.float

        """
        # [num_sources, num_targets, num_pairs]
        valid_wedges_idx = tf.where(wedge_idxs == -1, 0, wedge_idxs)
        # [num_sources, num_targets, num_pairs, 3]
        origins = tf.gather(self._wedges_origin, valid_wedges_idx)
        e_hat = tf.gather(self._wedges_e_hat, valid_wedges_idx)
        wedges_length = tf.gather(self._wedges_length, valid_wedges_idx)

        # [num_sources, num_targets, num_pairs, 3]
        targets_ext = targets[None, :, None, :]

        # [num_sources, num_targets, num_pairs, 3]
        u_t = origins - vertices
        # [num_sources, num_targets, num_pairs, 3]
        u_r = origins - targets_ext

        inter_point, new_wedge_idxs, mask_ = get_wd_points(u_t, u_r, e_hat, origins, wedges_length, wedge_idxs)

        return inter_point, new_wedge_idxs, mask_ 
    
    def check_diff_points_validity(self, diff_points, wedge_pair_idxs):
        r"""
        Check if diff_points are on wedges (between start and end points of a wedge)

        Inputs
        ------
        diff_points: [num_sources, num_targets, num_pairs, 2, 3], tf.float
            The diffraction points

        wedge_pair_idxs: [num_sources, num_targets, num_pairs, 2], int
            The indices of the wedges
        """

        wedge_pair_idxs = tf.where(wedge_pair_idxs == -1, 0, wedge_pair_idxs)
        
        # [num_sources, num_targets, num_pairs]
        w1_len = tf.gather(self._wedges_length, wedge_pair_idxs[:, :, :, 0], axis=0)
        w2_len = tf.gather(self._wedges_length, wedge_pair_idxs[:, :, :, 1], axis=0)
        # [num_sources, num_targets, num_pairs, 3]
        w1_e_hat = tf.gather(self._wedges_e_hat, wedge_pair_idxs[:, :, :, 0], axis=0)
        w2_e_hat = tf.gather(self._wedges_e_hat, wedge_pair_idxs[:, :, :, 1], axis=0)

        # [num_sources, num_targets, num_pairs, 3]
        w1_start_points = tf.gather(self._wedges_origin, wedge_pair_idxs[:, :, :, 0], axis=0)
        w2_start_points = tf.gather(self._wedges_origin, wedge_pair_idxs[:, :, :, 1], axis=0)
        w1_end_points = w1_start_points + w1_e_hat * tf.expand_dims(w1_len, axis=-1)
        w2_end_points = w2_start_points + w2_e_hat * tf.expand_dims(w2_len, axis=-1)
        
        # [num_sources, num_targets, num_pairs]
        valid_w1 = dot(w1_start_points - diff_points[:, :, :, 0], w1_end_points - diff_points[:, :, :, 0]) <= 0.0
        valid_w2 = dot(w2_start_points - diff_points[:, :, :, 1], w2_end_points - diff_points[:, :, :, 1]) <= 0.0
        valid = tf.logical_and(valid_w1, valid_w2)

        return valid
    
    def get_slopes(self, dd_pair_idxs):
        r"""
        Inputs
        ------

        dd_pairs : [2, num_targets, num_sources, num_pairs], int
            The indices of the diffraction pairs (wedge1, wedge2)
        """

        def foo(a, b, atol=1e-3, rtol=1e-3):
            mask = tf.abs(a - b) <= (atol + rtol * tf.abs(b))
            return tf.reduce_all(mask, axis=-1)

        # [num_targets, num_sources, num_pairs]
        idxs1 = dd_pair_idxs[0]
        idxs2 = dd_pair_idxs[1]

        ## remove idxs -1
        idxs1 = tf.where(idxs1 == -1, 0, idxs1)
        idxs2 = tf.where(idxs2 == -1, 0, idxs2)

        ### wedge objects
        # [num_targets, num_sources, num_pairs]
        w1_obj = tf.gather(self._wedges_objects[:, 0], idxs1, axis=0)
        w2_obj = tf.gather(self._wedges_objects[:, 0], idxs2, axis=0)
        mask_same_objs = tf.equal(w1_obj, w2_obj)

        # [num_targets, num_sources, num_pairs, 2, 3]
        w1_normals = tf.gather(self._wedges_normals, idxs1, axis=0)
        w2_normals = tf.gather(self._wedges_normals, idxs2, axis=0)

        # [num_targets, num_sources, num_pairs]
        mask1 = foo(w1_normals[..., 0, :], w2_normals[..., 0, :])
        mask2 = foo(w1_normals[..., 0, :], w2_normals[..., 1, :])
        mask3 = foo(w1_normals[..., 1, :], w2_normals[..., 0, :])
        mask4 = foo(w1_normals[..., 1, :], w2_normals[..., 1, :])

        mask = tf.logical_or(
            tf.logical_or(mask1, mask2),
            tf.logical_or(mask3, mask4)
        )

        ## no slopes for planar (knife) edges
        is_edge1 = tf.gather(self.solver_paths.is_edge, idxs1, axis=0)
        is_edge2 = tf.gather(self.solver_paths.is_edge, idxs2, axis=0)
        mask = tf.logical_and(mask, tf.logical_not(tf.logical_and(is_edge1, is_edge2)))

        mask = tf.logical_and(mask, mask_same_objs)

        return mask
        
    def check_preliminary_visibility(self, dd_pairs, junction_enabled=None):
        r"""
        Inputs
        ------

        dd_pairs : [num_pairs, 2], int
            The indices of the diffraction pairs (wedge1, wedge2)
        """

        # [num_pairs]
        idxs1 = dd_pairs[:, 0]
        idxs2 = dd_pairs[:, 1]

        ## use center points of the wedges
        # [num_pairs, 3]
        points1 = tf.gather(self._wedges_origin, idxs1, axis=0) + tf.gather(self._wedges_e_hat, idxs1, axis=0) *\
            tf.gather(self._wedges_length, idxs1, axis=0)[:, None]/2
        
        points2 = tf.gather(self._wedges_origin, idxs2, axis=0) + tf.gather(self._wedges_e_hat, idxs2, axis=0) *\
            tf.gather(self._wedges_length, idxs2, axis=0)[:, None]/2

        # [num_pairs, 2, 3]
        w1_normals = tf.gather(self._wedges_normals, idxs1, axis=0)
        w2_normals = tf.gather(self._wedges_normals, idxs2, axis=0)

        # [num_pairs, 3]
        shift_1 = self.solver_paths.DD_EPSILON_OBSTRUCTION * (w1_normals[:, 0] + w1_normals[:, 1]) / 3.0
        shift_2 = self.solver_paths.DD_EPSILON_OBSTRUCTION * (w2_normals[:, 0] + w2_normals[:, 1]) / 3.0

        d, maxt = tf.linalg.normalize((points2 + shift_2) - (points1 + shift_1), axis=1)
        maxt = tf.squeeze(maxt, axis=1)
        valid = tf.logical_not(self.solver_paths._test_obstruction_2(points1 + shift_1, d, maxt))

        ### add edges, as they fail for such visibility checks
        # is_edge1 = tf.gather(self.solver_paths.is_edge, idxs1, axis=0)
        is_edge = tf.math.reduce_any(tf.gather(self.solver_paths.is_edge, dd_pairs, axis=0), axis=-1)
        valid = tf.logical_or(valid, is_edge)

        if junction_enabled is not None:
            # Shifted midpoint rays can reject valid endpoint-to-endpoint paths.
            valid |= tf.reduce_any(self._junction_candidates(dd_pairs, junction_enabled), axis=-1)
        return tf.boolean_mask(dd_pairs, valid, axis=0)

    def _check_visibility(self, sources, targets, diff_points, dd_pair_idxs, junction_enabled=None):
        r"""
        Check visibility source -> diffraction_point1 -> diffraction_point2 -> target

        Inputs
        ------
        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        targets : [num_targets, 3], tf.float
            Coordinates of the targets.

        diff_points: [num_sources, num_targets, num_pairs, 2, 3], tf.float
            The diffraction points

        dd_pair_idxs: [num_sources, num_targets, num_pairs, 2], int
            The indices of the wedges

        Output
        ------
        visible: [num_sources, num_targets, num_pairs], bool
            The visibility of the diffraction points
        """

        dd_pair_idxs_valid = tf.where(dd_pair_idxs == -1, 0, dd_pair_idxs)

        num_sources, num_targets = tf.shape(sources)[0], tf.shape(targets)[0]
        num_pairs = tf.shape(diff_points)[2]

        # [num_sources, num_targets, num_pairs, 3]
        sources = insert_dim_num(sources, 1, num_targets)
        sources = insert_dim_num(sources, 2, num_pairs)
        targets = insert_dim_num(targets, 0, num_sources)
        targets = insert_dim_num(targets, 2, num_pairs)
        diff_points1 = diff_points[:, :, :, 0]
        diff_points2 = diff_points[:, :, :, 1]

        # [batch_size, 3]
        sources = tf.reshape(sources, (-1, 3))
        targets = tf.reshape(targets, (-1, 3))
        diff_points1 = tf.reshape(diff_points1, (-1, 3))
        diff_points2 = tf.reshape(diff_points2, (-1, 3))


        # 1) Check visibility between source and diff_points1
        # d : [batch_size, 3]
        # maxt : [batch_size]
        d, maxt = tf.linalg.normalize(diff_points1 - sources, axis=1)
        maxt = tf.squeeze(maxt, axis=1)
        # [batch_size]
        valid_s2w = tf.logical_not(self.solver_paths._test_obstruction(sources, d, maxt))

        # 2) Check visibility between diff_points1 and diff_points2  # TODO
        w1_normals = tf.gather(self._wedges_normals, dd_pair_idxs_valid[:, :, :, 0], axis=0)
        w2_normals = tf.gather(self._wedges_normals, dd_pair_idxs_valid[:, :, :, 1], axis=0)

        is_concave1 = tf.gather(self.solver_paths._is_concave_wedge, dd_pair_idxs_valid[:, :, :, 0], axis=0)
        is_concave2 = tf.gather(self.solver_paths._is_concave_wedge, dd_pair_idxs_valid[:, :, :, 1], axis=0)
        w1_normals = tf.where(is_concave1[..., None, None], -w1_normals, w1_normals)
        w2_normals = tf.where(is_concave2[..., None, None], -w2_normals, w2_normals)

        shift_1 = self.solver_paths.DD_EPSILON_OBSTRUCTION * (w1_normals[:,:,:,0] + w1_normals[:,:,:,1]) / 3.0
        shift_2 = self.solver_paths.DD_EPSILON_OBSTRUCTION * (w2_normals[:,:,:,0] + w2_normals[:,:,:,1]) / 3.0
        shift_1 = tf.reshape(shift_1, (-1, 3))
        shift_2 = tf.reshape(shift_2, (-1, 3))

        d, maxt = tf.linalg.normalize((diff_points2 + shift_2) - (diff_points1 + shift_1), axis=1)
        maxt = tf.squeeze(maxt, axis=1)
        valid_w2w = tf.logical_not(self.solver_paths._test_obstruction_2(diff_points1+shift_1, d, maxt))

        ## check edges separately, as they fail for such visibility checks
        valid_w2w_edges = tf.logical_not(self.solver_paths._test_obstruction(diff_points1, d, maxt))
        valid_w2w = tf.logical_or(valid_w2w, valid_w2w_edges)

        if junction_enabled is not None:
            special = tf.reduce_any(self._junction_candidates(dd_pair_idxs_valid, junction_enabled), axis=-1)
            special = tf.reshape(special, [-1])
            rows = tf.cast(tf.where(special)[:, 0], tf.int32)
            if tf.size(rows) > 0:
                exact = self._unshifted_middle_visibility(
                    tf.gather(diff_points1, rows), tf.gather(diff_points2, rows)
                )
                valid_w2w = tf.tensor_scatter_nd_update(valid_w2w, rows[:, None], exact)

        # 3) Check visibility between diff_points2 and target
        d, maxt = tf.linalg.normalize(targets - diff_points2, axis=1)
        maxt = tf.squeeze(maxt, axis=1)
        valid_w2t = tf.logical_not(self.solver_paths._test_obstruction(diff_points2, d, maxt, True))

        # [batch_size]
        valid = tf.logical_and(tf.logical_and(valid_s2w, valid_w2w), valid_w2t)
        #valid = tf.logical_and(valid_s2w, valid_w2t) a_s2w, a_w2w
        # [num_sources, num_targets, num_pairs]
        valid = tf.reshape(valid, (num_sources, num_targets, num_pairs))

        return valid

    def check_visibility_mixed(self, sources, targets, diff_points, w_idxs, v_idxs, junction_enabled=None):
        r"""
        Check visibility source -> diffraction_point1 -> diffraction_point2 -> target

        Inputs
        ------
        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        targets : [num_targets, 3], tf.float
            Coordinates of the targets.

        diff_points: [2, num_sources, num_targets, num_pairs, 3], tf.float
            The diffraction points

        dd_pair_idxs: [num_sources, num_targets, num_pairs, 2], int
            The indices of the wedges

        Output
        ------
        visible: [num_sources, num_targets, num_pairs], bool
            The visibility of the diffraction points
        """

        vaild_w_idxs = tf.where(w_idxs == -1, 0, w_idxs)
        vaild_v_idxs = tf.where(v_idxs == -1, 0, v_idxs)

        num_sources, num_targets = tf.shape(sources)[0], tf.shape(targets)[0]
        num_pairs = tf.shape(diff_points)[3]

        # [num_sources, num_targets, num_pairs, 3]
        sources = insert_dim_num(sources, 1, num_targets)
        sources = insert_dim_num(sources, 2, num_pairs)
        targets = insert_dim_num(targets, 0, num_sources)
        targets = insert_dim_num(targets, 2, num_pairs)
        diff_points1 = diff_points[0, ...]
        diff_points2 = diff_points[1, ...]
        # diff_points1 = diff_points[:, :, :, 0]
        # diff_points2 = diff_points[:, :, :, 1]

        # [batch_size, 3]
        sources = tf.reshape(sources, (-1, 3))
        targets = tf.reshape(targets, (-1, 3))
        diff_points1 = tf.reshape(diff_points1, (-1, 3))
        diff_points2 = tf.reshape(diff_points2, (-1, 3))


        # 1) Check visibility between source and diff_points1
        # d : [batch_size, 3]
        # maxt : [batch_size]
        d, maxt = tf.linalg.normalize(diff_points1 - sources, axis=1)
        maxt = tf.squeeze(maxt, axis=1)
        # [batch_size]
        valid_s2w = tf.logical_not(self.solver_paths._test_obstruction(sources, d, maxt))

        # 2) Check visibility between diff_points1 and diff_points2  # TODO
        w1_normals = tf.gather(self._wedges_normals, vaild_w_idxs, axis=0)
        is_concave1 = tf.gather(self.solver_paths._is_concave_wedge, vaild_w_idxs, axis=0)
        w1_normals = tf.where(is_concave1[..., None, None], -w1_normals, w1_normals)

        v_wedge_idxs = tf.gather(self.vertices_2_wedges, vaild_v_idxs, axis=0)
        _w2_normals = tf.gather(self._wedges_normals, v_wedge_idxs, axis=0)
        is_concave2 = tf.gather(self.solver_paths._is_concave_wedge, v_wedge_idxs, axis=0)
        _w2_normals = tf.where(is_concave2[..., None, None], -_w2_normals, _w2_normals)
        w2_normals = tf.reduce_mean(_w2_normals, axis=-3).to_tensor()
        
        # w2_normals = tf.gather(self._wedges_normals, dd_pair_idxs_valid[:, :, :, 1], axis=0)
        shift_1 = self.solver_paths.DD_EPSILON_OBSTRUCTION * (w1_normals[:,:,:,0] + w1_normals[:,:,:,1]) / 3.0
        shift_2 = self.solver_paths.DD_EPSILON_OBSTRUCTION * (w2_normals[:,:,:,0] + w2_normals[:,:,:,1]) / 3.0
        shift_1 = tf.reshape(shift_1, (-1, 3))
        shift_2 = tf.reshape(shift_2, (-1, 3))

        d, maxt = tf.linalg.normalize((diff_points2 + shift_2) - (diff_points1 + shift_1), axis=1)
        maxt = tf.squeeze(maxt, axis=1)
        valid_w2w = tf.logical_not(self.solver_paths._test_obstruction_2(diff_points1+shift_1, d, maxt))

        ## check edges separately, as they fail for such visibility checks
        valid_w2w_edges = tf.logical_not(self.solver_paths._test_obstruction(diff_points1, d, maxt))     
        valid_w2w = tf.logical_or(valid_w2w, valid_w2w_edges)

        if junction_enabled is not None:
            special = self._junction_candidates(vaild_w_idxs, junction_enabled)
            special |= self._vertex_has_junction(vaild_v_idxs, junction_enabled)
            special = tf.reshape(special, [-1])
            rows = tf.cast(tf.where(special)[:, 0], tf.int32)
            if tf.size(rows) > 0:
                exact = self._unshifted_middle_visibility(
                    tf.gather(diff_points1, rows), tf.gather(diff_points2, rows)
                )
                valid_w2w = tf.tensor_scatter_nd_update(valid_w2w, rows[:, None], exact)

        # 3) Check visibility between diff_points2 and target
        d, maxt = tf.linalg.normalize(targets - diff_points2, axis=1)
        maxt = tf.squeeze(maxt, axis=1)
        valid_w2t = tf.logical_not(self.solver_paths._test_obstruction(diff_points2, d, maxt, True))

        # [batch_size]
        valid = tf.logical_and(tf.logical_and(valid_s2w, valid_w2w), valid_w2t)
        #valid = tf.logical_and(valid_s2w, valid_w2t) a_s2w, a_w2w
        # [num_sources, num_targets, num_pairs]
        valid = tf.reshape(valid, (num_sources, num_targets, num_pairs))

        return valid

    def check_visibility_vertex_vertex(self, sources, targets, diff_points, v1_idxs, v2_idxs):
        r"""
        Check visibility source -> diffraction_point1 -> diffraction_point2 -> target

        Inputs
        ------
        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        targets : [num_targets, 3], tf.float
            Coordinates of the targets.

        diff_points: [2, num_sources, num_targets, num_pairs, 3], tf.float
            The diffraction points

        v1_idxs: [num_sources, num_targets, num_pairs], int
            The indices of the vertices1

        Output
        ------
        visible: [num_sources, num_targets, num_pairs], bool
            The visibility of the diffraction points
        """

        vaild_v1_idxs = tf.where(v1_idxs == -1, 0, v1_idxs)
        vaild_v2_idxs = tf.where(v2_idxs == -1, 0, v2_idxs)

        num_sources, num_targets = tf.shape(sources)[0], tf.shape(targets)[0]
        num_pairs = tf.shape(diff_points)[3]

        # [num_sources, num_targets, num_pairs, 3]
        sources = insert_dim_num(sources, 1, num_targets)
        sources = insert_dim_num(sources, 2, num_pairs)
        targets = insert_dim_num(targets, 0, num_sources)
        targets = insert_dim_num(targets, 2, num_pairs)
        diff_points1 = diff_points[0, ...]
        diff_points2 = diff_points[1, ...]
        # diff_points1 = diff_points[:, :, :, 0]
        # diff_points2 = diff_points[:, :, :, 1]

        # [batch_size, 3]
        sources = tf.reshape(sources, (-1, 3))
        targets = tf.reshape(targets, (-1, 3))
        diff_points1 = tf.reshape(diff_points1, (-1, 3))
        diff_points2 = tf.reshape(diff_points2, (-1, 3))


        # 1) Check visibility between source and diff_points1
        # d : [batch_size, 3]
        # maxt : [batch_size]
        d, maxt = tf.linalg.normalize(diff_points1 - sources, axis=1)
        maxt = tf.squeeze(maxt, axis=1)
        # [batch_size]
        valid_s2w = tf.logical_not(self.solver_paths._test_obstruction(sources, d, maxt))

        # 2) Check visibility between diff_points1 and diff_points2  # TODO
        v1_wedge_idxs = tf.gather(self.vertices_2_wedges, vaild_v1_idxs, axis=0)
        _w1_normals = tf.gather(self._wedges_normals, v1_wedge_idxs, axis=0)
        w1_normals = tf.reduce_mean(_w1_normals, axis=-3).to_tensor()

        v2_wedge_idxs = tf.gather(self.vertices_2_wedges, vaild_v2_idxs, axis=0)
        _w2_normals = tf.gather(self._wedges_normals, v2_wedge_idxs, axis=0)
        w2_normals = tf.reduce_mean(_w2_normals, axis=-3).to_tensor()
        
        # TODO: does this work for concave wedges?
        shift_1 = self.solver_paths.DD_EPSILON_OBSTRUCTION * (w1_normals[:,:,:,0] + w1_normals[:,:,:,1]) / 3.0
        shift_2 = self.solver_paths.DD_EPSILON_OBSTRUCTION * (w2_normals[:,:,:,0] + w2_normals[:,:,:,1]) / 3.0
        shift_1 = tf.reshape(shift_1, (-1, 3))
        shift_2 = tf.reshape(shift_2, (-1, 3))

        d, maxt = tf.linalg.normalize((diff_points2 + shift_2) - (diff_points1 + shift_1), axis=1)
        maxt = tf.squeeze(maxt, axis=1)
        valid_w2w = tf.logical_not(self.solver_paths._test_obstruction_2(diff_points1+shift_1, d, maxt))

        ## add edges separately, as they fail for such visibility checks
        valid_w2w_edges = tf.logical_not(self.solver_paths._test_obstruction(diff_points1, d, maxt))
        valid_w2w = tf.logical_or(valid_w2w, valid_w2w_edges)

        # 3) Check visibility between diff_points2 and target
        d, maxt = tf.linalg.normalize(targets - diff_points2, axis=1)
        maxt = tf.squeeze(maxt, axis=1)
        valid_w2t = tf.logical_not(self.solver_paths._test_obstruction(diff_points2, d, maxt, True))

        # [batch_size]
        valid = tf.logical_and(tf.logical_and(valid_s2w, valid_w2w), valid_w2t)
        #valid = tf.logical_and(valid_s2w, valid_w2t) a_s2w, a_w2w
        # [num_sources, num_targets, num_pairs]
        valid = tf.reshape(valid, (num_sources, num_targets, num_pairs))

        return valid
    
    def close_2_transition(self, sources, targets, diff_points):
        r"""
        Check if diff_points are close to transition regions

        Inputs
        ------
        sources : [num_sources, 3], tf.float

        targets : [num_targets, 3], tf.float

        diff_points: [2, num_sources, num_targets, num_pairs, 3], tf.float
            The diffraction points

        Output
        ------
        is_close: [num_sources, num_targets, num_pairs], bool
            If they are close
        """
        # [num_sources, num_targets, num_pairs, 3]
        r1_hat, _ = normalize(diff_points[0, :, :, :] - sources[:, None, None, :])
        l_hat, _ = normalize(diff_points[1, :, :, :] - diff_points[0, :, :, :])
        r2_hat, _ = normalize(targets[None, :, None, :] - diff_points[1, :, :, :])

        eps = 1e-3
        are_close = tf.norm(diff_points[0, :, :, :] - diff_points[1, :, :, :], axis=-1) < eps
        
        # [num_sources, num_targets, num_pairs]
        source_close = dot(r1_hat, l_hat) > (self.close_2_transition_eps)
        target_close = dot(l_hat, r2_hat) > (self.close_2_transition_eps)

        res = tf.logical_and(source_close, target_close)
        res = tf.logical_or(res, are_close)

        return res


    def close_2_transition2(self, sources, targets, diff_points):
        r"""
        Check if diff_points are close to transition regions

        Inputs
        ------
        sources : [num_sources, 3], tf.float

        targets : [num_targets, 3], tf.float

        diff_points: [2, num_sources, num_targets, num_pairs, 3], tf.float
            The diffraction points

        Output
        ------
        is_close: [num_sources, num_targets, num_pairs], bool
            If they are close
        """
        targets = targets[None, :, None, :]
        sources = sources[:, None, None, :]

        # [num_sources, num_targets, num_pairs, 3]
        tx_rx_hat, _ = normalize(targets - sources)
        p1_hat, _ = normalize(diff_points[0] - sources)
        p2_hat, _ = normalize(diff_points[1] - sources)

        u1 = dot(p1_hat, tx_rx_hat)
        u2 = dot(p2_hat, tx_rx_hat)

        # valid_order = u1 < u2
        valid_order = u2 - u1 > 0.0

        return valid_order
    
    def check_if_diff_points_close(self, diff_points):
        r"""
        Check if diff_poin1 is very close to diff_point2

        Inputs
        ------

        diff_points: [num_sources, num_targets, num_pairs, 2, 3], tf.float
            The diffraction points

        Output
        ------
        is_close: [num_sources, num_targets, num_pairs], bool
            If they are close
        """

        eps = 1e-3

        diff_points1 = diff_points[:, :, :, 0]
        diff_points2 = diff_points[:, :, :, 1]

        return tf.norm(diff_points1 - diff_points2, axis=-1) < eps

    
    def _gather_valid_paths(self, paths: Paths):
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

        # [2, num_targets, num_sources, max_num_candidates]
        path_objects = paths.objects
        # [2, num_targets, num_sources, max_num_candidates, 3]
        path_vertices = paths.vertices
        # [num_sources, 3]
        sources = paths.sources
        # [num_targets, 3]
        targets = paths.targets

        max_depth = paths.vertices.shape[0]  # 2

        num_sources = tf.shape(sources)[0]
        num_targets = tf.shape(targets)[0]

        # # [num_targets, num_sources, max_num_candidates]
        # valid = tf.reduce_all(tf.not_equal(path_objects, -1), axis=0)
        valid = paths.mask

        # [num_targets, num_sources]
        num_paths = tf.reduce_sum(tf.cast(valid, tf.int32), axis=-1)
        # Maximum number of valid paths
        # ()
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
        # [num_keep_paths]
        mask_ = tf.gather_nd(valid, gather_indices)
        # [num_targets, num_sources, max_num_paths]
        mask = tf.tensor_scatter_nd_update(mask, scatter_indices, mask_)

        # [2, num_targets, num_sources, max_num_paths, 3]
        valid_vertices = tf.zeros([max_depth, num_targets, num_sources, max_num_paths, 3],
                                dtype=self._rdtype)
        # [max_depth, num_targets, num_sources, max_num_paths]
        valid_objects = tf.fill([max_depth, num_targets, num_sources,
                                        max_num_paths], -1)
        
        for depth in tf.range(max_depth, dtype=tf.int64):
            scatter_indices_ = tf.pad(scatter_indices, [[0,0], [1,0]],
                                mode='CONSTANT', constant_values=depth)
            
            # [num_targets, num_sources, num_samples, 3]
            vertices_ = tf.gather(path_vertices, depth, axis=0)
            # [total_num_valid_paths, 3]
            vertices_ = tf.gather_nd(vertices_, gather_indices)
            # Store the valid intersection points
            # [max_depth, num_targets, num_sources, max_num_paths, 3]
            valid_vertices = tf.tensor_scatter_nd_update(valid_vertices,
                                            scatter_indices_, vertices_)
            
            # Intersected primitives
            # Extract only the valid paths
            # [num_targets, num_sources, num_samples]
            objects_ = tf.gather(path_objects, depth, axis=0)
            # [total_num_valid_paths]
            #objects_ = tf.gather(objects_, gather_indices[:,2])  # ?
            objects_ = tf.gather_nd(objects_, gather_indices)
            # Store the valid primitives]
            # [max_depth, num_targets, num_sources, max_num_paths]
            valid_objects = tf.tensor_scatter_nd_update(valid_objects,
                                    scatter_indices_, objects_)
            

        # [max_depth, num_targets, num_sources, max_num_candidates]
        paths.objects = valid_objects
        # [max_depth, num_targets, num_sources, max_num_candidates, 3]
        paths.vertices = valid_vertices
        # [num_targets, num_sources, max_num_candidates]
        paths.mask = mask
        paths.targets_sources_mask = mask

        return paths
    
    def _check_close_to_SB(self, sources, targets, diff_points, dd_pair_idxs):
        r"""
        Check if diff_points are close to the shadow boundary

        Inputs
        ------
        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        targets : [num_targets, 3], tf.float
            Coordinates of the targets.

        diff_points: [num_sources, num_targets, num_pairs, 2, 3], tf.float
            The diffraction points

        dd_pair_idxs: [num_sources, num_targets, num_pairs, 2], int
            The indices of the wedges

        Output
        ------
        close_to_SB: [num_sources, num_targets, num_pairs], bool
            The closeness to the shadow boundary of the diffraction points
        """

        # [num_sources, num_targets, num_pairs, 2]
        dd_pair_idxs_valid = tf.where(dd_pair_idxs == -1, 0, dd_pair_idxs)

        num_sources, num_targets = tf.shape(sources)[0], tf.shape(targets)[0]
        num_paths = tf.shape(diff_points)[2]


        # [num_sources, num_targets, num_sides=2, num_pairs=2, 3]
        normals = tf.gather(self._wedges_normals, dd_pair_idxs_valid, axis=0)

        n11 = normals[:, :, :, 0, 0]
        n12 = normals[:, :, :, 0, 1]
        n21 = normals[:, :, :, 1, 0]
        n22 = normals[:, :, :, 1, 1]

        zeros = tf.ones((num_sources, num_targets, num_paths), dtype=self._rdtype) * 1e-3
        cond1 = tf.math.less(tf.abs(dot(n11, n21)), zeros)
        cond2 = tf.math.less(tf.abs(dot(n12, n22)), zeros)
        
        #slope_normals = tf.

    def compute_fields(self, relative_permittivity,
                                scattering_coefficient, 
                                paths: Paths, 
                                paths_tmp: PathsTmpData, debug_resolve_edges=False):
        
        transmission = getattr(paths_tmp, "junction_transmission", False)
        mask_wedge_wedge = tf.reduce_all(tf.reduce_all(paths.dd_type == 0, axis=1), axis=0)
        mask_wedge_vertex = tf.reduce_all(tf.reduce_all(paths.dd_type == 1, axis=1), axis=0)
        mask_vertex_wedge = tf.reduce_all(tf.reduce_all(paths.dd_type == 2, axis=1), axis=0)
        mask_vertex_vertex = tf.reduce_all(tf.reduce_all(paths.dd_type == 3, axis=1), axis=0)
        mask_shared_vertex = tf.reduce_all(tf.reduce_all(paths.dd_type == 4, axis=1), axis=0)

        if debug_resolve_edges:
            # Dispatch by the same slot groups as the normal field calculation.
            groups = (
                (0, mask_wedge_wedge, self.compute_fields_wedge_wedge),
                (1, mask_wedge_vertex, self.compute_fields_wedge_vertex),
                (2, mask_vertex_wedge, self.compute_fields_vertex_wedge),
                (3, mask_vertex_vertex, None),
                (4, mask_shared_vertex, self.compute_fields_sv),
            )
            parents, pairs, mechanisms, fields = [], [], [], []
            for kind, mask, kernel in groups:
                slots = tf.cast(tf.where(mask)[:, 0], tf.int32)
                if tf.size(slots) == 0:
                    continue
                selected = paths.gather_paths(slots, is_finalized=False)
                selected_tmp = paths_tmp.gather_paths(slots)
                args = (relative_permittivity, scattering_coefficient, selected, selected_tmp)
                if kind in (1, 2):
                    indices, pair, field = kernel(
                        *args, transmission=transmission, debug_resolve_edges=True
                    )
                else:
                    indices = tf.cast(tf.where(selected.mask), tf.int32)
                    if kind == 3:
                        # VV has no field kernel in the standard dispatch.
                        pair = tf.fill([tf.shape(indices)[0], 2], -1)
                        field = tf.zeros([tf.shape(indices)[0], 2, 2], self._dtype)
                    else:
                        pair = tf.stack([
                            tf.gather_nd(selected.objects[d], indices) for d in (0, 1)
                        ], axis=1)
                        packed = kernel(*args, transmission=transmission) if kind == 0 else kernel(*args)
                        field = tf.gather_nd(packed, indices)
                parents.append(tf.concat([indices[:, :2], tf.gather(slots, indices[:, 2:3])], axis=1))
                pairs.append(tf.cast(pair, tf.int32))
                mechanisms.append(tf.fill([tf.shape(indices)[0]], kind))
                fields.append(field)
            if not parents:
                return (tf.zeros([0, 3], tf.int32), tf.zeros([0, 2], tf.int32),
                        tf.zeros([0], tf.int32), tf.zeros([0, 2, 2], self._dtype))
            return tuple(tf.concat(items, axis=0) for items in (parents, pairs, mechanisms, fields))

        # [max_num_paths, num_sources, num_targets, 2, 3]
        mat_t = tf.zeros((paths.vertices[0].shape[2], paths.targets.shape[0], paths.sources.shape[0], 2, 2), dtype=self._dtype)

        ### 1) wedge-wedge
        paths_wedge_wedge = paths.gather_paths(tf.where(mask_wedge_wedge)[:, 0], is_finalized=False)
        paths_tmp_wedge_wedge = paths_tmp.gather_paths(tf.where(mask_wedge_wedge)[:, 0])
        if tf.reduce_any(mask_wedge_wedge):
            mat_t_wedge_wedge = self.compute_fields_wedge_wedge(relative_permittivity, scattering_coefficient, 
                                                                paths_wedge_wedge, paths_tmp_wedge_wedge, transmission=transmission)
            
        #### 1a) wedge-wedge shared vertex
        paths_sv = paths.gather_paths(tf.where(mask_shared_vertex)[:, 0], is_finalized=False)
        paths_tmp_sv = paths_tmp.gather_paths(tf.where(mask_shared_vertex)[:, 0])
        if tf.reduce_any(mask_shared_vertex):
            mat_t_sv = self.compute_fields_sv(relative_permittivity, scattering_coefficient, 
                                                                paths_sv, paths_tmp_sv)

        ### 2) wedge-vertex
        paths_wedge_vertex = paths.gather_paths(tf.where(mask_wedge_vertex)[:, 0], is_finalized=False)
        paths_tmp_wedge_vertex = paths_tmp.gather_paths(tf.where(mask_wedge_vertex)[:, 0])
        if tf.reduce_any(mask_wedge_vertex):
            mat_t_wedge_vertex = self.compute_fields_wedge_vertex(relative_permittivity, scattering_coefficient,
                                                                   paths_wedge_vertex, paths_tmp_wedge_vertex, transmission=transmission)
            
        ### 3) vertex-wedge
        paths_vw = paths.gather_paths(tf.where(mask_vertex_wedge)[:, 0], is_finalized=False)
        paths_tmp_vw = paths_tmp.gather_paths(tf.where(mask_vertex_wedge)[:, 0])
        if tf.reduce_any(mask_vertex_wedge):
            mat_t_vw = self.compute_fields_vertex_wedge(
                relative_permittivity, scattering_coefficient, paths_vw, paths_tmp_vw, transmission=transmission
            )

        if tf.reduce_any(mask_wedge_wedge):
            mat_t = tf.tensor_scatter_nd_update(mat_t, tf.where(mask_wedge_wedge), tf.transpose(mat_t_wedge_wedge, perm=[2, 0, 1, 3, 4]))
        if tf.reduce_any(mask_shared_vertex):
            mat_t = tf.tensor_scatter_nd_update(mat_t, tf.where(mask_shared_vertex), tf.transpose(mat_t_sv, perm=[2, 0, 1, 3, 4]))
        if tf.reduce_any(mask_wedge_vertex):
            mat_t = tf.tensor_scatter_nd_update(mat_t, tf.where(mask_wedge_vertex), tf.transpose(mat_t_wedge_vertex, perm=[2, 0, 1, 3, 4]))
        if tf.reduce_any(mask_vertex_wedge):
            mat_t = tf.tensor_scatter_nd_update(mat_t, tf.where(mask_vertex_wedge), tf.transpose(mat_t_vw, perm=[2, 0, 1, 3, 4]))

        mat_t = tf.transpose(mat_t, perm=[1, 2, 0, 3, 4])

        return mat_t

    def compute_fields_wedge_wedge(self, relative_permittivity, scattering_coefficient,
                                paths: Paths, paths_tmp: PathsTmpData, transmission=False):
        r"""
        Compute EE fields for valid paths using flat intermediate tensors.

        Parameters
        ----------
        relative_permittivity, scattering_coefficient : tf.Tensor
            Retained for interface compatibility. The existing analytical EE
            coefficients do not use these material arrays.
        paths : Paths
            EE paths. The first two interaction slots contain wedge indices
            and diffraction points.
        paths_tmp : PathsTmpData
            Retained for interface compatibility.

        Returns
        -------
        tf.Tensor, complex, shape [targets, sources, paths, 2, 2]
            EE transition matrices in the global spherical polarization bases.
            Invalid paths have zero matrices.
        """

        if not self.is_ww_analytical:
            raise NotImplementedError("Flat EE calculation requires is_ww_analytical=True.")

        shape = tf.shape(paths.mask)
        output_shape = tf.concat([shape, [2, 2]], axis=0)
        valid = paths.mask & (paths.objects[0] >= 0) & (paths.objects[1] >= 0)
        indices = tf.cast(tf.where(valid), tf.int32)

        def compute_valid():
            solver = self.solver_paths
            wd = solver.solver_wedge_diffraction

            wedge1 = tf.gather_nd(paths.objects[0], indices)
            wedge2 = tf.gather_nd(paths.objects[1], indices)
            point1 = tf.gather_nd(paths.vertices[0], indices)
            point2 = tf.gather_nd(paths.vertices[1], indices)
            sources = tf.gather(paths.sources, indices[:, 1])
            targets = tf.gather(paths.targets, indices[:, 0])

            normals1, e_hat1, n1 = wd._wedge_geometry(wedge1)
            normals2, e_hat2, n2 = wd._wedge_geometry(wedge2)

            r1_hat, r1 = normalize(point1 - sources)
            l_hat, l = normalize(point2 - point1)
            r2_hat, r2 = normalize(targets - point2)

            normal1, normal2 = normals1[:, 0], normals2[:, 0]
            if transmission:
                enabled = solver._object_transmission_enabled
                normal1, e_hat1, n1, junction1, glass_obj1 = self._junction_geometry(
                    wedge1, normal1, e_hat1, n1, point1 - sources, point2 - point1, enabled
                )
                normal2, e_hat2, n2, junction2, glass_obj2 = self._junction_geometry(
                    wedge2, normal2, e_hat2, n2, point2 - point1, targets - point2, enabled
                )

            phi1_prime, phi12, beta1_prime, _ = calc_angles(
                normal1, e_hat1, r1_hat, l_hat
            )
            phi1_prime_c, phi12_c, _, _ = calc_angles_concave(
                normal1, e_hat1, r1_hat, l_hat, n1
            )
            phi1_prime = tf.where(n1 < 1.0, phi1_prime_c, phi1_prime)
            phi12 = tf.where(n1 < 1.0, phi12_c, phi12)

            phi12_prime, phi2, beta2_prime, _ = calc_angles(
                normal2, e_hat2, l_hat, r2_hat
            )
            phi12_prime_c, phi2_c, _, _ = calc_angles_concave(
                normal2, e_hat2, l_hat, r2_hat, n2
            )
            phi12_prime = tf.where(n2 < 1.0, phi12_prime_c, phi12_prime)
            phi2 = tf.where(n2 < 1.0, phi2_c, phi2)

            # Preserve the existing treatment of angles close to 2*pi.
            phi12 = tf.where(
                tf.experimental.numpy.isclose(phi12, 2 * PI, atol=1e-2),
                1e-3 * tf.ones_like(phi12), phi12
            )
            phi12_prime = tf.where(
                tf.experimental.numpy.isclose(phi12_prime, 2 * PI, atol=1e-2),
                1e-3 * tf.ones_like(phi12_prime), phi12_prime
            )

            matrices = self._ee_field_matrices_flat(
                phi1_prime, phi12, beta1_prime, phi12_prime, phi2, beta2_prime,
                r1, l, r2, n1, n2, e_hat1, e_hat2, l_hat,
                tf.stack([wedge1, wedge2], axis=0)
            )

            # Apply incident spherical spreading exactly once.
            matrices /= tf.cast(r1[:, None, None], self._dtype)

            phi1_prime_hat, _ = normalize(cross(r1_hat, e_hat1))
            beta1_prime_hat = cross(phi1_prime_hat, r1_hat)
            phi2_hat, _ = normalize(-cross(r2_hat, e_hat2))
            beta2_hat = cross(phi2_hat, r2_hat)

            obj_geom = solver.obj_geom
            if transmission:
                glass1 = self._junction_slab_matrix(
                    junction1, glass_obj1, point1, r1_hat, normal1,
                    phi1_prime_hat, beta1_prime_hat, relative_permittivity,
                )
                # Use the reverse outer-ray basis at the exit, as in the legacy EE model.
                glass2 = self._junction_slab_matrix(
                    junction2, glass_obj2, point2, -r2_hat, normal2,
                    phi2_hat, beta2_hat, relative_permittivity,
                )
                matrices = tf.matmul(glass2, tf.matmul(matrices, glass1))
            elif obj_geom is not None and obj_geom.is_glass_transmission:
                e_i_s1, e_i_p1, _, _ = compute_field_unit_vectors(
                    r1_hat, l_hat, normal1, solver.EPSILON
                )

                # Preserve the reverse-direction convention at the second edge.
                e_i_s2, e_i_p2, _, _ = compute_field_unit_vectors(
                    -r2_hat, -l_hat, normal2, solver.EPSILON
                )
                mask = tf.ones_like(n1, dtype=tf.bool)
                wavelength = solver._scene.wavelength

                glass1 = obj_geom.diffraction_glass_transmission(
                    valid_wedges_idx=wedge1, n=n1, mask=mask,
                    phi_prime_hat=phi1_prime_hat, beta_0_prime_hat=beta1_prime_hat,
                    e_i_s_0=e_i_s1, e_i_p_0=e_i_p1,
                    s_prime_hat=r1_hat, n_0_hat=normal1,
                    wavelength=wavelength, dtype=self._dtype
                )
                glass2 = obj_geom.diffraction_glass_transmission(
                    valid_wedges_idx=wedge2, n=n2, mask=mask,
                    phi_prime_hat=phi2_hat, beta_0_prime_hat=beta2_hat,
                    e_i_s_0=e_i_s2, e_i_p_0=e_i_p2,
                    s_prime_hat=-r2_hat, n_0_hat=normal2,
                    wavelength=wavelength, dtype=self._dtype
                )

                # Keep the original EE glass multiplication order.
                matrices = tf.matmul(glass2, tf.matmul(matrices, glass1))

            theta_t = tf.gather_nd(paths.theta_t, indices)
            phi_t = tf.gather_nd(paths.phi_t, indices)
            theta_r = tf.gather_nd(paths.theta_r, indices)
            phi_r = tf.gather_nd(paths.phi_r, indices)

            # Existing helper already supports flat leading dimensions.
            matrices = solver.convert_mat_t(
                matrices, tf.ones_like(n1, dtype=tf.bool),
                theta_t, phi_t, theta_r, phi_r,
                phi1_prime_hat, beta1_prime_hat, phi2_hat, beta2_hat
            )

            return tf.scatter_nd(indices, matrices, output_shape)

        return tf.cond(
            tf.shape(indices)[0] > 0,
            compute_valid,
            lambda: tf.zeros(output_shape, dtype=self._dtype)
        )

    def _ee_field_matrices_flat(self, phi1_prime, phi12, beta1_prime,
                                phi12_prime, phi2, beta2_prime,
                                r1, l, r2, n1, n2, e_hat1, e_hat2, l_hat,
                                wedge_idxs):
        r"""
        Evaluate analytical EE coefficients for a flat batch of paths.

        Parameters
        ----------
        phi1_prime, phi12, beta1_prime, phi12_prime, phi2, beta2_prime :
            Real tensors of shape [N] containing local ray angles.
        r1, l, r2 : tf.Tensor, real, shape [N]
            Lengths of the three propagation segments.
        n1, n2 : tf.Tensor, real, shape [N]
            Wedge parameters.
        e_hat1, e_hat2, l_hat : tf.Tensor, real, shape [N, 3]
            Edge tangents and propagation direction between the edges.
        wedge_idxs : tf.Tensor, integer, shape [2, N]
            Ordered wedge pairs used by the existing grazing correction.

        Returns
        -------
        tf.Tensor, complex, shape [N, 2, 2]
            EE matrices in the local diffraction bases, including the existing
            spreading factor. The incident factor 1/r1 is excluded.
        """

        solver = self.solver_paths
        k = 2.0 * PI / solver._scene.wavelength
        sin_beta1 = tf.sin(beta1_prime)
        sin_beta2 = tf.sin(beta2_prime)
        sin_product = sin_beta1 * sin_beta2

        # cos_ksi = (
        #     dot(e_hat1, l_hat) * dot(e_hat2, l_hat) - dot(e_hat1, e_hat2)
        # ) / sin_product
        # sin_ksi = dot(cross(e_hat2, e_hat1), l_hat) / sin_product

        l_unit, _ = normalize(l_hat)
        u1, transverse1 = normalize(cross(e_hat1, l_unit))
        u2, transverse2 = normalize(cross(e_hat2, l_unit))

        cos_ksi = -dot(u1, u2, clip=True)
        sin_ksi = dot(cross(u2, u1), l_unit)

        phi1_terms = _get_Phi1_pq(phi1_prime, phi12)
        phi2_terms = _get_Phi2_rs(phi2, phi12_prime)
        l_new = l + r1 * r2 * tf.square(sin_ksi) / (r1 + l + r2)

        # Four angular terms at each edge.
        a_terms = _get_a_pq(r1, l_new, beta1_prime, phi1_terms, n1, k)
        b_terms = _get_b_rs(r2, l_new, beta2_prime, phi2_terms, n2, k)

        # Joint transition functions require all 4x4 combinations per path.
        grid_shape = tf.concat([tf.shape(r1), [4, 4]], axis=0)
        a = tf.broadcast_to(a_terms[:, :, None], grid_shape)
        b = tf.broadcast_to(b_terms[:, None, :], grid_shape)

        w_scalar = cos_ksi * tf.sqrt(r1 * r2 / ((r1 + l) * (r2 + l)))
        w = tf.broadcast_to(w_scalar[:, None, None], grid_shape)
        gfis = _get_gfis2d(a, b, w, solver.obj_geom.diffraction_tables)

        angle1 = phi1_terms[:, :, None] / (2.0 * n1[:, None, None])
        angle2 = phi2_terms[:, None, :] / (2.0 * n2[:, None, None])

        coef_main = tf.complex(
            tf.zeros_like(r1), -1.0 / (8.0 * PI * k * n1 * n2 * sin_product)
        )
        coef_slope = tf.cast(
            -cos_ksi / (32.0 * PI * l_new * tf.square(k * n1 * n2 * sin_product)),
            self._dtype
        )

        ones = tf.ones([4], dtype=self._dtype)
        main_sign = tf.constant([1, -1, -1, 1], dtype=self._dtype)
        slope_sign1 = tf.constant([1, -1, 1, -1], dtype=self._dtype)
        slope_sign2 = tf.constant([1, 1, -1, -1], dtype=self._dtype)

        main_masks = (
            main_sign[:, None] * main_sign[None, :],  # soft
            ones[:, None] * ones[None, :],            # hard
            main_sign[:, None] * ones[None, :],       # sh
            ones[:, None] * main_sign[None, :]        # hs
        )
        slope_masks = (
            slope_sign2[:, None] * slope_sign2[None, :],
            slope_sign1[:, None] * slope_sign1[None, :],
            slope_sign2[:, None] * slope_sign1[None, :],
            slope_sign1[:, None] * slope_sign2[None, :]
        )

        gamma = (
            tf.cast(cos_ksi, self._dtype),
            tf.cast(cos_ksi, self._dtype),
            tf.cast(sin_ksi, self._dtype),
            -tf.cast(sin_ksi, self._dtype)
        )

        def contract(terms, masks, coefficient):
            """Sum angular terms in the original soft/hard/sh/hs order."""
            return tuple(
                g * tf.reduce_sum(terms * mask, axis=(-2, -1)) * coefficient
                for g, mask in zip(gamma, masks)
            )

        if self.dd_mode not in ("all", "main", "slope"):
            raise ValueError(f"Unknown EE mode: {self.dd_mode}")

        main = None
        slope = None

        if self.dd_mode in ("all", "main"):
            transition = _get_func_T(a, b, w, gfis)
            cot_product = (1.0 / tf.tan(angle1)) * (1.0 / tf.tan(angle2))
            terms = tf.cast(cot_product, self._dtype) * transition
            main = contract(terms, main_masks, coef_main)

        if self.dd_mode in ("all", "slope"):
            transition = _get_func_T_2_tilda(a, b, w, gfis)
            csc_product_squared = tf.square(
                (1.0 / tf.sin(angle1)) * (1.0 / tf.sin(angle2))
            )
            terms = tf.cast(csc_product_squared, self._dtype) * transition
            slope = contract(terms, slope_masks, coef_slope)

        if self.dd_mode == "all":
            assert main is not None and slope is not None
            coefficients = tuple(m + s for m, s in zip(main, slope))
        elif self.dd_mode == "main":
            assert main is not None
            coefficients = main
        else:
            assert slope is not None
            coefficients = slope

        # Existing half-weight correction for the selected wedge pairs.
        grazing = self.get_slopes(wedge_idxs)
        grazing_coef = tf.cast(tf.where(grazing, 0.5, 1.0), self._dtype)
        soft, hard, sh, hs = tuple(-grazing_coef * d for d in coefficients)

        row1 = tf.stack([hard, sh], axis=-1)
        row2 = tf.stack([hs, soft], axis=-1)
        matrices = tf.stack([row1, row2], axis=-2)

        # TensorFlow operations retain geometry gradients.
        spreading = tf.sqrt(r1) / (tf.sqrt(l * r2) * tf.sqrt(r1 + l + r2))
        return matrices * -tf.cast(spreading[:, None, None], self._dtype)

    def compute_fields_vertex_wedge(self, relative_permittivity, scattering_coefficient,
                                    paths: Paths, paths_tmp: PathsTmpData, transmission=False,
                                    debug_resolve_edges=False):
        """Reuse the EV calculation through the existing path reversal convention."""
        reversed_paths, reversed_tmp = invert_s_t(paths, paths_tmp)
        matrices = self.compute_fields_wedge_vertex(
            relative_permittivity, scattering_coefficient, reversed_paths, reversed_tmp,
            transmission=transmission, reverse_junction=True, debug_resolve_edges=debug_resolve_edges,
        )
        if debug_resolve_edges:
            parent, pairs, fields = matrices
            # Restore TX -> RX order and original target/source indices.
            return tf.gather(parent, [1, 0, 2], axis=1), tf.reverse(pairs, axis=[1]), fields
        return tf.transpose(matrices, perm=[1, 0, 2, 3, 4])

    def compute_fields_wedge_vertex(self, relative_permittivity, scattering_coefficient,
                                    paths: Paths, paths_tmp: PathsTmpData, transmission=False,
                                    reverse_junction=False, debug_resolve_edges=False):
        solver = self.solver_paths
        scene = solver._scene
        vd = solver.vertex_diffraction
        wavelength = scene.wavelength
        k = 2.0 * PI / wavelength

        path_shape = tf.shape(paths.mask)
        output_shape = tf.concat([path_shape, [2, 2]], axis=0)

        vertex_wedge_mask = getattr(paths, "vertex_wedge_mask", None)
        if vertex_wedge_mask is None:
            raise RuntimeError("Missing vertex_wedge_mask. Trace EV/VE paths again.")

        indices = tf.cast(tf.where(vertex_wedge_mask & paths.mask[..., None]), tf.int32)
        if tf.shape(indices)[0] == 0:
            if debug_resolve_edges:
                return indices[:, :3], tf.zeros([0, 2], tf.int32), tf.zeros([0, 2, 2], self._dtype)
            return tf.zeros(output_shape, dtype=self._dtype)

        # Each row selects one constituent edge of one EV path.
        path_indices = indices[:, :3]
        wedge_table = vd.vertices_2_wedges.to_tensor(default_value=-1)
        vertex_ids = tf.gather_nd(paths.objects[1], path_indices)
        vertex_edge_ids = tf.gather_nd(wedge_table, tf.stack([vertex_ids, indices[:, 3]], axis=-1))
        edge_ids = tf.gather_nd(paths.objects[0], path_indices)

        tf.debugging.assert_non_negative(edge_ids)
        tf.debugging.assert_non_negative(vertex_edge_ids)

        sources = tf.gather(paths.sources, path_indices[:, 1])
        targets = tf.gather(paths.targets, path_indices[:, 0])
        edge_points = tf.gather_nd(paths.vertices[0], path_indices)
        vertex_points = tf.gather_nd(paths.vertices[1], path_indices)

        r1_hat, r1 = normalize(edge_points - sources)
        l_hat, l = normalize(vertex_points - edge_points)
        r2_hat, r2 = normalize(targets - vertex_points)
        active = tf.ones_like(r1, dtype=tf.bool)

        edge_geometry_override = None
        vertex_geometry_override = None
        if transmission:
            enabled = solver._object_transmission_enabled
            normals1, e1, n1 = solver.solver_wedge_diffraction._wedge_geometry(edge_ids)
            e2, normal2, _, n2 = vd._vertex_edge_geometry(vertex_edge_ids, vertex_points)
            normal1, e1, n1, junction1, glass_obj1 = self._junction_geometry(
                edge_ids, normals1[:, 0], e1, n1, edge_points - sources, vertex_points - edge_points, enabled
            )
            normal2, e2, n2, junction2, glass_obj2 = self._junction_geometry(
                vertex_edge_ids, normal2, e2, n2, vertex_points - edge_points, targets - vertex_points, enabled,
                endpoint_swap=vd._vertex_edge_swap(vertex_edge_ids, vertex_points),
            )
            # Use overrides only on rows affected by the junction model.
            edge_geometry_override = (normal1, e1, n1, self._junction_candidates(edge_ids, enabled))
            vertex_geometry_override = (normal2, e2, n2)

        # Keep the original spreading factors included by these two helpers.
        d_a, a_values, edge_geometry = solver.solver_wedge_diffraction.simple_compute_fields(
            edge_ids, r1_hat, l_hat, r1, l, active, skip_F=True,
            geometry_override=edge_geometry_override
        )
        e_hat1, n_0_hat1, n1, phi_prime1, phi1, beta_prime1 = edge_geometry

        d_b, param_b, param_c, vertex_geometry = vd._vertex_edge_coefficients_flat(
            vertex_edge_ids, vertex_points, l_hat, r2_hat, l, r2, add_to_r1=r1,
            geometry_override=vertex_geometry_override
        )
        e_hat2, n_0_hat2, n2, phi_prime2, phi2, beta_prime2, beta2 = vertex_geometry

        param_a = tf.stack(a_values, axis=-1)
        a_ij = param_a[:, :, None]
        b_ij = param_b[:, None, :]
        c_ij = param_c[:, None, :]
        w = tf.sqrt(r1 * r2 / ((r1 + l) * (r2 + l)))[:, None, None]

        d_a_ij = tf.stack(d_a, axis=-1)[:, :, None]
        d_b_ij = tf.stack(d_b, axis=-1)[:, None, :]

        tables = solver.obj_geom.diffraction_tables
        t_even, t_odd = tables.ev_transition(
            tf.sqrt(a_ij), tf.sqrt(b_ij), tf.sqrt(c_ij), w
        )
        cascaded = d_a_ij * d_b_ij * t_even

        soft_signs = tf.cast([1.0, 1.0, -1.0, -1.0], self._dtype)
        soft_signs_ij = soft_signs[:, None] * soft_signs[None, :]
        d_hard = tf.reduce_sum(cascaded, axis=(-2, -1))
        d_soft = tf.reduce_sum(cascaded * soft_signs_ij, axis=(-2, -1))
        matrices = tf.linalg.diag(tf.stack([d_hard, d_soft], axis=-1))

        # e_hat2 is oriented away from the vertex, as in the original EV code.
        gamma = -tf.sign(dot(e_hat1, e_hat2))

        if self.is_add_slope:
            ds_a = slope_diffr_coef_2(
                param_a, phi_prime1, phi1, beta_prime1, n1, scene, skip_F=True
            )
            ds_b = slope_vertex_diffr_coef(
                param_b, param_c, None, phi_prime2, phi2, beta_prime2, beta2,
                n2, scene, tables, active, skip_F=True
            )

            slope_cascaded = tf.stack(ds_a, axis=-1)[:, :, None]
            slope_cascaded = slope_cascaded * tf.stack(ds_b, axis=-1)[:, None, :] * t_odd

            # Edge 1: derivative in phi; vertex edge: derivative in phi_prime.
            soft_a = tf.cast([-1.0, -1.0, -1.0, -1.0], self._dtype)
            hard_a = tf.cast([-1.0, -1.0, 1.0, 1.0], self._dtype)
            soft_b = tf.cast([1.0, 1.0, -1.0, -1.0], self._dtype)

            ds_hard = tf.reduce_sum(slope_cascaded * hard_a[:, None], axis=(-2, -1))
            ds_soft = tf.reduce_sum(slope_cascaded * soft_a[:, None] * soft_b[None, :], axis=(-2, -1))
            slope_matrices = tf.linalg.diag(tf.stack([ds_hard, ds_soft], axis=-1))

            sf = tf.sqrt(r1 / (l * (r1 + l))) / r2
            sf = sf * gamma / (l * k)
            slope_matrices *= tf.complex(tf.zeros_like(sf), sf)[:, None, None]

            if self.dd_mode == "all":
                matrices += slope_matrices
            elif self.dd_mode == "slope":
                matrices = slope_matrices

        grazing = self.get_slopes(tf.stack([edge_ids, vertex_edge_ids], axis=0))
        grazing_factor = tf.where(grazing, tf.ones_like(r1) * 0.5, tf.ones_like(r1))

        # Coplanarity and identical-edge rejection have already been applied.
        matrices *= tf.cast(grazing_factor * gamma, self._dtype)[:, None, None]

        phi1_prime_hat, _ = normalize(cross(r1_hat, e_hat1))
        beta1_prime_hat = cross(phi1_prime_hat, r1_hat)
        phi2_hat, _ = normalize(-cross(r2_hat, e_hat2))
        beta2_hat = cross(phi2_hat, r2_hat)

        if transmission:
            glass1 = self._junction_slab_matrix(
                junction1, glass_obj1, edge_points, r1_hat, n_0_hat1,
                phi1_prime_hat, beta1_prime_hat, relative_permittivity,
            )
            glass2 = self._junction_slab_matrix(
                junction2, glass_obj2, vertex_points, -r2_hat, n_0_hat2,
                phi2_hat, beta2_hat, relative_permittivity,
            )
            matrices = tf.matmul(glass2, tf.matmul(matrices, glass1))
        elif solver.obj_geom is not None and solver.obj_geom.is_glass_transmission:
            e_i_s1, e_i_p1, _, _ = compute_field_unit_vectors(
                r1_hat, l_hat, n_0_hat1, solver.EPSILON
            )
            e_i_s2, e_i_p2, _, _ = compute_field_unit_vectors(
                -r2_hat, -l_hat, n_0_hat2, solver.EPSILON
            )

            glass1 = solver.obj_geom.diffraction_glass_transmission(
                valid_wedges_idx=edge_ids, n=n1, mask=active,
                phi_prime_hat=phi1_prime_hat, beta_0_prime_hat=beta1_prime_hat,
                e_i_s_0=e_i_s1, e_i_p_0=e_i_p1, s_prime_hat=r1_hat,
                n_0_hat=n_0_hat1, wavelength=wavelength, dtype=self._dtype
            )
            glass2 = solver.obj_geom.diffraction_glass_transmission(
                valid_wedges_idx=vertex_edge_ids, n=n2, mask=active,
                phi_prime_hat=phi2_hat, beta_0_prime_hat=beta2_hat,
                e_i_s_0=e_i_s2, e_i_p_0=e_i_p2, s_prime_hat=-r2_hat,
                n_0_hat=n_0_hat2, wavelength=wavelength, dtype=self._dtype
            )

            # Preserve the original operator order: glass2 @ EV @ glass1.
            matrices = tf.matmul(glass2, tf.matmul(matrices, glass1))

        theta_t = tf.gather_nd(paths.theta_t, path_indices)
        phi_t = tf.gather_nd(paths.phi_t, path_indices)
        theta_r = tf.gather_nd(paths.theta_r, path_indices)
        phi_r = tf.gather_nd(paths.phi_r, path_indices)

        # Constituent edges have different local bases; transform before summing.
        matrices = solver.convert_mat_t(
            matrices, active, theta_t, phi_t, theta_r, phi_r,
            phi1_prime_hat, beta1_prime_hat, phi2_hat, beta2_hat
        )
        matrices /= tf.cast(r1[:, None, None], self._dtype)

        if transmission and reverse_junction:
            # Reversed EV uses swapped endpoint bases. Transpose the reciprocal
            # junction contribution before accumulating it into a VE path.
            has_junction = junction1 | junction2
            matrices = tf.where(has_junction[:, None, None], tf.linalg.matrix_transpose(matrices), matrices)

        if debug_resolve_edges:
            pairs = tf.stack([tf.cast(edge_ids, tf.int32), tf.cast(vertex_edge_ids, tf.int32)], axis=1)
            return path_indices, pairs, matrices

        path_ids = (path_indices[:, 0] * path_shape[1] + path_indices[:, 1]) * path_shape[2] + path_indices[:, 2]
        matrices = tf.math.unsorted_segment_sum(matrices, path_ids, tf.reduce_prod(path_shape))
        matrices = tf.reshape(matrices, output_shape)
        return tf.where(paths.mask[..., None, None], matrices, tf.zeros_like(matrices))

    #######################################################
    ## shared vertex 
    #####################################################
    def shared_vertex(self, sources, targets, dd_pairs):
        num_sources, num_targets = tf.shape(sources)[0], tf.shape(targets)[0]

        # [N_valid_pairs, 2]
        valid_pairs, valid_vertices = self.shared_vertex_valid_pairs(dd_pairs)

        # [num_sources, num_targets, max_num_paths, 2], [num_sources, num_targets, max_num_paths]
        dd_pair_idxs, active, (gather_indices, new_scatter_indices) = self.discard_obstructing(valid_pairs, sources, targets, return_indices=True)

        # [num_sources, num_targets, N_valid_pairs]
        valid_vertices_ = tf.repeat(valid_vertices[None, None, ...], num_sources, axis=0)
        valid_vertices_ = tf.repeat(valid_vertices_, num_targets, axis=1)

        num_paths = tf.shape(active)[-1]
        # [num_sources, num_targets, max_num_paths]
        valid_vertex_idxs = tf.fill([num_sources, num_targets, num_paths], -1)
        # [num_keep_paths]
        valid_vertices_ = tf.gather_nd(valid_vertices_, gather_indices)
        # [num_sources, num_targets, max_num_paths]
        valid_vertex_idxs = tf.tensor_scatter_nd_update(valid_vertex_idxs, new_scatter_indices, valid_vertices_)

        return dd_pair_idxs, active, valid_vertex_idxs


    def shared_vertex_valid_pairs(self, dd_pairs):
        wedge_vertices = self.solver_paths.obj_geom.wedge_2_vertices

        wv = tf.gather(wedge_vertices, dd_pairs)

        v1 = wv[:, 0, :]                                # [N_pairs, 2]
        v2 = wv[:, 1, :]                                # [N_pairs, 2]

        eq = v1[:, :, None] == v2[:, None, :]           # [N_pairs, 2, 2]

        has_common_vertex = tf.reduce_any(eq, axis=[1, 2])  # [N_pairs]

        common_vertex = tf.reduce_max(
            tf.where(eq, v1[:, :, None], -1),
            axis=[1, 2]
        ) 

        valid_pairs = tf.boolean_mask(dd_pairs, has_common_vertex)
        valid_vertices = tf.boolean_mask(common_vertex, has_common_vertex)

        return valid_pairs, valid_vertices

    def compute_fields_sv(self, relative_permittivity, scattering_coefficient, paths, paths_tmp):
        """Evaluate the local even/odd SV model in the existing Sionna bases."""
        solver = self.solver_paths
        dtype = solver._scene.dtype
        rdtype = dtype.real_dtype
        wavelength = solver._scene.wavelength
        k = tf.cast(2.0 * PI / wavelength, rdtype)

        if self.dd_mode not in ("all", "main", "slope"):
            raise ValueError(f"Unknown SV mode: {self.dd_mode}")
        use_even = self.dd_mode in ("all", "main")
        use_odd = self.dd_mode in ("all", "slope")

        shape = tf.shape(paths.mask)
        output_shape = tf.concat([shape, [2, 2]], axis=0)
        wedge1, wedge2 = paths.objects[0], paths.objects[1]
        valid = paths.mask & (wedge1 >= 0) & (wedge2 >= 0)
        indices = tf.cast(tf.where(valid), tf.int32)

        if tf.shape(indices)[0] == 0:
            self.delta_E = tf.zeros(shape, dtype)
            self.delta_E_odd = tf.zeros(shape, dtype)
            self.c_sv = tf.zeros(shape, rdtype)
            self.pole_s = tf.zeros(shape, rdtype)
            self.res_hard_base = tf.zeros(shape, dtype)
            self.res_soft_base = tf.zeros(shape, dtype)
            self.res_hard = tf.zeros(shape, dtype)
            self.res_soft = tf.zeros(shape, dtype)
            self.sv_saddle_exist = tf.zeros(shape, tf.bool)
            return tf.zeros(output_shape, dtype)

        diff_point = tf.gather_nd(paths.vertices[0], indices)
        sources = tf.gather(paths.sources, indices[:, 1])
        targets = tf.gather(paths.targets, indices[:, 0])

        def edge_geometry(wedge_ids):
            wedge_ids = tf.gather_nd(wedge_ids, indices)
            normals = tf.gather(self._wedges_normals, wedge_ids)
            cosine = dot(normals[:, 0], normals[:, 1], clip=True)
            wedge_angle = PI - tf.acos(cosine)
            n = (2.0 * PI - wedge_angle) / PI
            e_hat = tf.gather(self._wedges_e_hat, wedge_ids)
            origin = tf.gather(self._wedges_origin, wedge_ids)

            # Both tangents point away from the shared vertex.
            swap = ~tf.reduce_all(tf.equal(origin, diff_point), axis=-1)
            e_hat = tf.where(swap[:, None], -e_hat, e_hat)
            n_0_hat = tf.where(swap[:, None], normals[:, 1], normals[:, 0])
            return e_hat, n_0_hat, n

        e_hat_w1, n_0_hat_1, n1 = edge_geometry(wedge1)
        e_hat_w2, n_0_hat_2, n2 = edge_geometry(wedge2)
        s_prime_hat, s_prime = normalize(diff_point - sources)
        s_hat, s = normalize(targets - diff_point)

        cos_alpha = tf.clip_by_value(dot(e_hat_w1, e_hat_w2), -1.0, 1.0)
        alpha = tf.acos(cos_alpha)
        sin_alpha = tf.sin(alpha)

        # Keep the physical angular sector for both radial contour branches.
        NUM_THETA = 301
        THETA_EPS = 1e-6
        span = tf.cast(PI, rdtype) - alpha
        # tf.debugging.assert_greater(
        #     span, tf.cast(2.0 * THETA_EPS, rdtype),
        #     message="SV edges are too close to a straight angle.",
        # )
        # tf.debugging.assert_positive(sin_alpha, message="Degenerate SV edge angle.")
        grid = tf.linspace(tf.cast(0.0, rdtype), tf.cast(1.0, rdtype), NUM_THETA)
        # [NUM_THETA, N_valid]
        theta = tf.cast(THETA_EPS, rdtype) + grid[:, None] * (
            span[None, :] - tf.cast(2.0 * THETA_EPS, rdtype)
        )

        # Classification uses vertex-to-device directions, not theta samples.
        chi_S = tf.acos(tf.clip_by_value(-dot(s_prime_hat, e_hat_w1), -1.0, 1.0))
        chi_P = tf.acos(tf.clip_by_value(dot(s_hat, e_hat_w2), -1.0, 1.0))
        saddle_exist = chi_S + chi_P + alpha < tf.cast(PI, rdtype)

        g1, g2 = g1_g2_at_vertex(e_hat_w1, e_hat_w2, s_prime_hat, s_hat)
        f1 = tf.sin(theta + alpha[None, :]) / sin_alpha[None, :]
        f2 = tf.sin(theta) / sin_alpha[None, :]
        C = C_theta(theta, g1[None, :], g2[None, :], alpha[None, :])
        D = D_theta(
            f1, f2, s_prime_hat[None, :, :], s_hat[None, :, :],
            s_prime[None, :], s[None, :],
            e_hat_w1[None, :, :], e_hat_w2[None, :, :],
        )
        J = sv_radial_endpoint(C, D, k, saddle_exist, dtype)

        r12_hat = f2[..., None] * e_hat_w2[None, :, :] - f1[..., None] * e_hat_w1[None, :, :]
        r12_hat, _ = normalize(r12_hat)

        phi1_prime, phi12, beta1_prime, beta1 = calc_angles(
            n_0_hat_1[None, :, :], e_hat_w1[None, :, :],
            s_prime_hat[None, :, :], r12_hat,
        )
        phi12_prime, phi2, beta2_prime, beta2 = calc_angles(
            n_0_hat_2[None, :, :], e_hat_w2[None, :, :],
            r12_hat, s_hat[None, :, :],
        )

        # Map the common-face 2*pi representation to zero without moving it.
        phi12 = tf.where(
            tf.experimental.numpy.isclose(phi12, 2.0 * PI, atol=1e-2),
            tf.zeros_like(phi12), phi12,
        )
        phi12_prime = tf.where(
            tf.experimental.numpy.isclose(phi12_prime, 2.0 * PI, atol=1e-2),
            tf.zeros_like(phi12_prime), phi12_prime,
        )
        # Retain small transverse angles on the tight float32 theta grid.
        # acos(dot) can otherwise round beta to exactly zero or pi.
        def elevation(e_hat, direction):
            transverse = cross(e_hat, direction)
            sin_beta = tf.sqrt(tf.reduce_sum(transverse**2, axis=-1))
            return tf.math.atan2(sin_beta, dot(e_hat, direction))

        beta1_prime = elevation(e_hat_w1[None, :, :], s_prime_hat[None, :, :])
        beta1 = elevation(e_hat_w1[None, :, :], r12_hat)
        beta2_prime = elevation(e_hat_w2[None, :, :], r12_hat)
        beta2 = elevation(e_hat_w2[None, :, :], s_hat[None, :, :])
        u1 = solver.vertex_diffraction._get_param_u(beta1_prime, beta1)
        u2 = solver.vertex_diffraction._get_param_u(beta2_prime, beta2)

        # Preserve the original normalization and downstream propagation phase.
        radial_amp = (
            -tf.cast(2.0, dtype) * J
            * tf.cast(wavelength, dtype)
            * tf.cast(4.0 * PI / wavelength, dtype)
            / tf.cast(4.0 * PI * s_prime[None, :], dtype)
            / tf.cast(4.0 * PI * s[None, :], dtype)
            / tf.cast(4.0 * PI, dtype)
            / tf.cast(sin_alpha[None, :], dtype)
        )
        res_hard_base = tf.zeros_like(s_prime, dtype=dtype)
        res_soft_base = tf.zeros_like(s_prime, dtype=dtype)

        if use_even:
            G1 = G_full_hard(phi12, phi1_prime, u1, n1)
            G2 = G_full_hard(phi2, phi12_prime, u2, n2)
            res_hard_base = trapz_axis0(tf.cast(G1 * G2, dtype) * radial_amp, theta)

        if use_odd:
            G1_odd = G_full_odd_soft(phi1_prime, phi12, u1, n1, dtype)
            G2_odd = G_full_odd_soft(phi2, phi12_prime, u2, n2, dtype)
            # Match EE slope signs for two outward coplanar SV tangents.
            # Keep both u parameters in calc_angles' native convention.
            res_soft_base = trapz_axis0(-G1_odd * G2_odd * radial_amp, theta)

        delta_even = tf.zeros_like(s_prime, dtype=dtype)
        delta_odd = tf.zeros_like(s_prime, dtype=dtype)
        c = tf.zeros_like(s_prime)
        pole_s = tf.zeros_like(s_prime)
        if self.sv_add_delta:
            delta_even, delta_odd, c, pole_s = self._compute_sv_delta_flat(
                k, s_prime, s, s_prime_hat, s_hat, e_hat_w1, e_hat_w2,
                sin_alpha, cos_alpha, phi1_prime[0], phi2[0],
                phi12[0], phi12_prime[0], n1, n2, dtype,
                saddle_exist=saddle_exist,
            )
        if not use_even:
            delta_even = tf.zeros_like(delta_even)
        if not use_odd:
            delta_odd = tf.zeros_like(delta_odd)

        res_hard = res_hard_base + delta_even
        res_soft = res_soft_base + delta_odd

        # Dense diagnostics use the same path layout as paths.mask.
        self.res_hard_base = tf.scatter_nd(indices, res_hard_base, shape)
        self.res_soft_base = tf.scatter_nd(indices, res_soft_base, shape)
        self.delta_E = tf.scatter_nd(indices, delta_even, shape)
        self.delta_E_odd = tf.scatter_nd(indices, delta_odd, shape)
        self.res_hard = tf.scatter_nd(indices, res_hard, shape)
        self.res_soft = tf.scatter_nd(indices, res_soft, shape)
        self.c_sv = tf.scatter_nd(indices, c, shape)
        self.pole_s = tf.scatter_nd(indices, pole_s, shape)
        self.sv_saddle_exist = tf.scatter_nd(indices, saddle_exist, shape)

        mat_t = tf.linalg.diag(tf.stack([res_hard, res_soft], axis=-1))
        theta_t = tf.gather_nd(paths.theta_t, indices)
        phi_t = tf.gather_nd(paths.phi_t, indices)
        theta_r = tf.gather_nd(paths.theta_r, indices)
        phi_r = tf.gather_nd(paths.phi_r, indices)
        phi1_prime_hat, _ = normalize(cross(s_prime_hat, e_hat_w1))
        beta1_prime_hat = cross(phi1_prime_hat, s_prime_hat)
        phi2_hat, _ = normalize(-cross(s_hat, e_hat_w2))
        beta2_hat = cross(phi2_hat, s_hat)
        mat_t = solver.convert_mat_t(
            mat_t, tf.ones_like(s_prime, dtype=tf.bool),
            theta_t, phi_t, theta_r, phi_r,
            phi1_prime_hat, beta1_prime_hat, phi2_hat, beta2_hat,
        )
        return tf.scatter_nd(indices, mat_t, output_shape)

    def _compute_sv_delta_flat(
        self, k, s_prime, s, s_prime_hat, s_hat, e_hat_w1, e_hat_w2,
        sin_alpha, cos_alpha, phi1_prime, phi2, q1, q2, n1, n2, dtype,
        saddle_exist=None,
    ):
        """Local one-pole-pair corrections; returns even, odd, signed c and s.

        This is the existing asymptotic correction, continued to signed c<0.
        It is not a full four-pair uniformization. Exact joint pole limits
        remain outside its validity; those rows receive no correction.
        """
        output_rdtype = dtype.real_dtype
        shape = tf.shape(s_prime)
        # Evaluate endpoint differences in double precision on the flat batch.
        s_prime, s = tf.cast(s_prime, tf.float64), tf.cast(s, tf.float64)
        s_prime_hat = tf.cast(s_prime_hat, tf.float64)
        s_hat = tf.cast(s_hat, tf.float64)
        e_hat_w1 = tf.cast(e_hat_w1, tf.float64)
        e_hat_w2 = tf.cast(e_hat_w2, tf.float64)
        sin_alpha, cos_alpha = tf.cast(sin_alpha, tf.float64), tf.cast(cos_alpha, tf.float64)
        phi1_prime, phi2 = tf.cast(phi1_prime, tf.float64), tf.cast(phi2, tf.float64)
        q1, q2 = tf.cast(q1, tf.float64), tf.cast(q2, tf.float64)
        n1, n2, k = tf.cast(n1, tf.float64), tf.cast(n2, tf.float64), tf.cast(k, tf.float64)

        pS = -s_prime * dot(s_prime_hat, e_hat_w1)
        pP = s * dot(s_hat, e_hat_w2)
        hS = tf.sqrt(tf.maximum(s_prime**2 - pS**2, 0.0))
        hP = tf.sqrt(tf.maximum(s**2 - pP**2, 0.0))
        hS_star = pS * sin_alpha + hS * cos_alpha
        hP_star = pP * sin_alpha + hP * cos_alpha
        p_star = pP * cos_alpha - hP * sin_alpha
        Phi_s = tf.sqrt((pS - p_star)**2 + (hS + hP_star)**2)
        Phi_v = s_prime + s

        valid = (hS > 0.0) & (hP > 0.0) & (hS_star > 0.0) & (hP_star > 0.0) & (Phi_s > 0.0)
        rows = tf.cast(tf.where(valid), tf.int32)
        if tf.shape(rows)[0] == 0:
            zero = tf.zeros(shape, dtype)
            return zero, zero, tf.zeros(shape, output_rdtype), tf.zeros(shape, output_rdtype)

        def select(*values):
            return tuple(tf.gather_nd(value, rows) for value in values)

        pS, pP, hS, hP, hS_star, hP_star, Phi_s, Phi_v = select(
            pS, pP, hS, hP, hS_star, hP_star, Phi_s, Phi_v,
        )
        sin_alpha, cos_alpha, phi1_prime, phi2, q1, q2, n1, n2 = select(
            sin_alpha, cos_alpha, phi1_prime, phi2, q1, q2, n1, n2,
        )
        selected_saddle = None if saddle_exist is None else tf.gather_nd(saddle_exist, rows)

        kap = sin_alpha / (1.0 + cos_alpha)
        TS = tf.exp(tf.asinh(pS / hS))
        TP = tf.exp(-tf.asinh(pP / hP))
        f0 = TS - kap * TS * TP - kap - TP
        c = -f0 * tf.sqrt(k * hS * hP / (TS * TP * (1.0 + kap**2) * (Phi_v + Phi_s)))
        w = tf.sqrt(hS * hP / (hS_star * hP_star))
        q1 = tf.where(tf.abs(q1) <= tf.abs(q1 - n1 * PI), tf.zeros_like(q1), n1 * PI)
        q2 = tf.where(tf.abs(q2) <= tf.abs(q2 - n2 * PI), tf.zeros_like(q2), n2 * PI)
        alpha1 = tf.sqrt(k * hS * hP_star / (2.0 * Phi_s))
        alpha2 = tf.sqrt(k * hP * hS_star / (2.0 * Phi_s))
        a_bar = 2.0 * alpha1 * tf.cos((phi1_prime - q1) / 2.0)
        b_bar = 2.0 * alpha2 * tf.cos((phi2 - q2) / 2.0)
        pole_s = w * tf.abs(a_bar) + tf.abs(b_bar)
        sigma = tf.sign(a_bar * b_bar)
        # Each odd pole residue changes sign when its reference face changes.
        odd_pole_sign = (
            tf.sign(tf.sin((phi1_prime - q1) / 2.0))
            * tf.sign(tf.sin((phi2 - q2) / 2.0))
        )

        # Evaluate GFI only on admissible nonzero arguments.
        # Use the analytic one-sided c->0 limit at fixed pole_s>0.
        nonzero_s = pole_s > 0.0
        regular = nonzero_s & (tf.abs(c) > 1e-6)
        reg_rows = tf.cast(tf.where(regular), tf.int32)
        near_rows = tf.cast(tf.where(nonzero_s & ~regular), tf.int32)
        base = tf.zeros(tf.shape(c), tf.complex128)
        N = tf.cast(-8.0 * PI / (4.0 * PI)**3, tf.float64)

        if tf.shape(reg_rows)[0] > 0:
            cr, sr, vr = tuple(tf.gather_nd(v, reg_rows) for v in (c, pole_s, Phi_v))
            T_gfi = tf.cast(t_gfi_full_approximated_capolino(
                cr**2, sr**2, self.obj_geom.diffraction_tables,
            ), tf.complex128)
            F_c = tf.cast(transition_func(cr**2), tf.complex128)
            value = (
                tf.cast(2.0 * PI * N / vr, tf.complex128)
                / tf.cast(1j, tf.complex128)
                * tf.cast(sr / (cr * (cr**2 + sr**2)), tf.complex128)
                * (T_gfi - F_c)
            )
            base = tf.tensor_scatter_nd_update(base, reg_rows, value)

        if tf.shape(near_rows)[0] > 0:
            cn, sn, vn = tuple(tf.gather_nd(v, near_rows) for v in (c, pole_s, Phi_v))
            if selected_saddle is None:
                side = tf.where(cn < 0.0, -tf.ones_like(cn), tf.ones_like(cn))
            else:
                inside = tf.gather_nd(selected_saddle, near_rows)
                side = tf.where(inside, -tf.ones_like(cn), tf.ones_like(cn))
            factor = tf.sqrt(tf.cast(PI, tf.float64)) * tf.exp(
                tf.complex(tf.cast(0.0, tf.float64), tf.cast(PI / 4.0, tf.float64))
            )
            value = (
                tf.cast(2.0 * PI * N * side / vn, tf.complex128)
                / tf.cast(1j, tf.complex128)
                * factor
                * (tf.cast(transition_func(sn**2), tf.complex128) - 1.0)
                / tf.cast(sn, tf.complex128)
            )
            base = tf.tensor_scatter_nd_update(base, near_rows, value)

        delta_even = tf.cast(tf.cast(sigma, tf.complex128) * base, dtype)
        # Include both the transport sign and the two odd pole-residue signs.
        delta_odd = tf.cast(-tf.cast(odd_pole_sign, tf.complex128) * base, dtype)
        return (
            tf.scatter_nd(rows, delta_even, shape),
            tf.scatter_nd(rows, delta_odd, shape),
            tf.scatter_nd(rows, tf.cast(c, output_rdtype), shape),
            tf.scatter_nd(rows, tf.cast(pole_s, output_rdtype), shape),
        )



###################################################################
### Triple diffraction -> probably remove
    
    def ddd_discard_obstructing(self, candidate_pairs, sources, targets):
        r"""
        Discard wedges for which at least one of the source or target are
        "inside" the wedge

        Inputs
        ------
        candidate_pairs : [num_dd_candidates, 3], int
            wedge1, wedge2, wedge3 indexes.

        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        targets : [num_targets, 3], tf.float
            Coordinates of the targets.   

        Output
        -------
        pairs : [num_targets, num_sources, max_num_paths], tf.int
            Indices of the wedges that interacted with the diffracted paths
        """

        num_targets = tf.shape(targets)[0]
        num_sources = tf.shape(sources)[0]

        ## 1) prepare check sources and wedge1
        # [num_sources, num_dd_candidates1]
        _, mask1 = self._discard_obstructing_wedges(candidate_pairs[:, 0], sources)

        ### 2) prepare check targets and wedge2
        # [num_targets, num_dd_candidates2]
        _, mask2 = self._discard_obstructing_wedges(candidate_pairs[:, 2], targets)

        # [num_sources, num_targets, num_dd_candidates]
        mask1 = tf.expand_dims(mask1, axis=1)
        mask1 = tf.repeat(mask1, tf.shape(targets)[0], axis=1)

        # [num_sources, num_targets, num_dd_candidates]
        mask2 = tf.expand_dims(mask2, axis=0)
        mask2 = tf.repeat(mask2, tf.shape(sources)[0], axis=0)

        # [num_sources, num_targets, num_dd_candidates]
        pairs = candidate_pairs[None, None, ...]
        pairs = tf.repeat(pairs, tf.shape(sources)[0], axis=0)
        pairs = tf.repeat(pairs, tf.shape(targets)[0], axis=1)

        mask = tf.logical_and(mask1, mask2)
        #mask = mask[..., None]
        pairs = tf.where(mask[..., None], candidate_pairs, -1)

        gather_indices = tf.where(mask)
        # [num_sources, num_targets, max_num_candidates]
        path_indices = tf.cumsum(tf.cast(mask, tf.int32), axis=-1)
        # [num_valid_paths]
        path_indices2 = tf.gather_nd(path_indices, gather_indices) - 1
        scatter_indices = tf.transpose(gather_indices, [1,0])

        if tf.size(scatter_indices) == 0:
            new_scatter_indices = tf.zeros([3, tf.shape(scatter_indices)[1]], dtype=tf.int32)
        else:
            new_scatter_indices = tf.tensor_scatter_nd_update(scatter_indices,
                                [[2]], [path_indices2])
            
        # [num_valid_paths, 3]
        new_scatter_indices = tf.transpose(new_scatter_indices, [1,0])

        # [num_sources, num_targets]
        num_paths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
        # Maximum number of valid paths
        # ()
        max_num_paths = tf.reduce_max(num_paths)

        # [num_sources, num_targets, max_num_paths]
        new_mask = tf.fill([num_sources, num_targets, max_num_paths], False)
        # [num_keep_paths]
        mask_ = tf.gather_nd(mask, gather_indices)
        # [num_sources, num_targets, max_num_paths]
        new_mask = tf.tensor_scatter_nd_update(new_mask, new_scatter_indices, mask_)
        valid_pairs = -1 * tf.ones([num_sources, num_targets, max_num_paths, 3], dtype=tf.int32)

        pairs_ = tf.gather_nd(pairs, gather_indices)
        new_scatter_indices_0 = tf.pad(new_scatter_indices, [[0, 0], [0, 1]], constant_values=0)
        new_scatter_indices_1 = tf.pad(new_scatter_indices, [[0, 0], [0, 1]], constant_values=1)
        new_scatter_indices_2 = tf.pad(new_scatter_indices, [[0, 0], [0, 1]], constant_values=2)

        # [max_depth, num_sources, num_targets, max_num_paths]
        valid_pairs = tf.tensor_scatter_nd_update(valid_pairs,
                                new_scatter_indices_0, pairs_[:, 0])
        valid_pairs = tf.tensor_scatter_nd_update(valid_pairs,
                                new_scatter_indices_1, pairs_[:, 1])
        valid_pairs = tf.tensor_scatter_nd_update(valid_pairs,
                                new_scatter_indices_2, pairs_[:, 2])

        return valid_pairs, new_mask

    def ddd_compute_diffraction_points(self, sources, targets, wedge_idxs):
        r"""
        If (E1, E2) and (E2, E3) coplanar

        Output
        ------
        diff_points: [num_sources, num_targets, num_pairs, 3, 3], tf.float
            The diffraction points

        wedge_idxs: [num_sources, num_targets, num_pairs, 3], tf.int
        """
        valid_wedge_idxs = tf.where(wedge_idxs == -1, tf.zeros_like(wedge_idxs), wedge_idxs)
        w3_idxs = valid_wedge_idxs[..., 2]
        p3_guess = tf.gather(self._wedges_origin, w3_idxs) + \
                    0.5 * tf.gather(self._wedges_e_hat, w3_idxs) * tf.gather(self._wedges_length, w3_idxs)[..., None]
        
        point3 = p3_guess
        
        for _ in range(5):  # TODO
            diff_points1, valid1 = self.compute_diffraction_points_analytical(sources, point3, valid_wedge_idxs[..., :2])
            diff_points2, valid2 = self.compute_diffraction_points_analytical(diff_points1[..., 0, :], targets, valid_wedge_idxs[..., 1:])
            #tf.norm(diff_points2[..., 1, :] - point3, axis=-1)
            point3 = diff_points2[..., 1, :]

        valid = tf.logical_and(valid1, valid2)
        diff_points = tf.concat([diff_points1, diff_points2[..., 1:, :]], axis=-2)
        # diff_points1[..., 1, :], diff_points2[..., 0, :]

        return diff_points, valid

    def ddd_check_diff_points_validity(self, diff_points, wedge_idxs):
        r"""
        Check if diff_points are on wedges (between start and end points of a wedge)

        Inputs
        ------
        diff_points: [num_sources, num_targets, num_pairs, 3, 3], tf.float
            The diffraction points

        wedge_idxs: [num_sources, num_targets, num_pairs, 3], int
            The indices of the wedges
        """

        valid_wedge_idxs = tf.where(wedge_idxs == -1, 0, wedge_idxs)
        
        # [num_sources, num_targets, num_pairs]
        w1_len = tf.gather(self._wedges_length, valid_wedge_idxs[:, :, :, 0], axis=0)
        w2_len = tf.gather(self._wedges_length, valid_wedge_idxs[:, :, :, 1], axis=0)
        w3_len = tf.gather(self._wedges_length, valid_wedge_idxs[:, :, :, 2], axis=0)

        # [num_sources, num_targets, num_pairs, 3]
        w1_e_hat = tf.gather(self._wedges_e_hat, valid_wedge_idxs[:, :, :, 0], axis=0)
        w2_e_hat = tf.gather(self._wedges_e_hat, valid_wedge_idxs[:, :, :, 1], axis=0)
        w3_e_hat = tf.gather(self._wedges_e_hat, valid_wedge_idxs[:, :, :, 2], axis=0)

        # [num_sources, num_targets, num_pairs, 3]
        w1_start_points = tf.gather(self._wedges_origin, valid_wedge_idxs[:, :, :, 0], axis=0)
        w2_start_points = tf.gather(self._wedges_origin, valid_wedge_idxs[:, :, :, 1], axis=0)
        w3_start_points = tf.gather(self._wedges_origin, valid_wedge_idxs[:, :, :, 2], axis=0)

        w1_end_points = w1_start_points + w1_e_hat * tf.expand_dims(w1_len, axis=-1)
        w2_end_points = w2_start_points + w2_e_hat * tf.expand_dims(w2_len, axis=-1)
        w3_end_points = w3_start_points + w3_e_hat * tf.expand_dims(w3_len, axis=-1)
        
        # [num_sources, num_targets, num_pairs]
        valid_w1 = dot(w1_start_points - diff_points[:, :, :, 0], w1_end_points - diff_points[:, :, :, 0]) <= 0.0
        valid_w2 = dot(w2_start_points - diff_points[:, :, :, 1], w2_end_points - diff_points[:, :, :, 1]) <= 0.0
        valid_w3 = dot(w3_start_points - diff_points[:, :, :, 2], w3_end_points - diff_points[:, :, :, 2]) <= 0.0

        valid = tf.logical_and(tf.logical_and(valid_w1, valid_w2), valid_w3)

        return valid
            
        
def _get_func_T(a, b, w, gfi):
    r"""
    Input
    ------
    a: [num_targets, num_sources, max_num_paths, 4, 4]
    b: [num_targets, num_sources, max_num_paths, 4, 4]
    w: [num_targets, num_sources, max_num_paths, 4, 4]
    gfi: [num_targets, num_sources, max_num_paths, 4, 4, 4]

    Output
    ------
    T: [num_targets, num_sources, max_num_paths, 4, 4]
    """
    w_tmp = tf.sqrt(1 - w**2)
    coef = tf.complex(tf.zeros_like(w), 2 * PI * a * b / w_tmp)
    #coef = 2j * PI * a * b / w_tmp

    return coef * tf.reduce_sum(gfi, axis=-1)

def _get_func_T_2_tilda(a, b, w, gfi): # T_tilda_tilda
    r"""    
    Input
    ------
    a: [num_targets, num_sources, max_num_paths, 4, 4]
    b: [num_targets, num_sources, max_num_paths, 4, 4]
    w: [num_targets, num_sources, max_num_paths, 4, 4]
    gfi: [num_targets, num_sources, max_num_paths, 4, 4, 4]

    Output
    ------
    T_2_tilda: [num_targets, num_sources, max_num_paths, 4, 4]
    """

    w_tmp = tf.sqrt(1 - w**2)
    coef = tf.complex(-4 * PI *(a * b)**2 / (w * w_tmp), tf.zeros_like(w))

    return coef * (gfi[..., 0] + gfi[..., 1] - gfi[..., 2] - gfi[..., 3])



def _get_Phi1_pq(phi1_prime, phi12):
    r"""
    Input
    ------
    phi1_prime: [num_targets, num_sources, max_num_paths]
    phi12: [num_targets, num_sources, max_num_paths]

    Output
    ------
    Phi1_pq: [num_targets, num_sources, max_num_paths, 4]
    """

    # convert from numpy to tensorflow
    # Phi1_00 = np.pi + phi1_rad_prime + phi12_rad  # p=0, q=0
    # Phi1_01 = np.pi + phi1_rad_prime - phi12_rad  # p=0, q=1
    # Phi1_10 = np.pi - phi1_rad_prime + phi12_rad  # p=1, q=0
    # Phi1_11 = np.pi - phi1_rad_prime - phi12_rad  # p=1, q=1

    # return np.array([Phi1_00, Phi1_01, Phi1_10, Phi1_11])
    
    # [num_targets, num_sources, max_num_paths]
    Phi1_00 = PI + phi1_prime + phi12  # p=0, q=0
    Phi1_01 = PI + phi1_prime - phi12  # p=0, q=1
    Phi1_10 = PI - phi1_prime + phi12  # p=1, q=0
    Phi1_11 = PI - phi1_prime - phi12  # p=1, q=1

    return tf.stack([Phi1_00, Phi1_01, Phi1_10, Phi1_11], axis=-1)

    

def _get_Phi2_rs(phi2, phi12_prime):
    r"""
    Input
    ------
    phi2: [num_targets, num_sources, max_num_paths]
    phi12_prime: [num_targets, num_sources, max_num_paths]

    Output
    ------
    Phi2_rs: [num_targets, num_sources, max_num_paths, 4]
    """

    # [num_targets, num_sources, max_num_paths]
    Phi2_00 = PI + phi2 + phi12_prime  # r=0, s=0
    Phi2_01 = PI + phi2 - phi12_prime  # r=0, s=1
    Phi2_10 = PI - phi2 + phi12_prime  # r=1, s=0
    Phi2_11 = PI - phi2 - phi12_prime  # r=1, s=1

    return tf.stack([Phi2_00, Phi2_01, Phi2_10, Phi2_11], axis=-1)

def _get_a_pq(r1, l, beta1, Phi1_pq, n1, k):
    r"""
    Input
    ------
    r1: [num_targets, num_sources, max_num_paths]
    l: [num_targets, num_sources, max_num_paths]
    beta1: [num_targets, num_sources, max_num_paths]
    Phi1_pq: [num_targets, num_sources, max_num_paths, 4]
    n1: [num_targets, num_sources, max_num_paths]
    k : float

    Output
    ------
    a_pq: [num_targets, num_sources, max_num_paths, 4]
    """

    # [num_targets, num_sources, max_num_paths, 4]
    r1 = insert_dim_num(r1, -1, 4)
    l = insert_dim_num(l, -1, 4)
    beta1 = insert_dim_num(beta1, -1, 4)
    n1 = insert_dim_num(n1, -1, 4)

    # [num_targets, num_sources, max_num_paths, 4]
    N = _get_N(n1, Phi1_pq)
    a_pq = tf.sin(beta1) * tf.sqrt(2*k*r1*l / (r1+l)) * tf.sin((Phi1_pq - 2*n1*PI*N) / 2)
    # c1 = (2* tf.sin((Phi1_pq - 2*n1*PI*N) / 2)**2)[99,0,0,-1]  1.1*1e-5

    return a_pq

def _get_b_rs(r2, l, beta2, Phi2_rs, n2, k):
    r"""
    Input
    ------
    r2: [num_targets, num_sources, max_num_paths]
    l: [num_targets, num_sources, max_num_paths]
    beta2: [num_targets, num_sources, max_num_paths]
    Phi2_rs: [num_targets, num_sources, max_num_paths, 4]
    n2: [num_targets, num_sources, max_num_paths]
    k : float

    Output
    ------
    b_rs: [num_targets, num_sources, max_num_paths, 4]
    """
    
    # [num_targets, num_sources, max_num_paths, 4]
    r2 = insert_dim_num(r2, -1, 4)
    l = insert_dim_num(l, -1, 4)
    beta2 = insert_dim_num(beta2, -1, 4)
    n2 = insert_dim_num(n2, -1, 4)
    
    # [num_targets, num_sources, max_num_paths, 4]
    N = _get_N(n2, Phi2_rs)
    b_rs = tf.sin(beta2) * tf.sqrt(2*k*r2*l / (r2+l)) * tf.sin((Phi2_rs - 2*n2*PI*N) / 2)
    # c2 = 2 * tf.sin((Phi2_rs - 2*n2*PI*N) / 2)**2

    return b_rs


def _get_N(n, Phi):
    r"""
    Input
    ------
    n: [num_targets, num_sources, max_num_paths, 4]
    Phi: [num_targets, num_sources, max_num_paths, 4]

    Output
    ------
    N: [num_targets, num_sources, max_num_paths, 4]
    """
    # convert from numpy to tensorflow
    #return np.round(Phi / (2*n*np.pi))

    return tf.math.round(Phi / (2*n*PI))

def _get_gfis2d(a_pq, b_rs, w, diffraction_tables=None):
    r"""
    Input
    ------
    a_pq: [num_targets, num_sources, max_num_paths, 4, 4]
    b_rs: [num_targets, num_sources, max_num_paths, 4, 4]
    w: [num_targets, num_sources, max_num_paths, 4, 4]

    Output
    ------
    gfi: [num_targets, num_sources, max_num_paths, 4, 4, 4]
    """

    #[num_targets, num_sources, max_num_paths, 4, 4]
    # a = insert_dim_num(a_pq, -1, 4)
    # b = insert_dim_num(b_rs, -1, 4)
    w_coef = tf.sqrt(1 - w**2)   

    x = tf.stack([a_pq, b_rs, a_pq, b_rs], axis=-1)
    y = tf.stack([(b_rs + w*a_pq) / w_coef, (a_pq + w*b_rs) / w_coef,
                     (b_rs - w*a_pq) / w_coef, (a_pq - w*b_rs) / w_coef], axis=-1)
    
    gfis = func_G_approximated_capolino(x, y, diffraction_tables)

    return gfis


def my_angles(rot_mat, s_prime_hat, s_hat, facet_0_vector_hat,
              facet_n_vector_hat, e_hat):
    r"""
    Input
    rot_mat: [num_targets, num_sources, max_num_paths, 3, 3]
    s_prime_hat: [num_targets, num_sources, max_num_paths, 3]

    facet_0_vector_hat: [num_targets, num_sources, max_num_paths, 3]
    """

    r1_vector_hat = tf.einsum('...ij, ...j->...i', rot_mat, -s_prime_hat) # TODO
    r2_vector_hat = tf.einsum('...ij, ...j->...i', rot_mat, s_hat)  # TODO
    ksi_cos = facet_0_vector_hat[..., 0]*facet_n_vector_hat[..., 0] + \
                facet_0_vector_hat[..., 1]*facet_n_vector_hat[..., 1]
    ksi_sin = facet_0_vector_hat[..., 0]*facet_n_vector_hat[..., 1] - \
                facet_0_vector_hat[..., 1]*facet_n_vector_hat[..., 0]
    ksi = tf.math.atan2(ksi_sin, ksi_cos)
    needs_swap = tf.math.greater(0.0, ksi)
    facet_0_vector_hat = tf.where(tf.expand_dims(needs_swap, axis=-1), facet_n_vector_hat, facet_0_vector_hat)

    phi1_cos = r1_vector_hat[..., 0]*facet_0_vector_hat[..., 0] + r1_vector_hat[..., 1]*facet_0_vector_hat[..., 1]
    phi1_sin = r1_vector_hat[..., 0]*facet_0_vector_hat[..., 1] - r1_vector_hat[..., 1]*facet_0_vector_hat[..., 0]
    phi_prime = tf.math.atan2(phi1_sin, phi1_cos)
    phi_prime = tf.where(phi_prime < 0, 2*PI - tf.abs(phi_prime), phi_prime)

    # same for phi2
    phi2_cos = r2_vector_hat[..., 0]*facet_0_vector_hat[..., 0] + r2_vector_hat[..., 1]*facet_0_vector_hat[..., 1]
    phi2_sin = r2_vector_hat[..., 0]*facet_0_vector_hat[..., 1] - r2_vector_hat[..., 1]*facet_0_vector_hat[..., 0]
    phi = tf.math.atan2(phi2_sin, phi2_cos)
    phi = tf.where(phi < 0, 2*PI - tf.abs(phi), phi)

    beta_prime = acos_diff(dot(e_hat, s_prime_hat))
    beta = beta = acos_diff(dot(e_hat, s_hat))

    return phi_prime, phi, beta_prime, beta


####################################
## shared vertex
######################################


def C_theta(theta, g1, g2, alpha):
    sin_alpha = tf.sin(alpha)

    f1 = tf.cos(theta) + tf.sin(theta) * tf.cos(alpha) / sin_alpha
    f2 = tf.sin(theta) / sin_alpha

    return g1 * f1 + g2 * f2 + 1.0


def g1_g2_at_vertex(e_hat_w1, e_hat_w2, s_prime_hat, s_hat):
    g1 = tf.reduce_sum(s_prime_hat * e_hat_w1, axis=-1)
    g2 = -tf.reduce_sum(s_hat * e_hat_w2, axis=-1)

    return g1, g2


def D_theta(f1, f2, s_prime_hat, s_hat, s_prime, s, e_hat_w1, e_hat_w2):
    mu1 = tf.reduce_sum(e_hat_w1 * s_prime_hat, axis=-1)
    mu2 = tf.reduce_sum(e_hat_w2 * s_hat, axis=-1)

    h1 = (1.0 - mu1**2) / s_prime
    h2 = (1.0 - mu2**2) / s

    return 0.5 * (h1 * f1**2 + h2 * f2**2)


def B(Phi, u, n):
    return -(1.0 / (2.0 * n)) * tf.sin(Phi / n) / (
        tf.cos(Phi / n) - tf.cosh(u / n)
    )


def G_full_hard(phi, phip, u, n):
    term1 = (
        B(PI + phi - phip, u, n)
        + B(PI - phi + phip, u, n)
    )

    term2 = (
        B(PI + phi + phip, u, n)
        + B(PI - phi - phip, u, n)
    )

    return term1 + term2

def sv_radial_endpoint(C, D, k, saddle_exist, dtype):
    """Quadratic radial endpoint branch, with a finite C=0 limit.

    C, D: [num_theta, N_valid]; saddle_exist: [N_valid].
    The inside branch is the oriented negative radial continuation.
    """
    tf.debugging.assert_positive(D, message="SV radial model requires D > 0.")
    k = tf.cast(k, C.dtype)
    eps_C = tf.cast(1e-7 if C.dtype == tf.float32 else 1e-14, C.dtype)
    small = tf.abs(C) <= eps_C
    safe_C = tf.where(small, tf.ones_like(C), C)
    b_squared = k * C**2 / (4.0 * D)
    F_b = tf.cast(transition_func(b_squared), dtype)
    S_r = tf.cast(tf.sqrt(PI / (k * D)), dtype) * tf.exp(
        tf.complex(tf.zeros_like(b_squared), b_squared - tf.cast(PI / 4.0, C.dtype))
    )
    J = F_b / tf.cast(tf.complex(tf.zeros_like(C), k * safe_C), dtype)
    subtract_full = saddle_exist[None, :] & (C > eps_C)
    J -= tf.where(subtract_full, S_r, tf.zeros_like(S_r))
    side = tf.where(saddle_exist, -tf.ones_like(C[0]), tf.ones_like(C[0]))
    limit = tf.cast(0.5 * side[None, :], dtype) * S_r
    return tf.where(small, limit, J)


def G_full_odd_soft(phi_external, phi_internal, u, n, dtype):
    """Soft odd angular sum in the native outward SV edge convention.

    This is the [+,+,-,-] mask in the EE Phi1_pq/Phi2_rs order.
    The transport sign of the product is applied in compute_fields_sv.
    """
    return (
        vertex_B_odd(PI + phi_external + phi_internal, u, n, dtype)
        + vertex_B_odd(PI + phi_external - phi_internal, u, n, dtype)
        - vertex_B_odd(PI - phi_external + phi_internal, u, n, dtype)
        - vertex_B_odd(PI - phi_external - phi_internal, u, n, dtype)
    )



def trapz_axis0(y, x):
    """
    Trapezoidal integration over axis=0.

    y: [N, ...]
    x: [N] or [N, ...]
    returns: [...]
    """

    y0 = y[:-1, ...]
    y1 = y[1:, ...]

    dx = x[1:, ...] - x[:-1, ...]

    y_mid = 0.5 * (y0 + y1)

    return tf.reduce_sum(y_mid * tf.cast(dx, y.dtype), axis=0)


def F_erfcx(x, dtype):
    x = tf.cast(x, dtype)

    # pi = tf.constant(np.pi, dtype=tf.float32)
    j = tf.constant(1j, dtype=dtype)

    z = tf.exp(j * tf.cast(PI / 4.0, dtype)) * x

    return (
        j
        * tf.sqrt(tf.cast(PI, dtype))
        * x
        * tf.exp(-j * tf.cast(PI / 4.0, dtype))
        * simple_erfcx(z)
    )

def simple_erfcx(z):
    return tf.exp(z*z) * tf.math.erfc(z)


def calc_angles2(n_0_hat, e_hat, s_prime_hat, s_hat):
    r"""
    Input
    ------
    n_0_hat: [num_targets, num_sources, max_num_paths, 3]
    e_hat: [num_targets, num_sources, max_num_paths, 3]
    s_prime_hat: [num_targets, num_sources, max_num_paths, 3]
    s_hat: [num_targets, num_sources, max_num_paths, 3]
    """
    # [num_targets, num_sources, max_num_paths, 3]
    t_0_hat = cross(n_0_hat, e_hat)
    # t_0_hat = cross(e_hat, n_0_hat)

    # Compute s_t_prime_hat and s_t_hat
    # [num_targets, num_sources, max_num_paths, 3]
    s_t_prime_hat, _ = normalize(s_prime_hat
                            - dot(s_prime_hat, e_hat, keepdim=True)*e_hat)
    # [num_targets, num_sources, max_num_paths, 3]
    s_t_hat, _ = normalize(s_hat - dot(s_hat, e_hat, keepdim=True)*e_hat)

    my_phi_prime_tmp = acos_diff(dot(-s_t_prime_hat, t_0_hat))
    my_phi_prime = tf.where(sign(dot(e_hat, cross(t_0_hat, -s_t_prime_hat))) > 0.0, my_phi_prime_tmp, 2*PI - my_phi_prime_tmp)

    my_phi_tmp = acos_diff(dot(s_t_hat, t_0_hat))
    my_phi = tf.where(sign(dot(e_hat, cross(t_0_hat, s_t_hat))) > 0.0, my_phi_tmp, 2*PI - my_phi_tmp)

    # Compute elevation beta_prime
    # [num_targets, num_sources, max_num_paths]
    beta = acos_diff(dot(e_hat, s_hat))
    beta_prime = acos_diff(dot(e_hat, s_prime_hat))

    return my_phi_prime, my_phi, beta_prime, beta
