from __future__ import annotations

import mitsuba as mi
import drjit as dr
import tensorflow as tf
from typing import TYPE_CHECKING, Tuple

from sionna.constants import SPEED_OF_LIGHT, PI
from sionna.utils.tensors import expand_to_rank, insert_dims, flatten_dims,\
    split_dim
from ..paths import Paths, PathsTmpData
from ..utils import dot, phi_hat, theta_hat, theta_phi_from_unit_vec,\
        normalize, moller_trumbore, component_transform, mi_to_tf_tensor,\
        compute_field_unit_vectors, reflection_coefficient, slab_coefficients,\
        cot, cross, sign, rotation_matrix, acos_diff, material_reflection_coefficients
from sionna.rt.diffraction_funcs import calc_angles, calc_angles_concave, my_wd_compute_fields, transition_func,\
        my_wd_compute_fields2, cot_transition_stable, fresnel_integral_capolino

if TYPE_CHECKING:
    from ..solver_base import SolverBase
    from ..solver_paths import SolverPaths

class WedgeDiffraction:
    def __init__(self, solver: SolverPaths):
        self.solver = solver
        self._scene = solver._scene

        self._dtype = solver._dtype
        self._rdtype = solver._rdtype

    def tracing(self, sources, targets, los_prim, diffraction, edge_diffraction, case='both', 
                transmission=False) -> Tuple[Paths, PathsTmpData]:
        
        diff_paths = Paths(sources=sources, targets=targets, scene=self._scene,
                           types=Paths.DIFFRACTED)
        diff_paths_tmp = PathsTmpData(sources, targets, self._dtype)
        diff_paths_tmp.junction_transmission = transmission

        if not diffraction or (los_prim is None and not transmission):
            return diff_paths, diff_paths_tmp

        if los_prim is None:
            candidate_wedges = tf.zeros([0], tf.int32)
        else:
            candidate_wedges = self.solver._wedges_from_primitives(los_prim, edge_diffraction)

        # Keep material junctions even if candidate rays missed their faces.
        has_wedges = tf.shape(self.solver._wedges_objects)[0] > 0
        if transmission and has_wedges:
            array_size = tf.reduce_max(self.solver._primitives_2_objects) + 1
            _, enabled = self.solver._build_slab_properties_tensors(array_size)

            all_wedges = tf.range(tf.shape(self.solver._wedges_objects)[0], dtype=tf.int32)
            face_enabled = tf.gather(enabled, self.solver._wedges_objects)
            mixed = tf.math.logical_xor(face_enabled[:, 0], face_enabled[:, 1])
            junctions = tf.boolean_mask(all_wedges, self._is_flat_shared_wedge(all_wedges) & mixed)
            candidate_wedges, _ = tf.unique(tf.concat([candidate_wedges, junctions], axis=0))

        candidate_shape = tf.stack([tf.shape(targets)[0], tf.shape(sources)[0], tf.size(candidate_wedges)])
        diff_wedges_indices = tf.broadcast_to(candidate_wedges[None, None, :], candidate_shape)

        if transmission and has_wedges:
            origins = tf.gather(self.solver._wedges_origin, candidate_wedges)
            normals = tf.gather(self.solver._wedges_normals[:, 0], candidate_wedges)
            d_tx = dot(sources[:, None, :] - origins[None, :, :], normals[None, :, :])
            d_rx = dot(targets[:, None, :] - origins[None, :, :], normals[None, :, :])
            opposite = self._opposite_sides(d_tx[None, ...], d_rx[:, None, :])

            flat_shared = self._is_flat_shared_wedge(candidate_wedges)
            mixed = tf.gather(mixed, candidate_wedges)

            # The transmitted sector currently requires exactly one transmissive face.
            unsupported = opposite & (flat_shared & ~mixed)[None, None, :]
            diff_wedges_indices = tf.where(unsupported, -1, diff_wedges_indices)

        # Compute the intersection points with the wedges, and discard paths
        # for which the intersection point is not on the finite wedge.
        # diff_wedges_indices : [num_targets, num_sources, max_num_paths]
        #   Indices of the intersected wedges
        # diff_vertices : [num_targets, num_sources, max_num_paths, 3]
        #   Position of the intersection point on the wedges
        diff_wedges_indices, diff_vertices =\
            self._compute_diffraction_points(targets, sources,
                                            diff_wedges_indices)
        
        diff_paths.objects = tf.expand_dims(diff_wedges_indices, axis=0)
        diff_paths.vertices = tf.expand_dims(diff_vertices, axis=0)
        diff_paths.mask = diff_paths.objects != -1
        #diff_paths.targets_sources_mask = diff_paths.mask

        diff_paths.path_types = tf.fill(diff_paths.objects.shape, Paths.DIFFRACTED)

        #diff_paths.depth_mask = tf.ones_like(diff_wedges_indices, dtype=tf.bool)
        #diff_paths.depth_mask = diff_paths.objects != -1

        return diff_paths, diff_paths_tmp


    def check_obstructions(self, diff_paths, diff_paths_tmp, diffraction, case='both'):
        # Discard obstructed diffracted paths
        # Only check for wedge visibility if there is at least one candidate
        # diffracted path
        if not diffraction:
            return diff_paths, diff_paths_tmp
        
        targets = diff_paths.targets
        sources = diff_paths.sources
        diff_wedges_indices = diff_paths.objects[0]
        diff_vertices = diff_paths.vertices[0]

        if diff_wedges_indices.shape[2] == 0: # Number of diff. paths == 0
            return diff_paths, diff_paths_tmp

        if getattr(diff_paths_tmp, "junction_transmission", False) and case == "both":
            diff_wedges_indices = self.solver.solver_reflection._trace_diffraction_transmissions(
                diff_wedges_indices, diff_vertices, sources, targets, diff_paths_tmp
            )
        else:
            diff_wedges_indices, diff_vertices = self._check_wedges_visibility(
                targets, sources, diff_wedges_indices, diff_vertices, case
            )

        # diff_paths = Paths(sources=sources, targets=targets,
        #                 scene=self._scene, types=Paths.DIFFRACTED)
        diff_paths.objects = tf.expand_dims(diff_wedges_indices, axis=0)
        diff_paths.vertices = tf.expand_dims(diff_vertices, axis=0)
        diff_paths.depth_mask = diff_paths.objects != -1

        # Select only the valid paths
        diff_paths = self._gather_valid_diff_paths(diff_paths)

        #diff_paths.depth_mask = tf.ones_like(diff_wedges_indices, dtype=tf.bool)
        #diff_paths.depth_mask = diff_wedges_indices != -1

        # Computes paths length, delays, angles and directions of arrivals
        # and departures for the specular paths
        diff_paths, diff_paths_tmp =\
            self.solver._compute_directions_distances_delays_angles(diff_paths,
                                                    diff_paths_tmp, False)
        
        return diff_paths, diff_paths_tmp

    def _is_flat_shared_wedge(self, wedge_idxs):
        """Identify shared edges whose two faces lie in the same plane."""
        normals = tf.gather(self.solver._wedges_normals, wedge_idxs)
        n0, nn = normals[..., 0, :], normals[..., 1, :]
        parallel = tf.linalg.norm(cross(n0, nn), axis=-1) < 1e-6
        return ~tf.gather(self.solver._is_edge, wedge_idxs) & parallel & (dot(n0, nn) > 0.0)

    def _opposite_sides(self, d_tx, d_rx):
        """Test whether endpoints are strictly on opposite sides of a plane."""
        eps = self.solver.EPSILON
        return ((d_tx > eps) & (d_rx < -eps)) | ((d_tx < -eps) & (d_rx > eps))

    def _discard_obstructing_wedges_and_corners(self, candidate_wedges, targets,
                                                sources, case='both'):
        r"""
        Discard wedges for which at least one of the source or target are
        "inside" the wedge

        Inputs
        ------
        candidate_wedges : [num_candidate_wedges], int
            Candidate wedges.
            Entries correspond to wedges indices.

        targets : [num_targets, 3], tf.float
            Coordinates of the targets.

        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        Output
        -------
        wedges_indices : [num_targets, num_sources, max_num_paths], tf.int
            Indices of the wedges that interacted with the diffracted paths
        """

        epsilon = tf.cast(self.solver.EPSILON, self._rdtype)

        # [num_candidate_wedges, 3]
        origins = tf.gather(self.solver._wedges_origin, candidate_wedges)

        # Expand to broadcast with sources/targets and 0/n faces
        # [1, num_candidate_wedges, 1, 3]
        origins = tf.expand_dims(origins, axis=0)
        origins = tf.expand_dims(origins, axis=2)

        # Normals
        # [num_candidate_wedges, 2, 3]
        # [:,0,:] : 0-face
        # [:,1,:] : n-face
        normals = tf.gather(self.solver._wedges_normals, candidate_wedges)
        # Expand to broadcast with the sources or targets
        # [1, num_candidate_wedges, 2, 3]
        normals = tf.expand_dims(normals, axis=0)

        # Expand to broadcast with candidate and 0/n faces wedges
        # [num_sources, 1, 1, 3]
        sources = expand_to_rank(sources, 4, 1)
        # [num_targets, 1, 1, 3]
        targets = expand_to_rank(targets, 4, 1)
        # Sources/Targets vectors
        # [num_sources, num_candidate_wedges, 1, 3]
        u_t = sources - origins
        # [num_targets, num_candidate_wedges, 1, 3]
        u_r = targets - origins

        # [num_sources, num_candidate_wedges, 2]
        sources_valid_half_space = dot(u_t, normals)
        sources_valid_half_space = tf.greater(sources_valid_half_space,
                    tf.fill(tf.shape(sources_valid_half_space), epsilon))
        # [num_sources, num_candidate_wedges]
        sources_valid_half_space = tf.reduce_any(sources_valid_half_space,
                                                 axis=2)
        # Expand to broadcast with targets
        # [1, num_sources, num_candidate_wedges]
        sources_valid_half_space = tf.expand_dims(sources_valid_half_space,
                                                  axis=0)

        # [num_targets, num_candidate_wedges, 2]
        targets_valid_half_space = dot(u_r, normals)
        targets_valid_half_space = tf.greater(targets_valid_half_space,
                            tf.fill(tf.shape(targets_valid_half_space),epsilon))
        # [num_targets, num_candidate_wedges]
        targets_valid_half_space = tf.reduce_any(targets_valid_half_space,
                                                 axis=2)
        # Expand to broadcast with sources
        # [num_targets, 1, num_candidate_wedges]
        targets_valid_half_space = tf.expand_dims(targets_valid_half_space,
                                                  axis=1)

        # [num_targets, num_sources, max_num_paths = num_candidate_wedges]
        if case == 'both':
            mask = tf.logical_and(sources_valid_half_space,
                                targets_valid_half_space)
        elif case == 'tx':
            mask = sources_valid_half_space
        elif case == 'rx':
            mask = targets_valid_half_space

        ### planar egdes don't obstruct
        is_edge = tf.gather(self.solver._is_edge, candidate_wedges)
        mask = tf.where(is_edge, tf.fill(tf.shape(mask), True), mask)

        # Discard paths with no valid link
        # [max_num_paths]
        valid_paths = tf.where(tf.reduce_any(mask, axis=(0,1)))[:,0]
        # [num_targets, num_sources, max_num_paths]
        mask = tf.gather(mask, valid_paths, axis=2)
        # [max_num_paths]
        wedges_indices = tf.gather(candidate_wedges, valid_paths, axis=0)
        # Set invalid wedges to -1
        # [num_targets, num_sources, max_num_paths]
        wedges_indices = tf.where(mask, wedges_indices, -1)

        return wedges_indices

    def _compute_diffraction_points(self, targets, sources, wedges_indices):
        r"""
        Compute stationary diffraction points on finite wedges.

        The point minimizes the total source-to-edge-to-target path length.
        Candidates whose stationary points lie outside the finite edge are
        discarded. Wedges unused by all links are removed.

        The calculation is performed in double precision, then the resulting
        points are converted back to self._rdtype.

        Input
        ------
        targets : [num_targets, 3], tf.float
            Coordinates of the targets.

        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        wedges_indices : [num_targets, num_sources, max_num_paths], tf.int
            Candidate wedge indices. Invalid candidates have index -1.

        Output
        -------
        wedges_indices : [num_targets, num_sources, num_used_wedges], tf.int
            Wedge indices after discarding invalid candidates and unused wedges.

        vertices : [num_targets, num_sources, num_used_wedges, 3], tf.float
            Stationary interaction points, with dtype self._rdtype.
            Points corresponding to index -1 must be ignored.
        """

        sources = tf.cast(sources, tf.float64)[None, :, None, :]
        targets = tf.cast(targets, tf.float64)[:, None, None, :]

        # Replace invalid indices before gathering; their paths remain masked.
        valid_wedges_idx = tf.where(wedges_indices >= 0, wedges_indices, 0)
        origins = tf.cast(tf.gather(self.solver._wedges_origin, valid_wedges_idx), tf.float64)
        e_hat = tf.cast(tf.gather(self.solver._wedges_e_hat, valid_wedges_idx), tf.float64)
        lengths = tf.cast(tf.gather(self.solver._wedges_length, valid_wedges_idx), tf.float64)

        # Float32 unit vectors are not exactly normalized after casting.
        e_hat = tf.math.divide_no_nan(e_hat, tf.norm(e_hat, axis=-1, keepdims=True))
        origins = expand_to_rank(origins, 4, 0)
        e_hat = expand_to_rank(e_hat, 4, 0)
        lengths = expand_to_rank(lengths, 3, 0)

        source_offsets = sources - origins
        target_offsets = targets - origins

        z_source = dot(source_offsets, e_hat)
        z_target = dot(target_offsets, e_hat)
        source_perp = source_offsets - z_source[..., None] * e_hat
        target_perp = target_offsets - z_target[..., None] * e_hat

        def perpendicular_distance(vector):
            # Avoid evaluating SqrtGrad at zero for axial configurations.
            squared = tf.reduce_sum(tf.square(vector), axis=-1)
            positive = squared > 0.0
            safe_squared = tf.where(positive, squared, tf.ones_like(squared))
            return tf.where(positive, tf.sqrt(safe_squared), tf.zeros_like(squared))

        h_source = perpendicular_distance(source_perp)
        h_target = perpendicular_distance(target_perp)
        h_sum = h_source + h_target

        # Direct stationary-point formula, without a quadratic equation.
        t_min = tf.math.divide_no_nan(z_source * h_target + z_target * h_source, h_sum)

        # If both endpoints lie on the edge line, any point between their
        # projections minimizes the path length. Choose one on the finite edge
        # whenever the two intervals overlap.
        axial_low = tf.maximum(tf.minimum(z_source, z_target), tf.zeros_like(h_sum))
        axial_high = tf.minimum(tf.maximum(z_source, z_target), lengths)
        axial_valid = axial_low <= axial_high
        axial_t = 0.5 * (axial_low + axial_high)
        t_min = tf.where(h_sum > 0.0, t_min, axial_t)

        valid = (wedges_indices >= 0) & (t_min >= 0.0) & (t_min <= lengths)
        valid &= (h_sum > 0.0) | axial_valid
        wedges_indices = tf.where(valid, wedges_indices, -1)
        inter_point = origins + t_min[..., None] * e_hat

        used_wedges = tf.where(tf.reduce_any(wedges_indices >= 0, axis=(0, 1)))[:, 0]
        wedges_indices = tf.gather(wedges_indices, used_wedges, axis=2)
        inter_point = tf.gather(inter_point, used_wedges, axis=2)

        return wedges_indices, tf.cast(inter_point, self._rdtype)

    def _check_wedges_visibility(self, targets, sources, wedges_indices,
                                 vertices, case):
        r"""
        Discard the wedges that are not valid due to obstruction by updating the
        mask and removing the wedges related to no valid links.

        Input
        ------
        targets : [num_targets, 3], tf.float
            Coordinates of the targets.

        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        wedges_indices : [num_targets, num_sources, max_num_paths], tf.int
            Indices of the wedges that interacted with the diffracted paths

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
        wedges_indices : [num_targets, num_sources, max_num_paths], tf.int
            Indices of the wedges that interacted with the diffracted paths

        vertices : [num_targets, num_sources, max_num_paths, 3], tf.float
            Coordinates of the interaction points on the intersected wedges
        """

        max_num_paths = vertices.shape[2]
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
        wedges_points = tf.reshape(vertices, [-1, 3])

        if case == 'both' or case == 'tx':
            # Check visibility between transmitter and wedge
            # Ray origin
            # d : [batch_size, 3]
            # maxt : [batch_size]
            d,maxt = tf.linalg.normalize(wedges_points - sources, axis=1)
            maxt = tf.squeeze(maxt, axis=1)
            # [batch_size]
            valid_t2w = tf.logical_not(self.solver._test_obstruction(sources, d, maxt))
            valid = valid_t2w

        if case == 'both' or case == 'rx':
            # Check visibility between wedge and receiver
            # Ray origin
            # d : [batch_size, 3]
            # maxt : [batch_size]
            d,maxt = tf.linalg.normalize(wedges_points - targets, axis=1)
            maxt = tf.squeeze(maxt, axis=1)
            # [batch_size]
            valid_w2r = tf.logical_not(self.solver._test_obstruction(targets, d, maxt))
            valid = valid_w2r
        
        if case == 'both':
            # Mask obstructed wedges
            # [batch_size]
            valid = tf.logical_and(valid_t2w, valid_w2r)

        # # Mask obstructed wedges
        # [num_targets, num_sources, max_num_paths]
        valid = tf.reshape(valid, [num_targets, num_sources, max_num_paths])
        # Set wedge indices of blocked paths to -1
        wedges_indices = tf.where(valid, wedges_indices, -1)

        # Discard wedges not involved in any link
        # [max_num_paths]
        used_wedges = tf.where(tf.reduce_any(tf.not_equal(wedges_indices, -1),
                                             axis=(0,1)))[:,0]
        # [num_targets, num_sources, max_num_paths, 3]
        vertices = tf.gather(vertices, used_wedges, axis=2)
        # [num_targets, num_sources, max_num_paths]
        wedges_indices = tf.gather(wedges_indices, used_wedges, axis=2)

        return wedges_indices, vertices
    
    def _gather_valid_diff_paths(self, paths):
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
        wedges_indices = paths.objects[0]
        # [num_targets, num_sources, max_num_candidates, 3]
        vertices = paths.vertices[0]
        # [num_sources, 3]
        sources = paths.sources
        # [num_targets, 3]
        targets = paths.targets

        num_sources = tf.shape(sources)[0]
        num_targets = tf.shape(targets)[0]

        # [num_targets, num_sources, max_num_candidates]
        valid = tf.not_equal(wedges_indices, -1)

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
        mask = tf.tensor_scatter_nd_update(mask, scatter_indices,
                tf.fill([tf.shape(scatter_indices)[0]], True))

        # Locations of the interactions
        # [num_targets, num_sources, max_num_paths, 3]
        valid_vertices = tf.zeros([num_targets, num_sources, max_num_paths, 3],
                                  dtype=self._rdtype)
        # [total_num_valid_paths, 3]
        vertices = tf.gather_nd(vertices, gather_indices)
        valid_vertices = tf.tensor_scatter_nd_update(valid_vertices,
                                                     scatter_indices, vertices)

        # Intersected wedges
        # [num_targets, num_sources, max_num_paths]
        valid_wedges_indices = tf.fill([num_targets, num_sources,
                                        max_num_paths], -1)
        # [total_num_valid_paths]
        wedges_indices = tf.gather_nd(wedges_indices, gather_indices)
        valid_wedges_indices = tf.tensor_scatter_nd_update(valid_wedges_indices,
                                            scatter_indices, wedges_indices)

        # [1, num_targets, num_sources, max_num_candidates]
        paths.objects = tf.expand_dims(valid_wedges_indices, axis=0)
        paths.depth_mask = paths.objects != -1
        # [1, num_targets, num_sources, max_num_candidates, 3]
        paths.vertices = tf.expand_dims(valid_vertices, axis=0)
        # [num_targets, num_sources, max_num_candidates]
        paths.mask = mask
        #paths.targets_sources_mask = mask
        # [1, num_targets, num_sources, max_num_candidates]
        paths.path_types = tf.fill(paths.objects.shape, Paths.DIFFRACTED)

        return paths


    def compute_fields(self, relative_permittivity, scattering_coefficient, paths, paths_tmp):
        r"""
        Compute ordinary wedge-diffraction matrices for valid paths only.

        Parameters
        ----------
        relative_permittivity : tf.Tensor, complex, shape [num_objects]
            Relative permittivity indexed by scene object.
        scattering_coefficient : tf.Tensor, real, shape [num_objects]
            Scattering coefficients indexed by scene object.
        paths : Paths
            Traced wedge paths with mask [targets, sources, paths].
        paths_tmp : PathsTmpData
            Retained for compatibility with the field computation interface.

        Returns
        -------
        tf.Tensor, complex, shape [targets, sources, paths, 2, 2]
            Transition matrices in the global spherical polarization bases.
            Invalid paths have zero matrices.
        """

        shape = tf.shape(paths.mask)
        output_shape = tf.concat([shape, [2, 2]], axis=0)
        indices = tf.cast(tf.where(paths.mask & (paths.objects[0] >= 0)), tf.int32)

        junction_indices = indices[:0]

        if getattr(paths_tmp, "junction_transmission", False) and tf.shape(indices)[0] > 0:
            wedge_idxs = tf.gather_nd(paths.objects[0], indices)
            origins = tf.gather(self.solver._wedges_origin, wedge_idxs)
            normals = tf.gather(self.solver._wedges_normals[:, 0], wedge_idxs)
            sources = tf.gather(paths.sources, indices[:, 1])
            targets = tf.gather(paths.targets, indices[:, 0])

            d_tx = dot(sources - origins, normals)
            d_rx = dot(targets - origins, normals)
            face_objects = tf.gather(self.solver._wedges_objects, wedge_idxs)
            enabled = tf.gather(self.solver._object_transmission_enabled, face_objects)
            mixed = tf.math.logical_xor(enabled[:, 0], enabled[:, 1])

            junction = self._is_flat_shared_wedge(wedge_idxs) & mixed & self._opposite_sides(d_tx, d_rx)
            junction_indices = tf.boolean_mask(indices, junction)
            indices = tf.boolean_mask(indices, ~junction)

        def compute_valid():
            # Each selected path produces exactly one matrix.
            wedge_idxs = tf.gather_nd(paths.objects[0], indices)
            points = tf.gather_nd(paths.vertices[0], indices)
            sources = tf.gather(paths.sources, indices[:, 1])
            targets = tf.gather(paths.targets, indices[:, 0])

            theta_t = tf.gather_nd(paths.theta_t, indices)
            phi_t = tf.gather_nd(paths.phi_t, indices)
            theta_r = tf.gather_nd(paths.theta_r, indices)
            phi_r = tf.gather_nd(paths.phi_r, indices)

            normals, e_hat, n = self._wedge_geometry(wedge_idxs)
            etas, scattering = self._wedge_materials(
                relative_permittivity, scattering_coefficient, wedge_idxs, points
            )

            s_prime_hat, s_prime = normalize(points - sources)
            s_hat, s = normalize(targets - points)

            matrices = self._wedge_field_matrices(
                wedge_idxs, normals, e_hat, n, etas, scattering,
                s_prime_hat, s_hat, s_prime, s,
                theta_t, phi_t, theta_r, phi_r
            )

            # Incident spherical spreading is applied once, as in SolverPaths.
            matrices /= tf.cast(s_prime[:, None, None], self._dtype)
            return tf.scatter_nd(indices, matrices, output_shape)

        matrices = tf.cond(
            tf.shape(indices)[0] > 0,
            compute_valid,
            lambda: tf.zeros(output_shape, dtype=self._dtype),
        )

        if tf.shape(junction_indices)[0] > 0:
            junction_matrices = self._flat_junction_transmission_matrices(
                relative_permittivity, scattering_coefficient, paths, junction_indices
            )
            matrices += tf.scatter_nd(junction_indices, junction_matrices, output_shape)

        return matrices

    def _flat_junction_transmission_matrices(
        self, relative_permittivity, scattering_coefficient, paths, indices
    ):
        """Evaluate a virtual metal knife edge with incident-side slab transmission."""
        wedge_idxs = tf.gather_nd(paths.objects[0], indices)
        points = tf.gather_nd(paths.vertices[0], indices)
        sources = tf.gather(paths.sources, indices[:, 1])
        targets = tf.gather(paths.targets, indices[:, 0])

        s_prime_hat, s_prime = normalize(points - sources)
        s_hat, s = normalize(targets - points)

        face_objects = tf.gather(self.solver._wedges_objects, wedge_idxs)
        face_enabled = tf.gather(self.solver._object_transmission_enabled, face_objects)
        glass_face = tf.cast(~face_enabled[:, 0], tf.int32)
        metal_face = 1 - glass_face

        glass_objects = tf.gather(face_objects, glass_face, axis=1, batch_dims=1)
        metal_objects = tf.gather(face_objects, metal_face, axis=1, batch_dims=1)

        face_etas, face_scattering = self._wedge_materials(
            relative_permittivity, scattering_coefficient, wedge_idxs, points
        )
        glass_eta = tf.gather(face_etas, glass_face, axis=1, batch_dims=1)
        metal_eta = tf.gather(face_etas, metal_face, axis=1, batch_dims=1)
        metal_scattering = tf.gather(face_scattering, metal_face, axis=1, batch_dims=1)

        normal = tf.gather(self.solver._wedges_normals[:, 0], wedge_idxs)
        e_hat = tf.gather(self.solver._wedges_e_hat, wedge_idxs)

        # Point cross(normal, e_hat) into the metal half-plane.
        e_hat = tf.where((metal_face == 0)[:, None], e_hat, -e_hat)

        # Face 0 faces TX. Flipping both vectors preserves the metal half-plane.
        from_back = dot(s_prime_hat, normal) > 0.0
        normal = tf.where(from_back[:, None], -normal, normal)
        e_hat = tf.where(from_back[:, None], -e_hat, e_hat)

        # Both faces of the virtual wedge belong to the same metal sheet.
        normals = tf.stack([normal, -normal], axis=1)
        n = tf.fill(tf.shape(s_prime), tf.cast(2.0, self._rdtype))
        etas = tf.stack([metal_eta, metal_eta], axis=1)
        scattering = tf.stack([metal_scattering, metal_scattering], axis=1)
        object_idxs = tf.stack([metal_objects, metal_objects], axis=1)

        _, _, t_s, t_p = slab_coefficients(
            glass_eta, tf.abs(dot(s_prime_hat, normal, clip=True)),
            tf.gather(self.solver._object_thickness, glass_objects),
            self._scene.wavelength, dtype=self._dtype,
        )

        matrices = self._wedge_field_matrices(
            wedge_idxs, normals, e_hat, n, etas, scattering,
            s_prime_hat, s_hat, s_prime, s,
            tf.gather_nd(paths.theta_t, indices), tf.gather_nd(paths.phi_t, indices),
            tf.gather_nd(paths.theta_r, indices), tf.gather_nd(paths.phi_r, indices),
            object_idxs=object_idxs, incident_slab_coeffs=(t_s, t_p),
        )

        # Match the ordinary WD normalization in compute_fields().
        return matrices / tf.cast(s_prime[:, None, None], self._dtype)
    
    def _wedge_geometry(self, wedge_idxs):
        r"""
        Gather geometry without reversing wedge orientation.

        Parameters
        ----------
        wedge_idxs : tf.Tensor, integer, shape [N]
            Valid global wedge indices.

        Returns
        -------
        normals : tf.Tensor, real, shape [N, 2, 3]
            Normals of the 0- and n-faces.
        e_hat : tf.Tensor, real, shape [N, 3]
            Oriented unit edge tangents.
        n : tf.Tensor, real, shape [N]
            Wedge parameters, including concave wedges.
        """

        solver = self.solver
        normals = tf.gather(solver._wedges_normals, wedge_idxs)
        e_hat = tf.gather(solver._wedges_e_hat, wedge_idxs)

        wedge_angle = PI - acos_diff(dot(normals[:, 0], normals[:, 1], clip=True))
        is_concave = tf.gather(solver._is_concave_wedge, wedge_idxs)
        n = tf.where(is_concave, wedge_angle / PI, (2.0 * PI - wedge_angle) / PI)

        return normals, e_hat, n


    def _wedge_materials(self, relative_permittivity, scattering_coefficient, wedge_idxs, points):
        r"""
        Obtain the material properties of both faces for each selected path.

        Parameters
        ----------
        relative_permittivity : tf.Tensor, complex, shape [num_objects]
            Relative permittivity indexed by object.
        scattering_coefficient : tf.Tensor, real, shape [num_objects]
            Scattering coefficients indexed by object.
        wedge_idxs : tf.Tensor, integer, shape [N]
            Valid global wedge indices.
        points : tf.Tensor, real, shape [N, 3]
            Diffraction points supplied to the material callable.

        Returns
        -------
        etas : tf.Tensor, complex, shape [N, 2]
            Relative permittivity of both wedge faces.
        scattering : tf.Tensor, real, shape [N, 2]
            Scattering coefficients of both wedge faces.
        """

        object_idxs = tf.gather(self.solver._wedges_objects, wedge_idxs)
        rm_callable = self._scene.radio_material_callable

        if rm_callable is None:
            return tf.gather(relative_permittivity, object_idxs), tf.gather(scattering_coefficient, object_idxs)

        material_points = tf.repeat(points[:, None, :], 2, axis=1)
        etas, scattering, _ = rm_callable(object_idxs, material_points)
        return etas, scattering


    def _wedge_field_matrices(
        self, wedge_idxs, normals, e_hat, n, etas, scattering,
        s_prime_hat, s_hat, s_prime, s,
        theta_t, phi_t, theta_r, phi_r,
        object_idxs=None, incident_slab_coeffs=None,
    ):
        r"""
        Compute flat ordinary-WD matrices using the existing field convention.

        Parameters
        ----------
        wedge_idxs : tf.Tensor, integer, shape [N]
            Valid wedge indices.
        normals : tf.Tensor, real, shape [N, 2, 3]
            Face normals.
        e_hat : tf.Tensor, real, shape [N, 3]
            Oriented edge tangents.
        n : tf.Tensor, real, shape [N]
            Wedge parameters.
        etas, scattering : tf.Tensor, shape [N, 2]
            Complex permittivities and real scattering coefficients.
        s_prime_hat, s_hat : tf.Tensor, real, shape [N, 3]
            Incident and outgoing propagation directions.
        s_prime, s : tf.Tensor, real, shape [N]
            Source-to-edge and edge-to-target distances.
        theta_t, phi_t, theta_r, phi_r : tf.Tensor, real, shape [N]
            Global departure and arrival polarization-basis angles.

        Returns
        -------
        tf.Tensor, complex, shape [N, 2, 2]
            Matrices including outgoing spreading and optional glass transmission.
            The caller must apply the incident factor 1/s_prime.
        """

        wavelength = self._scene.wavelength
        k = 2.0 * PI / wavelength
        n_0_hat, n_n_hat = normals[:, 0], normals[:, 1]

        # Locate the source in the original oriented cross-sectional plane.
        t_0_hat = cross(n_0_hat, e_hat)
        source_phi = tf.math.floormod(
            tf.atan2(-dot(s_prime_hat, n_0_hat), -dot(s_prime_hat, t_0_hat)),
            2.0 * PI,
        )
        use_other_sector = source_phi > n * PI

        # Select the complementary sector while preserving physical face identities.
        n_0_hat = tf.where(use_other_sector[:, None], -n_0_hat, n_0_hat)
        n_n_hat = tf.where(use_other_sector[:, None], -n_n_hat, n_n_hat)
        e_hat = tf.where(use_other_sector[:, None], -e_hat, e_hat)
        n = tf.where(use_other_sector, 2.0 - n, n)

        # Construct all polarization bases after selecting the sector.
        phi_prime_hat, _ = normalize(cross(s_prime_hat, e_hat))
        beta_0_prime_hat = cross(phi_prime_hat, s_prime_hat)
        phi_hat_, _ = normalize(-cross(s_hat, e_hat))
        beta_0_hat = cross(phi_hat_, s_hat)

        phi_prime, phi, beta_prime, _ = calc_angles(n_0_hat, e_hat, s_prime_hat, s_hat)
        phi_prime_c, phi_c, _, _ = calc_angles_concave(n_0_hat, e_hat, s_prime_hat, s_hat, n)
        phi_prime = tf.where(n < 1.0, phi_prime_c, phi_prime)
        phi = tf.where(n < 1.0, phi_c, phi)

        # WD sector selection preserves physical face ordering.
        if object_idxs is None:
            object_idxs = tf.gather(self.solver._wedges_objects, wedge_idxs)

        face_thickness = tf.gather(self.solver._object_thickness, object_idxs)
        face_slab_enabled = tf.gather(self.solver._object_transmission_enabled, object_idxs)
        # object_idxs = tf.gather(self.solver._wedges_objects, wedge_idxs)
        # face_thickness = tf.gather(self.solver._object_thickness, object_idxs)
        # face_slab_enabled = tf.gather(self.solver._object_transmission_enabled, object_idxs)

        cos_theta_0 = tf.abs(dot(s_prime_hat, n_0_hat, clip=True))
        cos_theta_n = tf.abs(dot(s_prime_hat, n_n_hat, clip=True))
        face_cos_theta = tf.stack([cos_theta_0, cos_theta_n], axis=-1)

        face_r_s, face_r_p = material_reflection_coefficients(
            etas, face_cos_theta, face_thickness, face_slab_enabled, wavelength
        )

        def face_matrix(normal, r_s, r_p, scattering_coef):
            """Express one face reflection operator in the diffraction bases."""
            e_i_s, e_i_p, e_r_s, e_r_p = compute_field_unit_vectors(
                s_prime_hat, s_hat, normal, self.solver.EPSILON
            )
            reduction = tf.cast(tf.sqrt(1.0 - scattering_coef**2), self._dtype)
            reflection = tf.stack([r_s * reduction, r_p * reduction], axis=-1)

            mat_in = tf.cast(
                component_transform(phi_prime_hat, beta_0_prime_hat, e_i_s, e_i_p), self._dtype
            )
            mat_out = tf.cast(
                component_transform(e_r_s, e_r_p, phi_hat_, beta_0_hat), self._dtype
            )
            matrix = -tf.matmul(mat_out, reflection[:, :, None] * mat_in)
            return matrix, e_i_s, e_i_p

        r_0, e_i_s_0, e_i_p_0 = face_matrix(
            n_0_hat, face_r_s[:, 0], face_r_p[:, 0], scattering[:, 0]
        )
        r_n, _, _ = face_matrix(
            n_n_hat, face_r_s[:, 1], face_r_p[:, 1], scattering[:, 1]
        )

        phi_m, phi_p = phi - phi_prime, phi + phi_prime
        d_mul = -tf.cast(tf.exp(-1j * PI / 4.0), self._dtype) / tf.cast(
            2.0 * n * tf.sqrt(2.0 * PI * k) * tf.sin(beta_prime), self._dtype
        )
        ell = s_prime * s / (s_prime + s) * tf.square(tf.sin(beta_prime))
        k_ell = k * ell

        d1 = d_mul * tf.cast(cot_transition_stable(phi_m, n, k_ell, plus=True), self._dtype)
        d2 = d_mul * tf.cast(cot_transition_stable(phi_m, n, k_ell, plus=False), self._dtype)
        d3 = d_mul * tf.cast(cot_transition_stable(phi_p, n, k_ell, plus=True), self._dtype)
        d4 = d_mul * tf.cast(cot_transition_stable(phi_p, n, k_ell, plus=False), self._dtype)

        eye = tf.eye(2, batch_shape=tf.shape(n), dtype=self._dtype)
        matrices = (d1 + d2)[:, None, None] * eye
        matrices += d3[:, None, None] * r_n + d4[:, None, None] * r_0

        # Preserve the original spreading factor and its sign.
        spreading = tf.sqrt(s_prime / (s * (s_prime + s)))
        matrices *= -tf.cast(spreading[:, None, None], self._dtype)

        obj_geom = self.solver.obj_geom

        if incident_slab_coeffs is not None:
            t_s, t_p = incident_slab_coeffs

            # The glass and metal are coplanar, so face 0 supplies the slab TE/TM basis.
            mat_to_sp = tf.cast(component_transform(
                phi_prime_hat, beta_0_prime_hat, e_i_s_0, e_i_p_0
            ), self._dtype)
            mat_from_sp = tf.cast(component_transform(
                e_i_s_0, e_i_p_0, phi_prime_hat, beta_0_prime_hat
            ), self._dtype)

            transmission = tf.stack([t_s, t_p], axis=-1)
            glass = tf.matmul(mat_from_sp, transmission[..., None] * mat_to_sp)
            matrices = tf.matmul(matrices, glass)

        elif obj_geom is not None and obj_geom.is_glass_transmission:
            glass = obj_geom.diffraction_glass_transmission(
                valid_wedges_idx=wedge_idxs, n=n, mask=tf.ones_like(n, dtype=tf.bool),
                phi_prime_hat=phi_prime_hat, beta_0_prime_hat=beta_0_prime_hat,
                e_i_s_0=e_i_s_0, e_i_p_0=e_i_p_0,
                s_prime_hat=s_prime_hat, n_0_hat=n_0_hat,
                wavelength=wavelength, dtype=self._dtype,
            )
            if obj_geom.los_transmissions == "two":
                glass = tf.matmul(glass, glass)
            matrices = tf.matmul(matrices, glass)

        # Preserve the existing NaN handling for the first comparison.
        matrices = tf.where(tf.math.is_nan(tf.math.real(matrices)), tf.zeros_like(matrices), matrices)

        mat_from_gcs = tf.cast(component_transform(
            theta_hat(theta_t, phi_t), phi_hat(phi_t), phi_prime_hat, beta_0_prime_hat
        ), self._dtype)
        mat_to_gcs = tf.cast(component_transform(
            phi_hat_, beta_0_hat, theta_hat(theta_r, phi_r), phi_hat(phi_r)
        ), self._dtype)

        return tf.matmul(mat_to_gcs, tf.matmul(matrices, mat_from_gcs))

    def simple_compute_fields(self, wedge_idxs, s_wedge_prime_hat, s_wedge_hat, s_wedge_prime, s_wedge, mask,
                                ell_ = None, skip_F=False, geometry_override=None):
        # # [num_targets, num_sources, max_num_paths]
        valid_wedges_idx = tf.where(wedge_idxs == -1, 0, wedge_idxs)

        # [num_targets, num_sources, max_num_paths, 2, 3]
        normals = tf.gather(self.solver._wedges_normals, valid_wedges_idx, axis=0)

        # Compute the wedges angle
        # [num_targets, num_sources, max_num_paths]
        cos_wedges_angle = dot(normals[...,0,:],normals[...,1,:], clip=True)
        wedges_angle = PI - tf.math.acos(cos_wedges_angle)
        _n = (2.*PI-wedges_angle)/PI

        is_concave_wedge = tf.gather(self.solver._is_concave_wedge, valid_wedges_idx)
        n = tf.where(is_concave_wedge, wedges_angle / PI, _n)

        # [num_targets, num_sources, max_num_paths, 3]
        e_hat = tf.gather(self.solver._wedges_e_hat, valid_wedges_idx)

        # Extract surface normals
        # [num_targets, num_sources, max_num_paths, 3]
        n_0_hat = normals[...,0,:]
        # [num_targets, num_sources, max_num_paths, 3]
        n_n_hat = normals[...,1,:]

        if geometry_override is not None:
            new_normal, new_tangent, new_n, use_override = geometry_override
            n_0_hat = tf.where(use_override[:, None], new_normal, n_0_hat)
            e_hat = tf.where(use_override[:, None], new_tangent, e_hat)
            n = tf.where(use_override, new_n, n)

        w_phi_prime, w_phi, w_beta_prime, _ = calc_angles(n_0_hat, e_hat, s_wedge_prime_hat, s_wedge_hat)

        is_concave = n < 1.0
        w_phi_prime_, w_phi_, _, _ = calc_angles_concave(n_0_hat, e_hat, s_wedge_prime_hat, s_wedge_hat, n) # concave
        w_phi_prime = tf.where(is_concave, w_phi_prime_, w_phi_prime)
        w_phi = tf.where(is_concave, w_phi_, w_phi)

        w_phi_prime = tf.where(tf.experimental.numpy.isclose(w_phi_prime, 2*PI, atol=1e-2), 1e-3*tf.ones_like(w_phi_prime), w_phi_prime)
        w_phi = tf.where(tf.experimental.numpy.isclose(w_phi, 2*PI, atol=1e-2), 1e-3*tf.ones_like(w_phi), w_phi)

        if ell_ is None:
            ell = s_wedge_prime * s_wedge / (s_wedge_prime + s_wedge) * tf.sin(w_beta_prime)**2
        else:
            ell = ell_ * tf.sin(w_beta_prime)**2

        d_s, par_a_s = my_wd_compute_fields2(w_phi_prime, w_phi, w_beta_prime, ell, s_wedge_prime, 
                                            s_wedge, n, mask, self.solver._scene, skip_F=skip_F)
        
        return d_s, par_a_s, (e_hat, n_0_hat, n, w_phi_prime, w_phi, w_beta_prime)   
    
    def _diffraction_compute_fields(self, s_prime_hat, s_hat, n_0_hat, n_n_hat, e_hat, etas,
                                    scattering_coefficient, s_prime, s, n, mask, 
                                    theta_t, phi_t, theta_r, phi_r, in_local_coordinates=False, skip_sf=False):  
        wavelength = self._scene.wavelength
        k = 2.*PI/wavelength

        # [num_targets, num_sources, max_num_paths]
        eta_0 = etas[..., 0]
        eta_n = etas[..., 1]
        # [num_targets, num_sources, max_num_paths]
        scattering_coefficient_0 = scattering_coefficient[..., 0]
        scattering_coefficient_n = scattering_coefficient[..., 1]

        # Compute phi_prime_hat, beta_0_prime_hat, phi_hat, beta_0_hat
        # [num_targets, num_sources, max_num_paths, 3]
        phi_prime_hat, _ = normalize(cross(s_prime_hat, e_hat))
        # [num_targets, num_sources, max_num_paths, 3]
        beta_0_prime_hat = cross(phi_prime_hat, s_prime_hat)

        # [num_targets, num_sources, max_num_paths, 3]
        phi_hat_, _ = normalize(-cross(s_hat, e_hat))
        beta_0_hat = cross(phi_hat_, s_hat)

        is_concave = n < 1.0

        phi_prime, phi, beta_prime, _ = calc_angles(n_0_hat, e_hat, s_prime_hat, s_hat)
        phi_prime_, phi_, _, _ = calc_angles_concave(n_0_hat, e_hat, s_prime_hat, s_hat, n) # concave
        phi_prime = tf.where(is_concave, phi_prime_, phi_prime)
        phi = tf.where(is_concave, phi_, phi)

        # Compute field component vectors for reflections at both surfaces
        # [num_targets, num_sources, max_num_paths, 3]
        # pylint: disable=unbalanced-tuple-unpacking
        e_i_s_0, e_i_p_0, e_r_s_0, e_r_p_0 = compute_field_unit_vectors(
            s_prime_hat,
            s_hat,
            n_0_hat,#*sign(-dot(s_t_prime_hat, n_0_hat, keepdim=True)),
            self.solver.EPSILON
            )
        # [num_targets, num_sources, max_num_paths, 3]
        # pylint: disable=unbalanced-tuple-unpacking
        e_i_s_n, e_i_p_n, e_r_s_n, e_r_p_n = compute_field_unit_vectors(
            s_prime_hat,
            s_hat,
            n_n_hat,#*sign(-dot(s_t_prime_hat, n_n_hat, keepdim=True)),
            self.solver.EPSILON
            )

        # Compute Fresnel reflection coefficients for 0- and n-surfaces
        # [num_targets, num_sources, max_num_paths]
        r_s_0, r_p_0 = reflection_coefficient(eta_0, tf.abs(tf.sin(phi_prime)))
        r_s_n, r_p_n = reflection_coefficient(eta_n, tf.abs(tf.sin(n*PI-phi)))

        # Multiply the reflection coefficients with the
        # corresponding reflection reduction factor
        reduction_factor_0 = tf.sqrt(1 - scattering_coefficient_0**2)
        reduction_factor_0 = tf.complex(reduction_factor_0,
                                        tf.zeros_like(reduction_factor_0))
        reduction_factor_n = tf.sqrt(1 - scattering_coefficient_n**2)
        reduction_factor_n = tf.complex(reduction_factor_n,
                                        tf.zeros_like(reduction_factor_n))
        r_s_0 *= reduction_factor_0
        r_p_0 *= reduction_factor_0
        r_s_n *= reduction_factor_n
        r_p_n *= reduction_factor_n

        # Compute matrices R_0, R_n
        # [num_targets, num_sources, max_num_paths, 2, 2]
        w_i_0 = component_transform(phi_prime_hat,
                                    beta_0_prime_hat,
                                    e_i_s_0,
                                    e_i_p_0)
        w_i_0 = tf.complex(w_i_0, tf.zeros_like(w_i_0))
        # [num_targets, num_sources, max_num_paths, 2, 2]
        w_r_0 = component_transform(e_r_s_0,
                                    e_r_p_0,
                                    phi_hat_,
                                    beta_0_hat)
        w_r_0 = tf.complex(w_r_0, tf.zeros_like(w_r_0))
        # [num_targets, num_sources, max_num_paths, 2, 1]
        r_0 = tf.expand_dims(tf.stack([r_s_0, r_p_0], -1), -1) * w_i_0
        # [num_targets, num_sources, max_num_paths, 2, 1]
        r_0 = -tf.matmul(w_r_0, r_0)

        # [num_targets, num_sources, max_num_paths, 2, 2]
        w_i_n = component_transform(phi_prime_hat,
                                    beta_0_prime_hat,
                                    e_i_s_n,
                                    e_i_p_n)
        w_i_n = tf.complex(w_i_n, tf.zeros_like(w_i_n))
        # [num_targets, num_sources, max_num_paths, 2, 2]
        w_r_n = component_transform(e_r_s_n,
                                    e_r_p_n,
                                    phi_hat_,
                                    beta_0_hat)
        w_r_n = tf.complex(w_r_n, tf.zeros_like(w_r_n))
        # [num_targets, num_sources, max_num_paths, 2, 1]
        r_n = tf.expand_dims(tf.stack([r_s_n, r_p_n], -1), -1) * w_i_n
        # [num_targets, num_sources, max_num_paths, 2, 1]
        r_n = -tf.matmul(w_r_n, r_n)

        # Compute D_1, D_2, D_3, D_4
        # [num_targets, num_sources, max_num_paths]
        phi_m = phi - phi_prime
        phi_p = phi + phi_prime

        # [num_targets, num_sources, max_num_paths]
        cot_1 = cot((PI + phi_m)/(2*n))
        cot_2 = cot((PI - phi_m)/(2*n))
        cot_3 = cot((PI + phi_p)/(2*n))
        cot_4 = cot((PI - phi_p)/(2*n))

        def n_p(beta, n):
            return tf.math.round((beta + PI)/(2.*n*PI))

        def n_m(beta, n):
            return tf.math.round((beta - PI)/(2.*n*PI))

        def a_p(beta, n):
            return 2*tf.cos((2.*n*PI*n_p(beta, n)-beta)/2.)**2

        def a_m(beta, n):
            return 2*tf.cos((2.*n*PI*n_m(beta, n)-beta)/2.)**2


        d_mul = - tf.cast(tf.exp(-1j*PI/4.), self._dtype)/\
            tf.cast((2*n)*tf.sqrt(2*PI*k)*tf.math.sin(beta_prime), self._dtype)

        # [num_targets, num_sources, max_num_paths]
        ell = s_prime*s/(s_prime + s) * tf.math.sin(beta_prime)**2

        # [num_targets, num_sources, max_num_paths]
        cot_1 = tf.complex(cot_1, tf.zeros_like(cot_1))
        cot_2 = tf.complex(cot_2, tf.zeros_like(cot_2))
        cot_3 = tf.complex(cot_3, tf.zeros_like(cot_3))
        cot_4 = tf.complex(cot_4, tf.zeros_like(cot_4))
        
        d_1 = d_mul*cot_1 * transition_func(k*ell*a_p(phi_m, n))
        d_2 = d_mul*cot_2 * transition_func(k*ell*a_m(phi_m, n))
        d_3 = d_mul*cot_3 * transition_func(k*ell*a_p(phi_p, n))
        d_4 = d_mul*cot_4 * transition_func(k*ell*a_m(phi_p, n))

        if in_local_coordinates:
            d_soft = d_1 + d_2 - d_3 - d_4
            d_hard = d_1 + d_2 + d_3 + d_4
            # d_soft = d_1 + d_2 + rd_3 - d_4
            # d_hard = d_1 + d_2 + d_3 + d_4
            _mat_t1 = tf.stack([d_hard, tf.zeros_like(d_hard)], axis=-1)
            _mat_t2 = tf.stack([tf.zeros_like(d_soft), d_soft], axis=-1)
            mat_t = tf.stack([_mat_t1, _mat_t2], axis=-2)

            spreading_factor = tf.sqrt(s_prime / (s*(s_prime + s)))
            spreading_factor = tf.complex(spreading_factor, tf.zeros_like(spreading_factor))
        
            mat_t *= -spreading_factor[..., None, None]
            # [num_targets, num_sources, max_num_paths, 1, 1]
            mask_ = expand_to_rank(mask, 5, axis=3)
            # [num_targets, num_sources, max_num_paths, 2]
            mat_t = tf.where(mask_, mat_t, tf.zeros_like(mat_t))

            return mat_t, beta_0_prime_hat, phi_prime_hat, phi_hat_, beta_0_hat

        # [num_targets, num_sources, max_num_paths, 1, 1]
        d_1 = tf.reshape(d_1, tf.concat([tf.shape(d_1), [1,1]], axis=0))
        d_2 = tf.reshape(d_2, tf.concat([tf.shape(d_2), [1,1]], axis=0))
        d_3 = tf.reshape(d_3, tf.concat([tf.shape(d_3), [1,1]], axis=0))
        d_4 = tf.reshape(d_4, tf.concat([tf.shape(d_4), [1,1]], axis=0))

        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = (d_1+d_2)*tf.eye(2,2, batch_shape=tf.shape(r_0)[:3],
                                 dtype=self._dtype)
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t += d_3*r_n + d_4*r_0
        
        mat_from_gcs = component_transform(
                            theta_hat(theta_t, phi_t), phi_hat(phi_t),
                            phi_prime_hat, beta_0_prime_hat)
        mat_from_gcs = tf.complex(mat_from_gcs, tf.zeros_like(mat_from_gcs))

        mat_to_gcs = component_transform(phi_hat_, beta_0_hat,
                                      theta_hat(theta_r, phi_r), phi_hat(phi_r))
        mat_to_gcs = tf.complex(mat_to_gcs, tf.zeros_like(mat_to_gcs))

        mat_t = tf.linalg.matmul(mat_t, mat_from_gcs)
        mat_t = tf.linalg.matmul(mat_to_gcs, mat_t)

        # Set invalid paths to 0
        # Expand masks to broadcast with the field components
        # [num_targets, num_sources, max_num_paths, 1, 1]
        mask_ = expand_to_rank(mask, 5, axis=3)
        # Zeroing coefficients corresponding to non-valid paths
        # [num_targets, num_sources, max_num_paths, 2]
        mat_t = tf.where(mask_, mat_t, tf.zeros_like(mat_t))

        if skip_sf:
            return -mat_t

        # [num_targets, num_sources, max_num_paths]
        spreading_factor = tf.sqrt(s_prime / (s*(s_prime + s)))
        spreading_factor = tf.complex(spreading_factor,
                                      tf.zeros_like(spreading_factor))

        # [num_targets, num_sources, max_num_paths, 2]
        mat_t *= -spreading_factor[..., None]

        return mat_t
    
    def _compute_refl_diff(self,
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
        mask_11 = tf.reduce_all(tf.reduce_all(paths.dd_type == 11, axis=1), axis=0)
        mask_12 = tf.reduce_all(tf.reduce_all(paths.dd_type == 12, axis=1), axis=0)

        # [max_num_paths, num_sources, num_targets, 2, 3]
        mat_t = tf.zeros((paths.vertices[0].shape[2], paths.targets.shape[0], paths.sources.shape[0], 2, 2), dtype=self._dtype)

        ### 1) refl-diff
        paths_11 = paths.gather_paths(tf.where(mask_11)[:, 0], is_finalized=False)
        paths_tmp_11 = paths_tmp.gather_paths(tf.where(mask_11)[:, 0])
        if tf.reduce_any(mask_11):
            mat_t_11 = self._compute_refl_diff_11(relative_permittivity, scattering_coefficient, paths_11, paths_tmp_11)

        ### 2) diff-refl
        paths_12 = paths.gather_paths(tf.where(mask_12)[:, 0], is_finalized=False)
        paths_tmp_12 = paths_tmp.gather_paths(tf.where(mask_12)[:, 0])
        if tf.reduce_any(mask_12):
            mat_t_12 = self._compute_refl_diff_12(relative_permittivity, scattering_coefficient, paths_12, paths_tmp_12)

        if tf.reduce_any(mask_11):
            mat_t = tf.tensor_scatter_nd_update(mat_t, tf.where(mask_11), tf.transpose(mat_t_11, perm=[2, 0, 1, 3, 4]))
        if tf.reduce_any(mask_12):
            mat_t = tf.tensor_scatter_nd_update(mat_t, tf.where(mask_12), tf.transpose(mat_t_12, perm=[2, 0, 1, 3, 4]))

        mat_t = tf.transpose(mat_t, perm=[1, 2, 0, 3, 4])

        return mat_t
        

    def _compute_refl_diff_11(self,
                            relative_permittivity,
                            scattering_coefficient,
                            paths,
                            paths_tmp):
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

        wavelength = self.solver._scene.wavelength
        k = 2.0*PI/wavelength

        #### TX - Refl point - Diff point - RX
        # [num_targets, num_sources, max_num_paths, 3]
        refl_points = tf.gather(paths.vertices, 0, axis=0)
        diff_points = tf.gather(paths.vertices, 1, axis=0)
        # [num_targets, num_sources, max_num_paths]
        refl_idxs = tf.gather(objects, 0, axis=0)
        wedge_idxs = tf.gather(objects, 1, axis=0)
        valid_wedges_idx = tf.where(wedge_idxs == -1, 0, wedge_idxs)
        objects_indices_diff = tf.gather(self.solver._wedges_objects, valid_wedges_idx, axis=0)

        r1_hat, r1 = normalize(refl_points - sources)
        l_hat, l = normalize(diff_points - refl_points)
        r2_hat, r2 = normalize(targets - diff_points)

        # Relative perimittivities and scattering coefficients.
        # If a callable is defined to compute the radio material properties,
        # it is invoked. Otherwise, the radio materials of objects are used.
        rm_callable = self._scene.radio_material_callable
        if rm_callable is None:
            # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
            # This makes no difference on the resulting paths as such paths
            # are not flagged as active.
            # [max_depth, num_targets, num_sources, max_num_paths]
            valid_refl_object_idx = tf.where(refl_idxs == -1, 0, refl_idxs)

            # [max_depth, num_targets, num_sources, max_num_paths]
            etas_refl = tf.gather(relative_permittivity, valid_refl_object_idx)
            scattering_coefficient_refl = tf.gather(scattering_coefficient, valid_refl_object_idx)
            etas_diff = tf.gather(relative_permittivity, objects_indices_diff)
            scattering_coefficient_diff = tf.gather(scattering_coefficient, objects_indices_diff)
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
        mat_t_refl = self.solver.solver_reflection._foo(refl_idxs, _k_t, _k_r, _normals, reduction_factor_refl, etas_refl)

        #### 2) Wedge Diffraction
        # [num_targets, num_sources, max_num_paths]
        # valid_wedges_idx = tf.where(wedge_idxs == -1, 0, wedge_idxs)

        # [num_targets, num_sources, max_num_paths, 2, 3]
        normals = tf.gather(self.solver._wedges_normals, valid_wedges_idx, axis=0)

        # Compute the wedges angle
        # [num_targets, num_sources, max_num_paths]
        cos_wedges_angle = dot(normals[..., 0, :], normals[..., 1, :], clip=True)
        wedges_angle = PI - tf.math.acos(cos_wedges_angle)
        n = (2.*PI-wedges_angle)/PI

        is_concave_wedge = tf.gather(self.solver._is_concave_wedge, valid_wedges_idx)
        n = tf.where(is_concave_wedge, wedges_angle / PI, n)

        # [num_targets, num_sources, max_num_paths, 3]
        e_hat = tf.gather(self.solver._wedges_e_hat, valid_wedges_idx)

        # [num_targets, num_sources, max_num_paths, 3]
        n_0_hat = normals[...,0,:]
        # [num_targets, num_sources, max_num_paths, 3]
        n_n_hat = normals[...,1,:]

        _theta_t, _phi_t = theta_phi_from_unit_vec(k_i[1])

        mat_t_diff = self._diffraction_compute_fields(l_hat, r2_hat, n_0_hat, n_n_hat, e_hat, etas_diff,
                                    scattering_coefficient_diff, r1+l, r2, n, mask, _theta_t, _phi_t, theta_r, phi_r, skip_sf=True)

        ### 3) total field
        sf = 1 / tf.sqrt(r2 * (r1+l) * (r1+l+r2))
        mat_t = tf.multiply(mat_t_refl, mat_t_diff) * tf.cast(sf, self.solver._dtype)[..., None, None]

        return mat_t
    
    def _compute_refl_diff_12(self,
                            relative_permittivity,
                            scattering_coefficient,
                            paths,
                            paths_tmp):
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

        wavelength = self.solver._scene.wavelength
        k = 2.0*PI/wavelength

        #### TX - Diff_point - Refl_point - RX
        # [num_targets, num_sources, max_num_paths, 3]
        diff_points = tf.gather(paths.vertices, 0, axis=0)
        refl_points = tf.gather(paths.vertices, 1, axis=0)
        # [num_targets, num_sources, max_num_paths]
        wedge_idxs = tf.gather(objects, 0, axis=0)
        refl_idxs = tf.gather(objects, 1, axis=0)
        valid_wedges_idx = tf.where(wedge_idxs == -1, 0, wedge_idxs)
        objects_indices_diff = tf.gather(self.solver._wedges_objects, valid_wedges_idx, axis=0)

        rm_callable = self._scene.radio_material_callable
        if rm_callable is None:
            # [max_depth, num_targets, num_sources, max_num_paths]
            valid_refl_object_idx = tf.where(refl_idxs == -1, 0, refl_idxs)

            # [max_depth, num_targets, num_sources, max_num_paths]
            etas_refl = tf.gather(relative_permittivity, valid_refl_object_idx)
            scattering_coefficient_refl = tf.gather(scattering_coefficient, valid_refl_object_idx)
            etas_diff = tf.gather(relative_permittivity, objects_indices_diff)
            scattering_coefficient_diff = tf.gather(scattering_coefficient, objects_indices_diff)
        else:
            # [max_depth, num_targets, num_sources, max_num_paths]
            etas, scattering_coefficient, _  = rm_callable(objects, paths.vertices)

        ################################
        r1_hat, r1 = normalize(diff_points - sources)
        l_hat, l = normalize(refl_points - diff_points)
        r2_hat, r2 = normalize(targets - refl_points)

        #### 1) Wedge Diffraction
        # [num_targets, num_sources, max_num_paths, 2, 3]
        normals = tf.gather(self.solver._wedges_normals, valid_wedges_idx, axis=0)

        # Compute the wedges angle
        # [num_targets, num_sources, max_num_paths]
        cos_wedges_angle = dot(normals[..., 0, :], normals[..., 1, :], clip=True)
        wedges_angle = PI - tf.math.acos(cos_wedges_angle)
        n = (2.*PI-wedges_angle)/PI

        is_concave_wedge = tf.gather(self.solver._is_concave_wedge, valid_wedges_idx)
        n = tf.where(is_concave_wedge, wedges_angle / PI, n)

        # [num_targets, num_sources, max_num_paths, 3]
        e_hat = tf.gather(self.solver._wedges_e_hat, valid_wedges_idx)

        # [num_targets, num_sources, max_num_paths, 3]
        n_0_hat = normals[...,0,:]
        # [num_targets, num_sources, max_num_paths, 3]
        n_n_hat = normals[...,1,:]

        _theta_r, _phi_r = theta_phi_from_unit_vec(-l_hat)  # l_hat

        mat_t_diff = self._diffraction_compute_fields(r1_hat, l_hat, n_0_hat, n_n_hat, e_hat, etas_diff,
                                    scattering_coefficient_diff, r1, l+r2, n, mask, theta_t, phi_t, _theta_r, _phi_r, skip_sf=True)
        
        #### 2) Reflection
        # [max_depth, num_targets, num_sources, max_num_paths]
        reduction_factor_refl = tf.sqrt(1 - scattering_coefficient_refl**2)
        reduction_factor_refl = tf.complex(reduction_factor_refl, tf.zeros_like(reduction_factor_refl))
        _k_t = k_i[1] # l_hat
        _k_r = k_r[1] # r2_hat
        mat_t_refl = self.solver.solver_reflection._foo(refl_idxs, _k_t, _k_r, _normals, reduction_factor_refl, etas_refl)

        ### 3) total field
        sf = 1 / tf.sqrt(r1 * (r2+l) * (r1+l+r2))
        mat_t = tf.multiply(mat_t_diff, mat_t_refl) * tf.cast(sf, self.solver._dtype)[..., None, None]

        return mat_t