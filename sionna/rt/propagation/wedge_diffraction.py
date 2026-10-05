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
        compute_field_unit_vectors, reflection_coefficient, fibonacci_lattice,\
            cot, cross, sign, rotation_matrix, acos_diff
from sionna.rt.diffraction_funcs import calc_angles, calc_angles_concave, my_wd_compute_fields, transition_func,\
    my_wd_compute_fields2

if TYPE_CHECKING:
    from ..solver_base import SolverBase
    from ..solver_paths import SolverPaths

class WedgeDiffraction:
    def __init__(self, solver: SolverPaths):
        self.solver = solver
        self._scene = solver._scene

        self._dtype = solver._dtype
        self._rdtype = solver._rdtype

    def tracing(self, sources, targets, los_prim, diffraction, edge_diffraction, case='both') -> Tuple[Paths, PathsTmpData]:
        
        diff_paths = Paths(sources=sources, targets=targets, scene=self._scene,
                           types=Paths.DIFFRACTED)
        diff_paths_tmp = PathsTmpData(sources, targets, self._dtype)

        if (not diffraction) or (los_prim is None):
            return diff_paths, diff_paths_tmp
        
        # Get the candidate wedges for diffraction
        # Note: Only one-order diffraction is supported. Therefore, we
        # restrict the candidate wedges to the ones of primitives in
        # line-of-sight with the transmitter
        # candidate_wedges : [num_candidate_wedges], int
        #     Candidate wedges indices
        diff_wedges_indices = self.solver._wedges_from_primitives(los_prim,
                                                        edge_diffraction)

        # Discard concave wedges if needed
        
        # Discard paths for which at least one of the transmitter or
        # receiver is inside the wedge.
        # diff_wedges_indices : [num_targets, num_sources, max_num_paths]
        #   Indices of the intersected wedges
        if not(case is None or case == 'None'):
            diff_wedges_indices = self._discard_obstructing_wedges_and_corners(
                                                diff_wedges_indices, targets,
                                                sources, case)

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

        # Discard obstructed paths
        diff_wedges_indices, diff_vertices =\
            self._check_wedges_visibility(targets, sources,
                                        diff_wedges_indices,
                                        diff_vertices, case)

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
        Compute the interaction points on the wedges that minimizes the path
        length, and masks the wedges for which the interaction points is
        not on the finite wedge.

        Note: This calculation is done in double-precision (64bit).

        Input
        ------
        targets : [num_targets, 3], tf.float
            Coordinates of the targets.

        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        wedges_indices : [num_targets, num_sources, max_num_paths], tf.int
            Indices of the wedges that interacted with the diffracted paths

        Output
        -------
        wedges_indices : [num_targets, num_sources, max_num_paths], tf.int
            Indices of the wedges that interacted with the diffracted paths

        vertices : [num_targets, num_sources, max_num_paths, 3], tf.float
            Coordinates of the interaction points on the intersected wedges
        """

        sources = tf.cast(sources, tf.float64)
        targets = tf.cast(targets, tf.float64)

        # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
        # This makes no difference on the resulting paths as such paths
        # are not flagged as active.
        # [max_num_paths]
        valid_wedges_idx = tf.where(wedges_indices == -1, 0, wedges_indices)

        # [max_num_paths, 3]
        origins = tf.gather(self.solver._wedges_origin, valid_wedges_idx)
        origins = tf.cast(origins, tf.float64)
        # [1, 1, max_num_paths, 3]
        origins = expand_to_rank(origins, 4, 0)
        # [max_num_paths, 3]
        e_hat = tf.gather(self.solver._wedges_e_hat, valid_wedges_idx)
        e_hat = tf.cast(e_hat, tf.float64)
        # [1, 1, max_num_paths, 3]
        e_hat = expand_to_rank(e_hat, 4, 0)
        # [max_num_paths]
        wedges_length = tf.gather(self.solver._wedges_length, valid_wedges_idx)
        wedges_length = tf.cast(wedges_length, tf.float64)
        # [1, 1, max_num_paths]
        wedges_length = expand_to_rank(wedges_length, 3, 0)

        # Expand to broadcast with paths and sources/targets
        # [1, num_sources, 1, 3]
        sources = tf.expand_dims(tf.expand_dims(sources, axis=1), axis=0)
        # [num_targets, 1, 1, 3]
        targets = insert_dims(targets, 2, 1)
        # Sources/Targets vectors
        # [1, num_sources, max_num_paths, 3]
        u_t = origins - sources
        # [num_targets, 1, max_num_paths, 3]
        u_r = origins - targets

        # Quantites required for the computation of the interaction points
        # [1, num_sources, max_num_paths]
        a = dot(u_t,e_hat)
        # [num_targets, 1, max_num_paths]
        b = dot(u_r, e_hat)
        # [1, num_sources, max_num_paths]
        c = dot(u_t, u_t)
        # [num_targets, 1, max_num_paths]
        d = dot(u_r, u_r)

        # Quantites required for the computation of the interaction points
        # [num_targets, num_sources, max_num_paths]
        alpha = -tf.square(a) + tf.square(b) + c - d
        beta = 2.*(a*tf.square(b) - b*tf.square(a) + b*c - a*d)
        gamma = tf.square(b)*c - tf.square(a)*d

        # Normalized quantites to improve numerical preicion, only valid if
        # alpha != 0
        # [num_targets, num_sources, max_num_paths]
        beta_norm = tf.math.divide_no_nan(beta, alpha)
        gamma_norm = tf.math.divide_no_nan(gamma, alpha)
        delta = tf.square(beta_norm) - 4.*gamma_norm

        # Because of numerical imprecision, delta could be slighlty smaller than
        # 0
        # [num_targets, num_sources, max_num_paths]
        delta = tf.where(tf.less(delta, tf.zeros_like(delta)),
                         tf.zeros_like(delta), delta)


        # Four possible outcomes depending on the value of the previous
        # quantities.
        # Values of t that minimizes the path length for each outcome.
        # [num_targets, num_sources, max_num_paths]
        t_min_1 = -a
        t_min_2 = -tf.math.divide_no_nan(gamma, beta)
        t_min_3 = (-beta_norm + tf.sqrt(delta))*0.5
        t_min_4 = (-beta_norm - tf.sqrt(delta))*0.5
        # Condition for each outcome to be selected
        # If a == b and c == d, then set to t_min_1
        # [num_targets, num_sources, max_num_paths]
        cond_1 = tf.logical_and(tf.experimental.numpy.isclose(a, b),
                                tf.experimental.numpy.isclose(c, d))
        # If cond_1 does not hold and alpha == 0, then set to t_min_2
        # [num_targets, num_sources, max_num_paths]
        cond_2 = tf.logical_and(tf.logical_not(cond_1),
                                tf.experimental.numpy.isclose(alpha,
                                tf.zeros_like(alpha)))
        # If neither cond_1 nor cond_2 holds, then set to t_min_3 or t_min_4
        # depending on the signs of t+a and t+b
        # [num_targets, num_sources, max_num_paths]
        not_cond_12 = tf.logical_and(tf.logical_not(cond_1),
                                     tf.logical_not(cond_2))
        # [num_targets, num_sources, max_num_paths]
        t_min_3a = t_min_3 + a
        t_min_3b = t_min_3 + b
        # [num_targets, num_sources, max_num_paths]
        cond_3 = tf.logical_and(not_cond_12,
                tf.less_equal(tf.sign(t_min_3a)*tf.sign(t_min_3b), 0.0))
        # If none of conditions 1, 2, or 3 are satisfied, then all is left is
        # t_min_4
        # [num_targets, num_sources, max_num_paths]
        cond_4 = tf.logical_and(not_cond_12, tf.logical_not(cond_3))
        # Assign t_min according to the previously computed conditions
        # [num_targets, num_sources, max_num_paths]
        t_min = tf.zeros_like(cond_1, tf.float64)
        t_min = tf.where(cond_1, t_min_1, t_min)
        t_min = tf.where(cond_2, t_min_2, t_min)
        t_min = tf.where(cond_3, t_min_3, t_min)
        t_min = tf.where(cond_4, t_min_4, t_min)

        # Mask paths for which the interaction point is not on the finite
        # wedge
        # [num_targets, num_sources, max_num_paths]
        mask_ = tf.logical_and(
            tf.greater_equal(t_min, tf.zeros_like(t_min)),
            tf.less_equal(t_min, wedges_length))
        # [num_targets, num_sources, max_num_paths]
        wedges_indices = tf.where(mask_, wedges_indices, -1)

        # Interaction points
        # Expand to broadcast with coordinates
        # [num_targets, num_sources, max_num_paths, 1]
        t_min = tf.expand_dims(t_min, axis=3)
        # [num_targets, num_sources, max_num_paths, 3]
        inter_point = origins + t_min*e_hat

        # Discard wedges with no valid paths
        # [max_num_paths]
        used_wedges = tf.where(tf.reduce_any(tf.not_equal(wedges_indices, -1),
                                             axis=(0,1)))[:,0]
        # [num_targets, num_sources, max_num_paths, 3]
        inter_point = tf.gather(inter_point, used_wedges, axis=2)
        # [num_targets, num_sources, max_num_paths]
        wedges_indices = tf.gather(wedges_indices, used_wedges, axis=2)

        # Back to the required precision
        inter_point = tf.cast(inter_point, self._rdtype)

        return wedges_indices, inter_point
    
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

    def simple_compute_fields(self, wedge_idxs, s_wedge_prime_hat, s_wedge_hat, s_wedge_prime, s_wedge, mask,
                                ell_ = None, skip_F=False):
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