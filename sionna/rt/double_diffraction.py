
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
from .paths import Paths, invert_s_t
from .objects_geometry import ObjectsGeometry
from .diffraction_funcs import func_G_approximated_capolino, calc_angles, invert_angles,\
    my_wd_compute_fields, transition_func, calc_angles_concave,\
    get_wd_points
from .utils import dot, phi_hat, theta_hat, theta_phi_from_unit_vec,\
    normalize, moller_trumbore, component_transform, mi_to_tf_tensor,\
        compute_field_unit_vectors, reflection_coefficient, fibonacci_lattice,\
            cot, cross, sign, rotation_matrix, acos_diff, rot_mat_from_unit_vecs, \
            r_hat

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

        self.is_wedge_vertex = False
        self.is_vertex_wedge = False
        self.is_eee = False
        self.is_vertex_vertex = False

        self.is_only_in_nlos = True  # for VE, EV combinations 

        self.is_vw_reversed_compute_fields = True
        self.is_add_slope = False
        self.is_only_close_2_transition = True
        self.is_coef_ell_separate_in_vd = True

        self.close_2_transition_eps = 0.5
        r"""More value - less paths"""

        self.max_lim_param_b = 0.1

        self.w = None

        #self.is_only_slope_diffraction = False  # compute all double wedges by default

        # [num_wedges, 3]
        self._wedges_origin = self.solver_paths.wedges_origin
        self._wedges_normals = self.solver_paths.wedges_normals
        self._wedges_e_hat = self.solver_paths.wedges_e_hat
        self._facet_points = self.solver_paths._facet_points
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
        
        
    def _discard_obstructing_wedges2(self, candidate_wedges, points):
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
        # # Expand to broadcast with targets
        # # [1, num_points, num_candidate_wedges]
        # sources_valid_half_space = tf.expand_dims(sources_valid_half_space,
        #                                           axis=0)

        mask = sources_valid_half_space

        # Discard paths with no valid link
        # [max_num_paths]
        valid_paths = tf.where(tf.reduce_any(mask, axis=(0)))[:,0]
        # [num_points, max_num_paths]
        mask = tf.gather(mask, valid_paths, axis=1)
        # [max_num_paths]
        wedges_indices = tf.gather(candidate_wedges, valid_paths, axis=0)
        # Set invalid wedges to -1
        # [num_points, max_num_paths]
        wedges_indices = tf.where(mask, wedges_indices, -1)

        return wedges_indices
    
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
        # # Expand to broadcast with targets
        # # [1, num_points, num_candidate_wedges]
        # sources_valid_half_space = tf.expand_dims(sources_valid_half_space,
        #                                           axis=0)

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

    # def get_coplanar_pairs(self, dd_pairs):
    #     r"""

    #     Inputs
    #     ------
    #     dd_pairs : [num_pairs, 2], int
    #         Indices of the wedge pairs.
    #     """
    #     eps = 1e-3

    #     # [num_sources, num_targets, num_pairs, 3]
    #     e_hat_1 = tf.gather(self._wedges_e_hat, dd_pairs[:, 0], axis=0)
    #     e_hat_2 = tf.gather(self._wedges_e_hat, dd_pairs[:, :, :, 1], axis=0)
    #     w1_point = tf.gather(self._wedges_origin, dd_pairs[:, :, :, 0], axis=0)
    #     w2_point = tf.gather(self._wedges_origin, dd_pairs[:, :, :, 1], axis=0)
    #     l_hat, _ = normalize(w2_point - w1_point)

    #     mask_coplanar = tf.abs(dot(cross(e_hat_1, e_hat_2), l_hat)) < eps

    #     return mask_coplanar
    
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
        #tf.where(tf.reduce_all(dd_pair_idxs == [169, 264], axis=1))

        mask_coplanar = tf.abs(dot(cross(e_hat_1, e_hat_2), l_hat)) < self.coplanar_eps

        return mask_coplanar
    
    def discard_obstructing(self, candidate_pairs, sources, targets):
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

        ## 1) prepare check sources and wedge1
        # [num_sources, num_dd_candidates1]
        _, mask1 = self._discard_obstructing_wedges(candidate_pairs[:, 0], sources)

        ### 2) prepare check targets and wedge2
        # [num_targets, num_dd_candidates2]
        _, mask2 = self._discard_obstructing_wedges(candidate_pairs[:, 1], targets)

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

    def discard_obstructing_dd_wedge_vertex(self, candidate_pairs, sources, targets):
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
        
        ## 1) prepare check points and wedges
        # [num_sources, num_dd_candidates]
        _, mask1 = self._discard_obstructing_wedges(candidate_pairs[:, 0], sources)

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

    def discard_obstructing_dd_vertex_wedge(self, candidate_pairs, sources, targets):
        # TODO: make function from above general
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
        
        ## 1) prepare check points and wedges
        # [num_targets, num_dd_candidates]
        _, mask1 = self._discard_obstructing_wedges(candidate_pairs[:, 1], targets)

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
        
        #x, y = tf.raw_ops.UniqueV2(x=pairs, out_idx=tf.int32, axis=[1])
        
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

        # # [num_targets, num_sources, num_wedges]
        # wedge_idxs_t = tf.expand_dims(wedge_idxs_t, axis=1)
        # wedge_idxs_t = tf.repeat(wedge_idxs_t, tf.shape(wedge_idxs_s)[0], axis=1)

        # # [num_targets, num_sources, num_wedges]
        # wedge_idxs_s = tf.expand_dims(wedge_idxs_s, axis=0)
        # wedge_idxs_s = tf.repeat(wedge_idxs_s, tf.shape(wedge_idxs_t)[0], axis=0)


        # wedge_idxs_t = tf.reshape(wedge_idxs_t, (num_targets*num_sources*num_wedges1, -1))
        # wedge_idxs_s = tf.reshape(wedge_idxs_s, (num_targets*num_sources*num_wedges2, -1))

        # [num_targets, num_sources, num_wedges1, num_wedges2, 2]
        #pairs = tf.stack(tf.meshgrid(wedge_idxs_t, wedge_idxs_s), axis=-1)

        wedge_idxs_t = tf.reshape(wedge_idxs_t, (num_targets*num_wedges1, -1))
        wedge_idxs_s = tf.reshape(wedge_idxs_s, (num_sources*num_wedges2, -1))

        # [num_targets*num_wedges1, num_sources*num_wedges2, 2]
        pairs = tf.stack(tf.meshgrid(wedge_idxs_t, wedge_idxs_s), axis=-1)

        # [num_pairs, 2]
        pairs = tf.reshape(pairs, (-1, 2))

        # TODO
        
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
        #is_parallel = False
        #cond_is_parallel = tf.zeros((num_sources, num_targets, num_pairs), dtype=tf.bool)

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
        #l_dir = tf.sqrt(1.0 - n_dir**2)  # TODO
        # edge1_hat_local = tf.constant([1.0, 0.0, 0.0], dtype=self._rdtype)
        # edge2_hat_local = tf.constant([l_dir, 0.0, n_dir], dtype=self._rdtype)
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
        #local_R2 = - tf.norm(tmp_vec, axis=-1)

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

        cos_beta1_prime = dot(edge1_hat, w1_point - sources_ext) / tf.norm(w1_point - sources_ext, axis=-1)
        cos_beta1 = dot(edge1_hat, w2_point - w1_point) / tf.norm(w2_point - w1_point, axis=-1)
        cos_beta2_prime = dot(edge2_hat, w2_point - w1_point) / tf.norm(w2_point - w1_point, axis=-1)
        cos_beta2 = dot(edge2_hat, targets_ext - w2_point) / tf.norm(targets_ext - w2_point, axis=-1)

        # active1 = tf.experimental.numpy.isclose(cos_beta1_prime, cos_beta1, rtol=1e-2, atol=1e-3)
        # active2 = tf.experimental.numpy.isclose(cos_beta2_prime, cos_beta2, rtol=1e-2, atol=1e-3)
        # active = tf.logical_and(active1, active2)
        # #return tf.stack([w1_point, w2_point], axis=-2), active

        return tf.stack([w1_point, w2_point], axis=-2), new_active

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
        num_sources, num_targets = tf.shape(sources)[0], tf.shape(targets)[0]
        num_pairs = tf.shape(pair_idxs)[2]
        # [num_sources, num_targets, num_pairs]
    
        # Change -1 by 0
        pair_idxs = tf.where(pair_idxs == -1, 0, pair_idxs)
        wedge_idxs = pair_idxs[:, :, :, 0]
        vertex_idxs = pair_idxs[:, :, :, 1]

        # [num_sources, num_targets, num_pairs, 3]
        # w_start_points = tf.gather(self._wedges_origin, wedge_idxs, axis=0)
        # edge_hat = tf.gather(self._wedges_e_hat, vertex_idxs, axis=0)
        #vec_w1_w2 = w2_start_points - w1_start_points

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
        num_sources, num_targets = tf.shape(sources)[0], tf.shape(targets)[0]
        num_pairs = tf.shape(pair_idxs)[2]
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
        t_min = tf.zeros_like(cond_1, tf.float64)
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

        # # Discard wedges with no valid paths
        # # [max_num_paths]
        # used_wedges = tf.where(tf.reduce_any(tf.not_equal(new_wedge_idxs, -1),
        #                                      axis=(0,1)))[:,0]
        # # [num_sources, num_targets, max_num_pairs, 3]
        # inter_point = tf.gather(inter_point, used_wedges, axis=2)
        # # [num_sources, num_targets, max_num_pairs]
        # new_wedge_idxs = tf.gather(new_wedge_idxs, used_wedges, axis=2)

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

        mask = tf.logical_and(mask, mask_same_objs)

        return mask
    
    def _old_get_slopes(self, dd_pairs):
        r"""
        Inputs
        ------

        dd_pairs : [num_pairs, 2], int
            The indices of the diffraction pairs (wedge1, wedge2)
        """

        def foo(a, b, atol=1e-3, rtol=1e-3):
            mask = tf.abs(a - b) <= (atol + rtol * tf.abs(b))
            return tf.reduce_all(mask, axis=-1)

        # [num_pairs]
        idxs1 = dd_pairs[:, 0]
        idxs2 = dd_pairs[:, 1]

        # [num_pairs, 2, 3]
        w1_normals = tf.gather(self._wedges_normals, idxs1, axis=0)
        w2_normals = tf.gather(self._wedges_normals, idxs2, axis=0)

        # [num_pairs]
        mask1 = foo(w1_normals[:, 0], w2_normals[:, 0])
        mask2 = foo(w1_normals[:, 0], w2_normals[:, 1])
        mask3 = foo(w1_normals[:, 1], w2_normals[:, 0])
        mask4 = foo(w1_normals[:, 1], w2_normals[:, 1])

        mask = tf.logical_or(
            tf.logical_or(mask1, mask2),
            tf.logical_or(mask3, mask4)
        )

        return mask
        
    def check_preliminary_visibility(self, dd_pairs):
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

        return tf.boolean_mask(dd_pairs, valid, axis=0)

    def _check_visibility(self, sources, targets, diff_points, dd_pair_idxs):
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

    def check_visibility_mixed(self, sources, targets, diff_points, w_idxs, v_idxs):
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
        
        # w2_normals = tf.gather(self._wedges_normals, dd_pair_idxs_valid[:, :, :, 1], axis=0)
        # TODO: does this work for concave wedges?
        shift_1 = self.solver_paths.DD_EPSILON_OBSTRUCTION * (w1_normals[:,:,:,0] + w1_normals[:,:,:,1]) / 3.0
        shift_2 = self.solver_paths.DD_EPSILON_OBSTRUCTION * (w2_normals[:,:,:,0] + w2_normals[:,:,:,1]) / 3.0
        shift_1 = tf.reshape(shift_1, (-1, 3))
        shift_2 = tf.reshape(shift_2, (-1, 3))

        d, maxt = tf.linalg.normalize((diff_points2 + shift_2) - (diff_points1 + shift_1), axis=1)
        maxt = tf.squeeze(maxt, axis=1)
        valid_w2w = tf.logical_not(self.solver_paths._test_obstruction_2(diff_points1+shift_1, d, maxt))

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
        #l1_hat, l2_hat = l_hat, l_hat

        # TODO: when diff_points[0, :, :, :] == diff_points[1, :, :, :]
        # l1_hat <-- r2_hat, l2_hat <-- r1_hat
        eps = 1e-3
        are_close = tf.norm(diff_points[0, :, :, :] - diff_points[1, :, :, :], axis=-1) < eps
        # l1_hat = tf.where(are_close[..., None], r2_hat, l1_hat)
        # l2_hat = tf.where(are_close[..., None], r1_hat, l2_hat)
        
        # [num_sources, num_targets, num_pairs]
        source_close = dot(r1_hat, l_hat) > 1 - self.close_2_transition_eps
        target_close = dot(l_hat, r2_hat) > 1 - self.close_2_transition_eps

        res = tf.logical_and(source_close, target_close)
        res = tf.logical_or(res, are_close)

        return res
    
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

        # # Mask of valid paths
        # # [num_targets, num_sources, max_num_paths]
        # mask = tf.fill([num_targets, num_sources, max_num_paths], False)
        # mask = tf.tensor_scatter_nd_update(mask, scatter_indices,
        #         tf.fill([tf.shape(scatter_indices)[0]], True))

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
                                paths_tmp: PathsTmpData):
        
        mask_wedge_wedge = tf.reduce_all(tf.reduce_all(paths.dd_type == 0, axis=1), axis=0)
        mask_wedge_vertex = tf.reduce_all(tf.reduce_all(paths.dd_type == 1, axis=1), axis=0)
        mask_vertex_wedge = tf.reduce_all(tf.reduce_all(paths.dd_type == 2, axis=1), axis=0)
        mask_vertex_vertex = tf.reduce_all(tf.reduce_all(paths.dd_type == 3, axis=1), axis=0)

        # [max_num_paths, num_sources, num_targets, 2, 3]
        mat_t = tf.zeros((paths.vertices[0].shape[2], paths.targets.shape[0], paths.sources.shape[0], 2, 2), dtype=self._dtype)

        ### 1) wedge-wedge
        paths_wedge_wedge = paths.gather_paths(tf.where(mask_wedge_wedge)[:, 0], is_finalized=False)
        paths_tmp_wedge_wedge = paths_tmp.gather_paths(tf.where(mask_wedge_wedge)[:, 0])
        if tf.reduce_any(mask_wedge_wedge):
            mat_t_wedge_wedge = self.compute_fields_wedge_wedge(relative_permittivity, scattering_coefficient, 
                                                                paths_wedge_wedge, paths_tmp_wedge_wedge)

        ### 2) wedge-vertex
        paths_wedge_vertex = paths.gather_paths(tf.where(mask_wedge_vertex)[:, 0], is_finalized=False)
        paths_tmp_wedge_vertex = paths_tmp.gather_paths(tf.where(mask_wedge_vertex)[:, 0])
        if tf.reduce_any(mask_wedge_vertex):
            mat_t_wedge_vertex = self.compute_fields_wedge_vertex(relative_permittivity, scattering_coefficient,
                                                                   paths_wedge_vertex, paths_tmp_wedge_vertex)
            
        ### 3) vertex-wedge
        paths_vw = paths.gather_paths(tf.where(mask_vertex_wedge)[:, 0], is_finalized=False)
        paths_tmp_vw = paths_tmp.gather_paths(tf.where(mask_vertex_wedge)[:, 0])
        if tf.reduce_any(mask_vertex_wedge):
            if self.is_vw_reversed_compute_fields:
                _paths_vw, _paths_tmp_vw = invert_s_t(paths_vw, paths_tmp_vw)
                _mat_t_vw = self.compute_fields_wedge_vertex(relative_permittivity, scattering_coefficient,
                                                                   _paths_vw, _paths_tmp_vw)
                mat_t_vw = tf.transpose(_mat_t_vw, perm=[1, 0, 2, 3, 4])
            else:
                mat_t_vw = self.compute_fields_vertex_wedge(relative_permittivity, scattering_coefficient,
                                                                    paths_vw, paths_tmp_vw)
                
        ### 4) vertex-vertex
        # paths_vv = paths.gather_paths(tf.where(mask_vertex_vertex)[:, 0], is_finalized=False)
        # paths_tmp_vv = paths_tmp.gather_paths(tf.where(mask_vertex_vertex)[:, 0])
        # if tf.reduce_any(mask_vertex_vertex):
        #     pass

        if tf.reduce_any(mask_wedge_wedge):
            mat_t = tf.tensor_scatter_nd_update(mat_t, tf.where(mask_wedge_wedge), tf.transpose(mat_t_wedge_wedge, perm=[2, 0, 1, 3, 4]))
        if tf.reduce_any(mask_wedge_vertex):
            mat_t = tf.tensor_scatter_nd_update(mat_t, tf.where(mask_wedge_vertex), tf.transpose(mat_t_wedge_vertex, perm=[2, 0, 1, 3, 4]))
        if tf.reduce_any(mask_vertex_wedge):
            mat_t = tf.tensor_scatter_nd_update(mat_t, tf.where(mask_vertex_wedge), tf.transpose(mat_t_vw, perm=[2, 0, 1, 3, 4]))

        mat_t = tf.transpose(mat_t, perm=[1, 2, 0, 3, 4])

        return mat_t

    def compute_fields_wedge_vertex(self, relative_permittivity,
                                    scattering_coefficient, 
                                    paths: Paths, 
                                    paths_tmp: PathsTmpData):
        mask = paths.mask
        targets = paths.targets
        sources = paths.sources

        num_targets = targets.shape[0]
        num_sources = sources.shape[0]
        #num_vertices = paths.vertices[0].shape[2]

        wavelength = self.solver_paths._scene.wavelength
        k = 2.0*PI/wavelength

        # Reshape sources and targets
        # [1, num_sources, 1, 3]
        sources = tf.reshape(sources, [1, -1, 1, 3])
        # [num_targets, 1, 1, 3]
        targets = tf.reshape(targets, [-1, 1, 1, 3])
        num_paths = tf.shape(paths.vertices)[3]

        # sources -> wedge_diff_points -> vertex_diff_points -> targets
        # [num_targets, num_sources, max_num_paths, 3]
        wedge_diff_points = tf.gather(paths.vertices, 0, axis=0)
        vertex_diff_points = tf.gather(paths.vertices, 1, axis=0)

        r1_hat, r1 = normalize(wedge_diff_points - sources)
        r2_hat, r2 = normalize(targets - vertex_diff_points)
        l_hat, l = normalize(vertex_diff_points - wedge_diff_points)   

        # [num_targets, num_sources, max_num_paths]
        wedge_idxs = tf.gather(paths.objects, 0, axis=0)
        vertex_idxs = tf.gather(paths.objects, 1, axis=0)

        ###### 0.1) wedge diffraction
        # [num_targets, num_sources, max_num_paths]
        valid_wedges_idx = tf.where(wedge_idxs == -1, 0, wedge_idxs)

        ####### 0.2) vertex diffraction
        valid_vertex_idxs = tf.where(vertex_idxs == -1, 0, vertex_idxs)
        vertex_indices_mask = vertex_idxs != -1

        #[max_num_paths, max_num_wedges_per_vertex]
        non_valid_tensor_vertices_2_wedges = self.solver_paths.vertex_diffraction.vertices_2_wedges.to_tensor(default_value=-1)
        tensor_vertices_2_wedges = self.solver_paths.vertex_diffraction.vertices_2_wedges.to_tensor(default_value=0)

        #[num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex]
        mask_wedges_in_vertex_ = tf.gather(non_valid_tensor_vertices_2_wedges, valid_vertex_idxs)
        mask_wedges_in_vertex_ = tf.not_equal(mask_wedges_in_vertex_, -1)
        mask_wedges_in_vertex_ = tf.logical_and(mask_wedges_in_vertex_, tf.expand_dims(vertex_indices_mask, -1))

        #[num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex]
        tensor_valid_vertices_2_wedges = tf.gather(tensor_vertices_2_wedges, valid_vertex_idxs)

        # [num_targets, num_sources, max_num_paths_extended, 3]
        vertex_diff_points_ext = tf.repeat(vertex_diff_points, tensor_valid_vertices_2_wedges.shape[3], axis=2)

        #[num_targets, num_sources, max_num_paths_extended]
        valid_vertex_wedges_idx = tf.reshape(tensor_valid_vertices_2_wedges, [num_targets, num_sources, -1])

        #### a, b)
        eps = 1e-3
        are_close = tf.norm(wedge_diff_points - vertex_diff_points, axis=-1) < eps
        
        valid_wedges_idx_ext = tf.repeat(valid_wedges_idx, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        a1 = tf.gather(self.obj_geom.wedge_2_facets, valid_vertex_wedges_idx, axis=0)
        a2 = tf.gather(self.obj_geom.wedge_2_facets, valid_wedges_idx_ext, axis=0)
        same_facet = (a1 == a2)
        non_same_edge = valid_wedges_idx_ext != valid_vertex_wedges_idx
        same_facet2 = tf.logical_and(same_facet, non_same_edge[..., None])
        facet_idx = tf.where(same_facet2[..., 0], a1[..., 0], a1[..., 1])
        facet_normal = tf.gather(self.solver_paths._normals, facet_idx, axis=0)
        _facet_normal = tf.reshape(facet_normal, [num_targets, num_sources, num_paths, -1, 3])
        are_close_ext = tf.repeat(are_close, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        #tmp_hat_are_close = tf.broadcast_to(tf.convert_to_tensor([0.0, -1.0, 0.0], dtype=self._rdtype), tf.shape(r1_hat))
        _tmp_hat_are_close, _ = normalize(r1_hat + r2_hat) # TODO

        a3 = tf.abs(dot(_tmp_hat_are_close[..., None, :], _facet_normal))
        a4_idxs = tf.argmin(a3, axis=-1)
        # project tmp_hat_are_close onto chosen_normals
        chosen_normals = tf.gather(_facet_normal, a4_idxs, batch_dims=3, axis=-2)
        tmp_hat_are_close = _tmp_hat_are_close - dot(_tmp_hat_are_close, chosen_normals)[..., None] * chosen_normals
        tmp_hat_are_close, _ = normalize(tmp_hat_are_close)
        #(dot(tmp_hat_are_close, chosen_normals))

        eps_l = wavelength/5
        l_hat = tf.where(are_close[..., None], tmp_hat_are_close, l_hat)
        l = tf.where(are_close, eps_l * tf.ones_like(l), l)

        ####### 1) wedge diffraction
        coef_ell = (r1 + l) * (r2 + l) / (l * (r1 + r2))
        
        mat_t_wedge, new_coef_ell, _, (e_hat1, w_beta_prime) = self.solver_paths.solver_wedge_diffraction.simple_compute_fields(wedge_idxs, 
                              r1_hat, l_hat, r1, l, mask,
                              coef_ell=coef_ell, new_coef_ell=None)

        # ####### 2) vertex diffraction
        # [num_targets, num_sources, max_num_paths_extended, 2, 3]
        normals = tf.gather(self._wedges_normals, valid_vertex_wedges_idx, axis=0)

        # Compute the wedges angle
        # [num_targets, num_sources, max_num_paths]
        cos_wedges_angle = dot(normals[...,0,:],normals[...,1,:], clip=True)
        wedges_angle = PI - tf.math.acos(cos_wedges_angle)
        _v_wedge_n = (2.*PI-wedges_angle)/PI

        is_concave_wedge2 = tf.gather(self.solver_paths._is_concave_wedge, valid_vertex_wedges_idx)
        v_wedge_n = tf.where(is_concave_wedge2, wedges_angle / PI, _v_wedge_n)

        # [num_targets, num_sources, max_num_paths_extended, 3]
        e_hat2_ext = tf.gather(self._wedges_e_hat, valid_vertex_wedges_idx)
        wedge_origins_tmp = tf.gather(self._wedges_origin, valid_vertex_wedges_idx)

        # [num_targets, num_sources, max_num_paths_extended]
        # needs_swap = tf.logical_not(tf.reduce_all(tf.math.equal(wedge_origins_tmp, vertex_diff_points_ext), axis=3))
        needs_swap = tf.logical_not(tf.reduce_all(tf.experimental.numpy.isclose(wedge_origins_tmp, vertex_diff_points_ext), axis=3))
        # [num_targets, num_sources, max_num_paths_extended, 3]
        needs_swap = tf.expand_dims(needs_swap, axis=3)
        e_hat2_ext = tf.where(needs_swap, -e_hat2_ext, e_hat2_ext)

        n_0_hat = tf.where(needs_swap, normals[...,1,:], normals[...,0,:]) 
        n_n_hat = tf.where(needs_swap, normals[...,0,:], normals[...,1,:])

        v_s_prime_hat = tf.repeat(l_hat, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        v_s_prime = tf.repeat(l, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        v_s_hat = tf.repeat(r2_hat, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        v_s = tf.repeat(r2, tensor_valid_vertices_2_wedges.shape[3], axis=2)

        tmp_hat_are_close_ext = tf.repeat(tmp_hat_are_close, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        v_s_prime_hat = tf.where(are_close_ext[..., None], tmp_hat_are_close_ext, v_s_prime_hat)
        v_s_prime = tf.where(are_close_ext, eps_l * tf.ones_like(v_s_prime), v_s_prime)

        v_phi_prime, v_phi, v_beta_prime, v_beta = calc_angles(n_0_hat, e_hat2_ext, v_s_prime_hat, v_s_hat)

        is_concave2 = v_wedge_n < 1.0
        v_phi_prime_, v_phi_, _, _ = calc_angles_concave(n_0_hat, e_hat2_ext, v_s_prime_hat, v_s_hat, v_wedge_n) # concave
        v_phi_prime = tf.where(is_concave2, v_phi_prime_, v_phi_prime)
        v_phi = tf.where(is_concave2, v_phi_, v_phi)

        v_phi_prime = tf.where(tf.experimental.numpy.isclose(v_phi_prime, 2*PI, atol=1e-2), 1e-3*tf.ones_like(v_phi_prime), v_phi_prime)

        ext_new_coef_ell = tf.repeat(new_coef_ell, tensor_valid_vertices_2_wedges.shape[3], axis=2)

        # eps_phi = 1e-5  # TODO

        mat_t_vertex, tmp_v = self.solver_paths.vertex_diffraction.simple_compute_fields(v_phi_prime, v_phi, v_beta_prime, v_beta, 
                                    v_s_prime, v_s, v_wedge_n, mask_wedges_in_vertex_, new_coef_ell=ext_new_coef_ell,
                                    return_more=True)

        mat_t_wedge_ext = tf.repeat(mat_t_wedge, tensor_valid_vertices_2_wedges.shape[3], axis=2)

        mat_t_ext = tf.multiply(mat_t_wedge_ext, mat_t_vertex) / 2.0 # grazing incidence

        s_prime_hat_ext = tf.repeat(r1_hat, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        e_hat1_ext = tf.repeat(e_hat1, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        s_hat_ext = v_s_hat

        w_beta_prime_ext = tf.repeat(w_beta_prime, tensor_valid_vertices_2_wedges.shape[3], axis=2)

        cos_ksi = -tf.sign(dot(e_hat1_ext, e_hat2_ext))

        # take only coplanar edge pairs
        l_hat_ext = v_s_prime_hat
        mask_coplanar = tf.abs(dot(cross(e_hat1_ext, e_hat2_ext), l_hat_ext)) < self.coplanar_eps
        cos_ksi = tf.where(mask_coplanar, cos_ksi, tf.zeros_like(cos_ksi))

        # if edge pair is from two identical edges
        # then we set amplitude to zero
        valid_wedges_idx_ext = tf.repeat(valid_wedges_idx, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        same_edge = valid_wedges_idx_ext == valid_vertex_wedges_idx
        cos_ksi = tf.where(same_edge, tf.zeros_like(cos_ksi), cos_ksi)

        mat_t_ext = mat_t_ext * tf.complex(cos_ksi, tf.zeros_like(cos_ksi))[..., None, None]

        phi1_prime_hat, _ = normalize(cross(s_prime_hat_ext, e_hat1_ext))
        beta1_prime_hat = cross(phi1_prime_hat, s_prime_hat_ext)
        
        phi2_hat, _ = normalize(-cross(s_hat_ext, e_hat2_ext))
        beta2_hat = cross(phi2_hat, s_hat_ext)

        theta_t_ext = tf.repeat(paths.theta_t, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        phi_t_ext = tf.repeat(paths.phi_t, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        theta_r_ext = tf.repeat(paths.theta_r, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        phi_r_ext = tf.repeat(paths.phi_r, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        mask_ext = tf.repeat(mask, tensor_valid_vertices_2_wedges.shape[3], axis=2)

        mat_t_ext = self.solver_paths.convert_mat_t(mat_t_ext, mask_ext, theta_t_ext, phi_t_ext, theta_r_ext, phi_r_ext, 
                                                      phi1_prime_hat, beta1_prime_hat, phi2_hat, beta2_hat)

        new_shape = tensor_valid_vertices_2_wedges.shape.as_list() + [2, 2]
        mat_t_ext = tf.reshape(mat_t_ext, new_shape)

        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = tf.reduce_sum(mat_t_ext, axis=3)
        mat_t = mat_t / tf.complex(r1, tf.zeros_like(r1))[..., None, None]  # incidence

        return mat_t
    
    def compute_fields_vertex_wedge(self, relative_permittivity,
                                    scattering_coefficient, 
                                    paths: Paths, 
                                    paths_tmp: PathsTmpData):
        mask = paths.mask
        targets = paths.targets
        sources = paths.sources

        num_targets = targets.shape[0]
        num_sources = sources.shape[0]

        wavelength = self.solver_paths._scene.wavelength
        k = 2.0*PI/wavelength

        # sources -> vertex_diff_points -> wedge_diff_points -> targets
        # [num_targets, num_sources, max_num_paths, 3]
        vertex_diff_points = tf.gather(paths.vertices, 0, axis=0)
        wedge_diff_points = tf.gather(paths.vertices, 1, axis=0)
        # [num_targets, num_sources, max_num_paths]
        vertex_idxs = tf.gather(paths.objects, 0, axis=0)
        wedge_idxs = tf.gather(paths.objects, 1, axis=0)

        # Reshape sources and targets
        # [1, num_sources, 1, 3]
        sources = tf.reshape(sources, [1, -1, 1, 3])
        # [num_targets, 1, 1, 3]
        targets = tf.reshape(targets, [-1, 1, 1, 3])

        _, r1 = normalize(vertex_diff_points - sources)
        _, r2 = normalize(targets - wedge_diff_points)
        _, l = normalize(wedge_diff_points - vertex_diff_points)

        ####### 1) vertex diffraction
        valid_vertex_idxs = tf.where(vertex_idxs == -1, 0, vertex_idxs)
        vertex_indices_mask = vertex_idxs != -1

        #[max_num_paths, max_num_wedges_per_vertex]
        non_valid_tensor_vertices_2_wedges = self.solver_paths.vertex_diffraction.vertices_2_wedges.to_tensor(default_value=-1)
        tensor_vertices_2_wedges = self.solver_paths.vertex_diffraction.vertices_2_wedges.to_tensor(default_value=0)

        #[num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex]
        mask_wedges_in_vertex_ = tf.gather(non_valid_tensor_vertices_2_wedges, valid_vertex_idxs)
        mask_wedges_in_vertex_ = tf.not_equal(mask_wedges_in_vertex_, -1)
        mask_wedges_in_vertex_ = tf.logical_and(mask_wedges_in_vertex_, tf.expand_dims(vertex_indices_mask, -1))

        # #[num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex, 2, 2]
        # mask_wedges_in_vertex = tf.stack([mask_wedges_in_vertex_, mask_wedges_in_vertex_], axis=-1)
        # mask_wedges_in_vertex = tf.stack([mask_wedges_in_vertex, mask_wedges_in_vertex], axis=-1)

        #[num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex]
        tensor_valid_vertices_2_wedges = tf.gather(tensor_vertices_2_wedges, valid_vertex_idxs)

        # [num_targets, num_sources, max_num_paths_extended, 3]
        vertex_diff_points_ext = tf.repeat(vertex_diff_points, tensor_valid_vertices_2_wedges.shape[3], axis=2)

        #[num_targets, num_sources, max_num_paths_extended]
        valid_vertex_wedges_idx = tf.reshape(tensor_valid_vertices_2_wedges, [num_targets, num_sources, -1])

        # Normals
        # [num_targets, num_sources, max_num_paths_extended, 2, 3]
        normals = tf.gather(self._wedges_normals, valid_vertex_wedges_idx, axis=0)

        # Compute the wedges angle
        # [num_targets, num_sources, max_num_paths]
        cos_wedges_angle = dot(normals[...,0,:],normals[...,1,:], clip=True)
        wedges_angle = PI - tf.math.acos(cos_wedges_angle)
        v_wedge_n = (2.*PI-wedges_angle)/PI

        # [num_targets, num_sources, max_num_paths_extended, 3]
        e_hat1_ext = tf.gather(self._wedges_e_hat, valid_vertex_wedges_idx)
        wedge_origins_tmp = tf.gather(self._wedges_origin, valid_vertex_wedges_idx)

        # [num_targets, num_sources, max_num_paths_extended]
        needs_swap = tf.logical_not(tf.reduce_all(tf.math.equal(wedge_origins_tmp, vertex_diff_points_ext), axis=3))
        # [num_targets, num_sources, max_num_paths_extended, 3]
        needs_swap = tf.expand_dims(needs_swap, axis=3)
        e_hat1_ext = tf.where(needs_swap, -e_hat1_ext, e_hat1_ext)

        n_0_hat = tf.where(needs_swap, normals[...,1,:], normals[...,0,:])  # TODO
        n_n_hat = tf.where(needs_swap, normals[...,0,:], normals[...,1,:])

        wedge_diff_points_ext = tf.repeat(wedge_diff_points, tensor_valid_vertices_2_wedges.shape[3], axis=2)

        # [num_targets, num_sources, max_num_paths_ext, 3] and [num_targets, num_sources, max_num_paths_ext]
        v_s_prime_hat, v_s_prime = normalize(vertex_diff_points_ext - sources)
        v_s_hat, v_s = normalize(wedge_diff_points_ext - vertex_diff_points_ext)

        v_phi_prime, v_phi, v_beta_prime, v_beta = calc_angles(n_0_hat, e_hat1_ext, v_s_prime_hat, v_s_hat)
        v_phi = tf.where(tf.experimental.numpy.isclose(v_phi, 2*PI, atol=1e-2), 1e-3*tf.ones_like(v_phi), v_phi)

        r2_ext = tf.repeat(r2, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        # coef_ell = (v_s_prime + v_s) * (r2_ext + v_s) / (v_s * (v_s_prime + r2_ext))
        coef_ell = (r1 + l) * (r2 + l) / (l * (r1 + r2))

        # _new_coef_ell : [num_targets, num_sources, num_paths, num_wedges_per_vertex]
        mat_t_vertex, _new_coef_ell = self.solver_paths.vertex_diffraction.simple_compute_fields(v_phi_prime, v_phi, v_beta_prime, v_beta, 
                                    v_s_prime, v_s, v_wedge_n, mask_wedges_in_vertex_, coef_ell=coef_ell)   

        ###### 2) wedge diffraction
        # [num_targets, num_sources, max_num_paths]
        valid_wedges_idx = tf.where(wedge_idxs == -1, 0, wedge_idxs)

        # [num_targets, num_sources, max_num_paths, 2, 3]
        normals = tf.gather(self._wedges_normals, valid_wedges_idx, axis=0)

        # Compute the wedges angle
        # [num_targets, num_sources, max_num_paths]
        cos_wedges_angle = dot(normals[...,0,:],normals[...,1,:], clip=True)
        wedges_angle = PI - tf.math.acos(cos_wedges_angle)
        w_n = (2.*PI-wedges_angle)/PI

        # [num_targets, num_sources, max_num_paths, 3]
        e_hat2 = tf.gather(self._wedges_e_hat, valid_wedges_idx)

        # Extract surface normals
        # [num_targets, num_sources, max_num_paths, 3]
        n_0_hat = normals[...,0,:]
        # [num_targets, num_sources, max_num_paths, 3]
        n_n_hat = normals[...,1,:]

        # Relative permitivities and scattering coefficients
        # If a callable is defined to compute the radio material properties,
        # it is invoked. Otherwise, the radio materials of objects are used.
        rm_callable = self.solver_paths._scene.radio_material_callable
        # [num_targets, num_sources, max_num_paths, 2]
        objects_indices = tf.gather(self._wedges_objects, valid_wedges_idx,
                                    axis=0)
        if rm_callable is None:
            # [num_targets, num_sources, max_num_paths, 2]
            etas = tf.gather(relative_permittivity, objects_indices)
            scattering_coefficient = tf.gather(scattering_coefficient,
                                               objects_indices)
        else:
            # Harmonize the shapes of the radio material callables
            # [num_targets, num_sources, max_num_paths, 2, 3]
            wedge_diff_points_ = tf.tile(tf.expand_dims(wedge_diff_points, axis=-2),
                                   [1, 1, 1, 2, 1])
            # scattering_coefficient, etas : [num_targets, num_sources,
            #   max_num_paths, 2]
            etas, scattering_coefficient, _   = rm_callable(objects_indices,
                                                            wedge_diff_points_)

        # [num_targets, num_sources, max_num_paths, 3] or [num_targets, num_sources, max_num_paths]
        _, v_s_prime_non_ext = normalize(vertex_diff_points - sources)
        w_s_prime_hat, w_s_prime = normalize(wedge_diff_points - vertex_diff_points)
        w_s_hat, w_s = normalize(targets - wedge_diff_points)

        w_phi_prime, w_phi, w_beta_prime, _ = calc_angles(n_0_hat, e_hat2, w_s_prime_hat, w_s_hat)

        w_phi_prime = tf.where(tf.experimental.numpy.isclose(w_phi_prime, 2*PI, atol=1e-2), 1e-3*tf.ones_like(w_phi_prime), w_phi_prime)

        ell2 = w_s_prime * w_s / (w_s_prime + w_s) * tf.sin(w_beta_prime)**2

        # [num_targets, num_sources, num_paths, num_wedges_per_vertex]
        mat_t_wedge, _, _ = my_wd_compute_fields(w_phi_prime, w_phi, w_beta_prime, ell2, w_s_prime + v_s_prime_non_ext, 
                                           w_s, w_n, mask, self.solver_paths._scene, new_coef_ell=_new_coef_ell)
        
        mat_t_wedge_ext = tf.reshape(mat_t_wedge, mat_t_vertex.shape)

        mat_t_ext = tf.multiply(mat_t_wedge_ext, mat_t_vertex) / 2.0 # grazing incidence

        w_s_prime_hat_ext = tf.repeat(w_s_prime_hat, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        w_s_hat_ext = tf.repeat(w_s_hat, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        e_hat2_ext = tf.repeat(e_hat2, tensor_valid_vertices_2_wedges.shape[3], axis=2)

        l_hat_ext = w_s_prime_hat_ext
        cos_ksi = -tf.sign(dot(e_hat1_ext, e_hat2_ext))
        mask_coplanar = tf.abs(dot(cross(e_hat1_ext, e_hat2_ext), l_hat_ext)) < self.coplanar_eps
        cos_ksi = tf.where(mask_coplanar, cos_ksi, tf.zeros_like(cos_ksi))

        mat_t_ext = mat_t_ext * tf.complex(cos_ksi, tf.zeros_like(cos_ksi))[..., None, None]

        # ### 1)
        phi1_prime_hat, _ = normalize(cross(v_s_prime_hat, e_hat1_ext))
        beta1_prime_hat = cross(phi1_prime_hat, v_s_prime_hat)
        
        phi2_hat, _ = normalize(-cross(w_s_hat_ext, e_hat2_ext))
        beta2_hat = cross(phi2_hat, w_s_hat_ext)
        
        ### 2)
        theta_t_ext = tf.repeat(paths.theta_t, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        phi_t_ext = tf.repeat(paths.phi_t, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        theta_r_ext = tf.repeat(paths.theta_r, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        phi_r_ext = tf.repeat(paths.phi_r, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        mask_ext = tf.repeat(mask, tensor_valid_vertices_2_wedges.shape[3], axis=2)

        mat_t_ext = self.solver_paths.convert_mat_t(mat_t_ext, mask_ext, theta_t_ext, phi_t_ext, theta_r_ext, phi_r_ext, 
                                                      phi1_prime_hat, beta1_prime_hat, phi2_hat, beta2_hat)

        new_shape = tensor_valid_vertices_2_wedges.shape.as_list() + [2, 2]
        mat_t_ext = tf.reshape(mat_t_ext, new_shape)

        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = tf.reduce_sum(mat_t_ext, axis=3)
        mat_t = mat_t / tf.complex(r1, tf.zeros_like(r1))[..., None, None]  # incidence

        return mat_t

    def compute_fields_wedge_wedge(self, relative_permittivity,
                                scattering_coefficient, 
                                paths: Paths, 
                                paths_tmp: PathsTmpData):
        mask = paths.mask
        targets = paths.targets
        sources = paths.sources
        theta_t = paths.theta_t
        phi_t = paths.phi_t
        theta_r = paths.theta_r
        phi_r = paths.phi_r

        num_targets = targets.shape[0]
        num_sources = sources.shape[0]
        num_vertices = paths.vertices[0].shape[2]

        #normals = paths_tmp.normals

        wavelength = self.solver_paths._scene.wavelength
        k = 2.0*PI/wavelength

        # sources -> diff_points1 -> diff_points2 -> targets
        # [2, num_targets, num_sources, max_num_paths, 3]
        diff_points = paths.vertices
        # [2, num_targets, num_sources, max_num_paths]
        wedge_idxs = paths.objects

        # [num_targets, num_sources, max_num_paths]
        valid = tf.reduce_all(wedge_idxs != -1, axis=0)
        # [2, num_targets, num_sources, max_num_paths]
        valid_wedge_idxs = tf.where(valid, wedge_idxs, 0)

        # Normals
        # [2, num_targets, num_sources, max_num_paths, 2, 3]
        normals = tf.gather(self._wedges_normals, valid_wedge_idxs, axis=0)

        # Compute the wedges angle
        # [2, num_targets, num_sources, max_num_paths]
        cos_wedges_angle = dot(normals[...,0,:],normals[...,1,:], clip=True)
        wedges_angle = PI - tf.math.acos(cos_wedges_angle)
        _n = (2.*PI-wedges_angle)/PI

        is_concave_wedge = tf.gather(self.solver_paths._is_concave_wedge, valid_wedge_idxs)
        n = tf.where(is_concave_wedge, wedges_angle / PI, _n)

        # [num_targets, num_sources, max_num_paths]
        n1 = tf.gather(n, 0, axis=0)
        n2 = tf.gather(n, 1, axis=0)

        # [2, num_targets, num_sources, max_num_paths, 3]
        e_hat = tf.gather(self._wedges_e_hat, valid_wedge_idxs)

        # [num_targets, num_sources, max_num_paths, 3]
        e_hat_w1 = tf.gather(e_hat, 0, axis=0)
        e_hat_w2 = tf.gather(e_hat, 1, axis=0)

        # Reshape sources and targets
        # [1, num_sources, 1, 3]
        sources = tf.reshape(sources, [1, -1, 1, 3])
        # [num_targets, 1, 1, 3]
        targets = tf.reshape(targets, [-1, 1, 1, 3])

        # Extract surface normals
        # [2, num_targets, num_sources, max_num_paths, 3]
        n_0_hat = normals[...,0,:]
        # [2, num_targets, num_sources, max_num_paths, 3]
        n_n_hat = normals[...,1,:]

        # Compute s_prime_hat, s_hat, s_prime, s
        # s_prime_hat : [num_targets, num_sources, max_num_paths, 3]
        # s_prime : [num_targets, num_sources, max_num_paths]
        s_prime_hat, s_prime = normalize(diff_points[0]-sources)
        # s_hat : [num_targets, num_sources, max_num_paths, 3]
        # s : [num_targets, num_sources, max_num_paths]
        s_hat, s = normalize(targets-diff_points[1])
        # w1w2_hat : [num_targets, num_sources, max_num_paths, 3]
        # w1w2 : [num_targets, num_sources, max_num_paths]
        #w1w2_hat, w1w2 = normalize(diff_points[0] - diff_points[1])
        w1w2_hat, w1w2 = normalize(diff_points[1] - diff_points[0])

        # 1) sources (w1w2s) -> diff_points[0] -> diff_points[1]
        w1w2s_n_0_hat = tf.gather(n_0_hat, 0, axis=0)
        w1w2s_e_hat = e_hat_w1 # tf.gather(e_hat, 0, axis=0)
        w1w2s_s_prime_hat = s_prime_hat
        w1w2s_s_hat = w1w2_hat
        phi1_prime, phi12, beta1_prime, _ = calc_angles(w1w2s_n_0_hat, w1w2s_e_hat, w1w2s_s_prime_hat, w1w2s_s_hat)

        is_concave1 = n1 < 1.0
        phi1_prime_, phi12_, _, _ = calc_angles_concave(w1w2s_n_0_hat, w1w2s_e_hat, w1w2s_s_prime_hat, w1w2s_s_hat, n1) # concave
        phi1_prime = tf.where(is_concave1, phi1_prime_, phi1_prime)
        phi12 = tf.where(is_concave1, phi12_, phi12)

        
        # 2) diff_points[0] -> diff_points[1] -> targets (tw1w2)
        tw1w2_n_0_hat = tf.gather(n_0_hat, 1, axis=0)
        tw1w2_e_hat = e_hat_w2  # tf.gather(e_hat, 1, axis=0)
        tw1w2_s_prime_hat = w1w2_hat
        tw1w2_s_hat = s_hat
        phi12_prime, phi2, beta2_prime, _ = calc_angles(tw1w2_n_0_hat, tw1w2_e_hat, tw1w2_s_prime_hat, tw1w2_s_hat)

        is_concave2 = n2 < 1.0
        phi12_prime_, phi2_, _, _ = calc_angles_concave(tw1w2_n_0_hat, tw1w2_e_hat, tw1w2_s_prime_hat, tw1w2_s_hat, n2) # concave
        phi12_prime = tf.where(is_concave2, phi12_prime_, phi12_prime)
        phi2 = tf.where(is_concave2, phi2_, phi2)

        phi12 = tf.where(tf.experimental.numpy.isclose(phi12, 2*PI, atol=1e-2), 1e-3*tf.ones_like(phi12), phi12)
        phi12_prime = tf.where(tf.experimental.numpy.isclose(phi12_prime, 2*PI, atol=1e-2), 1e-3*tf.ones_like(phi12_prime), phi12_prime)

        # for derivatives in slope diffraction?
        # cond1 = tf.math.greater(phi12, 1e-2)
        # phi1_prime = tf.where(cond1, invert_angles(phi1_prime, n1), phi1_prime)
        # phi12 = tf.where(cond1, invert_angles(phi12, n1), phi12)
        # e_hat_w1 = tf.where(cond1[..., None], -e_hat_w1, e_hat_w1)

        # cond2 = tf.math.greater(phi12_prime, 1e-2)
        # phi12_prime = tf.where(cond2, invert_angles(phi12_prime, n2), phi12_prime)
        # phi2 = tf.where(cond2, invert_angles(phi2, n2), phi2)
        # e_hat_w2 = tf.where(cond2[..., None], -e_hat_w2, e_hat_w2)

        r1, l, r2 = s_prime, w1w2, s

        w = tf.sqrt(r1 * r2 / ((r1+l) * (r2+l)))
        self.w = w

        if self.is_ww_analytical:
            res = self.compute_fields_analytical(paths, paths_tmp, phi1_prime, phi12, beta1_prime, phi12_prime, phi2, beta2_prime, \
                                    r1, l, r2, n1, n2, e_hat_w1, e_hat_w2, k, s_prime_hat, s_hat, w1w2_hat)
            return res
        
        #### Cascaded

        # ##### I) New (to test)
        # path_vertices = paths.vertices
        # r1_hat, r1 = normalize(path_vertices[0] - sources)
        # l_hat, l = normalize(path_vertices[1] - path_vertices[0])
        # r2_hat, r2 = normalize(targets - path_vertices[1])
        # # compute_fields_wedge_vertex
        # # compute_fields_wedge_wedge

        # #### 1) wedge
        # ell1_ = r1 * l / (r1 + l)
        # coef_ell_12 = (r1 + l) * (l+r2) / (l * (r1 + r2))
        # mat_t_wedge1, new_coef_ell, _, (e_hat1, _) = self.solver_paths.solver_wedge_diffraction.simple_compute_fields(wedge_idxs[0], 
        #                       r1_hat, l_hat, r1, l, mask, ell1_,
        #                       coef_ell=coef_ell_12, new_coef_ell=None)
        
        # #### 2) wedge
        # ell2_ = l * r2 / (l + r2)
        # mat_t_wedge2, _, _, (e_hat2, _) = self.solver_paths.solver_wedge_diffraction.simple_compute_fields(wedge_idxs[1], 
        #                       l_hat, r2_hat, r1+l, r2, mask, ell2_,
        #                       coef_ell=None, new_coef_ell=new_coef_ell)
        
        # mat_casc = tf.matmul(mat_t_wedge1, mat_t_wedge2)
        # cos_ksi = tf.cast(-tf.sign(dot(e_hat1, e_hat2)), dtype=self._dtype)
        # mat_casc = mat_casc * cos_ksi[..., None, None] / 2.0
        # mat_casc = mat_casc / tf.cast(r1, dtype=self._dtype)[..., None, None]  # incidence
        # e_hat_w1 = e_hat1
        # e_hat_w2 = e_hat2
        # s_prime_hat = r1_hat
        # s_hat = r2_hat
        # ########

        ### II) Old
        w = tf.sqrt(r1 * r2 / ((r1+l) * (r2+l)))
        coef_ell = (r1 + l) * (r2 + l) / (l * (r1 + r2))

        ell1 = r1 * l / (r1 + l) * tf.sin(beta1_prime)**2
        
        mat_w1, new_coef_ell, _ = my_wd_compute_fields(phi1_prime, phi12, beta1_prime, ell1, r1, l, n1, 
                                                    mask, self.solver_paths._scene, coef_ell=coef_ell)

        mat_w1 = mat_w1 / tf.complex(r1, tf.zeros_like(r1))[..., None, None]

        ell2 = l * s / (l + s) * tf.sin(beta2_prime)**2

        mat_w2, _, _ = my_wd_compute_fields(phi12_prime, phi2, beta2_prime, ell2, r1+l, r2, n2, mask, self.solver_paths._scene, new_coef_ell=new_coef_ell)

        mat_casc = tf.matmul(mat_w1, mat_w2) / 2.0 # grazing inc

        cos_ksi = ((dot(e_hat_w1, w1w2_hat) * dot(e_hat_w2, w1w2_hat) - dot(e_hat_w1, e_hat_w2)) \
                        /(tf.sin(beta1_prime) * tf.sin(beta2_prime)))
        mat_casc = mat_casc * tf.complex(cos_ksi, tf.zeros_like(cos_ksi))[..., None, None]

        phi1_prime_hat, _ = normalize(cross(s_prime_hat, e_hat_w1))
        beta1_prime_hat = cross(phi1_prime_hat, s_prime_hat)

        
        phi2_hat, _ = normalize(-cross(s_hat, e_hat_w2))
        beta2_hat = cross(phi2_hat, s_hat)

        ### 2)
        mat_casc_new = self.solver_paths.convert_mat_t(mat_casc, mask, theta_t, phi_t, theta_r, phi_r,
                                                      phi1_prime_hat, beta1_prime_hat, phi2_hat, beta2_hat)
        
        return mat_casc_new

    def compute_fields_analytical(self, paths, paths_tmp, phi1_prime, phi12, beta1_prime, phi12_prime, phi2, beta2_prime, \
                                   r1, l, r2, n1, n2, e_hat_w1, e_hat_w2, k, s_prime_hat, s_hat, w1w2_hat):
        
        mat_t = self._compute_fields_analytical(phi1_prime, phi12, beta1_prime, phi12_prime, phi2, beta2_prime,
                                   r1, l, r2, n1, n2, e_hat_w1, e_hat_w2, k, s_prime_hat, s_hat, w1w2_hat,
                                   objects=paths.objects, mode = 'all')
        
        #incidence
        mat_t = mat_t / tf.cast(r1[..., None, None], self._dtype)

        theta_t = paths.theta_t
        phi_t = paths.phi_t
        theta_r = paths.theta_r
        phi_r = paths.phi_r

        # [num_targets, num_sources, max_num_paths, 3]
        phi1_prime_hat, _ = normalize(cross(s_prime_hat, e_hat_w1))
        beta1_prime_hat = cross(phi1_prime_hat, s_prime_hat)

        # [num_targets, num_sources, max_num_paths, 3]
        phi2_hat, _ = normalize(-cross(s_hat, e_hat_w2))
        beta2_hat = cross(phi2_hat, s_hat)

        mat_t = self.solver_paths.convert_mat_t(mat_t, paths.mask, theta_t, phi_t, theta_r, phi_r, 
                                                    phi1_prime_hat, beta1_prime_hat, phi2_hat, beta2_hat)

        return mat_t
        

    def _compute_fields_analytical(self, phi1_prime, phi12, beta1_prime, phi12_prime, phi2, beta2_prime,
                                   r1, l, r2, n1, n2, e_hat_w1, e_hat_w2, k, s_prime_hat, s_hat, w1w2_hat,
                                    objects=None, mode='all'):
        # [num_targets, num_sources, max_num_paths]
        cos_ksi = ((dot(e_hat_w1, w1w2_hat) * dot(e_hat_w2, w1w2_hat) - dot(e_hat_w1, e_hat_w2)) \
                        /(tf.sin(beta1_prime) * tf.sin(beta2_prime)))
        
        sin_ksi = dot(cross(e_hat_w2, e_hat_w1), w1w2_hat) / (tf.sin(beta1_prime) * tf.sin(beta2_prime))

        # eps12 = -tf.sign(dot(e_hat_w1, e_hat_w2))

        # [num_targets, num_sources, max_num_paths, 4]
        Phi1_pq = _get_Phi1_pq(phi1_prime, phi12)
        Phi2_rs = _get_Phi2_rs(phi2, phi12_prime)

        l_new = l + (r1*r2* sin_ksi**2 / (r1+l+r2))

        a_pq = _get_a_pq(r1, l_new, beta1_prime, Phi1_pq, n1, k)
        b_rs = _get_b_rs(r2, l_new, beta2_prime, Phi2_rs, n2, k)

        # [num_targets, num_sources, max_num_paths, 4, 4]
        w = cos_ksi * tf.sqrt(r1 * r2 / ((r1 + l) * (r2 + l)))
        w = insert_dim_num(w, -1, 4)
        w = insert_dim_num(w, -1, 4)
        a_pq = insert_dim_num(a_pq, -1, 4)
        b_rs = insert_dim_num(b_rs, -2, 4)  # TODO: check axis
        Phi1_pq = insert_dim_num(Phi1_pq, -1, 4)
        Phi2_rs = insert_dim_num(Phi2_rs, -2, 4)

        # [num_targets, num_sources, max_num_paths, 4, 4, 4]
        gfis_2d = _get_gfis2d(a_pq, b_rs, w, self.solver_paths.is_table_gfi)
        
        # [num_targets, num_sources, max_num_paths]
        coef_in_D = tf.complex(np.zeros_like(l), -1.0 / (8*PI*k*n1*n2 * tf.sin(beta1_prime) * tf.sin(beta2_prime)))
        #coef_in_D_2_tilda = tf.complex(-1.0 * cos_ksi / (32*PI*l_new * (k*n1*n2*tf.sin(beta1_prime)*tf.sin(beta2_prime))**2), np.zeros_like(n1))
        coef_in_D_2_tilda = tf.cast(-1.0 * cos_ksi / (32*PI*l_new * (k*n1*n2*tf.sin(beta1_prime)*tf.sin(beta2_prime))**2), self._dtype)

        # [num_targets, num_sources, max_num_paths, 4, 4]
        n1 = insert_dim_num(n1, -1, 4)
        n1 = insert_dim_num(n1, -1, 4)
        n2 = insert_dim_num(n2, -1, 4)
        n2 = insert_dim_num(n2, -1, 4)

        # [num_targets, num_sources, max_num_paths, 4, 4]
        beta1_prime = insert_dim_num(beta1_prime, -1, 4)
        beta1_prime = insert_dim_num(beta1_prime, -1, 4)
        beta2_prime = insert_dim_num(beta2_prime, -1, 4)
        beta2_prime = insert_dim_num(beta2_prime, -1, 4)

        def _get_D12():
            r""""
            Output
            ------
            D_soft : [num_targets, num_sources, max_num_paths]
            D_hard : [num_targets, num_sources, max_num_paths]
            """

            # [num_targets, num_sources, max_num_paths, 4, 4]
            func_T = _get_func_T(a_pq, b_rs, w, gfis_2d)
            terms_D_tmp = (1 / tf.tan(Phi1_pq/(2.0*n1))) * (1 / tf.tan(Phi2_rs/(2.0*n2))) 
            terms_D = tf.complex(terms_D_tmp, np.zeros_like(terms_D_tmp))* func_T

            mask_hard = tf.constant([   # phi1_prime, phi2_prime
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]
            ], dtype=self._dtype)

            mask_soft = tf.constant([   # beta1_prime, beta2_prime
                [1, -1, -1, 1],
                [-1, 1, 1, -1],
                [-1, 1, 1, -1],
                [1, -1, -1, 1]
            ], dtype=self._dtype)

            mask_sh = tf.constant([   # beta1_prime, phi2_prime
                [1, 1, 1, 1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [1, 1, 1, 1]
            ], dtype=self._dtype)

            mask_hs = tf.constant([   # phi1_prime, beta2_prime
                [1, -1, -1, 1],
                [1, -1, -1, 1],
                [1, -1, -1, 1],
                [1, -1, -1, 1]
            ], dtype=self._dtype)

            mask_hard = mask_hard[None, None, None, ...]
            mask_soft = mask_soft[None, None, None, ...]
            mask_sh = mask_sh[None, None, None, ...]
            mask_hs = mask_hs[None, None, None, ...]

            gamma_ss = tf.cast(cos_ksi[..., None, None], self._dtype)
            gamma_hh = tf.cast(cos_ksi[..., None, None], self._dtype)
            gamma_sh = tf.cast(sin_ksi[..., None, None], self._dtype)
            gamma_hs = -tf.cast(sin_ksi[..., None, None], self._dtype)

            soft = tf.reduce_sum(gamma_ss * terms_D * mask_soft, axis=[-2, -1]) * coef_in_D 
            hard = tf.reduce_sum(gamma_hh * terms_D * mask_hard, axis=[-2, -1]) * coef_in_D
            sh = tf.reduce_sum(gamma_sh * terms_D * mask_sh, axis=[-2, -1]) * coef_in_D
            hs = tf.reduce_sum(gamma_hs * terms_D * mask_hs, axis=[-2, -1]) * coef_in_D

            return soft, hard, sh, hs

        def _get_D12_2_tilda():
            r""""
            Input
            ------
            #eps12_sign : [num_targets, num_sources, max_num_paths]
            
            Output
            ------
            D_soft : [num_targets, num_sources, max_num_paths]
            D_hard : [num_targets, num_sources, max_num_paths]
            """
            

            # [num_targets, num_sources, max_num_paths, 4, 4] 
            func_T_2_tilda = _get_func_T_2_tilda(a_pq, b_rs, w, gfis_2d)

            coef1 = 1 / tf.sin(Phi1_pq/(2*n1))
            coef2 = 1 / tf.sin(Phi2_rs/(2*n2))
            terms_D = tf.complex((coef1 * coef2)**2, np.zeros_like(coef1)) * func_T_2_tilda

            mask_hard = tf.constant([   # phi1_prime, phi2_prime
                [1, -1, 1, -1],
                [-1, 1, -1, 1],
                [1, -1, 1, -1],
                [-1, 1, -1, 1]
            ], dtype=self._dtype)

            mask_soft = tf.constant([    # beta1_prime, beta2_prime
                [1, 1, -1, -1],
                [1, 1, -1, -1],
                [-1, -1, 1, 1],
                [-1, -1, 1, 1]
            ], dtype=self._dtype)

            mask_sh = tf.constant([   # beta1_prime, phi2_prime
                [1, -1, 1, -1],
                [1, -1, 1, -1],
                [-1, 1, -1, 1],
                [-1, 1, -1, 1]
            ], dtype=self._dtype)

            mask_hs = tf.constant([   # phi1_prime, beta2_prime
                [1, 1, -1, -1],
                [-1, -1, 1, 1],
                [1, 1, -1, -1],
                [-1, -1, 1, 1]
            ], dtype=self._dtype)

            mask_hard = mask_hard[None, None, None, ...]
            mask_soft = mask_soft[None, None, None, ...]
            mask_sh = mask_sh[None, None, None, ...]
            mask_hs = mask_hs[None, None, None, ...]

            gamma_ss = tf.cast(cos_ksi[..., None, None], self._dtype)
            gamma_hh = tf.cast(cos_ksi[..., None, None], self._dtype)
            gamma_sh = tf.cast(sin_ksi[..., None, None], self._dtype)
            gamma_hs = -tf.cast(sin_ksi[..., None, None], self._dtype)

            soft = tf.reduce_sum(gamma_ss * terms_D * mask_soft, axis=[-2, -1]) * coef_in_D_2_tilda
            hard = tf.reduce_sum(gamma_hh * terms_D * mask_hard, axis=[-2, -1]) * coef_in_D_2_tilda
            sh = tf.reduce_sum(gamma_sh * terms_D * mask_sh, axis=[-2, -1]) * coef_in_D_2_tilda
            hs = tf.reduce_sum(gamma_hs * terms_D * mask_hs, axis=[-2, -1]) * coef_in_D_2_tilda

            return soft, hard, sh, hs

        # [num_targets, num_sources, max_num_paths]
        D12_soft, D12_hard, D12_sh, D12_hs = _get_D12()

        # [num_targets, num_sources, max_num_paths]
        D12_2_tilda_soft, D12_2_tilda_hard, D12_2_tilda_sh, D12_2_tilda_hs = _get_D12_2_tilda()

        # get slopes: 0.5 coef for grazing incidence
        if objects is None:
            coef = tf.cast(0.5, dtype=self._dtype)
        else:
            # only for EE (objects are wedges)
            slopes_mask = self.get_slopes(objects)
            coef = tf.cast(tf.where(slopes_mask, 0.5, 1.0), dtype=self._dtype)

        if mode == 'all':
            # [num_targets, num_sources, max_num_paths]
            D_soft = -coef * (D12_soft + D12_2_tilda_soft)
            D_hard = -coef * (D12_hard + D12_2_tilda_hard)
            D_sh = -coef * (D12_sh + D12_2_tilda_sh)
            D_hs = -coef * (D12_hs + D12_2_tilda_hs)
        elif mode == 'main':
            D_soft = -coef * D12_soft
            D_hard = -coef * D12_hard
            D_sh = -coef * D12_sh
            D_hs = -coef * D12_hs
        elif mode == 'slope':
            D_soft = -coef * D12_2_tilda_soft
            D_hard = -coef * D12_2_tilda_hard
            D_sh = -coef * D12_2_tilda_sh
            D_hs = -coef * D12_2_tilda_hs
        else:
            raise ValueError("Wrong mode")

        # [num_targets, num_sources, max_num_paths]
        # (1 / r1) is incident field
        spreading_factor = np.sqrt(r1) / (np.sqrt(l * r2) * np.sqrt(r1 + l + r2))
        spreading_factor = tf.complex(spreading_factor, tf.zeros_like(spreading_factor))
        spreading_factor = tf.expand_dims(spreading_factor, axis=-1)
        spreading_factor = tf.expand_dims(spreading_factor, axis=-1)

        mat_t1 = tf.stack([D_hard, D_sh], axis=-1)
        mat_t2 = tf.stack([D_hs, D_soft], axis=-1)

        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = tf.stack([mat_t1, mat_t2], axis=-2)

        # # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t *= -spreading_factor

        return mat_t
    
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
    
    def ddd_compute_fields(self, paths: Paths, paths_tmp: PathsTmpData):
        mask = paths.mask
        targets = paths.targets
        sources = paths.sources
        theta_t = paths.theta_t
        phi_t = paths.phi_t
        theta_r = paths.theta_r
        phi_r = paths.phi_r

        # num_targets = targets.shape[0]
        # num_sources = sources.shape[0]
        # num_vertices = paths.vertices.shape[3]

        #normals = paths_tmp.normals

        wavelength = self.solver_paths._scene.wavelength
        k = 2.0*PI/wavelength

        wedge_idxs = paths.objects
        #valid_wedges_idx = tf.where(wedge_idxs == -1, 0, wedge_idxs)
        path_vertices = paths.vertices

        # [1, num_sources, 1, 3]
        sources = tf.reshape(sources, [1, -1, 1, 3])
        # [num_targets, 1, 1, 3]
        targets = tf.reshape(targets, [-1, 1, 1, 3])
        #num_paths = tf.shape(paths.vertices)[3]

        ###  I) 3 wedges
        r1_hat, r1 = normalize(path_vertices[0] - sources)
        l1_hat, l1 = normalize(path_vertices[1] - path_vertices[0])
        l2_hat, l2 = normalize(path_vertices[2] - path_vertices[1])
        r2_hat, r2 = normalize(targets - path_vertices[2])
        # compute_fields_wedge_vertex
        # compute_fields_wedge_wedge

        #### 1) wedge
        ell1_ = r1 * l1 / (r1 + l1)
        coef_ell_12 = (r1 + l1) * (l2 + l1) / (l1 * (r1 + l2))
        mat_t_wedge1, new_coef_ell_12, _, (e_hat1, _) = self.solver_paths.solver_wedge_diffraction.simple_compute_fields(wedge_idxs[0], 
                              r1_hat, l1_hat, r1, l1, mask, ell1_,
                              coef_ell=coef_ell_12, new_coef_ell=None)
        
        #### 2) wedge
        # ell2_ = l1 * l2 / (l1 + l2)
        ell2_ = l1 * l2 / (l1 + l2)
        coef_ell_23 = (l1 + l2) * (r2 + l2) / (l2 * (l1 + r2))

        mat_t_wedge2, new_coef_ell_23, _, (e_hat2, _) = self.solver_paths.solver_wedge_diffraction.simple_compute_fields(wedge_idxs[1], 
                              l1_hat, l2_hat, r1+l1, l2, mask, ell2_,
                              coef_ell=coef_ell_23, new_coef_ell=new_coef_ell_12)
        
        #### 3) wedge
        ell3_ = l2 * r2 / (l2 + r2)
        mat_t_wedge3, _, _, (e_hat3, _) = self.solver_paths.solver_wedge_diffraction.simple_compute_fields(wedge_idxs[2], 
                              l2_hat, r2_hat, r1+l1+l2, r2, mask, ell3_,
                              coef_ell=None, new_coef_ell=new_coef_ell_23)

        cos_ksi = tf.cast(tf.sign(dot(e_hat1, e_hat3)), dtype=self._dtype)

        mat_t = tf.matmul(tf.matmul(mat_t_wedge1, mat_t_wedge2), mat_t_wedge3)
        # mat_t = tf.matmul(mat_t_wedge1, mat_t_wedge2)
        mat_t = mat_t * cos_ksi[..., None, None]
        mat_t = mat_t / 4.0   # TODO

        # ##### II) 2 wedges 
        # r1_hat, r1 = normalize(path_vertices[0] - sources)
        # l_hat, l = normalize(path_vertices[1] - path_vertices[0])
        # r2_hat, r2 = normalize(targets - path_vertices[1])
        # # compute_fields_wedge_vertex
        # # compute_fields_wedge_wedge

        # #### 1) wedge
        # ell1_ = r1 * l / (r1 + l)
        # coef_ell_12 = (r1 + l) * (l+r2) / (l * (r1 + r2))
        # mat_t_wedge1, new_coef_ell_23, _, (e_hat1, _) = self.solver_paths.solver_wedge_diffraction.simple_compute_fields(wedge_idxs[0], 
        #                       r1_hat, l_hat, r1, l, mask, ell1_,
        #                       coef_ell=coef_ell_12, new_coef_ell=None)
        
        # #### 2) wedge
        # ell2_ = l * r2 / (l + r2)
        # #coef_ell23 = (l1 + l2) * (r2 + l2) / (l2 * (l1 + r2))
        # mat_t_wedge2, _, _, (e_hat2, _) = self.solver_paths.solver_wedge_diffraction.simple_compute_fields(wedge_idxs[1], 
        #                       l_hat, r2_hat, r1+l, r2, mask, ell2_,
        #                       coef_ell=None, new_coef_ell=new_coef_ell_23)
        
        # mat_t = tf.matmul(mat_t_wedge1, mat_t_wedge2)
        # cos_ksi = tf.cast(-tf.sign(dot(e_hat1, e_hat2)), dtype=self._dtype)
        # mat_t = mat_t * cos_ksi[..., None, None] / 2.0
        # ########

        ##### III) 1 wedge
        r1_hat, r1 = normalize(path_vertices[0] - sources)
        r2_hat, r2 = normalize(targets - path_vertices[0])

        # #### 1) wedge
        # ell1_ = r1 * r2 / (r1 + r2)
        # #coef_ell_12 = (r1 + l) * (l+r2) / (l * (r1 + r2))
        # mat_t_wedge1, new_coef_ell_23, _, (e_hat1, _) = self.solver_paths.solver_wedge_diffraction.simple_compute_fields(wedge_idxs[0], 
        #                       r1_hat, r2_hat, r1, r2, mask, ell1_,
        #                       coef_ell=None, new_coef_ell=None)

        # mat_t = mat_t_wedge1
        ############

        phi1_prime_hat, _ = normalize(cross(r1_hat, e_hat1))
        beta1_prime_hat = cross(phi1_prime_hat, r1_hat)
        phi2_hat, _ = normalize(-cross(r2_hat, e_hat3))  # TODO
        beta2_hat = cross(phi2_hat, r2_hat)

        mat_t = self.solver_paths.convert_mat_t(mat_t, mask, theta_t, phi_t, theta_r, phi_r,
                                                      phi1_prime_hat, beta1_prime_hat, phi2_hat, beta2_hat)

        # incidence
        mat_t = mat_t / tf.cast(r1[..., None, None], self._dtype)
        
        return mat_t



            
        
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

def _get_gfis2d(a_pq, b_rs, w, is_table=False):
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

    #[num_targets, num_sources, max_num_paths, 4, 4]
    gfi1 = func_G_approximated_capolino(a_pq, (b_rs + w*a_pq) / w_coef, is_table)
    gfi2 = func_G_approximated_capolino(b_rs, (a_pq + w*b_rs) / w_coef, is_table)
    gfi3 = func_G_approximated_capolino(a_pq, (b_rs - w*a_pq) / w_coef, is_table)
    gfi4 = func_G_approximated_capolino(b_rs, (a_pq - w*b_rs) / w_coef, is_table)

    return tf.stack([gfi1, gfi2, gfi3, gfi4], axis=-1)



# def calc_angles(n_0_hat, e_hat, s_prime_hat, s_hat):
#     r"""
#     Input
#     ------
#     n_0_hat: [num_targets, num_sources, max_num_paths, 3]
#     e_hat: [num_targets, num_sources, max_num_paths, 3]
#     s_prime_hat: [num_targets, num_sources, max_num_paths, 3]
#     s_hat: [num_targets, num_sources, max_num_paths, 3]
#     """
#     # [num_targets, num_sources, max_num_paths, 3]
#     t_0_hat = cross(n_0_hat, e_hat)

#     # Compute s_t_prime_hat and s_t_hat
#     # [num_targets, num_sources, max_num_paths, 3]
#     s_t_prime_hat, _ = normalize(s_prime_hat
#                             - dot(s_prime_hat,e_hat, keepdim=True)*e_hat)
#     # [num_targets, num_sources, max_num_paths, 3]
#     s_t_hat, _ = normalize(s_hat - dot(s_hat,e_hat, keepdim=True)*e_hat)

#     # Compute phi_prime and phi
#     # [num_targets, num_sources, max_num_paths]
#     # phi_prime = PI - \
#     #     (PI-acos_diff(-dot(s_t_prime_hat, t_0_hat)))\
#     #         * sign(-dot(s_t_prime_hat, n_0_hat))
#     # # [num_targets, num_sources, max_num_paths]
#     # phi = PI - (PI-acos_diff(dot(s_t_hat, t_0_hat)))\
#     #     * sign(dot(s_t_hat, n_0_hat))

#     my_phi_prime_tmp = acos_diff(dot(-s_t_prime_hat, t_0_hat))
#     #my_phi_prime = tf.where(sign(dot(e_hat, cross(n_0_hat, -s_t_prime_hat))) < 0.0, my_phi_prime_tmp, 2*PI - my_phi_prime_tmp)
#     my_phi_prime = tf.where(sign(dot(e_hat, cross(t_0_hat, -s_t_prime_hat))) > 0.0, my_phi_prime_tmp, 2*PI - my_phi_prime_tmp)

#     my_phi_tmp = acos_diff(dot(s_t_hat, t_0_hat))
#     my_phi = tf.where(sign(dot(e_hat, cross(t_0_hat, s_t_hat))) > 0.0, my_phi_tmp, 2*PI - my_phi_tmp)

#     # Compute elevation beta_prime
#     # [num_targets, num_sources, max_num_paths]
#     beta = acos_diff(dot(e_hat, s_hat))
#     beta_prime = acos_diff(dot(e_hat, s_prime_hat))
#     # beta_prime == acos_diff(dot(e_hat, s_hat))

#     #return phi_prime, phi, beta_prime
#     return my_phi_prime, my_phi, beta_prime, beta


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