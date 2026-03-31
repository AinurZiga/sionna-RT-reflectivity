
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
from .diffraction_funcs import calc_angles, calc_angles_concave, func_G_approximated_capolino, t_gfi_full_approximated_capolino, get_coef_ell
from .utils import dot, phi_hat, theta_hat, theta_phi_from_unit_vec,\
    normalize, moller_trumbore, component_transform, mi_to_tf_tensor,\
        compute_field_unit_vectors, reflection_coefficient, fibonacci_lattice,\
            cot, cross, sign, rotation_matrix, acos_diff, rot_mat_from_unit_vecs

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
        self.is_refl_vd = False

        # [num_primitives, 3]
        self._primitives_2_wedges = self.solver_paths._primitives_2_wedges

        # [num_objects+1]
        self._objects_2_primitives = [shape.effective_primitive_count() 
                    for shape in (self.solver_paths._mi_scene.shapes())]
        self._objects_2_primitives.insert(0, 0)
        self._objects_2_primitives = np.cumsum(self._objects_2_primitives)

        # [num_wedges, 3]
        self._wedges_origin = self.solver_paths._wedges_origin
        self._wedges_e_hat = self.solver_paths._wedges_e_hat
        self._wedges_length = self.solver_paths._wedges_length
        self._facet_points = self.solver_paths._facet_points

        # [num_wedges, 2, 3]
        self._wedges_normals = self.solver_paths._wedges_normals

        # [num_wedges]
        self._is_concave_wedge = self.solver_paths._is_concave_wedge
    
    def set_objects_geometry(self, objects_geometry: ObjectsGeometry):
        self.obj_geom = objects_geometry
        self.solver_paths.obj_geom = objects_geometry
        self.is_setup = True
        
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

        # Origin of the coordinate system of the edge is the vertex. 
        # If not, the direction of the edge is flipped

        # needs_swap = tf.reduce_all(tf.math.equal(vertex_wedges_origin_tmp, vertex_wedges_origin), axis=1)
        # needs_swap = tf.expand_dims(needs_swap, axis=1)
        # vertex_wedges_e_hat = tf.where(needs_swap, vertex_wedges_origin_tmp, vertex_wedges_e_hat_tmp)

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

    def tracing(self, sources, targets, case: Literal["tx", "rx", "both"]='both', is_dd=False):
        """
        Trace vertex diffracted paths between sources and targets.
        is_dd (double-diffraction) to get vertices for double-diffraction
        """

        vd_paths = Paths(sources=sources, targets=targets, scene=self.solver_paths._scene, types=Paths.VERTEX_DIFFRACTED)
        vd_paths_tmp = PathsTmpData(sources, targets, self.solver_paths._dtype)

        num_sources = tf.shape(sources)[0]
        num_targets = tf.shape(targets)[0]

        if is_dd:
            active = self.obj_geom.active_vertices_dd
        else:
            active = self.obj_geom.active_vertices

        vertices = tf.boolean_mask(self._vertices, active)
        vertices_2_wedges = tf.ragged.boolean_mask(self.obj_geom.vertex_2_wedges, active)

        num_vertices = vertices.shape[0]

        # [num_targets, num_sources, max_num_paths / num_wedges_per_vertex]
        vertices_2_edges_row_lengths = vertices_2_wedges.row_lengths()

        # [num_targets, num_sources, max_num_paths, 3]  
        #[ 1, max_num_paths, 3]
        diff_vertex_points = tf.expand_dims(vertices, axis=0)
        # [1, 1, max_num_paths, 3]
        diff_vertex_points = tf.expand_dims(diff_vertex_points, axis=0)
        # [num_targets, num_sources, max_num_paths, 3]
        # diff_vertex_points = tf.repeat(diff_vertex_points, num_targets, axis=0)
        # diff_vertex_points = tf.repeat(diff_vertex_points, num_sources, axis=1)

        diff_vertex_points = tf.broadcast_to(diff_vertex_points, [num_targets, num_sources, num_vertices, 3])

        # [num_vertex_wedges]
        # vertex_wedges_indices = self.vertices_2_wedges.values
        vertex_wedges_indices = vertices_2_wedges.values

        # # map back to original vertex indices (using active)
        vertex_indices = tf.cast(tf.where(active)[:, 0], dtype=tf.int32)

        # [1, max_num_paths / num_wedges_per_vertex]
        vertex_indices = tf.expand_dims(vertex_indices, axis=0)
        # [1, 1, max_num_paths / num_wedges_per_vertex]
        vertex_indices = tf.expand_dims(vertex_indices, axis=0)
        # [num_targets, num_sources, max_num_paths / num_wedges_per_vertex]
        # vertex_indices = tf.repeat(vertex_indices, num_targets, axis=0)
        # vertex_indices = tf.repeat(vertex_indices, num_sources, axis=1)
        vertex_indices = tf.broadcast_to(vertex_indices, [num_targets, num_sources, num_vertices])

        # Discard obstructed diffracted paths
        # Only check for vertex visibility if there is at least one vertex-candidate
        if vertex_indices.shape[2] > 0: # Number of diff. paths > 0
            # Discard obstructed paths
            vertex_indices, vertex_wedges_indices, diff_vertex_points, valid_vertex_idxs =\
                self._check_vertices_visibility(targets, sources,
                                                vertex_indices,
                                                vertex_wedges_indices,
                                                diff_vertex_points,
                                                vertices_2_edges_row_lengths,
                                                case,
                                                ) 
            vd_paths, vd_paths_tmp = \
                self._vertex_diffraction_create_paths(vd_paths, vd_paths_tmp, vertex_indices, \
                                                       diff_vertex_points)

        return vd_paths, vd_paths_tmp

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
      
    
    def compute_fields(self, relative_permittivity,
                                scattering_coefficient, 
                                paths: Paths, 
                                paths_tmp: PathsTmpData):
        """
        Main function for computing electric fields of vertex diffracted paths.
        """
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

        normals = paths_tmp.normals

        # [num_targets, num_sources, max_num_paths, 3]
        diff_points = paths.vertices[0]
        # [num_targets, num_sources, max_num_paths]
        vertex_indices = paths.objects[0]

        # [num_targets, num_sources, max_num_paths]
        valid_vertex_idxs = tf.where(vertex_indices == -1, 0, vertex_indices)

        vertex_indices_mask = vertex_indices != -1

        #[max_num_paths, max_num_wedges_per_vertex]
        non_valid_tensor_vertices_2_wedges = self.vertices_2_wedges.to_tensor(default_value=-1)
        tensor_vertices_2_wedges = self.vertices_2_wedges.to_tensor(default_value=0)

        # [num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex]
        mask_wedges_in_vertex_ = tf.gather(non_valid_tensor_vertices_2_wedges, valid_vertex_idxs)
        mask_wedges_in_vertex_ = tf.not_equal(mask_wedges_in_vertex_, -1)
        mask_wedges_in_vertex_ = tf.logical_and(mask_wedges_in_vertex_, tf.expand_dims(vertex_indices_mask, -1))

        #[num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex, 2, 2]
        mask_wedges_in_vertex = tf.stack([mask_wedges_in_vertex_, mask_wedges_in_vertex_], axis=-1)
        mask_wedges_in_vertex = tf.stack([mask_wedges_in_vertex, mask_wedges_in_vertex], axis=-1)

        #[num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex]
        tensor_valid_vertices_2_wedges = tf.gather(tensor_vertices_2_wedges, valid_vertex_idxs)

        # [num_targets, num_sources, max_num_paths_extended, 3]
        diff_points_ext = tf.repeat(diff_points, tensor_valid_vertices_2_wedges.shape[3], axis=2)
        # diff_points_ext = tf.broadcast_to(diff_points, [num_targets, num_sources, max_num_paths_extended, 3])

        #[num_targets, num_sources, max_num_paths_extended]
        valid_vertex_wedges_idx = tf.reshape(tensor_valid_vertices_2_wedges, [num_targets, num_sources, -1])

        # [num_targets, num_sources, max_num_paths_extended, 2, 3]
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
        needs_swap = tf.logical_not(tf.reduce_all(tf.math.equal(wedge_origins_tmp, diff_points_ext), axis=3))
        # [num_targets, num_sources, max_num_paths_extended, 3]
        needs_swap = tf.expand_dims(needs_swap, axis=3)
        e_hat = tf.where(needs_swap, -e_hat, e_hat)
        # [num_targets, num_sources, max_num_paths_extended, 2, 3]

        # Reshape sources and targets
        # [1, num_sources, 1, 3]
        sources = tf.reshape(sources, [1, -1, 1, 3])
        # [num_targets, 1, 1, 3]
        targets = tf.reshape(targets, [-1, 1, 1, 3])

        n_0_hat = tf.where(needs_swap, normals[...,1,:], normals[...,0,:]) 
        n_n_hat = tf.where(needs_swap, normals[...,0,:], normals[...,1,:])

        # Relative permitivities and scattering coefficients
        # If a callable is defined to compute the radio material properties,
        # it is invoked. Otherwise, the radio materials of objects are used.
        rm_callable = self.solver_paths._scene.radio_material_callable
        # [num_targets, num_sources, max_num_paths, 2]

        objects_indices = tf.gather(self.solver_paths._wedges_objects, valid_vertex_wedges_idx,
                                    axis=0)
        if rm_callable is None:
            # [num_targets, num_sources, max_num_paths, 2]
            etas = tf.gather(relative_permittivity, objects_indices)
            scattering_coefficient = tf.gather(scattering_coefficient,
                                               objects_indices)
        else:
            # Harmonize the shapes of the radio material callables
            # [num_targets, num_sources, max_num_paths, 2, 3]
            diff_points_ = tf.tile(tf.expand_dims(diff_points_ext, axis=-2),
                                   [1, 1, 1, 2, 1])
            # scattering_coefficient, etas : [num_targets, num_sources,
            #   max_num_paths, 2]
            etas, scattering_coefficient, _   = rm_callable(objects_indices,
                                                            diff_points_)
            
        # Compute s_prime_hat, s_hat, s_prime, s
        # s_prime_hat : [num_targets, num_sources, max_num_paths, 3]
        # s_prime : [num_targets, num_sources, max_num_paths]
        s_prime_hat, s_prime = normalize(diff_points_ext - sources)
        # s_hat : [num_targets, num_sources, max_num_paths, 3]
        # s : [num_targets, num_sources, max_num_paths]
        s_hat, s = normalize(targets - diff_points_ext)

        mat_t = self._compute_fields(s_prime_hat, s_hat, n_0_hat, n_n_hat, e_hat, etas,
                                    scattering_coefficient, s_prime, s, wedge_n, mask,
                                    mask_wedges_in_vertex_, mask_wedges_in_vertex,
                                    theta_t, phi_t, theta_r, phi_r, tensor_valid_vertices_2_wedges,
                                    diff_points_ext, valid_vertex_wedges_idx)
        
        _, s_prime = normalize(diff_points - sources)

        return mat_t / tf.complex(s_prime[..., None, None], tf.zeros_like(s_prime[..., None, None]))

    def _compute_fields(self, s_prime_hat, s_hat, n_0_hat, n_n_hat, e_hat, etas,
                                    scattering_coefficient, s_prime, s, wedge_n, mask,
                                    mask_wedges_in_vertex_, mask_wedges_in_vertex,
                                    theta_t, phi_t, theta_r, phi_r, tensor_valid_vertices_2_wedges,
                                    diff_points, valid_vertex_wedges_idx, skip_sf=False):
        wavelength = self.solver_paths._scene.wavelength
        k = 2.*PI/wavelength
        num_targets = s_prime_hat.shape[0]
        num_sources = s_prime_hat.shape[1]
        num_vertices = mask_wedges_in_vertex_.shape[2]
        num_wedges_per_vertex = mask_wedges_in_vertex_.shape[3]

        # [num_targets, num_sources, max_num_paths]
        eta_0 = etas[...,0]
        eta_n = etas[...,1]
        # [num_targets, num_sources, max_num_paths]
        scattering_coefficient_0 = scattering_coefficient[...,0]
        scattering_coefficient_n = scattering_coefficient[...,1]

        # [num_targets, num_sources, max_num_paths, 3]
        phi_prime_hat, _ = normalize(cross(s_prime_hat, e_hat))
        # [num_targets, num_sources, max_num_paths, 3]
        beta_0_prime_hat = cross(phi_prime_hat, s_prime_hat)

        # [num_targets, num_sources, max_num_paths, 3]
        phi_hat_, _ = normalize(-cross(s_hat, e_hat))
        beta_0_hat = cross(phi_hat_, s_hat)

        ## Compute phi_prime and phi
        is_concave = wedge_n < 1.0

        # check for concave wedges and compute angles accordingly
        phi_prime, phi, beta_prime, beta = calc_angles(n_0_hat, e_hat, s_prime_hat, s_hat)
        phi_prime_, phi_, beta_prime, beta = calc_angles_concave(n_0_hat, e_hat, s_prime_hat, s_hat, wedge_n)
        phi_prime = tf.where(is_concave, phi_prime_, phi_prime)
        phi = tf.where(is_concave, phi_, phi)
        
        # Compute field component vectors for reflections at both surfaces
        # [num_targets, num_sources, max_num_paths, 3]
        # pylint: disable=unbalanced-tuple-unpacking
        e_i_s_0, e_i_p_0, e_r_s_0, e_r_p_0 = compute_field_unit_vectors(
            s_prime_hat,
            s_hat,
            n_0_hat,#*sign(-dot(s_t_prime_hat, n_0_hat, keepdim=True)),
            self.solver_paths.EPSILON
            )
        # [num_targets, num_sources, max_num_paths, 3]
        # pylint: disable=unbalanced-tuple-unpacking
        e_i_s_n, e_i_p_n, e_r_s_n, e_r_p_n = compute_field_unit_vectors(
            s_prime_hat,
            s_hat,
            n_n_hat,#*sign(-dot(s_t_prime_hat, n_n_hat, keepdim=True)),
            self.solver_paths.EPSILON
            )

        # Compute Fresnel reflection coefficients for 0- and n-surfaces
        # [num_targets, num_sources, max_num_paths]
        r_s_0, r_p_0 = reflection_coefficient(eta_0, tf.abs(tf.sin(phi_prime)))
        r_s_n, r_p_n = reflection_coefficient(eta_n, tf.abs(tf.sin(wedge_n*PI-phi)))

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

        # [num_targets, num_sources, max_num_paths]
        coef_ = -1 / tf.complex(tf.cast(0.0, dtype=self._rdtype), 
                               (2.0 * k * PI * (tf.math.cos(beta_prime) - tf.math.cos(beta))))
        step_func = tf.logical_and(tf.math.greater(wedge_n * PI - phi + 1e-5, 0.0),
                                   tf.math.greater(wedge_n * PI - phi_prime + 1e-5, 0.0))
        step_func = tf.cast(step_func, self._dtype)
        #coef *= step_func

        # [num_targets, num_sources, max_num_paths]
        param_u = self._get_param_u(beta_prime, beta)
        param_b = self._get_param_b(k, s_prime, s, beta_prime, beta)

        ell = s_prime * s / (s_prime + s) * tf.sin(beta_prime) * tf.sin(beta)

        res, _, _ = self._get_coefs_D2(wedge_n, phi_prime, phi, k, ell, param_b, param_u)

        d1_, d2_, d3_, d4_ = res

        # [num_targets, num_sources, max_num_paths * max_num_wedges_per_vertex]
        coef = coef_ * step_func

        # [num_targets, num_sources, max_num_paths]
        d1 = _clear_diff_coef(coef*d1_, mask_wedges_in_vertex_, num_targets, num_sources, num_vertices, self._dtype)
        d2 = _clear_diff_coef(coef*d2_, mask_wedges_in_vertex_, num_targets, num_sources, num_vertices, self._dtype)
        d3 = _clear_diff_coef(coef*d3_, mask_wedges_in_vertex_, num_targets, num_sources, num_vertices, self._dtype)
        d4 = _clear_diff_coef(coef*d4_, mask_wedges_in_vertex_, num_targets, num_sources, num_vertices, self._dtype)

        d1 = tf.reshape(d1, tf.concat([tf.shape(d1), [1,1]], axis=0))
        d2 = tf.reshape(d2, tf.concat([tf.shape(d2), [1,1]], axis=0))
        d3 = tf.reshape(d3, tf.concat([tf.shape(d3), [1,1]], axis=0))
        d4 = tf.reshape(d4, tf.concat([tf.shape(d4), [1,1]], axis=0))

        # [num_targets, num_sources, max_num_paths * num_wedges_per_vertex]
        spreading_factor = 1.0 / s
        spreading_factor = tf.complex(spreading_factor,
                                      tf.zeros_like(spreading_factor))
        # [num_targets, num_sources, max_num_paths * num_werges_per_vertex, 1, 1]
        spreading_factor = tf.reshape(spreading_factor, tf.shape(d1))

        mat_t = (d1 + d2)*tf.eye(2,2, batch_shape=tf.shape(r_0)[:3],
                                 dtype=self._dtype)
        # [num_targets, num_sources, max_num_paths * num_wedges_per_vertex, 2, 2]
        mat_t += d3 * r_n + d4 * r_0 

        # [num_targets, num_sources, max_num_paths * num_wedges_per_vertex, 2, 2]
        if skip_sf:
            mat_t = -mat_t
        else:
            mat_t *= -(spreading_factor)

        # [num_targets, num_sources, max_num_paths_ext, 2, 2]
        nans_bool = tf.math.is_nan(tf.math.real(mat_t))
        mat_t = tf.where(nans_bool, tf.zeros_like(mat_t, dtype=self._dtype), mat_t)

        # Coordinate transform
        theta_t = tf.repeat(theta_t, tensor_valid_vertices_2_wedges.shape[-1], axis=-1)
        phi_t = tf.repeat(phi_t, tensor_valid_vertices_2_wedges.shape[-1], axis=-1)
        theta_r = tf.repeat(theta_r, tensor_valid_vertices_2_wedges.shape[-1], axis=-1)
        phi_r = tf.repeat(phi_r, tensor_valid_vertices_2_wedges.shape[-1], axis=-1)

        mat_from_gcs = component_transform(
                            theta_hat(theta_t, phi_t), phi_hat(phi_t),
                            phi_prime_hat, beta_0_prime_hat)
        mat_from_gcs = tf.complex(mat_from_gcs, tf.zeros_like(mat_from_gcs))


        mat_to_gcs = component_transform(phi_hat_, beta_0_hat,
                                      theta_hat(theta_r, phi_r), phi_hat(phi_r))
        mat_to_gcs = tf.complex(mat_to_gcs, tf.zeros_like(mat_to_gcs))

        mat_t = tf.linalg.matmul(mat_t, mat_from_gcs)
        mat_t = tf.linalg.matmul(mat_to_gcs, mat_t)

        # [num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex, 2, 2]
        new_shape = tensor_valid_vertices_2_wedges.shape.as_list() + [2, 2]
        mat_t = tf.reshape(mat_t, new_shape)
        mat_t = tf.where(mask_wedges_in_vertex, mat_t, tf.zeros_like(mat_t))

        # [num_targets, num_sources, max_num_paths, max_num_wedges_per_vertex]
        _s_prime = tf.reshape(s_prime, [num_targets, num_sources, num_vertices, num_wedges_per_vertex])
        self._mat_t = mat_t / tf.complex(_s_prime[..., None, None], tf.zeros_like(_s_prime[..., None, None]))
        self._wedge_idxs = tf.reshape(valid_vertex_wedges_idx, [num_targets, num_sources, num_vertices, num_wedges_per_vertex])

        # Finally, sum over the wedges for each vertex
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = tf.reduce_sum(mat_t, axis=3)

        # Set invalid paths to 0
        # Expand masks to broadcast with the field components
        # [num_targets, num_sources, max_num_paths, 1, 1]
        mask_ = expand_to_rank(mask, 5, axis=3)
        # Zeroing coefficients corresponding to non-valid paths
        # [num_targets, num_sources, max_num_paths, 2]
        mat_t = tf.where(mask_, mat_t, tf.zeros_like(mat_t))

        return mat_t
    
    def simple_compute_fields(self, phi_prime, phi, beta_prime, beta, s_prime, s, wedge_n, mask_wedges_in_vertex_, 
                              ell=None, coef_ell=None, new_coef_ell=None, skip_sf=False, return_more=False, 
                              step_1=None, step2=None):
        r"""
        Simple function, doesn't involve materials (PEC assumed)
        Trues in step_1 (or 2) ignore step_func1 (or 2)
        """
        eps_phi = 1e-4
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
        
        if step_1 is not None:
            step_func1 = tf.logical_or(step_func1, step_1)
        if step2 is not None:
            step_func2 = tf.logical_or(step_func2, step2)
        step_func = tf.logical_and(step_func1, step_func2)

        step_func = tf.cast(step_func, self._dtype)

        coef = coef_ * step_func

        if ell is None:
            ell = s_prime * s / (s_prime + s) * tf.math.sin(beta_prime) * tf.math.sin(beta)

        if new_coef_ell is not None:
            if tf.rank(new_coef_ell) > tf.rank(ell):
                ell = ell[..., None]
                phi_prime = phi_prime[..., None]
                phi = phi[..., None]
                beta_prime = beta_prime[..., None]
                beta = beta[..., None]
                wedge_n = wedge_n[..., None]
                s_prime = s_prime[..., None]
                s = s[..., None]
                coef = coef[..., None]
                mask_wedges_in_vertex_ = insert_dim_num(mask_wedges_in_vertex_, -1, new_coef_ell.shape[-1])

            ell = ell * new_coef_ell
            param_b = self._get_param_b(k, s_prime, s, beta_prime, beta, new_coef_ell)
        else:
            param_b = self._get_param_b(k, s_prime, s, beta_prime, beta)

        # [num_targets, num_sources, max_num_paths]
        param_u = self._get_param_u(beta_prime, beta)

        res, func_B, tmp = self._get_coefs_D2(wedge_n, phi_prime, phi, k, ell, param_b, param_u)
        d1_, d2_, d3_, d4_ = res

        # [num_targets, num_sources, max_num_paths, ...]
        d1 = _clear_diff_coef2(coef*d1_, mask_wedges_in_vertex_, num_targets, num_sources, num_vertices, self._dtype)
        d2 = _clear_diff_coef2(coef*d2_, mask_wedges_in_vertex_, num_targets, num_sources, num_vertices, self._dtype)
        d3 = _clear_diff_coef2(coef*d3_, mask_wedges_in_vertex_, num_targets, num_sources, num_vertices, self._dtype)
        d4 = _clear_diff_coef2(coef*d4_, mask_wedges_in_vertex_, num_targets, num_sources, num_vertices, self._dtype)

        d_soft = d1 + d2 - d3 - d4
        d_hard = d1 + d2 + d3 + d4
        _mat_t1 = tf.stack([d_hard, tf.zeros_like(d_hard)], axis=-1)
        _mat_t2 = tf.stack([tf.zeros_like(d_soft), d_soft], axis=-1)
        mat_t = tf.stack([_mat_t1, _mat_t2], axis=-2)

        # [num_targets, num_sources, max_num_paths * num_wedges_per_vertex]
        spreading_factor = 1.0 / s
        spreading_factor = tf.complex(spreading_factor, tf.zeros_like(spreading_factor))

        # [num_targets, num_sources, max_num_paths * num_wedges_per_vertex, 2, 2]
        if skip_sf:
            mat_t = -mat_t
        else:
            mat_t *= -spreading_factor[... , None, None]

        # [num_targets, num_sources, max_num_paths_ext, 2, 2]
        nans_bool = tf.math.is_nan(tf.math.real(mat_t))
        mat_t = tf.where(nans_bool, tf.zeros_like(mat_t, dtype=self._dtype), mat_t)

        if coef_ell is not None:
            _n = tf.cast(2*wedge_n, self._dtype)
            all_func_B = tf.stack([func_B[0]*_n, func_B[1]*_n, func_B[2]*_n, func_B[3]*_n], axis=-1)

            max_func_B = tf.reduce_max(tf.abs(all_func_B), axis=-1)
            max_func_B = tf.reshape(max_func_B, mask_wedges_in_vertex_.shape)
            new_coef_ell = get_coef_ell(max_func_B, coef_ell[..., None], 2.0, 400.0)

            return mat_t, new_coef_ell 

        if return_more:
            return mat_t, tmp
        
        return mat_t
        
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
    
    def slope(self, phi_prime, phi, beta_prime, beta, s_prime, s, l, wedge_n, mask_wedges_in_vertex_, is_der_prime=True):
        # wavelength = self.solver_paths._scene.wavelength
        # k = 2.*PI/wavelength
        # num_targets = phi_prime.shape[0]
        # num_sources = phi_prime.shape[1]
        # num_vertices = mask_wedges_in_vertex_.shape[2]

        phi_eps = 1e-3
        if is_der_prime:
            phi_prime2 = phi_prime + phi_eps
            phi2 = phi
        else:
            phi_prime2 = phi_prime
            phi2 = phi + phi_eps

        _mat_t1 = self.simple_compute_fields(phi_prime, phi, beta_prime, beta, s_prime, s, wedge_n, mask_wedges_in_vertex_)
        _mat_t2 = self.simple_compute_fields(phi_prime2, phi2, beta_prime, beta, s_prime, s, wedge_n, mask_wedges_in_vertex_)
        delta_mat_t = _mat_t2 - _mat_t1
        #delta_mat_t = delta_mat_t / tf.complex(l, tf.zeros_like(l))[..., None, None]

        return delta_mat_t / phi_eps

    def _get_param_u(self, beta1, beta2):  # Rubinowicz parameter
        r"""beta1, beta2 - elevation of incidence and diffraction"""

        tmp1 = tf.math.log(tf.math.tan(beta2 / 2))
        tmp2 = tf.math.log(tf.math.tan(beta1 / 2))

        return tmp1 - tmp2
    
    def _get_func_B(self, Phi, u, n):
        tmp1 = -1 / (2*n) * tf.math.sin(Phi / n)
        tmp2 = tf.math.cos(Phi / n) - tf.math.cosh(u / n)

        return tmp1 / tmp2
    
    def _get_param_b(self, k, r1, r2, beta1, beta2, coef_ell=None):
        ell = r1 * r2 / (r1 + r2)

        if coef_ell is not None:
            ell = coef_ell * ell

        tmp2 = 1 - tf.math.cos(beta2 - beta1)

        return k * ell * tmp2

    def _get_coefs_D2(self, wedge_n, phi1_rad, phi2_rad, k, ell, param_b, param_u):
        # combine param_b and a and find gfi first

        phi_m = phi2_rad - phi1_rad
        phi_p = phi2_rad + phi1_rad

        a_d1 = self._get_tip_a_plus(phi2_rad - phi1_rad, wedge_n, k, ell)
        a_d2 = self._get_tip_a_minus(phi2_rad - phi1_rad, wedge_n, k, ell)
        a_d3 = self._get_tip_a_plus(phi2_rad + phi1_rad, wedge_n, k, ell)
        a_d4 = self._get_tip_a_minus(phi2_rad + phi1_rad, wedge_n, k, ell)

        # [4, num_targets, num_sources, max_num_paths]
        _param_a = tf.stack([a_d1, a_d2, a_d3, a_d4], axis=0)
        _param_b = tf.repeat(param_b[None, ...], 4, axis=0)
        _t_gfi = t_gfi_full_approximated_capolino(_param_b, _param_a, self.obj_geom.is_table_gfi)

        D1, func_B1 = self._coef_D_plus2(wedge_n, phi2_rad - phi1_rad, param_u, _t_gfi[0])
        D2, func_B2 = self._coef_D_minus2(wedge_n, phi2_rad - phi1_rad, param_u, _t_gfi[1])
        D3, func_B3 = self._coef_D_plus2(wedge_n, phi2_rad + phi1_rad, param_u, _t_gfi[2])
        D4, func_B4 = self._coef_D_minus2(wedge_n, phi2_rad + phi1_rad, param_u, _t_gfi[3])

        return (D1, D2, D3, D4), (func_B1, func_B2, func_B3, func_B4), (param_b, a_d1, a_d2, a_d3, a_d4, _t_gfi, phi1_rad, phi2_rad, param_u)
    
    def _coef_D_plus2(self, wedge_n, Phi, param_u, t_gfi):
        func_B = tf.complex(self._get_func_B(PI + Phi, param_u, wedge_n), tf.cast(0.0, self._rdtype))

        return t_gfi * func_B, func_B
    
    def _coef_D_minus2(self, wedge_n, Phi, param_u, t_gfi):
        func_B = tf.complex(self._get_func_B(PI - Phi, param_u, wedge_n), tf.cast(0.0, self._rdtype))

        return t_gfi * func_B, func_B
    
    def _get_tip_a_plus(self, Phi, n, k, ell):
        tmp = 1 + tf.math.cos(Phi - 2 * PI * n * self._get_N_plus(Phi, n))

        return k * ell * tmp

    def _get_tip_a_minus(self, Phi, n, k, ell):
        tmp = 1 + tf.math.cos(Phi - 2 * PI * n * self._get_N_minus(Phi, n))

        return k * ell * tmp

    def _get_N_plus(self, Phi, n):
        return tf.math.round((Phi + PI) / (2*n*PI))

    def _get_N_minus(self, Phi, n):
        return tf.math.round((Phi - PI) / (2*n*PI))



    # def get_vertices(self, vertices, faces, primitives_2_wedges, num_vertices) -> tf.RaggedTensor:
    #     face_adjacency, adj_edges = trimesh.graph.face_adjacency(faces, return_edges=True)
    #     vertex_wedges = [[] for _ in range(num_vertices)]
    #     #self.wedge_2_vertices = tf.constant(adj_edges, dtype=tf.int32)

    #     # [num_wedges, 2]
    #     #self.wedge_2_primitives = tf.zeros((len(adj_edges), 2), dtype=tf.int32)

    #     for i in range(len(face_adjacency)):
    #         vertex1_idx, vertex2_idx = adj_edges[i][0], adj_edges[i][1]
    #         face1_idx, face2_idx = face_adjacency[i][0], face_adjacency[i][1]
            
    #         common_wedge_idx = set(primitives_2_wedges[face1_idx].numpy()).intersection(set(primitives_2_wedges[face2_idx].numpy()))
       
    #         if len(common_wedge_idx) == 0: 
    #             print("ERROR No common:", primitives_2_wedges[face1_idx].numpy(), primitives_2_wedges[face2_idx].numpy())
    #             continue

    #         common_wedge_idx = list(common_wedge_idx)[0]

    #         if common_wedge_idx == -1:
    #             continue

    #         #self.solver_paths._wedges_origin[common_wedge_idx]

    #         vertex_wedges[vertex1_idx].append(common_wedge_idx)
    #         vertex_wedges[vertex2_idx].append(common_wedge_idx)
    #     return vertex_wedges



def _clear_diff_coef(d, mask_wedges_in_vertex_, num_targets, num_sources, num_vertices, dtype):
    res = np.reshape(d, [num_targets, num_sources, num_vertices, -1])
    res = np.where(mask_wedges_in_vertex_, res, np.zeros_like(res))
    nans_bool = tf.math.is_nan(tf.math.real(res))
    res = tf.where(nans_bool, tf.zeros_like(res, dtype=dtype), res)
    res = np.reshape(res, [num_targets, num_sources, -1])
    #res = tf.reduce_sum(res, axis=3)

    return res

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
    #res = tf.reduce_sum(res, axis=3)

    return res