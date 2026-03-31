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
from .paths import Paths
from .diffraction_funcs import func_G_approximated_capolino, calc_angles, invert_angles,\
    my_wd_compute_fields, transition_func
from .utils import dot, phi_hat, theta_hat, theta_phi_from_unit_vec,\
    normalize, moller_trumbore, component_transform, mi_to_tf_tensor,\
        compute_field_unit_vectors, reflection_coefficient, fibonacci_lattice,\
            cot, cross, sign, rotation_matrix, acos_diff, rot_mat_from_unit_vecs, \
            r_hat

if TYPE_CHECKING:
    from .solver_paths import SolverPaths
    from .solver_paths import PathsTmpData

K_MAX = 100


class ObjectsGeometry:
    def __init__(self, solver_paths: SolverPaths):
        self.solver_paths = solver_paths
        self._dtype = self.solver_paths._dtype   # complex
        self._rdtype = self.solver_paths._rdtype  # float

        self.is_vertex_diffraction = True
        self.is_double_diffraction = False
        self.is_triple_diffraction = False
        self.add_obj_primitives = True
        self.is_sbr_rxs = True

        self.is_table_gfi = True

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
        self._primitives_2_objects = self.solver_paths._primitives_2_objects

    def to_dict(self):
        # pylint: disable=line-too-long
        r"""
        Returns the properties of the paths as a dictionary which values are
        tensors

        Output
        -------
        : `dict`
        """
        members_names = ['is_vertex_diffraction', 
                         'is_double_diffraction', 
                         'is_table_gfi',
                         'ids',
                         'ids_sionna',
                         'obj_names',
                         '_objects_2_primitives',
                         '_primitives_2_wedges',
                         'facet_2_adj_facets',
                        'wedge_2_facets',
                        'wedge_2_vertices',
                        'vertex_2_wedges',
                        'wedge_2_adj_wedges',
                        'dd_wedges',
                        'dd_wedge_vertex',
                        'vertices_2_objects',
                        'vertices_2_objects_sionna',
                        'dd_wedges_2_objects',
                        'dd_wedges_2_objects_sionna',
                        'all_facets',
                        'primitive_idxs',
                        '_primitives_2_objects',
                        '_wedges_objects']
        
        members_objects = [getattr(self, attr) for attr in members_names]
        data = {attr_name : attr_obj for (attr_obj, attr_name)
                in zip(members_objects, members_names)}
        
        return data
    
    def from_dict(self, data_dict):
        # pylint: disable=line-too-long
        r"""
        Set the paths from a dictionary which values are tensors

        The format of the dictionary is expected to be the same as the one
        returned by :meth:`~sionna.rt.Paths.to_dict()`.

        Input
        ------
        data_dict : `dict`
        """
        for attr_name in data_dict:
            attr_obj = data_dict[attr_name]
            setattr(self, '_' + attr_name, attr_obj)

    def init_objects(self, object_names, in_object_filenames=None):
        ids = [self.solver_paths._scene.objects[object_name].obj_id_vd for object_name in object_names]
        ids_sionna = [self.solver_paths._scene.objects[object_name].object_id for object_name in object_names]

        vertices = tf.ones([0, 3], dtype=self._rdtype)
        facet_2_adj_facets = tf.ones([0, 3, 2], dtype=tf.int32)
        wedge_2_facets = tf.ones([0, 2], dtype=tf.int32)
        wedge_2_vertices = tf.ones([0, 2], dtype=tf.int32)
        vertex_2_wedges = []
        wedge_2_adj_wedges = []
        dd_wedges = tf.ones([0, 2], dtype=tf.int32)
        dd_wedge_vertex = tf.ones([0, 2], dtype=tf.int32)
        vertices_2_objects = []
        vertices_2_objects_sionna = []
        dd_wedges_2_objects = []
        dd_wedges_2_objects_sionna = []
        all_facets = tf.ones([0, 3], dtype=tf.int32)
        primitive_idxs = tf.ones([0], dtype=tf.int32)

        for i, obj_id in enumerate(ids):
            primitives_id_start = self._objects_2_primitives[obj_id] # inclusive
            primitives_id_end = self._objects_2_primitives[obj_id+1] # exclusive
            primitive_idxs = tf.concat([primitive_idxs, tf.range(primitives_id_start, primitives_id_end, dtype=tf.int32)], axis=0)
            # [num_facets, 3]
            facet_2_wedges = tf.gather(self._primitives_2_wedges, tf.range(primitives_id_start, primitives_id_end), axis=0)

            num_facets = self.solver_paths._mi_scene.shapes()[obj_id].face_count()
            num_vertices = self.solver_paths._mi_scene.shapes()[obj_id].vertex_count()

            _vertices = self.solver_paths._mi_scene.shapes()[obj_id].vertex_position(dr.arange(mi.UInt32, num_vertices))
            _vertices = mi_to_tf_tensor(_vertices, dtype=self._rdtype)
            vertices = tf.concat([vertices, _vertices], axis=0)
            facets = np.array(self.solver_paths._mi_scene.shapes()[obj_id].face_indices(dr.arange(mi.UInt32, num_facets)))
            all_facets = tf.concat([all_facets, tf.convert_to_tensor(facets, dtype=tf.int32)], axis=0)

            # 1) load structures from the object file
            if in_object_filenames is not None and in_object_filenames[i] is not None:
                obj_filename = in_object_filenames[i]
                with open(obj_filename, "rb") as f:
                    _vertex_2_wedges, _wedge_2_facets, _wedge_2_vertices, _facet_2_adj_facets, _wedge_2_adj_wedges, \
                        _dd_wedges, _dd_wedge_vertex = pickle.load(f)

            # 2) compute structures
            else:
                _vertex_2_wedges, _wedge_2_facets, _wedge_2_vertices, _facet_2_adj_facets, _wedge_2_adj_wedges = \
                                    get_data_types(_vertices, facets, facet_2_wedges)  # + facet_2_wedges

            if self.is_double_diffraction:
                if in_object_filenames is None or in_object_filenames[i] is None:
                    _dd_wedges = self._get_dd_wedges(facet_2_wedges, _facet_2_adj_facets, _wedge_2_facets)
                    _dd_wedge_vertex = self._get_dd_edge_vertex(facet_2_wedges, facets, _wedge_2_vertices)

                dd_wedges = tf.concat([dd_wedges, _dd_wedges], axis=0)
                dd_wedge_vertex = tf.concat([dd_wedge_vertex, _dd_wedge_vertex], axis=0)

            if not len(_vertices) == num_vertices == len(_vertex_2_wedges):
                print("Wrong number of vertices in a mesh:", object_names[np.where(np.array(ids) == obj_id)[0][0]])

            vertex_2_wedges += _vertex_2_wedges
            wedge_2_adj_wedges += _wedge_2_adj_wedges
            wedge_2_facets = tf.concat([wedge_2_facets, _wedge_2_facets], axis=0)
            wedge_2_vertices = tf.concat([wedge_2_vertices, _wedge_2_vertices], axis=0)

            vertices_2_objects += [obj_id]*num_vertices 
            vertices_2_objects_sionna += [ids_sionna[i]]*num_vertices

        # [num_vertices, 3]
        self._vertices = vertices
        # [num_vertices, num_wedges_per_vertex]
        self.vertex_2_wedges = tf.ragged.constant(vertex_2_wedges, dtype=tf.int32)
        # [num_wedges, num_adj_wedges]
        self.wedge_2_adj_wedges = tf.ragged.constant(wedge_2_adj_wedges, dtype=tf.int32)
        # [num_wedges, 2]
        self.wedge_2_facets = wedge_2_facets
        # [num_facets, 3, 2]
        self.facet_2_adj_facets = facet_2_adj_facets
        # [num_wedges, 2]
        self.wedge_2_vertices = wedge_2_vertices
        # [num_dd_wedges, 2]
        self.dd_wedges = dd_wedges
        # [num_dd_wedge_vertex, 2]
        self.dd_wedge_vertex = dd_wedge_vertex
        # [num_dd_wedges]
        self.dd_wedges_2_objects = tf.convert_to_tensor(dd_wedges_2_objects, dtype=tf.int32)
        self.dd_wedges_2_objects_sionna = tf.convert_to_tensor(dd_wedges_2_objects_sionna, dtype=tf.int32)
        # [num_vertices]
        self.vertices_2_objects = tf.convert_to_tensor(vertices_2_objects, dtype=tf.int32)
        self.vertices_2_objects_sionna = tf.convert_to_tensor(vertices_2_objects_sionna, dtype=tf.int32)
        # [num_objects+1]
        self._objects_2_primitives = tf.convert_to_tensor(self._objects_2_primitives, dtype=tf.int32)
        # [num_facets, 3]
        self.all_facets = all_facets
        # [num_facets]
        self._primitive_idxs = primitive_idxs
        # [num_objects]
        self.ids = ids
        self.ids_sionna = ids_sionna
        self.obj_names = object_names

        self.active_objects = tf.fill([len(object_names)], True)
        self.active_vertices = tf.fill([self._vertices.shape[0]], True)
        self.active_primitives = tf.fill([self._primitive_idxs.shape[0]], True)
        
        self.active_objects_dd = tf.fill([len(object_names)], True)
        self.active_vertices_dd = tf.fill([self._vertices.shape[0]], True)
        self.active_primitives_dd = tf.fill([self._primitive_idxs.shape[0]], True)

        print('num vertices:', self._vertices.shape[0])

    def set_active_objects_dd(self, object_names):
        self.active_objects_dd = tf.fill([len(self.ids)], False)

        active_ids = [self.solver_paths._scene.objects[object_name].obj_id_vd for object_name in object_names]
        active_ids = tf.convert_to_tensor(active_ids, dtype=tf.int32)

        ids = tf.convert_to_tensor(self.ids, dtype=tf.int32)
        self.active_objects_dd = tf.reduce_any(ids[None, ...] == active_ids[..., None], axis=0)

        active_ids_sionna = tf.boolean_mask(self.ids_sionna, self.active_objects_dd)
        self.active_vertices_dd = tf.reduce_any(self.vertices_2_objects_sionna[..., None] == active_ids_sionna[None, ...], axis=-1)

        primitive_objects = tf.gather(self._primitives_2_objects, self._primitive_idxs)
        self.active_primitives_dd = tf.reduce_any(primitive_objects[..., None] == active_ids_sionna[None, ...], axis=-1)

    def set_inactive_objects(self, object_names):
        self.active_objects = tf.fill([len(self.ids)], True)
        inactive_ids = [self.solver_paths._scene.objects[object_name].obj_id_vd for object_name in object_names]
        inactive_ids = tf.convert_to_tensor(inactive_ids, dtype=tf.int32)
        find_match = tf.convert_to_tensor(self.ids)[None, ...] - inactive_ids[..., None]
        self.active_objects = tf.logical_not(tf.reduce_any(find_match == 0, axis=0))

        if tf.reduce_all(self.active_objects):
            self.active_vertices = tf.fill([self._vertices.shape[0]], True)
            self.active_primitives = tf.fill([self._primitives_2_wedges.shape[0]], True)

        else:
            active_ids_sionna = tf.boolean_mask(self.ids_sionna, self.active_objects)
            self.active_vertices = tf.reduce_any(self.vertices_2_objects_sionna[..., None] == active_ids_sionna[None, ...], axis=-1)

            primitive_objects = tf.gather(self._primitives_2_objects, self._primitive_idxs)
            self.active_primitives = tf.reduce_any(primitive_objects[..., None] == active_ids_sionna[None, ...], axis=-1)

    @property
    def primitive_idxs(self):
        return tf.boolean_mask(self._primitive_idxs, self.active_primitives)
    
    @property
    def primitive_idxs_dd(self):
        return tf.boolean_mask(self._primitive_idxs, self.active_primitives_dd)
    
    @property
    def vertex_2_wedges_dd(self):
        return tf.boolean_mask(self._vertex_2_wedges, self.active_vertices_dd)
    

    def save_object(self, obj_name, out_filename):
        obj_id = self.solver_paths._scene.objects[obj_name].obj_id_vd

        primitives_id_start = self._objects_2_primitives[obj_id] # inclusive
        primitives_id_end = self._objects_2_primitives[obj_id+1] # exclusive
        # [num_facets, 3]
        facet_2_wedges = tf.gather(self._primitives_2_wedges, tf.range(primitives_id_start, primitives_id_end), axis=0)

        num_facets = self.solver_paths._mi_scene.shapes()[obj_id].face_count()
        num_vertices = self.solver_paths._mi_scene.shapes()[obj_id].vertex_count()

        _vertices = self.solver_paths._mi_scene.shapes()[obj_id].vertex_position(dr.arange(mi.UInt32, num_vertices))
        _vertices = mi_to_tf_tensor(_vertices, dtype=self._rdtype)
        _facets = np.array(self.solver_paths._mi_scene.shapes()[obj_id].face_indices(dr.arange(mi.UInt32, num_facets)))

        _vertex_2_wedges, _wedge_2_facets, _wedge_2_vertices, _facet_2_adj_facets, _wedge_2_adj_wedges = \
                            get_data_types(_vertices, _facets, facet_2_wedges)

        if self.is_double_diffraction:
            _dd_wedges = self._get_dd_wedges(facet_2_wedges, _facet_2_adj_facets, _wedge_2_facets)
            _dd_wedge_vertex = self._get_dd_edge_vertex(facet_2_wedges, _facets, _wedge_2_vertices)
            with open(out_filename, "wb") as f:
                pickle.dump((_vertex_2_wedges, _wedge_2_facets, _wedge_2_vertices, _facet_2_adj_facets, _wedge_2_adj_wedges,
                            _dd_wedges, _dd_wedge_vertex), f)
        
        else:
            with open(out_filename, "wb") as f:
                pickle.dump((_vertex_2_wedges, _wedge_2_facets, _wedge_2_vertices, _facet_2_adj_facets, _wedge_2_adj_wedges,
                             None, None), f)

    def _get_dd_edge_vertex(self, facets_2_wedges, facets, wedge_2_vertices):
        # TODO : make all in tf
        edge_vertex = []
        for facet_idx in range(len(facets_2_wedges)):
            e1, e2, e3 = facets_2_wedges[facet_idx].numpy()  # idxs
            v1, v2, v3 = facets[facet_idx]  # idxs

            for e in [e1, e2, e3]:
                if e == -1:
                    continue

                v = set([v1, v2, v3]) - set(wedge_2_vertices[e].numpy())
                if len(v) != 1:
                    pass
                if len(v) != 1 or list(v)[0] == -1:
                    continue

                edge_vertex.append([e, list(v)[0]])
        
        return tf.convert_to_tensor(edge_vertex, dtype=tf.int32)
    
    def _get_dd_wedges(self, facet_2_wedges, facet_2_adj_facets, wedge_2_facets):
        def _routine(next_facet):
            next_facets = [next_facet]
            passed_facets = []
            res = []
            k = 0
            for next_facet in next_facets:
                passed_facets.append(next_facet)
                k += 1
                if k >= K_MAX:
                    print('k:', k)
                    break
                adj_facets = facet_2_adj_facets[next_facet]
                for adj_facet_idx, adj_wedge_idx in adj_facets:
                    if adj_facet_idx in passed_facets:
                        continue
                    if adj_wedge_idx == -1:
                        next_facets.append(adj_facet_idx)
                    else:
                        res.append(adj_wedge_idx)

            return res

        dd_wedges = []
        for wedge_idx in range(wedge_2_facets.shape[0]):
            wedge_dd_wedges = []
            wedge2_candidate_idxs = []
            facet_idx1, facet_idx2 = tf.gather(wedge_2_facets, wedge_idx,axis=0).numpy()
            next_facets = [facet_idx1, facet_idx2]
            wedge2_candidate_idxs = []

            for next_facet in next_facets:
                wedge2_candidate_idxs += _routine(next_facet)

            for w2_candidate_idx in wedge2_candidate_idxs:
                if w2_candidate_idx != -1 and w2_candidate_idx != wedge_idx:
                    wedge_dd_wedges.append([wedge_idx, w2_candidate_idx])
                    wedge_dd_wedges.append([w2_candidate_idx, wedge_idx])
                    
            dd_wedges += wedge_dd_wedges

        return tf.convert_to_tensor(dd_wedges, dtype=tf.int32)
    

def get_data_types(vertices, faces, primitives_2_wedges): # for an object
    num_vertices = len(vertices)
    num_faces = len(faces)
    num_wedges = tf.math.reduce_max(primitives_2_wedges) + 1

    face_adjacency, adj_edges = trimesh.graph.face_adjacency(faces, return_edges=True)

    vertex_2_wedges = [[] for _ in range(num_vertices)]  # ragged
    wedge_2_facets = -1 * tf.ones([num_wedges, 2], dtype=tf.int32)
    wedge_2_vertices = -1 * tf.ones([num_wedges, 2], dtype=tf.int32)

    facet_2_adj_facets = [[] for _ in range(num_faces)]   # TODO: tf

    facet_central_point = tf.reduce_mean(tf.gather(vertices, tf.convert_to_tensor(faces, dtype=tf.int32)), axis=1)  # [num_faces, 3]
    count = 0 # will be equal to the number of wedges

    for i in range(len(face_adjacency)):
        vertex1_idx, vertex2_idx = adj_edges[i][0], adj_edges[i][1]
        face1_idx, face2_idx = face_adjacency[i][0], face_adjacency[i][1]
        
        _common_wedge_idx = set(primitives_2_wedges[face1_idx].numpy()).intersection(set(primitives_2_wedges[face2_idx].numpy()))
    
        if len(_common_wedge_idx) == 0: 
            print("ERROR No common:", primitives_2_wedges[face1_idx].numpy(), primitives_2_wedges[face2_idx].numpy())
            continue

        common_wedge_idx = max(list(_common_wedge_idx))

        facet_2_adj_facets[face1_idx].append([face2_idx, common_wedge_idx])  # [adj_facet_idx, wedge_idx]
        facet_2_adj_facets[face2_idx].append([face1_idx, common_wedge_idx])

        if common_wedge_idx == -1:
            continue

        count += 1
        wedge_2_facets = tf.tensor_scatter_nd_update(wedge_2_facets, [[common_wedge_idx, 0]], [face1_idx])
        wedge_2_facets = tf.tensor_scatter_nd_update(wedge_2_facets, [[common_wedge_idx, 1]], [face2_idx])

        vertex_2_wedges[vertex1_idx].append(common_wedge_idx)
        vertex_2_wedges[vertex2_idx].append(common_wedge_idx)

        wedge_2_vertices = tf.tensor_scatter_nd_update(wedge_2_vertices, [[common_wedge_idx, 0]], [vertex1_idx])
        wedge_2_vertices = tf.tensor_scatter_nd_update(wedge_2_vertices, [[common_wedge_idx, 1]], [vertex2_idx])

    wedge_2_adj_wedges = [[] for _ in range(num_wedges)]
    for vertex_idx in range(num_vertices):
        wedge_idxs = vertex_2_wedges[vertex_idx]
        for i in range(len(wedge_idxs)):
            for j in range(len(wedge_idxs)):
                if i == j:
                    continue

                wedge1_idx, wedge2_idx = wedge_idxs[i], wedge_idxs[j]
                if wedge1_idx == -1:
                    continue

                wedge_2_adj_wedges[wedge1_idx].append([wedge2_idx, vertex_idx])  # [adj_wedge_idx, vertex_idx]

    return vertex_2_wedges, wedge_2_facets, wedge_2_vertices, facet_2_adj_facets, wedge_2_adj_wedges