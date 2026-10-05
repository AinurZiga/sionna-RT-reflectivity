#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Class representing objects in the scene
"""
import tensorflow as tf
import time
import os

from .object import Object
from .radio_material import RadioMaterial
import drjit as dr
import mitsuba as mi
from . import scene as scene_module
from .utils import mi_to_tf_tensor, angles_to_mitsuba_rotation, normalize, rotation_matrix,\
    theta_phi_from_unit_vec
from sionna.constants import PI

NO_NAME_COUNTER = 0
class SceneObject(Object):
    # pylint: disable=line-too-long
    r"""
    SceneObject()

    Every object in the scene is implemented by an instance of this class
    """

    NO_NAME_COUNTER = 0

    def __init__(self,
                 name,
                 scene,
                 mi_shape=None,
                 fname=None,
                 object_id=None,
                 radio_material=None):

        # Initialize the base class Object
        #super().__init__(name)

        if mi_shape:
            if not isinstance(mi_shape, mi.Mesh):
                raise ValueError("`mi_shape` must a Mitsuba Shape object")
            
        elif fname:
            # Mesh type
            mesh_type = os.path.splitext(fname)[1][1:]
            if mesh_type not in ('ply', 'obj'):
                raise ValueError("Invalid mesh type."
                                 " Supported types: `ply` and `obj`")
            if not isinstance(fname, str):
                raise ValueError("The `name` of the object to instantiate must"
                                 " be an str")
            if not isinstance(radio_material, RadioMaterial):
                raise ValueError("The `radio_material` for the object to"
                                 " instantiate must be a RadioMaterialBase")

            mi_shape = mi.load_dict({'type': mesh_type,
                                     'id' : name,
                                     'filename': fname,
                                     'flip_normals': True,
                                    #  'bsdf' : {'type': 'holder-material',
                                    #            'id': f'mat-holder-{name}'}
                                     })
            #mi_shape.bsdf().radio_material = radio_material
        else:
            raise ValueError("Either a Mitsuba Shape (mi_shape) or a filename"
                             " (fname) must be provided")

        

        # Set the object id
        #self._object_id = object_id
        self._object_id = dr.reinterpret_array_v(mi.UInt32,
                                               mi.ShapePtr(mi_shape))[0]

        # Scene
        self._scene = scene

        # Set the radio material
        self.radio_material = radio_material

        # Set the Mitsuba shape
        self._mi_shape = mi_shape

        if mi_shape.id() == "":
            SceneObject.NO_NAME_COUNTER += 1
            name = f"no-name-{SceneObject.NO_NAME_COUNTER}"
            mi_shape.set_id(name)

        # Increment the material counter of objects
        #self.radio_material.add_object()

        # Set velocity vector
        self.velocity = tf.cast([0,0,0], dtype=scene.dtype.real_dtype)

        # Orientation of the object is initialized to (0,0,0)
        self._orientation = tf.cast([0.,0.,0.], dtype=scene.dtype.real_dtype)


        ## rotational motion
        self.main_object_id = tf.cast(-1, dtype=tf.int32)
        self.point_on_axis = tf.cast([0.,0.,0.], dtype=scene.dtype.real_dtype)
        self.axis = tf.cast([0.,0.,-1.], dtype=scene.dtype.real_dtype)
        self.angular_velocity = tf.cast(0., dtype=scene.dtype.real_dtype)


        if scene.dtype == tf.complex64:
            self._mi_point_t = mi.Point3f
            self._mi_vec_t = mi.Vector3f
            self._mi_scalar_t = mi.Float
            self._mi_transform_t = mi.Transform4f
        else:
            self._mi_point_t = mi.Point3d
            self._mi_vec_t = mi.Vector3d
            self._mi_scalar_t = mi.Float64
            self._mi_transform_t = mi.Transform4d

    @property
    def scene(self):
        """
        Get/set the scene to which the object belongs. Note that the scene can
        only be set once.

        :type: :py:class:`sionna.rt.Scene`
        """
        return self._scene

    @scene.setter
    def scene(self, scene: scene_module):
        if not isinstance(scene, scene_module.Scene):
            raise ValueError("`scene` must be an instance of Scene")
        if (self._scene is not None) and (self._scene is not scene):
            msg = f"Radio material ('{self.name}') already used by another "\
                "scene"
            raise ValueError(msg)
        self._scene = scene

    @staticmethod
    def shape_id_to_name(shape_id):
        name = shape_id
        if shape_id.startswith("mesh-"):
            name = shape_id[5:]
        return name

    @property
    def name(self):
        r"""Name

        :type: :py:class:`str`
        """
        return SceneObject.shape_id_to_name(self._mi_shape.id())
    
    @property
    def object_id(self):
        r"""
        int : Return the identifier of this object
        """
        return self._object_id
    
    @property
    def mi_shape(self):
        r"""Get/set the Mitsuba shape

        :type: :py:class:`mi.Mesh`
        """
        return self._mi_shape

    @mi_shape.setter
    def mi_shape(self, v: mi.Mesh):
        self._mi_shape = v

    @property
    def radio_material(self):
        r"""
        :class:`~sionna.rt.RadioMaterial` : Get/set the radio material of the
        object. Setting can be done by using either an instance of
        :class:`~sionna.rt.RadioMaterial` or the material name (`str`).
        If the radio material is not part of the scene, it will be added. This
        can raise an error if a different radio material with the same name was
        already added to the scene.
        """
        return self._radio_material

    @radio_material.setter
    def radio_material(self, mat):
        # Note: _radio_material is set at __init__, but pylint doesn't see it.
        if mat is None:
            mat_obj = None

        elif isinstance(mat, str):
            mat_obj = self.scene.get(mat)
            if (mat_obj is None) or (not isinstance(mat_obj, RadioMaterial)):
                err_msg = f"Unknown radio material '{mat}'"
                raise TypeError(err_msg)

        elif not isinstance(mat, RadioMaterial):
            err_msg = ("The material must be a material name (str) or an "
                        "instance of RadioMaterial")
            raise TypeError(err_msg)

        else:
            mat_obj = mat

        # Remove the object from the set of the currently used material, if any
        # pylint: disable=access-member-before-definition
        if hasattr(self, '_radio_material') and self._radio_material:
            self._radio_material.discard_object_using(self.object_id)
        # Assign the new material
        # pylint: disable=access-member-before-definition
        self._radio_material = mat_obj

        # If the radio material is set to None, we can stop here
        # pylint: disable=access-member-before-definition
        if not self._radio_material:
            return

        # Add the object to the set of the newly used material
        # pylint: disable=access-member-before-definition
        self._radio_material.add_object_using(self.object_id)

        # Add the RadioMaterial to the scene if not already done
        self.scene.add(self._radio_material)

    @property
    def velocity(self):
        """
        [3], tf.float : Get/set the velocity vector [m/s]
        """
        return self._velocity

    @velocity.setter
    def velocity(self, v):
        if not tf.shape(v)==3:
            raise ValueError("`velocity` must have shape [3]")
        self._velocity = tf.cast(v, self._scene.dtype.real_dtype)

    @property
    def position(self):
        """
        [3], tf.float : Get/set the position vector [m] of the center
            of the object. The center is defined as the object's axis-aligned
            bounding box (AABB).
        """
        rdtype = self._scene.dtype.real_dtype
        # # Bounding box
        # # [3]
        # bbox_min = mi_to_tf_tensor(self._mi_shape.bbox().min, rdtype)
        # # [3]
        # bbox_max = mi_to_tf_tensor(self._mi_shape.bbox().max, rdtype)
        # # [3]
        # half = tf.cast(0.5, rdtype)
        # position = half*(bbox_min + bbox_max)

        position = tf.convert_to_tensor(self._mi_shape.bbox().center(), rdtype)

        return position

    @position.setter
    def position(self, new_position):
        return self._position(new_position)

    def _set_mesh_vertices(self, vertices, is_update=True):
        """Update the object's vertices in TensorFlow and Mitsuba."""
        solver = self._scene.solver_paths
        vertices = tf.cast(vertices, solver._rdtype)

        shapes = solver._mi_scene.shapes()
        shape_idx = int(solver._shape_indices[self.object_id])
        start = sum(shape.vertex_count() for shape in shapes[:shape_idx])
        indices = tf.range(start, start + self._mi_shape.vertex_count(), dtype=tf.int32)

        solver._vertices.scatter_nd_update(indices[:, None], vertices)

        params = self._scene.mi_scene_params
        params[f"{self._mi_shape.id()}.vertex_positions"] = self._mi_scalar_t(tf.reshape(vertices, [-1]))

        if is_update:
            params.update()
            self._refresh_geometry()
            self._scene.scene_geometry_updated()

    def _refresh_geometry(self):
        """Refresh derived geometry and diffraction caches."""
        solver = self._scene.solver_paths
        solver._update_geometry()

        # DoubleDiffraction keeps references to derived geometry tensors.
        dd = solver.double_diffraction
        for name in (
            "_wedges_origin", "_wedges_normals", "_wedges_e_hat", "_wedges_objects",
            "_is_edge", "_wedges_length", "_primitives_2_wedges",
        ):
            setattr(dd, name, getattr(solver, name))
        dd._facet_normals = solver._normals
        dd.dd_pairs_coplanar = None

        obj_geom = solver.obj_geom
        if obj_geom is not None:
            obj_geom._update_geometry()
            if solver.vertex_diffraction.is_setup:
                solver.vertex_diffraction.set_objects_geometry(obj_geom)
            dd.set_objects_geometry(obj_geom)

    def _position(self, new_position, is_update=True):
        rdtype = self._scene.dtype.real_dtype
        new_position = tf.cast(new_position, rdtype)

        params = self._scene.mi_scene_params
        key = f"{self._mi_shape.id()}.vertex_positions"
        vertices = tf.reshape(mi_to_tf_tensor(params[key], rdtype), [-1, 3])
        vertices = vertices + (new_position - self.position)

        self._set_mesh_vertices(vertices, is_update)

    def _rotate(self, new_orient, origin=None, is_update=True):
        rdtype = self._scene.dtype.real_dtype
        new_orient = tf.cast(new_orient, rdtype)
        origin = self.position if origin is None else tf.cast(origin, rdtype)

        params = self._scene.mi_scene_params
        key = f"{self._mi_shape.id()}.vertex_positions"
        vertices = tf.reshape(mi_to_tf_tensor(params[key], rdtype), [-1, 3])

        new_rotation = rotation_matrix(new_orient)
        old_rotation = rotation_matrix(self._orientation)
        rotation = tf.linalg.matmul(new_rotation, old_rotation, transpose_b=True)
        vertices = tf.linalg.matmul(vertices - origin, rotation, transpose_b=True) + origin

        self._orientation = new_orient
        self._set_mesh_vertices(vertices, is_update)

    @property
    def orientation(self):
        r"""
        [3], tf.float : Get/set the orientation :math:`(\alpha, \beta, \gamma)`
            [rad] specified through three angles corresponding to a
            3D rotation as defined in :eq:`rotation`.
        """
        return self._orientation

    @orientation.setter
    def orientation(self, new_orient):
        return self._rotate(new_orient)

    def look_at(self, target):
        # pylint: disable=line-too-long
        r"""
        Sets the orientation so that the x-axis points toward an
        ``Object``.

        Input
        -----
        target : [3], float | :class:`sionna.rt.Object` | str
            A position or the name or instance of an
            :class:`sionna.rt.Object` in the scene to point toward to
        """
        # Get position to look at
        if isinstance(target, str):
            obj = self.scene.get(target)
            if not isinstance(obj, Object):
                raise ValueError(f"No camera, device, or object named '{target}' found.")
            else:
                target = obj.position
        elif isinstance(target, Object):
            target = target.position
        else:
            target = tf.cast(target, dtype=self._rdtype)
            if not target.shape[0]==3:
                raise ValueError("`target` must be a three-element vector)")

        # Compute angles relative to LCS
        x = target - self.position
        x, _ = normalize(x)
        theta, phi = theta_phi_from_unit_vec(x)
        alpha = phi # Rotation around z-axis
        beta = theta-PI/2 # Rotation around y-axis
        gamma = 0.0 # Rotation around x-axis
        self.orientation = (alpha, beta, gamma)
