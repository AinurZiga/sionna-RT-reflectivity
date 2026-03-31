from __future__ import annotations

import mitsuba as mi
import drjit as dr
import tensorflow as tf
from typing import TYPE_CHECKING

from sionna.utils.tensors import expand_to_rank, insert_dims, flatten_dims,\
    split_dim
from ..paths import Paths, PathsTmpData
from ..utils import dot, phi_hat, theta_hat, theta_phi_from_unit_vec,\
    normalize, moller_trumbore, component_transform, mi_to_tf_tensor,\
        compute_field_unit_vectors, reflection_coefficient, fibonacci_lattice,\
            cot, cross, sign, rotation_matrix, acos_diff

if TYPE_CHECKING:
    from ..solver_base import SolverBase
    from ..solver_paths import SolverPaths

class Reflection:
    def __init__(self, solver: SolverPaths):
        self.solver = solver
        self._scene = solver._scene

        self._dtype = solver._dtype
        self._rdtype = solver._rdtype

    def get_images(self, sources, candidates):
        r"""Starting from the sources, mirror each point against the
        given candidate primitive. At this stage, we do not carry
        any verification about the visibility of the ray.
        Loop through the max_depth interactions. All candidate paths are
        processed in parallel.

        Input
        ------
        sources : [num_sources, 3], tf.float
            Positions of the sources from which rays (paths) are emitted

        candidates: [max_depth, num_samples], tf.int
            Set of candidate paths with depth up to ``max_depth``.

        Output
        -------
        
        mirrored_vertices : [max_depth, num_sources, num_samples, 3], tf.float
            Mirrored points coordinates
        
        tri_p0 : [max_depth, num_sources, num_samples, 3], tf.float
            Coordinates of the first vertex of potentially hitted triangles
        
        normals : [max_depth, num_sources, num_samples, 3], tf.float
            Normals to the potentially hitted triangles"""
        
        mirrored_vertices, tri_p0, normals =\
            self._spec_image_method_phase_1(candidates, sources)
        
        return mirrored_vertices, tri_p0, normals

    def tracing(self, sources, targets, los, reflection, candidates, mirrored_vertices, tri_p0, normals):
        r"""Traces the paths from the sources to the targets.

        Input
        ------
        sources : [num_sources, 3], tf.float
            Positions of the sources from which rays (paths) are emitted

        targets : [num_targets, 3], tf.float
            Positions of the targets to which rays (paths) are traced

        los : bool

        reflection : bool

        candidates: [max_depth, num_samples], tf.int
            Set of candidate paths with depth up to ``max_depth``.

        mirrored_vertices : [max_depth, num_sources, num_samples, 3], tf.float
            Mirrored points coordinates (source images)

        tri_p0 : [max_depth, num_sources, num_samples, 3], tf.float
            Coordinates of the first vertex of potentially hitted triangles

        normals : [max_depth, num_sources, num_samples, 3], tf.float
            Normals to the potentially hitted triangles

        Output
        ------
        spec_paths : :class:`~sionna.rt.Paths`
            Paths from sources to targets

        spec_paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Temporary data associated with the paths
        """
        
        spec_paths = Paths(sources=sources, targets=targets, scene=self._scene,
                           types=Paths.SPECULAR)
        spec_paths_tmp = PathsTmpData(sources, targets, self._dtype)

        if los or reflection:
            # Using the image method, computes the non-obstructed specular paths
            # interacting with the ``candidates`` primitives
            self._spec_image_method(candidates, spec_paths, spec_paths_tmp, mirrored_vertices, tri_p0, normals)

            # Compute paths length, delays, angles and directions of arrivals
            # and departures for the specular paths
            spec_paths, spec_paths_tmp =\
                self.solver._compute_directions_distances_delays_angles(spec_paths,
                                                        spec_paths_tmp, False)
        else:
            pass

        return spec_paths, spec_paths_tmp

    

    def _spec_image_method_phase_1(self, candidates, sources):
        r"""
        Implements the first phase of the image method.

        Starting from the sources, mirror each point against the
        given candidate primitive. At this stage, we do not carry
        any verification about the visibility of the ray.
        Loop through the max_depth interactions. All candidate paths are
        processed in parallel.

        Input
        ------
        candidates: [max_depth, num_samples], tf.int
            Set of candidate paths with depth up to ``max_depth``.
            For paths with depth lower than ``max_depth``, -1 must be used as
            padding value.
            The first path is the LoS one if LoS is requested.

        sources : [num_sources, 3], tf.float
            Positions of the sources from which rays (paths) are emitted

        Output
        -------
        mirrored_vertices : [max_depth, num_sources, num_samples, 3], tf.float
            Mirrored points coordinates

        tri_p0 : [max_depth, num_sources, num_samples, 3], tf.float
            Coordinates of the first vertex of potentially hitted triangles

        normals : [max_depth, num_sources, num_samples, 3], tf.float
            Normals to the potentially hitted triangles
        """

        # Max depth
        max_depth = candidates.shape[0]

        # Number of candidates
        num_samples = tf.shape(candidates)[1]

        # Number of sources and number of receivers
        num_sources = len(sources)

        # Sturctures are filled by the following loop
        # Indicates if a path is discarded
        # [num_samples]
        valid = tf.fill([num_samples], True)
        # Coordinates of the first vertex of potentially hitted triangles
        # [max_depth, num_sources, num_samples, 3]
        tri_p0 = tf.zeros([max_depth, num_sources, num_samples, 3],
                            dtype=self._rdtype)
        # Coordinates of the mirrored vertices
        # [max_depth, num_sources, num_samples, 3]
        mirrored_vertices = tf.zeros([max_depth, num_sources, num_samples, 3],
                                        dtype=self._rdtype)
        # Normals to the potentially hitted triangles
        # [max_depth, num_sources, num_samples, 3]
        normals = tf.zeros([max_depth, num_sources, num_samples, 3],
                           dtype=self._rdtype)

        # Position of the last interaction.
        # It is initialized with the sources position
        # Add an additional dimension for broadcasting with the paths
        # [num_sources, 1, xyz : 1]
        current = tf.expand_dims(sources, axis=1)
        current = tf.tile(current, [1, num_samples, 1])
        # Index of the last hit primitive
        prev_prim_idx = tf.fill([num_samples], -1)
        if max_depth > 0:
            for depth in tf.range(max_depth):

                # Primitive indices with which paths interact at this depth
                # [num_samples]
                prim_idx = tf.gather(candidates, depth, axis=0)

                # Flag indicating which paths are still active, i.e., should be
                # tested.
                # Paths that are shorter than depth are marked as inactive
                # [num_samples]
                active = tf.not_equal(prim_idx, -1)

                # Break the loop if no active paths
                # Could happen with empty scenes, where we have only LoS
                if tf.logical_not(tf.reduce_any(active)):
                    break

                # Eliminate paths that go through the same prim twice in a row
                # [num_samples]
                valid = tf.logical_and(
                    valid,
                    tf.logical_or(~active, tf.not_equal(prim_idx,prev_prim_idx))
                )

                # On CPU, indexing with -1 does not work. Hence we replace -1
                # by 0.
                # This makes no difference on the resulting paths as such paths
                # are not flagged as active.
                # valid_prim_idx = prim_idx
                valid_prim_idx = tf.where(prim_idx == -1, 0, prim_idx)

                # Mirroring of the current point with respected to the
                # potentially hitted triangle.
                # We need the coordinate of the first vertex of the potentially
                # hitted triangle.
                # To get this, we build the indexing tensor to gather only the
                # coordinate of the first index
                # [[num_samples, 1]]
                p0_index = tf.expand_dims(valid_prim_idx, axis=1)
                p0_index = tf.pad(p0_index, [[0,0], [0,1]], mode='CONSTANT',
                                    constant_values=0) # First vertex
                # [num_samples, xyz : 3]
                p0 = tf.gather_nd(self.solver._primitives, p0_index)
                # Expand rank and tile to broadcast with the number of
                # transmitters
                # [num_sources, num_samples, xyz : 3]
                p0 = tf.expand_dims(p0, axis=0)
                p0 = tf.tile(p0, [num_sources, 1, 1])
                # Gather normals to potentially intersected triangles
                # [num_samples, xyz : 3]
                normal = tf.gather(self.solver._normals, valid_prim_idx)
                # Expand rank and tile to broadcast with the number of
                # transmitters
                # [1, num_samples, xyz : 3]
                normal = tf.expand_dims(normal, axis=0)
                normal = tf.tile(normal, [num_sources, 1, 1])

                # Distance between the current intersection point (or sources)
                # and the plane the triangle is part of.
                # Note: `dist` is signed to compensate for backfacing normals
                # whenn needed.
                # [num_sources, num_samples, 1]
                dist = dot(current, normal, keepdim=True)\
                            - dot(p0, normal, keepdim=True)
                # Coordinates of the mirrored point
                # [num_sources, num_samples, xyz : 3]
                mirrored = current - 2. * dist * normal

                # Store these results
                # [max_depth, num_sources, num_samples, 3]
                mirrored_vertices = tf.tensor_scatter_nd_update(
                                    mirrored_vertices, [[depth]], [mirrored])
                # [max_depth, num_sources, num_samples, 3]
                tri_p0 = tf.tensor_scatter_nd_update(tri_p0, [[depth]], [p0])
                # [max_depth, num_sources, num_samples, 3]
                normals = tf.tensor_scatter_nd_update(normals,
                                                      [[depth]], [normal])


                # Prepare for the next interaction
                # [num_sources, num_samples, xyz : 3]
                current = mirrored
                # [num_samples]
                prev_prim_idx = prim_idx

        return mirrored_vertices, tri_p0, normals
    

    def _spec_image_method_phase_21(self, depth, candidates, valid,
                                    mirrored_vertices, tri_p0, normals, current,
                                    num_targets, num_sources):
        # pylint: disable=line-too-long
        r"""
        Implement the first part of phase 2 of the image method:

        For a given ``depth``:
        - Computes the intersection point with the ``depth``th primitive of the
        sequence of candidates for a ray originating from ``current``
        - Checks that the intersection point is within the primitive
        - Ensures the normal points toward the ``current`` point
        - Prepares the ray to test for blockage between ``current`` point and
        thecomputed intersection point

        The obstruction test is note performed in this function as it uses
        Mitsuba.

        Input
        -----
        depth : int
            Current interaction number

        candidates: [max_depth, num_samples], tf.int
            Set of candidate paths with depth up to ``max_depth``.
            For paths with depth lower than ``max_depth``, -1 must be used as
            padding value.
            The first path is the LoS one if LoS is requested.

        valid : [num_targets, num_sources, num_samples], tf.bool
            Mask indicating the valid paths

        mirrored_vertices : [max_depth, num_sources, num_samples, 3], tf.float
            Mirrored points

        tri_p0 : [max_depth, num_sources, num_samples, 3], tf.float
            Coordinates of the first vertex of potentially hitted triangles

        normals : [max_depth, num_sources, num_samples, 3], tf.float
            Normals to the potentially hitted triangles

        current : [num_targets, 1, 1, xyz : 3], tf.float
            Positions of the last interactions

        num_targets : int
            Number of targets

        num_sources : int
            Number of sources

        Output
        ------
        valid : [num_targets, num_sources, num_samples], tf.bool
            Mask indicating the valid paths

        current : [num_targets, num_sources, num_samples, 3], tf.float
            Positions of the last interactions

        p : [num_targets, num_sources, num_samples, 3], tf.float
            Intersection point on the ``depth`` primitive

        n : [num_targets, num_sources, num_samples, 3], tf.float
            Normals to the primitive at the ``depth`` intersection point

        maxt : [num_targets, num_sources, num_samples], tf.float
            Distance from current to intersection point

        d : [num_targets, num_sources, num_samples, 3], tf.float
            Ray direction to test for blockage between ``curent`` and the
            intersection point

        active : [num_samples], tf.bool
            Mask indicating paths that are not active, i.e., didn't start yet
        """

        # Number of candidates at this stage
        num_samples = tf.shape(candidates)[1]

        # Primitive indices with which paths interact at this depth
        # [num_samples]
        prim_idx = tf.gather(candidates, depth, axis=0)

        # Next mirrored point
        # [num_sources, num_samples, 3]
        next_pos = tf.gather(mirrored_vertices, depth, axis=0)
        # Expand rank for broadcasting
        # [1, num_sources, num_samples, 3]

        # Since paths can have different depths, we have to mask out paths
        # that have not started yet.
        # [num_samples]
        active = tf.not_equal(prim_idx, -1)

        # Expand rank to broadcast with receivers and transmitters
        # [1, 1, num_samples]
        active = expand_to_rank(active, 3, axis=0)

        # Invalid paths are marked as inactive
        # [num_targets, num_sources, num_samples]
        active = tf.logical_and(active, valid)

        # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
        # This makes no difference on the resulting paths as such paths
        # are not flagged as active.
        # valid_prim_idx = prim_idx
        valid_prim_idx = tf.where(prim_idx == -1, 0, prim_idx)

        # Trace a direct line from the current position to the next path
        # vertex.

        # Ray direction
        # [num_targets, num_sources, num_samples, 3]
        d,_ = normalize(next_pos - current)

        # Find where it intersects the primitive that we mirrored against.
        # If that falls out of the primitive, this whole path is invalid.

        # Vertices forming the triangle.
        # [num_sources, num_samples, xyz : 3]
        p0 = tf.gather(tri_p0, depth, axis=0)
        # Expand rank to broadcast with the target dimension
        # [1, num_sources, num_samples, xyz : 3]
        p0 = tf.expand_dims(p0, axis=0)
        # Build the indexing tensor to gather only the coordinate of the
        # second index
        # [[num_samples, 1]]
        p1_index = tf.expand_dims(valid_prim_idx, axis=1)
        p1_index = tf.pad(p1_index, [[0,0], [0,1]], mode='CONSTANT',
                            constant_values=1) # Second vertex
        # [num_samples, xyz : 3]
        p1 = tf.gather_nd(self.solver._primitives, p1_index)
        # Expand rank to broadcast with the target and sources
        # dimensions
        # [1, 1, num_samples, xyz : 3]
        p1 = expand_to_rank(p1, 4, axis=0)
        # Build the indexing tensor to gather only the coordinate of the
        # third index
        # [[num_samples, 1]]
        p2_index = tf.expand_dims(valid_prim_idx, axis=1)
        p2_index = tf.pad(p2_index, [[0,0], [0,1]], mode='CONSTANT',
                            constant_values=2) # Third vertex
        # [num_samples, xyz : 3]
        p2 = tf.gather_nd(self.solver._primitives, p2_index)
        # Expand rank to broadcast with the target and sources
        # dimensions
        # [1, 1, num_samples, xyz : 3]
        p2 = expand_to_rank(p2, 4, axis=0)
        # Intersection test.
        # We use the Moeller Trumbore algorithm
        # t : [num_targets, num_sources, num_samples]
        # hit : [num_targets, num_sources, num_samples]
        t, hit = moller_trumbore(current, d, p0, p1, p2, self.solver.EPSILON)
        # [num_targets, num_sources, num_samples]
        valid = tf.logical_and(valid, tf.logical_or(~active, hit))

        # Force normal to point towards our current position
        # [num_sources, num_samples, 3]
        n = tf.gather(normals, depth, axis=0)
        # Add dimension for broadcasting with receivers
        # [1, num_sources, num_samples, 3]
        n = tf.expand_dims(n, axis=0)
        # Force to point towards current position
        # [num_targets, num_sources, num_samples, 3]
        s = tf.sign(dot(n, current-p0, keepdim=True))
        n = n * s
        # Intersection point
        # [num_targets, num_sources, num_samples, 3]
        t = tf.expand_dims(t, axis=3)
        p = current + t*d

        # Prepare obstruction test.
        # There should be no obstruction between the actual
        # interaction point and the current point.
        # We use Mitsuba to test for obstruction efficiently.
        # We only compute here the origin and direction of the ray

        # Ensure current is already broadcasted
        # [num_targets, num_sources, num_samples, 3]
        current = tf.broadcast_to(current, [num_targets, num_sources,
                                            num_samples, 3])
        # Distance from current to intersection point
        # [num_targets, num_sources, num_samples]
        maxt = tf.norm(current - p, axis=-1)

        output = (
            valid,
            current,
            p,
            n,
            maxt,
            d,
            active
        )
        return output
    

    def _spec_image_method_phase_22(self, depth, valid, mirrored_vertices,
                    current, blk, num_targets, num_sources, maxt, p, active):
        r"""
        Implement the second part of phase 2 of the image method:

        - Discards paths that are blocked
        - Discards paths for which ``current`` point and the ``next_pos`` are
            not on the same side, as this would mean that the path is going
            through the surface

        Input
        -----
        depth : int
            Current interaction number

        valid : [num_targets, num_sources, num_samples], tf.bool
            Mask indicating the valid paths

        mirrored_vertices : [max_depth, num_sources, num_samples, 3], tf.float
            Mirrored points

        current : [num_targets, num_sources, num_samples, 3], tf.float
            Positions of the last interactions

        blk : [num_targets*num_sources*num_samples], tf.bool
            Mask indicating which blocked paths

        num_targets : int
            Number of targets

        num_sources : int
            Number of sources

        maxt : [num_targets, num_sources, num_samples], tf.float
            Distance from current to intersection point

        p : [num_targets, num_sources, num_samples, 3], tf.float
            Intersection point on the ``depth`` primitive

        active : [num_targets, num_sources, num_samples], tf.bool
            Flag indicating which paths are active

        Output
        ------
        valid : [num_targets, num_sources, num_samples], tf.bool
            Mask indicating the valid paths

        current : [num_targets, num_sources, num_samples, 3], tf.float
            Positions of the last interactions
        """

        # Number of candidates at this stage
        num_samples = tf.shape(valid)[2]

        # Next mirrored point
        # [num_sources, num_samples, 3]
        next_pos = tf.gather(mirrored_vertices, depth, axis=0)
        # Expand rank for broadcasting
        # [1, num_sources, num_samples, 3]
        next_pos = tf.expand_dims(next_pos, axis=0)

        # Discard paths if blocked
        # [num_targets, num_sources, num_samples]
        blk = tf.reshape(blk, [num_targets, num_sources, num_samples])
        valid = tf.logical_and(valid, tf.logical_or(~active, ~blk))

        # Discard paths for which the shooted ray has zero-length, i.e.,
        # when two consecutive intersection points have the same location,
        # or when the source and target have the same locations (RADAR).
        # [num_targets, num_sources, num_samples]
        blk = tf.less(maxt, self.solver.EPSILON)
        # [num_targets, num_sources, num_samples]
        valid = tf.logical_and(valid, tf.logical_or(~active, ~blk))

        # We must also ensure that the current point and the next_pos are
        # not on the same side, as this would mean that the path is going
        # through the surface

        # Vector from the intersection point to the current point
        # [num_targets, num_sources, num_samples, 3]
        v1 = current - p
        # Vector from the intersection point to the next point
        # [num_targets, num_sources, num_samples, 3]
        v2 = next_pos - p
        # Compute the scalar product. It must be negative, as we are using
        # the image (next_pos)
        # [num_targets, num_sources, num_samples]
        blk = dot(v1, v2)
        blk = tf.greater_equal(blk, tf.zeros_like(blk))
        valid = tf.logical_and(valid, tf.logical_or(~active, ~blk))

        # Update active state
        # [num_targets, num_sources, num_samples]
        active = tf.logical_and(active, valid)
        # Prepare for next path segment
        # [num_targets, num_sources, num_samples, 3]
        current = tf.where(tf.expand_dims(active, axis=-1), p, current)

        output = (
            valid,
            current
        )
        return output
    

    def _spec_image_method_phase_23(self, current, sources, num_targets):
        r"""
        Implements the third step of phase 2 of the image method.

        Prepares the rays for testing blockage between the last interaction
        point and the sources.

        Input
        ------
        current : [num_targets, num_sources, num_samples, 3], tf.float
            Positions of the last interactions

        sources : [num_sources, 3], tf.float
            Sources from which rays (paths) are emitted

        num_targets : int
            Number of targets

        Output
        ------
        current : [num_targets, num_sources, num_samples, 3], tf.float
            Positions of the last interactions

        d : [num_targets, num_sources, num_samples, 3], tf.float
            Ray direction between the last interaction point and the sources

        maxt : [num_targets, num_sources, num_samples], tf.float
            Distances between the last interaction point and the sources
        """

        # Number of candidates at this stage
        num_samples = tf.shape(current)[2]

        num_sources = tf.shape(sources)[0]

        # Check visibility to the transmitters
        # [1, num_sources, 1, 3]
        sources_ = tf.expand_dims(tf.expand_dims(sources, axis=0),
                                        axis=2)
        # Direction vector and distance to the transmitters
        # d : [num_targets, num_sources, num_samples, 3]
        # maxt : [num_targets, num_sources, num_samples]
        d,maxt = normalize(sources_ - current)
        # Ensure current is already broadcasted
        # [num_targets, num_sources, num_samples, 3]
        current = tf.broadcast_to(current, [num_targets, num_sources,
                                            num_samples, 3])
        d = tf.broadcast_to(d, [num_targets, num_sources, num_samples, 3])
        maxt = tf.broadcast_to(maxt, [num_targets, num_sources, num_samples])

        return current, d, maxt
    

    def _spec_image_method_phase_3(self, candidates, valid, num_targets,
                                num_sources, path_vertices, path_normals, blk):
        # pylint: disable=line-too-long
        r"""
        Implements the third phase of the image method.

        Post-process the valid paths, from transmitters to receivers, to put
        them in the expected output format.

        Input
        -----
        candidates: [max_depth, num_samples], tf.int
            Set of candidate paths with depth up to ``max_depth``.
            For paths with depth lower than ``max_depth``, -1 must be used as
            padding value.
            The first path is the LoS one if LoS is requested.

        valid : [num_targets, num_sources, num_samples], tf.bool
            Mask indicating the valid paths

        num_targets : int
            Number of targets

        num_sources : int
            Number of sources

        path_vertices : [max_depth, num_targets, num_sources, num_samples, xyz : 3]
            Positions of the intersection points

        path_normals : [max_depth, num_targets, num_sources, num_samples, 3]
            Normals to the surface at the intersection points

        blk : [num_targets*num_sources*num_samples], tf.bool
            Mask indicating which blocked paths

        Output
        ------
        mask : [num_targets, num_sources, max_num_paths], tf.bool
             Mask indicating if a path is valid

        valid_vertices : [max_depth, num_targets, num_sources, max_num_paths, 3], tf.float
            Positions of intersection points.

        valid_objects : [max_depth, num_targets, num_sources, max_num_paths], tf.int
            Indices of the intersected scene objects or wedges.
            Paths with depth lower than ``max_depth`` are padded with `-1`.

        valid_normals : [max_depth, num_targets, num_sources, max_num_paths, 3], tf.float
            Normals to the primitives at the intersection points.
        """

        # Discard blocked paths
        # [num_targets, num_sources, num_samples]
        valid = tf.logical_and(valid, ~blk)

        # Max depth
        max_depth = candidates.shape[0]

        # If at least one link has the LoS paths flagged as valid,
        # then we keep the LoS paths for all links.
        # This makes the tracking of paths types easier.
        # The LoS paths will be masked for links for which it is obstructed.
        # Note that there is only one entry that correspond to LoS paths
        # [1, num_samples]
        is_los = tf.reduce_all(tf.equal(candidates, -1), axis=0, keepdims=True)
        # Only keep the LoS path if it is valid (i.e., not obstructed) for at
        # least one link
        # [1, 1, num_samples]
        is_los = tf.expand_dims(is_los, 0)
        # [1, 1, num_samples]
        is_los = tf.reduce_any(tf.logical_and(is_los, valid), axis=(0,1),
                               keepdims=True)

        # Build indices for keeping only valid path
        # A path is kept if its valid or the LoS and there is at least one link
        # for which LoS is not obstructed
        # [num_targets, num_sources, num_samples]
        keep = tf.logical_or(valid, is_los)
        # [num_targets, num_sources]
        num_paths = tf.reduce_sum(tf.cast(keep, tf.int32), axis=-1)
        # Maximum number of paths
        # ()
        max_num_paths = tf.reduce_max(num_paths)
        # [num_valid, 3]
        gather_indices = tf.where(keep)
        # [num_targets, num_sources, num_samples]
        path_indices = tf.cumsum(tf.cast(keep, tf.int32), axis=-1)
        # [num_valid]
        path_indices = tf.gather_nd(path_indices, gather_indices) - 1
        # [3, num_valid]
        scatter_indices = tf.transpose(gather_indices, [1,0])
        if not tf.size(scatter_indices) == 0:
            # [3, num_valid]
            scatter_indices = tf.tensor_scatter_nd_update(scatter_indices,
                                [[2]], [path_indices])
        # [num_valid, 3]
        scatter_indices = tf.transpose(scatter_indices, [1,0])

        # Mask of valid paths
        # [num_targets, num_sources, max_num_paths]
        mask = tf.fill([num_targets, num_sources, max_num_paths], False)
        # [num_keep_paths]
        mask_ = tf.gather_nd(valid, gather_indices)
        # [num_targets, num_sources, max_num_paths]
        mask = tf.tensor_scatter_nd_update(mask, scatter_indices, mask_)

        # Locations of the interactions
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        valid_vertices = tf.zeros([max_depth, num_targets, num_sources,
                                    max_num_paths, 3], dtype=self._rdtype)
        # Normals at the intersection points
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        valid_normals = tf.zeros([max_depth, num_targets, num_sources,
                                    max_num_paths, 3], dtype=self._rdtype)
        # [max_depth, num_targets, num_sources, max_num_paths]
        valid_primitives = tf.fill([max_depth, num_targets, num_sources,
                                        max_num_paths], -1)

        if max_depth > 0:

            for depth in tf.range(max_depth, dtype=tf.int64):

                # Indices for storing the valid vertices/normals/primitives for
                # this depth
                scatter_indices_ = tf.pad(scatter_indices, [[0,0], [1,0]],
                                mode='CONSTANT', constant_values=depth)

                # Loaction of the interactions
                # Extract only the valid paths
                # [num_targets, num_sources, num_samples, 3]
                vertices_ = tf.gather(path_vertices, depth, axis=0)
                # [total_num_valid_paths, 3]
                vertices_ = tf.gather_nd(vertices_, gather_indices)
                # Store the valid intersection points
                # [max_depth, num_targets, num_sources, max_num_paths, 3]
                valid_vertices = tf.tensor_scatter_nd_update(valid_vertices,
                                                scatter_indices_, vertices_)

                # Normals at the interactions
                # Extract only the valid paths
                # [num_targets, num_sources, num_samples, 3]
                normals_ = tf.gather(path_normals, depth, axis=0)
                # [total_num_valid_paths, 3]
                normals_ = tf.gather_nd(normals_, gather_indices)
                # Store the valid normals
                # [max_depth, num_targets, num_sources, max_num_paths, 3]
                valid_normals = tf.tensor_scatter_nd_update(valid_normals,
                                        scatter_indices_, normals_)

                # Intersected primitives
                # Extract only the valid paths
                # [num_samples]
                primitives_ = tf.gather(candidates, depth, axis=0)
                # [total_num_valid_paths]
                primitives_ = tf.gather(primitives_, gather_indices[:,2])
                # Store the valid primitives]
                # [max_depth, num_targets, num_sources, max_num_paths]
                valid_primitives = tf.tensor_scatter_nd_update(valid_primitives,
                                        scatter_indices_, primitives_)

        # Add a dummy entry to primitives_2_objects with value -1 for invalid
        # reflection.
        # Invalid reflection, i.e., corresponding to paths with a depth lower
        # than max_depth, will be assigned -1 as index of the intersected
        # shape.
        # [num_samples + 1]
        primitives_2_objects = tf.pad(self.solver._primitives_2_objects, [[0,1]],
                                        constant_values=-1)
        # Replace all -1 by num_samples
        num_samples = tf.shape(self.solver._primitives_2_objects)[0]
        # [max_depth, num_targets, num_sources, max_num_paths]
        valid_primitives = tf.where(tf.equal(valid_primitives,-1),
                                    num_samples,
                                    valid_primitives)
        # [max_depth, num_targets, num_sources, max_num_paths]
        valid_objects = tf.gather(primitives_2_objects, valid_primitives)

        # Actual maximum depth
        if max_depth > 0:
            # Limit the depth to the actual max_depth
            # [max_depth]
            useless_depth = tf.reduce_all(tf.equal(valid_objects, -1),
                                          axis=(1,2,3))

            max_depth = tf.where(tf.reduce_any(useless_depth),
                                tf.argmax(tf.cast(useless_depth, tf.int32),
                                          output_type=tf.int32),
                                max_depth)
            max_depth = tf.maximum(max_depth, 1)
            # [max_depth, num_targets, num_sources, max_num_paths, 3]
            valid_vertices = valid_vertices[:max_depth]
            # [max_depth, num_targets, num_sources, max_num_paths, 3]
            valid_normals = valid_normals[:max_depth]
            # [max_depth, num_targets, num_sources, max_num_paths]
            valid_objects = valid_objects[:max_depth]

        return mask, valid_vertices, valid_objects, valid_normals
    

    def _spec_image_method(self, candidates, paths, spec_paths_tmp, mirrored_vertices, tri_p0, normals):
        # pylint: disable=line-too-long
        r"""
        Evaluates a list of candidate paths ``candidates`` and keep only the
        valid ones, i.e., the non-obstricted ones with valid reflections only,
        using the image method.

        Input
        -----
        candidates: [max_depth, num_samples], tf.int
            Set of candidate paths with depth up to ``max_depth``.
            For paths with depth lower than ``max_depth``, -1 must be used as
            padding value.
            The first path is the LoS one if LoS is requested.

        paths : :class:`~sionna.rt.Paths`
            Paths to update
        """

        sources = paths.sources
        targets = paths.targets

        # Max depth
        max_depth = candidates.shape[0]

        # Number of sources and number of receivers
        num_sources = len(sources)
        num_targets = len(targets)

        # --- Phase 1
        # Starting from the sources, mirror each point against the
        # given candidate primitive. At this stage, we do not carry
        # any verification about the visibility of the ray.
        # Loop through the max_depth interactions. All candidate paths are
        # processed in parallel.
        #
        # mirrored_vertices : [max_depth, num_sources, num_samples, 3], tf.float
        #     Mirrored points coordinates
        #
        # tri_p0 : [max_depth, num_sources, num_samples, 3], tf.float
        #     Coordinates of the first vertex of potentially hitted triangles
        #
        # normals : [max_depth, num_sources, num_samples, 3], tf.float
        #     Normals to the potentially hitted triangles
        # mirrored_vertices, tri_p0, normals =\
        #     self._spec_image_method_phase_1(candidates, sources)

        # --- Phase 2

        # Number of candidates at this stage
        num_samples = candidates.shape[1]

        # Starting from the receivers, go over the vertices in reverse
        # and check that connections are possible.

        # Mask indicating which paths are valid
        # [num_targets, num_sources, num_samples]
        valid = tf.fill([num_targets, num_sources, num_samples], True)
        # Positions of the last interactions.
        # Initialized with the positions of the receivers.
        # Add two additional dimensions for broadcasting with transmitters and
        # paths.
        # [num_targets, 1, 1, xyz : 3]
        current = expand_to_rank(targets, 4, axis=1)
        # Positions of the interactions.
        # [max_depth, num_targets, num_sources, num_samples, xyz : 3]
        # path_vertices = tf.zeros([max_depth, num_targets, num_sources,
        #                             num_samples, 3], dtype=self._rdtype)
        path_vertices = []
        # Normals at the interactions.
        # [max_depth, num_targets, num_sources, num_samples, xyz : 3]
        path_normals = []
        for depth in tf.range(max_depth-1, -1, -1):

            # The following call:
            # - Computes the intersection point with the ``depth``th primitive
            #   of the sequence of candidates for a ray originating from
            #   ``current``
            # - Checks that the intersection point is within the primitive
            # - Ensures the normal points toward the ``current`` point
            # - Prepares the ray to test for blockage between ``depth-1``th
            # point and the current point
            output = self._spec_image_method_phase_21(depth, candidates, valid,
                mirrored_vertices, tri_p0, normals, current, num_targets,
                num_sources)
            # [num_targets, num_sources, num_samples]
            #   Mask indicating the valid paths
            valid = output[0]
            # [num_targets, 1, 1, xyz : 3]
            #   Positions of the last interactions
            current = output[1]
            # : [num_targets, num_sources, num_samples, 3], tf.float
            #     Intersection point on the ``depth`` primitive
            path_vertices_ = output[2]
            # : [num_targets, num_sources, num_samples, 3], tf.float
            #    Normals to the primitive at the ``depth`` intersection point
            path_normals_ = output[3]
            # maxt : [num_targets, num_sources, num_samples], tf.float
            #     Distance from current to intersection point
            maxt = output[4]
            # d : [num_targets, num_sources, num_samples, 3], tf.float
            #     Ray direction to test for blockage between ``curent`` and the
            #     intersection point
            d = output[5]
            # [num_targets, num_sources, num_samples]
            #   Mask indicating paths that are not active, i.e., didn't start
            #   yet
            active = output[6]

            # Test for obstruction using Mitsuba
            # As Mitsuba only hanldes a single batch dimension, we flatten the
            # batch dims [num_targets, num_sources, num_samples]
            # [num_targets*num_sources*num_samples]
            blk = self.solver._test_obstruction(tf.reshape(current, [-1, 3]),
                                         tf.reshape(d, [-1, 3]),
                                         tf.reshape(maxt, [-1]))

            # The following call:
            # - Discards paths that are blocked
            # - Discards paths for which ``current`` point and the ``next_pos``
            #   are not on the same side, as this would mean that the path is
            #   going through the surface
            output = self._spec_image_method_phase_22(depth, valid,
                mirrored_vertices, current, blk, num_targets, num_sources, maxt,
                path_vertices_, active)
            # [num_targets, num_sources, num_samples]
            #   Mask indicating the valid paths
            valid = output[0]
            # [num_targets, num_sources, num_samples, xyz : 3]
            #   Positions of the last interactions
            current = output[1]

            path_vertices.append(path_vertices_)
            path_normals.append(path_normals_)

        path_vertices.reverse()
        path_normals.reverse()
        path_vertices = tf.stack(path_vertices, axis=0)
        path_normals = tf.stack(path_normals, axis=0)

        # Prepares the rays for testing blockage between the last
        # interaction point and the sources.
        #
        # current : [num_targets, num_sources, num_samples, 3], tf.float
        #     Positions of the last interactions
        #
        # d : [num_targets, num_sources, num_samples, 3], tf.float
        #     Ray direction between the last interaction point and the sources
        #
        # maxt : [num_targets, num_sources, num_samples], tf.float
        #     Distances between the last interaction point and the sources
        current, d, maxt = self._spec_image_method_phase_23(current, sources,
                                                      num_targets)

        # Test for obstruction using Mitsuba
        # [num_targets*num_sources*num_samples]
        val = self.solver._test_obstruction(tf.reshape(current, [-1, 3]),
                                     tf.reshape(d, [-1, 3]),
                                     tf.reshape(maxt, [-1]))
        # [num_targets, num_sources, num_samples, 3]
        blk = tf.reshape(val, tf.shape(maxt))
        # Discard paths for which the shooted ray has zero-length, i.e., when
        # two consecutive intersection points have the same location, or when
        # the source and target have the same locations (RADAR).
        # [num_targets, num_sources, num_samples]
        blk = tf.logical_or(blk, tf.less(maxt, self.solver.EPSILON))

        # --- Phase 3
        # Post-process the valid paths, from transmitters to receivers, to put
        # them in the expected output format.
        #
        # mask : [num_targets, num_sources, max_num_paths], tf.bool
        #      Mask indicating if a path is valid
        #
        # valid_vertices : [max_depth, num_targets, num_sources,
        #                   max_num_paths, 3], tf.float
        #     Positions of intersection points.
        #
        # valid_objects : [max_depth, num_targets, num_sources,
        #                   max_num_paths], tf.int
        #     Indices of the intersected scene objects or wedges.
        #     Paths with depth lower than ``max_depth`` are padded with `-1`.
        #
        # valid_normals : [max_depth, num_targets, num_sources,
        #                   max_num_paths, 3], tf.float
        #     Normals to the primitives at the intersection points.
        mask, valid_vertices, valid_objects, valid_normals =\
            self._spec_image_method_phase_3(candidates, valid, num_targets,
                                num_sources, path_vertices, path_normals, blk)

        # Update the object storing the paths
        paths.mask = mask
        #paths.targets_sources_mask = mask
        paths.vertices = valid_vertices
        paths.objects = valid_objects
        paths.path_types = tf.fill(paths.objects.shape, Paths.SPECULAR)
        #paths.depth_mask = valid_objects != -1

        spec_paths_tmp.normals = valid_normals

    ### For combinations (reflection+diffraction)

    def _spec_image_method_phase_21_depth_1(self, prim_idx, valid,
                                    next_pos, tri_p0, normals, current,
                                    num_targets, num_sources):
        # pylint: disable=line-too-long
        r"""
        Implement the first part of phase 2 of the image method:

        For a ``depth`` == 1:
        - Computes the intersection point with the primitive (candidate)
          for a ray originating from ``current``
        - Checks that the intersection point is within the primitive
        - Ensures the normal points toward the ``current`` point
        - Prepares the ray to test for blockage between ``current`` point and
        the computed intersection point

        The obstruction test is note performed in this function as it uses
        Mitsuba.

        Input
        -----
        prim_idx: [num_samples], tf.int
            Set of candidate primitives, -1 must be used as
            padding value.

        valid : [num_targets, num_sources, num_samples], tf.bool
            Mask indicating the valid paths

        next_pos : [num_sources, num_samples, 3], tf.float
            Mirrored points

        tri_p0 : [num_sources, num_samples, 3], tf.float
            Coordinates of the first vertex of potentially hitted triangles

        normals : [num_sources, num_samples, 3], tf.float
            Normals to the potentially hitted triangles

        current : [num_targets, num_sources, num_samples, 3] or [num_targets, 1, 1, xyz : 3], tf.float
            Positions of the last interactions

        num_targets : int
            Number of targets

        num_sources : int
            Number of sources

        Output
        ------
        valid : [num_targets, num_sources, num_samples], tf.bool
            Mask indicating the valid paths

        current : [num_targets, num_sources, num_samples, 3], tf.float
            Positions of the last interactions

        p : [num_targets, num_sources, num_samples, 3], tf.float
            Intersection point on the primitive

        n : [num_targets, num_sources, num_samples, 3], tf.float
            Normals to the primitive at intersection point

        maxt : [num_targets, num_sources, num_samples], tf.float
            Distance from current to intersection point

        d : [num_targets, num_sources, num_samples, 3], tf.float
            Ray direction to test for blockage between ``curent`` and the
            intersection point

        active : [num_targets, num_sources, num_samples], tf.bool
            Mask indicating paths that are not active, i.e., didn't start yet
        """

        # # Number of candidates at this stage
        # num_samples = tf.shape(candidates)[1]
        num_samples = tf.shape(prim_idx)[0]

        # Since paths can have different depths, we have to mask out paths
        # that have not started yet.
        # [num_samples]
        active = tf.not_equal(prim_idx, -1)

        # Expand rank to broadcast with receivers and transmitters
        # [1, 1, num_samples]
        active = expand_to_rank(active, 3, axis=0)

        # Invalid paths are marked as inactive
        # [num_targets, num_sources, num_samples]
        active = tf.logical_and(active, valid)

        # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
        # This makes no difference on the resulting paths as such paths
        # are not flagged as active.
        # valid_prim_idx = prim_idx
        valid_prim_idx = tf.where(prim_idx == -1, 0, prim_idx)

        # Trace a direct line from the current position to the next path
        # vertex.

        # Ray direction
        # [num_targets, num_sources, num_samples, 3]
        d,_ = normalize(next_pos - current)

        # Find where it intersects the primitive that we mirrored against.
        # If that falls out of the primitive, this whole path is invalid.

        # Vertices forming the triangle.
        # [num_sources, num_samples, xyz : 3]
        p0 = tri_p0
        # Expand rank to broadcast with the target dimension
        # [1, num_sources, num_samples, xyz : 3]
        p0 = tf.expand_dims(p0, axis=0)
        # Build the indexing tensor to gather only the coordinate of the
        # second index
        # [[num_samples, 1]]
        p1_index = tf.expand_dims(valid_prim_idx, axis=1)
        p1_index = tf.pad(p1_index, [[0,0], [0,1]], mode='CONSTANT',
                            constant_values=1) # Second vertex
        # [num_samples, xyz : 3]
        p1 = tf.gather_nd(self.solver._primitives, p1_index)
        # Expand rank to broadcast with the target and sources
        # dimensions
        # [1, 1, num_samples, xyz : 3]
        p1 = expand_to_rank(p1, 4, axis=0)
        # Build the indexing tensor to gather only the coordinate of the
        # third index
        # [[num_samples, 1]]
        p2_index = tf.expand_dims(valid_prim_idx, axis=1)
        p2_index = tf.pad(p2_index, [[0,0], [0,1]], mode='CONSTANT',
                            constant_values=2) # Third vertex
        # [num_samples, xyz : 3]
        p2 = tf.gather_nd(self.solver._primitives, p2_index)
        # Expand rank to broadcast with the target and sources
        # dimensions
        # [1, 1, num_samples, xyz : 3]
        p2 = expand_to_rank(p2, 4, axis=0)
        # Intersection test.
        # We use the Moeller Trumbore algorithm
        # t : [num_targets, num_sources, num_samples]
        # hit : [num_targets, num_sources, num_samples]
        t, hit = moller_trumbore(current, d, p0, p1, p2, self.solver.EPSILON)
        # [num_targets, num_sources, num_samples]
        valid = tf.logical_and(valid, tf.logical_or(~active, hit))

        # Force normal to point towards our current position
        # [num_sources, num_samples, 3]
        n = normals
        # Add dimension for broadcasting with receivers
        # [1, num_sources, num_samples, 3]
        n = tf.expand_dims(n, axis=0)
        # Force to point towards current position
        # [num_targets, num_sources, num_samples, 3]
        s = tf.sign(dot(n, current-p0, keepdim=True))
        n = n * s
        # Intersection point
        # [num_targets, num_sources, num_samples, 3]
        t = tf.expand_dims(t, axis=3)
        p = current + t*d

        # Prepare obstruction test.
        # There should be no obstruction between the actual
        # interaction point and the current point.
        # We use Mitsuba to test for obstruction efficiently.
        # We only compute here the origin and direction of the ray

        # Ensure current is already broadcasted
        # [num_targets, num_sources, num_samples, 3]
        current = tf.broadcast_to(current, [num_targets, num_sources,
                                            num_samples, 3])
        # Distance from current to intersection point
        # [num_targets, num_sources, num_samples]
        maxt = tf.norm(current - p, axis=-1)

        output = (
            valid,
            current,
            p,
            n,
            maxt,
            d,
            active
        )

        return output


    def _spec_image_method_phase_22_depth_1(self, valid, next_pos,
                    current, blk, num_targets, num_sources, maxt, p, active):
        r"""
        Implement the second part of phase 2 of the image method:

        - Discards paths that are blocked
        - Discards paths for which ``current`` point and the ``next_pos`` are
            not on the same side, as this would mean that the path is going
            through the surface

        Input
        -----
        valid : [num_targets, num_sources, num_samples], tf.bool
            Mask indicating the valid paths

        next_pos : [num_sources, num_samples, 3], tf.float
            Mirrored points

        current : [num_targets, num_sources, num_samples, 3], tf.float
            Positions of the last interactions

        blk : [num_targets*num_sources*num_samples], tf.bool
            Mask indicating which blocked paths

        num_targets : int
            Number of targets

        num_sources : int
            Number of sources

        maxt : [num_targets, num_sources, num_samples], tf.float
            Distance from current to intersection point

        p : [num_targets, num_sources, num_samples, 3], tf.float
            Intersection point on the primitive

        active : [num_targets, num_sources, num_samples], tf.bool
            Flag indicating which paths are active

        Output
        ------
        valid : [num_targets, num_sources, num_samples], tf.bool
            Mask indicating the valid paths

        current : [num_targets, num_sources, num_samples, 3], tf.float
            Positions of the last interactions
        """

        # Number of candidates at this stage
        num_samples = tf.shape(valid)[2]

        # Discard paths if blocked
        # [num_targets, num_sources, num_samples]
        blk = tf.reshape(blk, [num_targets, num_sources, num_samples])
        valid = tf.logical_and(valid, tf.logical_or(~active, ~blk))

        # Discard paths for which the shooted ray has zero-length, i.e.,
        # when two consecutive intersection points have the same location,
        # or when the source and target have the same locations (RADAR).
        # [num_targets, num_sources, num_samples]
        blk = tf.less(maxt, self.solver.EPSILON)
        # [num_targets, num_sources, num_samples]
        valid = tf.logical_and(valid, tf.logical_or(~active, ~blk))

        # We must also ensure that the current point and the next_pos are
        # not on the same side, as this would mean that the path is going
        # through the surface

        # Vector from the intersection point to the current point
        # [num_targets, num_sources, num_samples, 3]
        v1 = current - p
        # Vector from the intersection point to the next point
        # [num_targets, num_sources, num_samples, 3]
        v2 = next_pos - p
        # Compute the scalar product. It must be negative, as we are using
        # the image (next_pos)
        # [num_targets, num_sources, num_samples]
        blk = dot(v1, v2)
        blk = tf.greater_equal(blk, tf.zeros_like(blk))
        valid = tf.logical_and(valid, tf.logical_or(~active, ~blk))

        # Update active state
        # [num_targets, num_sources, num_samples]
        active = tf.logical_and(active, valid)
        # Prepare for next path segment
        # [num_targets, num_sources, num_samples, 3]
        current = tf.where(tf.expand_dims(active, axis=-1), p, current)

        output = (
            valid,
            current
        )
        return output


    def _paths_post_process(self, obj_idxs, valid, num_targets,
                                num_sources, path_vertices, blk):
        # pylint: disable=line-too-long
        r"""
        (from def _spec_image_method_phase_3)

        Post-process the valid paths, from transmitters to receivers, to put
        them in the expected output format.

        Input
        -----
        obj_idxs: [max_depth, num_targets, num_sources, num_samples], tf.int
            Set of candidate paths with depth up to ``max_depth``.
            For paths with depth lower than ``max_depth``, -1 must be used as
            padding value.
            The first path is the LoS one if LoS is requested.

        valid : [num_targets, num_sources, num_samples], tf.bool
            Mask indicating the valid paths

        num_targets : int
            Number of targets

        num_sources : int
            Number of sources

        path_vertices : [max_depth, num_targets, num_sources, num_samples, xyz : 3]
            Positions of the intersection points

        blk : [num_targets*num_sources*num_samples], tf.bool
            Mask indicating which blocked paths

        path_types : [max_depth, num_targets, num_sources, num_samples]
            Types of intersection points (reflection, diffraction)

        Output
        ------
        mask : [num_targets, num_sources, max_num_paths], tf.bool
             Mask indicating if a path is valid

        valid_vertices : [max_depth, num_targets, num_sources, max_num_paths, 3], tf.float
            Positions of intersection points.

        valid_objects : [max_depth, num_targets, num_sources, max_num_paths], tf.int
            Indices of the intersected scene objects or wedges.
            Paths with depth lower than ``max_depth`` are padded with `-1`.

        valid_path_types : [max_depth, num_targets, num_sources, max_num_paths, 3]
            Types of intersection points (reflection, diffraction)
        """

        # Discard blocked paths
        # [num_targets, num_sources, num_samples]
        valid = tf.logical_and(valid, ~blk)

        # Max depth
        max_depth = obj_idxs.shape[0]

        # If at least one link has the LoS paths flagged as valid,
        # then we keep the LoS paths for all links.
        # This makes the tracking of paths types easier.
        # The LoS paths will be masked for links for which it is obstructed.
        # Note that there is only one entry that correspond to LoS paths
        # [1, num_samples]
        is_los = tf.reduce_all(tf.equal(obj_idxs, -1), axis=0, keepdims=True)
        # Only keep the LoS path if it is valid (i.e., not obstructed) for at
        # least one link
        # [1, 1, num_samples]
        is_los = tf.expand_dims(is_los, 0)
        # [1, 1, num_samples]
        is_los = tf.reduce_any(tf.logical_and(is_los, valid), axis=(0,1),
                               keepdims=True)

        # Build indices for keeping only valid path
        # A path is kept if its valid or the LoS and there is at least one link
        # for which LoS is not obstructed
        # [num_targets, num_sources, num_samples]
        #keep = tf.logical_or(valid, is_los)
        keep = valid
        # [num_targets, num_sources]
        num_paths = tf.reduce_sum(tf.cast(keep, tf.int32), axis=-1)
        # Maximum number of paths
        # ()
        max_num_paths = tf.reduce_max(num_paths)
        # [num_valid, 3]
        gather_indices = tf.where(keep)
        # [num_targets, num_sources, num_samples]
        path_indices = tf.cumsum(tf.cast(keep, tf.int32), axis=-1)
        # [num_valid]
        path_indices = tf.gather_nd(path_indices, gather_indices) - 1
        # [3, num_valid]
        scatter_indices = tf.transpose(gather_indices, [1,0])
        if not tf.size(scatter_indices) == 0:
            # [3, num_valid]
            scatter_indices = tf.tensor_scatter_nd_update(scatter_indices,
                                [[2]], [path_indices])
        # [num_valid, 3]
        scatter_indices = tf.transpose(scatter_indices, [1,0])

        # Mask of valid paths
        # [num_targets, num_sources, max_num_paths]
        mask = tf.fill([num_targets, num_sources, max_num_paths], False)
        # [num_keep_paths]
        mask_ = tf.gather_nd(valid, gather_indices)
        # [num_targets, num_sources, max_num_paths]
        mask = tf.tensor_scatter_nd_update(mask, scatter_indices, mask_)

        # Locations of the interactions
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        valid_vertices = tf.zeros([max_depth, num_targets, num_sources,
                                    max_num_paths, 3], dtype=self._rdtype)
        # Normals at the intersection points
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        # valid_normals = tf.zeros([max_depth, num_targets, num_sources,
        #                             max_num_paths, 3], dtype=self._rdtype)
        # [max_depth, num_targets, num_sources, max_num_paths]
        valid_primitives = tf.fill([max_depth, num_targets, num_sources,
                                        max_num_paths], -1)
        
        # [num_targets, num_sources, max_num_paths]
        # valid_path_types = tf.zeros([num_targets, num_sources,
        #                             max_num_paths], dtype=tf.int32)

        if max_depth > 0:  # TODO: maybe max_depth==1
            for depth in tf.range(max_depth, dtype=tf.int64):
                # Indices for storing the valid vertices/normals/primitives for
                # this depth
                scatter_indices_ = tf.pad(scatter_indices, [[0,0], [1,0]],
                                mode='CONSTANT', constant_values=depth)

                # Loaction of the interactions
                # Extract only the valid paths
                # [num_targets, num_sources, num_samples, 3]
                vertices_ = tf.gather(path_vertices, depth, axis=0)
                # [total_num_valid_paths, 3]
                vertices_ = tf.gather_nd(vertices_, gather_indices)
                # Store the valid intersection points
                # [max_depth, num_targets, num_sources, max_num_paths, 3]
                valid_vertices = tf.tensor_scatter_nd_update(valid_vertices,
                                                scatter_indices_, vertices_)
                
                # [num_targets, num_sources, num_samples, 3]
                # path_types_ = tf.gather(path_types, depth, axis=0)
                # path_types_ = tf.gather_nd(path_types_, gather_indices)
                # # [max_depth, num_targets, num_sources, max_num_paths]
                # valid_path_types = tf.tensor_scatter_nd_update(valid_path_types,
                #                                     scatter_indices_, path_types_)

                # Intersected primitives
                # Extract only the valid paths
                # [num_samples]
                primitives_ = tf.gather(obj_idxs, depth, axis=0)
                # [total_num_valid_paths]
                primitives_ = tf.gather_nd(primitives_, gather_indices)
                # Store the valid primitives]
                # [max_depth, num_targets, num_sources, max_num_paths]
                valid_primitives = tf.tensor_scatter_nd_update(valid_primitives,
                                        scatter_indices_, primitives_)

        # primitives_2_objects = tf.pad(self.solver._primitives_2_objects, [[0,1]], constant_values=-1)
        # valid_objects = tf.gather(primitives_2_objects, valid_primitives)

        return mask, valid_vertices, valid_primitives, gather_indices, scatter_indices

    def _foo(self, objects, k_t, k_r, normals, reduction_factor, etas):
        # # Maximum depth
        # max_depth = tf.shape(vertices)[0]
        # # Number of targets
        num_targets = tf.shape(objects)[0]
        # Number of sources
        num_sources = tf.shape(objects)[1]
        # Maximum number of paths
        max_num_paths = tf.shape(objects)[2]

        theta_t, phi_t = theta_phi_from_unit_vec(k_t)
        theta_r, phi_r = theta_phi_from_unit_vec(k_r)

        # # Flag that indicates if a ray is valid
        # [num_targets, num_sources, max_num_paths]
        valid = tf.not_equal(objects, -1)

        # Compute cos(theta) at each reflection point
        # [max_depth, num_targets, num_sources, max_num_paths]
        cos_theta = -dot(k_t, normals, clip=True)

        # Compute e_i_s, e_i_p, e_r_s, e_r_p at each reflection point
        # all : [max_depth, num_targets, num_sources, max_num_paths,3]
        # pylint: disable=unbalanced-tuple-unpacking
        e_i_s, e_i_p, e_r_s, e_r_p = compute_field_unit_vectors(k_t, k_r, normals, self.solver.EPSILON)

        # Compute r_s, r_p at each reflection point
        # [num_targets, num_sources, max_num_paths]
        r_s, r_p = reflection_coefficient(etas, cos_theta)

        # # Compute the field transfer matrix.
        # # It is initialized with the identity matrix of size 2 (S and P
        # # polarization components)
        # # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = tf.eye(num_rows=2,
                    batch_shape=[num_targets, num_sources, max_num_paths],
                    dtype=self._dtype)
        # # Initialize last field unit vector with outgoing ones
        # # [num_targets, num_sources, max_num_paths, 3]
        # last_e_r_s = theta_hat(theta_t, phi_t)
        # last_e_r_p = phi_hat(phi_t)
        last_e_r_s = theta_hat(theta_t, phi_t)
        last_e_r_p = phi_hat(phi_t)

        # [num_targets, num_sources, max_num_paths]
        # reduction_factor_ = reduction_factor[depth]
        # [num_targets, num_sources, max_num_paths, 1, 1]
        reduction_factor_ = insert_dims(reduction_factor, 2, -1)

        # Early stopping if no active rays
        # if not tf.reduce_any(valid):
        #     return mat_t

        # Add dimension for broadcasting with coordinates
        # [num_targets, num_sources, max_num_paths, 1]
        valid_ = tf.expand_dims(valid, axis=-1)

        # Change of basis matrix
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_cob = component_transform(last_e_r_s, last_e_r_p,
                                        e_i_s, e_i_p)
        mat_cob = tf.complex(mat_cob, tf.zeros_like(mat_cob))
        # Only apply transform if valid reflection
        # [num_targets, num_sources, max_num_paths, 1, 1]
        valid__ = tf.expand_dims(valid_, axis=-1)
        # [num_targets, num_sources, max_num_paths, 2, 2]
        e = tf.where(valid__, tf.linalg.matmul(mat_cob, mat_t), mat_t)
        # Only update ongoing direction for next iteration if this
        # reflection is valid and if this is not the last step
        last_e_r_s = tf.where(valid_, e_r_s, last_e_r_s)  # last_e_r_s = e_r_s
        last_e_r_p = tf.where(valid_, e_r_p, last_e_r_p)

        # Fresnel coefficients
        # [num_targets, num_sources, max_num_paths, 2]
        r = tf.stack([r_s, r_p], -1)
        # Set the coefficients to one if non-valid reflection
        # [num_targets, num_sources, max_num_paths, 2]
        r = tf.where(valid_, r, tf.ones_like(r))
        # Add a dimension to broadcast with mat_t
        # [num_targets, num_sources, max_num_paths, 2, 1]
        r = tf.expand_dims(r, axis=-1)
        # Apply Fresnel coefficient
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = r*e

        mat_cob = component_transform(last_e_r_s, last_e_r_p,
                                        theta_hat(theta_r, phi_r),
                                        phi_hat(phi_r))
        mat_cob = tf.complex(mat_cob, tf.zeros_like(mat_cob))

        # Apply transformation
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = tf.linalg.matmul(mat_cob, mat_t)

        # Apply the reduction factor
        # [num_targets, num_sources, max_num_paths, 2]
        reduction_factor_ = tf.where(valid__, reduction_factor_,
                                    tf.ones_like(reduction_factor_))
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = mat_t*reduction_factor_

        return mat_t
    
    def _spec_transition_matrices(self, relative_permittivity,
                                  scattering_coefficient,
                                  paths, paths_tmp, scattering):
        # pylint: disable=line-too-long
        """
        Compute the transfer matrices, delays, angles of departures, and angles
        of arrivals, of paths from a set of valid reflection paths and the
        EM properties of the materials.

        Input
        ------
        relative_permittivity : [num_shape], tf.complex
            Tensor containing the relative permittivity of all shapes

        scattering_coefficient : [num_shape], tf.float
            Tensor containing the scattering coefficients of all shapes

        paths : :class:`~sionna.rt.Paths`
            Paths to update

        paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Addtional quantities required for paths computation

        scattering : bool
            Set to `True` if computing the scattered paths.

        Output
        -------
        mat_t : [num_targets, num_sources, max_num_paths, 2, 2], tf.complex
                Specular transition matrix for every path.
        """

        vertices = paths.vertices
        targets = paths.targets
        sources = paths.sources
        objects = paths.objects
        theta_t = paths.theta_t
        phi_t = paths.phi_t
        theta_r = paths.theta_r
        phi_r = paths.phi_r

        normals = paths_tmp.normals
        k_i = paths_tmp.k_i
        k_r = paths_tmp.k_r
        if scattering:
            # For scattering, only the distance up to the last intersection
            # point is considered for path loss.
            # [num_targets, num_sources, max_num_paths]
            total_distance = paths_tmp.scat_src_2_last_int_dist
        else:
            # [num_targets, num_sources, max_num_paths]
            total_distance = paths_tmp.total_distance

        # Maximum depth
        max_depth = tf.shape(vertices)[0]
        # Number of targets
        num_targets = tf.shape(targets)[0]
        # Number of sources
        num_sources = tf.shape(sources)[0]
        # Maximum number of paths
        max_num_paths = tf.shape(objects)[3]

        # Flag that indicates if a ray is valid
        # [max_depth, num_targets, num_sources, max_num_paths]
        valid_ray = tf.not_equal(objects, -1)
        # Pad to enable detection of the last valid reflection for scattering
        # [max_depth+1, num_targets, num_sources, max_num_paths]
        valid_ray = tf.pad(valid_ray, [[0,1], [0,0], [0,0], [0,0]],
                           constant_values=False)

        # Relative perimittivities and scattering coefficients.
        # If a callable is defined to compute the radio material properties,
        # it is invoked. Otherwise, the radio materials of objects are used.
        rm_callable = self._scene.radio_material_callable
        if rm_callable is None:
            # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
            # This makes no difference on the resulting paths as such paths
            # are not flagged as active.
            # [max_depth, num_targets, num_sources, max_num_paths]
            valid_object_idx = tf.where(objects == -1, 0, objects)
            if tf.shape(relative_permittivity)[0] == 0:
                # [max_depth, num_targets, num_sources, max_num_paths]
                etas = tf.zeros_like(valid_object_idx, dtype=self._dtype)
                scattering_coefficient = tf.zeros_like(valid_object_idx,
                                                       dtype=self._rdtype)

            else:
                # [max_depth, num_targets, num_sources, max_num_paths]
                etas = tf.gather(relative_permittivity, valid_object_idx)
                scattering_coefficient = tf.gather(scattering_coefficient,
                                                   valid_object_idx)
        else:
            # [max_depth, num_targets, num_sources, max_num_paths]
            etas, scattering_coefficient, _  = rm_callable(objects, vertices)

        # Compute cos(theta) at each reflection point
        # [max_depth, num_targets, num_sources, max_num_paths]
        cos_theta = -dot(k_i[:max_depth], normals, clip=True)

        # Compute e_i_s, e_i_p, e_r_s, e_r_p at each reflection point
        # all : [max_depth, num_targets, num_sources, max_num_paths,3]
        # pylint: disable=unbalanced-tuple-unpacking
        e_i_s, e_i_p, e_r_s, e_r_p = compute_field_unit_vectors(k_i[:max_depth],
                                            k_r, normals, self.solver.EPSILON)

        # Compute r_s, r_p at each reflection point
        # [max_depth, num_targets, num_sources, max_num_paths]
        r_s, r_p = reflection_coefficient(etas, cos_theta)

        # Multiply the reflection coefficients with the
        # reflection reduction factor
        # [max_depth, num_targets, num_sources, max_num_paths]
        reduction_factor = tf.sqrt(1 - scattering_coefficient**2)
        reduction_factor = tf.complex(reduction_factor,
                                      tf.zeros_like(reduction_factor))

        # Compute the field transfer matrix.
        # It is initialized with the identity matrix of size 2 (S and P
        # polarization components)
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = tf.eye(num_rows=2,
                    batch_shape=[num_targets, num_sources, max_num_paths],
                    dtype=self._dtype)
        # Initialize last field unit vector with outgoing ones
        # [num_targets, num_sources, max_num_paths, 3]
        last_e_r_s = theta_hat(theta_t, phi_t)
        last_e_r_p = phi_hat(phi_t)
        for depth in tf.range(0,max_depth):

            # Is this a valid reflection?
            # [num_targets, num_sources, max_num_paths]
            valid = valid_ray[depth]

            # Is the next reflection valid?
            # [num_targets, num_sources, max_num_paths]
            next_valid = valid_ray[depth+1]
            # Expand for broadcasting
            # [num_targets, num_sources, max_num_paths, 1, 1]
            next_valid = insert_dims(next_valid, 2)

            # [num_targets, num_sources, max_num_paths]
            reduction_factor_ = reduction_factor[depth]
            # [num_targets, num_sources, max_num_paths, 1, 1]
            reduction_factor_ = insert_dims(reduction_factor_, 2, -1)

            # Early stopping if no active rays
            if not tf.reduce_any(valid):
                break

            # Add dimension for broadcasting with coordinates
            # [num_targets, num_sources, max_num_paths, 1]
            valid_ = tf.expand_dims(valid, axis=-1)

            # Change of basis matrix
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_cob = component_transform(last_e_r_s, last_e_r_p,
                                          e_i_s[depth], e_i_p[depth])
            mat_cob = tf.complex(mat_cob, tf.zeros_like(mat_cob))
            # Only apply transform if valid reflection
            # [num_targets, num_sources, max_num_paths, 1, 1]
            valid__ = tf.expand_dims(valid_, axis=-1)
            # [num_targets, num_sources, max_num_paths, 2, 2]
            e = tf.where(valid__, tf.linalg.matmul(mat_cob, mat_t), mat_t)
            # Only update ongoing direction for next iteration if this
            # reflection is valid and if this is not the last step
            last_e_r_s = tf.where(valid_, e_r_s[depth], last_e_r_s)
            last_e_r_p = tf.where(valid_, e_r_p[depth], last_e_r_p)

            # Fresnel coefficients
            # [num_targets, num_sources, max_num_paths, 2]
            r = tf.stack([r_s[depth], r_p[depth]], -1)
            # Set the coefficients to one if non-valid reflection
            # [num_targets, num_sources, max_num_paths, 2]
            r = tf.where(valid_, r, tf.ones_like(r))
            # Add a dimension to broadcast with mat_t
            # [num_targets, num_sources, max_num_paths, 2, 1]
            r = tf.expand_dims(r, axis=-1)
            # Apply Fresnel coefficient
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_t = r*e

            # If scattering, then the reduction coefficient is not applied
            # to the last interaction as the outgoing ray is diffusely
            # reflected and not specularly reflected
            if scattering:
                # [num_targets, num_sources, max_num_paths, 1, 1]
                apply_reduction = tf.logical_and(valid__, next_valid)
            else:
                apply_reduction = valid__

            # Apply the reduction factor
            # [num_targets, num_sources, max_num_paths, 2]
            reduction_factor_ = tf.where(apply_reduction, reduction_factor_,
                                        tf.ones_like(reduction_factor_))
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_t = mat_t*reduction_factor_

        # Move to the targets frame
        # This is not done for scattering as we stop the last interaction point
        if not scattering:
            # Transformation matrix
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_cob = component_transform(last_e_r_s, last_e_r_p,
                                        theta_hat(theta_r, phi_r),
                                        phi_hat(phi_r))
            mat_cob = tf.complex(mat_cob, tf.zeros_like(mat_cob))
            # Apply transformation
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_t = tf.linalg.matmul(mat_cob, mat_t)

        # Divide by total distance to account for propagation loss
        # [num_targets, num_sources, max_num_paths, 1, 1]
        total_distance = expand_to_rank(total_distance, tf.rank(mat_t),
                                        axis=3)
        total_distance = tf.complex(total_distance,
                                    tf.zeros_like(total_distance))
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = tf.math.divide_no_nan(mat_t, total_distance)

        # Set invalid paths to 0 and stores the transition matrices
        # Expand masks to broadcast with the field components
        # [num_targets, num_sources, max_num_paths, 1, 1]
        mask_ = expand_to_rank(paths.mask, 5, axis=3)
        # Zeroing coefficients corresponding to non-valid paths
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = tf.where(mask_, mat_t, tf.zeros_like(mat_t))

        return mat_t