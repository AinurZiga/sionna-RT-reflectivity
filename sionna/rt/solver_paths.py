#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# type: ignore
"""
Ray tracing algorithm that uses the image method to compute all pure reflection
paths.
"""

import mitsuba as mi
import drjit as dr
import tensorflow as tf
import time

from sionna.constants import SPEED_OF_LIGHT, PI
from sionna.utils.tensors import expand_to_rank, insert_dims, flatten_dims,\
    split_dim
from .paths import Paths, PathsTmpData, invert_s_t
from .utils import dot, phi_hat, theta_hat, theta_phi_from_unit_vec,\
    normalize, moller_trumbore, component_transform, mi_to_tf_tensor,\
        compute_field_unit_vectors, reflection_coefficient, fibonacci_lattice,\
            cot, cross, sign, rotation_matrix, acos_diff
from .solver_base import SolverBase
from .scattering_pattern import ScatteringPattern
from .diffraction_funcs import calc_angles, calc_angles_concave, transition_func


class SolverPaths(SolverBase):
    # pylint: disable=line-too-long
    r"""SolverPaths(scene, solver=None, dtype=tf.complex64)

    Generates propagation paths consisting of the line-of-sight (LoS) paths,
    specular, and diffracted paths for the currently loaded scene.

    The main inputs of the solver are:

    * A set of sources, from which rays are emitted.

    * A set of targets, at which rays are received.

    * A maximum depth, corresponding to the maximum number of reflections. A
    depth of zero corresponds to LoS.

    Generation of paths is carried-out for every link, i.e., for every pair of
    source and target.

    The genration of specular paths consists in three steps:

    1. A list of candidate paths is generated. A candidate consists in a
    sequence of primitives on which a ray emitted by a source sequentially
    reflects until it reaches a target.

    2. The image method is applied to every candidates (in parallel) to discard
    candidates that do not correspond to valid paths, either because they
    are obstructed by another object in the scene, or because a reflection on
    one of the primitive in the sequence is impossible (reflection point outside
    of the primitive).

    3. For the valid paths, Fresnel coefficients for reflections are computed,
    considering the materials of the intersected objects, to compute transfer
    matrices for every paths.

    For diffracted paths, after step 1.:

    2. The wedges of primitives in LoS are selected, i.e., primitives for which
    a direct connection with the sources was found at step 1

    3. The intersection point of the diffracted path on the wedge is computed.
    This is the point that minimizes the total length of the paths.
    Paths for which the diffraction point does not belong to the finite wedge
    are discarded.

    4. Obstruction test: Paths that are blocked are discarded.

    5. The transmition matrices are computed, as well as the delays and angles
    of arrival and departure.

    The output of the solver consists in, for every valid path that was found:

    * A transfer matrix, which is a 2x2 complex-valued matrix that describes the
    linear transformation incurred by the emitted field. The two dimensions
    correspond to the two polarization components (S and P).

    * A delay

    * Azimuth and zenith angles of arrival

    * Azimuth and zenith angles of departure

    Concerning the first step, two search methods are available for the
    listing of candidates:

    * Exhaustive search, which lists all possible combinations of primitives up
    to the requested maximum depth. This method is deterministic and ensures
    that all paths are found. However, its complexity increases exponentially
    with the number of primitives and with the maximum depth. Therefore, it
    only works for scenes of low complexity and/or for small depth values.

    * Fibonacci sampling, which find candidates by shooting and bouncing rays,
    and such that initial directions of rays shot from the sources are arranged
    in a Fibonacci lattice on the unit sphere. At every intersection with a
    primitive, the rays are bounced assuming perfectly specular reflections
    until the maximum depth is reached. The intersected primitives makes the
    candidate. This method can be applied to very large scenes. However, there
    is no guarantee that all possible paths are found.

    Note: Only triangle mesh are supported.

    Parameters
    -----------
    scene : :class:`~sionna.rt.Scene`
        Sionna RT scene

    solver : :class:`~sionna.rt.SolverBase` | None
        Another solver from which to re-use some structures to avoid useless
        compute and memory use

    dtype : tf.complex64 | tf.complex128
        Datatype for all computations, inputs, and outputs.
        Defaults to `tf.complex64`.

    Input
    ------
    max_depth : int
        Maximum depth (i.e., number of interaction with objects in the scene)
        allowed for tracing the paths.

    sources : [num_sources, 3], tf.float
        Coordinates of the sources.

    targets : [num_targets, 3], tf.float
        Coordinates of the targets.

    method : str ("exhaustive"|"fibonacci")
        Method to be used to list candidate paths.
        The "exhaustive" method tests all possible combination of primitives as
        paths. This method is not compatible with scattering.
        The "fibonacci" method uses a shoot-and-bounce approach to find
        candidate chains of primitives. Intial rays direction are arranged
        in a Fibonacci lattice on the unit sphere. This method can be
        applied to very large scenes. However, there is no guarantee that
        all possible paths are found.

    num_samples: int
        Number of random rays to trace in order to generate candidates.
        A large sample count may exhaust GPU memory.

    los : bool
        If set to `True`, then the LoS paths are computed.

    reflection : bool
        If set to `True`, then the reflected paths are computed.

    diffraction : bool
        If set to `True`, then the diffracted paths are computed.

    scattering : bool
        if set to `True`, then the scattered paths are computed.
        Only works with the Fibonacci method.

    scat_keep_prob : float
        Probability with which to keep scattered paths.
        This is helpful to reduce the number of scattered paths computed,
        which might be prohibitively high in some setup.
        Must be in the range (0,1).

    edge_diffraction : bool
        If set to `False`, only diffraction on wedges, i.e., edges that
        connect two primitives, is considered.

    scat_random_phases : bool
        If set to `True` and if scattering is enabled, random uniform phase
        shifts are added to the scattered paths.

    Output
    -------
    paths : Paths
        The computed paths.
    """


    def trace_paths(self, max_depth, method, num_samples, los, reflection, diffraction,
                 refl_diff, scattering, scat_keep_prob, scat_sparsity_coef, edge_diffraction, vertex_diffraction, 
                 double_diffraction):
        # pylint: disable=line-too-long
        r"""
        Traces the paths.

        Computes the trajectories of the paths by shooting rays.
        No EM field computation is performed by this function.

        Input
        ------
        max_depth : int
            Maximum depth (i.e., number of interaction with objects in the scene)
            allowed for tracing the paths.

        method : str ("exhaustive"|"fibonacci")
            Method to be used to list candidate paths.
            The "exhaustive" method tests all possible combination of primitives as
            paths. This method is not compatible with scattering.
            The "fibonacci" method uses a shoot-and-bounce approach to find
            candidate chains of primitives. Intial rays direction are arranged
            in a Fibonacci lattice on the unit sphere. This method can be
            applied to very large scenes. However, there is no guarantee that
            all possible paths are found.

        num_samples: int
            Number of random rays to trace in order to generate candidates.
            A large sample count may exhaust GPU memory.

        los : bool
            If set to `True`, then the LoS paths are computed.

        reflection : bool
            If set to `True`, then the reflected paths are computed.

        diffraction : bool
            If set to `True`, then the diffracted paths are computed.

        scattering : bool
            if set to `True`, then the scattered paths are computed.
            Only works with the Fibonacci method.

        scat_keep_prob : float
            Probability with which to keep scattered paths.
            This is helpful to reduce the number of scattered paths computed,
            which might be prohibitively high in some setup.
            Must be in the range (0,1).

        scat_sparsity_coef : int
            To make the diffuse scattered paths more sparse

        edge_diffraction : bool
            If set to `False`, only diffraction on wedges, i.e., edges that
            connect two primitives, is considered.

        Output
        -------
        spec_paths : Paths
            The computed specular paths

        diff_paths : Paths
            The computed diffracted paths

        scat_paths : Paths
            The computed scattered paths

        spec_paths_tmp : PathsTmpData
            Additional data required to compute the EM fields of the specular
            paths

        diff_paths_tmp : PathsTmpData
            Additional data required to compute the EM fields of the diffracted
            paths

        scat_paths_tmp : PathsTmpData
            Additional data required to compute the EM fields of the scattered
            paths
        """

        scat_keep_prob = tf.cast(scat_keep_prob, self._rdtype)
        # Disable scattering if the probability of keeping a path is 0
        scattering = tf.logical_and(scattering,
                    tf.greater(scat_keep_prob, tf.zeros_like(scat_keep_prob)))

        # If reflection and scattering are disabled, no need for a max_depth
        # higher than 1.
        # This clipping can save some compute for the shoot-and-bounce
        if (not reflection) and (not scattering):
            max_depth = tf.minimum(max_depth, 1)

        # Rotation matrices corresponding to the orientations of the radio
        # devices
        # rx_rot_mat : [num_rx, 3, 3]
        # tx_rot_mat : [num_tx, 3, 3]
        rx_rot_mat, tx_rot_mat = self._get_tx_rx_rotation_matrices()

        #################################################
        # Prepares the sources (from which rays are shot)
        # and targets (which capture the rays)
        #################################################

        if not self._scene.synthetic_array:
            # Relative positions of the antennas of the transmitters and
            # receivers
            # rx_rel_ant_pos: [num_rx, rx_array_size, 3], tf.float
            #     Relative positions of the receivers antennas
            # tx_rel_ant_pos: [num_tx, rx_array_size, 3], tf.float
            #     Relative positions of the transmitters antennas
            rx_rel_ant_pos, tx_rel_ant_pos =\
                self._get_antennas_relative_positions(rx_rot_mat, tx_rot_mat)

        # Transmitters and receivers positions
        # [num_tx, 3]
        tx_pos = [tx.position for tx in self._scene.transmitters.values()]
        tx_pos = tf.stack(tx_pos, axis=0)
        # [num_rx, 3]
        rx_pos = [rx.position for rx in self._scene.receivers.values()]
        rx_pos = tf.stack(rx_pos, axis=0)

        if self._scene.synthetic_array:
            # With synthetic arrays, each radio device corresponds to a single
            # endpoint (source or target)
            # [num_sources = num_tx, 3]
            sources = tx_pos
            # [num_targets = num_rx, 3]
            targets = rx_pos
        else:
            # [num_tx, tx_array_size, 3]
            sources = tf.expand_dims(tx_pos, axis=1) + tx_rel_ant_pos
            # [num_sources = num_tx*tx_array_size, 3]
            sources = tf.reshape(sources, [-1, 3])
            # [num_rx, rx_array_size, 3]
            targets = tf.expand_dims(rx_pos, axis=1) + rx_rel_ant_pos
            # [num_targets = num_rx*rx_array_size, 3]
            targets = tf.reshape(targets, [-1, 3])


        ##############################################
        # Generate candidate paths
        ##############################################
        # Candidate paths are generated according to the specified `method`.

        if method == 'exhaustive':
            if scattering:
                msg = "The exhaustive method is not compatible with scattering and mixed reflection-diffraction"
                raise ValueError(msg)
            # List all possible sequences of primitives with length up to
            # ``max_depth``
            # candidates: [max_depth, num_samples], int
            #     All possible candidate paths with depth up to ``max_depth``.
            # los_candidates: [num_samples], int
            #     Primitives in LoS. For the exhaustive method, this is the
            #     list of all the primitives in the scene.
            candidates, los_prim = self._list_candidates_exhaustive(max_depth, los, reflection)
            los_prim_t = los_prim
            candidates_t = candidates
            _candidates = candidates
            
        elif method == 'fibonacci':
            # Sample sequences of primitives using shoot-and-bounce
            # with length up to ``max_depth`` and by arranging the initial
            # rays direction in a Fibonacci lattice on the unit sphere.
            # candidates: [max_depth, num paths], int
            #     All unique candidate paths found, with depth up to
            #       ``max_depth``.
            # los_candidates: [num_samples], int
            #     Candidate primitives found in LoS.
            # candidates_scat : [max_depth, num_sources, num_paths_per_source]
            #       Sequence of primitives hit at `hit_points`.
            # hit_points : [max_depth, num_sources, num_paths_per_source, 3]
            #     Coordinates of the intersection points.
            output = self._list_candidates_fibonacci(max_depth,
                                        sources, num_samples, los, reflection,
                                        scattering, scat_sparsity_coef)
            candidates = output[0]
            los_prim = output[1]
            candidates_scat = output[2]
            hit_points = output[3]
            _candidates = candidates # _candidates[0] is reflection candidate for refl-diff and refl-VD

            if candidates.shape[0] != max_depth:
                print("Decreasing max_depth")
                max_depth = candidates.shape[0]

            #### 1) Add all facets from obj_geom
            if self.obj_geom is not None and self.obj_geom.add_obj_primitives and vertex_diffraction:
                _candidates = candidates # TODO: remove all obj_geom primitives

                los_prim_obj_geom = self._my_list_candidates_depth_1(sources, self.obj_geom.primitive_idxs)

                if max_depth == 1:
                    additive_prims = los_prim_obj_geom[None, ...]
                elif max_depth == 2:
                    additive_prims = tf.stack([los_prim_obj_geom, -1 * tf.ones_like(los_prim_obj_geom)], axis=0)
                elif max_depth == 3:
                    additive_prims = tf.stack([los_prim_obj_geom, -1 * tf.ones_like(los_prim_obj_geom), 
                                               -1 * tf.ones_like(los_prim_obj_geom)], axis=0)
                else:
                    print("Obj_geom is not supported for max_depth > 3")

                candidates = tf.concat([candidates, additive_prims], axis=-1)
                los_prim = tf.concat([los_prim, los_prim_obj_geom], axis=0)

                candidates, _ = tf.raw_ops.UniqueV2(
                    x=candidates,
                    axis=[1]
                )

            if (refl_diff or double_diffraction) and self.obj_geom.is_sbr_rxs:
                output_targets = self._list_candidates_fibonacci(2, targets, num_samples, False, reflection, False, 0)  # max_depth
                _candidates_t = output_targets[0] #_candidates_t[0] is reflection candidate for refl-diff and refl-VD
                los_prim_t = output_targets[1]
                if vertex_diffraction and self.obj_geom is not None and self.obj_geom.add_obj_primitives:
                    los_prim_obj_geom_t = self._my_list_candidates_depth_1(targets, self.obj_geom.primitive_idxs)
                    los_prim_t = tf.concat([los_prim_t, los_prim_obj_geom_t], axis=0) 

            if self.obj_geom and not self.obj_geom.is_sbr_rxs:
                _candidates_t = None

        else:
            raise ValueError(f"Unknown method '{method}'")

        los_prim = tf.gather(los_prim, tf.where(tf.not_equal(los_prim, -1))[:,0])
        los_prim, _ = tf.unique(x=los_prim)

        ##############################################
        # LoS and Specular paths
        ##############################################
        mirrored_vertices, tri_p0, normals = self.solver_reflection.get_images(sources, candidates)

        spec_paths, spec_paths_tmp = self.solver_reflection.tracing(sources,\
            targets, los, reflection, candidates, mirrored_vertices, tri_p0, normals)

        spec_paths.my_types = Paths.SPECULAR * tf.ones_like(spec_paths.objects, tf.int32)

        spec_paths.true_objects = spec_paths.objects

        # [num_targets, num_sources]
        if los and tf.shape(spec_paths.mask)[2] > 0:
            self.is_los = spec_paths.mask[..., :1]
        else:
            self.is_los = tf.zeros((targets.shape[0], sources.shape[0], 1), dtype=tf.bool)

        ############################################
        # Diffracted paths
        ############################################
        diff_paths, diff_paths_tmp = self.solver_wedge_diffraction.tracing(sources,targets, \
                                        los_prim, diffraction, edge_diffraction)
        
        diff_paths, diff_paths_tmp = self.solver_wedge_diffraction.check_obstructions( \
                diff_paths, diff_paths_tmp, diffraction)

        diff_paths.my_types = Paths.DIFFRACTED * tf.ones_like(diff_paths.objects, tf.int32)

        objects_mask = diff_paths.objects == -1
        valid_wedge_idxs = tf.where(objects_mask, 0, diff_paths.objects)
        diff_paths.true_objects = tf.gather(self._wedges_objects[:, 0], valid_wedge_idxs, axis=0)
            
        ############################################
        # Reflected + Diffracted paths
        ############################################

        ### 1) Tx -> reflection -> diffraction -> Rx
        mixed_paths = Paths(sources=sources, targets=targets, scene=self._scene, types=12)
        mixed_paths_tmp = PathsTmpData(sources, targets, self._dtype)

        if refl_diff and diffraction and reflection and max_depth >= 2:
            ## a) TX - refl - diff - RX
            if candidates.shape[0] >= 2:
                mixed_paths, mixed_paths_tmp = self._comb_diff_refl(mirrored_vertices, tri_p0, normals, _candidates, targets, sources)
                mixed_paths.dd_type = tf.ones(mixed_paths.mask.shape, tf.int32) * 11
                mixed_paths.my_types = tf.stack([Paths.SPECULAR * tf.ones_like(mixed_paths.mask, tf.int32), 
                                                  Paths.DIFFRACTED * tf.ones_like(mixed_paths.mask, tf.int32)], 
                                                  axis=0)
                
                true_objects_1 = mixed_paths.objects[0]
                valid_wedge_idxs = tf.where(mixed_paths.objects[1] == -1, 0, mixed_paths.objects[1])
                true_objects_2 = tf.gather(self._wedges_objects[:, 0], valid_wedge_idxs, axis=0)
                mixed_paths.true_objects = tf.stack([true_objects_1, true_objects_2], axis=0)

            
            ## b) TX - diff - refl - RX
            if _candidates_t is not None:
                mirrored_vertices_t, tri_p0_t, normals_t = self.solver_reflection.get_images(targets, _candidates_t)

                mixed_paths_t, mixed_paths_tmp_t = \
                    self._comb_diff_refl(mirrored_vertices_t, tri_p0_t, normals_t, _candidates_t, sources, targets)
                
                mixed_paths_t, mixed_paths_tmp_t = invert_s_t(mixed_paths_t, mixed_paths_tmp_t)
                mixed_paths_t.dd_type = tf.ones(mixed_paths_t.mask.shape, tf.int32) * 12
                mixed_paths_t.my_types = tf.stack([Paths.DIFFRACTED * tf.ones_like(mixed_paths_t.mask, tf.int32), 
                                                    Paths.SPECULAR * tf.ones_like(mixed_paths_t.mask, tf.int32)], 
                                                    axis=0)

                valid_wedge_idxs = tf.where(mixed_paths_t.objects[0] == -1, 0, mixed_paths_t.objects[0])
                true_objects_1 = tf.gather(self._wedges_objects[:, 0], valid_wedge_idxs, axis=0)
                true_objects_2 = mixed_paths_t.objects[1]
                mixed_paths_t.true_objects = tf.stack([true_objects_1, true_objects_2], axis=0)
                
                mixed_paths = merge_paths([mixed_paths, mixed_paths_t], True)
                mixed_paths_tmp = merge_paths_tmp([mixed_paths_tmp, mixed_paths_tmp_t])

            mixed_paths, mixed_paths_tmp =\
                self._compute_directions_distances_delays_angles(mixed_paths, mixed_paths_tmp, False)

        ##################
        # Reflection + Vertex_diffraction
        ##################
        vd_refl_paths = Paths(sources=sources, targets=targets, scene=self._scene, types=Paths.VD_REFL)
        vd_refl_paths_tmp = PathsTmpData(sources, targets, self._dtype)
        
        if refl_diff and vertex_diffraction and reflection and max_depth >= 2 and self.vertex_diffraction.is_refl_vd:  # TODO
            ## a) TX - refl - VD - RX
            if _candidates.shape[0] >= 2:
                vd_refl_paths, vd_refl_paths_tmp = self._comb_vd_refl(mirrored_vertices, tri_p0, _candidates, normals, targets, sources)
                vd_refl_paths.dd_type = tf.ones(vd_refl_paths.mask.shape, tf.int32) * 21
                vd_refl_paths.my_types = tf.stack([Paths.SPECULAR * tf.ones_like(vd_refl_paths.mask, tf.int32), 
                                                  Paths.VERTEX_DIFFRACTED * tf.ones_like(vd_refl_paths.mask, tf.int32)], 
                                                  axis=0)

                if vd_refl_paths.objects.shape[-1] != 0:
                    #vd_refl_paths.my_types = 
                    true_objects_1 = vd_refl_paths.objects[0]
                    valid_vertex_idxs = tf.where(vd_refl_paths.objects[1] == -1, 0, vd_refl_paths.objects[1])
                    true_objects_2 = tf.gather(self.vertex_diffraction.vertices_2_objects_sionna, valid_vertex_idxs)
                    vd_refl_paths.true_objects = tf.stack([true_objects_1, true_objects_2], axis=0)
            
            ## b) TX - VD - refl - RX
            if _candidates_t.shape[0] >= 2:
                vd_refl_paths_t, vd_refl_paths_tmp_t = self._comb_vd_refl(mirrored_vertices_t, tri_p0_t, 
                                                                    _candidates_t, normals_t, sources, targets)
                vd_refl_paths_t, vd_refl_paths_tmp_t = invert_s_t(vd_refl_paths_t, vd_refl_paths_tmp_t)
                vd_refl_paths_t.dd_type = tf.ones(vd_refl_paths_t.mask.shape, tf.int32) * 22
                vd_refl_paths_t.my_types = tf.stack([Paths.VERTEX_DIFFRACTED * tf.ones_like(vd_refl_paths_t.mask, tf.int32),
                                                    Paths.SPECULAR * tf.ones_like(vd_refl_paths_t.mask, tf.int32)], 
                                                    axis=0)

                if vd_refl_paths_t.objects.shape[-1] != 0:
                    valid_vertex_idxs = tf.where(vd_refl_paths_t.objects[0] == -1, 0, vd_refl_paths_t.objects[0])
                    true_objects_1 = tf.gather(self.vertex_diffraction.vertices_2_objects_sionna, valid_vertex_idxs)
                    true_objects_2 = vd_refl_paths_t.objects[1]
                    vd_refl_paths_t.true_objects = tf.stack([true_objects_1, true_objects_2], axis=0)

                vd_refl_paths = merge_paths([vd_refl_paths, vd_refl_paths_t], True)
                vd_refl_paths_tmp = merge_paths_tmp([vd_refl_paths_tmp, vd_refl_paths_tmp_t])

            vd_refl_paths, vd_refl_paths_tmp =\
                self._compute_directions_distances_delays_angles(vd_refl_paths, vd_refl_paths_tmp, False)
                

        ############################################
        # Double diffraction paths
        ############################################
        dd_paths = Paths(sources=sources, targets=targets, scene=self._scene, types=Paths.DOUBLE_DIFF)
        dd_paths_tmp = PathsTmpData(sources, targets, self._dtype)

        if (los_prim is not None) and double_diffraction:
            ###### 1)
            if self.double_diffraction.is_wedge_wedge or self.double_diffraction.is_wedge_vertex or \
                    self.double_diffraction.is_vertex_wedge or self.double_diffraction.is_vertex_vertex:
                wedges_1 = self._wedges_from_primitives(los_prim, True)
                wedges_2 = self._wedges_from_primitives(los_prim_t, True)
                dd_pairs = tf.zeros((2, tf.shape(wedges_1)[0], tf.shape(wedges_2)[0]), tf.int32)
                b1 = tf.repeat(wedges_1[..., None], tf.shape(wedges_2)[0], axis=1)
                b2 = tf.repeat(wedges_2[None, ...], tf.shape(wedges_1)[0], axis=0)
                # [2, num_wedges_1, num_wedges_2]
                dd_pairs = tf.tensor_scatter_nd_update(dd_pairs, [[0]], [b1])
                dd_pairs = tf.tensor_scatter_nd_update(dd_pairs, [[1]], [b2])
                # [num_wedges_1, num_wedges_2, 2]
                dd_pairs = tf.transpose(dd_pairs, perm=[1, 2, 0])
                # [num_pairs, 2]
                dd_pairs = tf.reshape(dd_pairs, [-1, 2])

                ## discard identical wedges
                non_duplicates_mask = dd_pairs[:, 0] != dd_pairs[:, 1]
                # [num_pairs, 2]
                dd_pairs = tf.boolean_mask(dd_pairs, non_duplicates_mask, axis=0)

                ### preliminary visibility
                dd_pairs = self.double_diffraction.check_preliminary_visibility(dd_pairs)

                ## Check if coplanar edges
                if self.double_diffraction.is_only_coplanar:
                    mask_coplanar = self.double_diffraction.get_coplanar(dd_pairs)
                    dd_pairs = tf.boolean_mask(dd_pairs, mask_coplanar, axis=0)
                    self.double_diffraction.dd_pairs_coplanar = dd_pairs

                if self.double_diffraction.is_wedge_vertex or self.double_diffraction.is_vertex_vertex:
                    _v_idxs = tf.gather(self.obj_geom.wedge_2_vertices, dd_pairs[:, 1])
                    v_idxs2 = tf.reshape(_v_idxs, -1)
                    wv_pairs = tf.stack((tf.repeat(dd_pairs[:, 0], 2), v_idxs2), axis=1)
                    wv_pairs, _ = tf.raw_ops.UniqueV2(x=wv_pairs, axis=[0])
                    active_vertices_dd_idxs = tf.cast(tf.where(self.obj_geom.active_vertices_dd)[:, 0], tf.int32)
                    wv_pairs = tf.boolean_mask(wv_pairs, tf.reduce_any(
                        tf.equal(wv_pairs[:, 1][..., None], active_vertices_dd_idxs[None, ...]), axis=1), axis=0)
                    
                
                if self.double_diffraction.is_vertex_wedge or self.double_diffraction.is_vertex_vertex:
                    _v_idxs = tf.gather(self.obj_geom.wedge_2_vertices, dd_pairs[:, 0])
                    v_idxs1 = tf.reshape(_v_idxs, -1)
                    vw_pairs = tf.stack((v_idxs1, tf.repeat(dd_pairs[:, 1], 2)), axis=1)
                    vw_pairs, _ = tf.raw_ops.UniqueV2(x=vw_pairs, axis=[0])
                    active_vertices_dd_idxs = tf.cast(tf.where(self.obj_geom.active_vertices_dd)[:, 0], tf.int32)
                    vw_pairs = tf.boolean_mask(vw_pairs, tf.reduce_any(
                        tf.equal(vw_pairs[:, 0][..., None], active_vertices_dd_idxs[None, ...]), axis=1), axis=0)
                    
                if self.double_diffraction.is_vertex_vertex:
                    vv_pairs = tf.stack((v_idxs1, v_idxs2), axis=1)
                    vv_pairs, _ = tf.raw_ops.UniqueV2(x=vv_pairs, axis=[0])
                    active_vertices_dd_idxs = tf.cast(tf.where(self.obj_geom.active_vertices_dd)[:, 0], tf.int32)
                    vv_pairs = tf.boolean_mask(vv_pairs, tf.reduce_any(
                        tf.equal(vv_pairs[:, 0][..., None], active_vertices_dd_idxs[None, ...]), axis=1), axis=0)

            if self.double_diffraction.is_wedge_wedge:
                ## Discard obstructed by wedges
                # [num_sources, num_targets, num_pairs, 2]
                dd_pair_idxs, active = self.double_diffraction.discard_obstructing(dd_pairs, sources, targets)
                
                if tf.shape(dd_pair_idxs)[2] != 0:
                    # Find diffraction points
                    # [num_sources, num_targets, num_pairs, 2, 3] and [num_sources, num_targets, num_pairs]
                    if self.double_diffraction.is_only_coplanar:
                        diff_points, valid = self.double_diffraction.compute_diffraction_points_analytical(sources, targets, dd_pair_idxs)
                    else:
                        diff_points, valid = self.double_diffraction.compute_diffraction_points(sources, targets, dd_pair_idxs)

                    active = tf.logical_and(active, valid)

                    if self.double_diffraction.is_only_close_2_transition:
                        # tf.transpose(diff_points, perm=[3, 1, 0, 2, 4])
                        valid = self.double_diffraction.close_2_transition(sources, targets, tf.transpose(diff_points, perm=[3, 0, 1, 2, 4]))
                        active = tf.logical_and(active, valid)

                    # Find diff points which are located on finite wedges
                    valid = self.double_diffraction.check_diff_points_validity(diff_points, dd_pair_idxs)
                    active = tf.logical_and(active, valid)

                    # Discard obstructed paths
                    valid = self.double_diffraction._check_visibility(sources, targets, diff_points, dd_pair_idxs)
                    active = tf.logical_and(active, valid)

                    dd_paths.objects = tf.transpose(dd_pair_idxs, perm=[3, 1, 0, 2])
                    dd_paths.vertices = tf.transpose(diff_points, perm=[3, 1, 0, 2, 4])
                    dd_paths.mask = tf.transpose(active, perm=[1, 0, 2]) #active

                    dd_paths = self.double_diffraction._gather_valid_paths(dd_paths)
                    dd_paths.dd_type = 0 * tf.ones(dd_paths.mask.shape, dtype=tf.int32)  # 0: double diffraction
                    dd_paths.my_types = tf.stack([Paths.DIFFRACTED * tf.ones_like(dd_paths.mask, tf.int32), 
                                                Paths.DIFFRACTED * tf.ones_like(dd_paths.mask, tf.int32)], 
                                                axis=0)

                    valid_wedge_idxs1 = tf.where(dd_paths.objects[0] == -1, 0, dd_paths.objects[0])
                    valid_wedge_idxs2 = tf.where(dd_paths.objects[1] == -1, 0, dd_paths.objects[1])
                    true_objects_1 = tf.gather(self._wedges_objects[:, 0], valid_wedge_idxs1, axis=0)
                    true_objects_2 = tf.gather(self._wedges_objects[:, 0], valid_wedge_idxs2, axis=0)
                    dd_paths.true_objects = tf.stack([true_objects_1, true_objects_2], axis=0)

                    dd_paths, dd_paths_tmp =\
                            self._compute_directions_distances_delays_angles(dd_paths, dd_paths_tmp, False)
                
            ############# 1a) shared vertex
            if self.double_diffraction.is_only_coplanar and self.double_diffraction.is_wedge_wedge_sv:
                dd_pair_idxs, active, valid_vertex_idxs = self.double_diffraction.shared_vertex(sources, targets, dd_pairs)
                valid_vertex_idxs = tf.where(valid_vertex_idxs == -1, 0, valid_vertex_idxs)

                if tf.reduce_any(active):
                    dd_paths_sv = Paths(sources=sources, targets=targets, scene=self._scene, types=Paths.DOUBLE_DIFF)
                    dd_paths_sv_tmp = PathsTmpData(sources, targets, self._dtype)

                    # [num_sources, num_targets, num_paths]
                    vertex_pos = tf.gather(self.obj_geom._vertices, valid_vertex_idxs)
                    self.double_diffraction.valid_vertex_idxs = valid_vertex_idxs

                    # check visibility (source-vertex and vertex-target)
                    valid = self.vertex_diffraction._check_visibility(sources, targets, vertex_pos)
                    active = tf.logical_and(active, valid)

                    # close to transition (abs(LOS_dist - dist) < 10 wavelengths)
                    if self.double_diffraction.is_only_close_2_transition:
                        # [num_sources, num_targets, num_paths]
                        los_dist = tf.norm(targets[None, :, None, :] - sources[:, None, None, :], axis=-1)
                        dist1 = tf.norm(vertex_pos - sources[:, None, None, :], axis=-1)
                        dist2 = tf.norm(targets[None, :, None, :] - vertex_pos, axis=-1)
                        close_2_transition = tf.abs(los_dist - (dist1 + dist2)) < 10 * self._scene._wavelength
                        active = tf.logical_and(active, close_2_transition)

                    # [2, num_sources, num_targets, num_paths]
                    ext_vertex_pos = tf.stack([vertex_pos, vertex_pos], axis=0)

                    dd_paths_sv.objects = tf.transpose(dd_pair_idxs, perm=[3, 1, 0, 2])
                    dd_paths_sv.vertices = tf.transpose(ext_vertex_pos, perm=[0, 2, 1, 3, 4])
                    dd_paths_sv.mask = tf.transpose(active, perm=[1, 0, 2])

                    dd_paths_sv = self.double_diffraction._gather_valid_paths(dd_paths_sv)
                    dd_paths_sv.dd_type = 4 * tf.ones(dd_paths_sv.mask.shape, dtype=tf.int32)  # 4: shared vertex
                    dd_paths_sv.my_types = tf.stack([Paths.VERTEX_DIFFRACTED * tf.ones_like(dd_paths_sv.mask, tf.int32), 
                                                     Paths.VERTEX_DIFFRACTED * tf.ones_like(dd_paths_sv.mask, tf.int32)], 
                                                     axis=0)

                    # if dd_paths_sv.objects.shape[-1] != 0:
                    valid_wedge_idxs1 = tf.where(dd_paths_sv.objects[0] == -1, 0, dd_paths_sv.objects[0])
                    valid_wedge_idxs2 = tf.where(dd_paths_sv.objects[1] == -1, 0, dd_paths_sv.objects[1])
                    true_objects_1 = tf.gather(self._wedges_objects[:, 0], valid_wedge_idxs1, axis=0)
                    true_objects_2 = tf.gather(self._wedges_objects[:, 0], valid_wedge_idxs2, axis=0)
                    dd_paths_sv.true_objects = tf.stack([true_objects_1, true_objects_2], axis=0)

                    dd_paths_sv, dd_paths_sv_tmp =\
                            self._compute_directions_distances_delays_angles(dd_paths_sv, dd_paths_sv_tmp, False, True)

                    dd_paths = merge_paths([dd_paths, dd_paths_sv]) 
                    dd_paths_tmp = merge_paths_tmp([dd_paths_tmp, dd_paths_sv_tmp])
            
            ######## 2) wedge_vertex
            if self.double_diffraction.is_wedge_vertex:
                dd_wedge_vertex = wv_pairs

                # [num_sources, num_targets, num_pairs, 2]
                dd_wv_idxs, active =\
                        self.double_diffraction.discard_obstructing_dd_wedge_vertex(dd_wedge_vertex, sources, targets)
                
                if self.double_diffraction.is_only_in_nlos:
                    # set to False all paths in LOS
                    active = tf.logical_and(active, ~tf.transpose(self.is_los, [1, 0, 2]))
                
                # [2, num_sources, num_targets, num_pairs, 3] and [num_sources, num_targets, num_pairs]
                diff_points2, valid2 = self.double_diffraction.compute_diffraction_points_wedge_vertex(sources, targets, dd_wv_idxs)
                active = tf.logical_and(active, valid2)

                ## discard if the diffraction points coincide
                diff_points2, valid2 = self.double_diffraction.omit_identical_points(diff_points2)
                active = tf.logical_and(active, valid2)

                # Discard obstructed paths
                valid2 = self.double_diffraction.check_visibility_mixed(sources, targets, diff_points2, dd_wv_idxs[..., 0], dd_wv_idxs[..., 1])
                active = tf.logical_and(active, valid2)

                ###
                if self.double_diffraction.is_only_close_2_transition:
                    valid2 = self.double_diffraction.close_2_transition(sources, targets, diff_points2)
                    active = tf.logical_and(active, valid2)

                if tf.reduce_any(active):
                    dd_paths_wv = Paths(sources=sources, targets=targets, scene=self._scene, types=Paths.DOUBLE_DIFF)
                    dd_paths_wv_tmp = PathsTmpData(sources, targets, self._dtype)

                    dd_paths_wv.objects = tf.transpose(dd_wv_idxs, perm=[3, 1, 0, 2])
                    dd_paths_wv.vertices = tf.transpose(diff_points2, perm=[0, 2, 1, 3, 4])
                    dd_paths_wv.mask = tf.transpose(active, perm=[1, 0, 2])

                    dd_paths_wv = self.double_diffraction._gather_valid_paths(dd_paths_wv)
                    dd_paths_wv.dd_type = 1 * tf.ones(dd_paths_wv.mask.shape, dtype=tf.int32)  # 1: wedge-vertex
                    dd_paths_wv.my_types = tf.stack([Paths.DIFFRACTED * tf.ones_like(dd_paths_wv.mask, tf.int32), 
                                                     Paths.VERTEX_DIFFRACTED * tf.ones_like(dd_paths_wv.mask, tf.int32)], 
                                                     axis=0)

                    if dd_paths_wv.objects.shape[-1] != 0:
                        valid_wedge_idxs = tf.where(dd_paths_wv.objects[0] == -1, 0, dd_paths_wv.objects[0])
                        true_objects_1 = tf.gather(self._wedges_objects[:, 0], valid_wedge_idxs, axis=0)
                        valid_vertex_idxs = tf.where(dd_paths_wv.objects[1] == -1, 0, dd_paths_wv.objects[1])
                        true_objects_2 = tf.gather(self.vertex_diffraction.vertices_2_objects_sionna, valid_vertex_idxs)
                        dd_paths_wv.true_objects = tf.stack([true_objects_1, true_objects_2], axis=0)

                    dd_paths_wv, dd_paths_wv_tmp =\
                            self._compute_directions_distances_delays_angles(dd_paths_wv, dd_paths_wv_tmp, False, True)

                    dd_paths = merge_paths([dd_paths, dd_paths_wv]) 
                    dd_paths_tmp = merge_paths_tmp([dd_paths_tmp, dd_paths_wv_tmp])

            # ######## 3) vertex_wedge
            if self.double_diffraction.is_vertex_wedge:

                # [num_sources, num_targets, num_pairs, 2]
                dd_vw_idxs, active =\
                        self.double_diffraction.discard_obstructing_dd_vertex_wedge(vw_pairs, sources, targets)

                if self.double_diffraction.is_only_in_nlos:
                    # set to False all paths in LOS
                    active = tf.logical_and(active, ~tf.transpose(self.is_los, [1, 0, 2]))
                
                # [2, num_sources, num_targets, num_pairs, 3] and [num_sources, num_targets, num_pairs]
                diff_points3, valid3 = self.double_diffraction.compute_diffraction_points_vertex_wedge(sources, targets, dd_vw_idxs)
                active = tf.logical_and(active, valid3)

                ## discard if the diffraction points coincide
                diff_points3, valid3 = self.double_diffraction.omit_identical_points(diff_points3)
                active = tf.logical_and(active, valid3)

                # Discard obstructed paths
                valid3 = self.double_diffraction.check_visibility_mixed(targets, sources, diff_points3[::-1], dd_vw_idxs[..., 1], dd_vw_idxs[..., 0])
                valid3 = tf.transpose(valid3, perm=[1, 0, 2])  
                active = tf.logical_and(active, valid3)

                if self.double_diffraction.is_only_close_2_transition:
                    valid3 = self.double_diffraction.close_2_transition(sources, targets, diff_points3)
                    active = tf.logical_and(active, valid3)      

                if tf.reduce_any(active):
                    dd_paths_vw = Paths(sources=sources, targets=targets, scene=self._scene, types=Paths.DOUBLE_DIFF)
                    dd_paths_vw_tmp = PathsTmpData(sources, targets, self._dtype)

                    dd_paths_vw.objects = tf.transpose(dd_vw_idxs, perm=[3, 1, 0, 2])
                    dd_paths_vw.vertices = tf.transpose(diff_points3, perm=[0, 2, 1, 3, 4])
                    dd_paths_vw.mask = tf.transpose(active, perm=[1, 0, 2])

                    dd_paths_vw = self.double_diffraction._gather_valid_paths(dd_paths_vw)
                    dd_paths_vw.dd_type = 2 * tf.ones(dd_paths_vw.mask.shape, dtype=tf.int32)  # 2: vertex-wedge

                    dd_paths_vw.my_types = tf.stack([Paths.VERTEX_DIFFRACTED * tf.ones_like(dd_paths_vw.mask, tf.int32), 
                                                     Paths.DIFFRACTED * tf.ones_like(dd_paths_vw.mask, tf.int32)], 
                                                     axis=0)
                    if dd_paths_vw.objects.shape[-1] != 0:
                        valid_vertex_idxs = tf.where(dd_paths_vw.objects[0] == -1, 0, dd_paths_vw.objects[0])
                        true_objects_1 = tf.gather(self.vertex_diffraction.vertices_2_objects_sionna, valid_vertex_idxs)
                        valid_wedge_idxs = tf.where(dd_paths_vw.objects[1] == -1, 0, dd_paths_vw.objects[1])
                        true_objects_2 = tf.gather(self._wedges_objects[:, 0], valid_wedge_idxs, axis=0)
                        dd_paths_vw.true_objects = tf.stack([true_objects_1, true_objects_2], axis=0)

                    dd_paths_vw, dd_paths_vw_tmp =\
                            self._compute_directions_distances_delays_angles(dd_paths_vw, dd_paths_vw_tmp, False, True)
                    
                    dd_paths = merge_paths([dd_paths, dd_paths_vw]) 
                    dd_paths_tmp = merge_paths_tmp([dd_paths_tmp, dd_paths_vw_tmp])

            # ######## 4) vertex_vertex
            if self.double_diffraction.is_vertex_vertex:
                # [num_sources, num_targets, num_pairs, 2]
                dd_vv_idxs, active =\
                        self.double_diffraction.discard_obstructing_dd_vertex_vertex(vv_pairs, sources, targets)
                
                # [2, num_sources, num_targets, num_pairs, 3] and [num_sources, num_targets, num_pairs]
                diff_points4, valid4 = self.double_diffraction.compute_diffraction_points_vertex_vertex(sources, targets, dd_vv_idxs)
                active = tf.logical_and(active, valid4)

                # Discard obstructed paths
                valid4 = self.double_diffraction.check_visibility_vertex_vertex(targets, sources, diff_points4[::-1], dd_vv_idxs[..., 1], dd_vv_idxs[..., 0])
                valid4 = tf.transpose(valid4, perm=[1, 0, 2])  
                active = tf.logical_and(active, valid4)

                if self.double_diffraction.is_only_close_2_transition:
                    valid4 = self.double_diffraction.close_2_transition(sources, targets, diff_points4)
                    active = tf.logical_and(active, valid4)

                if tf.reduce_any(active):
                    dd_paths_vv = Paths(sources=sources, targets=targets, scene=self._scene, types=Paths.DOUBLE_DIFF)
                    dd_paths_vv_tmp = PathsTmpData(sources, targets, self._dtype)

                    dd_paths_vv.objects = tf.transpose(dd_vv_idxs, perm=[3, 1, 0, 2])
                    dd_paths_vv.vertices = tf.transpose(diff_points4, perm=[0, 2, 1, 3, 4])
                    dd_paths_vv.mask = tf.transpose(active, perm=[1, 0, 2])

                    dd_paths_vv = self.double_diffraction._gather_valid_paths(dd_paths_vv)
                    dd_paths_vv.dd_type = 3 * tf.ones(dd_paths_vv.mask.shape, dtype=tf.int32)  # 3: vertex-vertex
                    dd_paths_vv.my_types = tf.stack([Paths.VERTEX_DIFFRACTED * tf.ones_like(dd_paths_vv.mask, tf.int32), 
                                                     Paths.VERTEX_DIFFRACTED * tf.ones_like(dd_paths_vv.mask, tf.int32)], 
                                                     axis=0)
                    
                    if dd_paths_vv.objects.shape[-1] != 0:
                        valid_vertex_idxs1 = tf.where(dd_paths_vv.objects[0] == -1, 0, dd_paths_vv.objects[0])
                        valid_vertex_idxs2 = tf.where(dd_paths_vv.objects[1] == -1, 0, dd_paths_vv.objects[1])
                        true_objects_1 = tf.gather(self.vertex_diffraction.vertices_2_objects_sionna, valid_vertex_idxs1)
                        true_objects_2 = tf.gather(self.vertex_diffraction.vertices_2_objects_sionna, valid_vertex_idxs2)
                        dd_paths_vv.true_objects = tf.stack([true_objects_1, true_objects_2], axis=0)

                    dd_paths_vv, dd_paths_vv_tmp =\
                            self._compute_directions_distances_delays_angles(dd_paths_vv, dd_paths_vv_tmp, False)
                    
                    dd_paths = merge_paths([dd_paths, dd_paths_vv]) 
                    dd_paths_tmp = merge_paths_tmp([dd_paths_tmp, dd_paths_vv_tmp])

        ############################################
        # Vertex diffracted paths
        ############################################
        vertex_diff_paths = Paths(sources=sources, targets=targets, scene=self._scene,
                           types=Paths.VERTEX_DIFFRACTED)
        vertex_diff_paths_tmp = PathsTmpData(sources, targets, self._dtype)

        if vertex_diffraction:
            vertex_diff_paths, vertex_diff_paths_tmp = self.vertex_diffraction.tracing(sources, targets)

            vertex_diff_paths.my_types = Paths.VERTEX_DIFFRACTED * tf.ones_like(vertex_diff_paths.objects, tf.int32)

            valid_vertex_idxs = tf.where(vertex_diff_paths.objects == -1, 0, vertex_diff_paths.objects)
            vertex_diff_paths.true_objects = tf.gather(self.vertex_diffraction.vertices_2_objects_sionna, valid_vertex_idxs)


        ############################################
        # Tripple diffracted paths
        ############################################
        ddd_paths = Paths(sources=sources, targets=targets, scene=self._scene, types=Paths.TRIPLE_DIFF)
        ddd_paths_tmp = PathsTmpData(sources, targets, self._dtype)

        if (los_prim is not None) and double_diffraction and self.double_diffraction.is_eee:

            wedges_1 = self._wedges_from_primitives(los_prim, False)
            wedges_2 = self._wedges_from_primitives(los_prim_t, False)

            tmp = (self.obj_geom.dd_wedges[:, 1][..., None] - self.obj_geom.dd_wedges[:, 0][None, ...]) == 0
            idxs = tf.where(tmp)
            _ddd_pairs = tf.concat([tf.gather(self.obj_geom.dd_wedges, idxs[:, 0], axis=0), 
                      tf.gather(self.obj_geom.dd_wedges[:, 1], idxs[:, 1], axis=0)[..., None]],
                      axis=-1)

            w1_bool = tf.reduce_any((wedges_1[..., None] - _ddd_pairs[:, 0][None, ...]) == 0, axis=0)
            w2_bool = tf.reduce_any((wedges_2[..., None] - _ddd_pairs[:, 2][None, ...]) == 0, axis=0)
            ddd_pairs = tf.boolean_mask(_ddd_pairs, tf.logical_and(w1_bool, w2_bool), axis=0)

            # [num_sources, num_targets, num_pairs, 2]
            ddd_pair_idxs, active = self.double_diffraction.ddd_discard_obstructing(ddd_pairs, sources, targets)

            diff_points, valid = self.double_diffraction.ddd_compute_diffraction_points(sources, targets, ddd_pair_idxs)
            active = tf.logical_and(active, valid)

            valid = self.double_diffraction.ddd_check_diff_points_validity(diff_points, ddd_pair_idxs)
            active = tf.logical_and(active, valid)

            ddd_paths.objects = tf.transpose(ddd_pair_idxs, perm=[3, 1, 0, 2])
            ddd_paths.vertices = tf.transpose(diff_points, perm=[3, 1, 0, 2, 4])
            ddd_paths.mask = tf.transpose(active, perm=[1, 0, 2]) #active

            ddd_paths = self.double_diffraction._gather_valid_paths(ddd_paths)

            ddd_paths.my_types = tf.stack([Paths.DIFFRACTED * tf.ones_like(ddd_paths.mask, tf.int32), 
                                            Paths.DIFFRACTED * tf.ones_like(ddd_paths.mask, tf.int32),
                                            Paths.DIFFRACTED * tf.ones_like(ddd_paths.mask, tf.int32)], 
                                            axis=0)
            
            if ddd_paths.objects.shape[-1] != 0:
                valid_wedge_idxs1 = tf.where(ddd_paths.objects[0] == -1, 0, ddd_paths.objects[0])
                valid_wedge_idxs2 = tf.where(ddd_paths.objects[1] == -1, 0, ddd_paths.objects[1])
                valid_wedge_idxs3 = tf.where(ddd_paths.objects[2] == -1, 0, ddd_paths.objects[2])
                true_objects_1 = tf.gather(self._wedges_objects[:, 0], valid_wedge_idxs1, axis=0)
                true_objects_2 = tf.gather(self._wedges_objects[:, 0], valid_wedge_idxs2, axis=0)
                true_objects_3 = tf.gather(self._wedges_objects[:, 0], valid_wedge_idxs3, axis=0)
                ddd_paths.true_objects = tf.stack([true_objects_1, true_objects_2, true_objects_3], axis=0)
            
            ddd_paths, ddd_paths_tmp =\
                    self._compute_directions_distances_delays_angles(ddd_paths, ddd_paths_tmp, False)


        ############################################
        # Scattered paths
        ############################################
        scat_paths = Paths(sources=sources, targets=targets, scene=self._scene,
                           types=Paths.SCATTERED)
        scat_paths_tmp = PathsTmpData(sources, targets, self._dtype)
        if scattering and tf.shape(candidates_scat)[0] > 0:

            scat_paths, scat_paths_tmp = self._scat_test_rx_blockage(targets,sources,
                                                                candidates_scat,
                                                                hit_points)

            scat_paths, scat_paths_tmp =\
                self._compute_directions_distances_delays_angles(scat_paths,
                                                                 scat_paths_tmp,
                                                                 True)

            scat_paths, scat_paths_tmp =\
                self._scat_discard_crossing_paths(scat_paths, scat_paths_tmp,
                                                  scat_keep_prob)

            # Extract the valid prefixes as paths
            scat_paths, scat_paths_tmp = self._scat_prefixes_2_paths(scat_paths,
                                                                scat_paths_tmp)

            scat_paths.my_types = Paths.SCATTERED * tf.ones_like(scat_paths.objects, tf.int32)

            scat_paths.true_objects = scat_paths.objects

        # Additional data required to compute the field
        spec_paths_tmp.num_samples = num_samples
        spec_paths_tmp.scat_keep_prob = tf.cast(scat_keep_prob, self._rdtype)
        diff_paths_tmp.num_samples = num_samples
        diff_paths_tmp.scat_keep_prob = tf.cast(scat_keep_prob, self._rdtype)
        mixed_paths_tmp.num_samples = num_samples
        mixed_paths_tmp.scat_keep_prob = tf.cast(scat_keep_prob, self._rdtype)
        scat_paths_tmp.num_samples = num_samples // scat_sparsity_coef
        scat_paths_tmp.scat_keep_prob = tf.cast(scat_keep_prob, self._rdtype)
        vertex_diff_paths_tmp.num_samples = num_samples
        vertex_diff_paths_tmp.scat_keep_prob = tf.cast(scat_keep_prob, self._rdtype)
        vd_refl_paths_tmp.num_samples = num_samples
        vd_refl_paths_tmp.scat_keep_prob = tf.cast(scat_keep_prob, self._rdtype)
        dd_paths_tmp.num_samples = num_samples
        dd_paths_tmp.scat_keep_prob = tf.cast(scat_keep_prob, self._rdtype)

        return spec_paths, diff_paths, mixed_paths, mixed_paths_tmp, scat_paths, spec_paths_tmp,\
            diff_paths_tmp, scat_paths_tmp, vertex_diff_paths, vertex_diff_paths_tmp,\
            vd_refl_paths, vd_refl_paths_tmp, dd_paths, dd_paths_tmp, ddd_paths, ddd_paths_tmp

    def compute_fields(self, spec_paths, diff_paths, mixed_paths, mixed_paths_tmp, 
        scat_paths, spec_paths_tmp,
        diff_paths_tmp, scat_paths_tmp, vertex_diff_paths, 
        vertex_diff_paths_tmp, vd_refl_paths, vd_refl_paths_tmp,\
              dd_paths, dd_paths_tmp, ddd_paths, ddd_paths_tmp, scat_random_phases, testing):
        r"""
        Computes the EM fields for a set of traced paths.

        Input
        ------
        spec_paths : Paths
            Specular paths

        diff_paths : Paths
            Diffracted paths

        scat_paths : Paths
            Scattered paths

        spec_paths_tmp : PathsTmpData
            Additional data required to compute the EM fields of the specular
            paths

        diff_paths_tmp : PathsTmpData
            Additional data required to compute the EM fields of the diffracted
            paths

        scat_paths_tmp : PathsTmpData
            Additional data required to compute the EM fields of the scattered
            paths

        scat_random_phases : bool
            If set to `True` and if scattering is enabled, random uniform phase
            shifts are added to the scattered paths.

        testing : bool
            If set to `True`, then additional data is returned for testing.

        Output
        -------
        sources : [num_sources, 3], tf.float
            Coordinates of the sources

        targets : [num_targets, 3], tf.float
            Coordinates of the targets

        list : Paths as a list
            The computed paths as a dictionary of tensors, i.e., the output of
            `Paths.to_dict()`.
            Returning the paths as a list of tensors is required to enable
            the execution of this function in graph mode.

        list : PathsTmpData as a list
            Additional data required to compute the EM fields of the specular
            paths as list of tensors.
            Only returned if `testing` is set to `True`.

        list : PathsTmpData as a list
            Additional data required to compute the EM fields of the diffracted
            paths as list of tensors.
            Only returned if `testing` is set to `True`.

        list : PathsTmpData as a list
            Additional data required to compute the EM fields of the scattered
            paths as list of tensors.
            Only returned if `testing` is set to `True`.
        """

        sources = spec_paths.sources
        targets = spec_paths.targets
        num_sources = tf.shape(sources)[0]
        num_targets = tf.shape(targets)[0]

        # Create empty paths object
        all_paths = Paths(sources=sources,
                          targets=targets,
                          scene=self._scene)
        # Create empty objects for storing tensors that are required to compute
        # paths, but that will not be returned to the user
        all_paths_tmp = PathsTmpData(sources, targets, self._dtype)

        # Rotation matrices corresponding to the orientations of the radio
        # devices
        # rx_rot_mat : [num_rx, 3, 3]
        # tx_rot_mat : [num_tx, 3, 3]
        rx_rot_mat, tx_rot_mat = self._get_tx_rx_rotation_matrices()

        # Number of receive antennas (not counting for dual polarization)
        tx_array_size = self._scene.tx_array.array_size
        # Number of transmit antennas (not counting for dual polarization)
        rx_array_size = self._scene.rx_array.array_size

        #################################################
        # Extract the material properties of the scene
        #################################################

        # Returns: relative_permittivities, denoted by `etas`,
        # scattering_coefficients, xpd_coefficients,
        # alpha_r, alpha_i, lambda_, and velocities
        object_properties = self._build_scene_object_properties_tensors()
        etas = object_properties[0]
        scattering_coefficient = object_properties[1]
        xpd_coefficient = object_properties[2]
        alpha_r = object_properties[3]
        alpha_i = object_properties[4]
        lambda_ = object_properties[5]
        velocity = object_properties[6]
        self.rot_motion_point_on_axis = object_properties[7]
        self.rot_motion_axis = object_properties[8]
        self.rot_motion_angular_velocity  = object_properties[9]

        ##############################################
        # LoS and Specular paths
        ##############################################

        if spec_paths.objects.shape[3] > 0:
            # Compute the EM transition matrices and Doppler shifts
            spec_mat_t = self.solver_reflection._spec_transition_matrices(etas, 
                                scattering_coefficient, spec_paths, spec_paths_tmp, False)
            
            spec_paths.doppler = self._compute_doppler_shifts(spec_paths,
                                                              spec_paths_tmp,
                                                              velocity)
            all_paths = all_paths.merge(spec_paths)
            # Only the transition matrix, vector of incidence/reflection, and
            # Doppler shifts are required for the computation of the paths
            # coefficients
            all_paths_tmp.mat_t = tf.concat([all_paths_tmp.mat_t, spec_mat_t],
                                            axis=-3)
            all_paths_tmp.k_tx = tf.concat([all_paths_tmp.k_tx,
                                            spec_paths_tmp.k_tx],
                                           axis=-2)
            all_paths_tmp.k_rx = tf.concat([all_paths_tmp.k_rx,
                                            spec_paths_tmp.k_rx],
                                           axis=-2)
            # If testing, the transition matrices are also returned
            if testing:
                spec_paths_tmp.mat_t = spec_mat_t

        ############################################
        # Diffracted paths
        ############################################

        if diff_paths.objects.shape[3] > 0:
            # Compute the transition matrices and Doppler shifts
            diff_mat_t =\
                self._compute_diffraction_transition_matrices(etas,
                            scattering_coefficient, diff_paths, diff_paths_tmp)
            diff_paths.doppler = self._compute_doppler_shifts(diff_paths,
                                                              diff_paths_tmp,
                                                              velocity)
            all_paths = all_paths.merge(diff_paths)
            # Only the transition matrix and vector of incidence/reflection are
            # required for the computation of the paths coefficients
            all_paths_tmp.mat_t = tf.concat([all_paths_tmp.mat_t, diff_mat_t],
                                            axis=-3)
            all_paths_tmp.k_tx = tf.concat([all_paths_tmp.k_tx,
                                            diff_paths_tmp.k_tx],
                                           axis=-2)
            all_paths_tmp.k_rx = tf.concat([all_paths_tmp.k_rx,
                                            diff_paths_tmp.k_rx],
                                           axis=-2)
            # If testing, the transition matrices are also returned
            if testing:
                diff_paths_tmp.mat_t = diff_mat_t

        ############################################
        # Reflection-Diffraction
        ############################################
        if mixed_paths.objects.shape[3] > 0:
            ### TX-Refl-Diff-RX
            mixed_mat_t = self.solver_wedge_diffraction._compute_refl_diff(etas, scattering_coefficient, mixed_paths, mixed_paths_tmp)
            mixed_paths.doppler = self._compute_doppler_shifts(mixed_paths,
                                                              mixed_paths_tmp,
                                                              velocity)

            all_paths = all_paths.merge(mixed_paths)

            all_paths_tmp.mat_t = tf.concat([all_paths_tmp.mat_t, mixed_mat_t],
                                            axis=-3)
            all_paths_tmp.k_tx = tf.concat([all_paths_tmp.k_tx,
                                            mixed_paths_tmp.k_tx],
                                           axis=-2)
            all_paths_tmp.k_rx = tf.concat([all_paths_tmp.k_rx,
                                            mixed_paths_tmp.k_rx],
                                           axis=-2)

        ############################################
        # Reflection-VD
        ############################################
        if vd_refl_paths.objects.shape[3] > 0:
            ### TX-Refl-VD-RX
            vd_refl_paths_mat_t = self.vertex_diffraction.compute_refl_vd(etas, scattering_coefficient, vd_refl_paths, vd_refl_paths_tmp)
            vd_refl_paths.doppler = self._compute_doppler_shifts(vd_refl_paths,
                                                              vd_refl_paths_tmp,
                                                              velocity)

            all_paths = all_paths.merge(vd_refl_paths)

            all_paths_tmp.mat_t = tf.concat([all_paths_tmp.mat_t, vd_refl_paths_mat_t],
                                            axis=-3)
            all_paths_tmp.k_tx = tf.concat([all_paths_tmp.k_tx,
                                            vd_refl_paths_tmp.k_tx],
                                           axis=-2)
            all_paths_tmp.k_rx = tf.concat([all_paths_tmp.k_rx,
                                            vd_refl_paths_tmp.k_rx],
                                           axis=-2)

        ############################################
        # Vertex-diffracted paths
        ############################################
        #    
        if vertex_diff_paths.objects.shape[3] > 0:
            vertex_diff_mat_t =\
                self.vertex_diffraction.compute_fields(etas,
                            scattering_coefficient, vertex_diff_paths, vertex_diff_paths_tmp)
            
            vertex_diff_paths.doppler = self._compute_doppler_shifts(vertex_diff_paths,
                                                              vertex_diff_paths_tmp,
                                                              velocity)
            
            all_paths = all_paths.merge(vertex_diff_paths)
            # Only the transition matrix and vector of incidence/reflection are
            # required for the computation of the paths coefficients
            all_paths_tmp.mat_t = tf.concat([all_paths_tmp.mat_t, vertex_diff_mat_t],
                                            axis=-3)
            all_paths_tmp.k_tx = tf.concat([all_paths_tmp.k_tx,
                                            vertex_diff_paths_tmp.k_tx],
                                           axis=-2)
            all_paths_tmp.k_rx = tf.concat([all_paths_tmp.k_rx,
                                            vertex_diff_paths_tmp.k_rx],
                                           axis=-2)
            
        
        ############################################
        # Double-diffracted paths
        ############################################
        if dd_paths.objects.shape[3] > 0:
            # Compute the transition matrices and Doppler shifts
            dd_mat_t =\
                self.double_diffraction.compute_fields(etas,
                            scattering_coefficient, dd_paths, dd_paths_tmp)
            dd_paths.doppler = self._compute_doppler_shifts(dd_paths,
                                                              dd_paths_tmp,
                                                              velocity)
            all_paths = all_paths.merge(dd_paths)
            # Only the transition matrix and vector of incidence/reflection are
            # required for the computation of the paths coefficients
            all_paths_tmp.mat_t = tf.concat([all_paths_tmp.mat_t, dd_mat_t],
                                            axis=-3)
            all_paths_tmp.k_tx = tf.concat([all_paths_tmp.k_tx,
                                            dd_paths_tmp.k_tx],
                                           axis=-2)
            all_paths_tmp.k_rx = tf.concat([all_paths_tmp.k_rx,
                                            dd_paths_tmp.k_rx],
                                           axis=-2)
            # If testing, the transition matrices are also returned
            if testing:
                dd_paths_tmp.mat_t = dd_mat_t

        ############################################
        # Triple-diffracted paths
        ############################################
        if ddd_paths.objects.shape[3] > 0:
            # Compute the transition matrices and Doppler shifts
            ddd_mat_t =\
                self.double_diffraction.ddd_compute_fields(ddd_paths, ddd_paths_tmp)
            ddd_paths.doppler = self._compute_doppler_shifts(ddd_paths,
                                                              ddd_paths_tmp,
                                                              velocity)
            all_paths = all_paths.merge(ddd_paths)
            # Only the transition matrix and vector of incidence/reflection are
            # required for the computation of the paths coefficients
            all_paths_tmp.mat_t = tf.concat([all_paths_tmp.mat_t, ddd_mat_t],
                                            axis=-3)
            all_paths_tmp.k_tx = tf.concat([all_paths_tmp.k_tx,
                                            ddd_paths_tmp.k_tx],
                                           axis=-2)
            all_paths_tmp.k_rx = tf.concat([all_paths_tmp.k_rx,
                                            ddd_paths_tmp.k_rx],
                                           axis=-2)

        ############################################
        # Scattered paths
        ############################################

        if scat_paths.objects.shape[3] > 0:
            # Compute transition matrices up to the scattering point
            # as well as Doppler shifts
            scat_mat_t = self.solver_reflection._spec_transition_matrices(etas, 
                                    scattering_coefficient, scat_paths, scat_paths_tmp, True)
            
            scat_paths.doppler = self._compute_doppler_shifts(scat_paths,
                                                              scat_paths_tmp,
                                                              velocity)

            all_paths = all_paths.merge(scat_paths)
            # The transition matrix and vector of incidence/reflection are
            # required for the computation of the paths coefficients, as well
            # as other scattering specific quantities.
            all_paths_tmp.mat_t = tf.concat([all_paths_tmp.mat_t, scat_mat_t],
                                            axis=-3)
            all_paths_tmp.k_tx = tf.concat([all_paths_tmp.k_tx,
                                            scat_paths_tmp.k_tx],
                                           axis=-2)
            all_paths_tmp.k_rx = tf.concat([all_paths_tmp.k_rx,
                                            scat_paths_tmp.k_rx],
                                           axis=-2)
            all_paths_tmp.scat_last_objects = scat_paths_tmp.scat_last_objects
            all_paths_tmp.scat_last_k_i = scat_paths_tmp.scat_last_k_i
            all_paths_tmp.scat_k_s = scat_paths_tmp.scat_k_s
            all_paths_tmp.scat_last_normals = scat_paths_tmp.scat_last_normals
            all_paths_tmp.scat_src_2_last_int_dist\
                                = scat_paths_tmp.scat_src_2_last_int_dist
            all_paths_tmp.scat_2_target_dist = scat_paths_tmp.scat_2_target_dist
            all_paths_tmp.scat_last_vertices = scat_paths_tmp.scat_last_vertices
            # If testing, the transition matrices are also returned
            if testing:
                scat_paths_tmp.mat_t = scat_mat_t

        #################################################
        # Splitting the sources (targets) dimension into
        # transmitters (receivers) and antennas, or
        # applying the synthetic arrays
        #################################################

        # If not using synthetic array, then the paths for the different
        # antenna elements were generated and reshaping is needed.
        # Otherwise, expand with the antenna dimensions.
        # [num_targets, num_sources, max_num_paths]
        all_paths.targets_sources_mask = all_paths.mask
        if self._scene.synthetic_array:
            # [num_rx, num_tx, 2, 2]
            mat_t = all_paths_tmp.mat_t
            # [num_rx, 1, num_tx, 1, max_num_paths, 2, 2]
            mat_t = tf.expand_dims(tf.expand_dims(mat_t, axis=1), axis=3)
            all_paths_tmp.mat_t = mat_t
        else:
            num_rx = len(self._scene.receivers)
            num_tx = len(self._scene.transmitters)
            max_num_paths = tf.shape(all_paths.vertices)[3]
            batch_dims = [num_rx, rx_array_size, num_tx, tx_array_size,
                          max_num_paths]
            # [num_rx, tx_array_size, num_tx, tx_array_size, max_num_paths]
            all_paths.mask = tf.reshape(all_paths.mask, batch_dims)
            all_paths.tau = tf.reshape(all_paths.tau, batch_dims)
            all_paths.theta_t = tf.reshape(all_paths.theta_t, batch_dims)
            all_paths.phi_t = tf.reshape(all_paths.phi_t, batch_dims)
            all_paths.theta_r = tf.reshape(all_paths.theta_r, batch_dims)
            all_paths.phi_r = tf.reshape(all_paths.phi_r, batch_dims)
            all_paths.doppler = tf.reshape(all_paths.doppler, batch_dims)
            # [num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths, 2,2]
            all_paths_tmp.mat_t = tf.reshape(all_paths_tmp.mat_t,
                                             batch_dims + [2,2])
            # [num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths, 3]
            all_paths_tmp.k_tx = tf.reshape(all_paths_tmp.k_tx, batch_dims+[3])
            all_paths_tmp.k_rx = tf.reshape(all_paths_tmp.k_rx, batch_dims+[3])
        ####################################################
        # Compute the channel coefficients
        ####################################################
        scat_keep_prob = scat_paths_tmp.scat_keep_prob
        num_samples = scat_paths_tmp.num_samples
        all_paths.a = self._compute_paths_coefficients(rx_rot_mat,
                                                       tx_rot_mat,
                                                       all_paths,
                                                       all_paths_tmp,
                                                       num_samples,
                                                       scattering_coefficient,
                                                       xpd_coefficient,
                                                       etas, alpha_r, alpha_i,
                                                       lambda_, scat_keep_prob,
                                                       scat_random_phases)

        # If using synthetic array, adds the antenna dimentions by applying
        # synthetic phase shifts
        if self._scene.synthetic_array:
            all_paths.a = self._apply_synthetic_array(rx_rot_mat, tx_rot_mat,
                                                      all_paths, all_paths_tmp)

        ##################################################
        # If not using synthetic arrays, tile the AoAs,
        # AoDs, and delays to handle dual-polarization
        ##################################################
        if not self._scene.synthetic_array:
            num_rx_patterns = len(self._scene.rx_array.antenna.patterns)
            num_tx_patterns = len(self._scene.tx_array.antenna.patterns)
            # [num_rx, 1,rx_array_size, num_tx, 1,tx_array_size, max_num_paths]
            mask = tf.expand_dims(tf.expand_dims(all_paths.mask, axis=2),
                                 axis=5)
            tau = tf.expand_dims(tf.expand_dims(all_paths.tau, axis=2),
                                 axis=5)
            theta_t = tf.expand_dims(tf.expand_dims(all_paths.theta_t, axis=2),
                                     axis=5)
            phi_t = tf.expand_dims(tf.expand_dims(all_paths.phi_t, axis=2),
                                   axis=5)
            theta_r = tf.expand_dims(tf.expand_dims(all_paths.theta_r, axis=2),
                                     axis=5)
            phi_r = tf.expand_dims(tf.expand_dims(all_paths.phi_r, axis=2),
                                   axis=5)
            doppler = tf.expand_dims(tf.expand_dims(all_paths.doppler, axis=2),
                                   axis=5)
            # [num_rx, num_rx_patterns, rx_array_size, num_tx, num_tx_patterns,
            #   tx_array_size, max_num_paths]
            mask = tf.tile(mask, [1, num_rx_patterns, 1, 1, num_tx_patterns,
                                  1, 1])
            tau = tf.tile(tau, [1, num_rx_patterns, 1, 1, num_tx_patterns,
                                1, 1])
            theta_t = tf.tile(theta_t, [1, num_rx_patterns, 1, 1,
                                        num_tx_patterns, 1, 1])
            phi_t = tf.tile(phi_t, [1, num_rx_patterns, 1, 1,
                                    num_tx_patterns, 1, 1])
            theta_r = tf.tile(theta_r, [1, num_rx_patterns, 1, 1,
                                        num_tx_patterns, 1, 1])
            phi_r = tf.tile(phi_r, [1, num_rx_patterns, 1, 1,
                                    num_tx_patterns, 1, 1])
            doppler = tf.tile(doppler, [1, num_rx_patterns, 1, 1,
                                    num_tx_patterns, 1, 1])
            # [num_rx, num_rx_ant = num_rx_patterns*num_rx_ant,
            #   ... num_tx, num_tx_ant = num_tx_patterns*tx_array_size,
            #   ... max_num_paths]
            all_paths.mask = flatten_dims(flatten_dims(mask, 2, 1), 2, 3)
            all_paths.tau = flatten_dims(flatten_dims(tau, 2, 1), 2, 3)
            all_paths.theta_t = flatten_dims(flatten_dims(theta_t, 2, 1), 2, 3)
            all_paths.phi_t = flatten_dims(flatten_dims(phi_t, 2, 1), 2, 3)
            all_paths.theta_r = flatten_dims(flatten_dims(theta_r, 2, 1), 2, 3)
            all_paths.phi_r = flatten_dims(flatten_dims(phi_r, 2, 1), 2, 3)
            all_paths.doppler = flatten_dims(flatten_dims(doppler, 2, 1), 2, 3)

        # If testing, additinal data is returned
        if testing:
            output = (  sources, targets, all_paths.to_dict(),
                        # For testing
                        spec_paths_tmp.to_dict(),
                        diff_paths_tmp.to_dict(),
                        scat_paths_tmp.to_dict() )
        else:
            output = (sources, targets, all_paths.to_dict())
        return output

    ##################################################################
    # Methods for finding candiate primitives and edges for reflected
    # and diffracted paths
    ##################################################################

    def _list_candidates_exhaustive(self, max_depth, los, reflection):
        r"""
        Generate all possible candidate paths made of reflections only and the
        LoS.

        The number of candidate paths equals

            num_triangles**max_depth + 1

        where the additional path (+1) is the LoS.

        This can easily exhaust GPU memory if the number of triangles in the
        scene or the `max_depth` are too large.

        Input
        ------
        max_depth: int
            Maximum number of reflections.
            Set to 0 for LoS only.

        los : bool
            Set if the LoS paths are computed.

        reflection : bool
            Set if the reflected paths are computed.

        Output
        -------
        candidates: [max_depth, num_samples], int
            All possible candidate paths with depth up to ``max_depth``.
            Entries correspond to primitives indices.
            For paths with depth lower than ``max_depth``, -1 is used as
            padding value.
            The first path is the LoS one if LoS is requested.

        los_candidates: [num_samples], int or `None`
            Candidates in LoS. For the exhaustive method, this is the list of
            all candidates. `None` is returned if ``max_depth`` is 0 or for
            empty scenes.
        """
        # Number of triangles
        n_prims = self._primitives.shape[0]

        # List of all triangles
        # [n_prims]
        all_prims = tf.range(n_prims, dtype=tf.int32)

        # Empty scene or reflection disabled
        if (not reflection) or (n_prims == 0):
            if los:
                # Only LoS is added as candidate
                return tf.fill([0,1], -1), all_prims
            else:
                # No candidates
                return tf.fill([0,0], -1), all_prims

        # If reflection is disabled,

        # Number of candidate paths made of reflections only
        # num_samples = n_prims + n_prims^2 + ... + n_prims^max_depth
        if n_prims == 0:
            num_samples = 0
        elif n_prims == 1:
            num_samples = max_depth
        else:
            num_samples = (n_prims * (n_prims ** max_depth - 1))//(n_prims - 1)
        # Add LoS path
        if los:
            num_samples += 1
        # Tensor of all possible reflections
        # Shape : [max_depth , num_samples]
        # It is transposed to fit the expected output shape at the end of this
        # function.
        # all_candidates[i,j] correspond to the triangle index intersected
        # by the i^th path for at j^th reflection.
        # The first column corresponds to LoS, i.e., no interaction.
        # -1 is used as padding value for path with depth lower than
        # max_depth.
        # Initialized with -1.
        all_candidates = tf.fill([num_samples, max_depth], -1)
        # The next loop fill all_candidates with the list of intersected
        # primitives for all possible paths made of reflections only.
        # It starts from the paths with the 1 reflection, up to max_depth.
        # The variable `offset` corresponds to the index offset for storing the
        # paths in all_candidates.
        if los:
            # `offset` is initialized to 1 as the first path (depth = 0)
            # corresponds to LoS
            offset = 1
        else:
            # No LoS, `offset` is initialized to 0
            offset = 0
        for depth in range(1, max_depth+1):
            # Enumerate all possible interactions for this depth
            # List of `depth` tensors with shape
            # [n_prims, ..., n_prims] and rank `depth`
            candidates = tf.meshgrid(*([all_prims] * depth), indexing='ij')

            # Reshape to
            # [n_prims**depth,depth]
            candidates = tf.stack([tf.reshape(c, [-1]) for c in candidates],
                                    axis=1)

            # Pad with -1 for paths shorter than max_depth
            # [n_prims**depth,max_depth]
            candidates = tf.pad(candidates, [[0,0],[0,max_depth-depth]],
                                mode='CONSTANT', constant_values=-1)

            # Update all_candidates
            # Number of candidate paths for this depth
            num_candidates = candidates.shape[0]
            # Corresponding row indices in the all_candidates tensor
            indices = tf.range(offset, offset+num_candidates, dtype=tf.int32)
            indices = tf.expand_dims(indices, -1)
            # all_candidates : [max_depth , num_samples]
            all_candidates = tf.tensor_scatter_nd_update(all_candidates,
                                                         indices, candidates)

            # Prepare for next iteration
            offset += num_candidates

        # Transpose to fit the expected output shape.
        # [max_depth, num_samples]
        all_candidates = tf.transpose(all_candidates)

        # Primitives in LoS
        if max_depth > 0:
            los_candidates = all_prims
        else:
            los_candidates = None

        return all_candidates, los_candidates
    
    def _my_list_candidates_depth_1(self, sources, primitive_idxs=None):
        # self.obj
        # primitives_idxs = tf.gather(self._primitives_2_objects, obj_ids)
        if primitive_idxs is None:
            primitive_idxs = tf.range(tf.shape(self._primitives)[0], dtype=tf.int32)

        #primitives = self._primitives
        primitives = tf.gather(self._primitives, primitive_idxs)

        num_primitives = tf.shape(primitives)[0]
        num_sources = tf.shape(sources)[0]
        #primitive_idxs = tf.range(num_primitives, dtype=tf.int32)
            
        # [num_triangles, 3]
        c_point = tf.reduce_mean(primitives, axis=1)
        # [num_sources, num_triangles, 3]
        c_point = tf.repeat(c_point[None, ...], num_sources, axis=0)
        # [batch_size, 3]
        c_point = tf.reshape(c_point, [-1, 3])

        # [num_sources, num_triangles, 3]
        sources = tf.repeat(sources[:, None, :], num_primitives, axis=1)
        # [batch_size, 3]
        sources = tf.reshape(sources, [-1, 3])

        # Check visibility between c_point and sources
        # Ray origin
        # d : [batch_size, 3]
        # maxt : [batch_size]
        d, maxt = tf.linalg.normalize(c_point - sources, axis=1)
        maxt = tf.squeeze(maxt, axis=1)
        # [batch_size]
        valid_primitives = tf.logical_not(self._test_obstruction(sources, d, maxt))
        # [num_sources, num_triangles]
        valid_primitives = tf.reshape(valid_primitives, [num_sources, num_primitives])

        # [num_triangles]
        valid_primitives_all_sources = tf.reduce_any(valid_primitives, axis=0)
        los_candidate_idxs = tf.cast(tf.where(valid_primitives_all_sources)[:, 0], tf.int32)
        los_candidates = tf.gather(primitive_idxs, los_candidate_idxs)

        return los_candidates

    def _list_candidates_fibonacci(self, max_depth, sources, num_samples,
                                   los, reflection, scattering, scat_sparsity_coef):
        r"""
        Generate potential candidate paths made of reflections only and the
        LoS. Rays direction are arranged in a Fibonacci lattice on the unit
        sphere.

        This can be used when the triangle count or maximum depth make the
        exhaustive method impractical.

        A budget of ``num_samples`` rays is split equally over the given
        sources. Starting directions are sampled uniformly at random.
        Paths are simulated until the maximum depth is reached.
        We record all sequences of primitives hit and the prefixes of these
        sequences, and return unique sequences.

        Input
        ------
        max_depth: int
            Maximum number of reflections.
            Set to 0 for LoS only.

        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        num_samples: int
            Number of rays to trace in order to generate candidates.
            A large sample count may exhaust GPU memory.

        los : bool
            If set to `True`, then the LoS paths are computed.

        reflection : bool
            If set to `True`, then the reflected paths are computed.

        scattering : bool
            if set to `True`, then the scattered paths are computed

        scat_sparsity_coef : int


        Output
        -------
        candidates_ref: [max_depth, num paths], int
            Unique sequence of hitted primitives, with depth up to ``max_depth``.
            Entries correspond to primitives indices.
            For paths with depth lower than max_depth, -1 is used as
            padding value.
            The first path is the LoS one if LoS is requested.

        los_candidates: [num_samples], int or `None`
            Primitives in LoS. `None` is returned if ``max_depth`` is 0.

        candidates_scat : [max_depth, num_sources, num_paths_per_source], int
            Sequence of primitives hit at `hit_points`. Compared to
            `candidates_ref`, it does not need to be unique, as the
            intersection points are different for every sequence, and is
            dependant on the source, as the intersection point are specific to
            the sources positions.

        hit_points : [max_depth, num_sources, num_paths_per_source, 3], tf.float
            Intersection points.
        """
        mask_t = dr.mask_t(self._mi_scalar_t)

        # Ensure that sample count can be distributed over the emitters
        num_sources = sources.shape[0]
        samples_per_source = int(dr.ceil(num_samples / num_sources))
        num_samples = num_sources * samples_per_source

        # List of candidates
        candidates = []

        # Hit points
        hit_points = []

        # Is the scene empty?
        is_empty = dr.shape(self._shape_indices)[0] == 0

        # Only shoot if the scene is not empty
        if not is_empty:

            # Keep track of which paths are still active
            active = dr.full(mask_t, True, num_samples)

            # Initial ray: Arranged in a Fibonacci lattice on the unit
            # sphere.
            # [samples_per_source, 3]
            lattice = fibonacci_lattice(samples_per_source, self._rdtype)
            sampled_d = tf.tile(lattice, [num_sources, 1])
            sampled_d = self._mi_point2_t(sampled_d)
            sampled_d = mi.warp.square_to_uniform_sphere(sampled_d)
            source_i = dr.linspace(self._mi_scalar_t, 0, num_sources,
                                   num=num_samples, endpoint=False)
            source_i = mi.Int32(source_i)
            sources_dr = self._mi_tensor_t(sources)
            ray = mi.Ray3f(
                o=dr.gather(self._mi_vec_t, sources_dr.array, source_i),
                d=sampled_d,
            )

            for depth in range(max_depth):

                # Intersect ray against the scene to find the next hitted
                # primitive
                si = self._mi_scene.ray_intersect(ray, active)

                active &= si.is_valid()

                # Record which primitives were hit
                shape_i = dr.gather(mi.Int32, self._shape_indices,
                                    dr.reinterpret_array_v(mi.UInt32, si.shape),
                                    active)
                offsets = dr.gather(mi.Int32, self._prim_offsets, shape_i,
                                    active)
                prims_i = dr.select(active, offsets + si.prim_index, -1)
                candidates.append(prims_i)

                # Record the hit point
                hit_p = ray.o + si.t*ray.d
                hit_points.append(hit_p)

                # Prepare the next interaction, assuming purely specular
                # reflection
                ray = si.spawn_ray(si.to_world(mi.reflect(si.wi)))

        # For diffraction, we need only primitives in LoS
        # [num_los_primitives]
        if len(candidates) > 0:
            # max_depth > 0 or empty scene
            los_primitives = tf.reshape(tf.cast(candidates[0], tf.int32), [-1])
            los_primitives,_ = tf.unique(los_primitives)
            los_primitives = tf.gather(los_primitives,
                                       tf.where(los_primitives != -1)[:,0])
        else:
            # max_depth == 0
            los_primitives = None

        reflection = reflection and (max_depth > 0) and (len(candidates) > 0)
        scattering = scattering and (max_depth > 0) and (len(candidates) > 0)

        if scattering or reflection:
            # Stack all found interactions along the depth dimension
            # [max_depth, num_samples]
            candidates = tf.stack([mi_to_tf_tensor(r, tf.int32)
                                for r in candidates], axis=0)

        if reflection:
            # [max_depth, num_samples]
            candidates_ref = candidates
            # Compute the actual max_depth
            # [max_depth]
            useless_step = tf.reduce_all(tf.equal(candidates_ref, -1), axis=1)
            # ()
            max_depth_ref = tf.where(tf.reduce_any(useless_step),
                                     tf.argmax(tf.cast(useless_step, tf.int32),
                                        output_type=tf.int32),
                                     max_depth)
            # [max_depth, num_samples]
            candidates_ref = candidates_ref[:max_depth_ref]
        else:
            # No candidates
            candidates_ref = tf.fill([0, 0], -1)
            max_depth_ref = 0

        if scattering:
            # [max_depth, num_samples, 3]
            hit_points = tf.stack([mi_to_tf_tensor(r, self._rdtype)
                                for r in hit_points])
            
            ### make scattering sparse
            # [max_depth, num_samples, 3]
            hit_points = hit_points[:, ::scat_sparsity_coef, :]
            candidates_scat = candidates[:, ::scat_sparsity_coef]
            samples_per_source_sparse = samples_per_source // scat_sparsity_coef

            # hit_points = tf.reshape(hit_points,
            #             [max_depth, num_sources, samples_per_source, 3])
            # [max_depth, num_sources, samples_per_source]
            # candidates_scat = tf.reshape(candidates,
            #                     [max_depth, num_sources, samples_per_source])
            
            # [max_depth, num_sources, samples_per_source, 3]
            hit_points = tf.reshape(hit_points, [max_depth, num_sources, samples_per_source_sparse, 3])
            # [max_depth, num_sources, samples_per_source]
            candidates_scat = tf.reshape(candidates_scat, [max_depth, num_sources, samples_per_source_sparse])
            
            # Flag indicating no hits
            # [max_depth, num_sources, samples_per_source]
            no_hit = tf.equal(candidates_scat, -1)
            # Compute the actual max_depth
            # [max_depth]
            useless_step = tf.reduce_all(no_hit, axis=(1,2))
            # ()
            max_depth_scat = tf.where(tf.reduce_any(useless_step),
                                      tf.argmax(tf.cast(useless_step, tf.int32),
                                        output_type=tf.int32),
                                    max_depth)
            # [max_depth, num_sources, samples_per_source, 3]
            hit_points = hit_points[:max_depth_scat]
            # [max_depth, num_sources, samples_per_source]
            candidates_scat = candidates_scat[:max_depth_scat]
            # [max_depth, num_sources, samples_per_source]
            no_hit = no_hit[:max_depth_scat]
            # Remove useless paths
            # [samples_per_source]
            useful_samples = tf.logical_not(tf.reduce_all(no_hit, axis=(0,1)))
            useful_samples_index = tf.where(useful_samples)[:,0]
            # [max_depth, num_sources, num_paths_per_source, 3]
            hit_points = tf.gather(hit_points, useful_samples_index, axis=2)
            # [max_depth, num_sources, num_paths_per_source]
            candidates_scat = tf.gather(candidates_scat, useful_samples_index,
                                        axis=2)
            # [max_depth, num_sources, num_paths_per_source]
            no_hit = tf.gather(no_hit, useful_samples_index, axis=2)

            # Zero the hit masked points
            # [max_depth, num_sources, num_paths, 3]
            hit_points = tf.where(tf.expand_dims(no_hit, axis=-1),
                                tf.zeros_like(hit_points),
                                hit_points)
        else:
            # No hit points
            hit_points = tf.fill([0, num_sources, 1, 3],
                                 tf.cast(0., self._rdtype))
            candidates_scat = tf.fill([0, num_sources, 1], False)
            max_depth_scat = 0

        if ((not reflection) and (not scattering)):
            max_depth = 0

        # Remove duplicates
        if max_depth_ref > 0:
            candidates_ref, _ = tf.raw_ops.UniqueV2(
                x=candidates_ref,
                axis=[1]
            )

        # Add line-of-sight to list of candidates for reflection if
        # required
        if los:
            candidates_ref = tf.concat([tf.fill([max_depth_ref, 1], -1),
                                        candidates_ref],
                                       axis=1)
        else:
            # Ensure there is no LoS by removing all paths corresponding
            # to no hits
            # [num_samples]
            is_nlos = tf.logical_not(tf.reduce_all(candidates_ref == -1,
                                                   axis=0))
            is_nlos_ind = tf.where(is_nlos)[:,0]
            candidates_ref = tf.gather(candidates_ref, is_nlos_ind, axis=1)

        # The previous shoot and bounce process does not do next-event
        # estimation, and continues to trace until max_depth reflections occurs
        # or the ray does not intersect any primitive.
        # Therefore, we extend the set of rays with the prefixes of all
        # rays in `results_tf` to ensure we don't miss shorter paths than the
        # ones found.
        candidates_ref_ = [candidates_ref]
        for depth in range(1, max_depth_ref):
            # Extract prefix of length depth
            # [depth, num_samples]
            prefix = candidates_ref[:depth]
            # Pad with -1, i.e., not intersection
            # [max_depth, num_samples]
            prefix = tf.pad(prefix, [[0, max_depth_ref-depth], [0,0]],
                            constant_values=-1)
            # Add to the list of rays
            candidates_ref_.insert(0, prefix)
        # [max_depth, num_samples]
        candidates_ref = tf.concat(candidates_ref_, axis=1)

        # Extending the rays with prefixes might have created duplicates.
        # Remove duplicates
        if candidates_ref.shape[0] > 0:
            candidates_ref, _ = tf.raw_ops.UniqueV2(
                x=candidates_ref,
                axis=[1]
            )

        return candidates_ref, los_primitives, candidates_scat, hit_points


    ##################################################################
    # Methods used for computing the diffracted paths
    ##################################################################


    def _compute_diffraction_transition_matrices(self,
                                                 relative_permittivity,
                                                 scattering_coefficient,
                                                 paths,
                                                 paths_tmp):
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

        mask = paths.mask
        targets = paths.targets
        sources = paths.sources
        theta_t = paths.theta_t
        phi_t = paths.phi_t
        theta_r = paths.theta_r
        phi_r = paths.phi_r

        #normals = paths_tmp.normals

        # wavelength = self._scene.wavelength
        # k = 2.*PI/wavelength

        # [num_targets, num_sources, max_num_paths, 3]
        diff_points = paths.vertices[0]
        # [num_targets, num_sources, max_num_paths]
        wedges_indices = paths.objects[0]

        # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
        # This makes no difference on the resulting paths as such paths
        # are not flagged as active.
        # [num_targets, num_sources, max_num_paths]
        valid_wedges_idx = tf.where(wedges_indices == -1, 0, wedges_indices)

        # Normals
        # [num_targets, num_sources, max_num_paths, 2, 3]
        normals = tf.gather(self._wedges_normals, valid_wedges_idx, axis=0)

        # [num_targets, num_sources, max_num_paths, 3]
        e_hat = tf.gather(self._wedges_e_hat, valid_wedges_idx)

        # Compute the wedges angle
        # [num_targets, num_sources, max_num_paths]
        cos_wedges_angle = dot(normals[...,0,:], normals[...,1,:], clip=True)    
        wedges_angle = PI - acos_diff(cos_wedges_angle)
        n = (2.*PI-wedges_angle)/PI

        #sin_wedges_angle = dot(cross(normals[...,0,:], normals[...,1,:]), e_hat, clip=True)
        is_concave_wedge = tf.gather(self._is_concave_wedge, valid_wedges_idx)
        n = tf.where(is_concave_wedge, wedges_angle / PI, n)

        # Reshape sources and targets
        # [1, num_sources, 1, 3]
        sources = tf.reshape(sources, [1, -1, 1, 3])
        # [num_targets, 1, 1, 3]
        targets = tf.reshape(targets, [-1, 1, 1, 3])

        # Extract surface normals
        # [num_targets, num_sources, max_num_paths, 3]
        n_0_hat = normals[...,0,:]
        # [num_targets, num_sources, max_num_paths, 3]
        n_n_hat = normals[...,1,:]

        # Relative permitivities and scattering coefficients
        # If a callable is defined to compute the radio material properties,
        # it is invoked. Otherwise, the radio materials of objects are used.
        rm_callable = self._scene.radio_material_callable
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
            diff_points_ = tf.tile(tf.expand_dims(diff_points, axis=-2),
                                   [1, 1, 1, 2, 1])
            # scattering_coefficient, etas : [num_targets, num_sources,
            #   max_num_paths, 2]
            etas, scattering_coefficient, _   = rm_callable(objects_indices,
                                                            diff_points_)
        # # [num_targets, num_sources, max_num_paths]
        # eta_0 = etas[...,0]
        # eta_n = etas[...,1]
        # # [num_targets, num_sources, max_num_paths]
        # scattering_coefficient_0 = scattering_coefficient[...,0]
        # scattering_coefficient_n = scattering_coefficient[...,1]

        # Compute s_prime_hat, s_hat, s_prime, s
        # s_prime_hat : [num_targets, num_sources, max_num_paths, 3]
        # s_prime : [num_targets, num_sources, max_num_paths]
        s_prime_hat, s_prime = normalize(diff_points-sources)
        # s_hat : [num_targets, num_sources, max_num_paths, 3]
        # s : [num_targets, num_sources, max_num_paths]
        s_hat, s = normalize(targets-diff_points)

        mat_t = self._diffraction_compute_fields(s_prime_hat, s_hat, n_0_hat, n_n_hat, e_hat, etas,
                                scattering_coefficient, s_prime, s, n, mask, theta_t, phi_t, theta_r, phi_r, valid_wedges_idx)
        
        mat_t = mat_t / tf.complex(s_prime[..., None, None], tf.zeros_like(s_prime[..., None, None]))

        return mat_t

    def _diffraction_compute_fields(self, s_prime_hat, s_hat, n_0_hat, n_n_hat, e_hat, etas,
                                    scattering_coefficient, s_prime, s, n, mask, 
                                    theta_t, phi_t, theta_r, phi_r, valid_wedges_idx):  
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
            SolverBase.EPSILON
            )
        # [num_targets, num_sources, max_num_paths, 3]
        # pylint: disable=unbalanced-tuple-unpacking
        e_i_s_n, e_i_p_n, e_r_s_n, e_r_p_n = compute_field_unit_vectors(
            s_prime_hat,
            s_hat,
            n_n_hat,#*sign(-dot(s_t_prime_hat, n_n_hat, keepdim=True)),
            SolverBase.EPSILON
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

        d_mul = - tf.cast(tf.exp(-1j*PI/4.), self._dtype) / tf.cast((2*n)*tf.sqrt(2*PI*k)*tf.math.sin(beta_prime), self._dtype)

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

        # [num_targets, num_sources, max_num_paths, 1, 1]
        d_1 = tf.reshape(d_1, tf.concat([tf.shape(d_1), [1,1]], axis=0))
        d_2 = tf.reshape(d_2, tf.concat([tf.shape(d_2), [1,1]], axis=0))
        d_3 = tf.reshape(d_3, tf.concat([tf.shape(d_3), [1,1]], axis=0))
        d_4 = tf.reshape(d_4, tf.concat([tf.shape(d_4), [1,1]], axis=0))

        # [num_targets, num_sources, max_num_paths]
        spreading_factor = tf.sqrt(s_prime / (s*(s_prime + s)))
        spreading_factor = tf.complex(spreading_factor,
                                      tf.zeros_like(spreading_factor))
        #_spreading_factor = spreading_factor
        # [num_targets, num_sources, max_num_paths, 1, 1]
        spreading_factor = tf.reshape(spreading_factor, tf.shape(d_1))

        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = (d_1+d_2)*tf.eye(2,2, batch_shape=tf.shape(r_0)[:3],
                                 dtype=self._dtype)
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t += d_3*r_n + d_4*r_0
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t *= -spreading_factor

        ### glass transmission T_slab
        if self.obj_geom is not None and self.obj_geom.is_glass_transmission:
            glass_mat_t_slab = self.obj_geom.diffraction_glass_transmission(
                valid_wedges_idx=valid_wedges_idx,
                n=n,
                mask=mask,
                phi_prime_hat=phi_prime_hat,
                beta_0_prime_hat=beta_0_prime_hat,
                e_i_s_0=e_i_s_0,
                e_i_p_0=e_i_p_0,
                s_prime_hat=s_prime_hat,
                n_0_hat=n_0_hat,
                wavelength=wavelength,
                dtype=self._dtype
            )

            if self.obj_geom.los_transmissions == 'two':
                glass_mat_t_slab = tf.linalg.matmul(glass_mat_t_slab, glass_mat_t_slab)

            mat_t = tf.linalg.matmul(mat_t, glass_mat_t_slab)

        nans_bool = tf.math.is_nan(tf.math.real(mat_t))
        mat_t = tf.where(nans_bool, tf.zeros_like(mat_t, dtype=self._dtype), mat_t)

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

        return mat_t

    def convert_mat_t(self, mat_t, mask, theta_t, phi_t, theta_r, phi_r, phi1_prime_hat, beta1_prime_hat, phi2_hat, beta2_hat):
        mat_from_gcs = component_transform(
                            theta_hat(theta_t, phi_t), phi_hat(phi_t),
                            phi1_prime_hat, beta1_prime_hat)
        mat_from_gcs = tf.complex(mat_from_gcs, tf.zeros_like(mat_from_gcs))


        mat_to_gcs = component_transform(phi2_hat, beta2_hat,
                                      theta_hat(theta_r, phi_r), phi_hat(phi_r))
        mat_to_gcs = tf.complex(mat_to_gcs, tf.zeros_like(mat_to_gcs))

        mat_t = tf.linalg.matmul(mat_t, mat_from_gcs)
        mat_t = tf.linalg.matmul(mat_to_gcs, mat_t)

        nans_bool = tf.math.is_nan(tf.math.real(mat_t))
        mat_t = tf.where(nans_bool, tf.zeros_like(mat_t, dtype=self._dtype), mat_t)

        # Set invalid paths to 0
        # Expand masks to broadcast with the field components
        # [num_targets, num_sources, max_num_paths, 1, 1]
        # mask_ = expand_to_rank(mask, 5, axis=3)
        mask_ = expand_to_rank(mask, tf.rank(mat_t), axis=-1)
        # Zeroing coefficients corresponding to non-valid paths
        # [num_targets, num_sources, max_num_paths, 2]
        mat_t = tf.where(mask_, mat_t, tf.zeros_like(mat_t))

        return mat_t
    

    ##################################################################
    # Methods used for computing the scattered paths
    ##################################################################

    def _scat_test_rx_blockage(self, targets, sources, candidates, hit_points):
        r"""
        Test if the LoS between the hit points and the target is blocked.
        Blocked paths are masked out.

        Input
        -----
        targets : [num_targets, 3], tf.float
            Coordinates of the targets.

        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        candidates : [max_depth, num_sources, num_paths_per_source], int
            Sequence of primitives hit at `hit_points`.

        hit_points : [max_depth, num_sources, num_paths_per_source, 3], tf.float
            Intersection points.

        Output
        -------
        paths : :class:`~sionna.rt.Paths`
            Structure storing the scattered paths.

        paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Addtional quantities required for paths computation
        """

        num_sources = candidates.shape[1]
        num_targets = targets.shape[0]
        max_depth = tf.shape(candidates)[0]

        # Expand for broadcasting with max_depth, num_sources, and num_paths
        # [1, num_targets, 1, 1, 3]
        targets_ = tf.expand_dims(insert_dims(targets, 2, 1), axis=0)

        # Build the rays for shooting
        # Origins
        # [max_depth, num_targets, num_targets, num_paths, 3]
        hit_points = tf.tile(tf.expand_dims(hit_points, axis=1),
                              [1, num_targets, 1, 1, 1])
        # [max_depth * num_targets * num_sources * num_paths, 3]
        ray_origins = tf.reshape(hit_points, [-1, 3])
        # Directions
        # [max_depth, num_targets, num_sources, num_paths, 3]
        ray_directions,rays_lengths = normalize(targets_ - hit_points)
        # [max_depth * num_targets * num_sources * num_paths, 3]
        ray_directions = tf.reshape(ray_directions, [-1, 3])
        # [max_depth * num_targets * num_sources * num_paths]
        rays_lengths = tf.reshape(rays_lengths, [-1])

        # Test for blockage
        # [max_depth * num_targets * num_sources * num_paths]
        blocked = self._test_obstruction(ray_origins, ray_directions,
                                          rays_lengths)
        # [max_depth, num_targets, num_sources, num_paths]
        blocked = tf.reshape(blocked,
                             [max_depth, num_targets, num_sources, -1])

        # Mask blocked paths
        # [max_depth, num_targets, num_sources, num_paths]
        candidates = tf.tile(tf.expand_dims(candidates, axis=1),
                             [1, num_targets, 1, 1])
        # [max_depth, num_targets, num_sources, num_paths]
        prefix_mask = tf.logical_and(~blocked, tf.not_equal(candidates, -1))

        # Optimize tensor size by ensuring that the length of the paths
        # dimension correspond to the maximum number of paths over all links

        # Keep a path if at least one of its prefix is valid
        # [num_targets, num_sources, num_paths]
        prefix_mask_ = tf.reduce_any(prefix_mask, axis=0)
        prefix_mask_int_ = tf.cast(prefix_mask_, tf.int32)

        # Maximum number of valid paths over all links
        # [num_targets, num_sources]
        num_paths = tf.reduce_sum(prefix_mask_int_, axis=-1)
        # Maximum number of paths
        # ()
        max_num_paths = tf.reduce_max(num_paths)

        # [num_valid_paths, 3]
        gather_indices = tf.where(prefix_mask_)
        # To build the indices of the paths in the tensor with optimized size,
        # the path dimension is indexed by counting the valid path in the order
        # in which they appear
        # [num_targets, num_sources, num_paths]
        path_indices = tf.cumsum(prefix_mask_int_, axis=-1)
        # [num_valid_paths, 3]
        path_indices = tf.gather_nd(path_indices, gather_indices) - 1
        # The indices used to scatter the valid paths in the tensors with
        # optimized size are built by replacing the index of the paths by
        # the previous ones, which leads to skipping the invalid paths
         # [3, num_valid_paths]
        scatter_indices = tf.transpose(gather_indices, [1,0])
        if not tf.size(scatter_indices) == 0:
            scatter_indices = tf.tensor_scatter_nd_update(scatter_indices,
                                [[2]], [path_indices])
        # [num_valid_paths, 3]
        scatter_indices = tf.transpose(scatter_indices, [1,0])

        # Mask of valid paths
        # [num_targets, num_sources, max_num_paths]
        opt_prefix_mask = tf.fill([max_depth, num_targets, num_sources,
                                   max_num_paths], False)
        # Locations of the interactions
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        opt_hit_points = tf.zeros([max_depth, num_targets, num_sources,
                                   max_num_paths, 3], dtype=self._rdtype)
        # [max_depth, num_targets, num_sources, max_num_paths]
        opt_candidates = tf.fill([max_depth, num_targets, num_sources,
                                  max_num_paths], -1)

        if max_depth > 0:

            for depth in tf.range(max_depth, dtype=tf.int64):

                # Indices for storing the valid items for this depth
                scatter_indices_ = tf.pad(scatter_indices, [[0,0], [1,0]],
                                mode='CONSTANT', constant_values=depth)

                # Prefix mask
                # [num_targets, num_sources, num_samples]
                prefix_mask_ = tf.gather(prefix_mask, depth, axis=0)
                # [num_valid_paths, 3]
                prefix_mask_ = tf.gather_nd(prefix_mask_, gather_indices)
                # Store the valid intersection points
                # [max_depth, num_targets, num_sources, max_num_paths]
                opt_prefix_mask = tf.tensor_scatter_nd_update(opt_prefix_mask,
                                                scatter_indices_, prefix_mask_)

                # Location of the interactions
                # [num_targets, num_sources, num_samples, 3]
                hit_points_ = tf.gather(hit_points, depth, axis=0)
                # [num_valid_paths, 3]
                hit_points_ = tf.gather_nd(hit_points_, gather_indices)
                # Store the valid intersection points
                # [max_depth, num_targets, num_sources, max_num_paths, 3]
                opt_hit_points = tf.tensor_scatter_nd_update(opt_hit_points,
                                                scatter_indices_, hit_points_)

                # Intersected primitives
                # [num_targets, num_sources, num_samples]
                candidates_ = tf.gather(candidates, depth, axis=0)
                # [num_valid_paths, 3]
                candidates_ = tf.gather_nd(candidates_, gather_indices)
                # Store the valid intersection points
                # [max_depth, num_targets, num_sources, max_num_paths]
                opt_candidates = tf.tensor_scatter_nd_update(opt_candidates,
                                                scatter_indices_, candidates_)

        # Gather normals to the intersected primitives
        # Note: They are not oriented in the direction of the incoming wave.
        # This is done later.
        # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
        # This makes no difference on the resulting paths as such paths
        # are not flagged as active.
        # [max_depth, num_targets, num_sources, max_num_paths]
        opt_candidates_ = tf.where(opt_candidates == -1, 0, opt_candidates)
        # [max_depth, num_targets, num_sources, num_paths, 3]
        normals = tf.gather(self._normals, opt_candidates_)

        # Map primitives to the corresponding objects
        # Add a dummy entry to primitives_2_objects with value -1.
        # [num_samples + 1]
        primitives_2_objects = tf.pad(self._primitives_2_objects, [[0,1]],
                                        constant_values=-1)
        # Replace all -1 by num_samples
        num_primitives = tf.shape(self._primitives_2_objects)[0]
        # [max_depth, num_targets, num_sources, max_num_paths]
        opt_candidates_ = tf.where(opt_candidates == -1, num_primitives,
                                   opt_candidates)
        # [max_depth, num_targets, num_sources, max_num_paths]
        objects = tf.gather(primitives_2_objects, opt_candidates_)

        # Create and return the the objects storing the scattered paths
        paths = Paths(sources=sources,
                      targets=targets,
                      scene=self._scene,
                      types=Paths.SCATTERED)
        paths.vertices = opt_hit_points
        paths.objects = objects

        paths_tmp = PathsTmpData(sources, targets, self._dtype)
        paths_tmp.normals = normals
        paths_tmp.scat_prefix_mask = opt_prefix_mask
        paths_tmp.scat_prefix_k_s,_ = normalize(targets_ - opt_hit_points)

        return paths, paths_tmp

    def _scat_discard_crossing_paths(self, paths, paths_tmp, scat_keep_prob):
        r"""
        Discards paths:

        - for which the scattered ray is crossing the intersected
        primitive, and

        - randomly with probability `` 1 - scat_keep_prob``.

        Input
        ------
        paths : :class:`~sionna.rt.Paths`
            Structure storing the scattered paths.

        paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Addtional quantities required for paths computation

        scat_keep_prob : tf.float
            Probablity of keeping a valid scattered paths.
            Must be in )0,1).

        Output
        -------
        paths : :class:`~sionna.rt.Paths`
            Updates paths.

        paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Updated addtional quantities required for paths computation
        """

        theta_t = paths.theta_t
        phi_t = paths.phi_t
        objects = paths.objects
        vertices = paths.vertices

        normals = paths_tmp.normals
        mask = paths_tmp.scat_prefix_mask
        k_i = paths_tmp.k_i
        k_r = paths_tmp.k_r
        k_s = paths_tmp.scat_prefix_k_s
        total_distance = paths_tmp.total_distance

        max_depth = tf.shape(vertices)[0]

        # Ensure the normals point in the same direction as -k_i
        # [max_depth, num_targets, num_sources, max_num_paths, 1]
        s = -tf.math.sign(dot(k_i[:max_depth], normals, keepdim=True))
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        normals = normals * s

        # Mask paths for which k_s does not point in the same direction as the
        # normal
        # [max_depth, num_targets, num_sources, max_num_paths]
        same_side = dot(normals, k_s) > SolverBase.EPSILON
        # [max_depth, num_targets, num_sources, max_num_paths]
        mask = tf.logical_and(mask, same_side)

        # Keep valid path with probability `scat_keep_prob`
        # [max_depth, num_targets, num_sources, max_num_paths]
        random_mask = tf.random.uniform(tf.shape(mask), 0., 1., self._rdtype)
        # [max_depth, num_targets, num_sources, max_num_paths]
        random_mask = tf.less(random_mask, scat_keep_prob)
        # [max_depth, num_targets, num_sources, max_num_paths]
        mask = tf.logical_and(mask, random_mask)

        # Discard paths invalid for all links
        valid_indices = tf.where(tf.reduce_any(mask, axis=(0,1,2)))[:,0]
        # [num_targets, num_sources, max_num_paths]]
        theta_t = tf.gather(theta_t, valid_indices, axis=2)
        phi_t = tf.gather(phi_t, valid_indices, axis=2)
        # [max_depth, num_targets, num_sources, max_num_paths]
        objects = tf.gather(objects, valid_indices, axis=3)
        mask = tf.gather(mask, valid_indices, axis=3)
        total_distance = tf.gather(total_distance, valid_indices, axis=3)
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        normals = tf.gather(normals, valid_indices, axis=3)
        k_r = tf.gather(k_r, valid_indices, axis=3)
        k_i = tf.gather(k_i, valid_indices, axis=3)
        vertices = tf.gather(vertices, valid_indices, axis=3)

        paths.theta_t = theta_t
        paths.phi_t = phi_t
        paths.vertices = vertices
        paths.objects = objects

        paths_tmp.scat_prefix_mask = mask
        paths_tmp.k_i = k_i
        paths_tmp.k_r = k_r
        paths_tmp.k_tx = k_i[0]
        paths_tmp.total_distance = total_distance
        paths_tmp.normals = normals

        return paths, paths_tmp

    def _scat_prefixes_2_paths(self, paths, paths_tmp):
        """
        Extracts valid prefixes as invidual paths.

        Input
        ------
        paths : :class:`~sionna.rt.Paths`
            Structure storing the scattered paths.

        paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Addtional quantities required for paths computation

        Output
        -------
        paths : :class:`~sionna.rt.Paths`
            Updates paths.

        paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Updated addtional quantities required for paths computation
        """

        # [max_depth, num_targets, num_sources, max_num_paths]
        prefix_mask = paths_tmp.scat_prefix_mask
        prefix_mask_int = tf.cast(prefix_mask, tf.int32)
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        prefix_vertices = paths.vertices
        # [max_depth, num_targets, num_sources, max_num_paths]
        prefix_objects = paths.objects
        # [num_targets, num_sources, max_num_paths]
        prefix_theta_t = paths.theta_t
        # [num_targets, num_sources, max_num_paths]
        prefix_phi_t = paths.phi_t
        # [max_depth, num_targets, num_sources, num_paths, 3]
        prefix_normals = paths_tmp.normals
        # [max_depth + 1, num_targets, num_sources, num_paths, 3]
        prefix_k_i = paths_tmp.k_i
        # [max_depth, num_targets, num_sources, num_paths, 3]
        prefix_k_r = paths_tmp.k_r
        # [max_depth, num_targets, num_sources, num_paths]
        prefix_distances = paths_tmp.total_distance
        # [num_targets, num_sources, max_num_paths, 3]
        prefix_ktx = paths_tmp.k_tx

        max_depth = tf.shape(prefix_mask)[0]
        max_depth64 = tf.cast(max_depth, tf.int64)
        num_targets = tf.shape(prefix_mask)[1]
        num_sources = tf.shape(prefix_mask)[2]

        # Number of paths for each link and depth
        # [max_depth, num_targets, num_sources]
        paths_count = tf.reduce_sum(prefix_mask_int, axis=3)
        # Maximum number of paths for each depth over all the links
        # [max_depth]
        path_count_depth = tf.reduce_max(paths_count, axis=(1,2))
        # Upper bound on the total number of paths
        # ()
        max_num_paths = tf.reduce_sum(path_count_depth)

        # [num_valid_paths, 4]
        gather_indices = tf.where(prefix_mask)
        # To build the indices of the paths in the tensor with optimized size,
        # the path dimension is indexed by counting the valid path in the order
        # in which they appear
        # [max_depth, num_targets, num_sources, num_paths]
        path_indices = tf.cumsum(prefix_mask_int, axis=-1)
        # [num_valid_paths, 4]
        path_indices = tf.gather_nd(path_indices, gather_indices) - 1
        scatter_indices = tf.transpose(gather_indices, [1,0])
        if not tf.size(scatter_indices) == 0:
            scatter_indices = tf.tensor_scatter_nd_update(scatter_indices,
                                [[3]], [path_indices])
        # [num_valid_paths, 3]
        scatter_indices = tf.transpose(scatter_indices, [1,0])

        # Create the final tensors to update
        # [num_targets, num_sources, max_num_paths]
        mask = tf.fill([num_targets, num_sources, max_num_paths], False)
        # This tensor is created transposed as paths
        # are added with all the objects hit along the paths.
        # [num_targets, num_sources, max_num_paths, max_depth, 3]
        vertices = tf.zeros([num_targets, num_sources, max_num_paths, max_depth,
                             3], self._rdtype)
        # Last vertices that were hit
        # [num_targets, num_sources, max_num_paths, 3]
        last_vertices = tf.zeros([num_targets, num_sources, max_num_paths, 3],
                                 self._rdtype)
        # Objects that were hit. This tensor is created transposed as paths
        # are added with all the objects hit along the paths.
        # [num_targets, num_sources, max_num_paths, max_depth]
        objects = tf.fill([num_targets, num_sources, max_num_paths, max_depth],
                          -1)
        # Last objects that were hit
        # [num_targets, num_sources, max_num_paths]
        last_objects = tf.fill([num_targets, num_sources, max_num_paths], -1)
        # Angles of departure
        # [num_targets, num_sources, max_num_paths]
        theta_t = tf.zeros([num_targets, num_sources, max_num_paths],
                           self._rdtype)
        # [num_targets, num_sources, max_num_paths]
        phi_t = tf.zeros([num_targets, num_sources, max_num_paths],
                         self._rdtype)
        # Normal to the last intersected objects
        # [num_targets, num_sources, max_num_paths, 3]
        last_normals = tf.zeros([num_targets, num_sources, max_num_paths, 3],
                                self._rdtype)
        # Direction of incidence at the last interaction point
        # [num_targets, num_sources, max_num_paths, 3]
        last_k_i = tf.zeros([num_targets, num_sources, max_num_paths, 3],
                                self._rdtype)
        # Distance from the sources to the last interaction point
        # [num_targets, num_sources, max_num_paths]
        last_distance = tf.zeros([num_targets, num_sources, max_num_paths],
                                 self._rdtype)
        # [num_targets, num_sources, max_num_paths, 3]
        k_tx = tf.zeros([num_targets, num_sources, max_num_paths, 3],
                        self._rdtype)
        # Normals at the intersection points
        # [num_targets, num_sources, max_num_paths, max_depth, 3]
        normals = tf.zeros([num_targets, num_sources, max_num_paths,
                            max_depth, 3], self._rdtype)
        # Direction of reflection at intersection points
        # [num_targets, num_sources, max_num_paths, max_depth, 3]
        k_r = tf.zeros([num_targets, num_sources, max_num_paths, max_depth, 3],
                       self._rdtype)
        # Direction of incidence at intersection points
        # [num_targets, num_sources, max_num_paths, max_depth+1, 3]
        k_i = tf.zeros([num_targets, num_sources, max_num_paths, max_depth+1,3],
                       self._rdtype)

        # Need to transpose these tensors in order to gather paths from them
        # with all the interactions, i.e., extract the entire "max_depth"
        # dimension.
        # [max_depth, num_targets, num_sources, max_num_paths]
        prefix_objects_tp = tf.transpose(prefix_objects, [1,2,3,0])
        # [num_targets, num_sources, max_num_paths, max_depth, 3]
        prefix_vertices_tp = tf.transpose(prefix_vertices, [1,2,3,0,4])
        # [num_targets, num_sources, max_num_paths, max_depth, 3]
        normals_tp = tf.transpose(prefix_normals, [1,2,3,0,4])
        # [num_targets, num_sources, max_num_paths, max_depth, 3]
        k_i_tp = tf.transpose(prefix_k_i, [1,2,3,0,4])
        # [num_targets, num_sources, max_num_paths, max_depth, 3]
        k_r_tp = tf.transpose(prefix_k_r, [1,2,3,0,4])

        # We sequentially add the prefixes for each depth value.
        # To avoid overwriting the paths scattered at the previous
        # iterations, we incremdent the path index by the maximum number
        # of paths over all links, cumulated over the iterations.
        path_ind_incr = 0
        for depth in tf.range(max_depth, dtype=tf.int64):
            # Indices of valid paths with depth d
            # [num_valid_paths with depth=depth, 4]
            gather_indices_ = tf.gather(gather_indices,
                                tf.where(gather_indices[:,0] == depth)[:,0],
                                axis=0)
            # Depth is not needed for some tensors
            # [num_valid_paths with depth=depth, 3]
            gather_indices_nd_ = gather_indices_[:,1:]

            # Indices for scattering the results in the target tensor
            # [num_valid_paths with depth=depth, 4]
            scatter_indices_ = tf.gather(scatter_indices,
                                tf.where(scatter_indices[:,0] == depth)[:,0],
                                axis=0)


            # [1, 4]
            path_ind_incr_ = tf.cast([0, 0, 0, path_ind_incr], tf.int64)
            # [num_valid_paths with depth=depth, 4]
            scatter_indices_ = scatter_indices_ + path_ind_incr_
            # Depth is not needed for some tensors
            # [num_valid_paths with depth=depth, 3]
            scatter_indices_nd_ = scatter_indices_[:,1:]
            # Prepare for next iteration
            path_ind_incr = path_ind_incr + path_count_depth[depth]

            # Update the tensors

            # Mask
            # [num_valid_paths with depth=depth]
            prefix_mask_ = tf.fill([tf.shape(scatter_indices_nd_)[0]], True)
            # [num_targets, num_sources, max_num_paths]
            mask = tf.tensor_scatter_nd_update(mask, scatter_indices_nd_,
                                               prefix_mask_)

            # Vertices
            prefix_vertices_ = tf.gather_nd(prefix_vertices_tp,
                                            gather_indices_nd_)
            # [num_targets, num_sources, max_num_paths, max_depth, 3]
            vertices = tf.tensor_scatter_nd_update(vertices,
                                                   scatter_indices_nd_,
                                                   prefix_vertices_)

            # Last vertex
            prefix_vertex = tf.gather_nd(prefix_vertices, gather_indices_)
            # [num_targets, num_sources, max_num_paths, 3]
            last_vertices = tf.tensor_scatter_nd_update(last_vertices,
                                                        scatter_indices_nd_,
                                                        prefix_vertex)

            # Objects
            # [num_paths, max_depth]
            objects_ = tf.gather_nd(prefix_objects_tp, gather_indices_nd_)
            # Only keep the prefix of length depth
            # [num_paths, depth]
            objects_ = objects_[:,:depth+1]
            # [num_paths, max_depth]
            objects_ = tf.pad(objects_, [[0,0],[0,max_depth64-depth-1]],
                              constant_values=-1)
            # [num_targets, num_sources, max_num_paths, max_depth]
            objects = tf.tensor_scatter_nd_update(objects, scatter_indices_nd_,
                                                  objects_)

            # Normals at intersection points
            normals_ = tf.gather_nd(normals_tp, gather_indices_nd_)
            # [num_targets, num_sources, max_num_paths, max_depth, 3]
            normals = tf.tensor_scatter_nd_update(normals, scatter_indices_nd_,
                                                  normals_)

            # Direction of incidence at intersection points
            k_i_ = tf.gather_nd(k_i_tp, gather_indices_nd_)
            # [num_targets, num_sources, max_num_paths, max_depth+1, 3]
            k_i = tf.tensor_scatter_nd_update(k_i, scatter_indices_nd_, k_i_)

            # Direction of reflection at intersection points
            k_r_ = tf.gather_nd(k_r_tp, gather_indices_nd_)
            # [num_targets, num_sources, max_num_paths, max_depth, 3]
            k_r = tf.tensor_scatter_nd_update(k_r, scatter_indices_nd_, k_r_)

            # Last hit objects
            objects_ = tf.gather_nd(prefix_objects, gather_indices_)
            # [num_targets, num_sources, max_num_paths]
            last_objects = tf.tensor_scatter_nd_update(last_objects,
                                                       scatter_indices_nd_,
                                                       objects_)

            # Azimuth of departure
            phi_t_ = tf.gather_nd(prefix_phi_t, gather_indices_nd_)
            # [num_targets, num_sources, max_num_paths]
            phi_t = tf.tensor_scatter_nd_update(phi_t, scatter_indices_nd_,
                                                phi_t_)

            # Elevation of departure
            theta_t_ = tf.gather_nd(prefix_theta_t, gather_indices_nd_)
            # [num_targets, num_sources, max_num_paths]
            theta_t = tf.tensor_scatter_nd_update(theta_t, scatter_indices_nd_,
                                                  theta_t_)

            # Normals at the last intersected object
            normals_ = tf.gather_nd(prefix_normals, gather_indices_)
            # [num_targets, num_sources, max_num_paths, 3]
            last_normals = tf.tensor_scatter_nd_update(last_normals,
                                                scatter_indices_nd_, normals_)

            # Direction of incidence at the last interaction point
            k_i_ = tf.gather_nd(prefix_k_i, gather_indices_)
            # [num_targets, num_sources, max_num_paths, 3]
            last_k_i = tf.tensor_scatter_nd_update(last_k_i,
                                                   scatter_indices_nd_,
                                                   k_i_)

            # Distance from the sources to the last interaction point
            last_dist_ = tf.gather_nd(prefix_distances, gather_indices_)
            # [num_targets, num_sources, max_num_paths]
            last_distance = tf.tensor_scatter_nd_update(last_distance,
                                                        scatter_indices_nd_,
                                                        last_dist_)

            # Direction of tx
            k_tx_ = tf.gather_nd(prefix_ktx, gather_indices_nd_)
            # [num_targets, num_sources, max_num_paths, 3]
            k_tx = tf.tensor_scatter_nd_update(k_tx, scatter_indices_nd_, k_tx_)

        # Computes the angles of arrivals, direction of the scattered field,
        # and distance from the scattering point to the targets
        # [num_targets, 3]
        targets = paths.targets
        # [num_targets, 1, 1, 3]
        targets = insert_dims(targets, 2, 1)
        # k_s : [num_targets, num_sources, max_num_paths, 3]
        # scat_2_target_dist : [num_targets, num_sources, max_num_paths]
        k_s,scat_2_target_dist = normalize(targets - last_vertices)
        # Angles of arrivales
        # theta_r, phi_r : [num_targets, num_sources, max_num_paths]
        theta_r, phi_r = theta_phi_from_unit_vec(-k_s)
        # Compute the delays
        # [num_targets, num_sources, max_num_paths]
        tau = (last_distance + scat_2_target_dist)/SPEED_OF_LIGHT
        # [num_targets, num_sources, max_num_paths]
        tau = tf.where(mask, tau, -tf.ones_like(tau))
        # [max_depth, num_targets, num_sources, max_num_paths]
        objects = tf.transpose(objects, [3, 0, 1, 2])
        vertices = tf.transpose(vertices, [3, 0, 1, 2, 4])
        normals = tf.transpose(normals, [3, 0, 1, 2, 4])
        k_i = tf.transpose(k_i, [3, 0, 1, 2, 4])
        k_r = tf.transpose(k_r, [3, 0, 1, 2, 4])

        paths.mask = mask
        paths.vertices = vertices
        paths.objects = objects
        paths.tau = tau
        paths.phi_t = phi_t
        paths.theta_t = theta_t
        paths.phi_r = phi_r
        paths.theta_r = theta_r

        paths_tmp.scat_last_objects = last_objects
        paths_tmp.scat_last_normals = last_normals
        paths_tmp.scat_last_k_i = last_k_i
        paths_tmp.scat_last_vertices = last_vertices
        paths_tmp.scat_src_2_last_int_dist = last_distance
        paths_tmp.scat_k_s = k_s
        paths_tmp.scat_2_target_dist = scat_2_target_dist
        paths_tmp.k_tx = k_tx
        paths_tmp.k_rx = -k_s
        paths_tmp.normals = normals
        paths_tmp.k_i = k_i
        paths_tmp.k_r = k_r
        paths_tmp.total_distance = scat_2_target_dist + last_distance

        return paths, paths_tmp

    ##################################################################
    # Utilities
    ##################################################################

    def _compute_directions_distances_delays_angles(self, paths, paths_tmp,
                                                    scattering, skip_duplicates=False):
        # pylint: disable=line-too-long
        r"""
        Computes:
        - The direction of incidence and departure at every interaction points
        ``k_i`` and ``k_r``
        - The length of each path segment ``distances``
        - The delays of each path
        - The angles of departure (``theta_t``, ``phi_t``) and arrival
        (``theta_r``, ``phi_r``)

        Input
        ------
        paths : :class:`~sionna.rt.Paths`
            Paths to update

        paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Addtional quantities required for paths computation

        scattering : bool
            Set to `True` computing the scattered paths.

        Output
        -------
        paths : :class:`~sionna.rt.Paths`
            Updated paths

        paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Updated addtional quantities required for paths computation
        """

        objects = paths.objects
        vertices = paths.vertices
        sources = paths.sources
        targets = paths.targets
        if scattering:
            mask = paths_tmp.scat_prefix_mask
        else:
            mask = paths.mask

        # Maximum depth
        max_depth = tf.shape(vertices)[0]

        # Flag that indicates if a ray is valid
        # [max_depth, num_targets, num_sources, max_num_paths]
        valid_ray = tf.not_equal(objects, -1)

        # Vertices updated with the sources and targets
        # [1, num_sources, 1, 3]
        sources = tf.expand_dims(tf.expand_dims(sources, axis=0), axis=2)
        # [num_targets, num_sources, max_num_paths, 3]
        sources = tf.broadcast_to(sources, tf.shape(vertices)[1:])
        # [1, num_targets, num_sources, max_num_paths, 3]
        sources = tf.expand_dims(sources, axis=0)
        # [1 + max_depth, num_targets, num_sources, max_num_paths, 3]
        vertices = tf.concat([sources, vertices], axis=0)
        # For the targets, we need to account for the paths having different
        # depths.
        # Pad vertices with dummy values to create the required extra depth
        # [1 + max_depth + 1, num_targets, num_sources, max_num_paths, 3]
        vertices = tf.pad(vertices, [[0,1],[0,0],[0,0],[0,0],[0,0]])
        # [num_targets, 1, 1, 3]
        targets = tf.expand_dims(tf.expand_dims(targets, axis=1), axis=2)
        # [num_targets, num_sources, max_num_paths, 3]
        targets = tf.broadcast_to(targets, tf.shape(vertices)[1:])

        #  [max_depth, num_targets, num_sources, max_num_paths]
        target_indices = tf.cast(valid_ray, tf.int64)
        #  [num_targets, num_sources, max_num_paths]
        target_indices = tf.reduce_sum(target_indices, axis=0) + 1
        # [num_targets*num_sources*max_num_paths]
        target_indices = tf.reshape(target_indices, [-1,1])
        # Indices of all (target, source,paths) entries
        # [num_targets*num_sources*max_num_paths, 3]
        target_indices_ = tf.where(tf.fill(tf.shape(vertices)[1:4], True))
        # Indices of all entries in vertices
        # [num_targets*num_sources*max_num_paths, 4]
        target_indices = tf.concat([target_indices, target_indices_], axis=1)
        # Reshape targets
        # vertices : [max_depth + 1, num_targets, num_sources, max_num_paths, 3]
        targets = tf.reshape(targets, [-1,3])
        vertices = tf.tensor_scatter_nd_update(vertices, target_indices,
                                                targets)
        # Direction of arrivals (k_i)
        # The last item (k_i[max_depth]) correspond to the direction of arrival
        # at the target. Therefore, k_i is a tensor of length `max_depth + 1`,
        # where `max_depth` is the number of maximum interaction (which could be
        # zero if only LoS is requested).
        # k_i : [max_depth + 1, num_targets, num_sources, max_num_paths, 3]
        # ray_lengths : [max_depth + 1, num_targets, num_sources, max_num_paths]
        k_i = tf.roll(vertices, -1, axis=0) - vertices
        k_i,ray_lengths = normalize(k_i)
        k_i = k_i[:max_depth+1]
        ray_lengths = ray_lengths[:max_depth+1]

        # Direction of departures (k_r) at interaction points.
        # We do not need the direction of departure at the source, as it
        # is the same as k_i[0]. Therefore `k_r` only stores the directions of
        # departures at the `max_depth` interaction points.
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        k_r = tf.roll(vertices, -2, axis=0) - tf.roll(vertices, -1, axis=0)
        k_r,_ = normalize(k_r)
        k_r = k_r[:max_depth]

        # Compute the distances
        # [max_depth, num_targets, num_sources, max_num_paths]
        lengths_mask = tf.cast(valid_ray, self._rdtype)
        # First ray is always valid (LoS)
        # [1 + max_depth, num_targets, num_sources, max_num_paths]
        lengths_mask = tf.pad(lengths_mask, [[1,0],[0,0],[0,0],[0,0]],
                                constant_values=tf.ones((),self._rdtype))
        # Compute path distance
        # [1 + max_depth, num_targets, num_sources, max_num_paths]
        distances = lengths_mask*ray_lengths

        # Propagation delay [s]
        # Total length of the paths
        if scattering:
            # Distances of every path prefix, not including the one connecting
            # to the target
            # [max_depth, num_targets, num_sources, max_num_paths]
            total_distance = tf.cumsum(distances[:max_depth], axis=0)
        else:
            # [num_targets, num_sources, max_num_paths]
            total_distance = tf.reduce_sum(distances, axis=0)
            # [num_targets, num_sources, max_num_paths]
            tau = total_distance / SPEED_OF_LIGHT

        # Compute angles of departures and arrival
        # theta_t, phi_t: [num_targets, num_sources, max_num_paths]
        theta_t, phi_t = theta_phi_from_unit_vec(k_i[0])
        # In the case of scattering, the angles of arrival are not computed
        # by this function
        # if not scattering or skip_duplicates:
        if not scattering:
            # Depth of the rays
            # [num_targets, num_sources, max_num_paths]
            ray_depth = tf.reduce_sum(tf.cast(valid_ray, tf.int32), axis=0)
            k_rx = -tf.gather(tf.transpose(k_i, [1,2,3,0,4]), ray_depth,
                                    batch_dims=3, axis=3)
            # theta_r, phi_r: [num_targets, num_sources, max_num_paths]
            theta_r, phi_r = theta_phi_from_unit_vec(k_rx)

            if not skip_duplicates:
                # Remove duplicated paths.
                # Paths intersecting an edge belonging to two different triangles
                # can be considered twice.
                # Note that this is rare, as intersections rarely occur on edges.
                # The similarity measure used to distinguish paths if the distance
                # between the angles of arrivals and departures.
                # [num_targets, num_sources, max_num_paths, 4]
                sim = tf.stack([theta_t, phi_t, theta_r, phi_r], axis=3)
                # [num_targets, num_sources, max_num_paths, max_num_paths, 4]
                sim = tf.expand_dims(sim, axis=2) - tf.expand_dims(sim, axis=3)
                # [num_targets, num_sources, max_num_paths, max_num_paths]
                sim = tf.reduce_sum(tf.square(sim), axis=4)
                sim = tf.equal(sim, tf.zeros_like(sim))
                # Keep only the paths with no duplicates.
                # If many paths are identical, keep the one with the highest index
                # [num_targets, num_sources, max_num_paths, max_num_paths]
                sim = tf.logical_and(tf.linalg.band_part(sim, 0, -1),
                                    ~tf.eye(tf.shape(sim)[-1],
                                            dtype=tf.bool,
                                            batch_shape=tf.shape(sim)[:2]))
                sim = tf.logical_and(sim, tf.expand_dims(mask, axis=-2))
                # [num_targets, num_sources, max_num_paths]
                uniques = tf.reduce_all(~sim, axis=3)
                # Keep only the unique paths
                # [num_targets, num_sources, max_num_paths]
                mask = tf.logical_and(uniques, mask)

                # Setting -1 for delays corresponding to non-valid paths
                # [num_targets, num_sources, max_num_paths]
                tau = tf.where(mask, tau, -tf.ones_like(tau))

        # Updates the object storing the paths
        if not scattering:
            paths.mask = mask
            paths.tau = tau
            # In the case of scattering, the angles of arrival are not computed
            # by this function
            paths.theta_r = theta_r
            paths.phi_r = phi_r
            paths_tmp.k_rx = k_rx
        else:
            paths_tmp.scat_prefix_mask = mask

        paths.theta_t = theta_t
        paths.phi_t = phi_t
        paths_tmp.k_i = k_i
        paths_tmp.k_r = k_r
        paths_tmp.k_tx = k_i[0]
        paths_tmp.total_distance = total_distance

        return paths, paths_tmp

    def _compute_doppler_shifts(self, paths, paths_tmp, velocity):
        # pylint: disable=line-too-long
        """
        Computes the Doppler shift resulting from the movement
        of objects in the scene for every path.

        The Doppler shift resulting from the movement of the
        transmitter and receiver are added later when the function
        :method:`~sionna.rt.Paths.apply_doppler` is called.

        Input
        ------
        paths : :class:`~sionna.rt.Paths`
            Paths to update

        paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Addtional quantities required for paths computation

        velocity : [num_shapes, 3]
            Velocity vectors of all objects in the scene

        Output
        ------
        doppler : [num_targets, num_sources, max_num_paths]
            Doppler shifts for all paths due to the movement of objects
        """
        
        # Compute Doppler shift for every path segment
        # Difference of outgoing and incoming direction vectors for every
        # intersection point
        # k_diff : [max_depth, num_targets, num_sources, max_num_paths, 3]
        k_diff = paths_tmp.k_i[1:]-paths_tmp.k_i[:-1]

        objects_mask = paths.true_objects == -1
        valid_objects = tf.where(objects_mask, 0, paths.true_objects)

        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        velocity = tf.gather(velocity, valid_objects, axis=0)

        ### rotational motion
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        point_on_axis = tf.gather(self.rot_motion_point_on_axis, valid_objects, axis=0)
        axis = tf.gather(self.rot_motion_axis, valid_objects, axis=0)
        interaction_points = paths.vertices
        u = interaction_points - point_on_axis
        # [max_depth, num_targets, num_sources, max_num_paths]
        angular_velocity = tf.gather(self.rot_motion_angular_velocity, valid_objects, axis=0)

        # [max_depth, num_targets, num_sources, max_num_paths]
        D = -dot(point_on_axis, axis)
        #dist = tf.abs(dot(interaction_points, axis) + D)
        dist_along_axis = dot(interaction_points, axis)
        dist = tf.sqrt(tf.linalg.norm(u, axis=-1)**2 - dist_along_axis**2)
        inst_velocity_abs = -2.0 * PI * (angular_velocity / 60) * dist

        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        e3 = axis
        #u = interaction_points - point_on_axis
        u_projected_on_axis = dot(u, e3, keepdim=True) * e3
        e1, _ = normalize(u - u_projected_on_axis)
        e2 = cross(e3, e1)
        inst_velocity = e2 * inst_velocity_abs[..., None]

        # Compute Doppler shift per path
        #[num_targets, num_sources, max_num_paths]
        doppler = tf.reduce_sum((velocity + inst_velocity)*k_diff, axis=-1)
        doppler = tf.where(objects_mask, tf.zeros_like(doppler), doppler)
        doppler = tf.reduce_sum(doppler, axis=0)
        doppler /= self._scene.wavelength
        return doppler

    def _get_tx_rx_rotation_matrices(self):
        r"""
        Computes and returns the rotation matrices for rotating according to
        the orientations of the transmitters and receivers rotation matrices,

        Output
        -------
        rx_rot_mat : [num_rx, 3, 3], tf.float
            Matrices for rotating according to the receivers orientations

        tx_rot_mat : [num_tx, 3, 3], tf.float
            Matrices for rotating according to the receivers orientations
        """

        transmitters = self._scene.transmitters.values()
        receivers = self._scene.receivers.values()

        # Rotation matrices for transmitters
        # [num_tx, 3]
        tx_orientations = [tx.orientation for tx in transmitters]
        tx_orientations = tf.stack(tx_orientations, axis=0)
        # [num_tx, 3, 3]
        tx_rot_mat = rotation_matrix(tx_orientations)

        # Rotation matrices for receivers
        # [num_rx, 3]
        rx_orientations = [rx.orientation for rx in receivers]
        rx_orientations = tf.stack(rx_orientations, axis=0)
        # [num_rx, 3, 3]
        rx_rot_mat = rotation_matrix(rx_orientations)

        return rx_rot_mat, tx_rot_mat

    def _get_antennas_relative_positions(self, rx_rot_mat, tx_rot_mat):
        r"""
        Returns the positions of the antennas of the transmitters and receivers.
        The positions are relative to the center of the radio devices, but
        rotated to the GCS.

        Input
        ------
        rx_rot_mat : [num_rx, 3, 3], tf.float
            Matrices for rotating according to the receivers orientations

        tx_rot_mat : [num_tx, 3, 3], tf.float
            Matrices for rotating according to the receivers orientations

        Output
        -------
        rx_rel_ant_pos: [num_rx, rx_array_size, 3], tf.float
            Relative positions of the receivers antennas

        tx_rel_ant_pos: [num_tx, rx_array_size, 3], tf.float
            Relative positions of the transmitters antennas
        """

        # Rotated position of the TX and RX antenna elements
        # [1, tx_array_size, 3]
        tx_rel_ant_pos = tf.expand_dims(self._scene.tx_array.positions, axis=0)
        # [num_tx, 1, 3, 3]
        tx_rot_mat = tf.expand_dims(tx_rot_mat, axis=1)
        # [num_tx, tx_array_size, 3]
        tx_rel_ant_pos = tf.linalg.matvec(tx_rot_mat, tx_rel_ant_pos)

        # [1, rx_array_size, 3]
        rx_rel_ant_pos = tf.expand_dims(self._scene.rx_array.positions, axis=0)
        # [num_rx, 1, 3, 3]
        rx_rot_mat = tf.expand_dims(rx_rot_mat, axis=1)
        # [num_tx, tx_array_size, 3]
        rx_rel_ant_pos = tf.linalg.matvec(rx_rot_mat, rx_rel_ant_pos)

        return rx_rel_ant_pos, tx_rel_ant_pos

    def _apply_synthetic_array(self, rx_rot_mat, tx_rot_mat, paths, paths_tmp):
        # pylint: disable=line-too-long
        r"""
        Applies the phase shifts to simulate the effect of a synthetic array
        on a planar wave

        Input
        ------
        rx_rot_mat : [num_rx, 3, 3], tf.float
            Matrices for rotating according to the receivers orientations

        tx_rot_mat : [num_tx, 3, 3], tf.float
            Matrices for rotating according to the receivers orientations

        paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Addtional quantities required for paths computation

        Output
        -------
        paths : :class:`~sionna.rt.PathsTmpData`
            Updated paths
        """

        # [num_rx, num_rx_patterns, 1, num_tx, num_tx_patterns, 1,
        #   max_num_paths]
        a = paths.a
        # [num_rx, num_tx, samples_per_tx, 3]
        k_tx = paths_tmp.k_tx
        # [num_tx, num_tx, samples_per_tx, 3]
        k_rx = paths_tmp.k_rx

        two_pi = tf.cast(2.*PI, self._rdtype)

        # Relative positions of the antennas of the transmitters and receivers
        # rx_rel_ant_pos: [num_rx, rx_array_size, 3], tf.float
        #     Relative positions of the receivers antennas
        # tx_rel_ant_pos: [num_tx, rx_array_size, 3], tf.float
        #     Relative positions of the transmitters antennas
        rx_rel_ant_pos, tx_rel_ant_pos =\
            self._get_antennas_relative_positions(rx_rot_mat, tx_rot_mat)

        # Expand dims for broadcasting with antennas
        # The receive vector is flipped as we need vectors that point away
        # from the arrays.
        # [num_rx, 1, 1, num_tx, 1, 1, max_num_paths, 3]
        k_rx = insert_dims(insert_dims(k_rx, 2, 1), 2, 4)
        k_tx = insert_dims(insert_dims(k_tx, 2, 1), 2, 4)
        # Compute the synthetic phase shifts due to the antenna array
        # Transmitter side
        # Expand for broadcasting with receiver, receive antennas,
        # paths
        # [1, 1, 1, num_tx, tx_array_size, 3]
        tx_rel_ant_pos = insert_dims(tx_rel_ant_pos, 3, axis=0)
        # [1, 1, 1, num_tx, 1, tx_array_size, 1, 3]
        tx_rel_ant_pos = tf.expand_dims(tf.expand_dims(tx_rel_ant_pos, axis=4),
                                        axis=6)
        # [num_rx, 1, 1, num_tx, 1, tx_array_size, max_num_paths]
        tx_phase_shifts = dot(tx_rel_ant_pos, k_tx)
        # Receiver side
        # Expand for broadcasting with transmitter, transmit antennas,
        # paths
        # [num_rx, 1, rx_array_size, 1, 1, 1, 1, 3]
        rx_rel_ant_pos = insert_dims(tf.expand_dims(rx_rel_ant_pos, axis=1),
                                     4, axis=3)
        # [num_rx, 1, rx_array_size, num_tx, 1, 1, 1, max_num_paths]
        rx_phase_shifts = dot(rx_rel_ant_pos, k_rx)
        # Total phase shift
        # [num_rx, 1, rx_array_size, num_tx, 1, tx_array_size, max_num_paths]
        phase_shifts = rx_phase_shifts + tx_phase_shifts
        phase_shifts = two_pi*phase_shifts/self._scene.wavelength
        # Apply the phase shifts
        # Broadcast is not supported by TF for such high rank tensors.
        # We therefore do it manually
        # [num_rx, num_rx_patterns, rx_array_size, num_tx, num_tx_patterns,
        #   tx_array_size, max_num_paths]
        a = tf.tile(a, [1, 1, phase_shifts.shape[2], 1, 1,
                        phase_shifts.shape[5], 1])
        # [num_rx, num_rx_patterns, rx_array_size, num_tx, num_tx_patterns,
        #   tx_array_size, max_num_paths]
        a = a*tf.exp(tf.complex(tf.zeros_like(phase_shifts), phase_shifts))
        a = flatten_dims(flatten_dims(a, 2, 1), 2, 3)

        return a

    def _compute_paths_coefficients(self, rx_rot_mat, tx_rot_mat, paths,
                                    paths_tmp, num_samples,
                                    scattering_coefficient, xpd_coefficient,
                                    etas, alpha_r, alpha_i, lambda_,
                                    scat_keep_prob, scat_random_phases):
        # pylint: disable=line-too-long
        r"""
        Computes the paths coefficients.

        Input
        ------
        rx_rot_mat : [num_rx, 3, 3], tf.float
            Matrices for rotating according to the receivers orientations

        tx_rot_mat : [num_tx, 3, 3], tf.float
            Matrices for rotating according to the receivers orientations

        paths : :class:`~sionna.rt.Paths`
            Paths to update

        paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Updated addtional quantities required for paths computation

        num_samples : int
            Number of random rays to trace in order to generate candidates.
            A large sample count may exhaust GPU memory.

        scattering_coefficient : [num_shapes], tf.float
            Scattering coefficient :math:`S\in[0,1]` as defined in
            :eq:`scattering_coefficient`.

        xpd_coefficient: [num_shapes], tf.float
            Cross-polarization discrimination coefficient :math:`K_x\in[0,1]` as
            defined in :eq:`xpd`.

        etas : [num_shapes], tf.complex
            Complex relative permittivity :math:`\eta` :eq:`eta`

        alpha_r : [num_shapes], tf.int32
            Parameter related to the width of the scattering lobe in the
            direction of the specular reflection.

        alpha_i : [num_shapes], tf.int32
            Parameter related to the width of the scattering lobe in the
            incoming direction.

        lambda_ : [num_shapes], tf.float
            Parameter determining the percentage of the diffusely
            reflected energy in the lobe around the specular reflection.

        scat_keep_prob : float
            Probability with which to keep scattered paths.
            This is helpful to reduce the number of scattered paths computed,
            which might be prohibitively high in some setup.
            Must be in the range (0,1).

        scat_random_phases : bool
            If set to `True` and if scattering is enabled, random uniform phase
            shifts are added to the scattered paths.

        Output
        ------
        paths : :class:`~sionna.rt.Paths`
            Updated paths
        """

        # [num_rx, num_tx, max_num_paths, 2, 2]
        theta_t = paths.theta_t
        phi_t = paths.phi_t
        theta_r = paths.theta_r
        phi_r = paths.phi_r
        types = paths.types

        mat_t = paths_tmp.mat_t
        k_tx = paths_tmp.k_tx
        k_rx = paths_tmp.k_rx

        # Apply multiplication by wavelength/4pi
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths,2, 2]
        cst = tf.cast(self._scene.wavelength/(4.*PI), self._dtype)
        a = cst*mat_t

        # Get dimensions that are needed later on
        num_rx = a.shape[0]
        rx_array_size = a.shape[1]
        num_tx = a.shape[2]
        tx_array_size = a.shape[3]

        # Expand dimension for broadcasting with receivers/transmitters,
        # antenna dimensions, and paths dimensions
        # [1, 1, num_tx, 1, 1, 3, 3]
        tx_rot_mat = insert_dims(insert_dims(tx_rot_mat, 2, 0), 2, 3)
        # [num_rx, 1, 1, 1, 1, 3, 3]
        rx_rot_mat = insert_dims(rx_rot_mat, 4, 1)

        if self._scene.synthetic_array:
            # Expand for broadcasting with antenna dimensions
            # [num_rx, 1, num_tx, 1, max_num_paths, 3]
            k_rx = tf.expand_dims(tf.expand_dims(k_rx, axis=1), axis=3)
            k_tx = tf.expand_dims(tf.expand_dims(k_tx, axis=1), axis=3)
            # [num_rx, 1, num_tx, 1, max_num_paths]
            theta_t = tf.expand_dims(tf.expand_dims(theta_t,axis=1), axis=3)
            phi_t = tf.expand_dims(tf.expand_dims(phi_t, axis=1), axis=3)
            theta_r = tf.expand_dims(tf.expand_dims(theta_r,axis=1), axis=3)
            phi_r = tf.expand_dims(tf.expand_dims(phi_r, axis=1), axis=3)

        # Normalized wave transmit vector in the local coordinate system of
        # the transmitters
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        k_prime_t = tf.linalg.matvec(tx_rot_mat, k_tx, transpose_a=True)

        # Normalized wave receiver vector in the local coordinate system of
        # the receivers
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        k_prime_r = tf.linalg.matvec(rx_rot_mat, k_rx, transpose_a=True)

        # Angles of departure in the local coordinate system of the
        # transmitter
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        theta_prime_t, phi_prime_t = theta_phi_from_unit_vec(k_prime_t)

        # Angles of arrival in the local coordinate system of the
        # receivers
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        theta_prime_r, phi_prime_r = theta_phi_from_unit_vec(k_prime_r)

        # Spherical global frame vectors for tx and rx
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        theta_hat_t = theta_hat(theta_t, phi_t)
        phi_hat_t = phi_hat(phi_t)
        theta_hat_r = theta_hat(theta_r, phi_r)
        phi_hat_r = phi_hat(phi_r)

        # Spherical local frame vectors for tx and rx
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        theta_hat_prime_t = theta_hat(theta_prime_t, phi_prime_t)
        phi_hat_prime_t = phi_hat(phi_prime_t)
        theta_hat_prime_r = theta_hat(theta_prime_r, phi_prime_r)
        phi_hat_prime_r = phi_hat(phi_prime_r)

        # Rotation matrix for going from the spherical LCS to the spherical GCS
        # For transmitters
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths]
        tx_lcs2gcs_11 = dot(theta_hat_t,
                            tf.linalg.matvec(tx_rot_mat, theta_hat_prime_t))
        tx_lcs2gcs_12 = dot(theta_hat_t,
                            tf.linalg.matvec(tx_rot_mat, phi_hat_prime_t))
        tx_lcs2gcs_21 = dot(phi_hat_t,
                            tf.linalg.matvec(tx_rot_mat, theta_hat_prime_t))
        tx_lcs2gcs_22 = dot(phi_hat_t,
                            tf.linalg.matvec(tx_rot_mat, phi_hat_prime_t))
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths,2, 2]
        tx_lcs2gcs = tf.stack(
                    [tf.stack([tx_lcs2gcs_11, tx_lcs2gcs_12], axis=-1),
                     tf.stack([tx_lcs2gcs_21, tx_lcs2gcs_22], axis=-1)],
                    axis=-2)
        tx_lcs2gcs = tf.complex(tx_lcs2gcs, tf.zeros_like(tx_lcs2gcs))
        # For receivers
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths]
        rx_lcs2gcs_11 = dot(theta_hat_r,
                            tf.linalg.matvec(rx_rot_mat, theta_hat_prime_r))
        rx_lcs2gcs_12 = dot(theta_hat_r,
                            tf.linalg.matvec(rx_rot_mat, phi_hat_prime_r))
        rx_lcs2gcs_21 = dot(phi_hat_r,
                            tf.linalg.matvec(rx_rot_mat, theta_hat_prime_r))
        rx_lcs2gcs_22 = dot(phi_hat_r,
                            tf.linalg.matvec(rx_rot_mat, phi_hat_prime_r))
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths,2, 2]
        rx_lcs2gcs = tf.stack(
                    [tf.stack([rx_lcs2gcs_11, rx_lcs2gcs_12], axis=-1),
                     tf.stack([rx_lcs2gcs_21, rx_lcs2gcs_22], axis=-1)],
                    axis=-2)
        rx_lcs2gcs = tf.complex(rx_lcs2gcs, tf.zeros_like(rx_lcs2gcs))

        # List of antenna patterns (callables)
        tx_patterns = self._scene.tx_array.antenna.patterns
        rx_patterns = self._scene.rx_array.antenna.patterns

        tx_ant_fields_hat = []
        for pattern in tx_patterns:
            # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size,
            #   max_num_paths, 2]
            tx_ant_f = tf.stack(pattern(theta_prime_t, phi_prime_t), axis=-1)
            tx_ant_fields_hat.append(tx_ant_f)

        rx_ant_fields_hat = []
        for pattern in rx_patterns:
            # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size,
            #   max_num_paths, 2]
            rx_ant_f = tf.stack(pattern(theta_prime_r, phi_prime_r), axis=-1)
            rx_ant_fields_hat.append(rx_ant_f)

        # Stacking the patterns, corresponding to different polarization
        # directions, as an additional dimension
        # [num_rx, num_rx_patterns, 1/rx_array_size, num_tx, 1/tx_array_size,
        #   max_num_paths, 2]
        rx_ant_fields_hat = tf.stack(rx_ant_fields_hat, axis=1)
        # Expand for broadcasting with tx polarization
        # [num_rx, num_rx_patterns, 1/rx_array_size, num_tx, 1, 1,
        #   1/tx_array_size, max_num_paths, 2]
        rx_ant_fields_hat = tf.expand_dims(rx_ant_fields_hat, axis=4)

        # Stacking the patterns, corresponding to different polarization
        # [num_rx, 1/rx_array_size, num_tx, num_tx_patterns, 1/tx_array_size,
        #   max_num_paths, 2]
        tx_ant_fields_hat = tf.stack(tx_ant_fields_hat, axis=3)
        # Expand for broadcasting with rx polarization
        # [num_rx, 1, 1/rx_array_size, num_tx, num_tx_patterns, 1/tx_array_size,
        #   max_num_paths, 2]
        tx_ant_fields_hat = tf.expand_dims(tx_ant_fields_hat, axis=1)

        # Antenna patterns to spherical global coordinate system
        # Expand to broadcast with antenna patterns
        # [num_rx, 1, 1/rx_array_size, num_tx, 1, 1/tx_array_size,
        #   max_num_paths, 2, 2]
        rx_lcs2gcs = tf.expand_dims(tf.expand_dims(rx_lcs2gcs, axis=1), axis=4)
        # [num_rx, num_rx_patterns, 1/rx_array_size, num_tx, 1, 1/tx_array_size,
        #   max_num_paths, 2]
        rx_ant_fields = tf.linalg.matvec(rx_lcs2gcs, rx_ant_fields_hat)
        # Expand to broadcast with antenna patterns
        # [num_rx, 1, 1/rx_array_size, num_tx, 1, 1/tx_array_size,
        #   max_num_paths, 2, 2]
        tx_lcs2gcs = tf.expand_dims(tf.expand_dims(tx_lcs2gcs, axis=1), axis=4)
        # [num_rx, 1, 1/rx_array_size, num_tx, num_tx_patterns, 1/tx_array_size,
        #   max_num_paths, 2, 2]
        tx_ant_fields = tf.linalg.matvec(tx_lcs2gcs, tx_ant_fields_hat)

        # Expand the field to broadcast with the antenna patterns
        # [num_rx, 1, rx_array_size, num_tx, 1, tx_array_size, max_num_paths,
        #   2, 2]
        a = tf.expand_dims(tf.expand_dims(a, axis=1), axis=4)

        # Compute transmitted field
        # [num_rx, 1, 1/rx_array_size, num_tx, num_tx_patterns, 1/tx_array_size,
        #   max_num_paths, 2]
        a = tf.linalg.matvec(a, tx_ant_fields)

        ## Scattering: For scattering, a is the field specularly reflected by
        # the last interaction point. We need to compute the scattered field.
        # [num_scat_paths]
        scat_ind = tf.where(types == Paths.SCATTERED)[:,0]
        n_scat = tf.size(scat_ind)
        if n_scat > 0:
            n_other = a.shape[-2] - n_scat

            # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
            # This makes no difference on the resulting paths as such paths are
            # not flagged as active.
            # [max_num_paths]
            valid_object_idx = tf.where(paths_tmp.scat_last_objects == -1,
                                        0, paths_tmp.scat_last_objects)

            # Cross-polarization discrimination and scattering coefficients.
            # If a callable is defined to compute the radio material properties,
            # it is invoked. Otherwise, the radio materials of objects are used.
            rm_callable = self._scene.radio_material_callable
            if rm_callable is None:
                # [num_targets, num_sources, max_num_paths]
                k_x = tf.gather(xpd_coefficient, valid_object_idx)
                s = tf.gather(scattering_coefficient, valid_object_idx)
                etas = tf.gather(etas, valid_object_idx)
            else:
                # [num_targets, num_sources, max_num_paths]
                etas, s, k_x = rm_callable(paths_tmp.scat_last_objects,
                                           paths_tmp.scat_last_vertices)

            # Generate random phase shifts, and compute field vector
            phase_shape = tf.concat([tf.shape(k_x), [2]], axis=0)
            if scat_random_phases:
                # [num_targets, num_sources, max_num_paths, 2]
                phases = tf.random.uniform(phase_shape, maxval=2*PI,
                                        dtype=self._rdtype)
            else:
                phases = tf.zeros(phase_shape, dtype=self._rdtype)
            # [num_targets, num_sources, max_num_paths, 2]
            field_vec = tf.exp(tf.complex(tf.cast(0, self._rdtype), phases))
            # [num_targets, num_sources, max_num_paths, 2]
            k_x_ = tf.stack([tf.sqrt(1-k_x), tf.sqrt(k_x)], axis=-1)
            k_x_ = tf.complex(k_x_, tf.zeros_like(k_x_))
            field_vec *= k_x_

            # Evaluate scattering pattern for all paths.
            # If a callable is defined to compute the scattering pattern,
            # it is invoked. Otherwise, the radio materials of objects are used.
            sp_callable = self._scene.scattering_pattern_callable
            if sp_callable is None:
                # Get all material properties related to scattering for each
                # path
                # [num_targets, num_sources, max_num_paths]
                alpha_r = tf.gather(alpha_r, valid_object_idx)
                alpha_i = tf.gather(alpha_i, valid_object_idx)
                lambda_ = tf.gather(lambda_, valid_object_idx)
                # Flattening is needed here as the pattern cannot handle it
                # otherwise
                f_s = ScatteringPattern.pattern(
                                tf.reshape(paths_tmp.scat_last_k_i, [-1, 3]),
                                tf.reshape(paths_tmp.scat_k_s, [-1, 3]),
                                tf.reshape(paths_tmp.scat_last_normals,[-1, 3]),
                                tf.reshape(alpha_r, [-1]),
                                tf.reshape(alpha_i, [-1]),
                                tf.reshape(lambda_, [-1]))
                # Reshape f_s to original dimensions
                # [num_targets, num_sources, max_num_paths]
                f_s = tf.reshape(f_s, tf.shape(alpha_r))
            else:
                # [num_targets, num_sources, max_num_paths]
                f_s = sp_callable(paths_tmp.scat_last_objects,
                                  paths_tmp.scat_last_vertices,
                                  paths_tmp.scat_last_k_i,
                                  paths_tmp.scat_k_s,
                                  paths_tmp.scat_last_normals)

            # Complete the computation of the field
            # [num_targets, num_sources, max_num_paths]
            scaling = tf.sqrt(f_s)*s

            # The term cos(theta_i)*dA is equal to 4*PI/N*r^2
            # [num_targets, num_sources, max_num_paths]
            num_samples = tf.cast(num_samples, self._rdtype)
            scaling *= tf.sqrt(4*tf.cast(PI, self._rdtype)\
                /(scat_keep_prob*num_samples))
            scaling *= paths_tmp.scat_src_2_last_int_dist

            # Apply path loss due to propagation from scattering point
            # to target
            # [num_targets, num_sources, max_num_paths]
            scaling = tf.math.divide_no_nan(scaling,
                                            paths_tmp.scat_2_target_dist)

            # Compute scaled field vector
            # [num_targets, num_sources, max_num_paths, 2]
            field_vec *= tf.expand_dims(tf.complex(scaling,
                                                   tf.zeros_like(scaling)), -1)

            # Compute Fresnel reflection coefficients at hit point
            # These will be scaled by the reflection reduction factor
            # [num_targets, num_sources, max_num_paths]
            cos_theta = -dot(paths_tmp.scat_last_k_i,
                             paths_tmp.scat_last_normals, clip=True)

            # [num_targets, num_sources, max_num_paths]
            r_s, r_p = reflection_coefficient(etas, cos_theta)

            # [num_targets, num_sources, max_num_paths, 3]
            e_i_s, e_i_p = compute_field_unit_vectors(
                                        paths_tmp.scat_last_k_i,
                                        paths_tmp.scat_k_s,
                                        paths_tmp.scat_last_normals,
                                        SolverBase.EPSILON,
                                        return_e_r=False)

            # a_scat : [num_rx, 1, rx_array_size, num_tx, num_tx_patterns,
            #   tx_array_size, n_scat, 2]
            # a_other : [num_rx, 1, rx_array_size, num_tx, num_tx_patterns,
            #   tx_array_size, max_num_paths - n_scat, 2]
            a_other, a_scat = tf.split(a, [n_other, n_scat], axis=-2)
            # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size,
            #   max_num_paths, 3]
            _, scat_theta_hat_r = tf.split(theta_hat_r, [n_other, n_scat],
                                           axis=-2)
            # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size,
            #   max_num_paths, 3]
            _, scat_phi_hat_r = tf.split(phi_hat_r, [n_other, n_scat],
                                           axis=-2)

            # Compute incoming field
            # [num_rx, 1, 1/rx_array_size, num_tx, 1, 1/tx_array_size, n_scat,
            #   (3)]
            scat_k_i = paths_tmp.scat_last_k_i
            if self._scene.synthetic_array:
                r_s = insert_dims(r_s, 2, axis=1)
                r_s = insert_dims(r_s, 2, axis=4)
                r_p = insert_dims(r_p, 2, axis=1)
                r_p = insert_dims(r_p, 2, axis=4)
                e_i_s = insert_dims(e_i_s, 2, axis=1)
                e_i_s = insert_dims(e_i_s, 2, axis=4)
                e_i_p = insert_dims(e_i_p, 2, axis=1)
                e_i_p = insert_dims(e_i_p, 2, axis=4)
                scat_k_i = insert_dims(scat_k_i, 2, axis=1)
                scat_k_i = insert_dims(scat_k_i, 2, axis=4)
                field_vec = insert_dims(field_vec, 2, axis=1)
                field_vec = insert_dims(field_vec, 2, axis=4)
            else:
                num_rx = len(self._scene.receivers)
                num_tx = len(self._scene.transmitters)
                r_s = split_dim(r_s, [num_rx, -1], 0)
                r_s = tf.expand_dims(r_s, axis=1)
                r_s = split_dim(r_s, [num_tx, -1], 3)
                r_s = tf.expand_dims(r_s, axis=4)
                r_p = split_dim(r_p, [num_rx, -1], 0)
                r_p = tf.expand_dims(r_p, axis=1)
                r_p = split_dim(r_p, [num_tx, -1], 3)
                r_p = tf.expand_dims(r_p, axis=4)
                e_i_s = split_dim(e_i_s, [num_rx, -1], 0)
                e_i_s = tf.expand_dims(e_i_s, axis=1)
                e_i_s = split_dim(e_i_s, [num_tx, -1], 3)
                e_i_s = tf.expand_dims(e_i_s, axis=4)
                e_i_p = split_dim(e_i_p, [num_rx, -1], 0)
                e_i_p = tf.expand_dims(e_i_p, axis=1)
                e_i_p = split_dim(e_i_p, [num_tx, -1], 3)
                e_i_p = tf.expand_dims(e_i_p, axis=4)
                scat_k_i = split_dim(scat_k_i, [num_rx, -1], 0)
                scat_k_i = tf.expand_dims(scat_k_i, axis=1)
                scat_k_i = split_dim(scat_k_i, [num_tx, -1], 3)
                scat_k_i = tf.expand_dims(scat_k_i, axis=4)
                field_vec = split_dim(field_vec, [num_rx, -1], 0)
                field_vec = tf.expand_dims(field_vec, axis=1)
                field_vec = split_dim(field_vec, [num_tx, -1], 3)
                field_vec = tf.expand_dims(field_vec, axis=4)

            # [num_rx, 1, 1/rx_array_size, num_tx, 1, 1/tx_array_size, n_scat,2]
            scat_r = tf.stack([r_s, r_p], axis=-1)

            # [num_rx, 1, 1/rx_array_size, num_tx, num_tx_patterns,
            #   1/tx_array_size, n_scat, 2]
            a_in = tf.math.divide_no_nan(a_scat, scat_r)

            # Compute polarization field vector
            a_in_s, a_in_p = tf.split(a_in, 2, axis=-1)
            e_i_s = tf.complex(e_i_s, tf.zeros_like(e_i_s))
            e_i_p = tf.complex(e_i_p, tf.zeros_like(e_i_p))
            e_in_pol = a_in_s*e_i_s + a_in_p*e_i_p
            e_pol_hat, _ = normalize(tf.math.real(e_in_pol))
            e_xpol_hat = cross(e_pol_hat, scat_k_i)

            # Compute incoming spherical unit vectors in GCS
            scat_theta_i, scat_phi_i = theta_phi_from_unit_vec(-scat_k_i)
            scat_theta_hat_i = theta_hat(scat_theta_i, scat_phi_i)
            scat_phi_hat_i = phi_hat(scat_phi_i)

            # Transformation to theta_hat_i, phi_hat_i
            trans_mat = component_transform(e_pol_hat, e_xpol_hat,
                                            scat_theta_hat_i, scat_phi_hat_i)

            # Transformation from theta_hat_s, phi_hat_s to theta_hat_r, phi_hat_r
            # [num_targets, num_sources, max_num_paths, 3]
            # = [num_rx*1/rx_array_size, num_tx*1/tx_array_size, max_num_paths, 3]
            scat_theta_s, scat_phi_s = theta_phi_from_unit_vec(paths_tmp.scat_k_s)
            scat_theta_hat_s = theta_hat(scat_theta_s, scat_phi_s)
            scat_phi_hat_s = phi_hat(scat_phi_s)

            # [num_rx, 1/rx_array_size, num_sources, max_num_paths, 3]
            scat_theta_hat_s = split_dim(scat_theta_hat_s,
                                         [num_rx, rx_array_size], 0)
            scat_phi_hat_s = split_dim(scat_phi_hat_s,
                                       [num_rx, rx_array_size], 0)

            # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
            scat_theta_hat_s = split_dim(scat_theta_hat_s,
                                         [num_tx, tx_array_size], 2)
            scat_phi_hat_s = split_dim(scat_phi_hat_s,
                                       [num_tx, tx_array_size], 2)

            # [num_rx, 1,  1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
            scat_theta_hat_s = tf.expand_dims(scat_theta_hat_s, 1)
            scat_phi_hat_s = tf.expand_dims(scat_phi_hat_s, 1)

            # [num_rx, 1,  1/rx_array_size, num_tx, 1, 1/tx_array_size, max_num_paths, 3]
            scat_theta_hat_s = tf.expand_dims(scat_theta_hat_s, 4)
            scat_phi_hat_s = tf.expand_dims(scat_phi_hat_s, 4)

            # [num_rx, 1, 1/rx_array_size, num_tx, 1, 1/tx_array_size,
            #   max_num_scat_paths, 3]
            scat_theta_hat_r = tf.expand_dims(scat_theta_hat_r, axis=1)
            scat_theta_hat_r = tf.expand_dims(scat_theta_hat_r, axis=4)
            # [num_rx, 1, 1/rx_array_size, num_tx, 1, 1/tx_array_size,
            #   max_num_scat_paths, 3]
            scat_phi_hat_r = tf.expand_dims(scat_phi_hat_r, axis=1)
            scat_phi_hat_r = tf.expand_dims(scat_phi_hat_r, axis=4)

            trans_mat2 = component_transform(scat_theta_hat_s, scat_phi_hat_s,
                                             scat_theta_hat_r, scat_phi_hat_r)

            trans_mat = tf.matmul(trans_mat2, trans_mat)

            # Compute basis transform matrix for GCS
            # [num_rx, 1, rx_array_size, num_tx, num_tx_patterns, tx_array_size,
            #   max_num_scat_paths, 2, 2]
            trans_mat = tf.complex(trans_mat, tf.zeros_like(trans_mat))

            # Multiply a_scat by sqrt of reflected energy
            # The splitting along the last dim is done because
            # TF cannot handle reduce_sum for such high-dimensional
            # tensors
            #
            # [num_rx, 1, 1/rx_array_size, num_tx, num_tx_patterns,
            #   1/tx_array_size, max_num_paths-n_scat, 1]
            e_spec = tf.reduce_sum(tf.square(tf.abs(a_scat)), axis=-1,
                                   keepdims=True)
            e_spec = tf.sqrt(e_spec)

            # [num_rx, 1, 1/rx_array_size, num_tx, num_tx_patterns,
            #   1/tx_array_size, max_num_paths-n_scat, 2]
            e_spec = tf.complex(e_spec, tf.zeros_like(e_spec))
            a_scat = field_vec*e_spec

            # Basis transform
            a_scat = tf.linalg.matvec(trans_mat, a_scat)

            # Concat with other paths
            a = tf.concat([a_other, a_scat], axis=-2)

        # [num_rx, num_rx_patterns, 1/rx_array_size, num_tx, num_tx_patterns,
        #   1/tx_array_size, max_num_paths]
        a = dot(rx_ant_fields, a)

        if not self._scene.synthetic_array:
            # Reshape as expected to merge antenna and antenna patterns into one
            # dimension, as expected by Sionna
            # [ num_rx, num_rx_ant = num_rx_patterns*rx_array_size,
            #   num_tx, num_tx_ant = num_tx_patterns*tx_array_size,
            #   max_num_paths]
            a = flatten_dims(flatten_dims(a, 2, 1), 2, 3)

        return a

    # def _vertex_diffraction_create_paths(self, diff_paths, diff_paths_tmp, sources, targets, vertex_indices, \
    #                                      diff_vertex_points, vertex_wedges_indices, valid_vertex_idxs):
    #     diff_paths.objects = tf.expand_dims(vertex_indices, axis=0)
    #     diff_paths.vertices = tf.expand_dims(diff_vertex_points, axis=0)
    #     # diff_paths.vertex_wedges_indices = tf.expand_dims(vertex_wedges_indices, axis=0)
    #     # diff_paths.valid_vertex_idxs = tf.expand_dims(valid_vertex_idxs, axis=0)

    #     #diff_paths = self._gather_valid_diff_paths(diff_paths)
    #     diff_paths = self.vertex_diffraction._gather_valid_diff_paths(diff_paths)

    #     diff_paths, diff_paths_tmp =\
    #             self._compute_directions_distances_delays_angles(diff_paths,
    #                                                     diff_paths_tmp, False, True)
        
    #     return diff_paths, diff_paths_tmp
    
    ### Methods to find reflection and diffraction combinations
    def _comb_diff_refl(self, mirrored_vertices, tri_p0, normals, candidates, targets, sources):
        r"""
        Compute the combination of diffraction and reflection paths.
        Tx -> reflection -> diffraction -> Rx

        Input
        ------

        mirrored_vertices: [max_depth, num_sources, num_samples, 3], tf.float
            Source images

        tri_p0: [max_depth, num_sources, num_samples, 3], tf.float

        normals: [max_depth, num_sources, num_samples, 3], tf.float

        candidates: [max_depth, num_samples], int

        targets: [num_targets, 3], tf.float

        sources: [num_sources, 3], tf.float
        """
        def _get_unique_images(x):
            y, idxs = tf.unique(x=x)
            equal_matrix = tf.equal(tf.expand_dims(x, 0), tf.expand_dims(y, 1))
            my_idxs = tf.argmax(tf.cast(equal_matrix, tf.int32), axis=1)

            if tf.gather(candidates[0], my_idxs)[0] == -1:
                # Skip LOS primitive
                my_idxs = my_idxs[1:]

            return my_idxs

        num_sources = tf.shape(sources)[0]
        num_targets = tf.shape(targets)[0]

        # [num_images]
        my_idxs = _get_unique_images(candidates[0])
        mirror_candidates = tf.gather(candidates[0], my_idxs)

        #[num_sources, num_images, 3]
        p0 = tf.gather(tri_p0[0], my_idxs, axis=1)
        n = tf.gather(normals[0], my_idxs, axis=1)

        # [num_sources, num_images, 3]
        source_images = tf.gather(mirrored_vertices[0], my_idxs, axis=1)
        # [num_sources * num_images, 3]
        source_images = tf.reshape(source_images, [-1, 3])

        # los primitives for source images
        los_prim = candidates[1]

        if self.obj_geom is not None:
            los_prim = tf.concat((los_prim, self.obj_geom.primitive_idxs_dd), axis=0)

        los_prim = tf.gather(los_prim, tf.where(tf.not_equal(los_prim, -1))[:,0])
        los_prim, _ = tf.unique(x=los_prim)

        # find diffraction from source images to targets
        diff_paths, diff_paths_tmp = self.solver_wedge_diffraction.tracing(source_images, targets, \
                                        los_prim, True, False)
        
        diff_paths, diff_paths_tmp = self.solver_wedge_diffraction.check_obstructions( \
                diff_paths, diff_paths_tmp, True, case='rx')

        # diff_paths.vertices: [1, num_targets, num_sources * num_images, num_difr_paths, 3]
        num_difr_points = diff_paths.vertices.shape[3]
        # [num_targets, num_sources, num_images * num_difr_paths, 3]
        difr_vertices = tf.reshape(diff_paths.vertices[0], [num_targets, num_sources, -1, 3])      
        current = difr_vertices

        # [num_sources * num_images]
        prim_idxs = tf.repeat(mirror_candidates, num_difr_points)

        # [num_sources, num_images * num_difr_points, 3]
        p0 = tf.repeat(p0, num_difr_points, axis=1)
        n = tf.repeat(n, num_difr_points, axis=1)
        # [num_sources, num_images, 3]
        next_pos = tf.reshape(source_images, [num_sources, -1, 3])
        # [num_sources, num_images * num_difr_paths, 3]
        next_pos = tf.repeat(next_pos, num_difr_points, axis=1)

        # [num_targets, num_sources, num_images * num_difr_paths, 3]
        valid = tf.reshape(diff_paths.mask, [num_targets, num_sources, -1])

        # image method:
        # find intersections from diffraction points to source images 
        # via corresponding prim_idxs
        output = self.solver_reflection._spec_image_method_phase_21_depth_1(prim_idxs, valid,
                next_pos, p0, n, current, num_targets,
                num_sources)
        
        # [num_targets, num_sources, num_samples]
        valid = output[0]
        # [num_targets, num_sources, num_samples, 3]
        #   Positions of the last interactions
        current = output[1]
        # : [num_targets, num_sources, num_samples, 3], tf.float
        #     Intersection point on the primitive
        path_vertices = output[2]
        # : [num_targets, num_sources, num_samples, 3], tf.float
        #    Normals to the primitive at the intersection point
        path_normals = output[3]
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
        blk = self._test_obstruction(tf.reshape(current, [-1, 3]),
                                        tf.reshape(d, [-1, 3]),
                                        tf.reshape(maxt, [-1]))

        # The following call:
        # - Discards paths that are blocked
        # - Discards paths for which ``current`` point and the ``next_pos``
        #   are not on the same side, as this would mean that the path is
        #   going through the surface
        output = self.solver_reflection._spec_image_method_phase_22_depth_1(valid,
            next_pos, current, blk, num_targets, num_sources, maxt,
            path_vertices, active)
        # [num_targets, num_sources, num_samples]
        #   Mask indicating the valid paths
        valid = output[0]
        # [num_targets, num_sources, num_samples, xyz : 3]
        #   Positions of the last interactions
        current = output[1]

        # # [num_targets, num_sources, num_samples, 3]  ???
        # path_vertices = tf.stack(path_vertices, axis=0)
        # path_normals = tf.stack(path_normals, axis=0)

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
        current, d, maxt = self.solver_reflection._spec_image_method_phase_23(current, sources,
                                                      num_targets)

        # Test for obstruction using Mitsuba
        # [num_targets*num_sources*num_samples]
        val = self._test_obstruction(tf.reshape(current, [-1, 3]),
                                     tf.reshape(d, [-1, 3]),
                                     tf.reshape(maxt, [-1]))
        # [num_targets, num_sources, num_samples, 3]
        blk = tf.reshape(val, tf.shape(maxt))
        # Discard paths for which the shooted ray has zero-length, i.e., when
        # two consecutive intersection points have the same location, or when
        # the source and target have the same locations (RADAR).
        # [num_targets, num_sources, num_samples]
        blk = tf.logical_or(blk, tf.less(maxt, self.EPSILON))

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

        # [2, num_targets, num_sources, num_samples, 3]
        #path_vertices = tf.expand_dims(path_vertices, axis=0)
        vertices = tf.stack([path_vertices, difr_vertices], 
                              axis=0)
        
        # combine diffraction and reflection paths
        # diff_paths.objects : [1, num_targets, num_sources * num_images, num_difr_paths]
        # [num_targets, num_sources, num_samples]
        difr_objs = tf.reshape(diff_paths.objects[0], [num_targets, num_sources, -1])
        # [1, 1, num_samples, 1]
        refl_objs = tf.reshape(prim_idxs, [1, 1, -1])
        primitives_2_objects = tf.pad(self._primitives_2_objects, [[0,1]], constant_values=-1)
        valid_refl_objs = tf.gather(primitives_2_objects, refl_objs)
        # [num_targets, num_sources, num_samples, 1]
        valid_refl_objs = tf.repeat(valid_refl_objs, num_targets, axis=0)
        valid_refl_objs = tf.repeat(valid_refl_objs, num_sources, axis=1)
        # [2, num_targets, num_sources, num_samples, 1]
        obj_idxs = tf.stack([valid_refl_objs, difr_objs], axis=0)

        mask, valid_vertices, valid_obj_idxs, gather_indices, scatter_indices =\
            self.solver_reflection._paths_post_process(obj_idxs, valid, num_targets,
                                num_sources, vertices, blk)

        valid_normals = tf.zeros(valid_vertices.shape[1:], dtype=self._rdtype)
        valid_normals = tf.tensor_scatter_nd_update(valid_normals, scatter_indices, tf.gather_nd(path_normals, gather_indices))

        combined_paths = Paths(sources=sources, targets=targets, scene=self._scene, types=Paths.REFL_DIFF)
        combined_paths_tmp = PathsTmpData(sources, targets, self._dtype)
        combined_paths_tmp.normals = valid_normals

        combined_paths.mask = mask
        combined_paths.vertices = valid_vertices
        combined_paths.objects = valid_obj_idxs
        combined_paths.types = Paths.REFL_DIFF

        combined_paths, combined_paths_tmp =\
                self._compute_directions_distances_delays_angles(combined_paths,
                                                        combined_paths_tmp, False)

        return combined_paths, combined_paths_tmp
    

    def _comb_vd_refl(self, mirrored_vertices, tri_p0, candidates, normals, targets, sources):
        r"""
        Compute the combination of diffraction and reflection paths.
        Tx -> reflection -> diffraction -> Rx

        Input
        ------

        mirrored_vertices: [max_depth, num_sources, num_samples, 3], tf.float
            Source images

        tri_p0: [max_depth, num_sources, num_samples, 3], tf.float

        normals: [max_depth, num_sources, num_samples, 3], tf.float

        candidates: [max_depth, num_samples], int

        targets: [num_targets, 3], tf.float

        sources: [num_sources, 3], tf.float
        """

        def _get_unique_images(x):
            y, idxs = tf.unique(x=x)
            equal_matrix = tf.equal(tf.expand_dims(x, 0), tf.expand_dims(y, 1))
            my_idxs = tf.argmax(tf.cast(equal_matrix, tf.int32), axis=1)

            if tf.gather(candidates[0], my_idxs)[0] == -1:
                # Skip LOS primitive
                my_idxs = my_idxs[1:]

            return my_idxs

        num_sources = tf.shape(sources)[0]
        num_targets = tf.shape(targets)[0]

        # [num_images]
        my_idxs = _get_unique_images(candidates[0])
        mirror_candidates = tf.gather(candidates[0], my_idxs)

        #[num_sources, num_images, 3]
        p0 = tf.gather(tri_p0[0], my_idxs, axis=1)
        n = tf.gather(normals[0], my_idxs, axis=1)

        # [num_sources, num_images, 3]
        source_images = tf.gather(mirrored_vertices[0], my_idxs, axis=1)
        # [num_sources * num_images, 3]
        source_images = tf.reshape(source_images, [-1, 3])

        vd_paths, vd_paths_tmp = self.vertex_diffraction.tracing(source_images, targets, case='rx', is_dd=True)

        if vd_paths.vertices.shape[3] == 0:
            vd_refl_paths = Paths(sources=sources, targets=targets, scene=self._scene, types=Paths.VD_REFL)
            vd_refl_paths_tmp = PathsTmpData(sources, targets, self._dtype)
            return vd_refl_paths, vd_refl_paths_tmp

        diff_vertex_points = tf.reshape(vd_paths.vertices[0], [num_targets, num_sources, -1, 3])
        vertex_indices = tf.reshape(vd_paths.objects[0], [num_targets, num_sources, -1]) 

        num_difr_points = vd_paths.vertices.shape[3]

        current = diff_vertex_points

        # [num_sources * num_images]
        prim_idxs = tf.repeat(mirror_candidates, num_difr_points)

        # [num_sources, num_images * num_difr_points, 3]
        p0 = tf.repeat(p0, num_difr_points, axis=1)
        n = tf.repeat(n, num_difr_points, axis=1)
        # [num_sources, num_images, 3]
        next_pos = tf.reshape(source_images, [num_sources, -1, 3])
        # [num_sources, num_images * num_difr_paths, 3]
        next_pos = tf.repeat(next_pos, num_difr_points, axis=1)

        # [num_targets, num_sources, num_images * num_difr_paths, 3]
        valid = tf.reshape(vd_paths.mask, [num_targets, num_sources, -1])

        # image method:
        # find intersections from diffraction points to source images 
        # via corresponding prim_idxs
        output = self.solver_reflection._spec_image_method_phase_21_depth_1(prim_idxs, valid,
                next_pos, p0, n, current, num_targets,
                num_sources)
        
        # num_samples = num_images * num_difr_points
        # [num_targets, num_sources, num_samples]
        valid = output[0]
        # [num_targets, num_sources, num_samples, 3]
        #   Positions of the last interactions
        current = output[1]
        # : [num_targets, num_sources, num_samples, 3], tf.float
        #     Intersection point on the primitive
        path_vertices = output[2]
        # : [num_targets, num_sources, num_samples, 3], tf.float
        #    Normals to the primitive at the intersection point
        path_normals = output[3]
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
        blk = self._test_obstruction(tf.reshape(current, [-1, 3]),
                                        tf.reshape(d, [-1, 3]),
                                        tf.reshape(maxt, [-1]))

        output = self.solver_reflection._spec_image_method_phase_22_depth_1(valid,
            next_pos, current, blk, num_targets, num_sources, maxt,
            path_vertices, active)
        # [num_targets, num_sources, num_samples]
        #   Mask indicating the valid paths
        valid = output[0]
        # [num_targets, num_sources, num_samples, xyz : 3]
        #   Positions of the last interactions
        current = output[1]

        current, d, maxt = self.solver_reflection._spec_image_method_phase_23(current, sources,
                                                      num_targets)

        # Test for obstruction using Mitsuba
        # [num_targets*num_sources*num_samples]
        val = self._test_obstruction(tf.reshape(current, [-1, 3]),
                                     tf.reshape(d, [-1, 3]),
                                     tf.reshape(maxt, [-1]))
        # [num_targets, num_sources, num_samples, 3]
        blk = tf.reshape(val, tf.shape(maxt))
        # Discard paths for which the shooted ray has zero-length, i.e., when
        # two consecutive intersection points have the same location, or when
        # the source and target have the same locations (RADAR).
        # [num_targets, num_sources, num_samples]
        blk = tf.logical_or(blk, tf.less(maxt, self.EPSILON))

        _vertices = tf.stack([path_vertices, diff_vertex_points], axis=0)
        
        # combine diffraction and reflection paths
        # [num_targets, num_sources, num_samples]
        difr_objs = tf.reshape(vertex_indices, [num_targets, num_sources, -1])
        # [1, 1, num_samples, 1]
        refl_objs = tf.reshape(prim_idxs, [1, 1, -1])
        primitives_2_objects = tf.pad(self._primitives_2_objects, [[0,1]], constant_values=-1)
        valid_refl_objs = tf.gather(primitives_2_objects, refl_objs)
        # [num_targets, num_sources, num_samples, 1]
        valid_refl_objs = tf.repeat(valid_refl_objs, num_targets, axis=0)
        valid_refl_objs = tf.repeat(valid_refl_objs, num_sources, axis=1)
        # [2, num_targets, num_sources, num_samples, 1]
        obj_idxs = tf.stack([valid_refl_objs, difr_objs], axis=0)

        mask, valid_vertices, valid_obj_idxs, gather_indices, scatter_indices =\
            self.solver_reflection._paths_post_process(obj_idxs, valid, num_targets,
                                num_sources, _vertices, blk)

        valid_normals = tf.zeros(valid_vertices.shape[1:], dtype=self._rdtype)
        valid_normals = tf.tensor_scatter_nd_update(valid_normals, scatter_indices, tf.gather_nd(path_normals, gather_indices))
        
        vd_refl_paths = Paths(sources=sources, targets=targets, scene=self._scene, types=Paths.VD_REFL)
        vd_refl_paths_tmp = PathsTmpData(sources, targets, self._dtype)
        vd_refl_paths_tmp.normals = valid_normals

        vd_refl_paths.mask = mask
        vd_refl_paths.vertices = valid_vertices
        vd_refl_paths.objects = valid_obj_idxs

        vd_refl_paths, vd_refl_paths_tmp =\
                self._compute_directions_distances_delays_angles(vd_refl_paths,
                                                        vd_refl_paths_tmp, False, True)
     
        return vd_refl_paths, vd_refl_paths_tmp
    

def merge_paths(paths, is_dd=True):
    # both empty
    if paths[0].objects.shape[3] == 0 and paths[1].objects.shape[3] == 0:
        return paths[0]

    all_paths = paths[0]
    for i, path in enumerate(paths):
        if path.objects.shape[3] != 0:
            all_paths = path  # non-empty path
            idx = i
            break

    for i, path in enumerate(paths):
        if i == idx:
            continue

        if path.objects.shape[3] == 0:
            continue

        all_paths.objects = tf.concat([all_paths.objects, path.objects], axis=3)
        all_paths.true_objects = tf.concat([all_paths.true_objects, path.true_objects], axis=3)
        all_paths.my_types = tf.concat([all_paths.my_types, path.my_types], axis=3)
        all_paths.vertices = tf.concat([all_paths.vertices, path.vertices], axis=3)
        all_paths.mask = tf.concat([all_paths.mask, path.mask], axis=2)
        all_paths.targets_sources_mask = tf.concat([all_paths.targets_sources_mask, path.targets_sources_mask], axis=2)
        all_paths.types = all_paths.types
        all_paths.theta_t = tf.concat([all_paths.theta_t, path.theta_t], axis=2)
        all_paths.phi_t = tf.concat([all_paths.phi_t, path.phi_t], axis=2)
        all_paths.theta_r = tf.concat([all_paths.theta_r, path.theta_r], axis=2)
        all_paths.phi_r = tf.concat([all_paths.phi_r, path.phi_r], axis=2)
        all_paths.tau = tf.concat([all_paths.tau, path.tau], axis=2)

        if is_dd:
            all_paths.dd_type = tf.concat([all_paths.dd_type, path.dd_type], axis=2)

    return all_paths

def merge_paths_tmp(paths):
    if paths[0].k_i.shape[3] == 0 and paths[1].k_i.shape[3] == 0:
        return paths[0]

    all_paths = paths[0]
    for i, path in enumerate(paths):
        if path.k_i.shape[3] != 0:
            all_paths = path  # non-empty path
            idx = i
            break

    for i, path in enumerate(paths):
        if i == idx:
            continue

        if path.k_i.shape[3] == 0:
            continue

        all_paths.k_i = tf.concat([all_paths.k_i, path.k_i], axis=3)
        all_paths.k_r = tf.concat([all_paths.k_r, path.k_r], axis=3)
        all_paths.k_tx = tf.concat([all_paths.k_tx, path.k_tx], axis=2)
        all_paths.k_rx = tf.concat([all_paths.k_rx, path.k_rx], axis=2)
        all_paths.total_distance = tf.concat([all_paths.total_distance, path.total_distance], axis=2)

        try:
            all_paths.normals = tf.concat([all_paths.normals, path.normals], axis=2)
        except:
            pass

    return all_paths