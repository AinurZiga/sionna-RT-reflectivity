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


    # Enable before tracing to retain WD endpoint decisions for VD diagnostics.
    # Shared by reflection/WD and VD/WD warnings; never changes path selection.
    warn_sb_side_mismatch = False
    sb_warning_tol = 1e-5  # Radians

    def trace_paths(self, max_depth, method, num_samples, los, reflection, diffraction,
                 refl_diff, scattering, scat_keep_prob, scat_sparsity_coef, edge_diffraction, vertex_diffraction, 
                 double_diffraction, transmission):
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
            Keep every ``scat_sparsity_coef``th diffuse launch direction per source.

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
        geom = self.obj_geom
        has_objects = geom is not None and hasattr(geom, "active_primitives")
        env_rwd = (
            method == "fibonacci" and has_objects
            and geom.enable_env_object_rwd and max_depth >= 2
        )
        env_rvd = (
            method == "fibonacci" and has_objects
            and geom.enable_env_object_rvd and max_depth >= 2
        )
        if not (reflection or scattering or env_rwd or env_rvd):
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

        wd_devices = vd_candidates = None
        rwd_devices = rvd_devices = None
        env_first_hits = None
        registered_mask = self._registered_primitive_mask() if (has_objects and refl_diff) else None
        dd_primitive_ids = None
        dd_los_prim = dd_los_prim_t = None
        los_prim_t = _candidates_t = None
        scat_samples_per_source = 0

        if method == "exhaustive":
            if scattering:
                raise ValueError("Diffuse scattering requires method='fibonacci'.")
            candidates, los_prim = self._list_candidates_exhaustive(max_depth, los, reflection)
            los_prim_t = los_prim
            _candidates = candidates

        elif method == "env_driven":
            if scattering:
                raise ValueError("Diffuse scattering requires method='fibonacci'.")
            if refl_diff:
                raise ValueError("env_driven does not yet support mixed reflection-diffraction.")

            primitive_ids = (
                geom.primitive_idxs if has_objects
                else tf.range(tf.shape(self._primitives)[0], dtype=tf.int32)
            )
            tx_visible, rx_visible = self._build_env_candidates(
                sources, targets, primitive_ids, transmission
            )
            los_prim = tf.boolean_mask(primitive_ids, tf.reduce_any(tx_visible, axis=0))
            los_prim_t = tf.boolean_mask(primitive_ids, tf.reduce_any(rx_visible, axis=0))
            reflection_prims, _ = tf.unique(tf.concat([los_prim, los_prim_t], axis=0))
            if not reflection or max_depth == 0:
                reflection_prims = reflection_prims[:0]
            candidates = reflection_prims[None, :]
            if los:
                candidates = tf.concat([tf.constant([[-1]], tf.int32), candidates], axis=1)
            _candidates = candidates

            if diffraction or double_diffraction:
                env_edges = (
                    self._env_edge_rows(primitive_ids, tx_visible),
                    self._env_edge_rows(primitive_ids, rx_visible),
                )
                if diffraction:
                    wd_devices = env_edges
                if double_diffraction:
                    dd_primitive_ids = primitive_ids
                    dd_tx_visible, dd_rx_visible = tx_visible, rx_visible
                    dd_tx_edges, dd_rx_edges = env_edges
                    dd_los_prim, dd_los_prim_t = los_prim, los_prim_t
            if vertex_diffraction and has_objects:
                vd_candidates = self._build_env_vertex_rows(
                    primitive_ids, tx_visible, rx_visible, geom.active_vertices
                )

        elif method == "fibonacci":
            candidates, los_prim, candidates_scat, hit_points, scat_samples_per_source = (
                self._list_candidates_fibonacci(
                    max_depth, sources, num_samples, los, reflection, scattering,
                    scat_sparsity_coef,
                )
            )
            _candidates = candidates

            # Preserve the receiver-side SBR pass for experimental reflection-diffraction.
            if (refl_diff and reflection and max_depth >= 2
                    and (geom is None or geom.is_sbr_rxs)):
                _candidates_t, *_ = self._list_candidates_fibonacci(
                    2, targets, num_samples, False, True, False, 1
                )

            env_specular = has_objects and geom.add_obj_primitives and reflection and max_depth > 0
            env_rr = has_objects and geom.enable_env_object_rr and reflection and max_depth >= 2
            env_wd = has_objects and geom.add_obj_primitives and diffraction
            env_vd = has_objects and vertex_diffraction
            env_dd = has_objects and double_diffraction

            need_first_order = env_specular or env_rr or env_rwd or env_rvd or env_wd or env_vd
            if need_first_order or env_dd:
                # Trace the union once; first-order and double diffraction use
                # different object masks.
                first_order = geom.active_primitives if need_first_order else (
                    tf.zeros_like(geom.active_primitives)
                )
                second_order = geom.active_primitives_dd if env_dd else tf.zeros_like(first_order)
                selected = first_order | second_order
                selected_ids = tf.boolean_mask(geom._primitive_idxs, selected)
                env_tx, env_rx = self._build_env_candidates(
                    sources, targets, selected_ids, transmission
                )

                if need_first_order:
                    first = tf.boolean_mask(first_order, selected)
                    primitive_ids = tf.boolean_mask(selected_ids, first)
                    tx_visible = tf.boolean_mask(env_tx, first, axis=1)
                    rx_visible = tf.boolean_mask(env_rx, first, axis=1)

                    if env_specular:
                        facets = tf.boolean_mask(
                            primitive_ids, tf.reduce_any(tx_visible, axis=0)
                            | tf.reduce_any(rx_visible, axis=0)
                        )
                        candidates = self._add_env_reflection_candidates(candidates, facets)
                    if env_wd or env_rwd:
                        object_edges = (
                            self._env_edge_rows(primitive_ids, tx_visible),
                            self._env_edge_rows(primitive_ids, rx_visible),
                        )
                        if env_wd:
                            wd_devices = object_edges
                        if env_rwd:
                            rwd_devices = object_edges

                    if env_vd or env_rvd:
                        object_vertices = self._build_env_vertex_rows(
                            primitive_ids, tx_visible, rx_visible, geom.active_vertices
                        )
                        if env_vd:
                            vd_candidates = object_vertices
                        if env_rvd:
                            rvd_devices = object_vertices

                    if env_rr or env_rwd or env_rvd:
                        need_rx_hits = (env_rr and tf.reduce_any(tx_visible)) or (
                            env_rwd and tf.size(rwd_devices[0]) > 0
                        ) or (
                            env_rvd and tf.size(rvd_devices[0]) > 0
                        )
                        if registered_mask is None:
                            registered_mask = self._registered_primitive_mask()
                        env_first_hits = self._environment_first_hits(
                            los_prim, targets, need_rx_hits, registered_mask
                        )
                        if env_rr:
                            candidates = self._add_env_object_rr_candidates(
                                candidates, *env_first_hits, primitive_ids, tx_visible, rx_visible
                            )
                if env_dd:
                    second = tf.boolean_mask(second_order, selected)
                    dd_primitive_ids = tf.boolean_mask(selected_ids, second)
                    dd_tx_visible = tf.boolean_mask(env_tx, second, axis=1)
                    dd_rx_visible = tf.boolean_mask(env_rx, second, axis=1)
                    dd_tx_edges = self._env_edge_rows(dd_primitive_ids, dd_tx_visible)
                    dd_rx_edges = self._env_edge_rows(dd_primitive_ids, dd_rx_visible)
                    if self.double_diffraction.is_vertex_vertex or self.double_diffraction.is_eee:
                        dd_los_prim = tf.boolean_mask(
                            dd_primitive_ids, tf.reduce_any(dd_tx_visible, axis=0)
                        )
                        dd_los_prim_t = tf.boolean_mask(
                            dd_primitive_ids, tf.reduce_any(dd_rx_visible, axis=0)
                        )

        else:
            raise ValueError(f"Unknown method '{method}'")

        ##############################################
        # LoS and Specular paths
        ##############################################
        mirrored_vertices, tri_p0, normals = self.solver_reflection.get_images(sources, candidates)

        spec_paths, spec_paths_tmp = self.solver_reflection.tracing(sources,\
            targets, los, reflection, candidates, mirrored_vertices, tri_p0, normals, transmission)

        spec_paths.my_types = Paths.SPECULAR * tf.ones_like(spec_paths.objects, tf.int32)

        spec_paths.true_objects = spec_paths.objects

        # [num_targets, num_sources]
        # A transmitted direct path is not an unobstructed LOS link.
        if los and tf.shape(spec_paths.mask)[2] > 0:
            direct = tf.reduce_all(spec_paths.objects == -1, axis=0) & spec_paths.mask
            crossed = tf.scatter_nd(
                spec_paths_tmp.transmission_path_indices,
                tf.ones_like(spec_paths_tmp.transmission_primitives, dtype=tf.int32),
                tf.shape(spec_paths.mask),
            ) > 0
            self.is_los = tf.reduce_any(direct & ~crossed, axis=-1, keepdims=True)
        else:
            self.is_los = tf.zeros(
                [tf.shape(targets)[0], tf.shape(sources)[0], 1], dtype=tf.bool
            )

        # if los and tf.shape(spec_paths.mask)[2] > 0:
        #     self.is_los = spec_paths.mask[..., :1]
        # else:
        #     self.is_los = tf.zeros((targets.shape[0], sources.shape[0], 1), dtype=tf.bool)

        ############################################
        # Diffracted paths
        ############################################
        diff_paths, diff_paths_tmp = self.solver_wedge_diffraction.tracing(
            sources, targets, los_prim, diffraction, edge_diffraction,
            transmission=transmission, device_candidates=wd_devices,
            merge_global_candidates=method == "fibonacci",
        )

        diff_paths.my_types = Paths.DIFFRACTED * tf.ones_like(diff_paths.objects, tf.int32)

        objects_mask = diff_paths.objects == -1
        valid_wedge_idxs = tf.where(objects_mask, 0, diff_paths.objects)
        diff_paths.true_objects = tf.gather(self._wedges_objects[:, 0], valid_wedge_idxs, axis=0)

        ############################################
        # Vertex diffracted paths
        ############################################
        vertex_diff_paths = Paths(sources=sources, targets=targets, scene=self._scene,
                            types=Paths.VERTEX_DIFFRACTED)
        vertex_diff_paths_tmp = PathsTmpData(sources, targets, self._dtype)

        if vertex_diffraction and (method == "exhaustive" or vd_candidates is not None):
            vertex_diff_paths, vertex_diff_paths_tmp = self.vertex_diffraction.tracing(
                sources, targets, edge_diffraction=edge_diffraction,
                transmission=transmission, device_candidates=vd_candidates,
            )
            vertex_diff_paths_tmp.junction_transmission = transmission

            vertex_diff_paths.my_types = Paths.VERTEX_DIFFRACTED * tf.ones_like(vertex_diff_paths.objects, tf.int32)

            valid_vertex_idxs = tf.where(vertex_diff_paths.objects == -1, 0, vertex_diff_paths.objects)
            vertex_diff_paths.true_objects = tf.gather(self.vertex_diffraction.vertices_2_objects_sionna, valid_vertex_idxs)
            
        ############################################
        # Reflected + Diffracted paths
        ############################################

        mixed = []
        if refl_diff and diffraction and reflection and max_depth >= 2:
            # Legacy SBR candidates for paths with both interactions on the environment.
            mixed.extend((
                self._trace_environment_rwd(_candidates, sources, targets, registered_mask),
                self._trace_environment_rwd(_candidates_t, targets, sources, registered_mask,
                                            reverse=True),
            ))
        if rwd_devices is not None:
            env_tx_hits, env_rx_hits = env_first_hits
            tx_edges, rx_edges = rwd_devices
            mixed.extend((
                self._trace_reflection_diffraction(
                    sources, targets, env_tx_hits, rx_edges, edge_diffraction,
                    registered_mask=registered_mask,
                ),
                self._trace_reflection_diffraction(
                    targets, sources, env_rx_hits, tx_edges, edge_diffraction,
                    reverse=True, registered_mask=registered_mask,
                ),
            ))

        mixed = [(p, tmp) for p, tmp in mixed if p.objects.shape[3] > 0]
        if mixed:
            mixed_paths = merge_paths([p for p, _ in mixed])
            mixed_paths_tmp = mixed[0][1]
            for _, tmp in mixed[1:]:
                mixed_paths_tmp = merge_paths_tmp([mixed_paths_tmp, tmp])
        else:
            mixed_paths = Paths(sources, targets, self._scene, types=Paths.REFL_DIFF)
            mixed_paths_tmp = PathsTmpData(sources, targets, self._dtype)

        ##################
        # Reflection + Vertex_diffraction
        ##################
        vd_refl_paths = Paths(sources=sources, targets=targets, scene=self._scene, types=Paths.VD_REFL)
        vd_refl_paths_tmp = PathsTmpData(sources, targets, self._dtype)
        
        if rvd_devices is not None:
            env_tx_hits, env_rx_hits = env_first_hits
            tx_vertices, rx_vertices = rvd_devices
            vd_refl_paths, vd_refl_paths_tmp = self._trace_reflection_diffraction(
                sources, targets, env_tx_hits, rx_vertices, edge_diffraction, vertex=True
            )
            reverse_paths, reverse_tmp = self._trace_reflection_diffraction(
                targets, sources, env_rx_hits, tx_vertices, edge_diffraction,
                reverse=True, vertex=True,
            )
            vd_refl_paths = merge_paths([vd_refl_paths, reverse_paths])
            vd_refl_paths_tmp = merge_paths_tmp([vd_refl_paths_tmp, reverse_tmp])
            vd_refl_paths_tmp.env_object_rvd = True
            vd_refl_paths_tmp.edge_diffraction = edge_diffraction
        elif refl_diff and vertex_diffraction and reflection and max_depth >= 2 and self.vertex_diffraction.is_refl_vd:
            tx_images = (mirrored_vertices, tri_p0, normals)
            if _candidates is not candidates and _candidates.shape[0] >= 2:
                tx_images = self.solver_reflection.get_images(sources, _candidates)
            if _candidates_t is not None and _candidates_t.shape[0] >= 2:
                rx_images = self.solver_reflection.get_images(targets, _candidates_t)
            ## a) TX - refl - VD - RX
            if _candidates.shape[0] >= 2:
                vd_refl_paths, vd_refl_paths_tmp = self._comb_vd_refl(
                    tx_images[0], tx_images[1], _candidates, tx_images[2], targets, sources
                )
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
            if _candidates_t is not None and _candidates_t.shape[0] >= 2:
                vd_refl_paths_t, vd_refl_paths_tmp_t = self._comb_vd_refl(
                    rx_images[0], rx_images[1], _candidates_t, rx_images[2], sources, targets
                )
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

        junction_enabled = None
        if transmission and double_diffraction and tf.size(self._primitives_2_objects) > 0:
            array_size = tf.reduce_max(self._primitives_2_objects) + 1
            _, junction_enabled = self._build_slab_properties_tensors(array_size)

        use_env_dd = dd_primitive_ids is not None
        if double_diffraction and (use_env_dd or (method == "exhaustive" and los_prim is not None)):
            dd = self.double_diffraction
            use_link_candidates = use_env_dd or method == "exhaustive"
            trace_sv = dd.is_only_coplanar and dd.is_wedge_wedge_sv
            trace_ev_ve = dd.is_wedge_vertex or dd.is_vertex_wedge
            need_legacy_pairs = dd.is_vertex_vertex or (
                not use_link_candidates and (trace_ev_ve or trace_sv)
            )
            need_global_pairs = need_legacy_pairs or (
                dd.is_wedge_wedge and not use_env_dd
            )
            if need_global_pairs:
                wedges_1 = self._wedges_from_primitives(dd_los_prim if use_env_dd else los_prim, True)
                wedges_2 = self._wedges_from_primitives(dd_los_prim_t if use_env_dd else los_prim_t, True)

                if junction_enabled is not None and not use_env_dd:
                    all_wedges = tf.range(tf.shape(self._wedges_objects)[0], dtype=tf.int32)
                    junctions = tf.boolean_mask(
                        all_wedges, self.double_diffraction._junction_candidates(all_wedges, junction_enabled)
                    )
                    wedges_1, _ = tf.unique(tf.concat([wedges_1, junctions], axis=0))
                    wedges_2, _ = tf.unique(tf.concat([wedges_2, junctions], axis=0))
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

                pair_is_coplanar = self.double_diffraction.get_coplanar(dd_pairs)

                if self.double_diffraction.is_only_coplanar:
                    dd_pairs = tf.boolean_mask(dd_pairs, pair_is_coplanar)
                    self.double_diffraction.dd_pairs_coplanar = dd_pairs
                    pair_is_coplanar = tf.ones_like(dd_pairs[:, 0], dtype=tf.bool)

                ee_pairs = dd_pairs
                ee_pair_is_coplanar = pair_is_coplanar

                # Ordinary EE uses the final visibility check on the solved path.
                if need_legacy_pairs:
                    dd_pairs = dd.check_preliminary_visibility(
                        dd_pairs, junction_enabled=junction_enabled
                    )

                if (not use_link_candidates and dd.is_wedge_vertex) or dd.is_vertex_vertex:
                    _v_idxs = tf.gather(self.obj_geom.wedge_2_vertices, dd_pairs[:, 1])
                    v_idxs2 = tf.reshape(_v_idxs, -1)
                    wv_pairs = tf.stack((tf.repeat(dd_pairs[:, 0], 2), v_idxs2), axis=1)
                    wv_pairs, _ = tf.raw_ops.UniqueV2(x=wv_pairs, axis=[0])
                    active_vertices_dd_idxs = tf.cast(tf.where(self.obj_geom.active_vertices_dd)[:, 0], tf.int32)
                    wv_pairs = tf.boolean_mask(wv_pairs, tf.reduce_any(
                        tf.equal(wv_pairs[:, 1][..., None], active_vertices_dd_idxs[None, ...]), axis=1), axis=0)
                    
                
                if (not use_link_candidates and dd.is_vertex_wedge) or dd.is_vertex_vertex:
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

            if dd.is_wedge_wedge:
                if use_env_dd:
                    ee_rows = self._build_env_ee_rows(
                        sources, targets, dd_tx_edges, dd_rx_edges, junction_enabled
                    )
                    ee_is_coplanar = dd.get_coplanar(ee_rows[:, 2:4])
                    if dd.is_only_coplanar:
                        ee_rows = tf.boolean_mask(ee_rows, ee_is_coplanar)
                        ee_is_coplanar = tf.ones_like(ee_rows[:, 0], dtype=tf.bool)
                else:
                    # candidate_rows: [N, 3], columns [tx, rx, index in ee_pairs].
                    candidate_rows = dd.discard_obstructing_flat(
                        ee_pairs, sources, targets, junction_enabled=junction_enabled
                    )
                    pair_ids = candidate_rows[:, 2]
                    ee_rows = tf.concat([
                        candidate_rows[:, :2], tf.gather(ee_pairs, pair_ids)
                    ], axis=1)
                    ee_is_coplanar = tf.gather(ee_pair_is_coplanar, pair_ids)

                # ee_rows: [N, 4], columns [tx, rx, first wedge, second wedge].
                if tf.shape(ee_rows)[0] > 0:
                    ee_tx = tf.gather(sources, ee_rows[:, 0])  # [N, 3]
                    ee_rx = tf.gather(targets, ee_rows[:, 1])  # [N, 3]
                    ee_wedges = ee_rows[:, 2:4]               # [N, 2]
                    n_ee = tf.shape(ee_rows)[0]

                    # Solve coplanar candidates first.
                    cop_rows = tf.cast(tf.where(ee_is_coplanar)[:, 0], tf.int32)
                    if tf.size(cop_rows) > 0:
                        cop_points, cop_valid = (
                            self.double_diffraction.compute_diffraction_points_coplanar_flat(
                                tf.gather(ee_tx, cop_rows),
                                tf.gather(ee_rx, cop_rows),
                                tf.gather(ee_wedges, cop_rows),
                            )
                        )
                        flat_points = tf.scatter_nd(cop_rows[:, None], cop_points, [n_ee, 2, 3])
                        flat_valid = tf.scatter_nd(cop_rows[:, None], cop_valid, [n_ee])
                    else:
                        flat_points = tf.zeros([n_ee, 2, 3], dtype=self._rdtype)
                        flat_valid = tf.zeros([n_ee], dtype=tf.bool)

                    if not self.double_diffraction.is_only_coplanar:
                        # Includes skew pairs and failed analytical candidates.
                        num_rows = tf.cast(tf.where(~flat_valid)[:, 0], tf.int32)
                        if tf.size(num_rows) > 0:
                            num_points, num_valid = (
                                self.double_diffraction.compute_diffraction_points_noncoplanar_flat(
                                    tf.gather(ee_tx, num_rows),
                                    tf.gather(ee_rx, num_rows),
                                    tf.gather(ee_wedges, num_rows),
                                )
                            )
                            flat_points = tf.tensor_scatter_nd_update(
                                flat_points, num_rows[:, None], num_points
                            )
                            flat_valid = tf.tensor_scatter_nd_update(
                                flat_valid, num_rows[:, None], num_valid
                            )

                    if tf.reduce_any(flat_valid):
                        # Keep solved EE rows flat through all geometric and ray checks.
                        solved_rows, solved_points = dd.filter_ee_paths_flat(
                            sources, targets, tf.boolean_mask(ee_rows, flat_valid),
                            tf.boolean_mask(flat_points, flat_valid),
                            junction_enabled=junction_enabled,
                        )
                        dd_paths, dd_paths_tmp = dd.pack_ee_paths_flat(
                            sources, targets, solved_rows, solved_points
                        )

            if use_link_candidates and (trace_ev_ve or trace_sv):
                if self.obj_geom is None:
                    raise ValueError("EV, VE and SV tracing require ObjectsGeometry.")
                if use_env_dd:
                    if trace_ev_ve:
                        dd_tx_vertices, dd_rx_vertices = self._build_env_vertex_rows(
                            dd_primitive_ids, dd_tx_visible, dd_rx_visible, geom.active_vertices_dd
                        )
                else:
                    n_tx, n_rx = tf.shape(sources)[0], tf.shape(targets)[0]
                    dd_tx_edges = self._all_device_items(
                        n_tx, self._wedges_from_primitives(los_prim, True)
                    )
                    dd_rx_edges = self._all_device_items(
                        n_rx, self._wedges_from_primitives(los_prim_t, True)
                    )
                    if trace_ev_ve:
                        vertex_ids = tf.cast(tf.where(self.obj_geom.active_vertices_dd)[:, 0], tf.int32)
                        dd_tx_vertices = self._all_device_items(n_tx, vertex_ids)
                        dd_rx_vertices = self._all_device_items(n_rx, vertex_ids)

            # Shared-vertex candidates stay flat through the excess-length and ray checks.
            if trace_sv and use_link_candidates:
                sv_rows = self._build_sv_candidate_rows(sources, targets, dd_tx_edges, dd_rx_edges)
                if method == "fibonacci":
                    sv_rows = tf.boolean_mask(
                        sv_rows, tf.gather(geom.active_vertices_dd, sv_rows[:, 4])
                    )
                dd_paths_sv, dd_paths_sv_tmp = dd.trace_sv_flat(sources, targets, sv_rows)
                if tf.reduce_any(dd_paths_sv.mask):
                    dd_paths = merge_paths([dd_paths, dd_paths_sv])
                    dd_paths_tmp = merge_paths_tmp([dd_paths_tmp, dd_paths_sv_tmp])

            # Legacy SV tracing for the remaining candidate-selection methods.
            if trace_sv and not use_link_candidates:
                dd_pair_idxs, active, valid_vertex_idxs = dd.shared_vertex(sources, targets, dd_pairs)
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
            
            # EV/VE candidates remain flat until the final Paths assembly.
            if use_link_candidates:
                for vertex_depth, enabled in ((1, dd.is_wedge_vertex), (0, dd.is_vertex_wedge)):
                    if not enabled:
                        continue
                    if vertex_depth == 1:
                        mixed_rows = self._build_mixed_candidate_rows(
                            sources, targets, dd_tx_edges, dd_rx_vertices,
                            vertex_first=False, junction_enabled=junction_enabled,
                        )
                    else:
                        mixed_rows = self._build_mixed_candidate_rows(
                            sources, targets, dd_tx_vertices, dd_rx_edges,
                            vertex_first=True, junction_enabled=junction_enabled,
                        )
                    mixed, mixed_tmp = dd.trace_mixed_flat(
                        sources, targets, mixed_rows, vertex_depth,
                        junction_enabled=junction_enabled,
                    )
                    if tf.reduce_any(mixed.mask):
                        dd_paths = merge_paths([dd_paths, mixed])
                        dd_paths_tmp = merge_paths_tmp([dd_paths_tmp, mixed_tmp])

            ######## 2) wedge_vertex
            if self.double_diffraction.is_wedge_vertex and not use_link_candidates:
                dd_wv_idxs, active = dd.discard_obstructing_dd_wedge_vertex(
                    wv_pairs, sources, targets, junction_enabled=junction_enabled
                )

                if self.double_diffraction.is_only_in_nlos:
                    # set to False all paths in LOS
                    active = tf.logical_and(active, ~tf.transpose(self.is_los, [1, 0, 2]))
                
                if tf.reduce_any(active):
                    # [2, num_sources, num_targets, num_pairs, 3] and [num_sources, num_targets, num_pairs]
                    diff_points2, valid2 = self.double_diffraction.compute_diffraction_points_wedge_vertex(sources, targets, dd_wv_idxs)
                    active = tf.logical_and(active, valid2)

                    ## discard if the diffraction points coincide
                    diff_points2, valid2 = self.double_diffraction.omit_identical_points(diff_points2)
                    active = tf.logical_and(active, valid2)

                    # Discard obstructed paths
                    valid2 = self.double_diffraction.check_visibility_mixed(sources, targets, diff_points2, dd_wv_idxs[..., 0], dd_wv_idxs[..., 1], junction_enabled=junction_enabled)
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
                        dd_paths_wv.vertex_wedge_mask = self.double_diffraction._vertex_wedge_mask(dd_paths_wv, vertex_depth=1, junction_enabled=junction_enabled)
                        if use_link_candidates:
                            dd_paths_wv.mask &= tf.reduce_any(dd_paths_wv.vertex_wedge_mask, axis=-1)
                            dd_paths_wv.objects = tf.where(
                                dd_paths_wv.mask[None, ...], dd_paths_wv.objects, -1
                            )
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
            if self.double_diffraction.is_vertex_wedge and not use_link_candidates:
                dd_vw_idxs, active = dd.discard_obstructing_dd_vertex_wedge(
                    vw_pairs, sources, targets, junction_enabled=junction_enabled
                )

                if self.double_diffraction.is_only_in_nlos:
                    # set to False all paths in LOS
                    active = tf.logical_and(active, ~tf.transpose(self.is_los, [1, 0, 2]))
                
                if tf.reduce_any(active):
                    # [2, num_sources, num_targets, num_pairs, 3] and [num_sources, num_targets, num_pairs]
                    diff_points3, valid3 = self.double_diffraction.compute_diffraction_points_vertex_wedge(sources, targets, dd_vw_idxs)
                    active = tf.logical_and(active, valid3)

                    ## discard if the diffraction points coincide
                    diff_points3, valid3 = self.double_diffraction.omit_identical_points(diff_points3)
                    active = tf.logical_and(active, valid3)

                    # Discard obstructed paths
                    # Reverse interaction order and swap link axes for the EV visibility check.
                    reverse_points = tf.transpose(diff_points3[::-1], [0, 2, 1, 3, 4])
                    reverse_pairs = tf.transpose(dd_vw_idxs, [1, 0, 2, 3])
                    valid3 = self.double_diffraction.check_visibility_mixed(
                        targets, sources, reverse_points, reverse_pairs[..., 1], reverse_pairs[..., 0],
                        junction_enabled=junction_enabled,
                    )
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
                        dd_paths_vw.vertex_wedge_mask = self.double_diffraction._vertex_wedge_mask(dd_paths_vw, vertex_depth=0, junction_enabled=junction_enabled)
                        if use_link_candidates:
                            dd_paths_vw.mask &= tf.reduce_any(dd_paths_vw.vertex_wedge_mask, axis=-1)
                            dd_paths_vw.objects = tf.where(
                                dd_paths_vw.mask[None, ...], dd_paths_vw.objects, -1
                            )

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
        # Tripple diffracted paths
        ############################################
        ddd_paths = Paths(sources=sources, targets=targets, scene=self._scene, types=Paths.TRIPLE_DIFF)
        ddd_paths_tmp = PathsTmpData(sources, targets, self._dtype)

        eee_prims = dd_los_prim if use_env_dd else los_prim
        eee_prims_t = dd_los_prim_t if use_env_dd else los_prim_t
        if (double_diffraction and self.double_diffraction.is_eee and has_objects
                and eee_prims is not None and eee_prims_t is not None):

            wedges_1 = self._wedges_from_primitives(eee_prims, False)
            wedges_2 = self._wedges_from_primitives(eee_prims_t, False)

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
        if scattering and scat_samples_per_source > 0 and tf.shape(candidates_scat)[0] > 0:

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
        scat_paths_tmp.num_samples = scat_samples_per_source
        scat_paths_tmp.scat_keep_prob = tf.cast(scat_keep_prob, self._rdtype)
        vertex_diff_paths_tmp.num_samples = num_samples
        vertex_diff_paths_tmp.scat_keep_prob = tf.cast(scat_keep_prob, self._rdtype)
        vd_refl_paths_tmp.num_samples = num_samples
        vd_refl_paths_tmp.scat_keep_prob = tf.cast(scat_keep_prob, self._rdtype)
        dd_paths_tmp.junction_transmission = transmission
        dd_paths_tmp.num_samples = num_samples
        dd_paths_tmp.scat_keep_prob = tf.cast(scat_keep_prob, self._rdtype)

        return spec_paths, diff_paths, mixed_paths, mixed_paths_tmp, scat_paths, spec_paths_tmp,\
            diff_paths_tmp, scat_paths_tmp, vertex_diff_paths, vertex_diff_paths_tmp,\
            vd_refl_paths, vd_refl_paths_tmp, dd_paths, dd_paths_tmp, ddd_paths, ddd_paths_tmp

    def compute_fields(self, spec_paths, diff_paths, mixed_paths, mixed_paths_tmp, 
        scat_paths, spec_paths_tmp,
        diff_paths_tmp, scat_paths_tmp, vertex_diff_paths, 
        vertex_diff_paths_tmp, vd_refl_paths, vd_refl_paths_tmp,
        dd_paths, dd_paths_tmp, ddd_paths, ddd_paths_tmp, transmission, scat_random_phases, testing):
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
        self.rot_motion_angular_velocity = object_properties[9]

        thickness, transmission_enabled = self._build_slab_properties_tensors(tf.shape(velocity)[0])

        # Rebuild inside each field evaluation to preserve material gradients.
        self._object_thickness = thickness
        self._object_transmission_enabled = transmission_enabled

        ##############################################
        # LoS and Specular paths
        ##############################################

        if spec_paths.objects.shape[3] > 0:
            # Compute the EM transition matrices and Doppler shifts
            spec_mat_t = self.solver_reflection._spec_transition_matrices(
                etas, scattering_coefficient, spec_paths, spec_paths_tmp, False,
                thickness, transmission_enabled
            )

            spec_mat_t = self.solver_reflection._apply_direct_transmission(
                spec_paths, spec_paths_tmp, spec_mat_t, etas, thickness
            )
            
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
            diff_mat_t = self.solver_wedge_diffraction.compute_fields(
                etas, scattering_coefficient, diff_paths, diff_paths_tmp,
                spec_paths=spec_paths, spec_paths_tmp=spec_paths_tmp,
            )

            diff_mat_t = self.solver_reflection._apply_diffraction_transmission(
                diff_paths, diff_paths_tmp, diff_mat_t, etas, thickness
            )            
            
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
        # Vertex-diffracted paths
        ############################################
        #    
        if vertex_diff_paths.objects.shape[3] > 0:
            vertex_diff_mat_t =\
                self.vertex_diffraction.compute_fields2(etas,
                            scattering_coefficient, vertex_diff_paths, vertex_diff_paths_tmp,
                            diff_paths=diff_paths, diff_paths_tmp=diff_paths_tmp)

            vertex_diff_mat_t = self.solver_reflection._apply_diffraction_transmission(
                vertex_diff_paths, vertex_diff_paths_tmp, vertex_diff_mat_t, etas, thickness
            )                            
            
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
            if getattr(vd_refl_paths_tmp, "env_object_rvd", False):
                vd_refl_paths_mat_t = self.solver_wedge_diffraction._compute_refl_diff(
                    etas, scattering_coefficient, vd_refl_paths, vd_refl_paths_tmp,
                    vertex_solver=self.vertex_diffraction,
                    edge_diffraction=vd_refl_paths_tmp.edge_diffraction,
                )
            else:
                vd_refl_paths_mat_t = self.vertex_diffraction.compute_refl_vd(
                    etas, scattering_coefficient, vd_refl_paths, vd_refl_paths_tmp
                )
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
            scat_mat_t = self.solver_reflection._spec_transition_matrices(
                etas, scattering_coefficient, scat_paths, scat_paths_tmp, True,
                thickness, transmission_enabled,
            )
            
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

    def _intersect_env_candidates(self, tx_rows, rx_rows, num_sources, num_targets):
        """Intersect [device, item] rows per link.

        Returns item IDs [K] and their candidate mask [num_targets, num_sources, K].
        Both input row lists must be deduplicated.
        """
        item_ids, slots = tf.unique(tf.concat([tx_rows[:, 1], rx_rows[:, 1]], axis=0))
        n_items = tf.size(item_ids)
        n_tx_rows = tf.shape(tx_rows)[0]

        def device_mask(rows, item_slots, num_devices):
            indices = tf.stack([rows[:, 0], item_slots], axis=1)
            return tf.scatter_nd(
                indices, tf.ones_like(item_slots, dtype=tf.bool),
                tf.stack([num_devices, n_items]),
            )

        tx_mask = device_mask(tx_rows, slots[:n_tx_rows], num_sources)
        rx_mask = device_mask(rx_rows, slots[n_tx_rows:], num_targets)
        common = tf.reduce_any(tx_mask, axis=0) & tf.reduce_any(rx_mask, axis=0)
        cols = tf.where(common)[:, 0]
        tx_mask = tf.gather(tx_mask, cols, axis=1)
        rx_mask = tf.gather(rx_mask, cols, axis=1)
        return tf.gather(item_ids, cols), rx_mask[:, None, :] & tx_mask[None, :, :]

    def _build_env_vertex_candidates(self, primitive_ids, tx_visible, rx_visible):
        """Find per-link VD candidates from visible triangle vertices."""
        if self.obj_geom is None:
            raise ValueError("env_driven vertex diffraction requires ObjectsGeometry.")
        tx_rows, rx_rows = self._build_env_vertex_rows(
            primitive_ids, tx_visible, rx_visible, self.obj_geom.active_vertices
        )
        return self._intersect_env_candidates(
            tx_rows, rx_rows, tf.shape(tx_visible)[0], tf.shape(rx_visible)[0]
        )

    def _build_env_vertex_rows(self, primitive_ids, tx_visible, rx_visible, active_vertices):
        """Return separate [device, vertex] lists before any per-link intersection."""
        geom = self.obj_geom
        if geom is None:
            raise ValueError("env_driven vertex diffraction requires ObjectsGeometry.")

        # Map global mesh vertex IDs to ObjectsGeometry vertex rows.
        global_ids = tf.cast(geom._vertex_indices, tf.int32)
        global_to_vd = tf.tensor_scatter_nd_update(
            tf.fill([tf.shape(self._vertices)[0]], -1),
            global_ids[:, None],
            tf.range(tf.size(global_ids), dtype=tf.int32),
        )
        face_vertices = tf.gather(global_to_vd, tf.gather(self._faces, primitive_ids))

        def vertex_rows(visible):
            hits = tf.cast(tf.where(visible), tf.int32)
            vertices = tf.reshape(tf.gather(face_vertices, hits[:, 1]), [-1])
            rows = tf.stack([tf.repeat(hits[:, 0], 3), vertices], axis=1)
            rows = tf.boolean_mask(rows, rows[:, 1] >= 0)
            rows = tf.boolean_mask(rows, tf.gather(active_vertices, rows[:, 1]))
            rows, _ = tf.raw_ops.UniqueV2(x=rows, axis=[0])
            return rows

        return vertex_rows(tx_visible), vertex_rows(rx_visible)

    def _check_env_first_order_visibility(self, sources, targets, indices, points, case):
        """Trace only actual first-order candidates; keep the original slots."""
        rows = tf.cast(tf.where(indices >= 0), tf.int32)
        if tf.shape(rows)[0] == 0:
            return indices

        interactions = tf.gather_nd(points, rows)
        valid = tf.ones([tf.shape(rows)[0]], dtype=tf.bool)
        if case in ("both", "tx"):
            tx = tf.gather(sources, rows[:, 1])
            directions, distances = normalize(interactions - tx)
            valid &= ~self._test_obstruction(tx, directions, distances)
        if case in ("both", "rx"):
            rx = tf.gather(targets, rows[:, 0])
            directions, distances = normalize(interactions - rx)
            valid &= ~self._test_obstruction(rx, directions, distances)

        rejected = tf.boolean_mask(rows, ~valid)
        return tf.tensor_scatter_nd_update(
            indices, rejected, tf.fill([tf.shape(rejected)[0]], tf.cast(-1, indices.dtype))
        )

    @staticmethod
    def _all_device_items(num_devices, items):
        """List all [device, item] rows for exhaustive candidate selection."""
        items = tf.cast(items, tf.int32)
        return tf.stack([
            tf.repeat(tf.range(num_devices), tf.size(items)),
            tf.tile(items, [num_devices]),
        ], axis=1)

    def _pack_link_pair_rows(self, rows, sources, targets):
        """Adapt flat [tx, rx, item1, item2] rows to the existing solvers."""
        num_sources, num_targets = tf.shape(sources)[0], tf.shape(targets)[0]
        if tf.shape(rows)[0] == 0:
            return (tf.fill([num_sources, num_targets, 0, 2], -1),
                    tf.zeros([num_sources, num_targets, 0], dtype=tf.bool),
                    (tf.zeros([0, 3], dtype=tf.int32), tf.zeros([0, 3], dtype=tf.int32)))
        row_ids = tf.range(tf.shape(rows)[0], dtype=tf.int32)
        return self.double_diffraction.pack_ee_candidates(
            rows[:, 2:4], tf.concat([rows[:, :2], row_ids[:, None]], axis=1),
            num_sources, num_targets,
        )

    @staticmethod
    def _pair_device_items(tx_rows, rx_rows, n_tx, n_rx):
        """Form [tx, rx, item1, item2] products for each link."""
        def prepare(rows, num_devices):
            rows = tf.gather(rows, tf.argsort(rows[:, 0], stable=True))
            counts = tf.math.unsorted_segment_sum(
                tf.ones_like(rows[:, 0]), rows[:, 0], num_devices
            )
            return rows[:, 1], counts, tf.cumsum(counts, exclusive=True)

        items_tx, counts_tx, offsets_tx = prepare(tx_rows, n_tx)
        items_rx, counts_rx, offsets_rx = prepare(rx_rows, n_rx)
        counts = tf.reshape(counts_tx[:, None] * counts_rx[None, :], [-1])
        links = tf.repeat(tf.range(n_tx * n_rx), counts)
        local = tf.range(tf.reduce_sum(counts)) - tf.repeat(
            tf.cumsum(counts, exclusive=True), counts
        )
        tx, rx = links // tf.maximum(n_rx, 1), links % tf.maximum(n_rx, 1)
        rx_count = tf.gather(counts_rx, rx)
        item1 = tf.gather(items_tx, tf.gather(offsets_tx, tx) + local // rx_count)
        item2 = tf.gather(items_rx, tf.gather(offsets_rx, rx) + local % rx_count)
        return tf.stack([tx, rx, item1, item2], axis=1)

    def _build_mixed_candidate_rows(
        self, sources, targets, tx_rows, rx_rows, vertex_first, junction_enabled=None
    ):
        """Build EV/VE rows and check the exterior sector at the ordinary edge."""
        rows = self._pair_device_items(
            tx_rows, rx_rows, tf.shape(sources)[0], tf.shape(targets)[0]
        )
        if vertex_first:
            wedges, vertices = rows[:, 3], rows[:, 2]
            outer = tf.gather(targets, rows[:, 1])
        else:
            wedges, vertices = rows[:, 2], rows[:, 3]
            outer = tf.gather(sources, rows[:, 0])

        origins = tf.gather(self._wedges_origin, wedges)
        normals = tf.gather(self._wedges_normals, wedges)
        d_outer = dot((outer - origins)[:, None, :], normals)
        d_inner = dot((tf.gather(self.obj_geom._vertices, vertices) - origins)[:, None, :], normals)
        eps = tf.cast(self.EPSILON, d_outer.dtype)
        allowed = tf.reduce_any(d_outer > eps, axis=1) & tf.reduce_any(d_inner > -eps, axis=1)

        if junction_enabled is not None:
            wd = self.solver_wedge_diffraction
            crossed = wd._is_flat_shared_wedge(wedges) & wd._opposite_sides(
                d_outer[:, 0], d_inner[:, 0]
            )
            eligible = self.double_diffraction._junction_candidates(wedges, junction_enabled)
            allowed = tf.where(crossed, eligible, allowed)
        allowed |= tf.gather(self._is_edge, wedges)
        return tf.boolean_mask(rows, allowed)

    def _build_sv_candidate_rows(self, sources, targets, tx_edges, rx_edges):
        """Join edge candidates by their shared vertex, yielding [tx, rx, w1, w2, vertex]."""
        # Invert the actual VD incidence table; its rows use ObjectsGeometry IDs.
        vertex_to_wedges = self.obj_geom.vertex_2_wedges
        topology = tf.stack([
            tf.cast(vertex_to_wedges.values, tf.int32),
            tf.cast(vertex_to_wedges.value_rowids(), tf.int32),
        ], axis=1)
        topology = tf.boolean_mask(topology, topology[:, 0] >= 0)
        if tf.shape(topology)[0] == 0:
            return tf.zeros([0, 5], dtype=tf.int32)
        topology = tf.gather(topology, tf.argsort(topology[:, 0], stable=True))

        def expand_endpoints(edges, devices):
            device = tf.gather(devices, edges[:, 0])
            wedges = edges[:, 1]
            origins = tf.gather(self._wedges_origin, wedges)
            normals = tf.gather(self._wedges_normals, wedges)
            signed = dot((device - origins)[:, None, :], normals)
            outside = tf.reduce_any(signed > tf.cast(self.EPSILON, signed.dtype), axis=1)
            edges = tf.boolean_mask(edges, outside)

            lo = tf.searchsorted(topology[:, 0], edges[:, 1], side="left")
            hi = tf.searchsorted(topology[:, 0], edges[:, 1], side="right")
            incident = tf.gather(topology[:, 1], tf.ragged.range(lo, hi).flat_values)
            rows = tf.concat([tf.repeat(edges, hi - lo, axis=0), incident[:, None]], axis=1)
            rows, _ = tf.raw_ops.UniqueV2(x=rows, axis=[0])
            return rows

        tx = expand_endpoints(tx_edges, sources)
        rx = expand_endpoints(rx_edges, targets)
        if tf.shape(tx)[0] == 0 or tf.shape(rx)[0] == 0:
            return tf.zeros([0, 5], dtype=tf.int32)

        # Locate only RX edges sharing the vertex of each TX edge.
        rx = tf.gather(rx, tf.argsort(rx[:, 2], stable=True))
        lo = tf.searchsorted(rx[:, 2], tx[:, 2], side="left")
        hi = tf.searchsorted(rx[:, 2], tx[:, 2], side="right")
        left = tf.repeat(tf.range(tf.shape(tx)[0]), hi - lo)
        right = tf.ragged.range(lo, hi).flat_values
        tx = tf.gather(tx, left)
        rx = tf.gather(rx, right)
        rows = tf.stack([tx[:, 0], rx[:, 0], tx[:, 1], rx[:, 1], tx[:, 2]], axis=1)
        rows = tf.boolean_mask(rows, rows[:, 2] != rows[:, 3])

        # Preserve the SV sector checks for the other edge and the coplanar tolerance.
        wedges = rows[:, 2:4]
        origins = tf.gather(self._wedges_origin, wedges)
        normals = tf.gather(self._wedges_normals, wedges)
        delta = origins[:, 1] - origins[:, 0]
        d1 = dot(delta[:, None, :], normals[:, 0])
        d2 = dot(-delta[:, None, :], normals[:, 1])
        eps = tf.cast(self.EPSILON, d1.dtype)
        allowed = tf.reduce_any(d1 > -eps, axis=1) & tf.reduce_any(d2 > -eps, axis=1)
        return tf.boolean_mask(rows, allowed & self.double_diffraction.get_coplanar(wedges))

    def _build_env_ee_rows(self, sources, targets, tx_edges, rx_edges,
                           junction_enabled=None):
        """Build [N, 4] rows [tx, rx, w1, w2] from per-device edge lists."""
        num_sources = tf.shape(sources)[0]
        num_targets = tf.shape(targets)[0]

        def prepare(devices, edge_rows):
            # edge_rows: [M, 2], columns [device ID, wedge ID].
            device_ids, wedges = tf.unstack(edge_rows, num=2, axis=1)
            offsets = tf.gather(devices, device_ids) - tf.gather(self._wedges_origin, wedges)
            normals = tf.gather(self._wedges_normals, wedges)
            signed = dot(offsets[:, None, :], normals)  # [M, 2]
            allowed = tf.reduce_any(
                signed > tf.cast(self.EPSILON, signed.dtype), axis=1
            )
            allowed |= tf.gather(self._is_edge, wedges)
            if junction_enabled is not None:
                allowed |= self.double_diffraction._junction_candidates(wedges, junction_enabled)

            rows = tf.boolean_mask(edge_rows, allowed)
            rows = tf.gather(rows, tf.argsort(rows[:, 0], stable=True))
            counts = tf.math.unsorted_segment_sum(
                tf.ones_like(rows[:, 0]), rows[:, 0], tf.shape(devices)[0]
            )
            return rows[:, 1], counts, tf.cumsum(counts, exclusive=True)

        tx_wedges, tx_counts, tx_offsets = prepare(sources, tx_edges)
        rx_wedges, rx_counts, rx_offsets = prepare(targets, rx_edges)

        # Enumerate the edge products only for their corresponding links.
        counts = tf.reshape(tx_counts[:, None] * rx_counts[None, :], [-1])
        link = tf.repeat(tf.range(num_sources * num_targets), counts)
        local = tf.range(tf.reduce_sum(counts)) - tf.repeat(
            tf.cumsum(counts, exclusive=True), counts
        )
        tx = link // tf.maximum(num_targets, 1)
        rx = link % tf.maximum(num_targets, 1)
        n_rx = tf.gather(rx_counts, rx)
        w1 = tf.gather(tx_wedges, tf.gather(tx_offsets, tx) + local // n_rx)
        w2 = tf.gather(rx_wedges, tf.gather(rx_offsets, rx) + local % n_rx)

        rows = tf.stack([tx, rx, w1, w2], axis=1)
        return tf.boolean_mask(rows, w1 != w2)

    def _build_env_candidates(self, sources, targets, primitive_ids, transmission):
        """Return [num_sources, num_facets] and [num_targets, num_facets] visibility."""
        num_sources = tf.shape(sources)[0]
        visible = self._my_list_candidates_env_depth_1(
            tf.concat([sources, targets], axis=0), primitive_ids, transmission=transmission
        )
        return visible[:num_sources], visible[num_sources:]

    def _env_edge_rows(self, primitive_ids, visible):
        """Convert visible facets to unique [device, global wedge] rows."""
        hits = tf.cast(tf.where(visible), tf.int32)
        primitives = tf.gather(primitive_ids, hits[:, 1])
        wedges = tf.gather(self._primitives_2_wedges, primitives)
        rows = tf.stack([
            tf.repeat(hits[:, 0], tf.shape(wedges)[1]),
            tf.reshape(wedges, [-1]),
        ], axis=1)
        rows = tf.boolean_mask(rows, rows[:, 1] >= 0)
        rows, _ = tf.raw_ops.UniqueV2(x=rows, axis=[0])
        return rows

    @staticmethod
    def _add_env_reflection_candidates(candidates, facets):
        """Append [facet, -1, ...] columns, keeping the original SBR chains."""
        return SolverPaths._merge_reflection_candidates(candidates, facets[None, :], 1)

    @staticmethod
    def _merge_reflection_candidates(candidates, extra, depth):
        """Merge ordered primitive chains with shape [depth, num_candidates]."""
        if tf.shape(extra)[1] == 0:
            return candidates

        # A LOS-only SBR result can have shape [0, 1].
        old_depth = tf.shape(candidates)[0]
        merged_depth = tf.maximum(depth, old_depth)
        candidates = tf.pad(candidates, [[0, merged_depth - old_depth], [0, 0]],
                            constant_values=-1)
        padded = tf.pad(extra, [[0, merged_depth - depth], [0, 0]],
                        constant_values=-1)
        merged, _ = tf.raw_ops.UniqueV2(x=tf.concat([candidates, padded], axis=1), axis=[1])
        return merged

    def _registered_primitive_mask(self):
        """Mark all registered facets, including inactive object facets."""
        registered = tf.cast(self.obj_geom._primitive_idxs, tf.int32)
        return tf.scatter_nd(
            registered[:, None], tf.ones_like(registered), [tf.shape(self._primitives)[0]]
        ) > 0

    @staticmethod
    def _environment_facets(hits, registered_mask):
        if hits is None:
            return tf.zeros([0], tf.int32)
        hits = tf.boolean_mask(hits, hits >= 0)
        if registered_mask is not None:
            hits = tf.boolean_mask(hits, ~tf.gather(registered_mask, hits))
        return tf.unique(hits)[0]

    def _environment_first_hits(self, tx_hits, targets, need_rx_hits, registered_mask):
        """Share a single first-hit receiver launch between RR, R-WD and R-VD."""
        rx_hits = None
        if need_rx_hits:
            _, rx_hits, _, _, _ = self._list_candidates_fibonacci(
                1, targets, self.NUM_RX_FIBONACCI_SAMPLES * targets.shape[0],
                False, False, False, 1,
            )
        return (self._environment_facets(tx_hits, registered_mask),
                self._environment_facets(rx_hits, registered_mask))

    def _add_env_object_rr_candidates(self, candidates, env_tx, env_rx,
                                      object_facets, tx_visible, rx_visible):
        """Add environment/object reflection pairs in TX-to-RX order."""
        object_tx = tf.boolean_mask(object_facets, tf.reduce_any(tx_visible, axis=0))
        object_rx = tf.boolean_mask(object_facets, tf.reduce_any(rx_visible, axis=0))

        def pairs(first, second):
            return tf.stack([tf.repeat(first, tf.size(second)),
                             tf.tile(second, [tf.size(first)])], axis=0)

        # The image method tests the shared candidate list on every link.
        extra = tf.concat([pairs(env_tx, object_rx), pairs(object_tx, env_rx)], axis=1)
        return self._merge_reflection_candidates(candidates, extra, 2)

    def _my_list_candidates_env_depth_1(self, devices, primitive_idxs=None,
                                         transmission=False):
        """Return facet-center visibility with shape [num_devices, num_facets]."""
        if primitive_idxs is None:
            primitive_idxs = tf.range(tf.shape(self._primitives)[0], dtype=tf.int32)

        triangles = tf.gather(self._primitives, primitive_idxs)
        centers = tf.reduce_mean(triangles, axis=1)
        rays = centers[None, :, :] - devices[:, None, :]
        shape = tf.shape(rays)[:2]

        if tf.size(rays) == 0:
            return tf.zeros(shape, tf.bool)

        origins = tf.reshape(tf.broadcast_to(devices[:, None, :], tf.shape(rays)), [-1, 3])
        endpoints = tf.reshape(tf.broadcast_to(centers[None, :, :], tf.shape(rays)), [-1, 3])

        if transmission:
            # Only the blocked mask is needed for candidate selection.
            ray_ids = tf.range(tf.shape(origins)[0], dtype=tf.int32)[:, None]
            blocked, _ = self.solver_reflection._trace_transmission_segments(
                origins, endpoints, ray_ids
            )
        else:
            directions, distances = normalize(endpoints - origins)
            blocked = self._test_obstruction(origins, directions, distances)

        return tf.reshape(~blocked, shape)
    
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
            Keep every ``scat_sparsity_coef``th diffuse launch direction per source.

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

        samples_per_source_sparse : int
            Directions sampled per source after thinning, before discarding misses.
        """
        mask_t = dr.mask_t(self._mi_scalar_t)

        # Ensure that sample count can be distributed over the emitters
        num_sources = sources.shape[0]
        samples_per_source = (int(num_samples) + num_sources - 1) // num_sources
        num_samples = num_sources * samples_per_source
        samples_per_source_sparse = 0
        if scattering and scat_sparsity_coef < 1:
            raise ValueError("scat_sparsity_coef must be a positive integer.")

        # List of candidates
        candidates = []

        # Hit points
        hit_points = []

        # Is the scene empty?
        is_empty = dr.shape(self._shape_indices)[0] == 0

        # Only shoot when there are samples and surfaces to intersect.
        if not is_empty and samples_per_source > 0 and max_depth > 0:

            # Keep track of which paths are still active
            active = dr.full(mask_t, True, num_samples)

            # Initial ray: Arranged in a Fibonacci lattice on the unit
            # sphere.
            # [samples_per_source, 2] on the unit square.
            if samples_per_source == 1:
                lattice = tf.constant([[0.0, 0.5]], self._rdtype)
            else:
                lattice = fibonacci_lattice(samples_per_source, self._rdtype)
            sampled_d = tf.tile(lattice, [num_sources, 1])
            sampled_d = self._mi_point2_t(sampled_d)
            sampled_d = mi.warp.square_to_uniform_sphere(sampled_d)
            source_i = dr.arange(mi.UInt32, num_samples) // samples_per_source
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
            
            # Thin the sample axis within each source, keeping points and IDs aligned.
            candidates_scat = tf.reshape(
                candidates, [max_depth, num_sources, samples_per_source]
            )[:, :, ::scat_sparsity_coef]
            hit_points = tf.reshape(
                hit_points, [max_depth, num_sources, samples_per_source, 3]
            )[:, :, ::scat_sparsity_coef, :]
            samples_per_source_sparse = (samples_per_source + scat_sparsity_coef - 1) // scat_sparsity_coef

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
            candidates_scat = tf.fill([0, num_sources, 1], -1)

        # Remove duplicates
        if max_depth_ref > 0:
            candidates_ref, _ = tf.raw_ops.UniqueV2(
                x=candidates_ref,
                axis=[1]
            )
        else:
            # There may be many empty columns if every launched ray missed.
            # Keep only the explicit LOS column below.
            candidates_ref = candidates_ref[:, :0]

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

        return candidates_ref, los_primitives, candidates_scat, hit_points, samples_per_source_sparse


    ##################################################################
    # Methods used for computing the diffracted paths
    ##################################################################

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
                                [[2]], tf.cast(path_indices[None, :], scatter_indices.dtype))
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
                                [[3]], tf.cast(path_indices[None, :], scatter_indices.dtype))
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
    def _trace_environment_rwd(self, candidates, sources, targets, registered_mask, reverse=False):
        """Use the legacy first/second SBR hit sets, restricted to the environment."""
        faces = tf.zeros([0], tf.int32)
        edges = tf.zeros([0, 2], tf.int32)
        if candidates is not None and candidates.shape[0] >= 2:
            faces = self._environment_facets(candidates[0], registered_mask)
            wd_facets = self._environment_facets(candidates[1], registered_mask)
            wedges = self._wedges_from_primitives(wd_facets, False)
            if registered_mask is not None and tf.size(wedges) > 0:
                wedge_faces = tf.gather(self._wedges_2_primitives, wedges)
                is_registered = (wedge_faces >= 0) & tf.gather(
                    registered_mask, tf.maximum(wedge_faces, 0)
                )
                wedges = tf.boolean_mask(wedges, ~tf.reduce_any(is_registered, axis=1))
            edges = self._all_device_items(tf.shape(targets)[0], wedges)
        return self._trace_reflection_diffraction(sources, targets, faces, edges, False, reverse)

    def _trace_reflection_diffraction(self, sources, targets, reflecting_facets, target_items,
                                      edge_diffraction, reverse=False, registered_mask=None,
                                      vertex=False):
        """Trace reflection then diffraction and pack in physical TX-to-RX order.

        target_items : int32 tensor [num_items, 2]
            Each row is [target, wedge] or [target, vertex] in the working direction.
        """
        path_type = Paths.VD_REFL if vertex else Paths.REFL_DIFF
        tx, rx = (targets, sources) if reverse else (sources, targets)
        paths = Paths(tx, rx, self._scene, types=path_type)
        tmp = PathsTmpData(paths.sources, paths.targets, self._dtype)
        if tf.size(reflecting_facets) == 0 or tf.size(target_items) == 0:
            return paths, tmp
        if not vertex and not edge_diffraction:
            target_items = tf.boolean_mask(target_items, ~tf.gather(self._is_edge, target_items[:, 1]))
        if tf.size(target_items) == 0:
            return paths, tmp

        # Bound intermediate memory; columns are [source, target, facet, wedge_or_vertex].
        ns = tf.shape(sources)[0]
        items_per_batch = max(1, 32768 // (int(ns) * int(tf.size(reflecting_facets))))
        facets_per_batch = max(1, 32768 // int(ns))
        results = []
        for i in range(0, target_items.shape[0], items_per_batch):
            items = target_items[i:i + items_per_batch]
            ni = tf.shape(items)[0]
            for f in range(0, reflecting_facets.shape[0], facets_per_batch):
                facets = reflecting_facets[f:f + facets_per_batch]
                nf = tf.size(facets)
                sources_rows = tf.repeat(tf.range(ns), ni * nf)
                facets_rows = tf.tile(tf.repeat(facets, ni), [ns])
                items_rows = tf.tile(items, [ns * nf, 1])
                rows = tf.stack([sources_rows, items_rows[:, 0], facets_rows, items_rows[:, 1]], axis=1)
                rows, points, normals = self._solve_reflection_diffraction_rows(
                    sources, targets, rows, vertex, edge_diffraction
                )
                if tf.shape(rows)[0] > 0:
                    results.append((rows, points, normals))
        if not results:
            return paths, tmp
        rows, points, normals = [tf.concat(items, axis=0) for items in zip(*results)]

        # Shared triangle boundaries can yield the same reflection twice.
        # Deduplicate by both interaction points and the diffraction ID, per link.
        link_ids = rows[:, 1] * ns + rows[:, 0]
        key = tf.concat([
            tf.cast(link_ids[:, None], tf.float64), tf.cast(rows[:, 3:4], tf.float64),
            tf.round(tf.cast(tf.reshape(points, [-1, 6]), tf.float64) / 1e-5),
        ], axis=1)
        _, groups = tf.raw_ops.UniqueV2(x=key, axis=[0])
        keep = tf.math.unsorted_segment_min(
            tf.range(tf.shape(rows)[0]), groups, tf.reduce_max(groups) + 1
        )
        rows, points, normals = [tf.gather(x, keep) for x in (rows, points, normals)]

        reflect_objects = tf.gather(self._primitives_2_objects, rows[:, 2])
        diffraction_ids = rows[:, 3]
        if vertex:
            diffraction_objects = tf.gather(
                self.vertex_diffraction.vertices_2_objects_sionna, diffraction_ids
            )
            plain_type = Paths.VERTEX_DIFFRACTED
            forward_order = 21
        else:
            wedge_faces_objects = tf.gather(self._wedges_objects, diffraction_ids)
            diffraction_objects = wedge_faces_objects[:, 0]
            if registered_mask is not None:
                first_face = tf.gather(self._wedges_2_primitives[:, 0], diffraction_ids)
                first_registered = (first_face >= 0) & tf.gather(
                    registered_mask, tf.maximum(first_face, 0)
                )
                # Use the registered object's velocity at a shared interface.
                diffraction_objects = tf.where(
                    first_registered, diffraction_objects, wedge_faces_objects[:, 1]
                )
            plain_type = Paths.DIFFRACTED
            forward_order = 11
        interaction_ids = tf.stack([reflect_objects, diffraction_ids], axis=1)
        object_ids = tf.stack([reflect_objects, diffraction_objects], axis=1)
        if reverse:
            link_rows = rows[:, :2]
            points = tf.reverse(points, axis=[1])
            interaction_ids = tf.reverse(interaction_ids, axis=[1])
            object_ids = tf.reverse(object_ids, axis=[1])
            direction, types = forward_order + 1, [plain_type, Paths.SPECULAR]
        else:
            link_rows = tf.reverse(rows[:, :2], axis=[1])
            direction, types = forward_order, [Paths.SPECULAR, plain_type]

        nr, nt = tf.shape(paths.targets)[0], tf.shape(paths.sources)[0]
        link_ids = link_rows[:, 0] * nt + link_rows[:, 1]
        order = tf.argsort(link_ids, stable=True)
        link_rows, link_ids, points, normals, interaction_ids, object_ids = [
            tf.gather(x, order) for x in (link_rows, link_ids, points, normals, interaction_ids, object_ids)
        ]
        counts = tf.math.unsorted_segment_sum(tf.ones_like(link_ids), link_ids, nr * nt)
        slots = tf.range(tf.size(link_ids)) - tf.gather(tf.cumsum(counts, exclusive=True), link_ids)
        indices = tf.concat([link_rows, slots[:, None]], axis=1)
        shape = tf.stack([nr, nt, tf.reduce_max(counts)])

        def pack(values, fill=None):
            packed_shape = tf.concat([shape, tf.shape(values)[1:]], axis=0)
            if fill is None:
                return tf.scatter_nd(indices, values, packed_shape)
            return tf.tensor_scatter_nd_update(tf.fill(packed_shape, fill), indices, values)

        paths.mask = pack(tf.ones_like(link_ids, tf.bool))
        paths.objects = tf.transpose(pack(interaction_ids, -1), [3, 0, 1, 2])
        paths.true_objects = tf.transpose(pack(object_ids, -1), [3, 0, 1, 2])
        paths.vertices = tf.transpose(pack(points), [3, 0, 1, 2, 4])
        paths.dd_type = tf.fill(shape, direction)
        paths.my_types = tf.broadcast_to(tf.constant(types, tf.int32)[:, None, None, None],
                                         tf.shape(paths.objects))
        tmp.normals = pack(normals)
        paths, tmp = self._compute_directions_distances_delays_angles(paths, tmp, False, True)
        paths.tau = tf.where(paths.mask, paths.tau, -tf.ones_like(paths.tau))
        return paths, tmp

    def _solve_reflection_diffraction_rows(self, sources, targets, rows, vertex, edge_diffraction):
        """Solve the wall image, then test the three physical segments."""
        wd = self.solver_wedge_diffraction
        src = tf.gather(sources, rows[:, 0])
        dst = tf.gather(targets, rows[:, 1])
        facet, diffraction_ids = rows[:, 2], rows[:, 3]
        triangles = tf.cast(tf.gather(self._primitives, facet), tf.float64)
        normal = tf.cast(tf.gather(self._normals, facet), tf.float64)
        normal, _ = normalize(normal)
        p0, p1, p2 = tf.unstack(triangles, num=3, axis=1)
        src64 = tf.cast(src, tf.float64)
        side = dot(src64 - p0, normal)
        image = src64 - 2.0 * side[:, None] * normal
        if vertex:
            q = tf.gather(self.vertex_diffraction._vertices, diffraction_ids)
            valid = tf.ones(tf.shape(diffraction_ids), tf.bool)
        else:
            q, valid = wd._flat_diffraction_points(image, dst, diffraction_ids)
        q64 = tf.cast(q, tf.float64)
        direction, length = normalize(image - q64)
        distance, hit = moller_trumbore(q64, direction, p0, p1, p2, self.EPSILON)
        valid &= hit & (distance < length - self.EPSILON)
        valid &= (tf.abs(side) > self.EPSILON) & (
            tf.sign(side) * dot(q64 - p0, normal) > self.EPSILON
        )
        p = tf.cast(q64 + distance[:, None] * direction, self._rdtype)
        normal = tf.cast(tf.where(side[:, None] < 0, -normal, normal), self._rdtype)

        if not vertex:
            normals = tf.gather(self._wedges_normals, diffraction_ids)
            origin = tf.gather(self._wedges_origin, diffraction_ids)
            side_p = dot((p - origin)[:, None, :], normals)
            side_rx = dot((dst - origin)[:, None, :], normals)
            allowed = (tf.reduce_any(side_p > self.EPSILON, axis=1) &
                       tf.reduce_any(side_rx > self.EPSILON, axis=1))
            allowed |= tf.gather(self._is_edge, diffraction_ids)
            valid &= tf.where(wd._is_flat_shared_wedge(diffraction_ids),
                              wd._same_sides(side_p[:, 0], side_rx[:, 0]), allowed)
        rows, src, dst, p, q, normal = [tf.boolean_mask(x, valid)
                                        for x in (rows, src, dst, p, q, normal)]
        if vertex and tf.shape(rows)[0] > 0:
            edge_paths, _ = self.vertex_diffraction._mixed_vertex_edge_rows(
                rows[:, 3], p, dst, edge_diffraction
            )
            counts = tf.math.unsorted_segment_sum(
                tf.ones_like(edge_paths), edge_paths, tf.shape(rows)[0]
            )
            rows, src, dst, p, q, normal = [
                tf.boolean_mask(x, counts > 0) for x in (rows, src, dst, p, q, normal)
            ]
        if tf.shape(rows)[0] == 0:
            return rows, tf.stack([p, q], axis=1), normal

        starts = tf.concat([src, p, q], axis=0)
        ends = tf.concat([p, q, dst], axis=0)
        directions, distances = normalize(ends - starts)
        blocked = self._test_obstruction(starts, directions, distances)
        clear = ~(blocked | (distances <= self.EPSILON))
        keep = tf.reduce_all(tf.reshape(clear, [3, -1]), axis=0)
        return (tf.boolean_mask(rows, keep), tf.boolean_mask(tf.stack([p, q], axis=1), keep),
                tf.boolean_mask(normal, keep))

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
    nonempty = [p for p in paths if p.objects.shape[3] != 0]
    if not nonempty:
        return paths[0]

    all_paths = nonempty[0]
    for path in nonempty[1:]:
        mask_a = getattr(all_paths, "vertex_wedge_mask", None)
        mask_b = getattr(path, "vertex_wedge_mask", None)

        if mask_a is not None or mask_b is not None:
            num_slots = tf.shape(mask_a if mask_a is not None else mask_b)[-1]

            if mask_a is None:
                mask_a = tf.zeros(tf.concat([tf.shape(all_paths.mask), [num_slots]], axis=0), tf.bool)
            if mask_b is None:
                mask_b = tf.zeros(tf.concat([tf.shape(path.mask), [num_slots]], axis=0), tf.bool)

            tf.debugging.assert_equal(tf.shape(mask_a)[-1], tf.shape(mask_b)[-1],
                                      message="Vertex wedge masks have different slot counts.")
            all_paths.vertex_wedge_mask = tf.concat([mask_a, mask_b], axis=2)

        for name in ("objects", "true_objects", "my_types", "vertices"):
            value = tf.concat([getattr(all_paths, name), getattr(path, name)], axis=3)
            setattr(all_paths, name, value)

        for name in ("mask", "targets_sources_mask", "theta_t", "phi_t", "theta_r", "phi_r", "tau"):
            value = tf.concat([getattr(all_paths, name), getattr(path, name)], axis=2)
            setattr(all_paths, name, value)

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
