#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#


import sionna
import os
import unittest
import numpy as np
import tensorflow as tf
import mitsuba as mi
import drjit as dr

gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)

from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
from utils import *

class TestDicretizedSphere(unittest.TestCase):
    def test_dicretized_sphere(self):
        #sphere = os.path.dirname(os.path.realpath(__file__)) + "/sphere/sphere.xml"
        #scene = load_scene(sphere)
        scene = load_scene(sionna.rt.scene.sphere)

        # num_shapes = len(scene._solver_paths._mi_scene.shapes())
        # num_vertices = scene._solver_paths._mi_scene.shapes()[0].vertex_count()
        # vertices = scene._solver_paths._mi_scene.shapes()[0].vertex_position(dr.arange(mi.UInt32, num_vertices))

        num_shapes = len(scene._solver_paths._mi_scene.shapes())
        num_edges = tf.reduce_sum(tf.cast(scene._solver_paths._is_edge, tf.int32))
        num_wedges = scene._solver_paths._wedges_origin.shape[0]

        print("nums:", num_shapes, num_edges, num_wedges)
        print("all_edges", scene._solver_paths.all_edges)

        self.assertTrue(num_shapes == 1), num_shapes
        self.assertTrue(num_edges == 0), num_edges
        self.assertTrue(num_wedges == 120), num_wedges