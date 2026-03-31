#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Miscellaneous utility functions of the Sionna package."""

import numpy as np
import tensorflow as tf
from tensorflow.experimental.numpy import log10 as _log10
from tensorflow.experimental.numpy import log2 as _log2

def log10(x):
    # pylint: disable=C0301
    """TensorFlow implementation of NumPy's `log10` function.

    Simple extension to `tf.experimental.numpy.log10`
    which casts the result to the `dtype` of the input.
    For more details see the `TensorFlow <https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/log10>`__ and `NumPy <https://numpy.org/doc/1.16/reference/generated/numpy.log10.html>`__ documentation.
    """
    return tf.cast(_log10(x), x.dtype)


def log2(x):
    # pylint: disable=C0301
    """TensorFlow implementation of NumPy's `log2` function.

    Simple extension to `tf.experimental.numpy.log2`
    which casts the result to the `dtype` of the input.
    For more details see the `TensorFlow <https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/log2>`_ and `NumPy <https://numpy.org/doc/1.16/reference/generated/numpy.log2.html>`_ documentation.
    """
    return tf.cast(_log2(x), x.dtype)