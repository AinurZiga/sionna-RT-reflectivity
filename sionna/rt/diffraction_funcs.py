from __future__ import annotations
from pathlib import Path

import numpy as np
import mitsuba as mi
import drjit as dr
import tensorflow as tf
import trimesh
from typing import Tuple, TYPE_CHECKING

from sionna.constants import SPEED_OF_LIGHT, PI
from sionna.utils.tensors import expand_to_rank, insert_dims, flatten_dims,\
    split_dim
from .paths import Paths
from .utils import dot, phi_hat, theta_hat, theta_phi_from_unit_vec,\
    normalize, moller_trumbore, component_transform, mi_to_tf_tensor,\
        compute_field_unit_vectors, reflection_coefficient, fibonacci_lattice,\
            cot, cross, sign, rotation_matrix, acos_diff, rot_mat_from_unit_vecs, csc

CALC = 'TF'  # ['TF, 'Umul']

#########################################
## diffraction tables
######################################

# class DiffractionTables:
#     def __init__(self, dtype=tf.complex64):
#         self.dtype = dtype
#         self.gfi_DIV = 0.0025
#         file_gfi_table = Path(__file__).resolve().parent / "data" / "gfi_00025.npy"
#         self.gfi_table_tf = tf.convert_to_tensor(np.load(file_gfi_table), dtype=dtype)

#         self.file_T_ev_1 = Path(__file__).resolve().parent / "data" / "T_ev_even_linear_table_new.npz"
#         self.file_T_ev_2 = Path(__file__).resolve().parent / "data" / "T_ev_odd_linear_table_new.npz"

class DiffractionTables:
    """Load shared GFI tables once and retain paths to EV transition tables."""

    def __init__(self, dtype=tf.complex64, use_gfi_lookup=True, gfi_filename=None):
        self.dtype = tf.as_dtype(dtype)
        data_dir = Path(__file__).resolve().parent / "data"
        gfi_filename = data_dir / "gfi_hybrid_tables2.npz" if gfi_filename is None else Path(gfi_filename)

        self.gfi_lookup = GFILookup(gfi_filename, dtype=self.dtype) if use_gfi_lookup else None

        # These independent EV tables are still required by double diffraction.
        self.file_T_ev_1 = data_dir / "T_ev_even_linear_table_new.npz"
        self.file_T_ev_2 = data_dir / "T_ev_odd_linear_table_new.npz"


def calc_angles(n_0_hat, e_hat, s_prime_hat, s_hat):
    r"""
    Input
    ------
    n_0_hat: [num_targets, num_sources, max_num_paths, 3]
    e_hat: [num_targets, num_sources, max_num_paths, 3]
    s_prime_hat: [num_targets, num_sources, max_num_paths, 3]
    s_hat: [num_targets, num_sources, max_num_paths, 3]
    """
    # [num_targets, num_sources, max_num_paths, 3]
    t_0_hat = cross(n_0_hat, e_hat)

    # Compute s_t_prime_hat and s_t_hat
    # [num_targets, num_sources, max_num_paths, 3]
    s_t_prime_hat, _ = normalize(s_prime_hat
                            - dot(s_prime_hat, e_hat, keepdim=True)*e_hat)
    # [num_targets, num_sources, max_num_paths, 3]
    s_t_hat, _ = normalize(s_hat - dot(s_hat, e_hat, keepdim=True)*e_hat)

    my_phi_prime_tmp = acos_diff(dot(-s_t_prime_hat, t_0_hat))
    my_phi_prime = tf.where(sign(dot(e_hat, cross(t_0_hat, -s_t_prime_hat))) > 0.0, my_phi_prime_tmp, 2*PI - my_phi_prime_tmp)

    my_phi_tmp = acos_diff(dot(s_t_hat, t_0_hat))
    my_phi = tf.where(sign(dot(e_hat, cross(t_0_hat, s_t_hat))) > 0.0, my_phi_tmp, 2*PI - my_phi_tmp)

    # Compute elevation beta_prime
    # [num_targets, num_sources, max_num_paths]
    beta = acos_diff(dot(e_hat, s_hat))
    beta_prime = acos_diff(dot(e_hat, s_prime_hat))

    return my_phi_prime, my_phi, beta_prime, beta

def calc_angles_concave(n_0_hat, e_hat, s_prime_hat, s_hat, n):
    r"""
    Input
    ------
    n_0_hat: [num_targets, num_sources, max_num_paths, 3]
    e_hat: [num_targets, num_sources, max_num_paths, 3]
    s_prime_hat: [num_targets, num_sources, max_num_paths, 3]
    s_hat: [num_targets, num_sources, max_num_paths, 3]
    n: [num_targets, num_sources, max_num_paths]

    for n < 1 (concave angle) change the direction of t_0_hat
    CONCAVE
    """
    # [num_targets, num_sources, max_num_paths, 3]
    t_0_hat = -cross(n_0_hat, e_hat)

    # Compute s_t_prime_hat and s_t_hat
    # [num_targets, num_sources, max_num_paths, 3]
    s_t_prime_hat, _ = normalize(s_prime_hat
                            - dot(s_prime_hat, e_hat, keepdim=True)*e_hat)
    # [num_targets, num_sources, max_num_paths, 3]
    s_t_hat, _ = normalize(s_hat - dot(s_hat, e_hat, keepdim=True)*e_hat)

    # Compute phi_prime and phi
    # [num_targets, num_sources, max_num_paths]
    my_phi_prime = PI - \
        (PI-acos_diff(-dot(s_t_prime_hat, t_0_hat)))\
            * sign(-dot(s_t_prime_hat, n_0_hat))
    # [num_targets, num_sources, max_num_paths]
    my_phi = PI - (PI-acos_diff(dot(s_t_hat, t_0_hat)))\
        * sign(dot(s_t_hat, n_0_hat))

    # Compute elevation beta_prime
    # [num_targets, num_sources, max_num_paths]
    beta = acos_diff(dot(e_hat, s_hat))
    beta_prime = acos_diff(dot(e_hat, s_prime_hat))

    return my_phi_prime, my_phi, beta_prime, beta

def cot_transition_stable(beta, n, k_ell, plus=True):
    r"""
    Evaluate cot((pi +/- beta)/(2*n)) * F(k_ell * a) jointly.

    The expression is rearranged to avoid cancellation between singular
    derivatives of the cotangent and the transition function near a boundary.

    Parameters
    ----------
    beta : tf.Tensor, real
        Angular argument: phi - phi_prime or phi + phi_prime.
    n : tf.Tensor, real
        Wedge parameter. Must be positive.
    k_ell : tf.Tensor, real
        Nonnegative transition scale k * ell.
    plus : bool
        Select the plus or minus UTD term.

    Returns
    -------
    tf.Tensor, complex
        Joint cotangent-transition product, with broadcast input shape.
        At delta=0, the positive one-sided value is selected. This convention
        does not resolve consistency with the GO boundary classification.
    """

    beta = tf.convert_to_tensor(beta)
    n = tf.cast(n, beta.dtype)
    k_ell = tf.cast(k_ell, beta.dtype)
    pi = tf.constant(PI, dtype=beta.dtype)

    # Use the same branch selection as the original a_p/a_m functions.
    if plus:
        turns = tf.round((beta + pi) / (2.0 * n * pi))
        delta = beta + pi - 2.0 * n * pi * turns
    else:
        turns = tf.round((beta - pi) / (2.0 * n * pi))
        delta = pi - beta + 2.0 * n * pi * turns

    def sinc_unscaled(x):
        # sin(x)/x, including a stable derivative near zero.
        small = tf.abs(x) < tf.cast(1e-2, x.dtype)
        x2 = tf.square(x)
        series = 1.0 - x2 / 6.0 + x2**2 / 120.0 - x2**3 / 5040.0
        safe_x = tf.where(small, tf.ones_like(x), x)
        regular = tf.sin(safe_x) / safe_x
        return tf.where(small, series, regular)

    half_delta = 0.5 * delta
    cot_angle = half_delta / n
    scale = tf.sqrt(2.0 * k_ell)
    sqrt_x = scale * tf.abs(tf.sin(half_delta))

    side = tf.where(delta >= 0.0, tf.ones_like(delta), -tf.ones_like(delta))
    ratio = sinc_unscaled(half_delta) / sinc_unscaled(cot_angle)
    amplitude = scale * n * side * ratio * tf.cos(cot_angle)

    q = fresnel_integral_capolino(tf.complex(sqrt_x, tf.zeros_like(sqrt_x)), sqrt_x)
    return tf.complex(tf.zeros_like(amplitude), 2.0 * amplitude) * q

def t_gfi_full_approximated_capolino(b, a, table: DiffractionTables = None):
    r"""
    Evaluate the normalized GFI transition function T(b, a).

    Near a=0 with b>0, use a first-order expansion to avoid the indeterminate
    product G(sqrt(b), sqrt(a)) * sqrt(b/a).

    Parameters
    ----------
    b, a : tf.Tensor, real
        Broadcast-compatible arguments. Requires b>0 and a>=0.
    table : DiffractionTables or None
        Optional GFI lookup table for the regular branch.

    Returns
    -------
    tf.Tensor, complex
        Transition function with the broadcast input shape.

    Notes
    -----
    The small-a expansion preserves the limiting value and the first
    derivative with respect to a at a=0.
    The joint limit a=b=0 and the boundary b=0 are not handled.
    """

    b = tf.convert_to_tensor(b)
    a = tf.cast(a, b.dtype)
    tf.debugging.assert_positive(b, message="T_GFI: b=0 requires a separate limit.")
    tf.debugging.assert_greater_equal(a, tf.zeros_like(a), message="T_GFI: a must be nonnegative.")

    output_dtype = tf.complex128 if b.dtype == tf.float64 else tf.complex64
    output_shape = tf.broadcast_dynamic_shape(tf.shape(b), tf.shape(a))
    b = tf.reshape(tf.broadcast_to(b, output_shape), [-1])
    a = tf.reshape(tf.broadcast_to(a, output_shape), [-1])

    # Require both a/b and a to be small.
    expansion_limit = tf.cast(1e-4, b.dtype) * tf.minimum(b, tf.ones_like(b))
    small = a <= expansion_limit
    small_idx = tf.where(small)[:, 0]
    regular_idx = tf.where(~small)[:, 0]

    # Evaluate only the selected branch, avoiding inactive 0/0 operations.
    b_small = tf.gather(b, small_idx)
    a_small = tf.gather(a, small_idx)

    # Local double precision protects the subtraction 1-F(b).
    b64 = tf.cast(b_small, tf.float64)
    a64 = tf.cast(a_small, tf.float64)
    f = transition_func(b64)
    one_minus_f = tf.ones_like(f) - f

    t0 = tf.complex(tf.zeros_like(b64), 2.0 * b64) * one_minus_f
    t1 = (
        tf.complex(tf.zeros_like(b64), 2.0 * tf.ones_like(b64)) * (2.0 / 3.0 - f)
        - tf.cast((4.0 / 3.0) * b64, f.dtype) * one_minus_f
    )
    small_values = t0 + tf.cast(a64, f.dtype) * t1

    b_regular = tf.gather(b, regular_idx)
    a_regular = tf.gather(a, regular_idx)
    x = tf.sqrt(b_regular)
    y = tf.sqrt(a_regular)

    g = func_G_approximated_capolino(x, y, table)
    multiplier = 4.0 * PI * (x / y) * (a_regular + b_regular)
    regular_values = g * tf.cast(
        tf.complex(tf.zeros_like(multiplier), multiplier), g.dtype
    )

    result = tf.zeros(tf.shape(b), dtype=output_dtype)
    result = tf.tensor_scatter_nd_update(
        result, small_idx[:, None], tf.cast(small_values, output_dtype)
    )
    result = tf.tensor_scatter_nd_update(
        result, regular_idx[:, None], tf.cast(regular_values, output_dtype)
    )
    return tf.reshape(result, output_shape)

def func_G_approximated_capolino(x, y, table: DiffractionTables = None):
    r"""
    Evaluate GFI for real, broadcast-compatible arguments.

    Parameters
    ----------
    x, y : tf.Tensor, real
        GFI arguments. Signed arguments use odd extension in each variable.
    table : DiffractionTables or None
        Shared tables. If the GFI lookup is disabled, evaluate the existing
        analytical approximations directly.

    Returns
    -------
    tf.Tensor, complex
        GFI values with the broadcast input shape.

    Notes
    -----
    The lookup performs its own sign extension and shape handling.
    Its joint-origin and domain restrictions remain applicable.
    """

    lookup = None if table is None else table.gfi_lookup
    if lookup is not None:
        return lookup(x, y)

    x = tf.convert_to_tensor(x)
    y = tf.cast(y, x.dtype)
    initial_shape = tf.broadcast_dynamic_shape(tf.shape(x), tf.shape(y))
    x = tf.reshape(tf.broadcast_to(x, initial_shape), [-1])
    y = tf.reshape(tf.broadcast_to(y, initial_shape), [-1])

    gfi_sign = sign(x) * sign(y)
    values = _func_G_approximated_capolino2(tf.abs(x), tf.abs(y))
    values *= tf.cast(gfi_sign, values.dtype)

    return tf.reshape(values, initial_shape)

def _func_G_table2_regular(x, y, table: DiffractionTables):
    dtype = table.dtype
    table_tf = table.gfi_table_tf
    DIV = table.gfi_DIV

    cond_approx1_idxs = tf.where(tf.math.logical_or(x >= 3.0, y >= 3.0))[:, 0]
    table_cond_idxs = tf.where(tf.math.logical_and(x < 3.0, y < 3.0))[:, 0]

    def _from_table(x_table, y_table):
        x_idx = tf.cast(tf.math.floor(x_table / DIV), tf.int32)
        y_idx = tf.cast(tf.math.floor(y_table / DIV), tf.int32)
        ##
        x_idx = tf.where(x_idx < 0, tf.zeros_like(x_idx), x_idx)
        x_idx = tf.where(x_idx >= tf.shape(table_tf)[0]-2, tf.zeros_like(x_idx), x_idx)
        y_idx = tf.where(y_idx < 0, tf.zeros_like(y_idx), y_idx)
        y_idx = tf.where(y_idx >= tf.shape(table_tf)[1]-2, tf.zeros_like(y_idx), y_idx)
        ##
        idxs = tf.stack([x_idx, y_idx], axis=-1)
        idxs_10 = tf.stack([x_idx + 1, y_idx], axis=-1)
        idxs_01 = tf.stack([x_idx, y_idx + 1], axis=-1)
        idxs_11 = tf.stack([x_idx + 1, y_idx + 1], axis=-1)

        table_00 = tf.gather_nd(table_tf, idxs)
        table_10 = tf.gather_nd(table_tf, idxs_10)
        table_01 = tf.gather_nd(table_tf, idxs_01)
        table_11 = tf.gather_nd(table_tf, idxs_11)

        # bilinear interpolation
        x1 = tf.cast(x_idx, x.dtype) * DIV
        x2 = tf.cast(x_idx + 1, x.dtype) * DIV
        y1 = tf.cast(y_idx, x.dtype) * DIV
        y2 = tf.cast(y_idx + 1, x.dtype) * DIV
        res = table_00 * tf.cast((x2 - x_table)/DIV, dtype) * tf.cast((y2 - y_table)/DIV, dtype) + \
                table_10 * tf.cast((x_table - x1)/DIV, dtype) * tf.cast((y2 - y_table)/DIV, dtype) + \
                table_01 * tf.cast((x2 - x_table)/DIV, dtype) * tf.cast((y_table - y1)/DIV, dtype) + \
                table_11 * tf.cast((x_table - x1)/DIV, dtype) * tf.cast((y_table - y1)/DIV, dtype)
        
        return res

    res_approx1 = G_approx1(tf.gather(x, cond_approx1_idxs), tf.gather(y, cond_approx1_idxs))
    res_table_cond = _from_table(tf.gather(x, table_cond_idxs), tf.gather(y, table_cond_idxs))

    res = tf.zeros([tf.shape(x)[0]], dtype=dtype)
    res = tf.tensor_scatter_nd_update(res, tf.reshape(cond_approx1_idxs, (-1, 1)), res_approx1)
    res = tf.tensor_scatter_nd_update(res, tf.reshape(table_cond_idxs, (-1, 1)), res_table_cond)
    
    return res

def _func_G_table2(x, y, table: DiffractionTables):
    r"""
    Evaluate G using an analytical approximation near the origin and a table
    elsewhere.

    The small-argument region matches the existing non-table implementation.
    Direct evaluation preserves the dependence on x/y near the origin, which
    cannot be represented by bilinear interpolation in the first table cell.

    Parameters
    ----------
    x, y : tf.Tensor, real, shape [N]
        Nonnegative GFI arguments.
    table : DiffractionTables
        Lookup table and interpolation settings.

    Returns
    -------
    tf.Tensor, complex, shape [N]
        Approximated GFI values.

    Notes
    -----
    The joint origin has direction-dependent limits and is not assigned a
    unique physical value here.
    """

    small = (x <= 0.4) & (y <= 0.4)
    small_idx = tf.where(small)[:, 0]
    regular_idx = tf.where(~small)[:, 0]

    x_small = tf.gather(x, small_idx)
    y_small = tf.gather(y, small_idx)
    small_values = _func_G_reg1(
        tf.complex(x_small, tf.zeros_like(x_small)),
        tf.complex(y_small, tf.zeros_like(y_small)),
        x_small, y_small
    )

    regular_values = _func_G_table2_regular(
        tf.gather(x, regular_idx), tf.gather(y, regular_idx), table
    )

    result = tf.zeros(tf.shape(x), dtype=table.dtype)
    result = tf.tensor_scatter_nd_update(result, small_idx[:, None], tf.cast(small_values, table.dtype))
    result = tf.tensor_scatter_nd_update(result, regular_idx[:, None], tf.cast(regular_values, table.dtype))
    return result

def _func_G_approximated_capolino2(x, y):
    # new and faster
    # x, y - real

    cond1_idxs = tf.where(tf.math.logical_and(x <= 0.4, y <= 0.4))[:, 0]
    cond2_idxs = tf.where(tf.math.logical_and(x >= 0.4, tf.math.logical_and(x <= 24.0, x >= 4.0*y)))[:, 0]
    cond3_idxs = tf.where(tf.math.logical_and(x >= 24.0, x >= y))[:, 0]
    cond4_idxs = tf.where(tf.math.logical_and(tf.math.logical_and(x >= y/4.0, x <= y), tf.math.logical_and(y >= 0.4, y <= 24.0)))[:, 0]
    cond5_idxs = tf.where(tf.math.logical_and(y >= 0.4, tf.math.logical_and(y <= 24.0, y >= 4.0*x)))[:, 0]
    cond6_idxs = tf.where(tf.math.logical_and(y >= 24.0, y >= x))[:, 0]
    cond7_idxs = tf.where(tf.math.logical_and(tf.math.logical_and(y >= x/4.0, y <= x), tf.math.logical_and(x >= 0.4, x <= 24.0)))[:, 0]

    x_complex = tf.complex(x, tf.cast(0.0, x.dtype))
    y_complex = tf.complex(y, tf.cast(0.0, x.dtype))
    large = tf.complex(tf.cast(10.0, x.dtype), tf.cast(0.0, x.dtype))

    res1 = _func_G_reg1(tf.gather(x_complex, cond1_idxs), tf.gather(y_complex, cond1_idxs), tf.gather(x, cond1_idxs), tf.gather(y, cond1_idxs))
    res2 = _func_G_reg2(tf.gather(x_complex, cond2_idxs), tf.gather(y_complex, cond2_idxs), tf.gather(x, cond2_idxs), tf.gather(y, cond2_idxs))
    res3 = _func_G_reg3_new(tf.gather(x_complex, cond3_idxs), tf.gather(y_complex, cond3_idxs), tf.gather(x, cond3_idxs), tf.gather(y, cond3_idxs))
    res4 = _func_G_reg4(tf.gather(x_complex, cond4_idxs), tf.gather(y_complex, cond4_idxs), tf.gather(x, cond4_idxs), tf.gather(y, cond4_idxs))
    res5 = _func_G_reg5(tf.gather(x_complex, cond5_idxs), tf.gather(y_complex, cond5_idxs), tf.gather(x, cond5_idxs), tf.gather(y, cond5_idxs))
    res6 = _func_G_reg6(tf.gather(x_complex, cond6_idxs), tf.gather(y_complex, cond6_idxs), tf.gather(x, cond6_idxs), tf.gather(y, cond6_idxs))
    res7 = _func_G_reg7(tf.gather(x_complex, cond7_idxs), tf.gather(y_complex, cond7_idxs), tf.gather(x, cond7_idxs), tf.gather(y, cond7_idxs))

    res = tf.zeros([x.shape[0]], dtype=res1.dtype) * large
    res = tf.tensor_scatter_nd_update(res, tf.reshape(cond1_idxs, (-1, 1)), res1)
    res = tf.tensor_scatter_nd_update(res, tf.reshape(cond2_idxs, (-1, 1)), res2)
    res = tf.tensor_scatter_nd_update(res, tf.reshape(cond3_idxs, (-1, 1)), res3)
    res = tf.tensor_scatter_nd_update(res, tf.reshape(cond4_idxs, (-1, 1)), res4)
    res = tf.tensor_scatter_nd_update(res, tf.reshape(cond5_idxs, (-1, 1)), res5)
    res = tf.tensor_scatter_nd_update(res, tf.reshape(cond6_idxs, (-1, 1)), res6)
    res = tf.tensor_scatter_nd_update(res, tf.reshape(cond7_idxs, (-1, 1)), res7)
    
    return res

def _func_G_reg1(x, y, x_real, y_real):
    r"""
    Evaluate the small-argument GFI approximation.

    Parameters
    ----------
    x, y : tf.Tensor, complex
        Complex representations of the nonnegative real arguments.
    x_real, y_real : tf.Tensor, real
        Corresponding real arguments, with matching shapes.

    Returns
    -------
    tf.Tensor, complex
        Small-argument GFI approximation.

    Notes
    -----
    atan2(x_real, y_real) replaces atan(x/y) for nonnegative real arguments.
    The joint origin has direction-dependent limits; atan2(0, 0) does not
    resolve that physical ambiguity.
    """

    pi = tf.constant(PI, dtype=x_real.dtype)
    imag_i = tf.complex(tf.zeros_like(x_real), tf.ones_like(x_real))
    sqrt_pi = tf.cast(tf.sqrt(pi), x.dtype)

    # Preserve the angular dependence without dividing by a small y.
    angle = tf.cast(tf.math.atan2(x_real, y_real), x.dtype)
    phase = tf.exp(imag_i * tf.cast(pi / 4.0, x.dtype))
    fresnel = fresnel_integral_capolino(y, y_real) / sqrt_pi

    term = ((1.0 + imag_i * y**2) * angle - imag_i * x * y) / tf.cast(pi, x.dtype)
    return 0.5 * tf.exp(imag_i * x**2) * (phase * fresnel - term)

def _func_G_reg2(x, y, x_real, y_real):
    imag_i = tf.complex(tf.cast(0.0, x_real.dtype), tf.cast(1.0, x_real.dtype))

    tmp1 = y / (2*PI * x)
    tmp2 = 1 - 2 * imag_i * x * fresnel_integral_capolino(x, x_real)
    tmp3 = (y / x) ** 3 / (6*PI)
    tmp4 = 1 - 2 * imag_i * x*x - 4*x*x*x * fresnel_integral_capolino(x, x_real)

    return tmp1 * tmp2 - tmp3 * tmp4

def _func_G_reg3(x, y, x_real, y_real):
    tmp1 = 1 / (2*PI) 
    tmp2 = y / (x**2 + y**2)
    tmp3 = fresnel_integral_capolino(x, x_real)

    return tmp1 * tmp2 * tmp3

def _func_G_reg3_new(x, y, x_real, y_real):
    tmp1 = 1 / (2*PI) 
    tmp2 = y / ((x**2 + y**2) * 2j*x)
    tmp3 = (3*x**2 + y**2) / (x*(x**2 + y**2))
    tmp4 = fresnel_integral_capolino(x, x_real)

    return tmp1 * tmp2 * (1 - tmp3*tmp4)

def _func_G_reg4(x, y, x_real, y_real):
    imag_i = tf.complex(tf.cast(0.0, x_real.dtype), tf.cast(1.0, x_real.dtype))

    tmp1 = tf.math.exp(imag_i * x*x) / (2*PI)
    tmp2 = imag_i * \
            tf.math.exp(-imag_i * y*y) * \
                fresnel_integral_capolino(y, y_real)**2 + _integral_I(x, y, x_real, y_real)

    return tmp1 * tmp2

def _func_G_reg5(x, y, x_real, y_real):
    imag_i = tf.complex(tf.cast(0.0, x_real.dtype), tf.cast(1.0, x_real.dtype))

    return imag_i * fresnel_integral_capolino(y, y_real) * fresnel_integral_capolino(x, x_real) / \
        PI - _func_G_reg2(y, x, y_real, x_real)

def _func_G_reg6(x, y, x_real, y_real):
    imag_i = tf.complex(tf.cast(0.0, x_real.dtype), tf.cast(1.0, x_real.dtype))

    return imag_i * fresnel_integral_capolino(y, y_real) * fresnel_integral_capolino(x, x_real) / \
        PI - _func_G_reg3(y, x, y_real, x_real)

def _func_G_reg7(x, y, x_real, y_real):
    imag_i = tf.complex(tf.cast(0.0, x_real.dtype), tf.cast(1.0, x_real.dtype))

    return imag_i * fresnel_integral_capolino(y, y_real) * fresnel_integral_capolino(x, x_real) / \
        PI - _func_G_reg4(y, x, y_real, x_real)


def _integral_I(x, y, x_real, y_real):
    a0 = 0.98926061  # tf.constant or tf.complex
    a1 = 0.12808114
    a2 = -1.59843183
    a3 = 1.35987246
    a4 = -0.37895149
    imag_i = tf.complex(tf.cast(0.0, x_real.dtype), tf.cast(1.0, x_real.dtype))

    I0 = (fresnel_integral_capolino(x, x_real) * tf.math.exp(-imag_i * x*x) - \
          fresnel_integral_capolino(y, y_real) * tf.math.exp(-imag_i * y*y)) / y
    I1 = imag_i*(-tf.math.exp(-imag_i *x*x) + tf.math.exp(-imag_i * y*y)) / (2*y*y)
    I2 = imag_i*(-x/y * tf.math.exp(-imag_i*x*x) + tf.math.exp(-imag_i*y*y) - I0) / (2*y*y)
    I3 = imag_i*(-(x/y)**2 * tf.math.exp(-imag_i*x*x) + tf.math.exp(-imag_i*y*y) - 2*I1) / (2*y*y)
    I4 = imag_i*(-(x/y)**3 * tf.math.exp(-imag_i*x*x) + tf.math.exp(-imag_i*y*y) - 3*I2) / (2*y*y)

    return a0*I0 + a1*I1 + a2*I2 + a3*I3 + a4*I4

def fresnel_integral_capolino(ksi, ksi_real):
    imag_i = tf.complex(tf.cast(0.0, ksi_real.dtype), tf.cast(1.0, ksi_real.dtype))

    if CALC == 'TF':
        return tf.math.exp(imag_i * ksi**2) * fresnel_integral(ksi_real)
    
    elif CALC == 'Umul':
        phase = tf.cast(tf.sqrt(PI), ksi.dtype) * tf.exp(imag_i * (ksi**2 - PI / 4.0))
        return phase * fresnel_integral_umul(ksi, ksi_real)
    
    else:
        raise ValueError(f"Unknown CALC mode: {CALC}")

def fresnel_integral(y_real):
    r"""intergral from y to +inf, without any coefs"""

    sqrt_pi_div_2 = tf.math.sqrt(tf.cast(PI, y_real.dtype) / 2.0)
    sqrt_pi_div_8 = tf.math.sqrt(tf.cast(PI, y_real.dtype) / 8.0)
    
    s = sqrt_pi_div_2 * tf.math.special.fresnel_sin(y_real / sqrt_pi_div_2)
    c = sqrt_pi_div_2 * tf.math.special.fresnel_cos(y_real / sqrt_pi_div_2)

    c = sqrt_pi_div_8 - c
    s = s - sqrt_pi_div_8

    return tf.complex(c, s)

def fresnel_integral_umul(x, x_real):
    """
    Compute the UMUL approximation of the complex Fresnel integral.

    Parameters
    ----------
    x : complex or tf.Tensor
        complex-valued input.

    x_real : real
        Real part of the input.

    Returns
    -------
    tf.Tensor
        Complex-valued result.
    """
    zero = tf.cast(0.0, x_real.dtype)
    pi = tf.cast(PI, x_real.dtype)

    sqrt_term_complex = tf.cast(2.0 * tf.sqrt(pi), x.dtype) * x
    phase_pi_4 = tf.complex(zero, pi / 4.0)

    exp_func = tf.exp(phase_pi_4) * sqrt_term_complex
    term1 = 1.0 / (1.0 - tf.exp(exp_func))

    phase = -tf.cast(1j, x.dtype) * (x**2 + tf.cast(pi / 4.0, x.dtype))
    term2 = tf.exp(phase) / sqrt_term_complex

    return term1 + term2

def transition_func(x):
    sqrt_x = tf.sqrt(x)

    return (tf.complex(tf.zeros_like(x), 2.0 * sqrt_x) *  
            fresnel_integral_capolino(tf.complex(sqrt_x, tf.zeros_like(x)), sqrt_x))


####################################################
## GFI approximation for x > 3.0 or y > 3.0

def fresnel_tail_I0(x):
    """
    I0(x) = integral from x to infinity of exp(-j*t^2) dt
    for real x >= 0.
    """
    zero = tf.zeros_like(x)
    x_complex = tf.complex(x, zero)
    phase = tf.exp(tf.complex(zero, -tf.square(x)))

    return phase * fresnel_integral_capolino(x_complex, x)

def G_approx1(x, y):
    """
    First-order approximation:
    G1(x,y) = (y/(2π)) exp(j x^2) *
              [ I0/denom - correction/denom^2 ]
    where
    correction = (x/(2j)) exp(-j x^2) + (1/(2j) - x^2) I0
    """

    I0 = fresnel_tail_I0(x)
    denom = x**2 + y**2

    exp_jx2 = tf.exp(tf.complex(tf.zeros_like(x), x**2))
    exp_mjx2 = tf.exp(tf.complex(tf.zeros_like(x), -x**2))

    correction = (
        tf.complex(tf.zeros_like(x), -x / 2.0) * exp_mjx2
        + (tf.complex(-x**2, -0.5 * tf.ones_like(x))) * I0
    )

    bracket = I0 / tf.complex(denom, tf.zeros_like(x)) \
              - correction / tf.complex(denom**2, tf.zeros_like(x))

    pref = tf.complex(y / (2.0 * PI), tf.zeros_like(x))
    return pref * exp_jx2 * bracket

#####################################################################


def invert_angles(phi, n):
    return 2*PI - (2-n) * PI - phi

def wd_get_par_a(phi_prime, phi, ell, n, scene):
    """
    """
    # dtype = scene.dtype
    wavelength = scene.wavelength
    k = 2.*PI/wavelength

    # [num_targets, num_sources, max_num_paths]
    phi_m = phi - phi_prime
    phi_p = phi + phi_prime

    def n_p(beta, n):
        return tf.math.round((beta + PI)/(2.*n*PI))

    def n_m(beta, n):
        return tf.math.round((beta - PI)/(2.*n*PI))

    def a_p(beta, n):
        return 2*tf.cos((2.*n*PI*n_p(beta, n)-beta)/2.)**2

    def a_m(beta, n):
        return 2*tf.cos((2.*n*PI*n_m(beta, n)-beta)/2.)**2

    a1 = a_p(phi_m, n) * k * ell
    a2 = a_m(phi_m, n) * k * ell
    a3 = a_p(phi_p, n) * k * ell
    a4 = a_m(phi_p, n) * k * ell

    return a1, a2, a3, a4

def wd_compute_mat_t_using_par_a(par_a_tuple, phi_prime, phi, beta_prime, s_prime, s, n, mask, scene):
    """
    1) No material properties
    2) Heuristic update of the coefficient of ell is possible
    """
    dtype = scene.dtype
    wavelength = scene.wavelength
    k = 2.*PI/wavelength

    # Compute D_1, D_2, D_3, D_4
    # [num_targets, num_sources, max_num_paths]
    phi_m = phi - phi_prime
    phi_p = phi + phi_prime

    # [num_targets, num_sources, max_num_paths]
    cot_1 = cot((PI + phi_m)/(2*n))
    cot_2 = cot((PI - phi_m)/(2*n))
    cot_3 = cot((PI + phi_p)/(2*n))
    cot_4 = cot((PI - phi_p)/(2*n))

    d_mul = - tf.cast(tf.exp(-1j*PI/4.), dtype)/\
        tf.cast((2*n)*tf.sqrt(2*PI*k)*tf.math.sin(beta_prime), dtype)

    # [num_targets, num_sources, max_num_paths]
    cot_1 = tf.complex(cot_1, tf.zeros_like(cot_1))
    cot_2 = tf.complex(cot_2, tf.zeros_like(cot_2))
    cot_3 = tf.complex(cot_3, tf.zeros_like(cot_3))
    cot_4 = tf.complex(cot_4, tf.zeros_like(cot_4))

    a1, a2, a3, a4 = par_a_tuple

    f1 = transition_func(a1)
    f2 = transition_func(a2)
    f3 = transition_func(a3)
    f4 = transition_func(a4)

    d_1 = d_mul * cot_1 * f1
    d_2 = d_mul * cot_2 * f2
    d_3 = d_mul * cot_3 * f3
    d_4 = d_mul * cot_4 * f4

    d_soft = d_1 + d_2 - d_3 - d_4
    d_hard = d_1 + d_2 + d_3 + d_4

    _mat_t1 = tf.stack([d_hard, tf.zeros_like(d_hard)], axis=-1)
    _mat_t2 = tf.stack([tf.zeros_like(d_soft), d_soft], axis=-1)
    mat_t = tf.stack([_mat_t1, _mat_t2], axis=-2)

    spreading_factor = tf.sqrt(s_prime / (s*(s_prime + s)))
    spreading_factor = tf.complex(spreading_factor, tf.zeros_like(spreading_factor))

    mat_t *= -spreading_factor[..., None, None]
    # [num_targets, num_sources, max_num_paths, 1, 1]
    mask_ = expand_to_rank(mask, tf.rank(mat_t), axis=3)
    # [num_targets, num_sources, max_num_paths, 2]
    mat_t = tf.where(mask_, mat_t, tf.zeros_like(mat_t))

    return mat_t

def my_wd_compute_fields(phi_prime, phi, beta_prime, ell, s_prime, s, n, mask, scene, flag=False, coef_ell=None, 
                         new_coef_ell=None):
    """
    1) No material properties
    2) Heuristic update of the coefficient of ell is possible
    """
    dtype = scene.dtype
    wavelength = scene.wavelength
    k = 2.*PI/wavelength

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

    d_mul = - tf.cast(tf.exp(-1j*PI/4.), dtype)/\
        tf.cast((2*n)*tf.sqrt(2*PI*k)*tf.math.sin(beta_prime), dtype)

    # [num_targets, num_sources, max_num_paths]
    cot_1 = tf.complex(cot_1, tf.zeros_like(cot_1))
    cot_2 = tf.complex(cot_2, tf.zeros_like(cot_2))
    cot_3 = tf.complex(cot_3, tf.zeros_like(cot_3))
    cot_4 = tf.complex(cot_4, tf.zeros_like(cot_4))

    if new_coef_ell is not None:
        if tf.rank(new_coef_ell) > tf.rank(ell):
            ell = ell[..., None]  
            phi_m = phi_m[..., None]
            phi_p = phi_p[..., None]
            n = n[..., None]
            cot_1 = cot_1[..., None]
            cot_2 = cot_2[..., None]
            cot_3 = cot_3[..., None]
            cot_4 = cot_4[..., None]
            d_mul = d_mul[..., None]
            s_prime = s_prime[..., None]
            s = s[..., None]

        ell = ell * new_coef_ell

    a1 = a_p(phi_m, n) * k * ell
    a2 = a_m(phi_m, n) * k * ell
    a3 = a_p(phi_p, n) * k * ell
    a4 = a_m(phi_p, n) * k * ell

    f1 = transition_func(a1)
    f2 = transition_func(a2)
    f3 = transition_func(a3)
    f4 = transition_func(a4)

    d_1 = d_mul * cot_1 * f1
    d_2 = d_mul * cot_2 * f2
    d_3 = d_mul * cot_3 * f3
    d_4 = d_mul * cot_4 * f4

    d_soft = d_1 + d_2 - d_3 - d_4
    d_hard = d_1 + d_2 + d_3 + d_4

    _mat_t1 = tf.stack([d_hard, tf.zeros_like(d_hard)], axis=-1)
    _mat_t2 = tf.stack([tf.zeros_like(d_soft), d_soft], axis=-1)
    mat_t = tf.stack([_mat_t1, _mat_t2], axis=-2)

    if flag:
        return mat_t

    spreading_factor = tf.sqrt(s_prime / (s*(s_prime + s)))
    spreading_factor = tf.complex(spreading_factor, tf.zeros_like(spreading_factor))

    mat_t *= -spreading_factor[..., None, None]
    # [num_targets, num_sources, max_num_paths, 1, 1]
    mask_ = expand_to_rank(mask, tf.rank(mat_t), axis=3)
    # [num_targets, num_sources, max_num_paths, 2]
    mat_t = tf.where(mask_, mat_t, tf.zeros_like(mat_t))

    if coef_ell is not None:
        all_a = tf.stack([a1, a2, a3, a4], axis=-1)
        min_a = tf.reduce_min(tf.abs(all_a), axis=-1)
        # new_coef_ell = 1 + 1 / (1 + min_a) * (tf.sqrt(coef_ell) - 1)
        # new_coef_ell = 1 + 1 / (1 + min_a) * (coef_ell - 1)

        a_0 = 0.1 #4.0
        new_coef_ell = 1 + (coef_ell - 1) * (1 / (1 + min_a / a_0))

    else:
        new_coef_ell = None

    return mat_t, new_coef_ell, (a1, a2, a3, a4)

def my_wd_compute_fields2(phi_prime, phi, beta_prime, ell, s_prime, s, n, mask, scene, skip_sf=False, skip_F=False):
    """
    """
    dtype = scene.dtype
    wavelength = scene.wavelength
    k = 2.*PI/wavelength

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

    d_mul = - tf.cast(tf.exp(-1j*PI/4.), dtype)/\
        tf.cast((2*n)*tf.sqrt(2*PI*k)*tf.math.sin(beta_prime), dtype)

    # [num_targets, num_sources, max_num_paths]
    cot_1 = tf.complex(cot_1, tf.zeros_like(cot_1))
    cot_2 = tf.complex(cot_2, tf.zeros_like(cot_2))
    cot_3 = tf.complex(cot_3, tf.zeros_like(cot_3))
    cot_4 = tf.complex(cot_4, tf.zeros_like(cot_4))

    a1 = a_p(phi_m, n) * k * ell
    a2 = a_m(phi_m, n) * k * ell
    a3 = a_p(phi_p, n) * k * ell
    a4 = a_m(phi_p, n) * k * ell

    spreading_factor = tf.sqrt(s_prime / (s*(s_prime + s)))
    spreading_factor = tf.complex(spreading_factor, tf.zeros_like(spreading_factor))

    if skip_F and not skip_sf:
        d1 = d_mul * cot_1 * spreading_factor
        d2 = d_mul * cot_2 * spreading_factor
        d3 = d_mul * cot_3 * spreading_factor
        d4 = d_mul * cot_4 * spreading_factor

        return (d1, d2, d3, d4), (a1, a2, a3, a4)

    f1 = transition_func(a1)
    f2 = transition_func(a2)
    f3 = transition_func(a3)
    f4 = transition_func(a4)

    d1 = d_mul * cot_1 * f1
    d2 = d_mul * cot_2 * f2
    d3 = d_mul * cot_3 * f3
    d4 = d_mul * cot_4 * f4

    if skip_sf:
        return (d1, d2, d3, d4), (a1, a2, a3, a4)

    d1 *= spreading_factor
    d2 *= spreading_factor
    d3 *= spreading_factor
    d4 *= spreading_factor

    return (d1, d2, d3, d4), (a1, a2, a3, a4)




######
def slope_t_gfi_full_approximated_capolino(b, a, table: DiffractionTables):
    x = tf.sqrt(b)
    y = tf.sqrt(a)

    initial_shape = tf.shape(x)

    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])
    b = tf.reshape(b, [-1])
    a = tf.reshape(a, [-1])

    table_idxs = tf.where(
        tf.logical_and(x < 3.0, y < 3.0)
    )[:, 0]

    approx_idxs = tf.where(
        tf.logical_or(x >= 3.0, y >= 3.0)
    )[:, 0]

    table_values = _slope_gfi_from_table(
        tf.gather(x, table_idxs),
        tf.gather(y, table_idxs),
        table,
    )

    approx_values = _slope_t_gfi_approx1(
        tf.gather(b, approx_idxs),
        tf.gather(a, approx_idxs),
    )

    dtype = table.dtype

    result = tf.zeros(tf.shape(x), dtype=dtype)
    result = tf.tensor_scatter_nd_update(result, table_idxs[:, None], table_values)
    result = tf.tensor_scatter_nd_update(result, approx_idxs[:, None], approx_values)

    return tf.reshape(result, initial_shape)

def slope_gfi_a0_limit(b):
    F = transition_func(b)
    b_complex = tf.cast(b, F.dtype)

    return (
        tf.complex(tf.zeros_like(b), 2.0*b) * (F - tf.cast(1.0/3.0, F.dtype))
        + tf.cast(8.0/3.0, F.dtype) * b_complex**2 * (1.0 - F)
    )



def _slope_gfi_from_table(x, y, table: DiffractionTables):
    """
    Bilinear interpolation of normalized slope T_GFI.

    Parameters
    ----------
    x : tf.Tensor
        sqrt(b), flattened.

    y : tf.Tensor
        sqrt(a), flattened.

    table : DiffractionTables
        Contains:
        - slope_gfi_table_tf
        - gfi_DIV
        - dtype

    Returns
    -------
    tf.Tensor
        Normalized T_GFI_slope(b, a).
    """
    table_tf = table.slope_gfi_table_tf
    div = tf.cast(table.gfi_DIV, x.dtype)
    dtype = table.dtype

    num_x = tf.shape(table_tf)[0]
    num_y = tf.shape(table_tf)[1]

    x_idx = tf.cast(tf.floor(x / div), tf.int32)
    y_idx = tf.cast(tf.floor(y / div), tf.int32)

    x_idx = tf.clip_by_value(x_idx, 0, num_x - 2)
    y_idx = tf.clip_by_value(y_idx, 0, num_y - 2)

    idx_00 = tf.stack([x_idx, y_idx], axis=-1)
    idx_10 = tf.stack([x_idx + 1, y_idx], axis=-1)
    idx_01 = tf.stack([x_idx, y_idx + 1], axis=-1)
    idx_11 = tf.stack([x_idx + 1, y_idx + 1], axis=-1)

    table_00 = tf.gather_nd(table_tf, idx_00)
    table_10 = tf.gather_nd(table_tf, idx_10)
    table_01 = tf.gather_nd(table_tf, idx_01)
    table_11 = tf.gather_nd(table_tf, idx_11)

    x1 = tf.cast(x_idx, x.dtype) * div
    y1 = tf.cast(y_idx, y.dtype) * div

    wx = (x - x1) / div
    wy = (y - y1) / div

    wx = tf.cast(wx, dtype)
    wy = tf.cast(wy, dtype)

    return (
        table_00 * (1.0 - wx) * (1.0 - wy)
        + table_10 * wx * (1.0 - wy)
        + table_01 * (1.0 - wx) * wy
        + table_11 * wx * wy
    )

def _slope_t_gfi_approx1(b, a):
    x = tf.sqrt(b)

    I0 = fresnel_tail_I0(x)
    # denom = a + b
    denom = tf.maximum(a + b, tf.cast(1e-8, a.dtype))

    exp_jb = tf.exp(tf.complex(tf.zeros_like(b), b))
    exp_mjb = tf.exp(tf.complex(tf.zeros_like(b), -b))

    correction = (
        tf.complex(tf.zeros_like(x), -x / 2.0) * exp_mjb
        + tf.complex(-b, -0.5 * tf.ones_like(b)) * I0
    )

    bracket = (
        I0
        - 2.0
        * correction
        / tf.complex(denom, tf.zeros_like(denom))
    )

    prefactor = tf.complex(
        tf.zeros_like(x),
        2.0 * x,
    )

    return prefactor * exp_jb * bracket

#################################################

#######
def wd_numerical_derivative(phi_prime, phi, beta_prime, r1, r2, n, scene, eps):
    ell = r1*r2 / (r1+r2) * tf.sin(beta_prime)**2

    (d1_p, d2_p, d3_p, d4_p), _ = my_wd_compute_fields2(phi_prime+eps, phi, beta_prime, 
                          ell, r1, r2, n, None, scene, skip_sf=True)

    (d1_m, d2_m, d3_m, d4_m), _ = my_wd_compute_fields2(phi_prime-eps, phi, beta_prime, 
                              ell, r1, r2, n, None, scene, skip_sf=True)

    denom = tf.cast(2*eps * tf.sin(beta_prime), d1_p.dtype)
    
    res1 = (d1_p - d1_m) / denom
    res2 = (d2_p - d2_m) / denom
    res3 = (d3_p - d3_m) / denom
    res4 = (d4_p - d4_m) / denom

    return res1, res2, res3, res4

def vd_numerical_derivative(vd, phi_prime, phi, beta_prime, beta, r1, r2, n, mask_wedges_in_vertex, add_to_r1, eps, b_override=None):
    # only phi_prime derivative
    # ell = r1*r2 / (r1+r2) * tf.sin(phi_prime)**2

    (d1_p, d2_p, d3_p, d4_p), param_a1, param_b1 = vd.simple_compute_fields(phi_prime+eps, phi, beta_prime, beta, r1, r2, n, mask_wedges_in_vertex, 
                              add_to_r1=add_to_r1, skip_sf=True, b_override=b_override)

    (d1_m, d2_m, d3_m, d4_m), param_a2, param_b2 = vd.simple_compute_fields(phi_prime-eps, phi, beta_prime, beta, r1, r2, n, mask_wedges_in_vertex, 
                                  add_to_r1=add_to_r1, skip_sf=True, b_override=b_override)
    
    denom = -tf.cast(2*eps * tf.sin(beta_prime), d1_p.dtype)

    res1 = (d1_p - d1_m) / denom
    res2 = (d2_p - d2_m) / denom
    res3 = (d3_p - d3_m) / denom
    res4 = (d4_p - d4_m) / denom

    return res1, res2, res3, res4

def slope_transition_func(x):
    return tf.complex(tf.zeros_like(x), 2*x) * (1.0 - transition_func(x))

def slope_diffr_coef(phi_prime, phi, beta_prime, ell, l, n, mask, scene, is_der_prime=True, return_a=False):
    # l is slope distance
    dtype = scene.dtype
    wavelength = scene.wavelength
    k = 2.*PI/wavelength

    # Compute D_1, D_2, D_3, D_4
    # [num_targets, num_sources, max_num_paths]
    phi_m = phi - phi_prime
    phi_p = phi + phi_prime

    # [num_targets, num_sources, max_num_paths]
    csc1 = csc((PI + phi_m)/(2*n))**2
    csc2 = csc((PI - phi_m)/(2*n))**2
    csc3 = csc((PI + phi_p)/(2*n))**2
    csc4 = csc((PI - phi_p)/(2*n))**2

    def n_p(beta, n):
        return tf.math.round((beta + PI)/(2.*n*PI))

    def n_m(beta, n):
        return tf.math.round((beta - PI)/(2.*n*PI))

    def a_p(beta, n):
        return 2*tf.cos((2.*n*PI*n_p(beta, n)-beta)/2.)**2

    def a_m(beta, n):
        return 2*tf.cos((2.*n*PI*n_m(beta, n)-beta)/2.)**2

    d_mul = tf.cast(tf.exp(-1j*PI/4.), dtype) / tf.cast(4*n*n*tf.sqrt(2*PI*k)*tf.math.sin(beta_prime)**2, dtype)

    # [num_targets, num_sources, max_num_paths]
    csc1 = tf.complex(csc1, tf.zeros_like(csc1))
    csc2 = tf.complex(csc2, tf.zeros_like(csc2))
    csc3 = tf.complex(csc3, tf.zeros_like(csc3))
    csc4 = tf.complex(csc4, tf.zeros_like(csc4))

    # [num_targets, num_sources, max_num_paths]
    a1 = a_p(phi_m, n) * k * ell
    a2 = a_m(phi_m, n) * k * ell
    a3 = a_p(phi_p, n) * k * ell
    a4 = a_m(phi_p, n) * k * ell

    fs1 = slope_transition_func(a1)
    fs2 = slope_transition_func(a2)
    fs3 = slope_transition_func(a3)
    fs4 = slope_transition_func(a4)

    d1 = -d_mul * csc1 * fs1
    d2 = d_mul * csc2 * fs2
    d3 = d_mul * csc3 * fs3
    d4 = -d_mul * csc4 * fs4

    if is_der_prime:
        d_soft = d1 + d2 - d3 - d4
        d_hard = d1 + d2 + d3 + d4
    else:
        d_soft = -d1 - d2 - d3 - d4
        d_hard = -d1 - d2 + d3 + d4

    _mat_t1 = tf.stack([d_hard, tf.zeros_like(d_hard)], axis=-1)
    _mat_t2 = tf.stack([tf.zeros_like(d_soft), d_soft], axis=-1)
    mat_t = tf.stack([_mat_t1, _mat_t2], axis=-2)

    # [num_targets, num_sources, max_num_paths, 1, 1]
    mask_ = expand_to_rank(mask, 5, axis=3)
    # [num_targets, num_sources, max_num_paths, 2]
    mat_t = tf.where(mask_, mat_t, tf.zeros_like(mat_t))

    if return_a:
        _tmp_a = tf.stack([a1, a2, a3, a4], axis=-1)
        return mat_t, tf.reduce_min(_tmp_a, axis=-1)+1e-6

    return mat_t

def slope_diffr_coef_2(par_a, phi_prime, phi, beta_prime, n, scene, skip_F=False):
    """
    Compute the diffracted field coefficients for slope diffraction.

    par_a: [..., 4] parameters a1, a2, a3, a4
    phi_m: phi - phi_prime
    phi_p: phi + phi_prime
    is_der_prime: boolean indicating if the derivative with respect to phi_prime is taken
                    or to phi.
    """
    dtype = scene.dtype
    wavelength = scene.wavelength
    k = 2.*PI/wavelength

    # [num_targets, num_sources, max_num_paths]
    phi_m = phi - phi_prime
    phi_p = phi + phi_prime

    # [num_targets, num_sources, max_num_paths]
    csc1 = csc((PI + phi_m)/(2*n))**2
    csc2 = csc((PI - phi_m)/(2*n))**2
    csc3 = csc((PI + phi_p)/(2*n))**2
    csc4 = csc((PI - phi_p)/(2*n))**2

    d_mul = tf.cast(tf.exp(-1j*PI/4.), dtype) / tf.cast(4*n*n*tf.sqrt(2*PI*k)*tf.math.sin(beta_prime)**2, dtype)

    # [num_targets, num_sources, max_num_paths]
    csc1 = tf.complex(csc1, tf.zeros_like(csc1))
    csc2 = tf.complex(csc2, tf.zeros_like(csc2))
    csc3 = tf.complex(csc3, tf.zeros_like(csc3))
    csc4 = tf.complex(csc4, tf.zeros_like(csc4))

    d1 = -d_mul * csc1
    d2 = d_mul * csc2
    d3 = d_mul * csc3
    d4 = -d_mul * csc4

    if skip_F:
        return d1, d2, d3, d4

    # not skip_F, apply slope transition function
    fs = slope_transition_func(par_a)
    fs1, fs2, fs3, fs4 = tf.unstack(fs, axis=-1)

    d1 *= fs1
    d2 *= fs2
    d3 *= fs3
    d4 *= fs4

    return d1, d2, d3, d4


def slope_vertex_trig_func(phi, u, n):
    phi_n = phi / n
    u_n = u / n

    sin2 = tf.sin(0.5*phi_n)**2
    sinh2 = tf.sinh(0.5*u_n)**2

    numerator = sin2 - sinh2 * tf.cos(phi_n)
    denominator = 2.0 * n**2 * (sin2 + sinh2)**2

    return numerator / denominator

def slope_vertex_trig_func2(phi, u, n):
    phi_n = phi / n
    u_n = u / n

    u_n2 = u_n**2
    sinhc = tf.where(
        tf.abs(u_n) < 1e-4,
        1.0 + u_n2 / 6.0 + u_n2**2 / 120.0,
        tf.math.divide_no_nan(tf.sinh(u_n), u_n),
    )

    denominator = tf.sin(0.5 * phi_n)**2 + tf.sinh(0.5 * u_n)**2

    return sinhc / denominator


def vertex_B_odd(phi, u, n, complex_dtype=tf.complex64):
    real_dtype = complex_dtype.real_dtype

    phi = tf.cast(phi, real_dtype)
    u = tf.cast(u, real_dtype)
    n = tf.cast(n, real_dtype)

    phi_n = phi / n
    u_n = u / n
    u_n2 = tf.square(u_n)

    sinhc = tf.where(
        tf.abs(u_n) < tf.cast(1e-4, real_dtype),
        1.0 + u_n2 / 6.0 + tf.square(u_n2) / 120.0,
        tf.math.divide_no_nan(tf.sinh(u_n), u_n),
    )

    denominator = (
        tf.square(tf.sin(0.5 * phi_n))
        + tf.square(tf.sinh(0.5 * u_n))
    )

    amplitude = u * sinhc / (4.0 * tf.square(n) * denominator)

    return tf.complex(tf.zeros_like(amplitude), -amplitude)

def slope_vertex_diffr_coef(
    par_a,
    par_c,
    sign_c,
    phi_prime,
    phi,
    beta_prime,
    beta,
    n,
    scene,
    table,
    mask_wedges_in_vertex,
    skip_F=False,
):
    dtype = scene.dtype
    wavelength = scene.wavelength
    k = 2.0 * PI / wavelength

    eps_phi = 1e-2
    step_func1 = tf.math.greater(n * PI - phi + eps_phi, 0.0)
    step_func2 = tf.math.greater(n * PI - phi_prime + eps_phi, 0.0)
    
    step_func = tf.logical_and(step_func1, step_func2)

    step_func = tf.cast(step_func, dtype)

    # coef = coef_ * step_func

    phi_m = phi - phi_prime
    phi_p = phi + phi_prime

    Phi1 = PI + phi_m
    Phi2 = PI - phi_m
    Phi3 = PI + phi_p
    Phi4 = PI - phi_p

    eps = tf.cast(1e-8, beta.dtype)
    tan_beta = tf.maximum(tf.tan(beta / 2.0), eps)
    tan_beta_prime = tf.maximum(tf.tan(beta_prime / 2.0), eps)
    u = tf.math.log(tan_beta) - tf.math.log(tan_beta_prime)

    Bs1 = tf.cast(slope_vertex_trig_func2(Phi1, u, n), dtype)
    Bs2 = tf.cast(slope_vertex_trig_func2(Phi2, u, n), dtype)
    Bs3 = tf.cast(slope_vertex_trig_func2(Phi3, u, n), dtype)
    Bs4 = tf.cast(slope_vertex_trig_func2(Phi4, u, n), dtype)

    delta_cos = tf.cos(beta_prime) - tf.cos(beta)

    d_mul2 = step_func * tf.constant(1j, dtype=dtype) / tf.cast(
            8.0 * PI * k * n**2 * delta_cos,
            dtype,
        )

    
    d1 = -d_mul2 * Bs1
    d2 = d_mul2 * Bs2
    d3 = d_mul2 * Bs3
    d4 = -d_mul2 * Bs4

    d1 = clear_vd_coef(d1, mask_wedges_in_vertex, dtype)
    d2 = clear_vd_coef(d2, mask_wedges_in_vertex, dtype)
    d3 = clear_vd_coef(d3, mask_wedges_in_vertex, dtype)
    d4 = clear_vd_coef(d4, mask_wedges_in_vertex, dtype)

    return d1, d2, d3, d4

def slope_vertex_diffr_coef2(
    par_a,
    par_b,
    phi_prime,
    phi,
    beta_prime,
    beta,
    n,
    scene,
    table,
    mask_wedges_in_vertex,
):
    """
    Compute four slope vertex diffraction terms for derivative
    with respect to the incident azimuth phi_prime.

    Parameters
    ----------
    par_a : tf.Tensor
        Four pole optical distances [a1, a2, a3, a4].
        Shape: [..., 4].

    par_b : tf.Tensor
        Common vertex/end-point optical distance.
        Shape: [...].

    phi_prime, phi : tf.Tensor
        Incident and outgoing azimuth angles.

    beta_prime, beta : tf.Tensor
        Incident and outgoing elevation angles.

    n : tf.Tensor
        Wedge parameter.

    scene
        Sionna scene.

    table : DiffractionTables
        Contains the normalized slope GFI table.
    """
    dtype = scene.dtype
    wavelength = scene.wavelength
    k = 2.0 * PI / wavelength

    phi_m = phi - phi_prime
    phi_p = phi + phi_prime

    Phi1 = PI + phi_m
    Phi2 = PI - phi_m
    Phi3 = PI + phi_p
    Phi4 = PI - phi_p

    eps = tf.cast(1e-8, beta.dtype)

    tan_beta = tf.maximum(tf.tan(beta / 2.0), eps)
    tan_beta_prime = tf.maximum(tf.tan(beta_prime / 2.0), eps)

    u = (
        tf.math.log(tan_beta)
        - tf.math.log(tan_beta_prime)
    )

    Bs1 = slope_vertex_trig_func2(Phi1, u, n)
    Bs2 = slope_vertex_trig_func2(Phi2, u, n)
    Bs3 = slope_vertex_trig_func2(Phi3, u, n)
    Bs4 = slope_vertex_trig_func2(Phi4, u, n)

    fs = slope_t_gfi_full_approximated_capolino(par_b, par_a, table)
    fs1, fs2, fs3, fs4 = tf.unstack(fs, axis=-1)

    delta_cos = tf.cos(beta_prime) - tf.cos(beta)

    denominator = (
        4.0
        * k
        * PI
        * tf.sin(beta_prime)
        * delta_cos
    )

    # 1 / j = -j
    d_mul = tf.complex(
        tf.zeros_like(denominator),
        -1.0 / denominator,
    )

    d_mul = tf.cast(d_mul, dtype)

    Bs1 = tf.cast(Bs1, dtype)
    Bs2 = tf.cast(Bs2, dtype)
    Bs3 = tf.cast(Bs3, dtype)
    Bs4 = tf.cast(Bs4, dtype)


    # Therefore the resulting signs are [+1, -1, -1, +1].
    d1 = d_mul * Bs1 * fs1
    d2 = -d_mul * Bs2 * fs2
    d3 = -d_mul * Bs3 * fs3
    d4 = d_mul * Bs4 * fs4

    d1 = clear_vd_coef(d1, mask_wedges_in_vertex, dtype)
    d2 = clear_vd_coef(d2, mask_wedges_in_vertex, dtype)
    d3 = clear_vd_coef(d3, mask_wedges_in_vertex, dtype)
    d4 = clear_vd_coef(d4, mask_wedges_in_vertex, dtype)

    return d1, d2, d3, d4

def clear_vd_coef(d, mask_wedges_in_vertex, dtype):
    r""""
    d: [num_targets, num_sources, num_paths, ...]
    mask_wedges_in_vertex: [num_targets, num_sources, num_vertices, num_wedges_per_vertex, ...]
    """
    res = np.reshape(d, mask_wedges_in_vertex.shape)
    res = np.where(mask_wedges_in_vertex, res, np.zeros_like(res))
    nans_bool = tf.math.is_nan(tf.math.real(res))
    res = tf.where(nans_bool, tf.zeros_like(res, dtype=dtype), res)
    res = tf.reshape(res, d.shape)
    #res = tf.reduce_sum(res, axis=3)

    return res


#########################
# Tables

"""Two-region lookup for the shared GFI G(x, y), used by VD and EE."""
from pathlib import Path

class GFILookup:
    """Bilinear GFI interpolation; slopes may jump at cell and region boundaries.

    The first quadrant uses the positive one-sided limits on its axes. The
    joint origin is rejected because its value depends on approach direction.
    Tables approximate the generating formulas, not an exact quadrature oracle.
    """

    def __init__(self, filename, dtype=tf.complex64):
        self.dtype = tf.as_dtype(dtype)
        self.rdtype = self.dtype.real_dtype
        with np.load(Path(filename), allow_pickle=False) as data:
            if int(data['format_version']) != 1:
                raise ValueError('Unsupported GFI table format.')
            self.region = tf.constant(float(data['region']), self.rdtype)
            for name in ('r_nodes', 'theta_nodes', 'x_nodes', 'y_nodes'):
                nodes = data[name]
                if len(nodes) < 2 or not np.all(np.diff(nodes) > 0):
                    raise ValueError(f'{name} must be strictly increasing.')
                setattr(self, name, tf.constant(nodes, self.rdtype))
            self.polar = tf.constant(data['polar'], self.dtype)
            self.cartesian = tf.constant(data['cartesian'], self.dtype)
        if tuple(self.polar.shape) != (len(self.r_nodes), len(self.theta_nodes)):
            raise ValueError('Invalid polar table shape.')
        if tuple(self.cartesian.shape) != (len(self.x_nodes), len(self.y_nodes)):
            raise ValueError('Invalid Cartesian table shape.')
        self.theta_step = self.theta_nodes[1] - self.theta_nodes[0]

    @staticmethod
    def _find_interval(x, nodes):
        """Find intervals on strictly increasing nonuniform nodes."""
        i = tf.searchsorted(nodes, x, side='right') - 1
        i = tf.clip_by_value(i, 0, tf.shape(nodes)[0] - 2)
        x0, x1 = tf.gather(nodes, i), tf.gather(nodes, i + 1)
        return i, tf.clip_by_value((x - x0) / (x1 - x0), 0.0, 1.0)

    @staticmethod
    def _bilinear(values, i, j, u, v):
        """Interpolate complex values; weights retain coordinate gradients."""
        z00 = tf.gather_nd(values, tf.stack([i, j], axis=-1))
        z10 = tf.gather_nd(values, tf.stack([i + 1, j], axis=-1))
        z01 = tf.gather_nd(values, tf.stack([i, j + 1], axis=-1))
        z11 = tf.gather_nd(values, tf.stack([i + 1, j + 1], axis=-1))
        u, v = tf.cast(u, values.dtype), tf.cast(v, values.dtype)
        return (1-u)*((1-v)*z00 + v*z01) + u*((1-v)*z10 + v*z11)

    def __call__(self, x, y):
        """Evaluate G for broadcast-compatible real tensors.

        Signed arguments use odd extension in each variable. Exact zeros select
        the positive side. No extrapolation or analytical fallback is performed.
        At the joint origin, callers must handle the directional physical limit.
        """
        x, y = tf.cast(x, self.rdtype), tf.cast(y, self.rdtype)
        shape = tf.broadcast_dynamic_shape(tf.shape(x), tf.shape(y))
        x = tf.reshape(tf.broadcast_to(x, shape), [-1])
        y = tf.reshape(tf.broadcast_to(y, shape), [-1])
        side = tf.where(x < 0, -tf.ones_like(x), tf.ones_like(x))
        side *= tf.where(y < 0, -tf.ones_like(y), tf.ones_like(y))
        x, y = tf.abs(x), tf.abs(y)
        tf.debugging.assert_all_finite(x, 'GFI x contains NaN/Inf.')
        tf.debugging.assert_all_finite(y, 'GFI y contains NaN/Inf.')
        tf.debugging.assert_positive(tf.maximum(x, y), message='GFI joint origin needs a directional limit.')
        tf.debugging.assert_less_equal(x, self.x_nodes[-1], message='GFI x exceeds table range.')
        tf.debugging.assert_less_equal(y, self.y_nodes[-1], message='GFI y exceeds table range.')

        small = (x <= self.region) & (y <= self.region)
        polar_idx, cart_idx = tf.where(small)[:, 0], tf.where(~small)[:, 0]
        xp, yp = tf.gather(x, polar_idx), tf.gather(y, polar_idx)
        r = tf.sqrt(tf.square(xp) + tf.square(yp))
        theta = tf.math.atan2(yp, xp)
        i, u = self._find_interval(r, self.r_nodes)
        j = tf.cast(tf.floor(theta / self.theta_step), tf.int32)
        j = tf.clip_by_value(j, 0, tf.shape(self.theta_nodes)[0] - 2)
        v = tf.clip_by_value((theta - tf.gather(self.theta_nodes, j)) / self.theta_step, 0.0, 1.0)
        polar_values = self._bilinear(self.polar, i, j, u, v)

        xc, yc = tf.gather(x, cart_idx), tf.gather(y, cart_idx)
        i, u = self._find_interval(xc, self.x_nodes)
        j, v = self._find_interval(yc, self.y_nodes)
        cart_values = self._bilinear(self.cartesian, i, j, u, v)
        result = tf.zeros(tf.shape(x), self.dtype)
        result = tf.tensor_scatter_nd_update(result, polar_idx[:, None], polar_values)
        result = tf.tensor_scatter_nd_update(result, cart_idx[:, None], cart_values)
        return tf.reshape(result * tf.cast(side, self.dtype), shape)


class LinearTEVTF:
    def __init__(self, table_path, transition_func, slope_transition_func, case="even", real_dtype=tf.float32):
        data = np.load(table_path)

        self.real_dtype = real_dtype
        self.complex_dtype = tf.complex64 if real_dtype == tf.float32 else tf.complex128
        self.case = case
        self.transition_func = transition_func
        self.slope_transition_func = slope_transition_func

        self.alpha_nodes = tf.constant(data["alpha_nodes"], dtype=real_dtype)
        self.beta_nodes = tf.constant(data["beta_nodes"], dtype=real_dtype)
        self.chi_nodes = tf.constant(data["chi_nodes"], dtype=real_dtype)
        self.w_nodes = tf.constant(data["w_nodes"], dtype=real_dtype)
        self.values = tf.constant(data["values"], dtype=self.complex_dtype)

        self.alpha_max = self.alpha_nodes[-1]
        self.beta_min = self.beta_nodes[0]
        self.beta_max = self.beta_nodes[-1]
        self.chi_max = self.chi_nodes[-1]
        self.w_min = self.w_nodes[0]
        self.w_max = self.w_nodes[-1]

    @staticmethod
    def _find_interval(x, nodes):
        """Return the interval index and linear interpolation coordinate."""
        num_nodes = tf.shape(nodes)[0]
        i = tf.searchsorted(nodes, x, side="right") - 1
        i = tf.clip_by_value(i, 0, num_nodes - 2)

        x0 = tf.gather(nodes, i)
        x1 = tf.gather(nodes, i + 1)
        t = tf.clip_by_value((x - x0) / (x1 - x0), 0.0, 1.0)
        return i, t

    def _interpolate(self, alpha, beta, chi, w):
        alpha = tf.clip_by_value(alpha, self.alpha_nodes[0], self.alpha_max)
        beta = tf.clip_by_value(beta, self.beta_min, self.beta_max)
        chi = tf.clip_by_value(chi, self.chi_nodes[0], self.chi_max)
        w = tf.clip_by_value(w, self.w_min, self.w_max)

        ia, ta = self._find_interval(alpha, self.alpha_nodes)
        ib, tb = self._find_interval(beta, self.beta_nodes)
        ic, tc = self._find_interval(chi, self.chi_nodes)
        iw, tw = self._find_interval(w, self.w_nodes)
        result = tf.zeros_like(tf.cast(alpha, self.complex_dtype))

        # Multilinear interpolation over 16 corners.
        for da in range(2):
            wa = ta if da else 1.0 - ta
            for db in range(2):
                wb = tb if db else 1.0 - tb
                for dc in range(2):
                    wc = tc if dc else 1.0 - tc
                    for dw in range(2):
                        ww = tw if dw else 1.0 - tw
                        indices = tf.stack([ia + da, ib + db, ic + dc, iw + dw], axis=-1)
                        corner_value = tf.gather_nd(self.values, indices)
                        weight = wa * wb * wc * ww
                        result += tf.cast(weight, self.complex_dtype) * corner_value

        return result

    @staticmethod
    def _scatter_result(result, mask, updates):
        return tf.tensor_scatter_nd_update(result, tf.where(mask), updates)

    def __call__(self, alpha, beta, chi, w):
        alpha = tf.cast(alpha, self.real_dtype)
        beta = tf.cast(beta, self.real_dtype)
        chi = tf.cast(chi, self.real_dtype)
        w = tf.cast(w, self.real_dtype)

        broadcast_shape = tf.broadcast_dynamic_shape(tf.shape(alpha), tf.shape(beta))
        broadcast_shape = tf.broadcast_dynamic_shape(broadcast_shape, tf.shape(chi))
        broadcast_shape = tf.broadcast_dynamic_shape(broadcast_shape, tf.shape(w))

        alpha = tf.broadcast_to(alpha, broadcast_shape)
        beta = tf.broadcast_to(beta, broadcast_shape)
        chi = tf.broadcast_to(chi, broadcast_shape)
        w = tf.broadcast_to(w, broadcast_shape)
        output_shape = tf.shape(alpha)

        alpha = tf.reshape(alpha, [-1])
        beta = tf.reshape(beta, [-1])
        chi = tf.reshape(chi, [-1])
        w = tf.reshape(w, [-1])
        result = tf.zeros(tf.shape(alpha), dtype=self.complex_dtype)

        # Priority: large chi, large beta, interpolation.
        mask_large_chi = chi > self.chi_max
        mask_large_beta = ~mask_large_chi & (beta > self.beta_max)
        mask_grid = ~mask_large_chi & ~mask_large_beta

        def update_large_chi():
            selected_alpha = tf.boolean_mask(alpha, mask_large_chi)
            a = tf.square(selected_alpha)

            if self.case == "even":
                updates = tf.cast(self.transition_func(a), self.complex_dtype)
            else:
                updates = tf.cast(-self.slope_transition_func(a), self.complex_dtype)

            return self._scatter_result(result, mask_large_chi, updates)

        result = tf.cond(tf.reduce_any(mask_large_chi), update_large_chi, lambda: result)

        def update_large_beta():
            selected_alpha = tf.boolean_mask(alpha, mask_large_beta)
            selected_chi = tf.boolean_mask(chi, mask_large_beta)
            a = tf.square(selected_alpha)
            c = tf.square(selected_chi)

            if self.case == "even":
                F_a = tf.cast(self.transition_func(a), self.complex_dtype)
            elif self.case == "odd":
                F_a = tf.cast(self.slope_transition_func(a), self.complex_dtype)

            F_c = tf.cast(self.transition_func(c), self.complex_dtype)
            updates = F_a * F_c
            return self._scatter_result(result, mask_large_beta, updates)

        result = tf.cond(tf.reduce_any(mask_large_beta), update_large_beta, lambda: result)

        def update_grid():
            selected_alpha = tf.boolean_mask(alpha, mask_grid)
            selected_beta = tf.boolean_mask(beta, mask_grid)
            selected_chi = tf.boolean_mask(chi, mask_grid)
            selected_w = tf.boolean_mask(w, mask_grid)

            updates = self._interpolate(selected_alpha, selected_beta, selected_chi, selected_w)
            return self._scatter_result(result, mask_grid, updates)

        result = tf.cond(tf.reduce_any(mask_grid), update_grid, lambda: result)
        return tf.reshape(result, output_shape)

###############################################################

def get_wd_points(u_t, u_r, e_hat, origins, wedges_length, wedge_idxs):
    a = dot(u_t, e_hat)
    b = dot(u_r, e_hat)
    c = dot(u_t, u_t)
    d = dot(u_r, u_r)

    rdtype = e_hat.dtype

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
    t_min = tf.zeros_like(cond_1, rdtype)
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

    return inter_point, new_wedge_idxs, mask_