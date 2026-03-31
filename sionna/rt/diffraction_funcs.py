from __future__ import annotations

import os
from importlib_resources import files

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
from . import data


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

def t_gfi_full_approximated_capolino(b, a, is_table=False):  # x, y
    x = tf.math.sqrt(b)
    y = tf.math.sqrt(a)

    func_G = func_G_approximated_capolino(x, y, is_table) 

    res = func_G * tf.complex(tf.cast(0.0, x.dtype), 4*PI * tf.math.sqrt(b / a) * (a + b))

    return res

def func_G_approximated_capolino(x, y, is_table=False):
    initial_shape = x.shape
    x = tf.reshape(x, -1)
    y = tf.reshape(y, -1)
    gfi_sign = sign(x) * sign(y)
    x = tf.math.abs(x)
    y = tf.math.abs(y)

    if is_table:
        _func_G = _func_G_table(x, y)
    else:
        _func_G = _func_G_approximated_capolino(x, y)  # faster non-table impl

    res = tf.complex(gfi_sign, tf.cast(0.0, x.dtype)) * _func_G
    res = tf.reshape(res, initial_shape)

    return res

def _func_G_table(x, y):
    # table is for x < 3.0 and y < 3.0
    file_table = str(files(data).joinpath("gfi.npy"))
    DIV = 0.01

    table_tf = np.load(file_table)

    if x.dtype == tf.float32:
        dtype = tf.complex64
        table_tf = tf.convert_to_tensor(table_tf, dtype=tf.complex64)
    else:
        dtype = tf.complex128
        table_tf = tf.convert_to_tensor(table_tf, dtype=tf.complex128)

    cond_approx1_idxs = tf.where(tf.math.logical_or(x >= 3.0, y >= 3.0))[:, 0]
    table_cond_idxs = tf.where(tf.math.logical_and(x < 3.0, y < 3.0))[:, 0]

    def _from_table(x_table, y_table):
        x_idx = tf.cast(tf.math.floor(x_table / DIV), tf.int32)
        y_idx = tf.cast(tf.math.floor(y_table / DIV), tf.int32)
        ##
        x_idx = tf.where(x_idx < 0, tf.zeros_like(x_idx), x_idx)
        x_idx = tf.where(x_idx >= table_tf.shape[0]-2, tf.zeros_like(x_idx), x_idx)
        y_idx = tf.where(y_idx < 0, tf.zeros_like(y_idx), y_idx)
        y_idx = tf.where(y_idx >= table_tf.shape[1]-2, tf.zeros_like(y_idx), y_idx)
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

    res = tf.zeros([x.shape[0]], dtype=dtype)
    res = tf.tensor_scatter_nd_update(res, tf.reshape(cond_approx1_idxs, (-1, 1)), res_approx1)
    res = tf.tensor_scatter_nd_update(res, tf.reshape(table_cond_idxs, (-1, 1)), res_table_cond)
    
    return res

def _func_G_approximated_capolino(x, y):
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
    # res3 = _func_G_reg3(tf.gather(x_complex, cond3_idxs), tf.gather(y_complex, cond3_idxs), tf.gather(x, cond3_idxs), tf.gather(y, cond3_idxs))
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
    imag_i = tf.complex(tf.cast(0.0, x_real.dtype), tf.cast(1.0, x_real.dtype))
    sqrt_pi_complex = tf.complex(tf.math.sqrt(tf.cast(PI, dtype=x_real.dtype)), tf.cast(0.0, x_real.dtype))

    tmp1 = 0.5 * tf.math.exp(imag_i * x*x)
    fr_integral = fresnel_integral_capolino(y, y_real) / sqrt_pi_complex
    tmp2 = tf.math.exp(imag_i * PI / 4) * fr_integral 
    tmp3 = ((1 + imag_i * y*y) * tf.math.atan(x / y) 
            - imag_i * x * y) / PI

    return tmp1 * (tmp2 - tmp3)

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

    return tf.math.exp(imag_i * ksi**2) * fresnel_integral(ksi_real)

def fresnel_integral(y_real):
    r"""intergral from y to +inf, without any coefs"""

    sqrt_pi_div_2 = tf.math.sqrt(tf.cast(PI, y_real.dtype) / 2.0)
    sqrt_pi_div_8 = tf.math.sqrt(tf.cast(PI, y_real.dtype) / 8.0)
    
    s = sqrt_pi_div_2 * tf.math.special.fresnel_sin(y_real / sqrt_pi_div_2)
    c = sqrt_pi_div_2 * tf.math.special.fresnel_cos(y_real / sqrt_pi_div_2)

    c = sqrt_pi_div_8 - c
    s = s - sqrt_pi_div_8

    return tf.complex(c, s)

####################################################
## GFI approximation for x > 3.0 or y > 3.0

def tf_fresnel_tail_I0(x):
    """
    I0(x) = ∫_x^∞ exp(-j t^2) dt
    for real x >= 0
    """

    u = tf.cast(tf.sqrt(2.0 / PI), x.dtype) * x
    C = tf.math.special.fresnel_cos(u)
    S = tf.math.special.fresnel_sin(u)

    scale = tf.cast(tf.sqrt(PI / 2.0), x.dtype)
    real_part = scale * (0.5 - C)
    imag_part = -scale * (0.5 - S)

    return tf.complex(real_part, imag_part)

def G_approx1(x, y):
    """
    First-order approximation
    """

    I0 = tf_fresnel_tail_I0(x)
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

def transition_func(x):
    """F(x) Eq.(88) in [ITUR_P526]
    """
    sqrt_x = tf.sqrt(x)
    sqrt_pi_2 = tf.cast(tf.sqrt(PI/2.), x.dtype)

    # Fresnel integral
    arg = sqrt_x/sqrt_pi_2
    s = tf.math.special.fresnel_sin(arg)
    c = tf.math.special.fresnel_cos(arg)
    f = tf.complex(s, c)

    zero = tf.cast(0, x.dtype)
    one = tf.cast(1, x.dtype)
    two = tf.cast(2, f.dtype)
    factor = tf.complex(sqrt_pi_2*sqrt_x, zero)
    factor = factor*tf.exp(tf.complex(zero, x))
    res =  tf.complex(one, one) - two*f

    return factor * res

def invert_angles(phi, n):
    return 2*PI - (2-n) * PI - phi

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


    def f(x):
        """F(x) Eq.(88) in [ITUR_P526]
        """
        sqrt_x = tf.sqrt(x)
        sqrt_pi_2 = tf.cast(tf.sqrt(PI/2.), x.dtype)

        # Fresnel integral
        arg = sqrt_x/sqrt_pi_2
        s = tf.math.special.fresnel_sin(arg)
        c = tf.math.special.fresnel_cos(arg)
        f = tf.complex(s, c)

        zero = tf.cast(0, x.dtype)
        one = tf.cast(1, x.dtype)
        two = tf.cast(2, f.dtype)
        factor = tf.complex(sqrt_pi_2*sqrt_x, zero)
        factor = factor*tf.exp(tf.complex(zero, x))
        res =  tf.complex(one, one) - two*f

        return factor* res

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

    a1 = a_p(phi_m, n)
    a2 = a_m(phi_m, n)
    a3 = a_p(phi_p, n)
    a4 = a_m(phi_p, n)

    f1 = f(k*ell*a1)
    f2 = f(k*ell*a2)
    f3 = f(k*ell*a3)
    f4 = f(k*ell*a4)

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
        all_cot = tf.stack([cot_1, cot_2, cot_3, cot_4], axis=-1)
        max_cot = tf.reduce_max(tf.abs(all_cot), axis=-1)
        new_coef_ell = get_coef_ell(max_cot, coef_ell)

    else:
        new_coef_ell = None

    return mat_t, new_coef_ell, (a1, a2, a3, a4)

def slope_transition_func(x):
    return tf.complex(tf.zeros_like(x), 2*x) * (1.0 - transition_func(x))

def get_coef_ell(cot_val, coef_ell, min_val=2.0, max_val=400.0):
    # this worked for wedge-wedge
    abs_cot = tf.abs(cot_val)

    left_cot = tf.cast(min_val, dtype=abs_cot.dtype)
    right_cot = tf.cast(max_val, dtype=abs_cot.dtype)

    cond1 = abs_cot < left_cot
    cond2 = abs_cot > right_cot
    adjusted_coef_ell = (abs_cot - left_cot) / (right_cot - left_cot) * (coef_ell - 1.0) + 1.0
    return tf.where(cond1, tf.ones_like(coef_ell), 
                    tf.where(cond2, coef_ell, adjusted_coef_ell))


###############################################################

def get_wd_points(u_t, u_r, e_hat, origins, wedges_length, wedge_idxs):
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

    return inter_point, new_wedge_idxs, mask_