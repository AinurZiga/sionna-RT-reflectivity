import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv, hankel2, spherical_jn, assoc_legendre_p_all, h2vp, jvp


def cylinder_te_inc_scat_total(theta_arr, cylinder_radius, circle_radius, k, terms=512):
    """
    TE (Hz) scattering from a PEC circular cylinder.
    """

    a = cylinder_radius
    r_obs = circle_radius

    x = k * a             # ka
    rho = k * r_obs        # k r_obs

    u_inc = np.exp(-1j * k * r_obs * np.cos(theta_arr + np.pi))

    u_scat = np.zeros_like(theta_arr, dtype=np.complex128)

    for n in range(-terms // 2, terms // 2 + 1):
        H_rho = hankel2(n, rho)

        Jp_x = jvp(n, x, 1) 
        Hp_x = h2vp(n, x, 1)

        a_n = - Jp_x / Hp_x

        phase = (1j ** (-n)) * np.exp(1j * n * (theta_arr + np.pi))
        u_scat += phase * (a_n * H_rho)

    u_tot = u_inc + u_scat
    return u_inc, u_scat, u_tot

def cylinder_tm_inc_scat_total(theta_arr, cylinder_radius, circle_radius, k, terms=512):
    """
    TM (Ez) scattering from a PEC circular cylinder.
    """
    theta_arr = np.array(theta_arr, dtype=float)

    a = cylinder_radius
    r_obs = circle_radius

    x = k * a
    rho = k * r_obs

    E_inc = np.exp(-1j * k * r_obs * np.cos(theta_arr + np.pi))

    E_scat = np.zeros_like(theta_arr, dtype=np.complex128)

    for n in range(-terms // 2, terms // 2 + 1):
        J_x = jv(n, x)
        H_x = hankel2(n, x)
        H_rho = hankel2(n, rho)

        a_n = -J_x / H_x

        phase = (1j ** (-n)) * np.exp(1j * n * (theta_arr + np.pi))
        E_scat += phase * (a_n * H_rho)

    E_tot = E_inc + E_scat
    return E_inc, E_scat, E_tot

def sphere_tm(phi_arr, theta_arr, sphere_radius, circle_radius, k):
    # angles in radians
    theta_arr = np.array(theta_arr)
    beta_a = k * sphere_radius
    beta_r = k * circle_radius
    num_terms = 100

    E_r_i = np.sin(theta_arr) * np.cos(phi_arr) * np.exp(-1j*beta_r*np.cos(theta_arr))
    E_theta_i = np.cos(theta_arr) * np.cos(phi_arr) * np.exp(-1j*beta_r*np.cos(theta_arr))
    E_phi_i = -np.sin(phi_arr) * np.exp(-1j*beta_r*np.cos(theta_arr))

    E_r_s = np.zeros(len(phi_arr), dtype=np.complex128)
    E_theta_s = np.zeros(len(phi_arr), dtype=np.complex128)
    E_phi_s = np.zeros(len(phi_arr), dtype=np.complex128)

    for i in range(len(phi_arr)):
        phi = phi_arr[i]
        theta = theta_arr[i]
        
        # Pmn_z, Pmn_d_z = lpmn(1, num_terms, np.cos(theta))  # both [2, num_terms + 1]
        Pmn_z, Pmn_d_z = assoc_legendre_p_all(num_terms, 1, np.cos(theta))  # both [2, num_terms + 1]

        e_r_s_coef = 0.0 + 0.0j 
        e_theta_s_coef = 0.0 + 0.0j 
        e_phi_s_coef = 0.0 + 0.0j 
        
        for n in range(1, num_terms + 1):
            a_n = 1j**(-n) * (2*n + 1) / (n*(n + 1))
            legandre = Pmn_z[1, n]     
            legandre_derivative = Pmn_d_z[1, n]

            b_n = -a_n * _get_spherical_bessel_derivative1(n, beta_a) / _get_spherical_hankel2_derivative1(n, beta_a)
            c_n = -a_n * _get_spherical_bessel(n, beta_a) / _get_spherical_hankel2(n, beta_a)

            hankel_2_beta_r = _get_spherical_hankel2(n, beta_r)
            hankel_2_beta_r_der1 = _get_spherical_hankel2_derivative1(n, beta_r)
            hankel_2_beta_r_der2 = _get_spherical_hankel2_derivative2(n, beta_r)

            e_r_s_coef += b_n * (hankel_2_beta_r_der2 + hankel_2_beta_r) * legandre
            e_theta_s_coef += 1j * b_n * hankel_2_beta_r_der1 * np.sin(theta) * legandre_derivative \
                - (c_n * hankel_2_beta_r * legandre / np.sin(theta))
            e_phi_s_coef += 1j * b_n * hankel_2_beta_r_der1 * legandre / np.sin(theta) \
                - (c_n * hankel_2_beta_r * np.sin(theta) * legandre_derivative)
            
        E_r_s[i] = -1j * np.cos(phi) * e_r_s_coef
        E_theta_s[i] = 1 / beta_r * np.cos(phi) * e_theta_s_coef
        E_phi_s[i] = 1 / beta_r * np.sin(phi) * e_phi_s_coef

    E_s = np.array([E_r_s, E_theta_s, E_phi_s])
    E_t = np.array([E_r_i + E_r_s, E_theta_i + E_theta_s, E_phi_i + E_phi_s])

    return np.linalg.norm(E_t, axis=0), np.linalg.norm(E_s, axis=0)


def sphere_rcs_tm(phi_arr, theta_arr, sphere_radius, circle_radius, k):
    # angles in radians
    theta_arr = np.array(theta_arr)
    beta_a = k * sphere_radius
    num_terms = 100

    rcs = np.zeros(len(phi_arr), dtype=np.float64)

    for i in range(len(phi_arr)):
        phi = phi_arr[i]
        theta = theta_arr[i]
        
        # Pmn_z, Pmn_d_z = lpmn(1, num_terms, np.cos(theta))  # both [2, num_terms + 1]
        Pmn_z, Pmn_d_z = assoc_legendre_p_all(num_terms, 1, np.cos(theta))  # both [2, num_terms + 1]

        A_theta_tmp = 0.0 + 0.0j
        A_phi_tmp = 0.0 + 0.0j
        
        for n in range(1, num_terms + 1):
            a_n = 1j**(-n) * (2*n + 1) / (n*(n + 1))
            legandre = Pmn_z[1, n]     
            legandre_derivative = Pmn_d_z[1, n]

            b_n = -a_n * _get_spherical_bessel_derivative1(n, beta_a) / _get_spherical_hankel2_derivative1(n, beta_a)
            c_n = -a_n * _get_spherical_bessel(n, beta_a) / _get_spherical_hankel2(n, beta_a)

            A_theta = b_n * np.sin(theta) * legandre_derivative - c_n * legandre / np.sin(theta)
            A_phi = b_n * legandre / np.sin(theta) - c_n * np.sin(theta) * legandre_derivative

            A_theta_tmp += (1j)**(n) * A_theta
            A_phi_tmp += (1j)**(n) * A_phi

        rcs[i] = (np.cos(phi)**2 * np.abs(A_theta_tmp)**2 + np.sin(phi)**2 * np.abs(A_phi_tmp)**2) * (4*np.pi / k**2)

    return rcs


def _get_spherical_bessel(n, x):
    return x * spherical_jn(n, x)
    #return spherical_jn(n, x)

def _get_spherical_bessel_derivative1(n, x):
    return spherical_jn(n, x) + x * spherical_jn(n, x, derivative=True)

def _get_spherical_hankel2(n, x):
    return np.sqrt(np.pi * x / 2) * hankel2(n + 0.5, x)

def _get_spherical_hankel2_derivative1(n, x):
    term1_coef = 0.5 * np.sqrt(np.pi / 2 / x)
    term1_hankel = hankel2(n + 0.5, x)
    term2_coef = np.sqrt(np.pi * x / 2)
    term2_hankel = h2vp(n + 0.5, x, 1)

    return term1_coef * term1_hankel + term2_coef * term2_hankel

def _get_spherical_hankel2_derivative2(n, x):
    term1_coef = -0.25 * np.sqrt(np.pi / 2) * x**(-1.5)
    term1_hankel = hankel2(n + 0.5, x)
    term2_coef = np.sqrt(np.pi / 2) * np.sqrt(1/x)
    term2_hankel = h2vp(n + 0.5, x, 1)
    term3_coef = np.sqrt(np.pi * x / 2)
    term3_hankel = h2vp(n + 0.5, x, 2)

    return term1_coef * term1_hankel + term2_coef * term2_hankel + term3_coef * term3_hankel