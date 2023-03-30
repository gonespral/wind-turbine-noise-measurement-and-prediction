'''File to house the LBL-VS calculation procedure'''
import math
import numpy as np



def st_prime_one(Reynolds):
    st_prime = 0
    if Reynolds <= 1.3*10**5:
        st_prime = 0.18
    elif 1.3 * 10**5 < Reynolds <= 4.0 * 10**5:
        st_prime = 0.001756 * Reynolds**0.3931
    elif Reynolds > 4.0 * 10**5:
        st_prime = 0.28
    return st_prime

def st_peak(alpha_star, st_prime):
    st_prime_peak = st_prime * 10**(-0.04*alpha_star)
    return  st_prime_peak

def G_1(f, delta, U, st_prime_peak):
    st = f * delta / U
    e = st / st_prime_peak
    Spectral_shape = 0
    if e <= 0.5974:
        Spectral_shape = 39.8*np.log10(e) - 11.12
    elif 0.5974 < e <= 0.8545:
        Spectral_shape = 98.409*np.log10(e) + 2
    elif 0.8545 < e <= 1.17:
        Spectral_shape = -5.076 + (2.484 - 506.25* ( np.log10(e))**2)**0.5
    elif 1.17 < e <= 1.674:
        Spectral_shape = -98.409 * np.log10(e) + 2
    elif e > 1.674:
        Spectral_shape = -39.8 * np.log10(e) -11.12
    return Spectral_shape * e

def Reynolds_zero(alpha_star):
    if alpha_star <= 3:
        Reynolds_0 = 10**(0.215 * alpha_star + 4.978)
    else:
        Reynolds_0 = 10**(0.120 * alpha_star + 5.263)
    return Reynolds_0

def G_2(Reynolds, Reynolds_0):
    d = Reynolds / Reynolds_0

    if d <= 0.3237:
        Peak_scaled_level = 77.852 * np.log10(d) + 15.328
    elif 0.3237 < d <= 0.5689:
        Peak_scaled_level = 65.188 * np.log10(d) + 9.125
    elif 0.5689 < d <= 1.7579:
        Peak_scaled_level = -114.052 * (np.log10(d))**2
    elif 1.7579 < d <= 3.0889:
        Peak_scaled_level = -65.188 * np.log10(d) + 9.125
    else:
        Peak_scaled_level = -77.852* np.log10(d) + 15.328
    return  Peak_scaled_level * d

def G_3(alpha_star):
    Angle_dependent_level = 171.04 - 3.03 * alpha_star
    return Angle_dependent_level * alpha_star



def SPL_LBL(delta, M, L, Dh, r_e, spectral_shape, peak_scaled_level, angle_dependent_level):
    return SPL


