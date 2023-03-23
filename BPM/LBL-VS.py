'''File to house the LBL-VS calculation procedure'''
import math
def st_prime_one():
    st_prime = 0
    if Reynolds < 12.3*10**5:
        st_prime = 0.18
    elif Reynolds < 4.0 * 10**5:
        st_prime = 0.001756 * Reynolds ** 0.3931
    elif reynolds > 4.0 * 10**5:
        st_prime = 0.28
    return st_prime

def st_peak():
    st_prime_peak = st_prime_one() * 10 ** (-0.04*)
    return  st_prime_peak

def G_1():
    st = f * delta / U
    e = st / st_peak()
    Spectral_shape = 0
    if e < 0.5974:
        Spectral_shape = 39.8*math.log(e, 10) -11.12
    elif e <0.8545:
        Spectral_shape = 98.409*math.log(e, 10) + 2
    elif e < 1.17:
        Spectral_shape = -5.076 + (2.184 - 506.25* ( math.log(e, 10))**2)**0.5
    elif e < 1.674:
        Spectral_shape = -98.409 * math.log(e, 10) + 2
    elif e > 1.674:
        Spectral_shape = -39.8 * math.log(e, 10) -11.12
    return Spectral_shape

def Reynolds_zero():
    if alpha_star < 3:
        Reynolds_0 = 10 ** (0.215 * alpha_star + 4.978)
    elif:
        Reynolds_0 = 10 ** (0.120 * alpha_star + 5.263)
    return Reynolds_0

def G_2():
    d = Reynolds / Reynolds_zero()

    if d < 0.3237:
        Peak_scaled_level = 77.852 * math.log(d, 10) + 15.328
    elif d < 0.5689:
        Peak_scaled_level = 65.188 * math.log(d, 10) + 9.125
    elif d < 1.7579:
        Peak_scaled_level = -114.052 * (math.log(d, 10))**2
    elif d < 3.0889:
        Peak_scaled_level = -65.188 * math.log(d, 10) + 9.125
    else:
        Peak_scaled_level = -77.852* math.log(d, 10) + 15.238
    return  Peak_scaled_level

def G_3():
    Angle_dependent_level = 171.04 - 3.03 * alpha_star
    return Angle_dependent_level

def SPL_LBL():
    SPL = 10 * math.log(delta * M ** 5 * L * D_h() /( r_e ** 2), 10) + G_1() + G_2() + G_3()
    return SPL


