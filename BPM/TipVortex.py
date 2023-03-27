
import numpy as np
import math
import common

def lfunc(c, alpha_tip):
    """
    calculate separated flow region size (l)

    Keyword arguments:
    c -- the chord length (float, int)
    alpha_tip -- angle of attack of tip (degrees)
    """

    if (alpha_tip >= 0) and (alpha_tip <= 2):
        l = c * (0.023 + (0.0169 * alpha_tip))
    elif alpha_tip > 2:
        l = c * (0.0378 + (0.0095 * alpha_tip))
    else:
        print("Angle of attack is negative!!")
    return l


def M_maximum(M, alpha_tip):
    """
    calculate max mach number (M_max)

    Keyword arguments:
    M -- mach number for incoming flow
    alpha_tip -- angle of attack of tip (degrees)
    """
    return M * (1 + (0.036 * alpha_tip))

def stNum(f, l, U_max):
    """
    calculate strouhal number (l)

    Keyword arguments:
    f -- frequency
    l -- flow region size
    U_max -- maximum flow velocity
    """
    return (f * l) / U_max


def SPL_tip(M, M_max, l, D_h, r_e, st):
    """
    calculate SPL for tip vortex

    Keyword arguments:
    M -- mach number for incoming flow
    M_max -- max mach number 
    l -- flow region size 
    D_h -- directivity function
    r_e -- radial distance from observer
    st -- strouhal number
    """

    SPL = (10 * np.log10((M**2 * M_max**3 * l**2 * D_h) / (r_e**2))) - (30.5 * ((np.log10(st) + 0.3)**2)) + 126

    return SPL

    #print(nothing)
