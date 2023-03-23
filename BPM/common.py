'''File to house functions to calculate parameters used for multiple noise sources'''
import numpy as np

def Dh_bar(Theta_e, Phi_e, M, M_c):
    '''Directivity function for TE noise, D_h-bar, using Eq. B1.

    Keyword arguments:
        Theta_e -- Retarded angle from source streamwise x-axis to observer [rad]
        Phi_e -- Retarded angle from source lateral y-axis to observer [rad]
        M -- Mach number [-]
        M_c -- Convection Mach number [-]
    '''
    num = 2 * (np.sin(Theta_e/2))**2 * (np.sin(Phi_e))**2
    den = (1 + M * np.cos(Theta_e)) * (1 + (M - M_c) * np.cos(Theta_e))**2
    return num / den

def r_e(yObs, zObs, yTurb, zTurb):
    '''Calculates source-observer distance using retarded coordinates. Output distance units same as input.

    Keyword arguments:
        y and z coordinates of oberserver and turbine relative to given coordinate system.

    Output:
        Euclidean distance in same units as input.
    '''
    # Assume retarded coordinates are just over-complicating our project
    return np.sqrt( (yTurb - yObs)**2 + (zTurb - zObs)**2 )