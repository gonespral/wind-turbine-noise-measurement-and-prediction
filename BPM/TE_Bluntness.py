import numpy as np
import common



def delta_avg1(delta_p, detla_s):
    delta_avg = (delta_p + detla_s) / 2
    return delta_avg


def mu1(h, delta_avg): 
    if (h / delta_avg) < 0.25:
        mu = 0.1221 
    elif 0.25 <= (h / delta_avg) < 0.62:
        mu = -0.2175 * (h / delta_avg) + 0.1755
    elif 0.62 <= (h / delta_avg) < 1.15:
        mu = -0.0308 * (h / delta_avg) + 0.0596 
    elif (h / delta_avg) >= 1.15:
        mu = 0.0242
    return mu

def m1(h, delta_avg): 
    if (h / delta_avg) <= 0.02:
        m = 0
    elif 0.02 < (h / delta_avg) <= 0.5:
        m = 68.724 * ( h / delta_avg) - 1.35
    elif 0.5 < (h / delta_avg) <= 0.62:
        m = 308.475 * ( h / delta_avg) - 121.23
    elif 0.62 < (h / delta_avg) <= 1.15:
        m = 224.811 * (h / delta_avg) - 69.35
    elif 1.15 < (h / delta_avg) <= 1.2:
        m = 1583.28 * (h / delta_avg) - 1631.59
    elif (h / delta_avg) > 1.2:
        m = 268.344 
    return m

def eta01(m, mu):
    eta0 = -1 * (( m**2 * mu**4 )/ (6.25 + m**2 * mu**2))**0.5
    return eta0

def k1(eta0, m, mu):
    k = 2.5 * (1 - (eta0/mu)**2)**0.5 -2.5 - m * eta0
    return k

def St1(f, h , U):
    St = f * h / U
    return St



def G41(h, delta_avg, psi):
    if (h / delta_avg) <= 5:
        G4 = 17.5 * np.log10( h / delta_avg ) + 157.5 - 1.114 * psi 
    elif (h / delta_avg) > 5:
        G4 = 169.7 - 1.114 * psi
    return G4



def St_peak1(h, delta_avg, psi):
    if (h / delta_avg) >= 0.2:
        St_peak = (0.212 - 0.0045* psi)/( 1 + 0.235 * (h / delta_avg)**-1 - 0.0132 * (h / delta_avg)**-2)
    elif (h / delta_avg) < 0.2:
        St_peak = 0.1 * ( h / delta_avg) + 0.095 - 0.00243 * psi
    return St_peak 

def eta1(St,  St_peak):
    eta = np.log10(St / St_peak)
    return eta


def G51(eta, eta0, k, m, mu):
    if eta < eta0:
        G5 = m * eta + k
    elif eta0 <= eta < 0:
        G5 = 2.5 * (1- (eta/mu)**2)**0.5 - 2.5
    elif 0 <= eta < 0.03616:
        G5 = (1.5626 - 1194.99 * eta**2)**0.5 - 1.25
    elif 0.03616 <= eta:
        G5 = -155.543 * eta + 4.375
    return G5

def St_peak(h, delta_avg, psi):
    if (h / delta_avg) >= 0.2:
        St_peak = (0.212 - 0.0045* psi)/( 1 + 0.235 * (h / delta_avg)**(-1) - 0.0132 * (h / delta_avg)**-2)
    elif (h / delta_avg) < 0.2:
        St_peak = 0.1 * ( h / delta_avg) + 0.095 - 0.00243 * psi
    return St_peak 

def SPL_BLUNT1(h, M, L, D_h, r_e, G4, G5):
    SPL_BLUNT = 10* np.log10(h * M**5.5 * L * D_h / r_e**2 ) + G4 + G5
    return SPL_BLUNT

def hdelta_avg1():
    hdelta_avg1 = 6.724 * ( h / delta_avg)**2 - 4.019 * (h / detla_avg) + 1.107
    return hdelta_avg1














