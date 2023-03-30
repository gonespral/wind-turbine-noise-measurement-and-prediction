import common
import numpy as np
from TE_Bluntness import *
from TipVortex import *
from TBL_TE_functions import *
from LBL_VS import *
import matplotlib.pyplot as plt

"""
Experimental Variables

"""
#angle between sloping surfaces unstream
psi = 14
#bluntness 
h = 0.0025
#viscosity
visc = 1.48 * 10**-5
#density
rho = 1.225
#span
L = 1                    
#distance from source to receiver        
r_e = common.r_e(0, 0, 1.775, 1.34)   
#Boundary layer thickness              
delta = 0
#displacement thickness
delta_p = 0.01
#displacement thickness of suction side of airfoil
delta_s = 0.02
#Angle from source streamwise axis                           
Theta_e = np.pi / 2  
#Angle from source source lateral y axis                       
Phi_e = np.pi / 2   
#effective aerodynamic angle of attack                        
alpha_star = 2     
#temperature K                 
T = 288.15    
#chord m                           
c = 0.15    
#angle of attack of tip                          
alpha_tip = 18 
 #max flow vel                      
U_max = 8 
#speed of sound  
c_0 = np.sqrt(1.4 * 287 * T)   
#Free stream velocity    
U = 8       
#flow mach number                        
M = U / c_0    
#convection mach number                                 
M_c = 0.8 * M                  
#directivity func                    
Dh = common.Dh_bar(Theta_e, Phi_e, M, M_c)      
#Reynolds number based on chord length                       
Reynolds = rho * U * c / visc
# Reynolds number based on pressure side boundary layer thickness displacement
R_deltaPstar = delta_p*U/visc
# ---------

# Initializing frequency bands
def Blunt(f):
    delta_avg = delta_avg1(delta_p, delta_s)
    mu = mu1(h, delta_avg)
    m = m1(h, delta_avg)
    eta0 = eta01(m, mu)
    k = k1(eta0, m, mu)
    St = St1(f, h, U)
    G4 = G41(h, delta_avg, psi)
    St_peak = St_peak1(h, delta_avg, psi)
    eta = eta1(St, St_peak)
    G5 = G51(eta, eta0, k, m, mu) 
    return SPL_BLUNT1(h,M,L,Dh,r_e,G4,G5)

def TipVortex(f):
    l = lfunc(c, alpha_tip)
    M_max = M_maximum(M, alpha_tip)
    st = stNum(f, l, U_max)
    return SPL_tip(M, M_max, l, Dh, r_e, st)

def LBL(f):
    st_prime = st_prime_one(Reynolds)
    st_prime_peak = st_peak(alpha_star, st_prime)
    spectral_shape = G_1(f, delta, U, st_prime_peak)
    Reynolds_0 = Reynolds_zero(alpha_star)
    peak_scaled_level = G_2(Reynolds, Reynolds_0)
    angle_dependent_level = G_3(alpha_star)
    return SPL_LBL(delta, M, L, Dh, r_e, spectral_shape, peak_scaled_level, angle_dependent_level)

def TBL_TE(f):
    stp = calc_stp(f, delta_p, U)
    sts = calc_sts(f, delta_s, U)
    st1, st2, st1bar = calc_strouhall(M, alpha_star)
    a0 = calc_a0(Reynolds)
    b0 = calc_b0(Reynolds)
    K1, K2, deltak1 = amplitudefunctions(Reynolds, alpha_star, M, R_deltaPstar)
    a = calc_a(stp, sts, st1, st2, st1bar)
    b = calc_b(sts, st2)
    A = calc_A(a, a0)
    B = calc_B(b, b0)
    return SPL_TOT(A, B, stp, sts, st1, st2, K1, K2, deltak1)

def CalculateSPL(): 
    f = np.arange(0,10000)
    SPL = []
    for i in range(len(f)):
        SPL.append(LBL(f[i]))

    return f, SPL

def plot():
    f, SPL = CalculateSPL()
    plt.plot(f, SPL)
    plt.title("LBL")
    plt.xlabel("Frequency")
    plt.ylabel("SPL")
    plt.show()

plot()