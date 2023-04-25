import common
import numpy as np
from TE_Bluntness import *
from TipVortex import *
# import TBL_TE_functions as TBL
from LBL_VS import *
import matplotlib.pyplot as plt
import TBL_TE_Real as TBL

"""
Experimental Variables

"""
#angle between sloping surfaces unstream
psi = 14 * np.pi / 180
#bluntness 
h = 0.0005
#viscosity
visc = 1.4529 * 10**-5
#density
rho = 1.225
#span
L = 0.1                   
#distance from source to receiver        
r_e = 1 
#Boundary layer thickness              
delta = 0.1
#displacement thickness
delta_p = 0.1
#displacement thickness of suction side of airfoil
delta_s = 0.001
#Angle from source streamwise axis                           
Theta_e = np.pi / 2  
#Angle from source source lateral y axis                       
Phi_e = np.pi / 2   
#effective aerodynamic angle of attack                        
alpha_star = 1.516 
#temperature K                 
T = 288.15    
#chord m                           
c = 1   
#angle of attack of tip                          
alpha_tip = 18 
 #max flow vel                      
U_max = 100
#speed of sound  
c_0 = 340.46
#Free stream velocity    
U = 100  
#flow mach number                        
M = U / c_0    
#convection mach number                                 
M_c = 0.8 * M                  
#directivity func                    
Dh = common.Dh_bar(Theta_e, Phi_e, M, M_c)      
#Reynolds number based on chord length                       
Reynolds = rho * U * c / visc
# Reynolds number based on pressure side boundary layer thickness displacement
R_deltaPstar = delta_p * U / visc

"""

Initializing frequency bands

"""

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
    gamma = TBL.gamma1(M)
    gamma0 = TBL.gamma01(M)
    beta = TBL.beta1(M)
    beta0 = TBL.beta01(M)
    K1 = TBL.K11(Reynolds)
    K2 = TBL.K21(alpha_star, gamma, gamma0, beta0, beta, K1)
    deltaK1 = TBL.deltaK11(alpha_star, R_deltaPstar)
    St_p = TBL.St_p1(f, delta_p, U)
    St_s = TBL.St_s1(f, delta_s, U)
    St_1 = TBL.St_11(M)
    St_2 = TBL.St_21(St_1, alpha_star)
    St_1mean = TBL.St_1mean1(St_1, St_2)
    St_peak = St_peak1(St_1, St_2, St_1mean)
    b0 = TBL.b01(Reynolds)
    b = TBL.b1(St_p, St_2)
    B_min = TBL.B_min1(b0)
    B_max = TBL.B_max1(b0)
    B_R = TBL.B_R1(B_min, B_max)
    B = TBL.B1(B_min, B_R, B_max)
    a0 = TBL.a01(Reynolds) # I started from here 
    a = TBL.a1(St_s, St_peak) 
    A_min = TBL.A_min1(a)
    A_max = TBL.A_max1(a)
    A_R = TBL.A_R1(A_min, A_max)
    A = TBL.A1(A_min, A_R, A_max) 
    SPL_alpha = TBL.SPL_alpha1(delta_s, M, L, Dh, r_e, B, St_s, St_2, K2)
    SPL_s = TBL.SPL_s1(delta_s, M, L, Dh, r_e, A, St_s, St_1, K1)
    SPL_p = TBL.SPL_p1(delta_p, M, L, Dh, r_e, A, St_p, St_1, K1, deltaK1)
    SPL_tot =  TBL.SPL_tot1(SPL_alpha, SPL_s, SPL_p)
    return SPL_s, SPL_p, SPL_alpha, SPL_tot
    #TBL.SPL_TBLTE1(SPL_s, SPL_p)
    

def CalculateSPL(): 
    f = np.arange(1,10000)
    SPLTBL_tot = []
    SPLTBL_alpha = []
    SPLTBL_p = []
    SPLTBL_s = []
    for i in range(len(f)):
        # SPLTip.append(TipVortex(f[i]))
        # SPLBlunt.append(Blunt(f[i]))
        # SPLLBL.append(LBL(f[i]))
        SPLTBL_s.append(TBL_TE(f[i])[0])
        SPLTBL_p.append(TBL_TE(f[i])[1])
        SPLTBL_alpha.append(TBL_TE(f[i])[2])
        SPLTBL_tot.append(TBL_TE(f[i])[3])

    return f, SPLTBL_s, SPLTBL_p, SPLTBL_alpha, SPLTBL_tot

# def plot():
#     f, SPLTBL_s, SPLTBL_p, SPLTBL_alpha, SPLTBL_tot = CalculateSPL()
#     print(SPLTBL_tot)
#     fig, axs = plt.subplots(2, 2)
#     axs[0, 0].plot(f, SPLTBL_p)
#     axs[0, 0].set_title('Pressure Side SPL')
#     axs[0, 0].set_ylim((0, 100))
#     axs[0, 0].set_xscale('log')
#     axs[0, 1].plot(f, SPLTBL_s, 'tab:orange')
#     axs[0, 1].set_title('Suction Side SPL')
#     axs[0, 1].set_ylim((0, 100))
#     axs[0, 1].set_xscale('log')
#     axs[1, 0].plot(f, SPLTBL_alpha, 'tab:green')
#     axs[1, 0].set_title('SPL due to Alpha')
#     axs[1, 0].set_ylim((0, 100))
#     axs[1, 0].set_xscale('log')
#     axs[1, 1].plot(f, SPLTBL_tot, 'tab:red')
#     axs[1, 1].set_title('Total SPL')
#     axs[1, 1].set_ylim((0, 100))
#     axs[1, 1].set_xscale('log')


#     #print(SPLLBL)
    
#     for ax in axs.flat:
#         ax.set(xlabel='Frequency', ylabel='SPL')
#     for ax in axs.flat:
#         ax.label_outer()
#     plt.show()

# plot()


# def plotnew():
#     f, SPLTBL_s, SPLTBL_p, SPLTBL_alpha, SPLTBL_tot = CalculateSPL()
#     plt.plot(f, SPLTBL_tot)
#     plt.ylim((0,100))
#     plt.xlabel("Frequency")
#     plt.ylabel("SPL")
#     plt.title("TBL-TE")
#     plt.show()


# plotnew()

def plotone():
    f, SPLTBL_s, SPLTBL_p, SPLTBL_alpha, SPLTBL_tot = CalculateSPL()
    dotf = []
    dots = []
    dotp = []
    dota = []
    dottot = []
    for i in range(1, len(f), 1000):
        dotf.append(f[i])
        dots.append(SPLTBL_s[i])
        dotp.append(SPLTBL_p[i])
        dota.append(SPLTBL_alpha[i])
        dottot.append(SPLTBL_tot[i])

    plt.scatter(dotf, dotp)
    plt.scatter(dotf, dots)
    plt.scatter(dotf, dota)
    plt.scatter(dotf, dottot)
    plt.plot(f, SPLTBL_p)
    plt.plot(f, SPLTBL_s, 'tab:orange')
    plt.plot(f, SPLTBL_alpha, 'tab:green')
    plt.plot(f, SPLTBL_tot, 'tab:red')
    plt.legend(["Pressure Side", "Suction Side", "Alpha", "Total SPL"])
    plt.ylim((0, 100))
    plt.xscale('log')
    plt.title("SPL vs Frequency")
    plt.show()


plotone()