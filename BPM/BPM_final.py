import common
import numpy as np
import matplotlib.pyplot as plt
import TBL_TE_Real as TBL



# Adding the inputs
visc = 1.81 * 10**-5
Theta_e = np.pi / 2  
Phi_e = np.pi / 2   
c_0 = np.sqrt(1.4 * 287 * 288.15)
rho = 1.225
c = 0.15

"""
These arrays contain the inputs per section we found
these using XFOIL for each section and calculated the velocity
using the angular velocity of the turbine
"""

U = np.array([7.7333,12.405,17.626,23.026,28.503,34.021,39.561,45.116,50.681,56.253])
delta_p = np.array([0.082,0.062,0.05,0.0450101,0.0425,0.04,0.0385,0.0372,0.036,0.035])
delta_s = np.array([0.082,0.062,0.05,0.0450101,0.0425,0.04,0.0385,0.0372,0.036,0.035])
print(U.shape,delta_p.shape,delta_s.shape)
# U = 50 * np.ones(10)
# delta_p = 0.1 * np.ones(10)
# delta_s = 0.001 * np.ones(10)

M = U / c_0
M_c = 0.8 * M                  
alpha_star = np.degrees(np.arctan(1/7))
Reynolds = rho * U * c / visc
L = 0.1
r_e = 1.22
#Dh = common.Dh_bar(Theta_e, Phi_e, M, M_c)  
Dh = 1    
R_deltaPstar = delta_p * U / visc

"""
Implementing the TBL-TE noise source, these are taken from the
BPM paper we checked multiple times and are quite confident the 
functions are correct

"""
def TBL_TE(f, U, delta_s, delta_p, Reynolds, M, R_deltaPstar):
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
    St_peak = TBL.St_peak1(St_1, St_2, St_1mean)
    b0 = TBL.b01(Reynolds)
    b = TBL.b1(St_p, St_2)
    B_min = TBL.B_min1(b)
    B_max = TBL.B_max1(b)
    B_min1 = TBL.B_min1(b0)
    B_max1 = TBL.B_max1(b0)
    B_R = TBL.B_R1(B_min1, B_max1)
    B = TBL.B1(B_min, B_R, B_max)
    a0 = TBL.a01(Reynolds) # I started from here 
    a = TBL.a1(St_s, St_peak) 
    A_min = TBL.A_min1(a)
    A_max = TBL.A_max1(a)
    A_min1 = TBL.A_min1(a0)
    A_max1 = TBL.A_max1(a0)
    A_R = TBL.A_R1(A_min1, A_max1)
    A = TBL.A1(A_min, A_R, A_max) 
    SPL_alpha = TBL.SPL_alpha1(delta_s, M, L, Dh, r_e, B, St_s, St_2, K2)
    SPL_s = TBL.SPL_s1(delta_s, M, L, Dh, r_e, A, St_s, St_1, K1)
    SPL_p = TBL.SPL_p1(delta_p, M, L, Dh, r_e, A, St_p, St_1, K1, deltaK1)
    SPL_tot =  TBL.SPL_tot1(SPL_alpha, SPL_s, SPL_p)


    return SPL_s, SPL_p, SPL_alpha, SPL_tot

"""
Calculated the SPL generating matricies of (10, 9999)

Unfortunately it is messy since we had to adapt from
calculating without secitoning the airfoil

Simply put we loop through each section and each frequency and
calculate the SPL using the BPM functions. This creates matrices where
the rows correspond to 10 sections and the columns are frequencies


"""

def CalculateSPL(f, U, delta_s, delta_p, Reynolds, M, R_deltaPstar): 
    SPLTBL_tot = np.zeros((10, len(f)))
    SPLTBL_alpha = np.zeros((10, len(f)))
    SPLTBL_p = np.zeros((10, len(f)))
    SPLTBL_s = np.zeros((10, len(f)))
    for k in range(10):
        for i in range(len(f)):
            SPLTBL_s[k, i] = TBL_TE(f[i], U[k], delta_s[k], delta_p[k], Reynolds[k], M[k], R_deltaPstar[k])[0]
            SPLTBL_p[k, i] = TBL_TE(f[i], U[k], delta_s[k], delta_p[k], Reynolds[k], M[k], R_deltaPstar[k])[1]
            SPLTBL_alpha[k, i] = TBL_TE(f[i], U[k], delta_s[k], delta_p[k], Reynolds[k], M[k], R_deltaPstar[k])[2]
            SPLTBL_tot[k, i] = TBL_TE(f[i], U[k], delta_s[k], delta_p[k], Reynolds[k], M[k], R_deltaPstar[k])[3]
        print("Section", k)
        # print(Reynolds[k])
    return SPLTBL_s, SPLTBL_p, SPLTBL_alpha, SPLTBL_tot

"""
The following function sums the contributions from
each section to compile the final arrays for the SPL, it
can then be plotted

"""


def AddSections(a, b, c, d, f):
    SPL_s = np.zeros(len(f))
    SPL_p = np.zeros(len(f))
    SPL_alpha = np.zeros(len(f))
    SPL_tot = np.zeros(len(f))

    for i in range(len(f)):
        SPL_s[i] = 10 * np.log10(sum(10**(a[:,i]/10)))
        SPL_p[i] = 10 * np.log10(sum(10**(b[:,i]/10)))
        SPL_alpha[i] = 10 * np.log10(sum(10**(c[:,i]/10)))
        SPL_tot[i] = 10 * np.log10(sum(10**(d[:,i]/10)))

    return SPL_s, SPL_p, SPL_alpha, SPL_tot


# This function simply plots the SPL vs frequency with log scale on x axis
    
def plotone(f, SPLTBL_s, SPLTBL_p, SPLTBL_alpha, SPLTBL_tot):
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

    # plt.scatter(dotf, dotp)
    # plt.scatter(dotf, dots)
    # plt.scatter(dotf, dota)
    # plt.scatter(dotf, dottot)
    plt.plot(f, SPLTBL_p)
    plt.plot(f, SPLTBL_s, 'tab:orange')
    plt.plot(f, SPLTBL_alpha, 'tab:green')
    plt.plot(f, SPLTBL_tot, 'tab:red')
    plt.legend(["Pressure Side", "Suction Side", "Alpha", "Total SPL"])
    plt.ylim((0, 80))
    plt.xscale('log')
    plt.title("SPL vs Frequency")
    plt.show()

f = np.arange(1,5000)
SPLTBL_s, SPLTBL_p, SPLTBL_alpha, SPLTBL_tot = CalculateSPL(f, U, delta_s, delta_p, Reynolds, M, R_deltaPstar)
SPL_s, SPL_p, SPL_alpha, SPL_tot = AddSections(SPLTBL_s, SPLTBL_p, SPLTBL_alpha, SPLTBL_tot, f)
plotone(f, SPL_s, SPL_p, SPL_alpha, SPL_tot)
#np.set_printoptions(threshold=np.inf)
#print(SPL_tot)


