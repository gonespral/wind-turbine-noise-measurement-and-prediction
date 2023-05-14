import numpy as np
import matplotlib.pyplot as plt
import TBL_TE_Real as TBL


U = 56.253
delta_p = 0.004
delta_s = 0.031

rho = 1.225
visc = 1.81 * 10**-5
c_0 = 340.26
c = 0.15
M = U / c_0
alpha_star = np.degrees(np.arctan(2/21))
Reynolds = rho * U * c / visc
L = 0.1
r_e = 1
Dh = 1    
R_deltaPstar = delta_p * U / visc

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
    St_peak = TBL.St_peak1(St_1, St_2, St_1mean)
    b0 = TBL.b01(Reynolds)
    b = TBL.b1(St_p, St_2)
    B_min = TBL.B_min1(b)
    B_max = TBL.B_max1(b)
    B_min1 = TBL.B_min1(b0)
    B_max1 = TBL.B_max1(b0)
    B_R = TBL.B_R1(B_min1, B_max1)
    B = TBL.B1(B_min, B_R, B_max)
    a0 = TBL.a01(Reynolds)
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


def CalculateSPL(f): 
    SPLTBL_tot = []
    SPLTBL_alpha = []
    SPLTBL_p = []
    SPLTBL_s = []
    for i in range(len(f)):
        SPLTBL_s.append(TBL_TE(f[i])[0])
        SPLTBL_p.append(TBL_TE(f[i])[1])
        SPLTBL_alpha.append(TBL_TE(f[i])[2])
        SPLTBL_tot.append(TBL_TE(f[i])[3])
    return SPLTBL_s, SPLTBL_p, SPLTBL_alpha, SPLTBL_tot


def plotone(f, SPLTBL_s, SPLTBL_p, SPLTBL_alpha, SPLTBL_tot):
    plt.plot(f, SPLTBL_p)
    plt.plot(f, SPLTBL_s, 'tab:orange')
    plt.plot(f, SPLTBL_alpha, 'tab:green')
    plt.plot(f, SPLTBL_tot, 'tab:red')
    plt.legend(["Pressure Side", "Suction Side", "Alpha", "Total SPL"])
    plt.ylim((0, 80))
    plt.xscale('log')
    plt.title("SPL vs Frequency")
    plt.show()

f = np.arange(100,5000)
SPLTBL_s, SPLTBL_p, SPLTBL_alpha, SPLTBL_tot = CalculateSPL(f)
plotone(f, SPLTBL_s, SPLTBL_p, SPLTBL_alpha, SPLTBL_tot)