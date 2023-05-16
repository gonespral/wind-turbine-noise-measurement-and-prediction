import numpy as np
import matplotlib.pyplot as plt
import TBL_TE_Real as TBL
import pickle as pkl

#c = 0.15
c = 0.3048
rho = 1.225
visc = 1.7529 * 10**-5
c_0 = 340.46
#L = 0.1
L = 0.4572
r_e = 1.22
Dh = 1   
#alpha_star = np.rad2deg(np.arctan(2/21))
#alpha_star = 1.516
alpha_star = 0
U = 71.3

# delta_p = 0.00057
# delta_s = 0.00465
delta_p = 0.00192
delta_s = 0.00246

Reynolds = rho * U * c / visc
# delta_zero = c * 10**(3.411 - 1.5397*np.log10(Reynolds) + 0.1059*(np.log10(Reynolds))**2)
# delta_p = 10**(-0.0432*alpha_star + 0.00113*alpha_star**2) * delta_zero
# delta_s = 10**(0.0311*alpha_star) * delta_zero


M = U / c_0
R_deltaPstar = delta_p * U / visc


print(Reynolds, delta_p, delta_s, M, R_deltaPstar)


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
    b = TBL.b1(St_s, St_2)
    B_min = TBL.B_min1(b)
    B_max = TBL.B_max1(b)
    B_minb0 = TBL.B_min1(b0)
    B_maxb0 = TBL.B_max1(b0)
    B_R = TBL.B_R1(B_minb0, B_maxb0)
    B = TBL.B1(B_min, B_R, B_max)
    a0 = TBL.a01(Reynolds)

    ap = TBL.a1(St_p, 0.05) 
    A_minp = TBL.A_min1(ap)
    A_maxp = TBL.A_max1(ap)
    A_mina0 = TBL.A_min1(a0)
    A_maxa0 = TBL.A_max1(a0)
    A_Rp = TBL.A_R1(A_mina0, A_maxa0)
    Ap = TBL.A1(A_minp, A_Rp, A_maxp) 

    aS= TBL.a1(St_s, 0.05) 
    A_minS = TBL.A_min1(aS)
    A_maxS = TBL.A_max1(aS)
    A_RS = TBL.A_R1(A_mina0, A_maxa0)
    AS = TBL.A1(A_minS, A_RS, A_maxS) 

    SPL_alpha = TBL.SPL_alpha1(delta_s, M, L, Dh, r_e, B, St_s, St_2, K2)
    SPL_s = TBL.SPL_s1(delta_s, M, L, Dh, r_e, AS, St_s, St_1, K1)
    SPL_p = TBL.SPL_p1(delta_p, M, L, Dh, r_e, Ap, St_p, St_1, K1, deltaK1)
    SPL_tot =  TBL.SPL_tot1(SPL_alpha, SPL_s, SPL_p)


    return SPL_s, SPL_p, SPL_alpha, SPL_tot


def CalculateSPL(f): 
    ameanl = []
    SPLTBL_tot = []
    SPLTBL_alpha = []
    SPLTBL_p = []
    SPLTBL_s = []
    for i in f:
        SPLTBL_s.append(TBL_TE(i)[0])
        SPLTBL_p.append(TBL_TE(i)[1])
        SPLTBL_alpha.append(TBL_TE(i)[2])
        SPLTBL_tot.append(TBL_TE(i)[3])
    return SPLTBL_s, SPLTBL_p, SPLTBL_alpha, SPLTBL_tot, ameanl


def plotone(f, SPLTBL_s, SPLTBL_p, SPLTBL_alpha, SPLTBL_tot):
    plt.plot(f, SPLTBL_p)
    plt.plot(f, SPLTBL_s, 'tab:orange')
    plt.plot(f, SPLTBL_alpha, 'tab:green')
    plt.plot(f, SPLTBL_tot, 'tab:red')
    plt.legend(["Pressure Side", "Suction Side", "Alpha", "Total SPL"])
    plt.ylim((-30, 100))
    plt.xscale('log')
    plt.grid(True, linestyle='--', axis='x', which="both")
    plt.grid(True, linestyle='--', axis='y')
    #plt.xticks([800, 1000, 1500, 2000, 2500, 3000])
    plt.title("BPM output")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("SPL (dB)")
    #plt.gcf().set_size_inches(10, 10)
    #plt.tight_layout()
    #plt.savefig("../saves/BPM_spl.png", dpi=300)
    plt.show()

f = np.arange(200, 10000)
SPLTBL_s, SPLTBL_p, SPLTBL_alpha, SPLTBL_tot, ameanl = CalculateSPL(f)
plotone(f, SPLTBL_s, SPLTBL_p, SPLTBL_alpha, SPLTBL_tot)
# Save data to pickle file
# with open('saves/BPM_spl.pkl', 'wb') as f_:
#    pkl.dump([f, SPLTBL_s, SPLTBL_p, SPLTBL_alpha, SPLTBL_tot], f_)
