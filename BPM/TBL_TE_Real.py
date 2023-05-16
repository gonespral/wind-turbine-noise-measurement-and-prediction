import numpy as np


# we need :  ( M, U, L, Dh, re, f, alphastar,Rc, R_deltap, St_peak, delta_s, delta_p)



def gamma1(M):
    gamma = 27.094*M + 3.31
    return gamma

def gamma01(M):
    gamma0 = 23.43*M + 4.651
    return gamma0

def beta1(M):
    beta = 72.65*M + 10.74
    return beta

def beta01(M):
    beta0 = -34.19*M - 13.82
    return beta0

def K21(alphastar, gamma, gamma0, beta0, beta, K1):
    if alphastar < (gamma0 - gamma):
        K2 = K1 - 1000
    elif (gamma0 - gamma) <= alphastar <= (gamma0 + gamma):
        K2 = K1 + (beta**2 - (beta/gamma)**2 * (alphastar - gamma0)**2)**0.5 + beta0
    elif alphastar > (gamma0 + gamma):
        K2 = K1 - 12
    return K2

def K11(Rc):
    if Rc < (2.47 * 10**5):
        K1 = -4.31 * np.log10(Rc) + 156.3
    elif (2.47 * 10**5) <= Rc <= (8 * 10**5):
        K1 = -9 * np.log10(Rc) + 181.6
    elif Rc > (8 * 10**5):
        K1 = 128.5
    return K1

def deltaK11(alphastar, R_deltap):
    if R_deltap <= 5000:
        deltaK1 = alphastar * (1.43 * np.log10(R_deltap) - 5.29)
    else:
        deltaK1 = 0
    return deltaK1



#calculate all the St's



def St_p1(f, delta_p, U):
    St_p = f * delta_p / U
    return St_p

def St_s1(f, delta_s, U):
    St_s = f * delta_s / U
    return St_s

def St_11(M):
    St_1 = 0.02 * (M**(-0.6))
    return St_1

#this is in deg (alphastar)
def St_21(St_1, alphastar):
    if alphastar < 1.33:
        St_2 = St_1 * 1
    elif 1.33 <= alphastar <= 12.5:
        St_2 = St_1 * (10**(0.0054 * (alphastar - 1.33)**2 ))
    elif alphastar > 12.5:
        St_2 = St_1 * 4.72
    return St_2

def St_1mean1(St_1, St_2):
    St_1mean = (St_1 + St_2) / 2 
    return St_1mean

def St_peak1(St_1, St_2, St_1mean):
    St_peak = max(St_1, St_2, St_1mean)
    return St_peak



#calculate all the B's



def b01(Rc):
    if Rc < (9.52 * 10**4):
        b0 = 0.3
    elif (9.52 * 10**4) <= Rc <= (8.57 * 10**5):
        b0 = (-4.48 * 10**-13) * (Rc - 8.57 * 10**5)**2 + 0.56
    elif Rc > (8.57 * 10**5):
        b0 = 0.56
    return b0

def b1(St_s, St_2):
    b = abs(np.log10(St_s / St_2))
    return(b)

def B_min1(b):
    if b < 0.13:
        B_min = (16.888 - 886.788 * b**2)**0.5 - 4.109
    elif 0.13 <= b <= 0.145:
        B_min = -83.607 * b + 8.138
    elif b > 0.145:
        B_min = -817.810 * b**3 + 355.210 * b**2 - 135.024 * b + 10.619
    return B_min

def B_max1(b):
    if b < 0.1:
        B_max = (16.888 - 886.788 * b**2)**0.5 - 4.109
    elif 0.1 <= b <= 0.187:
        B_max = -31.330 * b + 1.854
    elif b > 0.187:
        B_max = -80.541 * b**3 + 44.174 * b**2 - 39.381 * b + 2.344
    return  B_max

def B_R1(B_min, B_max):
    B_R = (-20 - B_min) / (B_max - B_min)
    return B_R

def B1(B_min, B_R, B_max):
    B = B_min + B_R * (B_max - B_min)
    return B



#calculate all the A's



def a01(Rc):
    if Rc < (9.52 * 10**4):
        a0 = 0.57
    elif (9.52 * 10**4) <= Rc <= (8.57 * 10**5):
        a0 = (-9.57 * 10**-13) * (Rc - 8.57 * 10**5)**2 + 1.13
    elif Rc > (8.57 * 10**5):
        a0 = 1.13
    return a0

def a1(St, St_peak):
    a = np.abs(np.log10(St / St_peak))
    return(a)

def A_min1(a):
    if a < 0.204:
        A_min = (67.552 - 886.788 * a**2)**0.5 - 8.219
    elif 0.204 <= a <= 0.244:
        A_min = -32.665 * a + 3.981
    elif a > 0.244:
        A_min = -142.795 * a**3 + 103.656 * a**2 - 57.757 * a + 6.006
    return A_min

def A_max1(a):
    if a < 0.13:
        A_max = (67.552 - 886.788 * a**2)**0.5 - 8.219
    elif 0.13 <= a <= 0.321:
        A_max = -15.901 * a + 1.098
    elif a > 0.321:
        A_max = -4.669 * a**3 + 3.491 * a**2 - 16.699 * a + 1.149
    return A_max

def A_R1(A_min, A_max):
    A_R = (-20 - A_min) / (A_max - A_min)
    return A_R 

def A1(A_min, A_R, A_max):
    A = A_min + A_R * (A_max - A_min)
    return A



#calculate all the SPL's



def SPL_alpha1(delta_s, M, L, Dh, re, B, K2):
   SPL_alpha = 10 * np.log10( (delta_s * M**5 * L * Dh) / re**2) + B  + K2 
   return SPL_alpha

def SPL_s1(delta_s, M, L, Dh, re, A, K1):
    SPL_s = 10 * np.log10( (delta_s * M**5 * L * Dh) / re**2) + A  + (K1 - 3)
    return SPL_s

def SPL_p1(delta_p, M, L, Dh, re, A, K1, deltaK1):
    SPL_p = 10 * np.log10( (delta_p * M**5 * L * Dh) / re**2) + A  + (K1 - 3) + deltaK1
    return SPL_p

def SPL_tot1(SPL_alpha, SPL_s, SPL_p):
   SPL_tot = 10 * np.log10( 10**(SPL_alpha/10) + 10**(SPL_s/10) + 10**(SPL_p/10))
   return SPL_tot


