import numpy as np
from TBL_TE_Real import *



# Define input parameters
visc = 1.4529 * 10**-5
U = 71.3
L = 0.4572
Dh = 1
re = 1.22
f = 100
alphastar = 5
Rc = 1.225 * U * 0.3048 / visc
delta_s = 0.00246888
delta_p = 0.00192024
M = U / 340.46
R_deltap = delta_p * U / visc



# Calculate the necessary parameters
gamma = gamma1(M)
gamma0 = gamma01(M)
beta = beta1(M)
beta0 = beta01(M)
K1 = K11(Rc)
K2 = K21(alphastar, gamma, gamma0, beta0, beta, K1)
St_p = St_p1(f, delta_p, U)
St_s = St_s1(f, delta_s, U)
St_1 = St_11(M)
St_2 = St_21(St_1, alphastar)
St_1mean = St_1mean1(St_1, St_2)
St_peak = St_peak1(St_1, St_2, St_1mean)
b = b1(St_s, St_2)
B_min = B_min1(b)
B_max = B_max1(b)
B_R = B_R1(B_min, B_max)
B = B1(B_min, B_R, B_max)
a = a1(St_s, St_peak)
A_min = A_min1(a)
A_max = A_max1(a)
A_R = A_R1(A_min, A_max)
A = A1(A_min, A_R, A_max)

# Calculate the sound pressure levels (SPLs)
SPL_alpha = SPL_alpha1(delta_s, M, L, Dh, re, B, St_s, St_2, K2)
SPL_s = SPL_s1(delta_s, M, L, Dh, re, A, St_s, St_1, K1)
SPL_p = SPL_p1(delta_p, M, L, Dh, re, A, St_p, St_1, K1, deltaK11(alphastar, R_deltap))
SPL_tot = SPL_tot1(SPL_alpha, SPL_s, SPL_p)

# Print the results
print("SPL_alpha:", SPL_alpha)
print("SPL_s:", SPL_s)
print("SPL_p:", SPL_p)
print("SPL_tot:", SPL_tot)
