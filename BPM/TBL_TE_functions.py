import math

# GLOBAL VARIABLES:
# M
# f
# delta_p_star
# delta_s_star
# U
# alphastar
# Rc
# re
# D_bar_h

# ----------------------------------------------------------------------------------------------------

# Strouhall numbers:
# TODO: what is the difference between stp and sts?
def calc_stp(f,delta_p_star,U): #USED
    '''Calculates Strouhall number St_p (float) as a function of f, delta_p*, and U (floats).'''
    # eq 31a
    stp = (f*delta_p_star)/U
    return stp

def calc_sts(f,delta_s_star,U): #USED
    '''Calculates Strouhall number St_p (float) as a function of f, delta_p*, and U (floats).'''
    # eq 31b
    sts = (f*delta_s_star)/U
    return sts

def calc_strouhall(M, alphastar): #USED
    # eqs 32, 33, 34
    st1 = .02*M**(-.6)

    if alphastar < 1.33:
        st2 = st1
    elif 1.33 <= alphastar <= 12.5:
        st2 = st1*10**(.0054*(alphastar-1.33)**2)
    else:
        st2 = 4.72*st1
    
    st1bar = (st1+st2)/2

    return st1, st2, st1bar


def calc_a0(Rc): #USED
    '''Calculates a0, a value at which the spectrum has a value of -20dB.'''
    # eq 38
    if Rc < 9.52*10**4:
        a0 = .57
    elif 9.52*10**4 <= Rc <= 8.57*10**5:
        a0 = (-9.57*10**(-13))*(Rc-8.57*10**5)**2 + 1.13
    else:
        a0 = 1.13
    return a0

# ----------------------------------------------------------------------------------------------------------

# Amplitude functions
def amplitudefunctions(Rc,alphastar,M,R_deltaPstar): #USED
    '''Returns values for the amplitude functions K1, DeltaK1, and K2.'''
    # eqs 47, 48, 49, 50
    #coefficcients:
    gamma = 27.094*M+3.31
    gamma0 = 23.43*M+4.651
    beta = 72.65*M+10.74
    beta0 = -34.19*M - 13.82

    # K1:
    if Rc < 2.47*10**5:
        K1 = -4.31*math.log10(Rc) + 156.3
    elif 2.47*10**5 <= Rc <= 8.0*10**5:
        K1 = -9.0*math.log10(Rc) + 181.6
    else:
        K1 = 128.5

    # Delta K1:
    if R_deltaPstar <= 5000:
        deltaK1 = alphastar*(1.43*math.log10(R_deltaPstar) - 5.29)
    else:
        deltaK1 = 0

    # K2:
    if alphastar < (gamma0 - gamma):
        K2 = K1 - 1000
    elif (gamma0 - gamma) <= alphastar <= (gamma0 + gamma):
        K2 = math.sqrt(beta**2 - (beta/gamma)**2 * (alphastar - gamma0)**2) + beta0
    else:
        K2 = K1 - 12

    return K1, K2, deltaK1

def calc_a(stp, sts, st1, st2, st1bar): #USED
    # eq 37
    st = max(stp, sts)
    st_peak = max(st1, st2, st1bar)
    a = abs(math.log10(st/st_peak))
    return a

def calc_b(sts, st2): #USED
    # eq 43
    b = abs(math.log10(sts/st2))
    return b

def calc_b0(Rc): #USED
    # eq 44
    if Rc < 9.52*10**4:
        b0 = .30
    elif 9.52*10**4 <= Rc <= 8.57*10**5:
        b0 = (-4.48*10**(-13))*(Rc - 8.57*10**5)**2 + .56
    else:
        b0 = .56
    return b0
# -------------------------------------------------------------------------------------------------------------
def calc_Amin(a): #USED
    # eq 35
    if a < .204:
        Amin = math.sqrt(67.552 - 886.788*a**2) - 8.219
    elif .204 <= a <= .244:
        Amin = -32.665**a + 3.981
    else:
        Amin = -142.795*a**3 + 103.656*a**2 - 57.757*a +6.006
    return Amin

def calc_Amax(a): #USED
    #eq 36
    if a < .13:
        Amax = math.sqrt(67.552 - 886.788*a**2) - 8.219
    elif .13 <= a <= .321:
        Amax = -15.901**a + 1.098
    else:
        Amax = -4.669*a**3 + 3.491*a**2 - 16.699*a + 1.149
    return Amax

def calc_Bmin(b): #USED
    # eq 41
    if b < .13:
        Bmin = math.sqrt(16.888 - 886.788*b**2) - 4.109
    elif .13 <= b <= .145:
        Bmin = -83.607*b + 8.138
    else:
        Bmin = -817.810*b**3 + 355.210*b**2 -135.024*b + 10.619
    return Bmin

def calc_Bmax(b): #USED
    # eq 42
    if b < .1:
        Bmax = math.sqrt(16.888 - 886.788*b**2) - 4.109
    elif .1 <= b <= .187:
        Bmax = -31.33*b + 1.854
    else:
        Bmax = -80.541*b**3 + 44.174*b**2 - 39.381*b + 2.344
    return Bmax
# -------------------------------------------------------------------------------------------------------------
def calc_A(a,a0): #USED
    # eqs 39, 40
    AR = (-20 - calc_Amin(a0))/(calc_Amax(a0)- calc_Amin(a0))
    A = calc_Amin(a) + AR*(calc_Amax(a)-calc_Amin(a))
    return A

def calc_B(b,b0): #USED
    # eqs 45, 46
    BR = (-20 - calc_Bmin(b0))/(calc_Bmax(b0)- calc_Bmin(b0))
    B = calc_Bmin(b) + BR*(calc_Bmax(b)-calc_Bmin(b))
    return B


# --------------------------------------------------------------------------------------------------------------
def SPL_TOT(A, B, stp, sts, st1, st2, K1, K2, deltaK1):
    SPLp = 10*math.log10((delta_p*M**5*L*Dh)/r_e**2) + A*(stp/st1) + K1 - 3 + deltaK1
    SPLs = 10*math.log10((delta_s*M**5*L*Dh)/r_e**2) + A*(sts/st1) + K1 - 3
    SPLalpha = 10*math.log10((delta_s*M**5*L*Dh)/re**2) + B*(sts/st2) + K2
    SPL_TOT = 10*math.log10(10**(SPLalpha/10) + 10**(SPLs/10) + 10**(SPLp/10))
    return SPL_TOT