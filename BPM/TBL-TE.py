import math

# GLOBAL VARIABLES:
# M
# f
# delta_p_star
# delta_s_star
# U
# alphastar
# Rc
# ----------------------------------------------------------------------------------------------------

# Strouhall numbers:
# TODO: what is the difference between stp and sts?
def calc_stp(f,delta_p_star,U):
    '''Calculates Strouhall number St_p (float) as a function of f, delta_p*, and U (floats).'''
    # eq 31a
    stp = (f*delta_p_star)/U
    return stp

def calc_sts(f,delta_s_star,U):
    '''Calculates Strouhall number St_p (float) as a function of f, delta_p*, and U (floats).'''
    # eq 31b
    sts = (f*delta_s_star)/U
    return sts

def calc_strouhall(M, alphastar):
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


def calc_a0(Rc):
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
# TODO: rewrite into one function
def amplitudefunctions(Rc,alphastar,M,R_deltaPstar):
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

def calc_a(stp, sts, st1, st2, st1bar):
    # eq 37
    st = max(stp, sts)
    st_peak = max(st1, st2, st1bar)
    a = abs(math.log10(st/st_peak))
    return a
# -------------------------------------------------------------------------------------------------------------

# Sound Pressure Levels
# TODO: rewrite this bad boy to actually be useable
def calc_SPLtot(delta_p_star,delta_s_star,M,L,D_h_bar,A,B,stp,sts,st1,st2,K1,delta_K1,K2,re):
    '''Calculate total SPL for angles of attack up to a0*'''
    SPLp = 10*math.log10((delta_p_star*M**5*L*D_h_bar)/(re**2)) + A*(stp/st1) + (K1-3) + delta_K1
    SPLs = 10*math.log10((delta_s_star*M**5*L*D_h_bar)/(re**2)) + A*(sts/st1) + (K1-3)
    SPLalpha = 10*math.log10((delta_s_star*M**5*L*D_h_bar)/(re**2)) + B*(sts/st2) + K2

    SPLtotal = 10*math.log10(10**(SPLalpha/10) + 10**(SPLs/10) + 10**(SPLp/10))
    return SPLtotal