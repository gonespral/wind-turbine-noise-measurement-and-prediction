import common.py
import math
import numpy as np
import math
import TE_Bluntness.py
import TipVortex.py
import LBL_VS.py
import TBL_TE.py

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
delta_p = 0
#displacement thickness of suction side of airfoil
delta_s = 0
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
alpha_tip = 2  
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

"""
Initializing frequency bands

"""

def Blunt(f):
    delta_avg = TE_Bluntness.delta_avg(delta_p, delta_s)
    mu = TE_Bluntness.mu(h, delta_avg)
    m = TE_Bluntness.m(h, delta_avg)
    eta0 = TE_Bluntness.eta0(m, mu)
    k = TE_Bluntness.k(eta0, m)
    St = TE_Bluntness.St(f, h, U)
    G4 = TE_Bluntness.G4(h, delta_avg, psi)
    G5 = TE_Bluntness.G5(eta, eta0, k, m)
    St_peak = TE_Bluntness.St_peak(h, delta_avg, psi)
    eta = TE_Bluntness.eta(St, St_peak)
    return TE_Bluntness(h,M,L,Dh,r_e,G4,G5)

def TipVortex(f):
    l = TipVortex.l(c, alpha_tip)
    M_max = TipVortex.M_max(M, alpha_tip)
    st = TipVortex.st(f, l, U_max)
    return TipVortex.SPL_tip(M, M_max, l, Dh, r_e, st)

def LBLVS(f):
    return LBL_VS.SPL_LBL()


SPL = TipVortex(200)
print(SPL)