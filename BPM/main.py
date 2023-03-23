import common.py
import math
import numpy as np
import math

"""
Experimental Variables

"""
#viscosity
visc = 1.48 * 10**-5
#density
rho = 1.225
#span
Span = 1                    
#distance from source to receiver        
r_e = common.r_e(0, 0, 1.775, 1.34)   
#Boundary layer thickness              
delta = 
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

def plot_SPL()