import common.py
import numpy as np
import math

"""
Experimental Variables

"""
Reynolds =                         #Reynolds number based on chord length
Span =                             #Wind turbine span
r_e = common.r_e(0, 0, 1.775, 1.34)                 #distance from source to observer (meters)
U =                                #Free stream velocity
delta =                            #Boundary layer thiccness
Theta_e =                          #Angle from source streamwise axis
Phi_e =                            #Angle from source source lateral y axis
f =                                #frequency
alpha_star =                       #effective aerodynamic angle of attack
M =                                #flow mach number
M_c = 0.8 * M                      #convection mach number
T =                                #temperature
c_0 = np.sqrt(1.4 * 287 * T)       #speed of sound
c =                                #chord
alpha_tip =                        #angle of attack of tip
U_max =                            #max flow vel
f =                                #frequency
Dh = common.Dh_bar(Theta_e, Phi_e, M, M_c)                             #directivity func


