#python2.7

import _bpmacoustic
import numpy as np
import csv

# -------------- Parameters --------------
#    x-positions of all the turbines heard by an observer (east to west, meter)
#turby : array
#    y-positions of all the turbines heard by an observer (north to south, meter)
#obs : array
#    x-, y-, and z-position of a specified observer (E-W, N-S, height; meter)
#winddir : float
#    direction the wind blows from (180=N, 270=E, 0=S, 90=W; degree)
#windvel : array
#    wind velocity at each turbine in the specified wind direction (m/s)
#rpm : array
#    rotation rate of each turbine (RPM)
#B : float
#    number of blades on a turbine
#h : float
#    height of a turbine hub (meter)
#rad : array
#    radial positions of the blade geometry (meter)
#c : array
#    chord length at each radial segment (meter)
#c1 : array
#    distance from the pitch axis to leading edge at each radial segment (meter)
#alpha : array
#    angle of attack of each radial segment (degree)
#nu : float
#    kinematic viscosity of the air (m^2/s)
#c0 : float
#    speed of sound of the air (m/s)
#psi : float
#    solid angle of turbine blades between upper and lower sides of trailing edge (degree)
#AR : float
#    aspect ratio of turbine blades
#noise_corr : float
#    correction factor for SPL calculations (1=none, use if calculations differ from expected)


# Specifiy input parameters
turbx = np.array([0.])
turby = np.array([0.])
obs = np.array([0., 1.34, 1.775])
winddir = 0.
B = 2.
h = 1.775
rad = np.arange(0, 1.1, 0.1)
c = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
c1 = c * 0.25
alpha = np.array([5.44, 5.44, 5.44, 5.44, 5.44, 5.44, 5.44, 5.44, 5.44, 5.44])
nu = 1.78e-5
c0 = 343.2
psi = 0
AR = 6.66
noise_corr = 0  # 0.8697933840957954

# Iterate over velocities
for vel in [8., 9., 10., 11., 12.]:

	windvel = np.array([vel])
	rpm = np.array([vel * 7 * 9.549297])

	SPL_HAWT = _bpmacoustic.turbinepos(turbx, turby, obs, winddir, windvel, rpm, B, h, rad, c, c1, alpha, nu, c0, psi, AR, noise_corr)

	print "Windvel: " + str(windvel)
	print "OASPL: " + str(SPL_HAWT[0])
	for f, SPL in zip(SPL_HAWT[1], SPL_HAWT[2]):
		print str(f) + " : " + str(SPL)

	rows = zip(SPL_HAWT[1], SPL_HAWT[2])
	with open("data/data" + str(int(vel)) + ".csv", "wb") as file:
		writer = csv.writer(file)
		writer.writerow(["freq", "spl"])
		writer.writerows(rows)
	print "Data written to data/data" + str(int(vel)) + ".csv"

