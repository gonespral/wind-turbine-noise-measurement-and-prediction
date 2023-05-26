#python2.7

import _bpmacoustic
import numpy as np
import csv

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

