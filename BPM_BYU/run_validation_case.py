#python2.7

import _bpmacoustic
import numpy as np
import csv

# Specifiy input parameters
turbx = np.array([0.])
turby = np.array([0.])
obs = np.array([0., 0., 1.22])
winddir = 90.
B = 3.
h = 1.22
rad = np.linspace(0, 0.4572, 1000)
c = np.ones(len(rad) - 1) * 0.3048
c1 = c * 0.25
alpha = np.ones(len(rad) - 1) * 1.516
nu = 1.4529e-5
c0 = 340.46
psi = 0.
AR = 0.4572 / 0.3048
noise_corr = 0.8697933840957954

print len(rad)
print rad
print len(c)
print c
print len(c1)
print c1
print len(alpha)
print alpha

# Iterate over velocities
for vel in [71.3]:

	windvel = np.array([vel])
	rpm = np.array([vel * 5 * 9.549297])

	SPL_HAWT = _bpmacoustic.turbinepos(turbx, turby, obs, winddir, windvel, rpm, B, h, rad, c, c1, alpha, nu, c0, psi, AR, noise_corr)

	print "Windvel: " + str(windvel)
	print "OASPL: " + str(SPL_HAWT[0])
	for f, SPL in zip(SPL_HAWT[1], SPL_HAWT[2]):
		print str(f) + " : " + str(SPL)

	rows = zip(SPL_HAWT[1], SPL_HAWT[2])
	with open("data_validation_case/data" + str(int(vel)) + ".csv", "wb") as file:
		writer = csv.writer(file)
		writer.writerow(["freq", "spl"])
		writer.writerows(rows)
	print "Data written to data_validation_case/data" + str(int(vel)) + ".csv"

