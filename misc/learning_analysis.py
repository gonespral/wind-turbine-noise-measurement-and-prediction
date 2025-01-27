import numpy as np
from scipy import fft, signal
from matplotlib import pyplot as plt

# the signal doc gave us
A1 = 0.02 * np.sqrt(2)
A2 = 0.002 * np.sqrt(2)
f1 = 500  # Hz
f2 = 2000  # hz
fs = 51200  # Hz
T = 20  # s
TLarge = 100

# Building the signal
t = np.arange(0, T + 1 / fs, 1 / fs)  # Start at 0, end at 20s, and have 51200 samples per second
tLarge = np.arange(-TLarge / 2, TLarge / 2, 1 / fs)


def signalTot(time):
    return A1 * np.sin(2 * np.pi * f1 * time + np.pi / 4) + A2 * np.cos(2 * np.pi * f2 * time + np.pi / 6)


def signal1(time):
    return A1 * np.sin(2 * np.pi * f1 * time + np.pi / 4)


def signal2(time):
    return A2 * np.cos(2 * np.pi * f2 * time + np.pi / 6)


# For the components of the signal and the signal itself
s = signalTot(t)
sLarge = signalTot(tLarge)
s1Large = signal1(tLarge)
s2Large = signal2(tLarge)

# Constants needed
pref = 2e-5


# Oasl - Task 2

# Spl function - Task 2
def spl(p):  # in dB
    return 20 * np.log10(p / pref)


def rms(p):
    return (np.mean(p ** 2)) ** 0.5


def trapezoid(list):
    I = 0
    for i in range(len(list)):
        if i != len(list) - 1:
            I += (list[i] + list[i + 1]) / 2 * 1 / fs
    return I


# Integrating the signals
p0 = 1 / TLarge * trapezoid(sLarge)
p0_1 = 1 / TLarge * trapezoid(s1Large)
p0_2 = 1 / TLarge * trapezoid(s2Large)
print(f"p^0 s  = {p0} [Pa]")
print(f"p^0 s1 = {p0_1} [Pa]")
print(f"p^0 s2 = {p0_2} [Pa]")

# Methods to calculate the SPL of the components and the combined
pprime = sLarge - p0
pprime1 = s1Large - p0_1
pprime2 = s2Large - p0_2

ptilda1 = rms(pprime1)
ptilda2 = rms(pprime2)

SPL1 = spl(ptilda1)
SPL2 = spl(ptilda2)
print(f"SPL1: {SPL1} [dB]")
print(f"SPL2: {SPL2} [dB]")

OSPL = 10 * np.log10(10 ** (SPL1 / 10) + 10 ** (SPL2 / 10))
print(f"OSPL: {OSPL} [dB]")

# Plotting - Task 3
"""
plt.figure()
plt.plot(tLarge, sLarge, "r")
#plt.plot(tLarge, s1Large, "g")
#plt.plot(tLarge, s2Large, "b")
plt.xlim(0, 0.01)
plt.show()
"""

# Fourier Transform - Task 4

N = T * fs  # Window length = total # points
y = fft.fft(s)
x = fft.fftfreq(N, 1 / fs)
print(x)
print(y)

# Plot of signal in f domain
plt.figure()
plt.plot(x, 2 / N * np.abs(y)[:-1])
plt.grid()
plt.xlim(0, 2500)
plt.show()


PSDt = fft.fft(signal.correlate(pprime, pprime, mode="full"))
PSDf = np.abs(y)**2

plt.figure()
plt.plot(x, PSDt[::10] * 2, "r")
plt.plot(x, PSDf[:-1] * 2, "b")
plt.xlim(0, 2500)
plt.yscale("log")
plt.show()

# Welch
f, Pxx_den = signal.welch(s, fs, window="hann", nperseg=256)
plt.figure()
plt.semilogy(f, Pxx_den)
plt.xlim(0, 2500)
plt.show()
# Task 5
