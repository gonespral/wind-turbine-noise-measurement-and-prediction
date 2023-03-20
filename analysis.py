"""
This script contains various plotting functions for the processed data. Pick which one to use at the bottom.
"""
# Use python magic to run self in ipython
# %run analysis.py

import seaborn as sns  # for plotting
import pandas as pd  # for data manipulation
import numpy as np  # for FFT
import matplotlib.pyplot as plt  # for plotting
import pickle

# Sample rate of the data
sample_rate = 48128

# Number of points to smooth FFT by
n_smoothing = 10

# x-axis limits for plots (frequency)
x_lim_0 = 1
x_lim_1 = 20000

# Path to pickle files
pickle_paths = ["data/processed/8m_s.pkl",
                "data/processed/9m_s.pkl",
                "data/processed/10m_s.pkl",
                "data/processed/11m_s.pkl",
                "data/processed/12m_s.pkl"]


def info():
    """
    Print a list of commands that can be used in this script.
    :return:
    """
    print("--- Available commands ---")
    # Get dir() of functions of  global namespace written by user
    for name, obj in globals().items():
        if callable(obj) and obj.__module__ == "__main__":
            print(f"-> {name}()")
    print()

    print("--- Available variables ---")
    print(f"-> sample_rate = {sample_rate}")
    print(f"-> n_smoothing = {n_smoothing}")
    print(f"-> x_lim_0 = {x_lim_0}")
    print(f"-> x_lim_1 = {x_lim_1}")
    print()

    print("--- Loaded data ---")
    for i, measurement in enumerate(measurements):
        print(f"-> measurements[{i}] | {measurement['v_inf']}m/s | {pickle_paths[i]}")
    print()

def plot_denoised_fft_all():
    """
    Display the de-noised FFT of all measurements in a single plot. Uses rolling mean to smooth the data.
    :return:
    """
    fig, ax = plt.subplots()
    for measurement in measurements:
        df_wt_bg_fft = measurement["df_wt_bg_fft"]
        df_wt_bg_fft["fft"] = df_wt_bg_fft["fft"].rolling(n_smoothing).mean()
        ax.plot(df_wt_bg_fft["freq"],
                df_wt_bg_fft["fft"],
                label=f"{measurement['v_inf']}m/s")
    ax.set_xlim(x_lim_0, x_lim_1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Wind Turbine - Background FFT")
    ax.legend()
    plt.show()


def plot_denoised_fft_single(i: int):
    """
    Display denoised FFT of a single measurement.
    :param i: measurement index
    :return:
    """
    fig, ax = plt.subplots()
    df_wt_bg_fft = measurements[i]["df_wt_bg_fft"]
    ax.plot(df_wt_bg_fft["freq"],
            df_wt_bg_fft["fft"],
            label=f"{measurements[i]['v_inf']}m/s")
    ax.set_xlim(x_lim_0, x_lim_1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Wind Turbine - Background FFT")
    ax.legend()
    plt.show()


def plot_denoised_psd_single(i: int):
    """
    Plot Power Spectral Density of a single measurement, using plt.psd Uses rolling mean to smooth the data.
    :param i: measurement index
    :return:
    """
    fig, ax = plt.subplots()
    df_wt_bg_fft = measurements[i]["df_wt_bg_fft"]
    ax.psd(df_wt_bg_fft["fft"],
           Fs=sample_rate,
           label=f"{measurements[i]['v_inf']}m/s")
    ax.set_xscale("log")
    ax.set_xlim(x_lim_0, x_lim_1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Wind Turbine - Background PSD FFT")
    ax.legend()
    plt.show()


def plot_denoised_psd_all():
    """
    Plot Power Spectral Density of all measurements, using plt.psd Uses rolling mean to smooth the data.
    :param i: measurement index
    :return:
    """
    fig, ax = plt.subplots()
    for measurement in measurements:
        df_wt_bg_fft = measurement["df_wt_bg_fft"]
        ax.psd(df_wt_bg_fft["fft"],
               Fs=sample_rate,
               NFFT=1024,
               label=f"{measurement['v_inf']}m/s")
    ax.set_xscale("log")
    ax.set_xlim(x_lim_0, x_lim_1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Wind Turbine - Background PSD FFT")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # Load data from pickle files
    measurements = []
    for path in pickle_paths:
        with open(path, "rb") as f:
            measurements.append(pickle.load(f))
        print(f"Loaded {path}. Found keys: {measurements[-1].keys()}")
    print()

    # Format for accessing data in measurements:
    # eg. df_bg_mean = measurements[0]["df_bg_mean"] # background mean
    # eg. v_inf = measurements[3]["v_inf"] # inflow velocity in m/s

    print("--- WIND TURBINE NOISE ANALYSIS TOOL ---")
    print("Type \'info()\' for a list of commands.")
