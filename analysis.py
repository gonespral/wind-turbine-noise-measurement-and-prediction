"""
This script contains various plotting functions for the processed data. Pick which one to use at the bottom.
"""

import mat73  # for loading data files
import seaborn as sns  # for plotting
import pandas as pd  # for data manipulation
import numpy as np  # for FFT
import matplotlib.pyplot as plt  # for plotting
import tqdm  # for progress bar
import pickle

# Sample rate of the data
sample_rate = 48128

# Number of points to smooth FFT by
n_smoothing = 10

# x-axis limits for plots (frequency)
x_lim_0 = 0
x_lim_1 = 200

# Path to pickle files
pickle_paths = ["data/processed/8m_s.pkl",
                "data/processed/9m_s.pkl",
                "data/processed/10m_s.pkl",
                "data/processed/11m_s.pkl",
                "data/processed/12m_s.pkl"]

# Load data from pickle files
measurements = []
for path in pickle_paths:
    with open(path, "rb") as f:
        measurements.append(pickle.load(f))
    print(f"Loaded {path}. Found keys: {measurements[-1].keys()}")


# Format for accessing data in measurements:
# eg. df_bg_mean = measurements[0]["df_bg_mean"] # background mean
# eg. v_inf = measurements[3]["v_inf"] # inflow velocity in m/s

def plot_1():
    """
    Display the de-noised FFT of all measurements in a single plot.
    :return:
    """
    fig, ax = plt.subplots()
    for measurement in measurements:
        df_wt_bg_fft = measurement["df_wt_bg_fft"]
        ax.plot(df_wt_bg_fft["freq"],
                df_wt_bg_fft["fft"],
                label=f"{measurement['v_inf']}m/s")
    ax.set_xlim(x_lim_0, x_lim_1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Background FFT")
    ax.legend()
    plt.show()


if __name__ in "__main__":
    # Pick what plotting function to use
    plot_1()
