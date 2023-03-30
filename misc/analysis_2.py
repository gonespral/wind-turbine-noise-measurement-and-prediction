import mat73
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample rate of the data
sample_rate = 48128

# Paths to the data files. Run download_data.py to download the data.
bg_data_path = "../data/downloads/U09_Background.mat"
wt_data_path = "../data/downloads/U09_Wind%20turbine.mat"

# Extract data from mat file
print(f"Processing files: {bg_data_path} and {wt_data_path}")

# Extract data from mat file
print("Extracting data from mat files...")
bg_data_dict = mat73.loadmat(bg_data_path)
wt_data_dict = mat73.loadmat(wt_data_path)

# Convert to dataframe
df_bg = pd.DataFrame(bg_data_dict['Sig_Mic_bg'])
df_wt = pd.DataFrame(wt_data_dict['Sig_Mic_rotating'])

# Swap sample number for seconds
df_bg.columns = df_bg.columns / sample_rate
df_wt.columns = df_wt.columns / sample_rate

# Get mean of each col with concat. Index is time in s, value is mean of all mics (rows)
df_bg_mean = pd.concat([df_bg.mean(axis=0)], axis=1)
df_wt_mean = pd.concat([df_wt.mean(axis=0)], axis=1)

# Rename cols
df_bg_mean.columns = ["mean"]
df_wt_mean.columns = ["mean"]

# Apply hamming window to data
print("Applying hamming window...")
for col in df_bg_mean.columns:
    df_bg_mean[col] = df_bg_mean[col] * np.hamming(df_bg_mean.shape[0])
for col in df_wt_mean.columns:
    df_wt_mean[col] = df_wt_mean[col] * np.hamming(df_wt_mean.shape[0])

# Calculate the FFT
print("Evaluating FFT...")
ax_fft_bg = np.fft.fft(df_bg_mean["mean"].values)
ax_fft_wt = np.fft.fft(df_wt_mean["mean"].values)
ax_freq_bg = np.fft.fftfreq(df_bg_mean["mean"].values.size, d=1 / sample_rate)
ax_freq_wt = np.fft.fftfreq(df_wt_mean["mean"].values.size, d=1 / sample_rate)
# Create new dataframe for FFT results (absolute value, only positive frequencies)
df_bg_fft = pd.DataFrame({"freq": ax_freq_bg, "fft": np.abs(ax_fft_bg)})
df_bg_fft = df_bg_fft.loc[df_bg_fft["freq"] > 0]
df_wt_fft = pd.DataFrame({"freq": ax_freq_wt, "fft": np.abs(ax_fft_wt)})
df_wt_fft = df_wt_fft.loc[df_wt_fft["freq"] > 0]
print(f"Frequency resolution: {ax_freq_bg[1] - ax_freq_bg[0]} Hz")

# De-noise the WT signal - this is done by subtracting the BG FFT from the WT FFT.
print("De-noising WT signal...")
df_wt_bg_fft = df_wt_fft.copy()
df_wt_bg_fft["fft"] = df_wt_bg_fft["fft"] - df_bg_fft["fft"]
# Remove negative values
df_wt_bg_fft.loc[df_wt_bg_fft["fft"] < 0, "fft"] = 0
# Set all positive FFT values to 1
df_wt_bg_fft.loc[df_wt_bg_fft["fft"] > 0, "fft"] = 1

# Convert FFT value to SPL (dB) : SPL = 20 * log10(FFT / 2e-5)
print("Converting FFT to SPL...")
df_bg_fft["fft"] = 20 * np.log10(df_bg_fft["fft"] / 2e-5)
df_wt_fft["fft"] = 20 * np.log10(df_wt_fft["fft"] / 2e-5)

# Rename column to SPL
df_bg_fft.columns = ["freq", "SPL"]
df_wt_fft.columns = ["freq", "SPL"]
df_wt_bg_fft.columns = ["freq", "value"]

# Now calculate time-dependent FFT
# Iterate over every sample_rate/2 samples

print(f"Evaluating FFT in time-frequency plane with time_step of {sample_step} samples...")
df_bg_fft_t = pd.DataFrame()
df_wt_fft_t = pd.DataFrame()