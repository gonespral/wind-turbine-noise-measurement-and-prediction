import mat73
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration Parameters ---
sample_rate = 48128
p_ref = 2e-5
bg_data_path = "data/downloads/U08_Background.mat"
wt_data_path = "data/downloads/U08_Wind%20turbine.mat"

# Extract data from mat file
print(f"[*] Processing files: {bg_data_path} and {wt_data_path}")
df_bg = pd.DataFrame(mat73.loadmat(bg_data_path)['Sig_Mic_bg'])
df_wt = pd.DataFrame(mat73.loadmat(wt_data_path)['Sig_Mic_rotating'])

# Swap sample number for seconds
df_bg.columns = df_bg.columns / sample_rate
df_wt.columns = df_wt.columns / sample_rate

# Keep only row 1 - microphone 1
df_bg = df_bg.iloc[1]
df_wt = df_wt.iloc[1]

# ---------------------------------- FFT ----------------------------------

# FFT for microphone 1
print("[*] Calculating FFT...")
df_bg_fft = pd.DataFrame(np.fft.fft(df_bg))  # Same length as df_bg.
df_wt_fft = pd.DataFrame(np.fft.fft(df_wt))
print("Length of df_bg_fft:", len(df_bg_fft))
print("Length of df_wt_fft:", len(df_wt_fft))

# Get modulus of complex numbers
print("[*] Getting modulus of complex numbers...")
df_bg_fft = df_bg_fft.applymap(lambda x: np.abs(x))
df_wt_fft = df_wt_fft.applymap(lambda x: np.abs(x))

# Get frequency
print("[*] Getting frequency...")
df_bg_fft['freq'] = np.fft.fftfreq(len(df_bg_fft), 1 / sample_rate)
df_wt_fft['freq'] = np.fft.fftfreq(len(df_wt_fft), 1 / sample_rate)

# Keep only positive frequencies
df_bg_fft = df_bg_fft[df_bg_fft['freq'] >= 0]
df_wt_fft = df_wt_fft[df_wt_fft['freq'] >= 0]

# Double the values
#df_bg_fft = df_bg_fft * 2
#df_wt_fft = df_wt_fft * 2

# Print results
#print("[*] Printing results for FFT...")
#print(df_bg_fft)
#print(df_wt_fft)

# Plot results
print("[*] Plotting results...")
sns.lineplot(data=df_bg_fft, x='freq', y=0, label='Background')
plt.xscale('log')
plt.grid(True)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Pressure (Pa)')
plt.show()

sns.lineplot(data=df_wt_fft, x='freq', y=0, label='Wind turbine')
plt.xscale('log')
plt.grid(True)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Pressure (Pa)')
plt.show()

# ---------------------------------- PSD ----------------------------------

# Evaluate PSD in the frequency domain
print("[*] Calculating PSD...")
df_bg_psd = df_bg_fft.applymap(lambda x: x ** 2 / sample_rate)
df_wt_psd = df_wt_fft.applymap(lambda x: x ** 2 / sample_rate)

# Convert to dB
print("[*] Converting to dB...")
df_bg_psd = df_bg_psd.applymap(lambda x: 10 * np.log10(x / p_ref ** 2))
df_wt_psd = df_wt_psd.applymap(lambda x: 10 * np.log10(x / p_ref ** 2))

# Print results
#print("[*] Printing results for PSD...")
#print(df_bg_psd)
#print(df_wt_psd)

# Average PSD over chunks

# Plot results
print("[*] Plotting results...")
sns.lineplot(data=df_bg_psd, x='freq', y=0, label='Background')
plt.xscale('log')
plt.grid(True)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Pressure (dB/Hz)')
plt.show()

sns.lineplot(data=df_wt_psd, x='freq', y=0, label='Wind turbine')
plt.xscale('log')
plt.grid(True)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Pressure (dB/Hz)')
plt.show()






