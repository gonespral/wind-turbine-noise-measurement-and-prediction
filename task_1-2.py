import mat73
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration Parameters ---
sample_rate = 48128
p_ref = 2e-5
num_chunks = 10
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

# Split data into chunks
print("[*] Splitting data into chunks...")
chunk_size = int(len(df_bg) / num_chunks) # samples
print("Chunk size:", chunk_size)
df_bg_chunks = np.array_split(df_bg, num_chunks)
df_wt_chunks = np.array_split(df_wt, num_chunks)

print(df_bg_chunks)

# ---------------------------------- FFT ----------------------------------

# FFT for each microphone
df_bg_fft_chunks = []
df_wt_fft_chunks = []
print("[*] Calculating FFT...")
for chunk in df_bg_chunks:
    df_bg_fft_chunks.append(pd.DataFrame(np.fft.fft(chunk)))
for chunk in df_wt_chunks:
    df_wt_fft_chunks.append(pd.DataFrame(np.fft.fft(chunk)))
print("Length of df_bg_fft_chunks:", len(df_bg_fft_chunks))
print("Length of df_wt_fft_chunks:", len(df_wt_fft_chunks))

# Get modulus of complex numbers
print("[*] Getting modulus of complex numbers...")
for chunk in df_bg_fft_chunks:
    chunk = chunk.applymap(lambda x: np.abs(x))
for chunk in df_wt_fft_chunks:
    chunk = chunk.applymap(lambda x: np.abs(x))

# Get frequency
print("[*] Getting frequency...")
for chunk in df_bg_fft_chunks:
    chunk['freq'] = np.fft.fftfreq(len(chunk), 1 / sample_rate)
for chunk in df_wt_fft_chunks:
    chunk['freq'] = np.fft.fftfreq(len(chunk), 1 / sample_rate)

# Keep only positive frequencies
print("[*] Keeping only positive frequencies...")
for chunk in df_bg_fft_chunks:
    chunk = chunk[chunk['freq'] >= 0]
for chunk in df_wt_fft_chunks:
    chunk = chunk[chunk['freq'] >= 0]

# Print results
#print("[*] Printing results for FFT...")
#print(df_bg_fft)
#print(df_wt_fft)

# Plot results
print("[*] Plotting results...")
sns.lineplot(data=df_bg_fft_chunks[0], x='freq', y=0, label='Background')
plt.xscale('log')
plt.grid(True)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Pressure (Pa)')
plt.show()

sns.lineplot(data=df_wt_fft_chunks[0], x='freq', y=0, label='Wind turbine')
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

# Average PSD over chunks, for each microphone
print("[*] Averaging PSD over chunks...")

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






