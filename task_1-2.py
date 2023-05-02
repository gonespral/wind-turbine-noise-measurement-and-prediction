import mat73
import pandas as pd
import numpy as np
import tqdm
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# --- Configuration Parameters ---
sample_rate = 48128
chunk_size = 1000
bg_data_path = "data/downloads/U08_Background.mat"
wt_data_path = "data/downloads/U08_Wind%20turbine.mat"

with open("misc/windmill.txt", "r") as f:
    print(f.read())
print("--- WIND TURBINE NOISE ANALYSIS TOOL ---\n")

# Extract data from mat file
print(f"[*] Processing files: {bg_data_path} and {wt_data_path}")
df_bg = pd.DataFrame(mat73.loadmat(bg_data_path)['Sig_Mic_bg'])
df_wt = pd.DataFrame(mat73.loadmat(wt_data_path)['Sig_Mic_rotating'])

# Swap sample number for seconds
df_bg.columns = df_bg.columns / sample_rate
df_wt.columns = df_wt.columns / sample_rate

# Split data into chunks. Splitting is done with chunk_size columns. Each new dataframe is added to an array bg_chunks.
print(f"[*] Splitting data into chunks of size {chunk_size}")
bg_chunks = []
wt_chunks = []
for i in range(0, len(df_bg.columns), chunk_size):
    bg_chunks.append(df_bg.iloc[:, i:i + chunk_size])
    wt_chunks.append(df_wt.iloc[:, i:i + chunk_size])
print(f"[*] {len(bg_chunks)} chunks created")
print(f"[*] Frequency resolution: {sample_rate / chunk_size} Hz")

# Each chunk has columns indicating sample number, and rows indicating microphone number.
# Do FFT for each microphone and print the results.
print("[*] Performing FFT on each chunk")
bg_chunks_fft = []
wt_chunks_fft = []
for bg_chunk, wt_chunk in tqdm.tqdm(zip(bg_chunks, wt_chunks),
                                    total=len(bg_chunks),
                                    unit="chunk pair"):
    for chunk in [bg_chunk, wt_chunk]:
        # Create a new dataframe to store the FFT results for this chunk
        chunk_fft = pd.DataFrame()  # index is mic, columns are freq, values are FFT magnitude
        for i, row in chunk.iterrows():  # "i" is mic number, row is the data for that mic
            fft = np.abs(np.fft.fft(row))  # taking absolute value to get magnitude
            freq = np.fft.fftfreq(len(row), 1 / sample_rate)  # get frequency for each FFT bin
            fft = 2 * np.abs(fft[:len(fft) // 2])  # only take first half of FFT bins
            freq = freq[:len(freq) // 2]  # only take first half of frequencies
            # Add row to chunk_fft dataframe
            chunk_fft = chunk_fft.append(pd.Series(fft, index=freq, name=i))
        # Add chunk_fft to bg_chunks_fft or wt_chunks_fft
        if chunk is bg_chunk:
            bg_chunks_fft.append(chunk_fft)
        else:
            wt_chunks_fft.append(chunk_fft)

    break

# Plot chunk 1 : x freq, y value, hue mic
print("[*] Plotting chunk 1")
sns.lineplot(data=bg_chunks_fft[0].T, x=bg_chunks_fft[0].T.index, y=0, hue=bg_chunks_fft[0].T.columns)

