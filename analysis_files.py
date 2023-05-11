import mat73
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Get paths for data. Background data is labelled with _Background and wind turbine data is labelled with _Wind in the filenames.
bg_paths = []
wt_paths = []
for file in os.listdir("data/downloads"):
    if file.endswith(".mat"):
        if "Background" in file:
            bg_paths.append(f"data/downloads/{file}")
        elif "Wind" in file:
            wt_paths.append(f"data/downloads/{file}")

# ------------------------- Prepare data --------------------------------

df_bg_list = []
df_wt_list = []
v_inf_list = []
for bg_data_path, wt_data_path in zip(bg_paths, wt_paths):
    # Configuration parameters
    p_ref = 2E-5  # Reference pressure (Pa)
    sample_rate = 48128  # Hz
    v_inf = bg_data_path.split("_")[0].split("U")[1] # Wind speed (m/s)

    # Extract data from mat file
    print(f"[*] Processing files:\n    {bg_data_path}\n    {wt_data_path}")
    df_bg = pd.DataFrame(mat73.loadmat(bg_data_path)['Sig_Mic_bg'])
    df_wt = pd.DataFrame(mat73.loadmat(wt_data_path)['Sig_Mic_rotating'])

    # Swap sample number for seconds
    df_bg.columns = df_bg.columns / sample_rate
    df_wt.columns = df_wt.columns / sample_rate

    # Keep only row 1 for microphone 1
    df_bg = df_bg.iloc[1]
    df_wt = df_wt.iloc[1]

    # Apply hanning window
    df_bg = df_bg * np.hanning(len(df_bg))
    df_wt = df_wt * np.hanning(len(df_wt))

    # Append to lists
    df_bg_list.append(df_bg)
    df_wt_list.append(df_wt)
    v_inf_list.append(v_inf)

# ---------------------------------- FFT ----------------------------------

df_bg_fft_list = []
df_wt_fft_list = []

for df_bg, df_wt, v_inf in zip(df_bg_list, df_wt_list, v_inf_list):
    print(f"[*] Calculating FFT for v_inf = {v_inf} m/s...")

    # Calculate frequency components
    df_bg_fft = pd.DataFrame(np.fft.fft(df_bg))
    df_wt_fft = pd.DataFrame(np.fft.fft(df_wt))

    # Get absolute value of components
    df_bg_fft = df_bg_fft.applymap(lambda x: np.abs(x))
    df_wt_fft = df_wt_fft.applymap(lambda x: np.abs(x))

    # Get frequency axis
    df_bg_fft['freq'] = np.fft.fftfreq(n=len(df_bg_fft), d=1/sample_rate)
    df_wt_fft['freq'] = np.fft.fftfreq(n=len(df_wt_fft), d=1/sample_rate)

    # Keep only positive frequencies
    df_bg_fft = df_bg_fft[df_bg_fft['freq'] >= 0]
    df_wt_fft = df_wt_fft[df_wt_fft['freq'] >= 0]

    # Append to lists
    df_bg_fft_list.append(df_bg_fft)
    df_wt_fft_list.append(df_wt_fft)

print(f"[*] Frequency resolution: {df_bg_fft['freq'][1]} Hz")

# Plot all FFT results in df_bg_fft_list and df_wt_fft_list
print("[*] Plotting results...")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
for df_bg_fft, df_wt_fft, v_inf in zip(df_bg_fft_list, df_wt_fft_list, v_inf_list):
    sns.lineplot(data=df_bg_fft, x='freq', y=0, label=f'Background (v_inf = {v_inf} m/s)', ax=ax1)
    sns.lineplot(data=df_wt_fft, x='freq', y=0, label=f'Wind turbine (v_inf = {v_inf} m/s)', ax=ax2)
ax1.set_title("FFT")
ax1.grid(True)
ax1.set_ylabel('Pressure [Pa]')
ax2.grid(True)
ax2.set_ylabel('Pressure [Pa]')
ax2.set_xlabel('Frequency [Hz]')
plt.xscale('log')
plt.show()


# ---------------------------------- PSD ----------------------------------

df_bg_psd_db_list = []
df_wt_psd_db_list = []

for df_bg_fft, df_wt_fft, v_inf in zip(df_bg_fft_list, df_wt_fft_list, v_inf_list):
    # Evaluate PSD in the frequency domain
    print(f"[*] Calculating PSD for v_inf = {v_inf} m/s...")

    df_bg_psd = df_bg_fft.applymap(lambda x: x ** 2)
    df_wt_psd = df_wt_fft.applymap(lambda x: x ** 2)

    # Double the amplitudes to account for one sided spectrum
    df_bg_psd = df_bg_psd.applymap(lambda x: 2 * x)
    df_wt_psd = df_wt_psd.applymap(lambda x: 2 * x)

    # Convert to dB
    df_bg_psd_db = df_bg_psd.applymap(lambda x: 10 * np.log10(x / (p_ref ** 2)))
    df_wt_psd_db = df_wt_psd.applymap(lambda x: 10 * np.log10(x / (p_ref ** 2)))

    # Append to lists
    df_bg_psd_db_list.append(df_bg_psd_db)
    df_wt_psd_db_list.append(df_wt_psd_db)

# Plot results
print("[*] Plotting results...")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
for df_bg_psd_db, df_wt_psd_db, v_inf in zip(df_bg_psd_db_list, df_wt_psd_db_list, v_inf_list):
    sns.lineplot(data=df_bg_psd_db, x='freq', y=0, label=f'Background (v_inf = {v_inf} m/s)', ax=ax1)
    sns.lineplot(data=df_wt_psd_db, x='freq', y=0, label=f'Wind turbine (v_inf = {v_inf} m/s)', ax=ax2)
ax1.set_title("PSD")
ax1.grid(True)
ax1.set_ylabel('Pressure [dB]')
ax2.grid(True)
ax2.set_ylabel('Pressure [dB]')
ax2.set_xlabel('Frequency [Hz]')
plt.xscale('log')
plt.show()

# ---------------------------------- SPL ----------------------------------

df_bg_spl_list = []
df_wt_spl_list = []

freq_step = 50
freq_bands = np.arange(0, max(df_bg_fft["freq"]), freq_step)

for df_bg_psd_db, df_wt_psd_db, v_inf in zip(df_bg_psd_db_list, df_wt_psd_db_list, v_inf_list):

    # Evaluate SPL in the frequency domain over freq_step Hz bands
    print(f"[*] Calculating SPL for v_inf = {v_inf} m/s...")

    df_bg_spl = pd.DataFrame(columns=['freq', 'spl'])
    df_wt_spl = pd.DataFrame(columns=['freq', 'spl'])

    for l, c, u in zip(freq_bands[:-1], freq_bands[1:], freq_bands[2:]):
        # Sum PSD in band
        sum_bg = df_bg_psd[(df_bg_psd['freq'] >= l) & (df_bg_psd['freq'] < u)].sum()
        sum_wt = df_wt_psd[(df_wt_psd['freq'] >= l) & (df_wt_psd['freq'] < u)].sum()

        # Add row to dataframe
        df_bg_spl = df_bg_spl.append({'freq': c, 'spl': sum_bg[0]}, ignore_index=True)
        df_wt_spl = df_wt_spl.append({'freq': c, 'spl': sum_wt[0]}, ignore_index=True)

    # Convert freq column to dB
    df_bg_spl['spl'] = df_bg_spl['spl'].apply(lambda x: 10 * np.log10(x / (p_ref ** 2)))
    df_wt_spl['spl'] = df_wt_spl['spl'].apply(lambda x: 10 * np.log10(x / (p_ref ** 2)))

    # Remove frequencies outside of range 0 - 500Hz
    df_bg_spl = df_bg_spl[(df_bg_spl['freq'] >= 2000) & (df_bg_spl['freq'] <= 7000)]
    df_wt_spl = df_wt_spl[(df_wt_spl['freq'] >= 2000) & (df_wt_spl['freq'] <= 7000)]

    # Append to lists
    df_bg_spl_list.append(df_bg_spl)
    df_wt_spl_list.append(df_wt_spl)

# Plot results
print("[*] Plotting results...")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
for df_bg_spl, df_wt_spl, v_inf in zip(df_bg_spl_list, df_wt_spl_list, v_inf_list):
    sns.lineplot(data=df_bg_spl, x='freq', y='spl', label=f'Background (v_inf = {v_inf} m/s)', ax=ax1)
    sns.lineplot(data=df_wt_spl, x='freq', y='spl', label=f'Wind turbine (v_inf = {v_inf} m/s)', ax=ax2)
ax1.set_title("SPL")
ax1.grid(True)
ax1.set_ylabel('Pressure [dB]')
ax2.grid(True)
ax2.set_ylabel('Pressure [dB]')
ax2.set_xlabel('Frequency [Hz]')
plt.xscale('log')
plt.show()

# ---------------------------------- SPL_1/3 ----------------------------------

df_bg_spl_1_3_list = []
df_wt_spl_1_3_list = []

# Evaluate SPL in the frequency domain in 3rd octave bands
freq_centre = 10 ** (0.1 * np.arange(12, 43))
freq_d = 10 ** 0.05
freq_upper = freq_centre * freq_d
freq_lower = freq_centre / freq_d

for df_bg_psd, df_wt_psd, v_inf in zip(df_bg_psd_db_list, df_wt_psd_db_list, v_inf_list):

    print(f"[*] Calculating SPL_1/3 for v_inf = {v_inf} m/s...")

    df_bg_spl = pd.DataFrame(columns=['freq', 'spl'])
    df_wt_spl = pd.DataFrame(columns=['freq', 'spl'])

    # Clear values in dataframe
    df_bg_spl = df_bg_spl.iloc[0:0]
    df_wt_spl = df_wt_spl.iloc[0:0]

    for l, c, u in zip(freq_lower, freq_centre, freq_upper):
        # Sum PSD in band
        sum_bg = df_bg_psd[(df_bg_psd['freq'] >= l) & (df_bg_psd['freq'] < u)].sum()
        sum_wt = df_wt_psd[(df_wt_psd['freq'] >= l) & (df_wt_psd['freq'] < u)].sum()

        # Add row to dataframe
        df_bg_spl = df_bg_spl.append({'freq': c, 'spl': sum_bg[0]}, ignore_index=True)
        df_wt_spl = df_wt_spl.append({'freq': c, 'spl': sum_wt[0]}, ignore_index=True)

    # Convert freq column to dB
    df_bg_spl['spl'] = df_bg_spl['spl'].apply(lambda x: 10 * np.log10(x / (p_ref ** 2)))
    df_wt_spl['spl'] = df_wt_spl['spl'].apply(lambda x: 10 * np.log10(x / (p_ref ** 2)))

    # Remove frequencies outside of range 0 - 500Hz
    df_bg_spl = df_bg_spl[(df_bg_spl['freq'] >= 2000) & (df_bg_spl['freq'] <= 7000)]
    df_wt_spl = df_wt_spl[(df_wt_spl['freq'] >= 2000) & (df_wt_spl['freq'] <= 7000)]

    # Append to lists
    df_bg_spl_1_3_list.append(df_bg_spl)
    df_wt_spl_1_3_list.append(df_wt_spl)

    for df in df_bg_spl_1_3_list:
        print(df.head())

# Plot results
print("[*] Plotting results...")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
for df_bg_spl, df_wt_spl, v_inf in zip(df_bg_spl_1_3_list, df_wt_spl_1_3_list, v_inf_list):
    sns.lineplot(data=df_bg_spl, x='freq', y='spl', label=f'Background (v_inf = {v_inf} m/s)', ax=ax1)
    sns.lineplot(data=df_wt_spl, x='freq', y='spl', label=f'Wind turbine (v_inf = {v_inf} m/s)', ax=ax2)
ax1.set_title("SPL 1/3 octave")
ax1.grid(True)
ax1.set_ylabel('Pressure [dB]')
ax2.grid(True)
ax2.set_ylabel('Pressure [dB]')
ax2.set_xlabel('Frequency [Hz]')
plt.xscale('log')
plt.show()

# ---------------------------------- OSPL ----------------------------------

# Evaluate OSPL for freq range
freq_lower = 800
freq_upper = 3000

OSPL_bg_list = []
OSPL_wt_list = []

for df_bg_psd, df_wt_psd, v_inf in zip(df_bg_psd_db_list, df_wt_psd_db_list, v_inf_list):
    # Sum PSD in band
    sum_bg = df_bg_psd[(df_bg_psd['freq'] >= freq_lower) & (df_bg_psd['freq'] <= freq_upper)].sum()
    sum_wt = df_wt_psd[(df_wt_psd['freq'] >= freq_lower) & (df_wt_psd['freq'] <= freq_upper)].sum()

    # Calculate OSPL
    ospl_bg = 10 * np.log10(sum_bg[0] / (p_ref ** 2))
    ospl_wt = 10 * np.log10(sum_wt[0] / (p_ref ** 2))

    # Print results
    print(f"[*] v_inf: {v_inf} m/s")
    print(f"    OSPL background: {ospl_bg} dB")
    print(f"    OSPL wind turbine: {ospl_wt} dB")

    # Append to lists
    OSPL_bg_list.append(ospl_bg)
    OSPL_wt_list.append(ospl_wt)

# Plot OSPL vs v_inf
print("[*] Plotting results...")
plt.plot(v_inf_list, OSPL_bg_list, label='Background')
plt.plot(v_inf_list, OSPL_wt_list, label='Wind turbine')
plt.title("OSPL")
plt.grid(True)
plt.ylabel('Pressure [dB]')
plt.xlabel('Wind speed [m/s]')
plt.legend()
plt.show()
