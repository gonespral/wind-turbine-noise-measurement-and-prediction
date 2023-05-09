import mat73
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

test_mode = False # Choose whether to run with test function (task 1) or with wind tunnel data.

# ------------------------- Prepare data --------------------------------

if not test_mode:
    # Configuration parameters
    p_ref = 2E-5  # Reference pressure (Pa)
    sample_rate = 48128  # Hz
    bg_data_path = "data/downloads/U12_Background.mat" # Background noise
    wt_data_path = "data/downloads/U12_Wind%20turbine.mat" # Wind turbine noise
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

else:
    # Configuration parameters
    p_ref = 2E-5  # Reference pressure (Pa)
    sample_rate = 51200 # Hz
    t_max = 20 # seconds
    A_1 = 0.02 * np.sqrt(2)
    A_2 = 0.002 * np.sqrt(2)
    f_1 = 500 # Hz
    f_2 = 2000 # Hz\
    v_inf = 0

    # Generate test data sigal
    print("[*] Running with test function...")
    def f(t):
        A = A_1 * np.sin((2 * np.pi * f_1 * t) + (np.pi / 4))
        B = A_2 * np.cos((2 * np.pi * f_2 * t) + (np.pi / 6))
        return A + B

    # Generate values
    x = np.arange(0, t_max, 1 / sample_rate)
    y = f(x)

    # Apply hanning window
    y = y * np.hanning(len(y))

    # Assemble pandas series with index number: time and values: pressure
    df_bg = pd.Series(y, index=x)
    df_wt = pd.Series(y, index=x)


# ---------------------------------- FFT ----------------------------------

print("[*] Calculating FFT...")

# Calculate frequency components
df_bg_fft = pd.DataFrame(np.fft.fft(df_bg))
df_wt_fft = pd.DataFrame(np.fft.fft(df_wt))

# Get absolute value of components and normalize (divide by number of samples) to get an estimation of pressure level
df_bg_fft = df_bg_fft.applymap(lambda x: np.abs(x)/len(df_bg_fft))
df_wt_fft = df_wt_fft.applymap(lambda x: np.abs(x)/len(df_wt_fft))

# Get frequency axis
df_bg_fft['freq'] = np.fft.fftfreq(n=len(df_bg_fft), d=1/sample_rate)
df_wt_fft['freq'] = np.fft.fftfreq(n=len(df_wt_fft), d=1/sample_rate)

print(f"[*] Frequency resolution: {df_bg_fft['freq'][1]} Hz")

# Keep only positive frequencies
df_bg_fft = df_bg_fft[df_bg_fft['freq'] >= 0]
df_wt_fft = df_wt_fft[df_wt_fft['freq'] >= 0]

# Double the values
df_bg_fft = df_bg_fft * 2
df_wt_fft = df_wt_fft * 2

# Plot results
print("[*] Plotting results...")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
sns.lineplot(data=df_bg_fft, x='freq', y=0, label='Background', ax=ax1)
sns.lineplot(data=df_wt_fft, x='freq', y=0, label='Wind turbine', ax=ax2)
ax1.set_title(f"FFT (v_inf = {v_inf} m/s)")
ax1.grid(True)
ax1.set_ylabel('Pressure [Pa]')
ax2.grid(True)
ax2.set_ylabel('Pressure [Pa]')
ax2.set_xlabel('Frequency [Hz]')
plt.xscale('log')
plt.show()

# ---------------------------------- PSD ----------------------------------

# Evaluate PSD in the frequency domain
print("[*] Calculating PSD...")
df_bg_psd = df_bg_fft.applymap(lambda x: x ** 2)
df_wt_psd = df_wt_fft.applymap(lambda x: x ** 2)

# Plot results
print("[*] Plotting results...")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
sns.lineplot(data=df_bg_psd, x='freq', y=0, label='Background', ax=ax1)
sns.lineplot(data=df_wt_psd, x='freq', y=0, label='Wind turbine', ax=ax2)
ax1.set_title(f"PSD (v_inf = {v_inf} m/s)")
ax1.grid(True)
ax1.set_ylabel('PSD [Pa^2/Hz]')
ax2.grid(True)
ax2.set_ylabel('PSD [Pa^2/Hz]')
ax2.set_xlabel('Frequency [Hz]')
plt.xscale('log')
plt.show()

# Convert to dB
df_bg_psd_db = df_bg_psd.applymap(lambda x: 10 * np.log10(x / (p_ref ** 2)))
df_wt_psd_db = df_wt_psd.applymap(lambda x: 10 * np.log10(x / (p_ref ** 2)))

# Plot results
print("[*] Plotting results...")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
sns.lineplot(data=df_bg_psd_db, x='freq', y=0, label='Background', ax=ax1)
sns.lineplot(data=df_wt_psd_db, x='freq', y=0, label='Wind turbine', ax=ax2)
ax1.set_title(f"PSD (v_inf = {v_inf} m/s)")
ax1.grid(True)
ax1.set_ylabel('PSD [dB/Hz]')
ax2.grid(True)
ax2.set_ylabel('PSD [dB/Hz]')
ax2.set_xlabel('Frequency [Hz]')
plt.xscale('log')
plt.show()

# Add some smoothing with rolling average
df_bg_psd_db['smooth'] = df_bg_psd_db[0].rolling(10).mean()
df_wt_psd_db['smooth'] = df_wt_psd_db[0].rolling(10).mean()

# Plot results
print("[*] Plotting results...")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
sns.lineplot(data=df_bg_psd_db, x='freq', y='smooth', label='Background', ax=ax1)
sns.lineplot(data=df_wt_psd_db, x='freq', y='smooth', label='Wind turbine', ax=ax2)
ax1.set_title(f"PSD rolling average (v_inf = {v_inf} m/s)")
ax1.grid(True)
ax1.set_ylabel('PSD [dB/Hz]')
ax2.grid(True)
ax2.set_ylabel('PSD [dB/Hz]')
ax2.set_xlabel('Frequency [Hz]')
plt.xscale('log')
plt.show()

# ---------------------------------- SPL ----------------------------------

# Evaluate SPL in the frequency domain over freq_step Hz bands
print("[*] Calculating SPL...")
freq_step = 50
freq_bands = np.arange(0, max(df_bg_fft["freq"]), freq_step)

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

# Plot results
print("[*] Plotting results...")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
sns.lineplot(data=df_bg_spl, x='freq', y='spl', label='Background', ax=ax1)
sns.lineplot(data=df_wt_spl, x='freq', y='spl', label='Wind turbine', ax=ax2)
ax1.set_title(f"SPL_{freq_step} (v_inf = {v_inf} m/s)")
ax1.grid(True)
ax1.set_ylabel('SPL [dB]')
ax2.grid(True)
ax2.set_ylabel('SPL [dB]')
ax2.set_xlabel('Frequency [Hz]')
plt.xscale('linear')
plt.show()

# ---------------------------------- SPL_1/3 ----------------------------------

# Evaluate SPL in the frequency domain in 3rd octave bands
print("[*] Calculating SPL_1/3...")
freq_centre = 10 ** (0.1 * np.arange(12, 43))
freq_d = 10 ** 0.05
freq_upper = freq_centre * freq_d
freq_lower = freq_centre / freq_d

df_bg_spl = pd.DataFrame(columns=['freq', 'spl'])
df_wt_spl = pd.DataFrame(columns=['freq', 'spl'])
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

# Plot results
print("[*] Plotting results...")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
sns.lineplot(data=df_bg_spl, x='freq', y='spl', label='Background', ax=ax1)
sns.lineplot(data=df_wt_spl, x='freq', y='spl', label='Wind turbine', ax=ax2)
ax1.set_title(f"SPL_1/3 (v_inf = {v_inf} m/s)")
ax1.grid(True)
ax1.set_ylabel('SPL [dB]')
ax2.grid(True)
ax2.set_ylabel('SPL [dB]')
ax2.set_xlabel('Frequency [Hz]')
plt.xscale('linear')
plt.show()

# ---------------------------------- OSPL ----------------------------------

# Evaluate OSPL for freq range
print("[*] Calculating OSPL...")
freq_lower = 800
freq_upper = 3000

# Sum PSD in band
sum_bg = df_bg_psd[(df_bg_psd['freq'] >= freq_lower) & (df_bg_psd['freq'] <= freq_upper)].sum()
sum_wt = df_wt_psd[(df_wt_psd['freq'] >= freq_lower) & (df_wt_psd['freq'] <= freq_upper)].sum()

# Calculate OSPL
ospl_bg = 10 * np.log10(sum_bg[0] / (p_ref ** 2))
ospl_wt = 10 * np.log10(sum_wt[0] / (p_ref ** 2))

# Print results
print(f"OSPL background: {ospl_bg} dB")
print(f"OSPL wind turbine: {ospl_wt} dB")