import mat73
import pandas as pd
import numpy as np
import scipy.signal as signal
import seaborn as sns
import matplotlib.pyplot as plt
import os

with open('misc/windmill.txt', 'r') as f:
    print(f.read())

# ----------------------------- Config ----------------------------------
p_ref = 2E-5  # Reference pressure (Pa)
sample_rate = 48128  # Hz
f_lower = 500  # Hz
f_upper = 5000 # Hz
scaling_factor = 1
size_x = 6.5 * scaling_factor
size_y = 5 * scaling_factor
x_ticks = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
color_scheme = 'viridis'

# Get paths for data files
bg_paths = []
wt_paths = []
for file in os.listdir('data/downloads'):
    if file.endswith('.mat'):
        if 'Background' in file:
            bg_paths.append(f'data/downloads/{file}')
        elif 'Wind' in file:
            wt_paths.append(f'data/downloads/{file}')

# Uncomment the following lines to use only one background and one wind turbine file
#bg_paths = ['data/downloads/U12_Background.mat']
#wt_paths = ['data/downloads/U12_Wind%20turbine.mat']

# ------------------------- Prepare data --------------------------------

df_bg_list = []
df_wt_list = []
df_bg_no_hanning_list = []
df_wt_no_hanning_list = []
v_inf_list = []
for bg_data_path, wt_data_path in zip(bg_paths, wt_paths):
    v_inf = bg_data_path.split('_')[0].split('U')[1] # Wind speed (m/s)

    # Extract data from mat file
    print(f'[*] Processing files:\n    {bg_data_path}\n    {wt_data_path}')
    df_bg = pd.DataFrame(mat73.loadmat(bg_data_path)['Sig_Mic_bg'])
    df_wt = pd.DataFrame(mat73.loadmat(wt_data_path)['Sig_Mic_rotating'])

    # Swap sample number for seconds
    df_bg.columns = df_bg.columns / sample_rate
    df_wt.columns = df_wt.columns / sample_rate

    # Keep only one row for one microphone
    df_bg = df_bg.iloc[100]
    df_wt = df_wt.iloc[100]

    # Create backup without hanning window
    df_bg_no_hanning = df_bg.copy()
    df_wt_no_hanning = df_wt.copy()

    # Apply hanning window
    df_bg = df_bg * np.hanning(len(df_bg))
    df_wt = df_wt * np.hanning(len(df_wt))

    # Scaling factor defined as sum of squared samples of window function
    hann_scaling_factor = np.sum(np.hanning(len(df_bg)) ** 2)

    # Append to lists
    df_bg_list.append(df_bg)
    df_wt_list.append(df_wt)
    df_bg_no_hanning_list.append(df_bg_no_hanning)
    df_wt_no_hanning_list.append(df_wt_no_hanning)
    v_inf_list.append(float(v_inf))

# Load BPM model data form pickle file
# with open('BPM/saves/BPM_spl.pkl', 'rb') as f:
#     bpm_model = pkl.load(f)
#     bpm_f = bpm_model[0]
#     SPLTBL_s = bpm_model[1]
#     SPLTBL_p = bpm_model[2]
#     SPLTBL_alpha = bpm_model[3]
#     SPLTBL_tot = bpm_model[4]

# Import data from BPM model (data/BPM/data.csv)
df_bpm_list = []
for v_inf in v_inf_list:
    print(f'[*] Importing BPM data from BPM_BYU/data/data{int(v_inf)}.csv...')
    df_bpm = pd.read_csv(f'BPM_BYU/data/data{int(v_inf)}.csv', sep=',')
    # Remove data outside of range
    df_bpm = df_bpm[(df_bpm['freq'] >= f_lower) & (df_bpm['freq'] <= f_upper)]
    df_bpm_list.append(df_bpm)

# Prepare colors for plots
colors = []
for i in range(len(v_inf_list) + 10):
    colors.append(plt.cm.get_cmap(color_scheme, len(v_inf_list))(i))


# ---------------------------------- FFT ----------------------------------

df_bg_fft_list = []
df_wt_fft_list = []

print('[*] Calculating FFT...')
for df_bg, df_wt, v_inf in zip(df_bg_list, df_wt_list, v_inf_list):
    print(f'    v_inf = {v_inf} m/s...')

    # Calculate frequency components
    df_bg_fft = pd.DataFrame(np.fft.fft(df_bg))
    df_wt_fft = pd.DataFrame(np.fft.fft(df_wt))

    # Get absolute value of components and normalize
    df_bg_fft = df_bg_fft.applymap(lambda x: np.abs(x) / len(df_bg))
    df_wt_fft = df_wt_fft.applymap(lambda x: np.abs(x) / len(df_wt))

    # Get frequency axis
    df_bg_fft['freq'] = np.fft.fftfreq(n=len(df_bg_fft), d=1/sample_rate)
    df_wt_fft['freq'] = np.fft.fftfreq(n=len(df_wt_fft), d=1/sample_rate)

    # Keep only positive frequencies
    df_bg_fft = df_bg_fft[df_bg_fft['freq'] >= 0]
    df_wt_fft = df_wt_fft[df_wt_fft['freq'] >= 0]

    # Remove frequencies outside of range
    df_bg_fft = df_bg_fft[(df_bg_fft['freq'] >= f_lower) & (df_bg_fft['freq'] <= f_upper)]
    df_wt_fft = df_wt_fft[(df_wt_fft['freq'] >= f_lower) & (df_wt_fft['freq'] <= f_upper)]

    # Reset index
    df_bg_fft = df_bg_fft.reset_index(drop=True)
    df_wt_fft = df_wt_fft.reset_index(drop=True)

    # Append to lists
    df_bg_fft_list.append(df_bg_fft)
    df_wt_fft_list.append(df_wt_fft)

freq_res_fft = df_bg_fft_list[0]['freq'][1] - df_bg_fft_list[0]['freq'][0]
print(f'    Frequency resolution: {freq_res_fft} Hz')

# --------------------------- Welch PSD -----------------------------------

df_bg_welch_psd_list = []
df_wt_welch_psd_list = []
df_wt_bg_welch_psd_list = []
df_bg_welch_psd_db_list = []
df_wt_welch_psd_db_list = []
df_wt_bg_welch_psd_db_list = []

print('[*] Calculating PSD (Welch)...')
for df_bg_no_hanning, df_wt_no_hanning, v_inf in zip(df_bg_no_hanning_list, df_wt_no_hanning_list, v_inf_list):
    # Evaluate PSD using Welch's method
    print(f'    v_inf = {v_inf} m/s...')

    # Get first column of the dataframe (pressure) - this applies a hanning window
    f, Pxx_den = signal.welch(df_bg_no_hanning,
                              fs=sample_rate,
                              nperseg=2048,
                              return_onesided=True,
                              scaling='density',
                              noverlap=0)
    df_bg_welch_psd = pd.DataFrame({'psd': Pxx_den, 'freq': f})

    f, Pxx_den = signal.welch(df_wt_no_hanning,
                              fs=sample_rate,
                              nperseg=2048,
                              return_onesided=True,
                              scaling='density',
                              noverlap=0)
    df_wt_welch_psd = pd.DataFrame({'psd': Pxx_den, 'freq': f})

    # Denoise PSD
    df_wt_bg_welch_psd = df_wt_welch_psd.copy()
    df_wt_bg_welch_psd['psd'] = df_wt_bg_welch_psd['psd'] - df_bg_welch_psd['psd']

    # Convert to dB
    df_bg_welch_psd_db = df_bg_welch_psd.copy()
    df_wt_welch_psd_db = df_wt_welch_psd.copy()
    df_wt_bg_welch_psd_db = df_wt_bg_welch_psd.copy()
    df_bg_welch_psd_db['psd'] = df_bg_welch_psd_db['psd'].apply(lambda x: 10 * np.log10(x / (p_ref ** 2)))
    df_wt_welch_psd_db['psd'] = df_wt_welch_psd_db['psd'].apply(lambda x: 10 * np.log10(x / (p_ref ** 2)))
    df_wt_bg_welch_psd_db['psd'] = df_wt_bg_welch_psd_db['psd'].apply(lambda x: 10 * np.log10(x / (p_ref ** 2)))

    # Remove frequencies outside of range
    df_bg_welch_psd_db = df_bg_welch_psd_db[(df_bg_welch_psd_db['freq'] >= f_lower) & (df_bg_welch_psd_db['freq'] <= f_upper)]
    df_wt_welch_psd_db = df_wt_welch_psd_db[(df_wt_welch_psd_db['freq'] >= f_lower) & (df_wt_welch_psd_db['freq'] <= f_upper)]
    df_wt_bg_welch_psd_db = df_wt_bg_welch_psd_db[(df_wt_bg_welch_psd_db['freq'] >= f_lower) & (df_wt_bg_welch_psd_db['freq'] <= f_upper)]

    # Reset index
    df_bg_welch_psd_db = df_bg_welch_psd_db.reset_index(drop=True)
    df_wt_welch_psd_db = df_wt_welch_psd_db.reset_index(drop=True)

    # Append to lists
    df_bg_welch_psd_list.append(df_bg_welch_psd)
    df_wt_welch_psd_list.append(df_wt_welch_psd)
    df_bg_welch_psd_db_list.append(df_bg_welch_psd_db)
    df_wt_welch_psd_db_list.append(df_wt_welch_psd_db)
    df_wt_bg_welch_psd_list.append(df_wt_bg_welch_psd)
    df_wt_bg_welch_psd_db_list.append(df_wt_bg_welch_psd_db)

freq_res_psd = df_bg_welch_psd_db['freq'][1] - df_bg_welch_psd_db['freq'][0]
print(f'    Frequency resolution: {freq_res_psd} Hz')

# Plot results for wind turbine per v_inf
fig, ax = plt.subplots(figsize=(size_x, size_y))
for df_wt_welch_psd_db, v_inf, color in zip(df_wt_welch_psd_db_list, v_inf_list, colors):
    ax.plot(df_wt_welch_psd_db['freq'], df_wt_welch_psd_db['psd'], color=color, label=f'{v_inf} m/s')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('PSD [dB\Hz]')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.xticks(x_ticks, x_ticks)
plt.savefig('saves/PSD.png')
plt.show()


# ---------------------------------- SPL ----------------------------------

df_bg_spl_list = []
df_wt_spl_list = []
df_wt_bg_spl_list = []

freq_step = 10 # Hz
freq_bands = np.arange(0, max(df_bg_fft['freq']), freq_step)

print('[*] Calculating SPL...')
for df_bg_welch_psd, df_wt_welch_psd, df_wt_bg_welch_psd, v_inf in zip(df_bg_welch_psd_list, df_wt_welch_psd_list, df_wt_bg_welch_psd_list, v_inf_list):
    # Calculate SPL
    print(f'    v_inf = {v_inf} m/s with freq_step = {freq_step} Hz...')

    df_bg_spl = pd.DataFrame(columns=['freq', 'spl'])
    df_wt_spl = pd.DataFrame(columns=['freq', 'spl'])
    df_wt_bg_spl = pd.DataFrame(columns=['freq', 'spl'])

    for l, c, u in zip(freq_bands[:-1], freq_bands[1:], freq_bands[2:]):
        # Sum PSD in band
        sum_bg = df_bg_welch_psd[(df_bg_welch_psd['freq'] >= l) & (df_bg_welch_psd['freq'] <= u)].sum()
        sum_wt = df_wt_welch_psd[(df_wt_welch_psd['freq'] >= l) & (df_wt_welch_psd['freq'] <= u)].sum()
        sum_wt_bg = df_wt_bg_welch_psd[(df_wt_bg_welch_psd['freq'] >= l) & (df_wt_bg_welch_psd['freq'] <= u)].sum()

        # Add row to dataframe
        df_bg_spl = df_bg_spl.append({'freq': c, 'spl': sum_bg[0] * freq_step}, ignore_index=True)
        df_wt_spl = df_wt_spl.append({'freq': c, 'spl': sum_wt[0] * freq_step}, ignore_index=True)
        df_wt_bg_spl = df_wt_bg_spl.append({'freq': c, 'spl': sum_wt_bg[0] * freq_step}, ignore_index=True)

    # Convert spl column to dB
    df_bg_spl['spl'] = df_bg_spl['spl'].apply(lambda x: 10 * np.log10(x / (p_ref ** 2)))
    df_wt_spl['spl'] = df_wt_spl['spl'].apply(lambda x: 10 * np.log10(x / (p_ref ** 2)))
    df_wt_bg_spl['spl'] = df_wt_bg_spl['spl'].apply(lambda x: 10 * np.log10(x / (p_ref ** 2)))

    # Remove frequencies outside of range
    df_bg_spl = df_bg_spl[(df_bg_spl['freq'] >= f_lower) & (df_bg_spl['freq'] <= f_upper)]
    df_wt_spl = df_wt_spl[(df_wt_spl['freq'] >= f_lower) & (df_wt_spl['freq'] <= f_upper)]
    df_wt_bg_spl = df_wt_bg_spl[(df_wt_bg_spl['freq'] >= f_lower) & (df_wt_bg_spl['freq'] <= f_upper)]

    # Append to lists
    df_bg_spl_list.append(df_bg_spl)
    df_wt_spl_list.append(df_wt_spl)
    df_wt_bg_spl_list.append(df_wt_bg_spl)

# Plot results for wind turbine per v_inf
fig, ax = plt.subplots(figsize=(size_x, size_y))
for df_wt_spl, v_inf, color in zip(df_wt_spl_list, v_inf_list, colors):
    sns.lineplot(x='freq', y='spl', data=df_wt_spl, color=color, label=f'{v_inf} m/s')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('SPL [dB]')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.xticks(x_ticks, x_ticks)
plt.savefig('saves/SPL.png')
plt.show()


# ---------------------------------- SPL_1/3 ----------------------------------

df_bg_spl_1_3_list = []
df_wt_spl_1_3_list = []
df_wt_bg_spl_1_3_list = []

# Evaluate SPL in the frequency domain in 3rd octave bands
# freq_centre = 10 ** (0.1 * np.arange(12, 43))
freq_centre = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
               8000, 10000, 12500, 16000, 20000, 25000, 31500, 40000])
freq_d = 10 ** 0.05
f_upper_1_3 = freq_centre * freq_d
f_lower_1_3 = freq_centre / freq_d

#f = (/ 100.0_dp, 125.0_dp, 160.0_dp, 200.0_dp, 250.0_dp, 315.0_dp, 400.0_dp, 500.0_dp, &
#     630.0_dp, 800.0_dp, 1000.0_dp, 1250.0_dp, 1600.0_dp, 2000.0_dp, 2500.0_dp, 3150.0_dp, &
#     4000.0_dp, 5000.0_dp, 6300.0_dp, 8000.0_dp, 10000.0_dp, 12500.0_dp, 16000.0_dp, &
#     20000.0_dp, 25000.0_dp, 31500.0_dp, 40000.0_dp /)

print('[*] Calculating SPL_1/3...')
for df_bg_welch_psd, df_wt_welch_psd, df_wt_bg_welch_psd, v_inf in zip(df_bg_welch_psd_list, df_wt_welch_psd_list, df_wt_bg_welch_psd_list, v_inf_list):
    # Calculate SPL
    print(f'    v_inf = {v_inf} m/s...')

    df_bg_spl = pd.DataFrame(columns=['freq', 'spl'])
    df_wt_spl = pd.DataFrame(columns=['freq', 'spl'])
    df_wt_bg_spl = pd.DataFrame(columns=['freq', 'spl'])

    # Clear values in dataframe
    df_bg_spl = df_bg_spl.iloc[0:0]
    df_wt_spl = df_wt_spl.iloc[0:0]
    df_wt_bg_spl = df_wt_bg_spl.iloc[0:0]

    for l, c, u in zip(f_lower_1_3, freq_centre, f_upper_1_3):
        # Sum PSD in band
        sum_bg = df_bg_welch_psd[(df_bg_welch_psd['freq'] >= l) & (df_bg_welch_psd['freq'] < u)].sum()
        sum_wt = df_wt_welch_psd[(df_wt_welch_psd['freq'] >= l) & (df_wt_welch_psd['freq'] < u)].sum()
        sum_wt_bg = df_wt_bg_welch_psd[(df_wt_bg_welch_psd['freq'] >= l) & (df_wt_bg_welch_psd['freq'] < u)].sum()

        # Add row to dataframe
        df_bg_spl = df_bg_spl.append({'freq': c, 'spl': sum_bg[0] * (u - l)}, ignore_index=True)
        df_wt_spl = df_wt_spl.append({'freq': c, 'spl': sum_wt[0] * (u - l)}, ignore_index=True)
        df_wt_bg_spl = df_wt_bg_spl.append({'freq': c, 'spl': sum_wt_bg[0] * (u - l)}, ignore_index=True)

    # Convert spl column to dB
    df_bg_spl['spl'] = df_bg_spl['spl'].apply(lambda x: 10 * np.log10(x / (p_ref ** 2)))
    df_wt_spl['spl'] = df_wt_spl['spl'].apply(lambda x: 10 * np.log10(x / (p_ref ** 2)))
    df_wt_bg_spl['spl'] = df_wt_bg_spl['spl'].apply(lambda x: 10 * np.log10(x / (p_ref ** 2)))

    # Remove frequencies outside of range
    df_bg_spl = df_bg_spl[(df_bg_spl['freq'] >= f_lower) & (df_bg_spl['freq'] <= f_upper)]
    df_wt_spl = df_wt_spl[(df_wt_spl['freq'] >= f_lower) & (df_wt_spl['freq'] <= f_upper)]
    df_wt_bg_spl = df_wt_bg_spl[(df_wt_bg_spl['freq'] >= f_lower) & (df_wt_bg_spl['freq'] <= f_upper)]
    # Append to lists
    df_bg_spl_1_3_list.append(df_bg_spl)
    df_wt_spl_1_3_list.append(df_wt_spl)
    df_wt_bg_spl_1_3_list.append(df_wt_bg_spl)

# Estimate error between BPM model and wind turbine
df_error_list = []
for df_wt_spl_1_3, df_bpm, v_inf in zip(df_wt_spl_1_3_list, df_bpm_list, v_inf_list):
    df_error = pd.DataFrame(columns=['freq', 'error'])
    df_error = df_error.iloc[0:0]
    for i in range(len(df_wt_spl_1_3)):
        df_error = df_error.append({'freq': df_wt_spl_1_3.iloc[i]['freq'], 'error': df_wt_spl_1_3.iloc[i]['spl'] - df_bpm.iloc[i]['spl']}, ignore_index=True)
    df_error_list.append(df_error)

# Plot results for wind turbine per v_inf
fig, ax = plt.subplots(figsize=(size_x, size_y))
for df_wt_spl_1_3, df_bpm, v_inf, color in zip(df_wt_spl_1_3_list, df_bpm_list, v_inf_list, colors):
    ax.plot(df_wt_spl_1_3['freq'], df_wt_spl_1_3['spl'], color=color, label=f'{v_inf} m/s')
    ax.plot(df_bpm['freq'], df_bpm['spl'], color=color, linestyle='--')
ax.set_xscale('log')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('SPL [dB]')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.xticks(x_ticks, x_ticks)
plt.savefig('saves/SPL_1_3.png')
plt.show()

# Plot error between wind turbine and BPM model per v_inf
fig, ax = plt.subplots(figsize=(size_x, size_y))
for df_error, v_inf, color in zip(df_error_list, v_inf_list, colors):
    ax.plot(df_error['freq'], df_error['error'], color=color, label=f'{v_inf} m/s')
ax.set_xscale('log')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Error [dB]')
ax.set_title('Error (SPL 1/3)')
ax.legend()
ax.grid(True)
plt.xticks(x_ticks, x_ticks)
plt.savefig('saves/Error_1_3.png')
plt.show()

# ---------------------------------- OSPL ----------------------------------

OSPL_bg_list = []
OSPL_wt_list = []

print('[*] Calculating OSPL...')
for df_bg_welch_psd, df_wt_welch_psd, v_inf in zip(df_bg_welch_psd_list, df_wt_welch_psd_list, v_inf_list):
    # Sum PSD in band f_lower to f_upper
    sum_bg = df_bg_welch_psd[(df_bg_welch_psd['freq'] >= 800) & (df_bg_welch_psd['freq'] <= 3000)].sum()
    sum_wt = df_wt_welch_psd[(df_wt_welch_psd['freq'] >= 800) & (df_wt_welch_psd['freq'] <= 3000)].sum()

    # Calculate OSPL
    ospl_bg = 10 * np.log10((sum_bg[0] * freq_res_psd) / (p_ref ** 2))
    ospl_wt = 10 * np.log10((sum_wt[0] * freq_res_psd) / (p_ref ** 2))

    # Print results
    print(f'    v_inf: {v_inf} m/s')
    print(f'    OSPL background: {ospl_bg} dB')
    print(f'    OSPL wind turbine: {ospl_wt} dB')

    # Append to lists
    OSPL_bg_list.append(ospl_bg)
    OSPL_wt_list.append(ospl_wt)

# Trendline A * log(x) + B
trend_ideal_fn = np.polyfit(np.log10(v_inf_list), OSPL_wt_list, 1)
print(f'[*] Trendline: {trend_ideal_fn[0]}*log(x) + {trend_ideal_fn[1]}')
print(f'[*] R^2: {np.corrcoef(np.log10(v_inf_list), OSPL_wt_list)[0, 1] ** 2}')

# Plot OSPL vs v_inf as scatter plot with line connecting points
fig, ax = plt.subplots(figsize=(size_x, size_y))
sns.scatterplot(x=v_inf_list, y=OSPL_wt_list, ax=ax, label='Wind turbine OSPL', color=colors[1])

# Plot trend line (dB scale)
x = np.linspace(min(v_inf_list), max(v_inf_list), 100)
y = trend_ideal_fn[0] * np.log10(x) + trend_ideal_fn[1]
sns.lineplot(x=x, y=y, ax=ax, color=colors[0], label=f'{trend_ideal_fn[0]:.2f}*log(x) + {trend_ideal_fn[1]:.2f}')

ax.grid(True)
ax.set_ylabel('OSPL [dB]')
ax.set_xlabel('v_inf [m/s]')
ax.grid(True)
plt.tight_layout()
plt.savefig(f'saves/OSPL_800_3000.png', dpi=300)
plt.show()