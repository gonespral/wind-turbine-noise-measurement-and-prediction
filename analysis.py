import mat73
import pandas as pd
import numpy as np
import scipy.signal as signal
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle as pkl
from playsound import playsound

with open('misc/windmill.txt', 'r') as f:
    print(f.read())

# ----------------------------- Config ----------------------------------
p_ref = 2E-5  # Reference pressure (Pa)
sample_rate = 48128  # Hz
f_lower = 500  # Hz
f_upper = 5000 # Hz
size_x = 8
size_y = 4
x_ticks = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

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
#bg_paths = ['data/downloads/U08_Background.mat']
#wt_paths = ['data/downloads/U08_Wind%20turbine.mat']

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
with open('BPM/saves/BPM_spl.pkl', 'rb') as f:
    bpm_model = pkl.load(f)
    bpm_f = bpm_model[0]
    SPLTBL_s = bpm_model[1]
    SPLTBL_p = bpm_model[2]
    SPLTBL_alpha = bpm_model[3]
    SPLTBL_tot = bpm_model[4]

# ---------------------------------- FFT ----------------------------------

df_bg_fft_list = []
df_wt_fft_list = []

for df_bg, df_wt, v_inf in zip(df_bg_list, df_wt_list, v_inf_list):
    print(f'[*] Calculating FFT for v_inf = {v_inf} m/s...')

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
print(f'[*] Frequency resolution: {freq_res_fft} Hz')

# Plot all FFT results in df_bg_fft_list and df_wt_fft_list
print('[*] Plotting results...')
fig, ax = plt.subplots()
fig.set_size_inches(size_x, size_y)
for df_bg_fft, df_wt_fft, v_inf in zip(df_bg_fft_list, df_wt_fft_list, v_inf_list):
    sns.lineplot(data=df_wt_fft, x='freq', y=0, label=f'v_inf = {v_inf} m/s', ax=ax)
#ax.set_title('FFT')
ax.grid(True)
ax.set_ylabel('Pressure [Pa]')
plt.xscale('log')
plt.xticks(x_ticks, x_ticks)
plt.tight_layout()
plt.savefig(f'saves/fft.png', dpi=300)
plt.show()


# ---------------------------------- PSD ----------------------------------

# df_bg_psd_db_list = []
# df_wt_psd_db_list = []
#
# for df_bg_fft, df_wt_fft, v_inf in zip(df_bg_fft_list, df_wt_fft_list, v_inf_list):
#     # Evaluate PSD in the frequency domain
#     print(f'[*] Calculating PSD for v_inf = {v_inf} m/s...')
#
#     df_bg_psd = df_bg_fft[0] ** 2
#     df_wt_psd = df_wt_fft[0] ** 2
#
#     # Double the amplitudes to account for one-sided spectrum, and normalize
#     df_bg_psd = df_bg_psd.apply(lambda x:  2 * x / (sample_rate * hann_scaling_factor))
#     df_wt_psd = df_wt_psd.apply(lambda x: 2 * x / (sample_rate * hann_scaling_factor))
#
#     # Convert to dB
#     df_bg_psd_db = df_bg_psd.apply(lambda x: 10 * np.log10(x / (p_ref ** 2)))
#     df_wt_psd_db = df_wt_psd.apply(lambda x: 10 * np.log10(x / (p_ref ** 2)))
#
#     # Assemble into dataframe
#     df_bg_psd_db = pd.DataFrame(df_bg_psd_db)
#     df_wt_psd_db = pd.DataFrame(df_wt_psd_db)
#
#     # Get frequency axis
#     df_bg_psd_db['freq'] = df_bg_fft['freq']
#     df_wt_psd_db['freq'] = df_wt_fft['freq']
#
#     # Append to lists
#     df_bg_psd_db_list.append(df_bg_psd_db)
#     df_wt_psd_db_list.append(df_wt_psd_db)
#
# # Plot results
# print('[*] Plotting results...')
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# fig.set_size_inches(10, 10)
# for df_bg_psd_db, df_wt_psd_db, v_inf in zip(df_bg_psd_db_list, df_wt_psd_db_list, v_inf_list):
#     sns.lineplot(data=df_bg_psd_db, x='freq', y=0, label=f'Background (v_inf = {v_inf} m/s)', ax=ax1)
#     sns.lineplot(data=df_wt_psd_db, x='freq', y=0, label=f'Wind turbine (v_inf = {v_inf} m/s)', ax=ax2)
# ax1.set_title('PSD')
# ax1.grid(True)
# ax1.set_ylabel('PSD [dB/Hz]')
# ax2.grid(True)
# ax2.set_ylabel('PSD [dB/Hz]')
# ax2.set_xlabel('Frequency [Hz]')
# plt.xscale('log')
# plt.show()

# --------------------------- Welch PSD -----------------------------------

df_bg_welch_psd_list = []
df_wt_welch_psd_list = []
df_wt_bg_welch_psd_list = []
df_bg_welch_psd_db_list = []
df_wt_welch_psd_db_list = []
df_wt_bg_welch_psd_db_list = []

for df_bg_no_hanning, df_wt_no_hanning, v_inf in zip(df_bg_no_hanning_list, df_wt_no_hanning_list, v_inf_list):
    # Evaluate PSD using Welch's method
    print(f'[*] Calculating Welch PSD for v_inf = {v_inf} m/s...')

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
print(f'[*] Frequency resolution: {freq_res_psd} Hz')

# Plot results
print('[*] Plotting results...')
fig, ax = plt.subplots()
fig.set_size_inches(size_x, size_y)
for df_bg_welch_psd_db, df_wt_welch_psd_db, v_inf in zip(df_bg_welch_psd_db_list, df_wt_welch_psd_db_list, v_inf_list):
    sns.lineplot(data=df_wt_welch_psd_db, x='freq', y='psd', label=f'v_inf = {v_inf} m/s', ax=ax)
#ax.set_title('PSD')
ax.grid(True)
ax.set_ylabel('PSD [dB/Hz]')
ax.set_xlabel('Frequency [Hz]')
plt.xscale('log')
plt.xticks(x_ticks, x_ticks)
plt.tight_layout()
plt.savefig('saves/welch_psd.png', dpi=300)
plt.show()

# Plot denoised results (dB)
fig, ax = plt.subplots()
fig.set_size_inches(size_x, size_y)
for df_wt_bg_welch_psd_db, v_inf in zip(df_wt_bg_welch_psd_db_list, v_inf_list):
    sns.lineplot(data=df_wt_bg_welch_psd_db, x='freq', y='psd', label=f'v_inf = {v_inf} m/s', ax=ax)
#ax.set_title('PSD (wt - bg)')
ax.grid(True)
ax.set_ylabel('PSD [dB/Hz]')
ax.set_xlabel('Frequency [Hz]')
plt.xscale('log')
plt.xticks(x_ticks, x_ticks)
plt.tight_layout()
plt.savefig('saves/welch_psd_denoised.png', dpi=300)
plt.show()

# ---------------------------------- SPL ----------------------------------

df_bg_spl_list = []
df_wt_spl_list = []
df_wt_bg_spl_list = []

freq_step = 10 # Hz
freq_bands = np.arange(0, max(df_bg_fft['freq']), freq_step)

for df_bg_welch_psd, df_wt_welch_psd, df_wt_bg_welch_psd, v_inf in zip(df_bg_welch_psd_list, df_wt_welch_psd_list, df_wt_bg_welch_psd_list, v_inf_list):

    # Evaluate SPL in the frequency domain over freq_step Hz bands
    print(f'[*] Calculating SPL for v_inf = {v_inf} m/s with freq_step = {freq_step} Hz...')

    df_bg_spl = pd.DataFrame(columns=['freq', 'spl'])
    df_wt_spl = pd.DataFrame(columns=['freq', 'spl'])
    df_wt_bg_spl = pd.DataFrame(columns=['freq', 'spl'])

    for l, c, u in zip(freq_bands[:-1], freq_bands[1:], freq_bands[2:]):
        # Sum PSD in band
        sum_bg = df_bg_welch_psd[(df_bg_welch_psd['freq'] >= l) & (df_bg_welch_psd['freq'] < u)].sum()
        sum_wt = df_wt_welch_psd[(df_wt_welch_psd['freq'] >= l) & (df_wt_welch_psd['freq'] < u)].sum()
        sum_wt_bg = df_wt_bg_welch_psd[(df_wt_bg_welch_psd['freq'] >= l) & (df_wt_bg_welch_psd['freq'] < u)].sum()

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

# Plot results
print('[*] Plotting results...')
fig, ax = plt.subplots()
fig.set_size_inches(size_x, size_y)
for df_bg_spl, df_wt_spl, v_inf in zip(df_bg_spl_list, df_wt_spl_list, v_inf_list):
    sns.lineplot(data=df_wt_spl, x='freq', y='spl', label=f'v_inf = {v_inf} m/s', ax=ax)
bpm_df = pd.DataFrame({'freq': bpm_f, 'spl': SPLTBL_tot})
bpm_df = bpm_df[(bpm_df['freq'] >= f_lower) & (bpm_df['freq'] <= f_upper)]
#sns.lineplot(data=bpm_df, x='freq', y='spl', label=f'(BPM) v_inf = x m/s', ax=ax)
#ax.set_title('SPL')
ax.grid(True)
ax.set_ylabel('L_p [dB]')
ax.set_xlabel('Frequency [Hz]')
plt.xscale('log')
plt.xticks(x_ticks, x_ticks)
plt.tight_layout()
plt.savefig('saves/spl.png', dpi=300)
plt.show()

# Plot df_wt_bg_spl
fig, ax = plt.subplots()
fig.set_size_inches(size_x, size_y)
for df_wt_bg_spl, v_inf in zip(df_wt_bg_spl_list, v_inf_list):
    sns.lineplot(data=df_wt_bg_spl, x='freq', y='spl', label=f'v_inf = {v_inf} m/s')
#sns.lineplot(data=bpm_df, x='freq', y='spl', label=f'(BPM) v_inf = {v_inf} m/s')
#ax.set_title('SPL (wt - bg)')
ax.grid(True)
ax.set_ylabel('L_p [dB]')
ax.set_xlabel('Frequency [Hz]')
plt.xscale('log')
plt.xticks(x_ticks, x_ticks)
plt.tight_layout()
plt.savefig('saves/spl_wt_bg.png', dpi=300)
plt.show()

# ---------------------------------- SPL_1/3 ----------------------------------

df_bg_spl_1_3_list = []
df_wt_spl_1_3_list = []

# Evaluate SPL in the frequency domain in 3rd octave bands
freq_centre = 10 ** (0.1 * np.arange(12, 43))
freq_d = 10 ** 0.05
f_upper_1_3 = freq_centre * freq_d
f_lower_1_3 = freq_centre / freq_d

for df_bg_welch_psd, df_wt_welch_psd, v_inf in zip(df_bg_welch_psd_list, df_wt_welch_psd_list, v_inf_list):

    print(f'[*] Calculating SPL_1/3 for v_inf = {v_inf} m/s...')

    df_bg_spl = pd.DataFrame(columns=['freq', 'spl'])
    df_wt_spl = pd.DataFrame(columns=['freq', 'spl'])

    # Clear values in dataframe
    df_bg_spl = df_bg_spl.iloc[0:0]
    df_wt_spl = df_wt_spl.iloc[0:0]

    for l, c, u in zip(f_lower_1_3, freq_centre, f_upper_1_3):
        # Sum PSD in band
        sum_bg = df_bg_welch_psd[(df_bg_welch_psd['freq'] >= l) & (df_bg_welch_psd['freq'] < u)].sum()
        sum_wt = df_wt_welch_psd[(df_wt_welch_psd['freq'] >= l) & (df_wt_welch_psd['freq'] < u)].sum()

        # Add row to dataframe
        df_bg_spl = df_bg_spl.append({'freq': c, 'spl': sum_bg[0] * (u - l)}, ignore_index=True)
        df_wt_spl = df_wt_spl.append({'freq': c, 'spl': sum_wt[0] * (u - l)}, ignore_index=True)

    # Convert spl column to dB
    df_bg_spl['spl'] = df_bg_spl['spl'].apply(lambda x: 10 * np.log10(x / (p_ref ** 2)))
    df_wt_spl['spl'] = df_wt_spl['spl'].apply(lambda x: 10 * np.log10(x / (p_ref ** 2)))

    # Remove frequencies outside of range
    df_bg_spl = df_bg_spl[(df_bg_spl['freq'] >= f_lower) & (df_bg_spl['freq'] <= f_upper)]
    df_wt_spl = df_wt_spl[(df_wt_spl['freq'] >= f_lower) & (df_wt_spl['freq'] <= f_upper)]

    # Append to lists
    df_bg_spl_1_3_list.append(df_bg_spl)
    df_wt_spl_1_3_list.append(df_wt_spl)

# Plot results
print('[*] Plotting results...')
fig, ax = plt.subplots()
fig.set_size_inches(size_x, size_y)
for df_bg_spl, df_wt_spl, v_inf in zip(df_bg_spl_1_3_list, df_wt_spl_1_3_list, v_inf_list):
    sns.lineplot(data=df_wt_spl, x='freq', y='spl', label=f'v_inf = {v_inf} m/s', ax=ax)
#ax.set_title('SPL 1/3')
ax.grid(True)
ax.set_ylabel('Pressure [dB]')
ax.set_xlabel('Frequency [Hz]')
plt.xscale('log')
plt.xticks(x_ticks, x_ticks)
plt.tight_layout()
plt.savefig('saves/spl_1_3.png', dpi=300)
plt.show()


# ---------------------------------- OSPL ----------------------------------

OSPL_bg_list = []
OSPL_wt_list = []

for df_bg_welch_psd, df_wt_welch_psd, v_inf in zip(df_bg_welch_psd_list, df_wt_welch_psd_list, v_inf_list):
    # Sum PSD in band f_lower to f_upper
    sum_bg = df_bg_welch_psd[(df_bg_welch_psd['freq'] >= 800) & (df_bg_welch_psd['freq'] <= 3000)].sum()
    sum_wt = df_wt_welch_psd[(df_wt_welch_psd['freq'] >= 800) & (df_wt_welch_psd['freq'] <= 3000)].sum()

    # Calculate OSPL
    ospl_bg = 10 * np.log10((sum_bg[0] * freq_res_psd) / (p_ref ** 2))
    ospl_wt = 10 * np.log10((sum_wt[0] * freq_res_psd) / (p_ref ** 2))

    # Print results
    print(f'[*] v_inf: {v_inf} m/s')
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
print('[*] Plotting results...')
fig, ax = plt.subplots()
fig.set_size_inches(size_x, size_y)
sns.scatterplot(x=v_inf_list, y=OSPL_wt_list, ax=ax, color='blue', label='Wind turbine OSPL')

# Plot trend line (dB scale)
x = np.linspace(min(v_inf_list), max(v_inf_list), 100)
y = trend_ideal_fn[0] * np.log10(x) + trend_ideal_fn[1]
sns.lineplot(x=x, y=y, ax=ax, color='blue', label=f'{trend_ideal_fn[0]:.2f}*log(x) + {trend_ideal_fn[1]:.2f}')

#ax.set_title(f'OSPL ({800} - {3000} Hz)')
ax.grid(True)
ax.set_ylabel('OSPL [dB]')
ax.set_xlabel('v_inf [m/s]')
plt.savefig(f'saves/OSPL_800_3000.png', dpi=300)
plt.show()

playsound('./misc/bell.wav')
print('[*] Done!')