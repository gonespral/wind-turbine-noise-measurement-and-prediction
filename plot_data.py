import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl

f_lower = 300  # Hz
f_upper = 5000 # Hz
scaling_factor = 1.2
size_x = 6.5 * scaling_factor
size_y = 5 * scaling_factor
x_ticks = [500, 1000, 1500, 2000, 3000, 4000, 5000]
color_scheme = 'viridis'
n_smooth = 10

# Load data from pickle file
with open('saves/processed_data.pkl', 'rb') as f:
    data = pkl.load(f)
    bg_psd_list = data['bg_psd_list'] # List of dataframes => Index: freq, Value: PSD
    wt_psd_list = data['wt_psd_list'] # List of dataframes => Index: freq, Value: PSD
    denoised_psd_list = data['denoised_psd_list'] # List of dataframes => Index: freq, Value: PSD
    bg_spl_1_3_list = data['bg_spl_1_3_list'] # List of dataframes => Index: freq, Value: SPL
    wt_spl_1_3_list = data['wt_spl_1_3_list'] # List of dataframes => Index: freq, Value: SPL
    denoised_spl_1_3_list = data['denoised_spl_1_3_list'] # List of dataframes => Index: freq, Value: SPL
    v_inf_list = data['v_inf_list'] # List of velocities

# Convert spl dataframes to series
bg_spl_1_3_list = [bg_spl_1_3_list[i][0] for i in range(len(bg_spl_1_3_list))]
wt_spl_1_3_list = [wt_spl_1_3_list[i][0] for i in range(len(wt_spl_1_3_list))]

# Import data from BPM model (data/BPM/data.csv)
bpm_list = []
for v_inf in v_inf_list:
    print(f'[*] Importing BPM data from BPM_BYU/data/data{int(v_inf)}.csv...')
    bpm = pd.read_csv(f'BPM_BYU/data/data{int(v_inf)}.csv', sep=',')
    # Move column freq to index
    bpm = bpm.set_index('freq')
    bpm.index = bpm.index.astype(int)
    bpm = bpm['spl']
    bpm_list.append(bpm)

# Prepare colors for plots
colors = []
for i in range(len(v_inf_list) + 10):
    colors.append(plt.cm.get_cmap(color_scheme, len(v_inf_list))(i))

# Plot PSD for wt
fig, ax = plt.subplots(figsize=(size_x, size_y))
for wt, v_inf, color in zip(wt_psd_list, v_inf_list, colors):
    wt = wt.rolling(n_smooth, center=True).mean()
    wt.plot(ax=ax, color=color, label=f'{v_inf} m/s')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
plt.title('Wind Turbine')
plt.xticks(x_ticks, x_ticks)
plt.savefig('saves/wt_psd.png', dpi=300)
plt.show()

# Plot SPL for wt
fig, ax = plt.subplots(figsize=(size_x, size_y))
for wt, bpm, v_inf, color in zip(wt_spl_1_3_list, bpm_list, v_inf_list, colors):
    wt.plot(ax=ax, color=color, label=f'{v_inf} m/s')
    ax.plot(bpm, color=color, linestyle='--')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
plt.title('Wind Turbine')
plt.xticks(x_ticks, x_ticks)
plt.savefig('saves/wt_spl.png', dpi=300)
plt.show()

# Plot SPL for denoised with BPM
fig, ax = plt.subplots(figsize=(size_x, size_y))
for denoised, bpm, v_inf, color in zip(denoised_spl_1_3_list, bpm_list, v_inf_list, colors):
    denoised.plot(ax=ax, color=color, label=f'{v_inf} m/s')
    ax.plot(bpm, color=color, linestyle='--')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
plt.title('Denoised')
plt.xticks(x_ticks, x_ticks)
plt.savefig('saves/denoised_spl.png', dpi=300)
plt.show()

# Plot abs error between SPL for wt and BPM
fig, ax = plt.subplots(figsize=(size_x, size_y))
for wt, bpm, v_inf, color in zip(wt_spl_1_3_list, bpm_list, v_inf_list, colors):
    error = np.abs(wt - bpm)
    ax.plot(error, color=color, label=f'{v_inf} m/s')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
plt.title('Error wt - BPM')
plt.xticks(x_ticks, x_ticks)
plt.savefig('saves/wt_error.png', dpi=300)
plt.show()

# Plot abs error between SPL for denoised and BPM
fig, ax = plt.subplots(figsize=(size_x, size_y))
for denoised, bpm, v_inf, color in zip(denoised_spl_1_3_list, bpm_list, v_inf_list, colors):
    error = np.abs(denoised[0] - bpm)
    ax.plot(error, color=color, label=f'{v_inf} m/s')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
plt.title('Error denoised - BPM')
plt.xticks(x_ticks, x_ticks)
plt.savefig('saves/denoised_error.png', dpi=300)
plt.show()




