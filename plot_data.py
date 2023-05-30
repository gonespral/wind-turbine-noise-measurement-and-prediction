import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl

f_lower = 300  # Hz
f_upper = 5000 # Hz
scaling_factor = 1.4
size_x = 6.5 * scaling_factor
size_y = 5 * scaling_factor
x_ticks = [500, 1000, 1500, 2000, 3000, 4000, 5000]
color_scheme = 'rocket'
n_smooth = 10

# Load data from pickle file
with open('saves/processed_data.pkl', 'rb') as f:
    data = pkl.load(f)
    bg_psd_list = data['bg_psd_list'] # List of dataframes => Index: freq, Value: PSD
    wt_psd_list = data['wt_psd_list'] # List of dataframes => Index: freq, Value: PSD
    bg_spl_list = data['bg_spl_list'] # List of dataframes => Index: freq, Value: SPL
    wt_spl_list = data['wt_spl_list'] # List of dataframes => Index: freq, Value: SPL
    bg_spl_1_3_list = data['bg_spl_1_3_list'] # List of dataframes => Index: freq, Value: SPL
    wt_spl_1_3_list = data['wt_spl_1_3_list'] # List of dataframes => Index: freq, Value: SPL
    bg_ospl_list = data['bg_ospl_list'] # List of SPL values
    wt_ospl_list = data['wt_ospl_list'] # List of SPL values
    v_inf_list = data['v_inf_list'] # List of velocities

# Convert spl dataframes to spl series
bg_spl_list = [bg_spl.iloc[:,0] for bg_spl in bg_spl_list]
wt_spl_list = [wt_spl.iloc[:,0] for wt_spl in wt_spl_list]

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
for i in range(len(v_inf_list) + 1):
    colors.append(plt.cm.get_cmap(color_scheme, len(v_inf_list) + 1)(i))

# Plot PSD for wt
fig, ax = plt.subplots(figsize=(size_x, size_y))
for wt, v_inf, color in zip(wt_psd_list, v_inf_list, colors):
    wt = wt.rolling(n_smooth, center=True).mean()
    wt.plot(ax=ax, color=color, label=f'{v_inf} m/s')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (dB/Hz)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
ax.set_ylim(20, 50)
plt.xticks(x_ticks, x_ticks)
plt.savefig('saves/wt_psd.png', dpi=300)
plt.show()

# Plot SNR for wt/bg
SNR_list = []
for wt, bg, v_inf in zip(wt_psd_list, bg_psd_list, v_inf_list):
    # Convert from db to linear
    wt_ = 10 ** (wt / 10)
    bg_ = 10 ** (bg / 10)
    SNR_list.append(10 * np.log10(wt_ / bg_))
fig, ax = plt.subplots(figsize=(size_x, size_y))
for SNR, v_inf, color in zip(SNR_list, v_inf_list, colors):
    SNR = SNR.rolling(n_smooth, center=True).mean()
    SNR.plot(ax=ax, color=color, label=f'{v_inf} m/s')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SNR (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
plt.xticks(x_ticks, x_ticks)
plt.savefig('saves/wt_bg_snr.png', dpi=300)
plt.show()

# Plot SPL for wt
fig, ax = plt.subplots(figsize=(size_x, size_y))
for wt, v_inf, color in zip(wt_spl_list, v_inf_list, colors):
    wt.plot(ax=ax, color=color, label=f'{v_inf} m/s')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
ax.set_ylim(30,67)
plt.xticks(x_ticks, x_ticks)
plt.savefig('saves/wt_spl.png', dpi=300)
plt.show()

# Plot wide spectrum SPL for wt
fig, ax = plt.subplots(figsize=(size_x, size_y))
for wt, v_inf, color in zip(wt_spl_list, v_inf_list, colors):
    wt.plot(ax=ax, color=color, label=f'{v_inf} m/s')
ax.set_xscale('linear')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(1, 24500)
ax.set_ylim(25, 80)
plt.savefig('saves/wt_spl=wideband.png', dpi=300)
plt.show()

# Plot SPL 1/3 for wt
fig, ax = plt.subplots(figsize=(size_x, size_y))
for wt, bpm, v_inf, color in zip(wt_spl_1_3_list, bpm_list, v_inf_list, colors):
    wt.plot(ax=ax, color=color, label=f'wt: {v_inf} m/s')
    ax.plot(bpm, color=color, linestyle='--')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
plt.xticks(x_ticks, x_ticks)
ax.set_ylim(30, 80)
plt.savefig('saves/wt_spl.png', dpi=300)
plt.show()

# Plot SPL 1/3 for bg
fig, ax = plt.subplots(figsize=(size_x, size_y))
for bg, bpm, v_inf, color in zip(bg_spl_1_3_list, bpm_list, v_inf_list, colors):
    bg.plot(ax=ax, color=color, label=f'bg: {v_inf} m/s')
    ax.plot(bpm, color=color, linestyle='--')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
plt.xticks(x_ticks, x_ticks)
ax.set_ylim(30, 80)
plt.savefig('saves/wt_spl.png', dpi=300)
plt.show()

# Plot abs error between SPL for wt and BPM
fig, ax = plt.subplots(figsize=(size_x, size_y))
for wt, bpm, v_inf, color in zip(wt_spl_1_3_list, bpm_list, v_inf_list, colors):
    error = np.abs(wt - bpm)
    ax.plot(error, color=color, label=f'e: {v_inf} m/s')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
plt.xticks(x_ticks, x_ticks)
ax.set_ylim(-2.5, 20)
plt.savefig('saves/wt_error.png', dpi=300)
plt.show()

# Get trendline A * log10(x) + B
trend_ideal_fn = np.polyfit(np.log10(v_inf_list), wt_ospl_list, 1)
print(f'    Trendline: {trend_ideal_fn[0]} * log10(v_inf) + {trend_ideal_fn[1]}')
print(f'    R^2: {np.corrcoef(np.log10(v_inf_list), wt_ospl_list)[0, 1] ** 2}')

# Plot OSPL for wt and trendline
fig, ax = plt.subplots(figsize=(size_x, size_y))
# Plot points
for wt_ospl, v_inf in zip(wt_ospl_list, v_inf_list):
    # Get color from cmap but fixed
    color = plt.cm.get_cmap('viridis')(50)
    ax.scatter(v_inf, wt_ospl, color=color)
ax.plot(v_inf_list, trend_ideal_fn[0] * np.log10(v_inf_list) + trend_ideal_fn[1], color=color, label=f'{round(trend_ideal_fn[0], 2)} * log10(v_inf) + {round(trend_ideal_fn[1], 2)}')
ax.set_xscale('log')
ax.set_xlabel('Velocity (m/s)')
ax.set_ylabel('OSPL (dB)')
ax.legend()
ax.grid(True, which='both')
plt.xticks(v_inf_list, v_inf_list)
plt.savefig('saves/wt_ospl.png', dpi=300)
plt.show()








