import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl

f_lower = 300  # Hz
f_upper = 5000 # Hz
scaling_factor = 0.95
size_x = 6.5 * scaling_factor
size_y = 5 * scaling_factor
x_ticks = [500, 1000, 1500, 2000, 3000, 4000, 5000]
color_scheme = 'rocket'
n_smooth = 10
v_inf_symb = '$V_{\\infty}$'

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

print(f'Frequency resolution: {bg_psd_list[0].index[1] - bg_psd_list[0].index[0]} Hz')

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

#Import data for validation graph
val_list = []
val = pd.read_csv(f'BPM_BYU/data_validation_case/data71.csv', sep=',')
# Move column freq to index
val = val.set_index('freq')
val.index = val.index.astype(int)
val = val['spl']
val_list.append(val)
print(val_list)

valf_list = []
valf = pd.read_csv(f'BPM_BYU/data_validation_case/data71.csv', sep=',')
# Move column freq to index
valf = valf.set_index('spl')
valf.index = valf.index.astype(int)
valf = valf['freq']
valf_list.append(valf)
print(valf_list)

#import data for default dataset
default_list = []
default = pd.read_csv(f'BPM_BYU/data_validation_case/Default Dataset.csv', sep=',')
# Move column freq to index
default = default.set_index('freq')
default.index = default.index.astype(int)
default = default['spl']
default_list.append(default)
print(default_list)
defaultf_list = []
defaultf = pd.read_csv(f'BPM_BYU/data_validation_case/Default Dataset.csv', sep=',')
# Move column freq to index
defaultf = defaultf.set_index('spl')
defaultf.index = defaultf.index.astype(int)
defaultf = defaultf['freq']
defaultf_list.append(defaultf)
print(default_list)
# Prepare colors for plots
# Prepare colors for plots
colors = []
for i in range(len(v_inf_list) + 1):
    colors.append(plt.cm.get_cmap(color_scheme, len(v_inf_list) + 1)(i))

# Plot PSD for wt
fig, ax = plt.subplots(figsize=(size_x, size_y))
for wt, v_inf, color in zip(wt_psd_list, v_inf_list, colors):
    wt = wt.rolling(n_smooth, center=True).mean()
    wt.plot(ax=ax, color=color, label=f'{v_inf_symb} = {v_inf} m/s')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (dB/Hz)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
ax.set_ylim(20, 50)
plt.xticks(x_ticks, x_ticks)
plt.savefig('saves/wt_psd.png', dpi=300)
plt.tight_layout()
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
    SNR.plot(ax=ax, color=color, label=f'{v_inf_symb} = {v_inf} m/s')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SNR (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
plt.xticks(x_ticks, x_ticks)
plt.savefig('saves/wt_bg_snr.png', dpi=300)
plt.tight_layout()
plt.show()

# Plot SPL for wt
fig, ax = plt.subplots(figsize=(size_x, size_y))
for wt, v_inf, color in zip(wt_spl_list, v_inf_list, colors):
    wt.plot(ax=ax, color=color, label=f'{v_inf_symb} = {v_inf} m/s')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
ax.set_ylim(35,65)
plt.xticks(x_ticks, x_ticks)
plt.savefig('saves/wt_spl.png', dpi=300)
plt.tight_layout()
plt.show()

# Plot wide spectrum SPL for wt
fig, ax = plt.subplots(figsize=(size_x, size_y))
for wt, v_inf, color in zip(wt_spl_list, v_inf_list, colors):
    wt.plot(ax=ax, color=color, label=f'{v_inf_symb} = {v_inf} m/s')
ax.set_xscale('linear')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(100, 23500)
ax.set_ylim(1, 80)
plt.savefig('saves/wt_spl_wideband.png', dpi=300)
plt.tight_layout()
plt.show()

# Plot wide spectrum SPL for bg
fig, ax = plt.subplots(figsize=(size_x, size_y))
for bg, v_inf, color in zip(bg_spl_list, v_inf_list, colors):
    bg.plot(ax=ax, color=color, label=f'{v_inf_symb} = {v_inf} m/s')
ax.set_xscale('linear')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(100, 23500)
ax.set_ylim(1, 80)
plt.savefig('saves/bg_spl_wideband.png', dpi=300)
plt.tight_layout()
plt.show()

# Plot SPL 1/3 for wt
fig, ax = plt.subplots(figsize=(size_x, size_y))
for wt, bpm, v_inf, color in zip(wt_spl_1_3_list, bpm_list, v_inf_list, colors):
    wt.plot(ax=ax, color=color, label=f'{v_inf_symb} = {v_inf} m/s')
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
plt.tight_layout()
plt.show()

# Plot SPL 1/3 for bg
fig, ax = plt.subplots(figsize=(size_x, size_y))
for bg, bpm, v_inf, color in zip(bg_spl_1_3_list, bpm_list, v_inf_list, colors):
    bg.plot(ax=ax, color=color, label=f'{v_inf_symb} = {v_inf} m/s')
    ax.plot(bpm, color=color, linestyle='--')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
plt.xticks(x_ticks, x_ticks)
ax.set_ylim(30, 80)
plt.savefig('saves/bg_spl.png', dpi=300)
plt.tight_layout()
plt.show()

denoised_spl_1_3_list = []
for wt, bg in zip(wt_spl_1_3_list, bg_spl_1_3_list):
    # Remove from dB and subtract
    wt = 10**(wt/10)
    bg = 10**(bg/10)
    denoised = 10*np.log10(wt - bg)
    denoised_spl_1_3_list.append(denoised)

# Plot denoised SPL 1/3
fig, ax = plt.subplots(figsize=(size_x, size_y))
for denoised, bpm, v_inf, color in zip(denoised_spl_1_3_list, bpm_list, v_inf_list, colors):
    ax.plot(denoised, color=color, label=f'{v_inf_symb} = {v_inf} m/s')
    ax.plot(bpm, color=color, linestyle='--')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
plt.xticks(x_ticks, x_ticks)
ax.set_ylim(30, 80)
plt.savefig('saves/denoised_spl.png', dpi=300)
plt.tight_layout()
plt.show()

# Plot abs error between SPL for wt and BPM
fig, ax = plt.subplots(figsize=(size_x, size_y))
for wt, bpm, v_inf, color in zip(wt_spl_1_3_list, bpm_list, v_inf_list, colors):
    error = np.abs(wt - bpm)
    ax.plot(error, color=color, label=f'{v_inf_symb} = {v_inf} m/s')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
plt.xticks(x_ticks, x_ticks)
ax.set_ylim(-2.5, 20)
plt.savefig('saves/wt_error.png', dpi=300)
plt.tight_layout()
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
    color = plt.cm.get_cmap('viridis')(20)
    ax.scatter(v_inf, wt_ospl, color=color)
ax.plot(v_inf_list, trend_ideal_fn[0] * np.log10(v_inf_list) + trend_ideal_fn[1], color=color, label=f'{round(trend_ideal_fn[0], 2)/10} * 10 * log10({v_inf_symb}) + {round(trend_ideal_fn[1], 2)}')
ax.set_xscale('log')
ax.set_xlabel(f'{v_inf_symb} = {v_inf} m/s')
ax.set_ylabel('OSPL (dB)')
ax.legend()
ax.grid(True, which='both')
plt.xticks(v_inf_list, v_inf_list)
plt.savefig('saves/wt_ospl.png', dpi=300)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(size_x, size_y))
# Plot points
for wt_ospl, v_inf in zip(val_list, valf_list):
    # Get color from cmap but fixed
    color = plt.cm.get_cmap('viridis')(20)
    ax.scatter(v_inf, wt_ospl, color=color, label = "validation")
for wt_ospl, v_inf in zip(default_list, defaultf_list):
    color = plt.cm.get_cmap('viridis')(20)
    ax.plot(v_inf, wt_ospl, color=color, label = "BPM")
ax.set_xscale('log')
ax.set_xlim(f_lower, 20000)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel('OSPL (dB)')
ax.legend()
ax.grid(True, which='both')
plt.savefig('saves/wt_ospl.png', dpi=300)
plt.tight_layout()
plt.show()




