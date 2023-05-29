import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl

f_lower = 300  # Hz
f_upper = 5000 # Hz
scaling_factor = 1
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
    #bg_spl_1_3_list = data['bg_spl_1_3_list'] # List of dataframes => Index: freq, Value: SPL
    #wt_spl_1_3_list = data['wt_spl_1_3_list'] # List of dataframes => Index: freq, Value: SPL
    v_inf_list = data['v_inf_list'] # List of velocities

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
plt.show()

# Plot PSD for bg
fig, ax = plt.subplots(figsize=(size_x, size_y))
for bg, v_inf, color in zip(bg_psd_list, v_inf_list, colors):
    bg = bg.rolling(n_smooth, center=True).mean()
    bg.plot(ax=ax, color=color, label=f'{v_inf} m/s')
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (dB)')
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(f_lower, f_upper)
plt.title('Background')
plt.xticks(x_ticks, x_ticks)
plt.show()
