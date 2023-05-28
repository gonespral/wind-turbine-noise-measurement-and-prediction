import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl

f_lower = 1  # Hz
f_upper = 24000 # Hz
scaling_factor = 1
size_x = 6.5 * scaling_factor
size_y = 5 * scaling_factor
x_ticks = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
color_scheme = 'viridis'

with open('saves/processed_data.pkl', 'rb') as f:
    data = pkl.load(f)
    bg_psd_list = data['bg_psd_list']
    wt_psd_list = data['wt_psd_list']
    v_inf_list = data['v_inf_list']

# Remove frequencies outside of range
for bg_psd, wt_psd in zip(bg_psd_list, wt_psd_list):
    bg_psd = bg_psd.loc[(bg_psd.index >= f_lower) & (bg_psd.index <= f_upper)]
    wt_psd = wt_psd.loc[(wt_psd.index >= f_lower) & (wt_psd.index <= f_upper)]

# Prepare colors for plots
colors = []
for i in range(len(v_inf_list) + 10):
    colors.append(plt.cm.get_cmap(color_scheme, len(v_inf_list))(i))

fig, ax = plt.subplots(figsize=(size_x, size_y))
for bg_psd, wt_psd, v_inf, color in zip(bg_psd_list, wt_psd_list, v_inf_list, colors):
    print(f'Plotting v_inf = {v_inf} m/s')
    ax.plot(bg_psd.index, bg_psd, color=color, label=f'v_inf = {v_inf} m/s')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB)')
    ax.legend()
    ax.grid(True, which='both')
    ax.set_xlim(f_lower, f_upper)
    plt.show()

