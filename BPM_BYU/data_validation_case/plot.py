import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

f_lower = 95  # Hz
f_upper = 30001 # Hz
scaling_factor = 1
size_x = 6.5 * scaling_factor
size_y = 5 * scaling_factor
x_ticks = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]

# Load BPM data
df_bpm = pd.read_csv('data71.csv', sep=',')
df_bpm = df_bpm[(df_bpm['freq'] >= f_lower) & (df_bpm['freq'] <= f_upper)]
df_val = pd.read_csv('Default Dataset.csv', sep=',')
df_val = df_val[(df_val['freq'] >= f_lower) & (df_val['freq'] <= f_upper)]

# Plot comparison on same axes
fig, ax = plt.subplots(figsize=(size_x, size_y))
sns.lineplot(x='freq', y='spl', data=df_bpm, ax=ax, label='BPM')
sns.scatterplot(x='freq', y='spl', data=df_val, ax=ax, label='Val')
ax.set_xscale('log')
ax.set_yscale('linear')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.set_xlim([f_lower, f_upper])
plt.grid(True)
plt.show()

