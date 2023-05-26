import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl

f_lower = 300  # Hz
f_upper = 5000 # Hz
scaling_factor = 1
size_x = 6.5 * scaling_factor
size_y = 5 * scaling_factor
x_ticks = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]

# Load BPM data
with open('BPM/saves/BPM_spl.pkl', 'rb') as f:
    bpm_model = pkl.load(f)
    bpm_f = bpm_model[0]
    SPLTBL_s = bpm_model[1]
    SPLTBL_p = bpm_model[2]
    SPLTBL_alpha = bpm_model[3]
    SPLTBL_tot = bpm_model[4]

df_bpm = pd.DataFrame({'freq': bpm_f, 'SPLTBL_s': SPLTBL_s, 'SPLTBL_p': SPLTBL_p, 'SPLTBL_alpha': SPLTBL_alpha, 'SPLTBL_tot': SPLTBL_tot})
df_bpm = df_bpm[(df_bpm['freq'] >= f_lower) & (df_bpm['freq'] <= f_upper)]

# Load BPM_BYU data
df_bpm_byu = pd.read_csv('BPM_BYU/data/data12.csv', sep=',')
df_bpm_byu = df_bpm_byu[(df_bpm_byu['freq'] >= f_lower) & (df_bpm_byu['freq'] <= f_upper)]

# Plot comparison on same axes
fig, ax = plt.subplots(figsize=(size_x, size_y))
sns.lineplot(x='freq', y='SPLTBL_tot', data=df_bpm, ax=ax, label='BPM')
sns.lineplot(x='freq', y='spl', data=df_bpm_byu, ax=ax, label='BPM_BYU')
ax.set_xscale('log')
ax.set_yscale('linear')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.set_xlim([f_lower, f_upper])
plt.grid(True)
plt.show()

