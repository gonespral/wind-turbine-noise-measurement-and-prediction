import mat73
import pandas as pd
import numpy as np
import scipy.signal as signal
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle as pkl

with open('misc/windmill.txt', 'r') as f:
    print(f.read())

# ----------------------------- Config ----------------------------------
p_ref = 2E-5  # Reference pressure (Pa)
sample_rate = 48128  # Hz
num_chunks = 10

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
bg_paths = ['data/downloads/U12_Background.mat']
wt_paths = ['data/downloads/U12_Wind%20turbine.mat']

# ------------------------- Prepare data --------------------------------

df_bg_list = []
df_wt_list = []
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

    # Append to lists
    df_bg_list.append(df_bg)
    df_wt_list.append(df_wt)
    v_inf_list.append(float(v_inf))

# Import data from BPM model (data/BPM/data.csv)
df_bpm_list = []
for v_inf in v_inf_list:
    print(f'[*] Importing BPM data from BPM_BYU/data/data{int(v_inf)}.csv...')
    df_bpm = pd.read_csv(f'BPM_BYU/data/data{int(v_inf)}.csv', sep=',')
    # Remove data outside of range
    df_bpm_list.append(df_bpm)

# ------------------------- Prepare chunks ------------------------------

bg_list = [] # [v_inf][chunk] = [df]
wt_list = []

print('[*] Preparing chunks...')

chunk_size = int(len(df_bg_list[0].columns) / num_chunks)
chunk_size_t = chunk_size / sample_rate
chunk_t = []
for i in range(num_chunks):
    chunk_t.append(chunk_size_t * i)
for df_bg, df_wt in zip(df_bg_list, df_wt_list):
    bg_list.append([])
    wt_list.append([])
    for t in chunk_t:
        bg_list[-1].append(df_bg.loc[:, t:t + chunk_size_t])
        wt_list[-1].append(df_wt.loc[:, t:t + chunk_size_t])

# ------------------------- PSD (Welch) --------------------------------

bg_psd_list = []
wt_psd_list = []

print('[*] Calculating PSD (Welch)...')
for bg, wt, v_inf in zip(bg_list, wt_list, v_inf_list):
    print(f'    v_inf = {v_inf} m/s')
    bg_psd_list.append([])
    wt_psd_list.append([])
    i = 1
    for bg_chunk, wt_chunk in zip(bg, wt):
        print(f'            chunk {i}')
        i += 1
        bg_psd_list[-1].append(pd.DataFrame())
        wt_psd_list[-1].append(pd.DataFrame())
        for j, chunk in zip([0,1],[bg_chunk, wt_chunk]):
            for index, row in chunk.iterrows():
                f, psd = signal.welch(row,
                                      fs=sample_rate,
                                      nperseg=chunk_size,
                                      return_onesided=True,
                                      scaling='density',
                                      noverlap=0)

                # Create columns for each freq if they don't exist
                if len(bg_psd_list[-1][-1].columns) == 0:
                    bg_psd_list[-1][-1] = pd.DataFrame(columns=f)
                    wt_psd_list[-1][-1] = pd.DataFrame(columns=f)

                # Append row to dataframe
                if j == 0:
                    bg_psd_list[-1][-1] = bg_psd_list[-1][-1].append(pd.Series(psd, index=f), ignore_index=True)
                elif j == 1:
                    wt_psd_list[-1][-1] = wt_psd_list[-1][-1].append(pd.Series(psd, index=f), ignore_index=True)

freq_res = f[1] - f[0]
print(f'    freq_res = {freq_res} Hz')

bg_psd_avg_list = []
wt_psd_avg_list = []

print('[*] Averaging PSD...')
for bg, wt, v_inf in zip(bg_psd_list, wt_psd_list, v_inf_list):
    print(f'    v_inf = {v_inf} m/s')
    bg_values = []
    wt_values = []
    for bg_chunk, wt_chunk in zip(bg, wt):
        bg_values.append(bg_chunk.values)
        wt_values.append(wt_chunk.values)

    bg_avg = pd.DataFrame(np.mean(bg_values, axis=0), columns=bg[0].columns)
    wt_avg = pd.DataFrame(np.mean(wt_values, axis=0), columns=wt[0].columns)

    bg_psd_avg_list.append(bg_avg)
    wt_psd_avg_list.append(wt_avg)

bg_psd_avg_db_list = []
wt_psd_avg_db_list = []

for bg, wt in zip(bg_psd_avg_list, wt_psd_avg_list):
    # Create row for average
    bg.loc['avg'] = bg.mean()
    wt.loc['avg'] = wt.mean()

    # Remove all other rows
    bg = bg.loc['avg']
    wt = wt.loc['avg']

    # Convert all values to dB (relative to p_ref)
    bg = 10 * np.log10(bg / p_ref)
    wt = 10 * np.log10(wt / p_ref)

    bg_psd_avg_db_list.append(bg)
    wt_psd_avg_db_list.append(wt)

# ------------------------- SPL (1/3) ----------------------------------

df_bg_spl_1_3_list = []
df_wt_spl_1_3_list = []

# Evaluate SPL in the frequency domain in 3rd octave bands
# freq_centre = 10 ** (0.1 * np.arange(12, 43))
freq_centre = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
               8000, 10000, 12500, 16000, 20000, 25000, 31500, 40000])
freq_d = 10 ** 0.05
f_upper_1_3 = freq_centre * freq_d
f_lower_1_3 = freq_centre / freq_d

print('[*] Calculating SPL (1/3)...')
for bg, wt, v_inf in zip(bg_psd_avg_db_list, wt_psd_avg_db_list, v_inf_list):
    print(f'    v_inf = {v_inf} m/s')
    df_bg_spl_1_3_list.append(pd.DataFrame())
    df_wt_spl_1_3_list.append(pd.DataFrame())
    for index, row in bg.iterrows():
        spl_bg = []
        spl_wt = []
        for f_lower, f_upper in zip(f_lower_1_3, f_upper_1_3):
            spl_bg.append(10 * np.log10(np.sum(10 ** (row.loc[f_lower:f_upper] / 10))))
            spl_wt.append(10 * np.log10(np.sum(10 ** (wt.loc[f_lower:f_upper] / 10))))
        df_bg_spl_1_3_list[-1] = df_bg_spl_1_3_list[-1].append(pd.Series(spl_bg, index=freq_centre), ignore_index=True)
        df_wt_spl_1_3_list[-1] = df_wt_spl_1_3_list[-1].append(pd.Series(spl_wt, index=freq_centre), ignore_index=True)


# ------------------------- Save data ---------------------------------
print('[*] Saving data...')
with open('saves/processed_data.pkl', 'wb') as f:
    pkl.dump({'bg_psd_list': bg_psd_avg_db_list,
              'wt_psd_list': wt_psd_avg_db_list,
              'bg_spl_1_3_list': df_bg_spl_1_3_list,
              'wt_spl_1_3_list': df_wt_spl_1_3_list,
              'v_inf_list': v_inf_list}, f)


