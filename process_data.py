import mat73
import pandas as pd
import numpy as np
import scipy.signal as signal
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
#bg_paths = ['data/downloads/U12_Background.mat']
#wt_paths = ['data/downloads/U12_Wind%20turbine.mat']

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

    # Create row for average
    bg_avg.loc['avg'] = bg_avg.mean()
    wt_avg.loc['avg'] = wt_avg.mean()

    # Remove all other rows
    bg_avg = bg_avg.loc['avg']
    wt_avg = wt_avg.loc['avg']

    bg_psd_avg_list.append(bg_avg)
    wt_psd_avg_list.append(wt_avg)

bg_psd_avg_db_list = []
wt_psd_avg_db_list = []

for bg_avg, wt_avg in zip(bg_psd_avg_list, wt_psd_avg_list):
    # Convert all values to dB (relative to p_ref)
    bg_avg = 10 * np.log10(bg_avg / (p_ref ** 2))
    wt_avg = 10 * np.log10(wt_avg / (p_ref ** 2))

    bg_psd_avg_db_list.append(bg_avg)
    wt_psd_avg_db_list.append(wt_avg)

# ------------------------- SPL -----------------------------------------

bg_spl_list = []
wt_spl_list = []

freq_step = 10 # Hz
freq_bands = np.arange(0, max(f), freq_step)

print('[*] Calculating SPL...')

for bg_psd, wt_psd, v_inf in zip(bg_psd_avg_list, wt_psd_avg_list, v_inf_list):
    # Calculate SPL
    print(f'    v_inf = {v_inf} m/s')

    bg_spl = pd.DataFrame()
    wt_spl = pd.DataFrame()

    for l, c, u in zip(freq_bands[:-1], freq_bands[1:], freq_bands[2:]):
        # Sum PSD in band
        bg_sum = bg_psd.loc[(bg_psd.index >= l) & (bg_psd.index < u)].sum() * freq_res
        wt_sum = wt_psd.loc[(wt_psd.index >= l) & (wt_psd.index < u)].sum() * freq_res

        # Convert to SPL
        bg_spl = bg_spl.append(pd.Series(10 * np.log10(bg_sum / (p_ref ** 2)), name=c))
        wt_spl = wt_spl.append(pd.Series(10 * np.log10(wt_sum / (p_ref ** 2)), name=c))

    bg_spl_list.append(bg_spl)
    wt_spl_list.append(wt_spl)

# ------------------------- SPL (1/3) ----------------------------------

bg_spl_1_3_list = []
wt_spl_1_3_list = []

# Evaluate SPL in the frequency domain in 3rd octave bands
freq_centre = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
               8000, 10000, 12500, 16000, 20000, 25000, 31500, 40000])
freq_d = 10 ** 0.05
f_upper_1_3 = freq_centre * freq_d
f_lower_1_3 = freq_centre / freq_d

print('[*] Calculating SPL (1/3)...')
for bg_psd, wt_psd, v_inf in zip(bg_psd_avg_list, wt_psd_avg_list, v_inf_list):
    print(f'    v_inf = {v_inf} m/s')

    bg_spl = pd.DataFrame()
    wt_spl = pd.DataFrame()
    denoised_spl = pd.DataFrame()

    for l, c, u in zip(f_lower_1_3, freq_centre, f_upper_1_3):
        # Integrate PSD in band
        bg_sum = bg_psd.loc[(bg_psd.index >= l) & (bg_psd.index <= u)].sum() * freq_res
        wt_sum = wt_psd.loc[(wt_psd.index >= l) & (wt_psd.index <= u)].sum() * freq_res

        # Convert to SPL
        bg_spl = bg_spl.append(pd.Series(10 * np.log10(bg_sum / (p_ref ** 2)), name=c))
        wt_spl = wt_spl.append(pd.Series(10 * np.log10(wt_sum / (p_ref ** 2)), name=c))

    bg_spl_1_3_list.append(bg_spl)
    wt_spl_1_3_list.append(wt_spl)

# Convert all SPL dataframes to series
bg_spl_1_3_list = [bg_spl_1_3_list[i].iloc[:, 0] for i in range(len(bg_spl_1_3_list))]
wt_spl_1_3_list = [wt_spl_1_3_list[i].iloc[:, 0] for i in range(len(wt_spl_1_3_list))]

# -------------------------- OSPL --------------------------------------

bg_ospl_list = []
wt_ospl_list = []

f_lower = 800
f_upper = 3000

print('[*] Calculating OSPL...')
for bg_psd_avg, wt_psd_avg, v_inf in zip(bg_psd_avg_list, wt_psd_avg_list, v_inf_list):
    # Sum PSD in band f_lower to f_upper
    bg_sum = bg_psd_avg.loc[(bg_psd_avg.index >= f_lower) & (bg_psd_avg.index <= f_upper)].sum() * freq_res
    wt_sum = wt_psd_avg.loc[(wt_psd_avg.index >= f_lower) & (wt_psd_avg.index <= f_upper)].sum() * freq_res

    # Convert to OSPL
    bg_ospl = 10 * np.log10(bg_sum / (p_ref ** 2))
    wt_ospl = 10 * np.log10(wt_sum / (p_ref ** 2))

    bg_ospl_list.append(bg_ospl)
    wt_ospl_list.append(wt_ospl)

    print(f'    v_inf = {v_inf} m/s')
    print(f'    bg_ospl = {bg_ospl} dB')
    print(f'    wt_ospl = {wt_ospl} dB')


# ------------------------- Save data ---------------------------------
print('[*] Saving data...')
with open('saves/processed_data.pkl', 'wb') as f:
    pkl.dump({'bg_psd_list': bg_psd_avg_db_list,
              'wt_psd_list': wt_psd_avg_db_list,
              'bg_spl_list': bg_spl_list,
              'wt_spl_list': wt_spl_list,
              'bg_spl_1_3_list': bg_spl_1_3_list,
              'wt_spl_1_3_list': wt_spl_1_3_list,
              'bg_ospl_list': bg_ospl_list,
              'wt_ospl_list': wt_ospl_list,
              'v_inf_list': v_inf_list}, f)


