import mat73  # for loading data files
import pandas as pd  # for data manipulation
import numpy as np  # for FFT
import os
import pickle

# Check if processed data directory exists, if not create it
if not os.path.exists("data/processed"):
    os.makedirs("data/processed")

# Clear processed data directory
for file in os.listdir("data/processed"):
    os.remove(os.path.join("data/processed", file))

# Sample rate of the data
sample_rate = 48128

# Number of samples in each time step for time-frequency analysis
sample_step = sample_rate / 2

# Paths to the data files. Run download_files.py to download the data.
# Corresponding files must be in the same order.
bg_data_path_list = ["data/downloads/U08_Background.mat",
                     "data/downloads/U09_Background.mat",
                     "data/downloads/U10_Background.mat",
                     "data/downloads/U11_Background.mat",
                     "data/downloads/U12_Background.mat"]
wt_data_path_list = ["data/downloads/U08_Wind%20turbine.mat",
                     "data/downloads/U09_Wind%20turbine.mat",
                     "data/downloads/U10_Wind%20turbine.mat",
                     "data/downloads/U11_Wind%20turbine.mat",
                     "data/downloads/U12_Wind%20turbine.mat"]

v_inf_list = [8, 9, 10, 11, 12]  # m/s

df_bg_mean_list = []
df_wt_mean_list = []
df_bg_fft_list = []
df_wt_fft_list = []
df_bg_wt_fft_list = []

for bg_data_path, wt_data_path, v_inf in zip(bg_data_path_list, wt_data_path_list, v_inf_list):

    print(f"\n[*] Processing files: {bg_data_path} and {wt_data_path}")

    # Extract data from mat file
    print("[*] Extracting data from mat files...")
    bg_data_dict = mat73.loadmat(bg_data_path)
    wt_data_dict = mat73.loadmat(wt_data_path)

    # Convert to dataframe
    df_bg = pd.DataFrame(bg_data_dict['Sig_Mic_bg'])
    df_wt = pd.DataFrame(wt_data_dict['Sig_Mic_rotating'])

    # Swap sample number for seconds
    df_bg.columns = df_bg.columns / sample_rate
    df_wt.columns = df_wt.columns / sample_rate

    # Get mean of each col with concat. Index is time in s, value is mean of all mics (rows)
    df_bg_mean = pd.concat([df_bg.mean(axis=0)], axis=1)
    df_wt_mean = pd.concat([df_wt.mean(axis=0)], axis=1)
    df_bg_mean.columns = ["mean"] # rename column
    df_wt_mean.columns = ["mean"]

    # Apply hamming window to data
    print("[*] Applying hamming window...")
    for col in df_bg_mean.columns:
        df_bg_mean[col] = df_bg_mean[col] * np.hamming(df_bg_mean.shape[0])
    for col in df_wt_mean.columns:
        df_wt_mean[col] = df_wt_mean[col] * np.hamming(df_wt_mean.shape[0])

    # Calculate the FFT and calculate the frequency axis - this creates an array of the same size as the signal with
    # the frequency of each bin (in Hz).
    print("[*] Evaluating FFT...")
    signal_fft_bg = np.fft.fft(df_bg_mean["mean"].values)
    signal_fft_wt = np.fft.fft(df_wt_mean["mean"].values)
    ax_freq_bg = np.fft.fftfreq(df_bg_mean["mean"].values.size, d=1 / sample_rate)
    ax_freq_wt = np.fft.fftfreq(df_wt_mean["mean"].values.size, d=1 / sample_rate)
    print(f"[*] Frequency resolution: {ax_freq_bg[1] - ax_freq_bg[0]} Hz")

    # Create new dataframe for FFT results (absolute value, only positive frequencies)
    df_bg_fft = pd.DataFrame({"freq": ax_freq_bg, "fft": np.abs(signal_fft_bg)})
    df_bg_fft = df_bg_fft.loc[df_bg_fft["freq"] > 0]
    df_wt_fft = pd.DataFrame({"freq": ax_freq_wt, "fft": np.abs(signal_fft_wt)})
    df_wt_fft = df_wt_fft.loc[df_wt_fft["freq"] > 0]

    # De-noise the WT signal - set negative values to 0, positive values to 1
    print("[*] De-noising WT signal...")
    df_wt_bg_fft = df_wt_fft.copy()
    df_wt_bg_fft["fft"] = df_wt_bg_fft["fft"] - df_bg_fft["fft"]
    # Remove negative values
    df_wt_bg_fft.loc[df_wt_bg_fft["fft"] < 0, "fft"] = 0
    # Set all positive FFT values to 1
    df_wt_bg_fft.loc[df_wt_bg_fft["fft"] > 0, "fft"] = 1

    # Calculate SPL (dB) : SPL = 20 * log10(FFT / 2e-5)
    print("[*] Calculating SPL...")
    df_bg_fft["SPL"] = 20 * np.log10(df_bg_fft["fft"] / 2e-5)
    df_wt_fft["SPL"] = 20 * np.log10(df_wt_fft["fft"] / 2e-5)

    # Now calculate time-dependent FFT
    # Iterate over every sample_rate/2 samples

    ##########################################################################

    print(f"[*] Evaluating FFT in time-frequency plane with time_step of {sample_step} samples...")
    df_bg_fft_t = pd.DataFrame()
    df_wt_fft_t = pd.DataFrame()

    for sample in range(0, len(df_bg_mean), int(sample_step)):
        # Calculate the FFT
        signal_fft_bg = np.fft.fft(df_bg_mean["mean"].values[sample:sample + int(sample_rate / 2)])
        signal_fft_wt = np.fft.fft(df_wt_mean["mean"].values[sample:sample + int(sample_rate / 2)])

        # Calculate the frequency axis - this creates an array of the same size as the signal with the frequency of
        # each bin (in Hz).
        ax_freq_bg = np.fft.fftfreq(df_bg_mean["mean"].values[sample:sample + int(sample_rate / 2)].size, d=1 / sample_rate)
        ax_freq_wt = np.fft.fftfreq(df_wt_mean["mean"].values[sample:sample + int(sample_rate / 2)].size, d=1 / sample_rate)

        # Create new dataframe for FFT results (absolute value, only positive frequencies)
        df_bg_fft_sample = pd.DataFrame({"freq": ax_freq_bg, "fft": np.abs(signal_fft_bg)})
        df_bg_fft_sample = df_bg_fft_sample.loc[df_bg_fft_sample["freq"] > 0]
        df_wt_fft_sample = pd.DataFrame({"freq": ax_freq_wt, "fft": np.abs(signal_fft_wt)})
        df_wt_fft_sample = df_wt_fft_sample.loc[df_wt_fft_sample["freq"] > 0]

        # Add sample number to df
        df_bg_fft_sample["sample"] = sample
        df_wt_fft_sample["sample"] = sample

        # Append to df
        df_bg_fft_t = pd.concat([df_bg_fft_t, df_bg_fft_sample], ignore_index=True)
        df_wt_fft_t = pd.concat([df_wt_fft_t, df_wt_fft_sample], ignore_index=True)

    # Denoise the WT signal - this is done by subtracting the BG FFT from the WT FFT.
    df_wt_bg_fft_t = df_wt_fft_t.copy()
    df_wt_bg_fft_t["fft"] = df_wt_bg_fft_t["fft"] - df_bg_fft_t["fft"]

    # Save processed data to pickle file
    filename = f"data/processed/{v_inf}m_s.pkl"
    print(f"Saving data to {filename}...")
    data_dict = {"v_inf": v_inf,
                 "df_bg_mean": df_bg_mean,
                 "df_wt_mean": df_wt_mean,
                 "df_bg_fft": df_bg_fft,
                 "df_wt_fft": df_wt_fft,
                 "df_wt_bg_fft": df_wt_bg_fft,
                 "df_bg_fft_t": df_bg_fft_t,
                 "df_wt_fft_t": df_wt_fft_t,
                 "df_wt_bg_fft_t": df_wt_bg_fft_t}

    with open(filename, "wb") as f:
        pickle.dump(data_dict, f)

    # Confirm file has been created
    if os.path.exists(filename):
        print("[*] File created successfully")
    else:
        raise FileNotFoundError("[!] File not created")

    for item in data_dict:
        if isinstance(data_dict[item], pd.DataFrame):
            print(f"[*] {item} columns: {data_dict[item].columns}")


# Questions:
# Go over method, check it is okay (as in, order of things)
# For spectral subtraction, should we subtract SPL or FFT?
