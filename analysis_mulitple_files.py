import mat73  # for loading data files
import seaborn as sns  # for plotting
import pandas as pd  # for data manipulation
import numpy as np  # for FFT
import matplotlib.pyplot as plt  # for plotting
import tqdm  # for progress

# Sample rate of the data
sample_rate = 48128

# Number of points to smooth FFT by
n_smoothing = 10

# Plot options
x_lim_0 = 1
x_lim_1 = 500


# Paths to the data files. Run download_data.py to download the data.
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
np_bg_mean_list = []
np_wt_mean_list = []
df_bg_fft_list = []
df_wt_fft_list = []
df_bg_wt_fft_list = []

for bg_data_path, wt_data_path in zip(bg_data_path_list, wt_data_path_list):
    print(f"\nProcessing files: {bg_data_path} and {wt_data_path}")

    # Extract data from mat file
    bg_data_dict = mat73.loadmat(bg_data_path)
    wt_data_dict = mat73.loadmat(wt_data_path)

    # Convert to dataframe
    df_bg = pd.DataFrame(bg_data_dict['Sig_Mic_bg'])
    df_wt = pd.DataFrame(wt_data_dict['Sig_Mic_rotating'])

    # Swap sample number for seconds
    df_bg.columns = df_bg.columns/sample_rate
    df_wt.columns = df_wt.columns/sample_rate

    # Get mean of each col with concat. Index is time in s, value is mean of all mics (rows)
    df_bg_mean = pd.concat([df_bg.mean(axis=0)], axis=1)
    df_wt_mean = pd.concat([df_wt.mean(axis=0)], axis=1)

    # Rename cols
    df_bg_mean.columns = ["mean"]
    df_wt_mean.columns = ["mean"]

    # Apply hamming window to data
    print("Applying hamming window...")
    for col in df_bg_mean.columns:
        df_bg_mean[col] = df_bg_mean[col] * np.hamming(df_bg_mean.shape[0])
    for col in df_wt_mean.columns:
        df_wt_mean[col] = df_wt_mean[col] * np.hamming(df_wt_mean.shape[0])

    # Create numpy array as input for np.fft
    np_bg_mean = df_bg_mean["mean"].values
    np_wt_mean = df_wt_mean["mean"].values

    # Calculate the FFT
    print("Evaluating FFT...")
    signal_fft_bg = np.fft.fft(np_bg_mean)
    signal_fft_wt = np.fft.fft(np_wt_mean)

    # Calculate the frequency axis - this creates an array of the same size as the signal with the frequency of each
    # bin (in Hz).
    ax_freq_bg = np.fft.fftfreq(np_bg_mean.size, d=1 / sample_rate)
    ax_freq_wt = np.fft.fftfreq(np_wt_mean.size, d=1 / sample_rate)

    # Create new dataframe for FFT results (absolute value, only positive frequencies)
    df_bg_fft = pd.DataFrame({"freq": ax_freq_bg, "fft": np.abs(signal_fft_bg)})
    df_bg_fft = df_bg_fft.loc[df_bg_fft["freq"] > 0]
    df_wt_fft = pd.DataFrame({"freq": ax_freq_wt, "fft": np.abs(signal_fft_wt)})
    df_wt_fft = df_wt_fft.loc[df_wt_fft["freq"] > 0]

    # Smooth the FFT results - this is done by taking the mean of the n_smoothing values around each point.
    df_bg_fft["fft"] = df_bg_fft["fft"].rolling(n_smoothing).mean()
    df_wt_fft["fft"] = df_wt_fft["fft"].rolling(n_smoothing).mean()

    # Denoise the WT signal - this is done by subtracting the BG FFT from the WT FFT.
    df_wt_bg_fft = df_wt_fft.copy()
    df_wt_bg_fft["fft"] = df_wt_bg_fft["fft"] - df_bg_fft["fft"]
    df_wt_bg_fft.loc[df_wt_bg_fft["fft"] < 0, "fft"] = 0

    # Append to list
    df_bg_mean_list.append(df_bg_mean)
    df_wt_mean_list.append(df_wt_mean)
    np_bg_mean_list.append(np_bg_mean)
    np_wt_mean_list.append(np_wt_mean)
    df_bg_fft_list.append(df_bg_fft)
    df_wt_fft_list.append(df_wt_fft)
    df_bg_wt_fft_list.append(df_wt_bg_fft)

    print("Done")


# Plot the mean ffts, noise reduced
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

for i, df_bg_wt_fft in enumerate(df_bg_wt_fft_list):
    # Get label based on file name in v_inf_lsit (e.g. U09_Wind turbine.mat -> U09)
    label = f"v_inf = {v_inf_list[i]} m/s"
    ax.plot(df_bg_wt_fft["freq"], df_bg_wt_fft["fft"], label=label)


ax.set_xscale("log")
ax.set_xlim(x_lim_0, x_lim_1)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")
ax.set_title("FFT")
ax.legend()

plt.show()

