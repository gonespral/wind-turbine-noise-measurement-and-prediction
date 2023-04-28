"""
This script contains various plotting functions for the processed data.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Configuration parameters
sample_rate = 48128
x_lim_0 = 1
x_lim_1 = 20000
x_scale = "log"
y_scale = "linear"
NFFT_val = 256
vmin = 0
vmax = 10

# Path to pickle files
pickle_paths = ["data/processed/8m_s.pkl",
                "data/processed/9m_s.pkl",
                "data/processed/10m_s.pkl",
                "data/processed/11m_s.pkl",
                "data/processed/12m_s.pkl"]

# Load data from pickle files
pickle_files = []
for path in pickle_paths:
    with open(path, "rb") as f:
        pickle_files.append(pickle.load(f))
    print(f"Loaded {path}. Found keys: {pickle_files[-1].keys()}")
print()

# Set matplotlib colour scheme for plots
# Possible values: "deep", "muted", "pastel", "bright", "dark", "colorblind", "Paired", "Set2", "Set1", "Set3",
# "tab10", "tab20", "tab20b", "tab20c", "Accent", "Dark2", "Paired", "Pastel1", "Pastel2", "Set1", "Set2", "Set3"
sns.set_palette("bright")


# Format for accessing data in data_files:
# eg. df_bg_mean = data_files[0]["df_bg_mean"] # background mean
# eg. v_inf = data_files[3]["v_inf"] # inflow velocity in m/s


def info():
    """
    Print a list of commands that can be used in this script.
    :return:
    """
    print("\n--- Available functions ---")
    # Get dir() of functions of  global namespace written by user
    for name, obj in globals().items():
        if callable(obj) and obj.__module__ == "__main__":
            print(f"    {name}()")
    print()

    print("--- Available variables ---")
    print(f"    sample_rate = {sample_rate}")
    print(f"    x_lim_0 = {x_lim_0}")
    print(f"    x_lim_1 = {x_lim_1}")
    print(f"    x_scale = {x_scale}")
    print(f"    y_scale = {y_scale}")
    print(f"    NFFT_val = {NFFT_val}\n")

    print(f"--- Loaded pickle files ---")
    for i, path in enumerate(pickle_paths):
        print(f"    {i}: {path}")
    print()


def plot_fft(indices: list = [0, 1, 2, 3, 4], modes: list = ["denoised"]) -> None:
    """
    Plot FFT of a set of measurements. Can plot background noise, wind turbine noise, or de-noised wind turbine noise.
    :param modes: mode to plot. Can be "bg", "wt", or "denoised"
    :param indices: indices of the data files to plot
    :return:
    """

    valid_modes = ["bg", "wt", "denoised"]
    for mode in modes:
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are {valid_modes}")
    print(f"Plotting {modes} FFT for indices {indices}.\n")

    # Multiple plots in one figure (per mode)
    fig, ax = plt.subplots(len(modes), 1, figsize=(10, 10))

    # AxesSubplot object is not subscriptable, so we need to convert it to a list
    if len(modes) == 1:
        ax = [ax]

    j = 0
    # Background noise
    if "bg" in modes:
        for i in indices:
            df_bg_fft_ = pickle_files[i]["df_bg_fft"]
            ax[j].plot(df_bg_fft_["freq"],
                       df_bg_fft_["SPL"],
                       label=f"{pickle_files[i]['v_inf']}m/s")
        ax[j].set_xscale(x_scale)
        ax[j].set_yscale(y_scale)
        ax[j].set_xlim(x_lim_0, x_lim_1)
        ax[j].set_xlabel("Frequency (Hz)")
        ax[j].set_ylabel("SPL (dB)")
        ax[j].set_title("Background FFT")
        ax[j].legend()

        j += 1

    # Wind turbine noise
    if "wt" in modes:
        for i in indices:
            df_wt_fft_ = pickle_files[i]["df_wt_fft"]
            ax[j].plot(df_wt_fft_["freq"],
                       df_wt_fft_["SPL"],
                       label=f"{pickle_files[i]['v_inf']}m/s")
        ax[j].set_xscale(x_scale)
        ax[j].set_yscale(y_scale)
        ax[j].set_xlim(x_lim_0, x_lim_1)
        ax[j].set_xlabel("Frequency (Hz)")
        ax[j].set_ylabel("SPL (dB)")
        ax[j].set_title("Wind Turbine FFT")
        ax[j].legend()

        j += 1

    # De-noised wind turbine noise
    if "denoised" in modes:
        for i in indices:
            df_wt_bg_fft_ = pickle_files[i]["df_wt_bg_fft"]
            ax[j].plot(df_wt_bg_fft_["freq"],
                       df_wt_bg_fft_["fft"],
                       label=f"{pickle_files[i]['v_inf']}m/s")
        ax[j].set_xscale(x_scale)
        ax[j].set_yscale(y_scale)
        ax[j].set_xlim(x_lim_0, x_lim_1)
        ax[j].set_xlabel("Frequency (Hz)")
        ax[j].set_ylabel("FFT")
        ax[j].set_title("Wind Turbine - Background (de-noised) FFT")
        ax[j].legend()

    plt.show()


def plot_psd(indices: list = [0, 1, 2, 3, 4], modes: list = ["denoised"]) -> None:
    """
    Plot PSD of a set of measurements. Can plot background noise, wind turbine noise, or de-noised wind turbine noise.
    :param indices: indices of the data files to plot
    :param modes: mode to plot. Can be "bg", "wt", or "denoised"
    :return:
    """

    valid_modes = ["bg", "wt", "denoised"]
    for mode in modes:
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are {valid_modes}")
    print(f"Plotting {modes} PSD for indices {indices}.\n")

    # Multiple plots in one figure (per mode)
    fig, ax = plt.subplots(len(modes), 1, figsize=(10, 10))

    # AxesSubplot object is not subscriptable, so we need to convert it to a list
    if len(modes) == 1:
        ax = [ax]

    j = 0
    # Background noise
    if "bg" in modes:
        for i in indices:
            df_bg_fft_ = pickle_files[i]["df_bg_fft"]
            # Use ax.psd() instead of ax.plot() to plot PSD
            ax[j].psd(x=df_bg_fft_["SPL"],
                      Fs=sample_rate,
                      label=f"{pickle_files[i]['v_inf']}m/s")
        ax[j].set_xscale(x_scale)
        ax[j].set_yscale(y_scale)
        ax[j].set_xlim(x_lim_0, x_lim_1)
        ax[j].set_xlabel("Frequency (Hz)")
        ax[j].set_ylabel("SPL (dB)")
        ax[j].set_title("Background PSD")
        ax[j].legend()

        j += 1

    # Wind turbine noise
    if "wt" in modes:
        for i in indices:
            df_wt_fft_ = pickle_files[i]["df_wt_fft"]
            ax[j].psd(x=df_wt_fft_["SPL"],
                      Fs=sample_rate,
                      label=f"{pickle_files[i]['v_inf']}m/s")
        ax[j].set_xscale(x_scale)
        ax[j].set_yscale(y_scale)
        ax[j].set_xlim(x_lim_0, x_lim_1)
        ax[j].set_xlabel("Frequency (Hz)")
        ax[j].set_ylabel("SPL (dB)")
        ax[j].set_title("Wind Turbine PSD")
        ax[j].legend()

        j += 1

    # De-noised wind turbine noise
    if "denoised" in modes:
        for i in indices:
            df_wt_bg_fft_ = pickle_files[i]["df_wt_bg_fft"]
            ax[j].psd(x=df_wt_bg_fft_["SPL"],
                      Fs=sample_rate,
                      label=f"{pickle_files[i]['v_inf']}m/s")
        ax[j].set_xscale(x_scale)
        ax[j].set_yscale(y_scale)
        ax[j].set_xlim(x_lim_0, x_lim_1)
        ax[j].set_xlabel("Frequency (Hz)")
        ax[j].set_ylabel("SPL (dB)")
        ax[j].set_title("Wind Turbine - Background (de-noised) PSD")
        ax[j].legend()

    plt.show()


def plot_waterfall(index: int = 0, modes: list = ["denoised"]) -> None:
    """
    Plot waterfall of a set of measurements. Can plot background noise, wind turbine noise, or de-noised wind turbine noise.
    :param index: index of the data file to plot
    :param modes: mode to plot. Can be "bg", "wt", or "denoised"
    :return:
    """
    valid_modes = ["bg", "wt", "denoised"]
    for mode in modes:
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are {valid_modes}")

    print(f"Plotting {modes} waterfall for index {index}.\n")

    # Multiple plots in one figure (per mode)
    fig, ax = plt.subplots(len(modes), 1, figsize=(10, 10))

    # AxesSubplot object is not subscriptable, so we need to convert it to a list
    if len(modes) == 1:
        ax = [ax]

    j = 0
    # Background noise
    if "bg" in modes:
        df_bg_ = pickle_files[index]["df_bg_fft_t"]
        sns.heatmap(df_bg_.pivot("sample", "freq", "fft"), ax=ax[j], cmap="viridis", vmin=vmin, vmax=vmax)
        ax[j].set_xscale(x_scale)
        ax[j].set_xlim(x_lim_0, x_lim_1)
        ax[j].set_xlabel("Frequency (Hz)")
        ax[j].set_ylabel("Sample")
        ax[j].set_title(f"Background {pickle_files[index]['v_inf']}m/s")

        j += 1

    # Wind turbine noise
    if "wt" in modes:
        df_wt_ = pickle_files[index]["df_wt_fft_t"]
        sns.heatmap(df_wt_.pivot("sample", "freq", "fft"), ax=ax[j], cmap="viridis", vmin=vmin, vmax=vmax)
        ax[j].set_xscale(x_scale)
        ax[j].set_xlim(x_lim_0, x_lim_1)
        ax[j].set_xlabel("Frequency (Hz)")
        ax[j].set_ylabel("Sample")
        ax[j].set_title(f"Wind Turbine {pickle_files[index]['v_inf']}m/s")

        j += 1

    # De-noised wind turbine noise
    if "denoised" in modes:
        df_wt_bg_ = pickle_files[index]["df_wt_bg_fft_t"]
        sns.heatmap(df_wt_bg_.pivot("sample", "freq", "fft"), ax=ax[j], cmap="viridis", vmin=vmin, vmax=vmax)
        ax[j].set_xscale(x_scale)
        ax[j].set_xlim(x_lim_0, x_lim_1)
        ax[j].set_xlabel("Frequency (Hz)")
        ax[j].set_ylabel("Sample")
        ax[j].set_title(f"Wind Turbine - Background (de-noised) {pickle_files[index]['v_inf']}m/s")

    plt.show()


if __name__ == "__main__":
    with open("misc/windmill.txt", "r") as f:
        print(f.read())
    print("--- WIND TURBINE NOISE ANALYSIS TOOL ---\n Type 'info()' for a list of commands.")

    # noinspection PyUnresolvedReferences
    get_ipython()




