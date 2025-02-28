import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from ipywidgets import widgets
from scipy import signal

def acceleration(df: pd.DataFrame, label: str, color: str = "red"):
    """
    Plots the acceleration signal from a DataFrame.

    Args:
        df: DataFrame containing 'Time' and '{label} Acceleration' columns.
        label: Direction of the acceleration signal to plot.
        color: Color of the plot line (default is 'red').

    Returns:
        None
    """
    plt.figure(figsize=(15, 6))
    plt.plot(
        df["Time"],
        df[f"{label} Acceleration"],
        color=color,
        linestyle="-",
        label=label,
    )

    plt.title(f"{label} Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel(r"Acceleration ($m/s^{2}$)")
    plt.legend()
    plt.savefig(f"results/{label} Acceleration.png", dpi=300)
    plt.show()


def normalized_fft_results(df: pd.DataFrame, label: str, color: str = "red") -> None:
    """
    Plots the normalized FFT magnitude spectrum from a DataFrame.

    Args:
        df: DataFrame containing '{label} Frequency' and '{label} Magnitude' columns.
        label: The label of the vibration (e.g., 'X', 'Y', 'Z').
        color: Color of the plot line (default is 'red').

    Returns:
        None

    Assumptions:
        - 'df_fft' contains '{label} Frequency' and '{label} Magnitude' columns.
    """
    frequencies = df["Frequency"]
    magnitudes = df[f"{label} Magnitude"]

    plt.figure(figsize=(10, 6))
    plt.plot(
        frequencies[:],
        magnitudes[:] / max(magnitudes[:]),
        color=color,
        linestyle="-",
        label=label,
    )
    plt.title("FFT Normalized Magnitude Spectrum ")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.savefig(f"results/normalized_fft_for_{label}.png", dpi=300)
    plt.show()


def fft_results(df: pd.DataFrame, label: str, color: str = "red") -> None:
    """
    Plots the FFT magnitude spectrum from a DataFrame and saves the figure.

    Args:
        df: DataFrame containing 'Frequency' and '{label} Magnitude' columns.
        label: The label of the vibration (e.g., 'X', 'Y', 'Z').
        color: Color of the plot line (default is 'red').

    Returns:
        None

    Assumptions:
        - 'df' contains 'Frequency' and '{label} Magnitude' columns.
    """
    frequencies = df["Frequency"]
    magnitudes = df[f"{label} Magnitude"]

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies[:], magnitudes[:], color=color, linestyle="-", label=label)
    plt.title("FFT Magnitude Spectrum ")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.savefig(f"results/fft_for_{label}.png", dpi=300)
    plt.show()


def fft_results_period_domain(df: pd.DataFrame, label: str, color: str = "red", log_scale: bool = False) -> None:
    """
    Plots the FFT magnitude spectrum from a DataFrame and saves the figure.

    Args:
        df: DataFrame containing 'Frequency' and '{label} Magnitude' columns.
        label: The label of the vibration (e.g., 'X', 'Y', 'Z').
        color: Color of the plot line (default is 'red').
        log_scale: Whether to use a logarithmic scale for the period axis (default is False).

    Returns:
        None

    Assumptions:
        - 'df' contains 'Frequency' and '{label} Magnitude' columns.
    """
    frequencies = df["Frequency"]
    magnitudes = df[f"{label} Magnitude"]

    periods = 1 / frequencies

    plt.figure(figsize=(10, 6))
    plt.plot(periods, magnitudes, color=color, linestyle="-", label=label)

    if log_scale:
        plt.xscale('log')

    plt.title("FFT Period Domain Spectrum")
    plt.xlabel("T (s)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"results/fft_period_domain {label}.png", dpi=300)

def wavelet_spectrum_views(
    df: pd.DataFrame,
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    label: str,
    elevation: int = 0,
    rotation: int = 0,
    label_size: int = 10,
    label_offset: float = 0.1,
):
    """
    Plots the time-frequency-magnitude wavelet spectrum in four subplots: XY, XZ, YZ, and 3D and saves the figure.

    Args:
        df: DataFrame containing the 'Time' column.
        spectrum: Magnitude spectrum of the CWT coefficients.
        frequencies: Frequencies corresponding to the scales.
        label: Direction of the acceleration data (e.g., 'X', 'Y', 'Z').
        elevation: Elevation angle for the 3D plot (default: 0).
        rotation: Rotation angle for the 3D plot (default: 0).
        label_size: Font size for labels (default: 10).
        label_offset: Offset for labels (default: 0.1).

    Returns:
        None (displays the subplots).
    """
    time_step = df["Time"].iloc[1] - df["Time"].iloc[0]
    sampling_rate = 1 / time_step
    nyquist_freq = sampling_rate / 2

    time_min, time_max = df["Time"].min(), df["Time"].max()
    freq_min, freq_max = frequencies.min(), nyquist_freq

    mask_x = (df["Time"] >= time_min) & (df["Time"] <= time_max)
    mask_y = (frequencies >= freq_min) & (frequencies <= freq_max)

    time_filtered = df["Time"][mask_x].values
    frequencies_filtered = frequencies[mask_y]
    spectrum_filtered = spectrum[np.ix_(mask_y, mask_x)]

    X, Y = np.meshgrid(time_filtered, frequencies_filtered)

    if X.shape != spectrum_filtered.shape:
        print(
            f"Shape mismatch: X{X.shape}, Y{Y.shape}, spectrum{spectrum_filtered.shape}"
        )
        return

    fig = plt.figure(figsize=(20, 15))

    # XY subplot (Top View)
    ax1 = fig.add_subplot(221)
    c1 = ax1.contourf(X, Y, spectrum_filtered, cmap="viridis")
    ax1.set_xlabel("Time (s)", fontsize=label_size, labelpad=label_offset)
    ax1.set_ylabel("Frequency (Hz)", fontsize=label_size, labelpad=label_offset)
    ax1.set_title(f"{label} Wavelet Spectrum (Top View)", fontsize=label_size)
    fig.colorbar(c1, ax=ax1)

    # XZ subplot (Side View 1)
    ax2 = fig.add_subplot(222)
    c2 = ax2.contourf(
        X, spectrum_filtered, Y, cmap="viridis"
    )  # Swapped Y and spectrum_filtered for side view
    ax2.set_xlabel("Time (s)", fontsize=label_size, labelpad=label_offset)
    ax2.set_ylabel("Magnitude", fontsize=label_size, labelpad=label_offset)
    ax2.set_title(f"{label} Wavelet Spectrum (Side View 1)", fontsize=label_size)
    fig.colorbar(c2, ax=ax2)

    # YZ subplot (Side View 2)
    ax3 = fig.add_subplot(223)
    c3 = ax3.contourf(
        spectrum_filtered, Y, X, cmap="viridis"
    )  # Swapped X and spectrum_filtered for side view
    ax3.set_xlabel("Magnitude", fontsize=label_size, labelpad=label_offset)
    ax3.set_ylabel("Frequency (Hz)", fontsize=label_size, labelpad=label_offset)
    ax3.set_title(f"{label} Wavelet Spectrum (Side View 2)", fontsize=label_size)
    fig.colorbar(c3, ax=ax3)

    # 3D subplot
    ax4 = fig.add_subplot(224, projection="3d")
    ax4.plot_surface(X, Y, spectrum_filtered, cmap="viridis")
    ax4.set_xlabel("Time (s)", fontsize=label_size, labelpad=label_offset)
    ax4.set_ylabel("Frequency (Hz)", fontsize=label_size, labelpad=label_offset)
    ax4.set_zlabel("Magnitude", fontsize=label_size, labelpad=label_offset)
    ax4.set_title(f"{label} Wavelet Spectrum (3D View)", fontsize=label_size)
    ax4.view_init(elev=elevation, azim=rotation)

    box = ax4.get_position()
    y_height = box.height * 1.2  # Increase height by 20%
    ax4.set_position([box.x0, box.y0, box.width, y_height])

    plt.tight_layout()
    plt.savefig(f"results/ws_views_for_for_{label}.png", dpi=300)
    plt.show()


def wavelet_spectrum_time_frequency(
    df: pd.DataFrame,
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    label: str,
    label_size: int = 10,
    label_offset: float = 0.1,
):
    """
    Plots the time-frequency wavelet spectrum (Top View) and saves the figure.

    Args:
        df: DataFrame containing the 'Time' column.
        spectrum: Magnitude spectrum of the CWT coefficients.
        frequencies: Frequencies corresponding to the scales.
        label: Direction of the acceleration data (e.g., 'X', 'Y', 'Z').
        label_size: Font size for labels (default: 10).
        label_offset: Offset for labels (default: 0.1).

    Returns:
        None (displays the plot).
    """
    time_step = df["Time"].iloc[1] - df["Time"].iloc[0]
    sampling_rate = 1 / time_step
    nyquist_freq = sampling_rate / 2

    time_min, time_max = df["Time"].min(), df["Time"].max()
    freq_min, freq_max = frequencies.min(), nyquist_freq

    mask_x = (df["Time"] >= time_min) & (df["Time"] <= time_max)
    mask_y = (frequencies >= freq_min) & (frequencies <= freq_max)

    time_filtered = df["Time"][mask_x].values
    frequencies_filtered = frequencies[mask_y]
    spectrum_filtered = spectrum[np.ix_(mask_y, mask_x)]

    X, Y = np.meshgrid(time_filtered, frequencies_filtered)

    if X.shape != spectrum_filtered.shape:
        print(
            f"Shape mismatch: X{X.shape}, Y{Y.shape}, spectrum{spectrum_filtered.shape}"
        )
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    contour = ax.contourf(X, Y, spectrum_filtered, cmap="viridis")
    ax.set_xlabel("Time (s)", fontsize=label_size, labelpad=label_offset)
    ax.set_ylabel("Frequency (Hz)", fontsize=label_size, labelpad=label_offset)
    ax.set_title(f"{label} Wavelet Spectrum (Top View)", fontsize=label_size)
    fig.colorbar(contour, ax=ax)

    plt.tight_layout()
    plt.savefig(f"results/ws_tf_for_{label}.png", dpi=300)
    plt.show()

def wavelet_spectrum_time_magnitude(
    df: pd.DataFrame,
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    label: str,
    label_size: int = 10,
    label_offset: float = 0.1,
):
    """
    Plots the time-magnitude wavelet spectrum (Side View 1) and saves the figure.

    Args:
        df: DataFrame containing the 'Time' column.
        spectrum: Magnitude spectrum of the CWT coefficients.
        frequencies: Frequencies corresponding to the scales.
        label: Direction of the acceleration data (e.g., 'X', 'Y', 'Z').
        label_size: Font size for labels (default: 10).
        label_offset: Offset for labels (default: 0.1).

    Returns:
        None (displays the plot).
    """
    time_min, time_max = df["Time"].min(), df["Time"].max()
    freq_min, freq_max = frequencies.min(), frequencies.max()

    mask_x = (df["Time"] >= time_min) & (df["Time"] <= time_max)
    mask_y = (frequencies >= freq_min) & (frequencies <= freq_max)

    time_filtered = df["Time"][mask_x].values
    frequencies_filtered = frequencies[mask_y]
    spectrum_filtered = spectrum[np.ix_(mask_y, mask_x)]

    X, Y = np.meshgrid(time_filtered, frequencies_filtered)

    if X.shape != spectrum_filtered.shape:
        print(
            f"Shape mismatch: X{X.shape}, Y{Y.shape}, spectrum{spectrum_filtered.shape}"
        )
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    contour = ax.contourf(
        X, spectrum_filtered, Y, cmap="viridis"
    )  # Transpose spectrum

    ax.set_xlabel("Time (s)", fontsize=label_size, labelpad=label_offset)
    ax.set_ylabel("Magnitude", fontsize=label_size, labelpad=label_offset)
    ax.set_title(f"{label} Wavelet Spectrum (Side View 1)", fontsize=label_size)

    fig.colorbar(contour, ax=ax)

    plt.tight_layout()
    plt.savefig(f"results/ws_tm_for_{label}.png", dpi=300)
    plt.show()


def wavelet_spectrum_frequency_magnitude(
    df: pd.DataFrame,
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    label: str,
    label_size: int = 10,
    label_offset: float = 0.1,
):
    """
    Plots the frequency-magnitude wavelet spectrum (Side View 2).

    Args:
        df: DataFrame containing the 'Time' column.
        spectrum: Magnitude spectrum of the CWT coefficients.
        frequencies: Frequencies corresponding to the scales.
        label: Direction of the acceleration data (e.g., 'X', 'Y', 'Z').
        label_size: Font size for labels (default: 10).
        label_offset: Offset for labels (default: 0.1).

    Returns:
        None (displays the plot).
    """
    time_min, time_max = df["Time"].min(), df["Time"].max()
    freq_min, freq_max = frequencies.min(), frequencies.max()

    mask_x = (df["Time"] >= time_min) & (df["Time"] <= time_max)
    mask_y = (frequencies >= freq_min) & (frequencies <= freq_max)

    time_filtered = df["Time"][mask_x].values
    frequencies_filtered = frequencies[mask_y]
    spectrum_filtered = spectrum[np.ix_(mask_y, mask_x)]

    X, Y = np.meshgrid(time_filtered, frequencies_filtered)

    if X.shape != spectrum_filtered.shape:
        print(
            f"Shape mismatch: X{X.shape}, Y{Y.shape}, spectrum{spectrum_filtered.shape}"
        )
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    c = ax.contourf(
        spectrum_filtered, Y, X, cmap="viridis"
    )  # Swapped X and spectrum_filtered for side view
    ax.set_xlabel("Magnitude", fontsize=label_size, labelpad=label_offset)
    ax.set_ylabel("Frequency (Hz)", fontsize=label_size, labelpad=label_offset)
    ax.set_title(f"{label} Wavelet Spectrum (Side View 2)", fontsize=label_size)
    fig.colorbar(c, ax=ax)

    plt.tight_layout()
    plt.savefig(f"results/ws_fm_for_{label}.png", dpi=300)
    plt.show()


def peaks(
    df: pd.DataFrame,
    label: str,
    height: float,
    distance: float,
    log_scale: bool = False,
    file_location: str = "results/fft_peaks.png",
):
    """
    Finds and plots peaks in the FFT magnitude spectrum.

    Args:
        df: DataFrame containing 'Frequency' and '{label} Magnitude' columns.
        label: The label of the vibration ('X', 'Y' or 'Z').
        height: Required height of peaks.
        distance: Required minimal horizontal distance (in samples) between neighboring peaks.
        log_scale: Whether to use a logarithmic scale for the frequency axis (default is False).
        file_location: Path where the plot will be saved (default is "results/fft_peaks.png").

    Returns:
        None (displays the plot).
    """
    frequencies = df["Frequency"]
    magnitudes = df[f"{label} Magnitude"]

    peaks_scipy, _ = signal.find_peaks(magnitudes, height=height, distance=distance)

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, magnitudes, label=label)
    plt.plot(
        frequencies[peaks_scipy],
        magnitudes[peaks_scipy],
        "^",
        label="Peaks",
    )

    if log_scale:
        plt.xscale("log")

    plt.title(f"{label} and peaks found")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    plt.savefig(file_location)
    plt.show()

