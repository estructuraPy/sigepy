from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from ipywidgets import interactive, IntSlider, FloatSlider
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
    height: float=0.1,
    distance: float=1,
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
        "o",
        label="Peaks",
    )

    for peak in peaks_scipy:
        plt.annotate(
            f"{frequencies[peak]:.2f}",
            (frequencies[peak], magnitudes[peak]),
            textcoords="offset points",
            xytext=(10, 0),  # Offset to the right of the marker
            ha='left'
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


def wavelet_spectrum(
    time: np.ndarray,
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    min_time: float,
    max_time: float,
    min_frequency: float,
    max_frequency: float,
    label_size: int = 14,
    elevation: float = 30.0,
    rotation: float = 30.0,
) -> plt.Figure:
    """
    Plots the 3D wavelet spectrum based on provided parameters.

    Args:
        time: Array of time values.
        frequencies: Array of frequency values.
        spectrum: 2D array representing the wavelet spectrum.
        min_time: Minimum time value for the plot.
        max_time: Maximum time value for the plot.
        min_frequency: Minimum frequency value for the plot.
        max_frequency: Maximum frequency value for the plot.
        label_size: Font size for the axis labels.
        elevation: Elevation angle for the 3D plot.
        rotation: Azimuthal (horizontal) rotation angle for the 3D plot.

    Returns:
        The matplotlib figure.

    Assumptions:
        - Time and frequency ranges are valid.
    """
    mask_x = (time >= min_time) & (time <= max_time)
    mask_y = (frequencies >= min_frequency) & (frequencies <= max_frequency)

    time_filtered = time[mask_x]
    frequencies_filtered = frequencies[mask_y]
    spectrum_filtered = spectrum[np.ix_(mask_y, mask_x)]

    X, Y = np.meshgrid(time_filtered, frequencies_filtered)

    if X.shape != spectrum_filtered.shape:
        print(f"Shape mismatch: X{X.shape}, Y{Y.shape}, Z{spectrum_filtered.shape}")
        return None  # Errores: Avoids the code to continue running and errors to come

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, spectrum_filtered, cmap="viridis")

    ax.set_xlabel("Time (s)", fontsize=label_size)
    ax.set_ylabel("Frequency (Hz)", fontsize=label_size)
    ax.set_zlabel("Magnitude", fontsize=label_size)
    ax.set_title("Time - Frequency - Magnitude Wavelet Spectrum", fontsize=label_size)
    ax.view_init(elev=elevation, azim=rotation)
    return fig


def wavelet_spectrum_gif(
    time: np.ndarray,
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    file_location: str,
    min_time: float,
    max_time: float,
    min_frequency: float,
    max_frequency: float,
):
    """
    Saves a GIF animation of the rotating 3D wavelet spectrum.

    Args:
        time: Array of time values.
        frequencies: Array of frequency values.
        spectrum: 2D array representing the wavelet spectrum.
        file_location: Path to save the GIF file.
        min_time: Minimum time value for the plot.
        max_time: Maximum time value for the plot.
        min_frequency: Minimum frequency value for the plot.
        max_frequency: Maximum frequency value for the plot.

    Returns:
        None
    """
    frames = []
    for angle in range(0, 360, 10):
        fig = wavelet_spectrum(
            time,
            frequencies,
            spectrum,
            min_time,
            max_time,
            min_frequency,
            max_frequency,
            elevation=30,
            rotation=angle,
        )  # Fixed elevation, varying rotation

        if fig is None:
            continue

        try:
            buf = BytesIO()
            fig.canvas.draw()
            fig.savefig(buf, format="png")
            buf.seek(0)
            image = Image.open(buf)
            frames.append(image)
            plt.close(fig)
        except Exception as e:
            print(f"Error processing frame: {e}")
            if fig:
                plt.close(fig)  # Ensure the figure is closed even on error
            continue

    if frames:  # Only save the GIF if there are frames
        frames[0].save(
            file_location,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=100,
        )
        print("GIF saved as 'wavelet_spectrum.gif'")
    else:
        print("No frames were generated. GIF not saved.")


def interactive_wavelet_spectrum(
    time: np.ndarray,
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    min_time: float,
    max_time: float,
    min_frequency: float,
    max_frequency: float,
):
    """
    Displays an interactive plot of the wavelet spectrum using ipywidgets and Matplotlib.

    Args:
        time: Array of time values.
        frequencies: Array of frequency values.
        spectrum: 2D array representing the wavelet spectrum.
        min_time: Minimum time value for the plot.
        max_time: Maximum time value for the plot.
        min_frequency: Minimum frequency value for the plot.
        max_frequency: Maximum frequency value for the plot.

    Returns:
        The interactive plot widget.
    """

    def plot_surface(elevation, rotation, min_time_val, max_time_val, min_frequency_val, max_frequency_val):
        """
        Helper function to create the surface plot with given parameters.
        """
        mask_x = (time >= min_time_val) & (time <= max_time_val)
        mask_y = (frequencies >= min_frequency_val) & (frequencies <= max_frequency_val)

        time_filtered = time[mask_x]
        frequencies_filtered = frequencies[mask_y]
        spectrum_filtered = spectrum[np.ix_(mask_y, mask_x)]

        X, Y = np.meshgrid(time_filtered, frequencies_filtered)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, spectrum_filtered, cmap='viridis')

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_zlabel("Magnitude")
        ax.set_title("Interactive Wavelet Spectrum")

        # Set view angle
        ax.view_init(elev=elevation, azim=rotation)

        plt.show()  # Show the plot

    elevation_slider = IntSlider(value=30, min=0, max=90, step=5, description="Elevation")
    rotation_slider = IntSlider(value=0, min=0, max=360, step=10, description="Rotation")
    min_time_slider = FloatSlider(value=min_time, min=min_time, max=max_time, step=(max_time - min_time) / 50, description="Min Time")
    max_time_slider = FloatSlider(value=max_time, min=min_time, max=max_time, step=(max_time - min_time) / 50, description="Max Time")
    min_frequency_slider = FloatSlider(value=min_frequency, min=min_frequency, max=max_frequency, step=(max_frequency - min_frequency) / 50, description="Min Frequency")
    max_frequency_slider = FloatSlider(value=max_frequency, min=min_frequency, max=max_frequency, step=(max_frequency - min_frequency) / 50, description="Max Frequency")

    interactive_plot = interactive(
        plot_surface,
        elevation=elevation_slider,
        rotation=rotation_slider,
        min_time_val=min_time_slider,
        max_time_val=max_time_slider,
        min_frequency_val=min_frequency_slider,
        max_frequency_val=max_frequency_slider
    )

    return interactive_plot


def ssi_cov(df, label, stability_data=None):
    """
    Plots the SSCOV results and optionally the stability diagram.

    Args:
        frequencies (array-like): Frequencies corresponding to the power spectrum.
        power_spectrum (array-like): Power spectrum values.
        label (str): Data label (e.g., 'X').
        stability_data (tuple, optional): Tuple containing (frequencies_stab, num_poles). If None, stability plot is skipped.
    """
    frequencies = df['Frequency']
    power_spectrum = df[f'Power Spectrum {label}']
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(frequencies, power_spectrum, label=f"Power Spectrum {label}", color='tab:blue')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power', color='tab:blue')
    ax1.tick_params('y', labelcolor='tab:blue')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # Plot stability diagram on a second y-axis
    if stability_data:
        try:
            frequencies_stab, num_poles = stability_data
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.plot(frequencies_stab, num_poles, marker='o', linestyle='-', color='tab:orange', label='Number of Poles')
            ax2.set_ylabel('Number of Poles', color='tab:orange')  # we already handled the x-label with ax1
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            ax2.legend(loc='upper right')  # Add legend for the second y-axis

            # Check that both axes are covered
            fig.tight_layout()
            plt.title(f'Power Spectrum and Stability Diagram ({label})')
        except ValueError as ve:
            print(f"ValueError plotting stability data: {ve}. Skipping stability plot.")
        except Exception as e:
            print(f"Error plotting stability data: {e}. Skipping stability plot.")
    else:
        plt.title(f'Power Spectrum ({label})')

    plt.show()