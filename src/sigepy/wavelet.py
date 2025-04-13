import warnings
import pandas as pd
import pywt
from scipy.signal import detrend, butter, filtfilt
from scipy.linalg import hankel, svd, eig
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import detrend, butter, filtfilt
from scipy.linalg import hankel, svd, eig, toeplitz
from IPython.display import display
from collections import OrderedDict
from typing import Tuple,Dict, Union, List
from dataclasses import dataclass, field
from tqdm import tqdm
import gc
from io import BytesIO
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import IntSlider, FloatSlider
import os


def calculate_cwt(df: pd.DataFrame, label: str, wavelet_function: str = "morl",
        min_scale: int = 2, max_scale: int = 32, magnitude_type: str='calculated',
        magnitude_factor: float=1.0) -> pd.DataFrame:
    """
    Performs Continuous Wavelet Transform (CWT) analysis on acceleration data.

    Args:
        magnitude_factor: Factor to multiply the magnitude spectrum by when magnitude_type is 'normalized'.
        magnitude_type: Type of magnitude spectrum to return ('normalized' or 'calculated').
        df: DataFrame containing 'Time' and '{label} Acceleration' columns.
        label: Direction of the acceleration data (e.g., 'X', 'Y', 'Z').
        wavelet_function: Wavelet function to use (default: 'morl').
        min_scale: Minimum scale for CWT (default: 1).
        max_scale: Maximum scale for CWT (default: 32).

    Returns:
        A tuple containing:
            - spectrum: Magnitude spectrum of the CWT coefficients.
            - frequencies: Frequencies corresponding to the scales.

    Assumptions:
        - 'df' contains 'Time' and '{label} Acceleration' columns.
        - 'Time' data is uniformly sampled.
        - 'label' is a valid column in df
    """
    # Validations
    if f"{label} Acceleration" not in df.columns:
        raise ValueError(f"{label} Acceleration is not found in DataFrame.")
    if "Time" not in df.columns:
        raise ValueError("The DataFrame must contain a 'Time' column.")

    time_step = df["Time"].iloc[1] - df["Time"].iloc[0]

    scales = np.arange(min_scale, max_scale)
    coefficients, frequencies = pywt.cwt(
        df[f"{label} Acceleration"], scales, wavelet_function, time_step
    )

    # Magnitude spectrum
    spectrum = np.abs(coefficients)

    if magnitude_type == 'normalized':
        modified_spectrum = spectrum / np.max(spectrum) * magnitude_factor
    elif magnitude_type == 'calculated':
        modified_spectrum = spectrum
    else:
        raise ValueError("'normalized' or 'calculated'")

    return modified_spectrum, frequencies


def plot_spectrum_gif(
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
    # Crear el directorio results si no existe
    os.makedirs(os.path.dirname(file_location), exist_ok=True)
    
    frames = []
    X, Y = np.meshgrid(time, frequencies)
    
    # Usar menos ángulos y mayor paso
    for angle in range(0, 360, 45):  # 8 frames en total
        fig = plt.figure(figsize=(6, 4), dpi=80)  # Figura más pequeña y menor DPI
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, spectrum, cmap='viridis', 
                             rcount=50, ccount=50)  # Reducir la resolución de la superficie
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_zlabel('Magnitude')
        ax.view_init(elev=30, azim=angle)
        
        plt.tight_layout()
        
        # Capturar frame de manera más eficiente
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', 
                   pad_inches=0.1, dpi=80)
        buf.seek(0)
        img = Image.open(buf)
        img = img.convert('P', palette=Image.ADAPTIVE, colors=256)  # Reducir colores
        frames.append(img)
        
        plt.close(fig)
        buf.close()

    if frames:
        frames[0].save(
            file_location,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=200,  # Más tiempo entre frames
            loop=0
        )
        print(f"GIF saved as '{file_location}'")
    else:
        print("No frames were generated. GIF not saved.")


def plot_interactive_wavelet_spectrum(
    time: np.ndarray,
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    min_time: float,
    max_time: float,
    min_frequency: float,
    max_frequency: float,
) -> None:
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
        None
    """
    def update_plot(elevation, rotation, min_time_val, max_time_val, min_frequency_val, max_frequency_val):
        """Update the plot with new parameters."""
        mask_x = (time >= min_time_val) & (time <= max_time_val)
        mask_y = (frequencies >= min_frequency_val) & (frequencies <= max_frequency_val)

        time_filtered = time[mask_x]
        frequencies_filtered = frequencies[mask_y]
        spectrum_filtered = spectrum[np.ix_(mask_y, mask_x)]

        X, Y = np.meshgrid(time_filtered, frequencies_filtered)

        plt.clf()
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, spectrum_filtered, cmap='viridis')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_zlabel("Magnitude")
        ax.set_title("Interactive Wavelet Spectrum")
        ax.view_init(elev=elevation, azim=rotation)
        plt.show()

    elevation_slider = IntSlider(value=30, min=0, max=90, step=5, description="Elevation")
    rotation_slider = IntSlider(value=0, min=0, max=360, step=10, description="Rotation")
    min_time_slider = FloatSlider(value=min_time, min=min_time, max=max_time, step=(max_time - min_time) / 50, description="Min Time")
    max_time_slider = FloatSlider(value=max_time, min=min_time, max=max_time, step=(max_time - min_time) / 50, description="Max Time")
    min_frequency_slider = FloatSlider(value=min_frequency, min=min_frequency, max=max_frequency, step=(max_frequency - min_frequency) / 50, description="Min Frequency")
    max_frequency_slider = FloatSlider(value=max_frequency, min=min_frequency, max=max_frequency, step=(max_frequency - min_frequency) / 50, description="Max Frequency")

    from ipywidgets import interact
    interact(
        update_plot,
        elevation=elevation_slider,
        rotation=rotation_slider,
        min_time_val=min_time_slider,
        max_time_val=max_time_slider,
        min_frequency_val=min_frequency_slider,
        max_frequency_val=max_frequency_slider
    )

def spectrum(
    df: pd.DataFrame,
    label: str,
    wavelet: str = "morl",
    min_scale: float = 2.0,
    max_scale: float = 32.0,
    save_gif: bool = False,
    file_location: str = "results/wavelet_spectrum.gif",
    magnitude_type: str='calculated',
    magnitude_factor: float=1.0
) -> None:
    """
    Applies Continuous Wavelet Transform (CWT) to acceleration data and visualizes the spectrum.

    Args:
        magnitude_factor:
        magnitude_type:
        df: DataFrame with 'Time' and '{label} Acceleration' column.
        label: Name of the acceleration column to analyze '{label} Acceleration'.
        wavelet: Wavelet function to use.
        min_scale: Minimum scale for the wavelet transform.
        max_scale: Maximum scale for the wavelet transform.
        save_gif: If True, saves the 3D plot rotation as a GIF.
        file_location: Path to save the GIF file.

    Returns:
        None
    """
    try:
        spectrum, frequencies = calculate_cwt(df, label, wavelet, min_scale, max_scale, magnitude_type, magnitude_factor)

        time_min, time_max = df["Time"].min(), df["Time"].max()
        freq_min, freq_max = frequencies.min(), frequencies.max()

        # Call interactive plotting function directly with all required arguments
        plot_interactive_wavelet_spectrum(
            df["Time"].values,
            frequencies,
            spectrum,
            time_min,
            time_max,
            freq_min,
            freq_max
        )

        if save_gif:
            plot_spectrum_gif(
                df["Time"].values,
                frequencies,
                spectrum,
                file_location,
                time_min,
                time_max,
                freq_min,
                freq_max,
            )

    except Exception as e:
        warnings.warn(f"An error occurred during wavelet spectrum processing: {e}")


def plot_spectrum_views(
    df: pd.DataFrame,
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    label: str,
    elevation: int = 30,
    rotation: int = 30,
    label_size: int = 10,
    label_offset: float = 0.1,
    display_plot: bool = True,
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
        display_plot: Whether to display the plot (default: True).

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
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/ws_views_for_for_{label}.png", dpi=300)
    if display_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_spectrum_time_frequency(
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
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/ws_tf_for_{label}.png", dpi=300)
    plt.show()

def plot_spectrum_time_magnitude(
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
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/ws_tm_for_{label}.png", dpi=300)
    plt.show()


def plot_spectrum_frequency_magnitude(
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
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/ws_fm_for_{label}.png", dpi=300)
    plt.show()


def plotly_spectrum_views(
    df: pd.DataFrame,
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    label: str,
    elevation: int = 0,
    rotation: int = 0,
    label_size: int = 10,
    label_offset: float = 0.1,
) -> go.Figure:
    """
    Plots the time-frequency-magnitude wavelet spectrum in four subplots: XY, XZ, YZ, and 3D and saves the figure (Plotly version).

    Args:
        df: DataFrame containing the 'Time' column.
        spectrum: Magnitude spectrum of the CWT coefficients.
        frequencies: Frequencies corresponding to the scales.
        label: Direction of the acceleration data (e.g., 'X', 'Y', 'Z').
        elevation: Elevation angle for the 3D plot.
        rotation: Rotation angle for the 3D plot.
        label_size: Font size for labels.
        label_offset: Offset for labels.

    Returns:
        The Plotly figure.
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
        print(f"Shape mismatch: X{X.shape}, Y{Y.shape}, spectrum{spectrum_filtered.shape}")
        return

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "contour"}, {"type": "contour"}], [{"type": "contour"}, {"type": "surface"}]],
        subplot_titles=(
            f"{label} Wavelet Spectrum (Top View)",
            f"{label} Wavelet Spectrum (Side View 1)",
            f"{label} Wavelet Spectrum (Side View 2)",
            f"{label} Wavelet Spectrum (3D View)",
        ),
    )

    # XY subplot (Top View)
    fig.add_trace(
        go.Contour(x=time_filtered, y=frequencies_filtered, z=spectrum_filtered, colorscale="Viridis"), row=1, col=1
    )
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)

    # XZ subplot (Side View 1)
    fig.add_trace(
        go.Contour(x=time_filtered, y=spectrum_filtered.sum(axis=0), z=frequencies_filtered, colorscale="Viridis"),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Magnitude", row=1, col=2)

    # YZ subplot (Side View 2)
    fig.add_trace(
        go.Contour(x=spectrum_filtered.sum(axis=1), y=frequencies_filtered, z=time_filtered, colorscale="Viridis"),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Magnitude", row=2, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)

    # 3D subplot
    fig.add_trace(go.Surface(x=X, y=Y, z=spectrum_filtered, colorscale="Viridis"), row=2, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=2)
    fig.update_layout(scene=dict(zaxis_title="Magnitude"), scene_camera=dict(eye=dict(x=2, y=2, z=0.5)))

    fig.update_layout(title_text=f"{label} Wavelet Spectrum Views", template="plotly_white")

    os.makedirs("results", exist_ok=True)
    fig.write_html(f"results/ws_views_for_for_{label}.html")
    return fig


def plotly_spectrum_time_frequency(
    df: pd.DataFrame,
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    label: str,
    label_size: int = 10,
    label_offset: float = 0.1,
) -> go.Figure:
    """
    Plots the time-frequency wavelet spectrum (Top View) using Plotly and saves the figure.

    Args:
        df: DataFrame containing the 'Time' column.
        spectrum: Magnitude spectrum of the CWT coefficients.
        frequencies: Frequencies corresponding to the scales.
        label: Direction of the acceleration data (e.g., 'X', 'Y', 'Z').
        label_size: Font size for labels.
        label_offset: Offset for labels.

    Returns:
        The Plotly figure.
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
        print(f"Shape mismatch: X{X.shape}, Y{Y.shape}, spectrum{spectrum_filtered.shape}")
        return

    fig = go.Figure(
        data=go.Contour(x=time_filtered, y=frequencies_filtered, z=spectrum_filtered, colorscale="Viridis")
    )

    fig.update_layout(
        title=f"{label} Wavelet Spectrum (Top View)",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        template="plotly_white",
    )

    os.makedirs("results", exist_ok=True)
    fig.write_html(f"results/ws_tf_for_{label}.html")
    return fig


def plotly_spectrum_time_magnitude(
    df: pd.DataFrame,
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    label: str,
    label_size: int = 10,
    label_offset: float = 0.1,
) -> go.Figure:
    """
    Plots the time-magnitude wavelet spectrum (Side View 1) using Plotly and saves the figure.

    Args:
        df: DataFrame containing the 'Time' column.
        spectrum: Magnitude spectrum of the CWT coefficients.
        frequencies: Frequencies corresponding to the scales.
        label: Direction of the acceleration data (e.g., 'X', 'Y', 'Z').
        label_size: Font size for labels.
        label_offset: Offset for labels.

    Returns:
        The Plotly figure.
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
        print(f"Shape mismatch: X{X.shape}, Y{Y.shape}, spectrum{spectrum_filtered.shape}")
        return

    fig = go.Figure(
        data=go.Contour(
            x=time_filtered,
            y=spectrum_filtered.sum(axis=0),  # Sum over frequency to get magnitude
            z=frequencies_filtered,
            colorscale="Viridis",
        )
    )

    fig.update_layout(
        title=f"{label} Wavelet Spectrum (Side View 1)",
        xaxis_title="Time (s)",
        yaxis_title="Magnitude",
        template="plotly_white",
    )

    os.makedirs("results", exist_ok=True)
    fig.write_html(f"results/ws_tm_for_{label}.html")
    return fig


def plotly_wavelet_spectrum_frequency_magnitude(
    df: pd.DataFrame,
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    label: str,
    label_size: int = 10,
    label_offset: float = 0.1,
) -> go.Figure:
    """
    Plots the frequency-magnitude wavelet spectrum (Side View 2) using Plotly.

    Args:
        df: DataFrame containing the 'Time' column.
        spectrum: Magnitude spectrum of the CWT coefficients.
        frequencies: Frequencies corresponding to the scales.
        label: Direction of the acceleration data (e.g., 'X', 'Y', 'Z').
        label_size: Font size for labels.
        label_offset: Offset for labels.

    Returns:
        The Plotly figure.
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
        print(f"Shape mismatch: X{X.shape}, Y{Y.shape}, spectrum{spectrum_filtered.shape}")
        return

    fig = go.Figure(
        data=go.Contour(
            x=spectrum_filtered.sum(axis=1),  # Summing to get magnitude
            y=frequencies_filtered,
            z=time_filtered,
            colorscale="Viridis",
        )
    )

    fig.update_layout(
        title=f"{label} Wavelet Spectrum (Side View 2)",
        xaxis_title="Magnitude",
        yaxis_title="Frequency (Hz)",
        template="plotly_white",
    )

    os.makedirs("results", exist_ok=True)
    fig.write_html(f"results/ws_fm_for_{label}.html")
    return fig

