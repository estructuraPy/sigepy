import warnings
import numpy as np
import pandas as pd
import pywt
from scipy.signal import detrend, butter, filtfilt
from scipy.linalg import hankel, svd, eig
from numpy.typing import NDArray
from . import plot
from . import utils
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
from signalepy import plot, utils
from collections import OrderedDict
from typing import Tuple,Dict, Union, List
from dataclasses import dataclass, field
from tqdm import tqdm
import gc
from plotly import graph_objs as go


def calculate_fft(df: pd.DataFrame, labels: List[str], magnitude_type: str='calculated', magnitude_factor: float=1.0) -> pd.DataFrame:
    """
    Calculates the FFT of multiple acceleration data columns and returns a DataFrame with the results.

    Args:
        df: DataFrame containing 'Time' and '{label} Acceleration' columns.
        labels: List of labels of the vibration (e.g., ['X', 'Y', 'Z']).
        magnitude_type: 'calculated' or 'normalized'
        magnitude_factor: float number to be used as reference to normalize the fft spectrum. Only works
            with magnitude_type = 'normalized'

    Returns:
        DataFrame containing the FFT results with 'Frequency' and '{label} Magnitude' columns for each label.

    Assumptions:
        - 'df' contains 'Time' and '{label} Acceleration' columns for each label.
        - Time is uniformly sampled.
    """
    result = pd.DataFrame()
    n = len(df["Time"])
    time_step = df["Time"].iloc[1] - df["Time"].iloc[0]
    sampling_rate = 1 / time_step
    frequencies = np.fft.fftfreq(n, 1 / sampling_rate)

    result["Frequency"] = frequencies[: n // 2]

    for label in labels:
        accelerations = np.fft.fft(df[f"{label} Acceleration"])
        magnitudes = np.abs(accelerations)

        if magnitude_type == 'normalized':
            modified_magnitudes = magnitudes / np.max(magnitudes) * magnitude_factor
        elif magnitude_type == 'calculated':
            modified_magnitudes = magnitudes
        else:
            raise ValueError("'normalized' or 'calculated'")

        result[f"{label} Magnitude"] = modified_magnitudes[: n // 2]

    return result


def filter_with_fft(
    df: pd.DataFrame,
    labels: List[str],
    threshold: float,
) -> pd.DataFrame:
    """
    Filters FFT output by zeroing frequencies with magnitudes below a threshold and reconstructs the filtered signal for multiple labels.

    Args:
        df: DataFrame containing 'Time' and '{label} Acceleration' columns.
        labels: List of labels of the acceleration data to filter (e.g., ['X', 'Y', 'Z']).
        threshold: Percentage of the maximum magnitude below which frequencies are filtered out (0-100).

    Returns:
        A Pandas DataFrame with new columns containing the filtered acceleration data for each label.

    Assumptions:
        - 'df' contains 'Time' and '{label} Acceleration' columns for each label.
        - Time data is uniformly sampled.
        - 'threshold' is a float between 0 and 100.
    """
    if not 0 <= threshold <= 100:
        raise ValueError("threshold must be between 0 and 100.")

    for label in labels:
        # Clarity: Apply FFT to the acceleration data for the current label.
        fft_data = fft(df[f"{label} Acceleration"])
        magnitude = np.abs(fft_data)

        # Clarity: Calculate the threshold value based on the maximum magnitude and the specified threshold percentage.
        threshold_value = np.max(magnitude) * threshold / 100

        # Clarity: Create a copy of the FFT data to apply the filter.
        fft_filtered = fft_data.copy()
        fft_filtered[magnitude < threshold_value] = 0

        # Clarity: Apply inverse FFT to reconstruct the filtered acceleration data.
        df[f"{label} Filtered Acceleration"] = ifft(fft_filtered).real

    return df


def plot_normalized_fft_results(df: pd.DataFrame, label: str, color: str = "red") -> None:
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


def plot_fft_results(df: pd.DataFrame, label: str, color: str = "red") -> None:
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


def plot_fft_results_period_domain(df: pd.DataFrame, label: str, color: str = "red", log_scale: bool = False) -> None:
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


def plot_peaks(
    df: pd.DataFrame,
    label: str,
    height: float=1,
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

def plotly_normalized_fft_results(df: pd.DataFrame, label: str, color: str = "red") -> go.Figure:
    """
    Plots the normalized FFT magnitude spectrum from a DataFrame using Plotly.

    Args:
        df: DataFrame containing 'Frequency' and '{label} Magnitude' columns.
        label: The label of the vibration (e.g., 'X', 'Y', 'Z').
        color: Color of the plot line.

    Returns:
        The Plotly figure.
    """
    frequencies = df["Frequency"]
    magnitudes = df[f"{label} Magnitude"]
    normalized_magnitudes = magnitudes / np.max(magnitudes)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=normalized_magnitudes,
            mode="lines",
            name=label,
            line=dict(color=color),
        )
    )

    fig.update_layout(
        title="FFT Normalized Magnitude Spectrum",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        template="plotly_white",
    )

    fig.write_html(f"results/normalized_fft_for_{label}.html")
    return fig


def plotly_fft_results(df: pd.DataFrame, label: str, color: str = "red") -> go.Figure:
    """
    Plots the FFT magnitude spectrum from a DataFrame using Plotly and saves the figure.

    Args:
        df: DataFrame containing 'Frequency' and '{label} Magnitude' columns.
        label: The label of the vibration (e.g., 'X', 'Y', 'Z').
        color: Color of the plot line.

    Returns:
        The Plotly figure.
    """
    frequencies = df["Frequency"]
    magnitudes = df[f"{label} Magnitude"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=magnitudes,
            mode="lines",
            name=label,
            line=dict(color=color),
        )
    )

    fig.update_layout(
        title="FFT Magnitude Spectrum",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        template="plotly_white",
    )

    fig.write_html(f"results/fft_for_{label}.html")
    return fig


def plotly_fft_results_period_domain(
    df: pd.DataFrame, label: str, color: str = "red", log_scale: bool = False
) -> go.Figure:
    """
    Plots the FFT magnitude spectrum in the period domain from a DataFrame using Plotly and saves the figure.

    Args:
        df: DataFrame containing 'Frequency' and '{label} Magnitude' columns.
        label: The label of the vibration (e.g., 'X', 'Y', 'Z').
        color: Color of the plot line.
        log_scale: Whether to use a logarithmic scale for the period axis.

    Returns:
        The Plotly figure.
    """
    frequencies = df["Frequency"]
    magnitudes = df[f"{label} Magnitude"]

    periods = 1 / frequencies

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=magnitudes,
            mode="lines",
            name=label,
            line=dict(color=color),
        )
    )

    fig.update_layout(
        title="FFT Period Domain Spectrum",
        xaxis_title="T (s)",
        yaxis_title="Magnitude",
        xaxis_type="log" if log_scale else "linear",
        template="plotly_white",
    )

    fig.write_html(f"results/fft_period_domain {label}.html")
    return fig


def plotly_peaks(
    df: pd.DataFrame,
    label: str,
    height: float = 1,
    distance: float = 1,
    log_scale: bool = False,
    file_location: str = "results/fft_peaks.html",
) -> go.Figure:
    """
    Finds and plots peaks in the FFT magnitude spectrum using Plotly.

    Args:
        df: DataFrame containing 'Frequency' and '{label} Magnitude' columns.
        label: The label of the vibration ('X', 'Y' or 'Z').
        height: Required height of peaks.
        distance: Required minimal horizontal distance (in samples) between neighboring peaks.
        log_scale: Whether to use a logarithmic scale for the frequency axis.
        file_location: Path where the plot will be saved.

    Returns:
        The Plotly figure.
    """
    frequencies = df["Frequency"]
    magnitudes = df[f"{label} Magnitude"]

    peaks_scipy, _ = signal.find_peaks(magnitudes, height=height, distance=distance)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frequencies, y=magnitudes, mode="lines", name=label))
    fig.add_trace(
        go.Scatter(
            x=frequencies[peaks_scipy],
            y=magnitudes[peaks_scipy],
            mode="markers+text",
            marker=dict(size=8),
            text=[f"{freq:.2f} Hz" for freq in frequencies[peaks_scipy]],  # Etiquetas en los picos
            textposition="top center",
            name="Peaks"
        )
    )

    fig.update_layout(
        title=f"{label} and peaks found",
        xaxis_title="Frequency [Hz]",
        yaxis_title="Magnitude",
        xaxis_type="log" if log_scale else "linear",
        template="plotly_white",
    )

    fig.write_html(file_location)
    return fig