import warnings

import numpy as np
import pandas as pd
import pywt
from scipy.fft import fft, ifft
from scipy.signal import detrend, butter, filtfilt
from scipy.linalg import hankel, svd, eig

from . import plot


def generate_vibration_signal_dataframe(
        total_time: float,
        sampling_rate: float,
        frequency_inputs: list,
        amplitude_inputs: list,
        noise_amplitude: float,
        label: str,
) -> pd.DataFrame:
    """
    Generates a synthetic vibration signal with time-varying or constant frequencies and returns it as a DataFrame.

    Args:
        total_time: Total duration of the signal (in seconds).
        sampling_rate: Sampling rate (in Hz).
        frequency_inputs: List of functions or constant values describing how frequencies vary over time.
        amplitude_inputs: List of functions or constant values describing how amplitudes vary over time.
        noise_amplitude: Amplitude of random noise added to the signal.
        label: Direction of the vibration signal.

    Returns:
        DataFrame with columns 'Time' and '{label} Acceleration'.
    """
    t = np.linspace(0, total_time, int(total_time * sampling_rate), endpoint=False)

    acceleration = np.zeros_like(t)

    for freq_input, amp_input in zip(frequency_inputs, amplitude_inputs):
        if callable(freq_input):
            instantaneous_frequencies = freq_input(t)
        else:
            instantaneous_frequencies = np.full_like(t, freq_input)

        if callable(amp_input):
            instantaneous_amplitudes = amp_input(t)
        else:
            instantaneous_amplitudes = np.full_like(t, amp_input)

        acceleration += instantaneous_amplitudes * np.sin(
            2 * np.pi * instantaneous_frequencies * t
        )

    acceleration += noise_amplitude * np.random.normal(size=len(t))

    return pd.DataFrame({"Time": t, f"{label} Acceleration": acceleration})


def baseline_correction(
    df: pd.DataFrame, label: str, label_corrected: str
) -> pd.DataFrame:
    """
    Preprocesses the data by removing outliers and detrending the acceleration signal.

    Args:
        df: DataFrame containing 'Time' and '{label} Acceleration' columns.
        label: The label of the vibration (e.g., 'X', 'Y', 'Z').
        label_corrected: Name of the new column

    Returns:
        DataFrame with outliers removed and the acceleration signal detrended.

    Assumptions:
        - 'df' contains 'Time' and '{label} Acceleration' columns.
    """
    # Remove outliers (values beyond 3 standard deviations)
    mean = df[f"{label} Acceleration"].mean()
    std = df[f"{label} Acceleration"].std()
    df[f"{label} Acceleration"] = np.where(
        np.abs(df[f"{label} Acceleration"] - mean) <= 3 * std,
        df[f"{label} Acceleration"],
        mean,
    )

    df["Time"] = df["Time"]
    # Detrend the data
    df[f"{label_corrected} Acceleration"] = detrend(df[f"{label} Acceleration"])

    return df


def fft_filter(
    df: pd.DataFrame,
    label: str,
    label_filtered: str,
    threshold_percentage: float,
) -> pd.DataFrame:
    """
    Filters FFT output by zeroing frequencies with magnitudes below a threshold and reconstructs the filtered signal.

    Args:
        df: DataFrame containing 'Time' and '{label} Acceleration' columns.
        label: The label of the acceleration data to filter (e.g., 'X', 'Y', 'Z').
        label_filtered: The label for the filtered acceleration data column.
        threshold_percentage: Percentage of the maximum magnitude below which frequencies are filtered out (0-100).

    Returns:
        A Pandas DataFrame with a new column containing the filtered acceleration data.

    Assumptions:
        - 'df' contains 'Time' and '{label} Acceleration' columns.
        - Time data is uniformly sampled.
        - 'threshold_percentage' is a float between 0 and 100.
    """
    if not 0 <= threshold_percentage <= 100:
        raise ValueError("threshold_percentage must be between 0 and 100.")

    fft_data = fft(df[f"{label} Acceleration"])
    magnitude = np.abs(fft_data)
    threshold = np.max(magnitude) * threshold_percentage / 100

    fft_filtered = fft_data.copy()
    fft_filtered[magnitude < threshold] = 0

    df[f"{label_filtered} Acceleration"] = ifft(fft_filtered).real

    return df


def denoising_with_fft(
    df: pd.DataFrame, threshold_percentage: float, label: str, label_filtered: str
) -> pd.DataFrame:
    """
    Applies FFT filtering to a specified acceleration data column in a DataFrame.

    Args:
        df: DataFrame containing 'Time' and '{label} Acceleration' columns.
        threshold_percentage: Percentage of the maximum magnitude below which frequencies are filtered out (0-100).
        label: The label of the acceleration data to filter (e.g., 'X', 'Y', 'Z').
        label_filtered: The label for the filtered acceleration data column.

    Returns:
        A Pandas DataFrame with the filtered acceleration data in a new column.

    Assumptions:
        - 'df' contains 'Time' and '{label} Acceleration' columns.
        - The `filter_with_fft` function is correctly implemented and available.
    """

    df[f"{label_filtered} Acceleration"] = fft_filter(
        df, label, label_filtered, threshold_percentage
    )[f"{label_filtered} Acceleration"]

    return df


def butter_bandpass_filter(
    df: pd.DataFrame,
    lowcut: int = 3,
    highcut: int = 40,
    order: int = 4,
    label: str = "X",
) -> pd.Series:
    """
    Applies a Butterworth bandpass filter to the specified acceleration data.

    Args:
        df: DataFrame containing 'Time' and '{label} Acceleration' columns.
        lowcut: Lower cutoff frequency in Hz (default: 3).
        highcut: Upper cutoff frequency in Hz (default: 40).
        order: Order of the Butterworth filter (default: 4).
        label: Direction of the acceleration data to filter (default: 'X').

    Returns:
        A Pandas Series containing the filtered acceleration data.

    Assumptions:
        - 'df' contains 'Time' and '{label} Acceleration' columns.
        - Time data is uniformly sampled.
        - The sampling rate is determined from the time step in the 'Time' column.
        - Cutoff frequencies are given in Hz.
    """
    time_step = df["Time"].iloc[1] - df["Time"].iloc[0]
    sampling_rate = 1 / time_step

    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")

    return pd.Series(filtfilt(b, a, df[f"{label} Acceleration"]))


def calculate_fft(df: pd.DataFrame, label: str, magnitude_type: str='calculated', magnitude_factor: float=1.0) -> pd.DataFrame:
    """
    Calculates the FFT of a single acceleration data column and returns a DataFrame with the results.

    Args:
        df: DataFrame containing 'Time' and '{label} Acceleration' columns.
        label: The label of the vibration (e.g., 'X', 'Y', 'Z').
        magnitude_type: 'calculated' or 'normalized'
        magnitude_factor: float number to be use as reference to normalized the fft spectrum. Only works
            with magnitude_type = 'normalized'

    Returns:
        DataFrame containing the FFT results with 'Frequency' and '{label} Magnitude' columns.

    Assumptions:
        - 'df' contains 'Time' and '{label} Acceleration' columns.
        - Time is uniformly sampled.
    """

    n = len(df[f"{label} Acceleration"])
    time_step = df["Time"].iloc[1] - df["Time"].iloc[0]
    sampling_rate = 1 / time_step

    accelerations = np.fft.fft(df[f"{label} Acceleration"])
    frequencies = np.fft.fftfreq(n, 1 / sampling_rate)

    magnitudes = np.abs(accelerations)

    if magnitude_type == 'normalized':
        modified_magnitudes = magnitudes / np.max(magnitudes) * magnitude_factor
    elif magnitude_type == 'calculated':
        modified_magnitudes = magnitudes
    else:
        raise ValueError("'normalized' or 'calculated'")

    return pd.DataFrame(
        {
            "Frequency": frequencies[: n // 2],
            f"{label} Magnitude": modified_magnitudes[: n // 2],
        }
    )


def cwt(df: pd.DataFrame, label: str, wavelet: str = "morl",
        min_scale: int = 2, max_scale: int = 32, magnitude_type: str='calculated',
        magnitude_factor: float=1.0) -> pd.DataFrame:
    """
    Performs Continuous Wavelet Transform (CWT) analysis on acceleration data.

    Args:
        magnitude_factor:
        magnitude_type:
        df: DataFrame containing 'Time' and '{label} Acceleration' columns.
        label: Direction of the acceleration data (e.g., 'X', 'Y', 'Z').
        wavelet: Wavelet function to use (default: 'cmor1.5-1.0').
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
        df[f"{label} Acceleration"], scales, wavelet, time_step
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


def wavelet_spectrum(
    df: pd.DataFrame,
    label: str,
    wavelet: str = "morl",
    min_scale: float = 2.0,
    max_scale: float = 32.0,
    save_gif: bool = False,
    file_location: str = "results/wavelet_spectrum.gif",
    magnitude_type: str='calculated',
    magnitude_factor: float=1.0):
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
        spectrum, frequencies = cwt(df, label, wavelet, min_scale, max_scale, magnitude_type, magnitude_factor)

        time_min, time_max = df["Time"].min(), df["Time"].max()
        freq_min, freq_max = frequencies.min(), frequencies.max()

        # Call interactive plotting function
        interactive_plot = plot.interactive_wavelet_spectrum(
            df["Time"].values,
            frequencies,
            spectrum,
            time_min,
            time_max,
            freq_min,
            freq_max,
        )

        # Display the interactive plot if it was successfully created
        if interactive_plot is not None:
            from IPython.display import display
            display(interactive_plot)
        else:
            warnings.warn("Interactive plot could not be created.")

        if save_gif:
            plot.wavelet_spectrum_gif(
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


def ssi_cov(
    df: pd.DataFrame,
    label: str,
    Ts: int = 100,
    Nmin: int = 2,
    Nmax: int = 20,
    threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs Stochastic Subspace Identification (SSI) analysis on acceleration data.

    Args:
        df: DataFrame containing the acceleration data.
        label: Column name of the acceleration data.
        Ts: Number of rows for the Hankel matrix.
        Nmin: Minimum number of modes to consider. (Unused in the current implementation)
        Nmax: Maximum number of modes to consider.
        threshold: Frequency threshold for pole counting.

    Returns:
        A tuple containing:
            frequencies: Frequency axis for the power spectrum.
            power_spectrum: Power spectrum of the acceleration data.
            fn0: Natural frequencies of the identified modes.
            zeta0: Damping ratios of the identified modes.
            number_of_poles: Number of poles within the frequency threshold.

    Assumptions:
        - Acceleration data is uniformly sampled.
    """
    # Extract acceleration data
    acceleration = df[f'{label} Acceleration'].values

    # Construct Hankel matrix
    H = hankel(acceleration[:Ts], acceleration[-Ts:])

    # Perform Singular Value Decomposition (SVD)
    U, S, V = svd(H, full_matrices=False)

    # Determine number of modes
    num_modes = min(Nmax, len(S))
    U = U[:, :num_modes]
    S = np.diag(S[:num_modes])
    V = V[:num_modes, :]

    # Calculate state-space matrices
    A = V @ np.diag(1 / np.diag(S)) @ U.T @ H
    C = U[:, :num_modes]

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(A)

    # Calculate natural frequencies and damping ratios
    fn0 = np.abs(np.arctan2(eigenvalues.imag, eigenvalues.real)) / (2 * np.pi)
    zeta0 = -eigenvalues.real / np.abs(eigenvalues)

    # Filter valid modes
    valid_modes = (fn0 > 0) & (zeta0 < 1)
    fn0 = fn0[valid_modes]
    zeta0 = zeta0[valid_modes]

    # Calculate power spectrum
    power_spectrum = np.abs(np.fft.fft(acceleration)) ** 2
    frequencies = np.fft.fftfreq(len(acceleration))

    # Ensure frequencies and power_spectrum are positive
    positive_freqs = frequencies > 0
    frequencies = frequencies[positive_freqs]
    power_spectrum = power_spectrum[positive_freqs]

    # Count number of poles within frequency range
    number_of_poles = np.zeros_like(frequencies)
    for i, freq in enumerate(frequencies):
        number_of_poles[i] = np.sum((fn0 >= freq - threshold) & (fn0 <= freq + threshold))

    return frequencies, power_spectrum, fn0, zeta0, number_of_poles