from io import BytesIO

import ipywidgets as widgets
import numpy as np
import pandas as pd
import pywt
from PIL import Image
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import detrend, butter, filtfilt


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


def calculate_fft(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Calculates the FFT of a single acceleration data column and returns a DataFrame with the results.

    Args:
        df: DataFrame containing 'Time' and '{label} Acceleration' columns.
        label: The label of the vibration (e.g., 'X', 'Y', 'Z').

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

    return pd.DataFrame(
        {
            "Frequency": frequencies[: n // 2],
            f"{label} Magnitude": magnitudes[: n // 2],
        }
    )


def cwt(df: pd.DataFrame, label: str, wavelet: str = "morl",
        min_scale: int = 2, max_scale: int = 32,
):
    """
    Performs Continuous Wavelet Transform (CWT) analysis on acceleration data.

    Args:
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

    return spectrum, frequencies


def wavelet_spectrum(
        df: pd.DataFrame,
        label: str,
        wavelet: str = "morl",
        min_scale: float = 2.0,
        max_scale: float = 32.0,
        save_gif: bool = False,
        file_location: str = "results/wavelet_spectrum.gif"
):
    """
    Applies Continuous Wavelet Transform (CWT) to filtered acceleration data,
    calculates the average spectrum, and visualizes the time-frequency-magnitude spectrum.

    Args:
        file_location:
        df: DataFrame with 'Time' and at least '{label} Acceleration' column.
        label: Name of the acceleration column to analyze '{label} Acceleration'.
        wavelet: Wavelet function to use.
        min_scale: Minimum scale for the wavelet transform.
        max_scale: Maximum scale for the wavelet transform.
        save_gif: If True, saves the 3D plot rotation as a GIF.
    """

    spectrum, frequencies = cwt(df, label, wavelet, min_scale, max_scale)
    time_step = df["Time"].iloc[1] - df["Time"].iloc[0]
    sampling_rate = 1 / time_step
    nyquist_freq = sampling_rate / 2

    time_min, time_max = df["Time"].min(), df["Time"].max()
    freq_min, freq_max = frequencies.min(), nyquist_freq

    def plot_spectrum(
            min_time=time_min,
            max_time=time_max,
            min_frequency=freq_min,
            max_frequency=freq_max,
            label_size=None,
            elevation=None,
            rotation=None,
    ):
        """
        Plots the 3D wavelet spectrum.

        Args:
            min_time: Minimum time value for the plot.
            max_time: Maximum time value for the plot.
            min_frequency: Minimum frequency value for the plot.
            max_frequency: Maximum frequency value for the plot.
            label_size: Font size for the axis labels.
            elevation: Elevation angle for the 3D plot.
            rotation: Azimuthal (horizontal) rotation angle for the 3D plot.
        """
        mask_x = (df["Time"] >= min_time) & (df["Time"] <= max_time)
        mask_y = (frequencies >= min_frequency) & (frequencies <= max_frequency)

        time_filtered = df["Time"][mask_x].values
        frequencies_filtered = frequencies[mask_y]
        spectrum_filtered = spectrum[np.ix_(mask_y, mask_x)]

        X, Y = np.meshgrid(time_filtered, frequencies_filtered)

        if X.shape != spectrum_filtered.shape:
            print(
                f"Shape mismatch: X{X.shape}, Y{Y.shape}, Z{spectrum_filtered.shape}"
            )
            return

        figx = plt.figure(figsize=(12, 12))
        ax = figx.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, spectrum_filtered, cmap="viridis")

        ax.set_xlabel("Time (s)", fontsize=label_size)
        ax.set_ylabel("Frequency (Hz)", fontsize=label_size)
        ax.set_zlabel("Magnitude", fontsize=label_size)
        ax.set_title("Time - Frequency - Magnitude Wavelet Spectrum", fontsize=label_size)
        ax.view_init(elev=elevation, azim=rotation)
        return figx

    if save_gif:
        frames = []
        for angle in range(0, 360, 10):
            fig = plot_spectrum(
                elevation=30, rotation=angle
            )  # Fixed elevation, varying rotation

            if fig is None:
                continue

            try:
                buf = BytesIO()
                fig.canvas.draw()
                fig.savefig(buf, format='png')
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

    else:
        widgets.interact(
            plot_spectrum,
            elevation=(0, 360, 10),
            rotation=(0, 360, 10),
            label_size=(6, 32, 2),
            min_time=(time_min, time_max, (time_max - time_min) / 100),
            max_time=(time_min, time_max, (time_max - time_min) / 100),
            min_frequency=(freq_min, freq_max, (freq_max - freq_min) / 50),
            max_frequency=(freq_min, freq_max, (freq_max - freq_min) / 50),
        )