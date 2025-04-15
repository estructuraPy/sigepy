"""
Utility module providing support functions for signal processing operations.

This module contains various utility functions used across the sigepy package, including:
- File handling and path management
- Data import from various file formats
- Signal generation for testing and simulation
- Signal preprocessing and filtering operations
- Data visualization utilities
- Acceleration data processing

Key Classes:
    SignalProcessor: Class for processing time-series data with filtering and corrections

Key Functions:
    import_sts_acceleration_txt: Imports acceleration data from TXT files
    import_csv_acceleration: Imports acceleration data from CSV files
    import_cscr_fed: Imports data from TXT files with specific format
    generate_vibration_signal: Generates synthetic vibration signals
    generate_vibration_signals: Generates multiple synthetic vibration signals
"""

import pathlib
import pandas as pd
import numpy as np
import json
import os
from scipy.signal import detrend, butter, filtfilt
from typing import List, Dict, Union
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import plotly.graph_objs as go


def estimate_power_of_two(df: pd.DataFrame) -> int:
    """
    Estimates the smallest integer n such that delta_t * 2**n >= time_n.

    Args:
        df: DataFrame containing the 'Time' column.

    Returns:
        The estimated integer n.

    Assumptions:
        - df is not empty.
        - df contains a 'Time' column.
    """
    if df.empty:
        raise ValueError("The DataFrame is empty.")

    if "Time" not in df.columns:
        raise KeyError("The 'Time' column does not exist in the DataFrame.")

    delta_t = df["Time"].iloc[1] - df["Time"].iloc[0]
    time_n = df["Time"].iloc[-1]

    n = 0
    while delta_t * 2**n < time_n:
        n += 1

    return n


def get_tests_files_location(file_name: str) -> Path:
    """
    Gets the file location in the current directory.

    Args:
        file_name: Name of the file.

    Returns:
        The file location as a Path object.
    """
    current_directory = pathlib.Path().absolute()
    file_location = current_directory / file_name
    return file_location


def get_results_files_location(file_name: str) -> Path:
    """
    Gets the file location in the results directory.

    Args:
        file_name: Name of the file.

    Returns:
        The file location as a Path object.
    """
    notebook_location = pathlib.Path().absolute()
    parent_directory = notebook_location.parent
    data_folder = parent_directory / "tutorials/results"
    file_location = data_folder / file_name
    return file_location


def get_data_files_location(file_name: str) -> Path:
    """
    Gets the file location in the data directory.

    Args:
        file_name: Name of the file.

    Returns:
        The file location as a Path object.
    """
    source_code_location = pathlib.Path(__file__).resolve().parent
    data_folder = source_code_location / "data"
    file_location = data_folder / file_name
    return file_location


def import_sts_acceleration_txt(file_location: str, labels: List[str]) -> pd.DataFrame:
    """
    Processes acceleration data from a txt file.

    Args:
        file_location: Path to the file.
        labels: List of labels (e.g., ['X', 'Y', 'Z']).

    Returns:
        DataFrame containing the time and acceleration data.

    Assumptions:
        - File exists and is readable.
        - File encoding is 'latin1'.
        - File contains at least two lines, with the second line being a header.
        - Each data line has at least 5 fields, with the 2nd, 3rd, 4th and 5th fields representing time, x, y, and z accelerations respectively.
    """
    try:
        with open(file_location, "r", encoding="latin1") as file:
            lines = [line.strip() for line in file.readlines()[2:]]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_location}") from e
    except Exception as e:
        raise IOError(f"Error reading file: {file_location}") from e

    times = []
    accelerations = {label: [] for label in labels}

    for line in lines:
        fields = line.split()
        if len(fields) >= 5:
            try:
                time = float(fields[1])
                times.append(time)
                if "X" in labels:
                    accelerations["X"].append(float(fields[2]) * 9.81)
                if "Y" in labels:
                    accelerations["Y"].append(float(fields[3]) * 9.81)
                if "Z" in labels:
                    accelerations["Z"].append(float(fields[4]) * 9.81)
            except ValueError as e:
                print(f"Skipping line due to invalid data: {line}. Error: {e}")

    data = {"Time": np.array(times)}
    for label in labels:
        data[f"{label} Acceleration"] = np.array(accelerations[label])

    return pd.DataFrame(data)


def import_csv_acceleration(file_location: str, labels: List[str] = None) -> pd.DataFrame:
    """
    Processes acceleration data from a CSV file containing multiple accelerometer readings.

    Args:
        file_location: Path to the CSV file containing the acceleration data.
        labels: List of labels to return as series ['X', 'Y', 'Z']. If None, returns all available axes.

    Returns:
        A Pandas DataFrame containing the time and acceleration data averaged across all accelerometers.

    Assumptions:
        - The file exists and is readable
        - CSV headers follow the pattern: Timestamp, Expected_Time, Accel{n}_X, Accel{n}_Y, Accel{n}_Z, Accel{n}_Magnitude
        - Where n is the accelerometer number (1, 2, etc.)
    """
    try:
        df = pd.read_csv(file_location)

        result_df = pd.DataFrame({"Time": df["Expected_Time"]})

        axes = labels if labels is not None else ["X", "Y", "Z"]
        for axis in axes:
            axis_columns = [col for col in df.columns if f"_{axis}" in col and "Magnitude" not in col]

            if axis_columns:
                avg_acceleration = df[axis_columns].mean(axis=1)
                result_df[f"{axis} Acceleration"] = avg_acceleration * 9.81

        return result_df

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_location}") from e
    except pd.errors.EmptyDataError as e:
        raise ValueError("The CSV file is empty") from e
    except Exception as e:
        raise IOError(f"Error processing file {file_location}: {str(e)}") from e


def import_cscr_fed(file_location: str, json_location: str) -> pd.DataFrame:
    """
    Imports data from a TXT file and returns a Pandas DataFrame.

    Args:
        file_location: Path to the TXT file.
        json_location: Path to the JSON file containing default values.

    Returns:
        A Pandas DataFrame with 'Frequency' and 'FED' columns, or None if an error occurs.

    Assumptions:
        - The TXT file is tab-separated and has two columns: 'T' and 'FED'.
        - The JSON file exists and is accessible.
    """
    try:
        file_location = os.path.abspath(file_location)
        json_location = os.path.abspath(json_location)

        with open(json_location, "r", encoding="utf-8") as f:
            default_values = json.load(f)

        try:
            df = pd.read_csv(file_location, sep="\t", header=None, names=["T", "FED"], encoding="utf-16")
        except UnicodeDecodeError:
            df = pd.read_csv(file_location, sep="\t", header=None, names=["T", "FED"], encoding="utf-8")

        df["T"] = pd.to_numeric(df["T"], errors="coerce")
        df = df.dropna(subset=["T"])
        df["Frequency"] = 1 / df["T"]

        return df[["Frequency", "FED"]]

    except FileNotFoundError as e:
        print(f"Error: {e}. Check that the JSON file and the TXT file are in the specified path.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def generate_vibration_signal(
    label: str,
    total_time: float,
    sampling_rate: float,
    frequency_inputs: list,
    amplitude_inputs: list,
    noise_amplitude: float,
) -> pd.DataFrame:
    """
    Generates a synthetic vibration signal.

    Args:
        label: Direction of the vibration signal.
        total_time: Total duration of the signal (in seconds).
        sampling_rate: Sampling rate (in Hz).
        frequency_inputs: List of functions or constant values describing how frequencies vary over time.
        amplitude_inputs: List of functions or constant values describing how amplitudes vary over time.
        noise_amplitude: Amplitude of random noise added to the signal.

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
        acceleration += instantaneous_amplitudes * np.sin(2 * np.pi * instantaneous_frequencies * t)
    acceleration += noise_amplitude * np.random.normal(size=len(t))

    return pd.DataFrame({"Time": t, f"{label} Acceleration": acceleration})


def generate_vibration_signals(
    labels: List[str],
    total_time: float,
    sampling_rate: float,
    frequency_inputs: Dict[str, List[Union[float, callable]]],
    amplitude_inputs: Dict[str, List[Union[float, callable]]],
    noise_amplitudes: Dict[str, float],
) -> pd.DataFrame:
    """
    Generates synthetic vibration signals for multiple directions.

    Args:
        labels: List of directions of the vibration signals.
        total_time: Total duration of the signal (in seconds).
        sampling_rate: Sampling rate (in Hz).
        frequency_inputs: Dictionary of lists of functions or constant values describing how frequencies vary over time for each label.
        amplitude_inputs: Dictionary of lists of functions or constant values describing how amplitudes vary over time for each label.
        noise_amplitudes: Dictionary of noise amplitudes for each label.

    Returns:
        DataFrame with columns 'Time' and '{label} Acceleration' for each label.
    """
    t = np.linspace(0, total_time, int(total_time * sampling_rate), endpoint=False)
    df = {"Time": t}

    for label in labels:
        acceleration = np.zeros_like(t)
        for freq_input, amp_input in zip(frequency_inputs[label], amplitude_inputs[label]):
            if callable(freq_input):
                instantaneous_frequencies = freq_input(t)
            else:
                instantaneous_frequencies = np.full_like(t, freq_input)
            if callable(amp_input):
                instantaneous_amplitudes = amp_input(t)
            else:
                instantaneous_amplitudes = np.full_like(t, amp_input)
            acceleration += instantaneous_amplitudes * np.sin(2 * np.pi * instantaneous_frequencies * t)
        acceleration += noise_amplitudes[label] * np.random.normal(size=len(t))
        df[f"{label} Acceleration"] = acceleration

    return pd.DataFrame(df)


@dataclass
class SignalProcessor:
    """
    A class for processing time-series data, including filtering and baseline correction.

    Args:
        df: DataFrame containing time-series data.
        labels: List of acceleration labels (e.g., ['X', 'Y', 'Z']).
        lowcut: Lower cutoff frequency for bandpass filter (default is 3 Hz).
        highcut: Higher cutoff frequency for bandpass filter (default is 40 Hz).
        order: Order of the Butterworth filter (default is 2).
        start_time: Start time for filtering a time window (default is 0).
        power_of_two: Exponent to calculate the time window size.

    Returns:
        None

    Assumptions:
        - df contains 'Time' and '{label} Acceleration' columns for each label.
        - Sampling rate is uniform.
    """

    df: pd.DataFrame
    labels: List[str]
    lowcut: int = 3
    highcut: int = 40
    order: int = 2
    start_time: float = 0
    power_of_two: float = 1

    def __post_init__(self):
        """
        Post-initialization to perform a deep copy of the DataFrame.
        """
        try:
            self.df = self.df.copy()
        except Exception as e:
            raise ValueError(f"Error copying DataFrame: {e}")

        time_step = self.df["Time"].iloc[1] - self.df["Time"].iloc[0]
        self.window_size = time_step * 2**self.power_of_two

    def filter_time_window(self) -> pd.DataFrame:
        """
        Filters a DataFrame to include only data within a specified time window.

        Returns:
            A new DataFrame containing only the data within the specified time window.
        """
        if self.start_time is None or self.window_size is None:
            return self.df

        end_time = self.start_time + self.window_size

        filtered_df = self.df[
            (self.df["Time"] >= self.start_time) & (self.df["Time"] <= end_time)
        ].copy()

        if self.labels is not None:
            for label in self.labels:
                if f"{label} Acceleration" in self.df.columns:
                    filtered_df.loc[:, f"{label} filtered Acceleration"] = self.df.loc[
                        filtered_df.index, f"{label} Acceleration"
                    ]

        return filtered_df

    def remove_outliers(self, label: str) -> pd.DataFrame:
        """
        Removes outliers from the acceleration signal.

        Args:
            label: The label of the vibration component (e.g., 'X', 'Y', 'Z').

        Returns:
            DataFrame with outliers replaced by the mean.
        """
        mean = self.df[f"{label} Acceleration"].mean()
        std = self.df[f"{label} Acceleration"].std()

        self.df.loc[:, f"{label} Acceleration"] = np.where(
            np.abs(self.df[f"{label} Acceleration"] - mean) <= 3 * std,
            self.df[f"{label} Acceleration"],
            mean,
        )

        return self.df

    def baseline_correction(self) -> pd.DataFrame:
        """
        Preprocesses the data by removing outliers and detrending the acceleration signal.

        Returns:
            DataFrame with outliers removed and the acceleration signal detrended.
        """
        for label in self.labels:
            self.df = self.remove_outliers(label)
            self.df.loc[:, f"{label} Acceleration"] = detrend(self.df[f"{label} Acceleration"])

        return self.df

    def butter_bandpass_filter(self) -> pd.DataFrame:
        """
        Apply a Butterworth bandpass filter to the acceleration data.

        Returns:
            DataFrame with filtered acceleration data.
        """
        time_step = self.df["Time"].iloc[1] - self.df["Time"].iloc[0]
        sampling_rate = 1 / time_step
        nyquist = 0.5 * sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype="band")

        for label in self.labels:
            try:
                filtered_acceleration = filtfilt(b, a, self.df[f"{label} Acceleration"])
                self.df.loc[:, f"{label} filtered Acceleration"] = filtered_acceleration
            except ValueError as e:
                print(f"Error filtering {label} Acceleration: {e}")

        return self.df

    def execute_preparing_signal(self) -> pd.DataFrame:
        """
        Executes the signal preparation pipeline.

        Returns:
            A processed DataFrame.
        """
        self.df = self.filter_time_window()
        self.df = self.baseline_correction()
        return self.butter_bandpass_filter()


def plot_acceleration(df: pd.DataFrame, label: str, color: str = "red"):
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
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{label} Acceleration.png", dpi=300)
    plt.show()


def plotly_acceleration(df: pd.DataFrame, label: str, color: str = "red") -> go.Figure:
    """
    Plots the acceleration signal from a DataFrame using Plotly.

    Args:
        df: DataFrame containing 'Time' and '{label} Acceleration' columns.
        label: Direction of the acceleration signal to plot.
        color: Color of the plot line.

    Returns:
        The Plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df[f"{label} Acceleration"],
            mode="lines",
            name=label,
            line=dict(color=color),
        )
    )
    fig.update_layout(
        title=f"{label} Acceleration",
        xaxis_title="Time (s)",
        yaxis_title="Acceleration (m/s^2)",
        template="plotly_white",
    )
    os.makedirs("results", exist_ok=True)
    fig.write_html(f"results/{label} Acceleration.html")
    return fig