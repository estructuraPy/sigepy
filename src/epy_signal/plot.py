import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pywt
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import detrend, butter, filtfilt
import ipywidgets as widgets
from IPython.display import display



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
    plt.show()

    