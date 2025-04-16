"""
SigePy: A Python Library for Structural Vibration Analysis and System Identification.

Modules:
- `fourier`: Tools for Fourier analysis, including FFT computation and filtering.
- `ssi_cov`: Implementation of Stochastic Subspace Identification (SSI-COV) for modal analysis.
- `utils`: Helper functions for data handling, signal generation, and preprocessing.
- `wavelet`: Wavelet-based tools for time-frequency analysis and visualization.
"""

# Import core modules for easier access
from .fo_urier import calculate_fft, filter_with_fft, plot_fft_results, plot_peaks
from .ss_i_cov import SSICov
from .util_s import (
    estimate_power_of_two,
    get_tests_files_location,
    get_results_files_location,
    get_data_files_location,
    import_sts_acceleration_txt,
    import_csv_acceleration,
    import_cscr_fed,
    generate_vibration_signal,
    generate_vibration_signals,
)
from .wa_velet import calculate_cwt, plot_spectrum_views, spectrum