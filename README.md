# SignalePy

## A Comprehensive Signal Processing Library for Python

SignalePy is an advanced Python library dedicated to signal processing tasks across various domains including audio processing, biomedical signal analysis, telecommunications, and general time series analysis.

## Core Capabilities

### Time Domain Analysis
- **Statistical Analysis**: Mean, variance, skewness, kurtosis, correlation functions
- **Peak Detection**: Multiple algorithms including local maxima/minima and threshold-based methods
- **Envelope Detection**: Hilbert transform and other envelope extraction techniques
- **Feature Extraction**: Zero-crossing rate, energy, entropy

### Frequency Domain Analysis
- **Fourier Transforms**: Fast Fourier Transform (FFT) implementations based on the Cooley-Tukey algorithm
- **Spectral Analysis**: Power Spectral Density (PSD), spectrograms
- **Harmonic Analysis**: Fundamental frequency estimation, harmonic-to-noise ratio

### Time-Frequency Analysis
- **Short-Time Fourier Transform (STFT)**: Windowed Fourier analysis
- **Wavelet Transforms**: Continuous Wavelet Transform (CWT) and Discrete Wavelet Transform (DWT)
- **Empirical Mode Decomposition (EMD)**: Adaptive decomposition for non-stationary signals

### Filtering
- **FIR Filters**: Window-based and Parks-McClellan optimal design
- **IIR Filters**: Butterworth, Chebyshev, Elliptic filter implementations
- **Adaptive Filters**: LMS, RLS, Kalman filtering

### Signal Enhancement
- **Noise Reduction**: Spectral subtraction, Wiener filtering
- **Deconvolution**: Blind and non-blind techniques
- **Signal Separation**: Independent Component Analysis (ICA)

## Technical Implementation

SignalePy leverages NumPy, SciPy, and optimized C/C++ extensions for maximum performance. The library emphasizes:

- Memory efficiency for large datasets
- Parallelization for multi-core processing
- GPU acceleration for selected algorithms
- Comprehensive validation against industry standards

## Installation

```bash
pip install signalepy
```

## Basic Usage

```python
import signalepy as sp
import numpy as np

# Generate a test signal
fs = 1000  # Sampling frequency (Hz)
t = np.arange(0, 1, 1/fs)
signal = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t) + 0.1*np.random.randn(len(t))

# Apply a bandpass filter
filtered = sp.filters.bandpass(signal, lowcut=40, highcut=60, fs=fs, order=4)

# Perform FFT analysis
frequencies, spectrum = sp.fft.magnitude_spectrum(signal, fs=fs)

# Compute spectrogram
time_bins, freq_bins, Sxx = sp.timefreq.spectrogram(signal, fs=fs, window='hann', nperseg=256)
```

## References

The implementation is based on established signal processing literature:

1. Oppenheim, A.V. & Schafer, R.W. (2009). *Discrete-Time Signal Processing* (3rd ed.). Prentice Hall.
2. Proakis, J.G. & Manolakis, D.G. (2006). *Digital Signal Processing* (4th ed.). Prentice Hall.
3. Mallat, S. (2008). *A Wavelet Tour of Signal Processing: The Sparse Way* (3rd ed.). Academic Press.
4. Haykin, S. (2013). *Adaptive Filter Theory* (5th ed.). Pearson.
5. Cohen, L. (1995). *Time-Frequency Analysis*. Prentice Hall.

## Documentation

For complete API reference and examples, visit our [documentation](https://signalepy.readthedocs.io/).

## License

SignalePy is released under the MIT License.
