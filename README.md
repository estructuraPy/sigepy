[EstructuraPy Logo](https://github.com/estructuraPy/sigepy/blob/main/estructurapy.png)

# SigePy

## A Python Library for Structural Vibration Analysis

SigePy is an advanced Python library specialized in structural vibration analysis and system identification, with robust capabilities for processing experimental and operational modal data. It's particularly suited for civil/mechanical engineers, researchers, and practitioners working with structural health monitoring, modal analysis, and vibration-based damage detection.

## Core Capabilities

### System Identification and Modal Analysis
- **SSI-COV**: Covariance-driven Stochastic Subspace Identification
  - Automated model order selection
  - Stabilization diagrams
  - Modal parameter extraction
  - Robust handling of noisy data
- **SSI-DATA**: Data-driven Stochastic Subspace Identification
- **Operational Modal Analysis**: Output-only modal identification
- **Modal Validation**: MAC (Modal Assurance Criterion), COMAC (Coordinate Modal Assurance Criterion), and mode complexity indicators

### Time Domain Analysis
- **Peak Detection**: Impact and transient response identification
- **Envelope Analysis**: Structural response characterization
- **Feature Extraction**: Time-domain vibration indicators

### Frequency Domain Analysis
- **Fourier Analysis**: Enhanced FFT for structural dynamics
- **Peak Detection**: Frequency domain peak detection
- **Spectral Analysis**: Power Spectral Density (PSD), Frequency Response Function (FRF) computation
- **Modal Parameters**: Natural frequencies and damping estimation
- **Order Analysis**: For rotating machinery diagnostics

### Time-Frequency Analysis
- **Short-Time Fourier Transform (STFT)**: Non-stationary response analysis
- **Wavelet Analysis**: Continuous Wavelet Transform (CWT) for damage localization and transient detection
- **Hilbert-Huang Transform (HHT)**: Empirical Mode Decomposition (EMD) for nonlinear systems

### Signal Processing
- **FIR/IIR Filters**: Digital filtering for noise reduction
- **Adaptive Filtering**: LMS, RLS algorithms
- **Signal Enhancement**: Advanced denoising techniques
- **Bandpass Filtering**: For isolating specific frequency ranges

### Visualization
- **Stabilization Diagrams**: For SSI-COV and SSI-DATA
- **Time-Frequency Spectra**: Interactive 3D plots for wavelet and STFT analysis
- **Modal Shapes**: Visualization of extracted mode shapes
- **Acceleration Plots**: Time-domain signal visualization

## Installation

SigePy requires Python 3.13 or higher. Install it using pip:

```bash
pip install sigepy
```

## Application Areas

- Structural Health Monitoring
- Bridge and Building Dynamics
- Seismic Response Analysis
- Wind-induced Vibrations
- Machine Foundation Analysis
- Modal Testing and Analysis

## References

1. Peeters, B., & De Roeck, G. (1999). *Reference-based stochastic subspace identification for output-only modal analysis*. Mechanical Systems and Signal Processing.
2. Brincker, R., & Ventura, C. (2015). *Introduction to Operational Modal Analysis*. Wiley.
3. Ewins, D.J. (2000). *Modal Testing: Theory, Practice and Application*. Research Studies Press.
4. Rainieri, C., & Fabbrocino, G. (2014). *Operational Modal Analysis of Civil Engineering Structures*. Springer.
5. Oppenheim, A.V. & Schafer, R.W. (2009). *Discrete-Time Signal Processing*. Prentice Hall.
6. Tomassini, E., García-Macías, E., & Ubertini, F. (2025). *Fast Stochastic Subspace Identification of Densely Instrumented Bridges Using Randomized SVD*. Mechanical Systems and Signal Processing.

## Documentation

For complete API reference and examples, visit our [documentation](https://sigepy.readthedocs.io/).

## License

SigePy is released under the MIT License.
