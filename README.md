# SigePy

## A Python Library for Structural Vibration Analysis

SigePy is an advanced Python library specialized in structural vibration analysis and system identification, with robust capabilities for processing experimental and operational modal data. It's particularly suited for civil/mechanical engineers, researchers, and practitioners working with structural health monitoring, modal analysis, and vibration-based damage detection.

## Core Capabilities

### System Identification and Modal Analysis
- **SSI-COV**: Covariance-driven Stochastic Subspace Identification
  - Automated model order selection
  - Stabilization diagrams
  - Modal parameter extraction
- **SSI-DATA**: Data-driven Stochastic Subspace Identification
- **Operational Modal Analysis**: Output-only modal identification
- **Modal Validation**: MAC, COMAC, and mode complexity indicators

### Time Domain Analysis
- **Statistical Analysis**: RMS, crest factor, kurtosis for vibration assessment
- **Peak Detection**: Impact and transient response identification
- **Envelope Analysis**: Structural response characterization
- **Feature Extraction**: Time-domain vibration indicators

### Frequency Domain Analysis
- **Fourier Analysis**: Enhanced FFT for structural dynamics
- **Spectral Analysis**: Power Spectral Density (PSD), FRF computation
- **Modal Parameters**: Natural frequencies and damping estimation
- **Order Analysis**: For rotating machinery diagnostics

### Time-Frequency Analysis
- **Short-Time Fourier Transform (STFT)**: Non-stationary response analysis
- **Wavelet Analysis**: Damage localization and transient detection
- **EMD/HHT**: Hilbert-Huang Transform for nonlinear systems

### Signal Processing
- **FIR/IIR Filters**: Digital filtering for noise reduction
- **Adaptive Filtering**: LMS, RLS algorithms
- **Signal Enhancement**: Advanced denoising techniques

## Technical Implementation

SigePy leverages NumPy, SciPy, and optimized C/C++ extensions for maximum performance in:

- Real-time structural monitoring
- Multi-channel sensor arrays
- Large-scale vibration data
- High-frequency sampling applications

## Installation

```bash
pip install signalepy
```

## Basic Usage

```python
import siglepy as sp
import numpy as np

# Load acceleration data
data = np.loadtxt('structural_response.txt')
fs = 100  # Sampling frequency (Hz)

# Perform SSI-COV analysis
frequencies, damping, modes = sp.modal.ssi_cov(
    data,
    fs=fs,
    n_block_rows=40,
    system_order=50
)

# Calculate Modal Assurance Criterion
mac_matrix = sp.modal.mac(modes)

# Generate stabilization diagram
sp.modal.plot_stabilization(
    frequencies, 
    damping,
    show_stable=True
)

# Operational Modal Analysis
modal_params = sp.modal.oma(
    data,
    fs=fs,
    method='ssi-cov',
    n_modes=5
)
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

## Documentation

For complete API reference and examples, visit our [documentation](https://signalepy.readthedocs.io/).

## License

SigePy is released under the MIT License.
