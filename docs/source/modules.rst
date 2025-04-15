.. SigePy documentation master file.

Welcome to SigePy's Documentation
=================================
SigePy is an advanced Python library developed under estructuraPy, a trademark of ANM Ingeniería, for structural vibration analysis and system identification. 
It integrates both existing libraries and custom-built algorithms to enhance the implementation of advanced modal analysis and signal processing methods.
Core Features:

- **System Identification & Modal Analysis**: Covariance-driven and data-driven stochastic subspace identification.
- **Time & Frequency Domain Analysis**: Peak detection, spectral analysis, and Fourier transforms.
- **Signal Processing**: Digital filtering, adaptive filtering, and signal enhancement.
- **Integration of Existing Libraries**: Utilizes NumPy, SciPy, Matplotlib, and other powerful Python libraries for efficient computation.
- **Custom-Built Code**: Implements proprietary algorithms to optimize and extend standard methodologies.
- **Visualization Tools**: Stabilization diagrams, time-frequency spectra, and modal shape representations.
- **Artificial Signal Generation for Testing**: While some algorithms for signal generation are included, SigePy is not primarily designed to generate signals. The provided methods may not conform to all criteria required for signal synthesis, but they are useful for creating artificial signals to test the library and validate signal decomposition techniques.

Key Features
------------

* **Fourier Analysis**: Enhanced FFT for structural dynamics, peak detection, and spectral analysis
* **Stochastic Subspace Identification**: Covariance-driven SSI for modal identification
* **Wavelet Analysis**: Time-frequency analysis for non-stationary signals
* **Signal Processing**: Time-domain analysis, filtering, and preprocessing

SigePy it's a Python development of estructuraPy a trademark of ANM Ingenieria https://www.anmingenieria.com/.

.. image:: https://github.com/estructuraPy/sigepy/raw/main/estructurapy.png
   :alt: estructuraPy Logo
   :align: center
   :width: 300px

:Author: Angel Navarro-Mora
:Contact: ahnavarro@anmingenieria.com
:Copyright: 2025, estructuraPy
:Version: 0.1.5
:License: MIT

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules/fourier
   modules/ssi_cov
   modules/wavelet
   modules/utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

* :ref:`search`