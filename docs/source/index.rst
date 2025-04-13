.. SigePy documentation master file.

Welcome to SigePy's Documentation
================================

.. image:: _static/estructurapy.png
   :alt: SigePy Logo
   :align: center
   :width: 300px

**SigePy** is an advanced Python library specialized in structural vibration analysis 
and system identification, with robust capabilities for processing experimental and 
operational modal data.

Key Features
-----------

* **Fourier Analysis**: Enhanced FFT for structural dynamics, peak detection, and spectral analysis
* **Stochastic Subspace Identification**: Covariance-driven SSI for modal identification
* **Signal Processing**: Time-domain analysis, filtering, and preprocessing
* **Wavelet Analysis**: Time-frequency analysis for non-stationary signals

Installation
-----------

Install SigePy using pip:

.. code-block:: bash

   pip install sigepy

Quick Example
-----------

.. code-block:: python

   import sigepy
   import pandas as pd
   import numpy as np
   
   # Create synthetic vibration data
   t = np.linspace(0, 10, 1000)
   accel = np.sin(2*np.pi*2*t) + 0.5*np.sin(2*np.pi*5*t) + 0.1*np.random.randn(len(t))
   df = pd.DataFrame({"Time": t, "X Acceleration": accel})
   
   # Calculate FFT
   fft_results = sigepy.calculate_fft(df, labels=["X"])
   
   # Plot results
   sigepy.plot_fft_results(fft_results, label="X")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
