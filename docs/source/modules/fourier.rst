Fourier Analysis
===============

.. automodule:: sigepy.fourier
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``fourier`` module provides functions for frequency domain analysis of vibration signals, including:

* FFT calculation and normalization
* Frequency filtering
* Peak detection in the frequency domain
* Visualization of frequency spectra

Example Usage
------------

.. code-block:: python

    import sigepy
    import pandas as pd
    
    # Load data
    df = pd.DataFrame({"Time": [0.0, 0.1, 0.2], "X Acceleration": [0.1, 0.2, 0.3]})
    
    # Calculate FFT
    fft_results = sigepy.calculate_fft(df, labels=["X"])
    
    # Plot results
    sigepy.plot_fft_results(fft_results, label="X")
