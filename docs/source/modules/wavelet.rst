Wavelet Analysis
==============

.. automodule:: sigepy.wavelet
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``wavelet`` module provides functions for time-frequency analysis using wavelets:

* Continuous Wavelet Transform (CWT) calculation
* Interactive visualization of wavelet spectra
* Time-frequency analysis of non-stationary signals
* Multiple visualization options (3D plots, contour plots, etc.)

Example Usage
------------

.. code-block:: python

    import sigepy
    import pandas as pd
    
    # Load data
    df = pd.DataFrame({"Time": [0.0, 0.1, 0.2], "X Acceleration": [0.1, 0.2, 0.3]})
    
    # Calculate CWT
    spectrum_data, frequencies = sigepy.calculate_cwt(df, label="X")
    
    # Plot spectrum views
    sigepy.plot_spectrum_views(df, spectrum_data, frequencies, label="X")
