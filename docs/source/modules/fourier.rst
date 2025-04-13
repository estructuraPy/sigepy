Fourier Analysis
================


.. automodule:: sigepy.fourier
   :members:
   :undoc-members:
   :show-inheritance:

This module provides functions for frequency domain analysis of vibration signals, including FFT calculation, filtering, and visualization.

Core Analysis Functions
-----------------------

.. autofunction:: sigepy.fourier.calculate_fft
.. autofunction:: sigepy.fourier.filter_with_fft

Matplotlib Visualization
------------------------

.. autofunction:: sigepy.fourier.plot_normalized_fft_results
.. autofunction:: sigepy.fourier.plot_fft_results
.. autofunction:: sigepy.fourier.plot_fft_results_period_domain
.. autofunction:: sigepy.fourier.plot_peaks

Interactive Plotly Visualization
--------------------------------

.. autofunction:: sigepy.fourier.plotly_normalized_fft_results
.. autofunction:: sigepy.fourier.plotly_fft_results
.. autofunction:: sigepy.fourier.plotly_fft_results_period_domain
.. autofunction:: sigepy.fourier.plotly_peaks