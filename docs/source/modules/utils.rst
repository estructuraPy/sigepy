Utilities
========

.. automodule:: sigepy.utils
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``utils`` module provides general-purpose functions for:

* Data loading and preprocessing
* Signal generation for testing and simulation
* File path management
* Signal processing (filtering, outlier removal, etc.)
* Visualization of time domain signals

Example Usage
------------

.. code-block:: python

    import sigepy
    import pandas as pd
    
    # Generate a synthetic vibration signal
    df = sigepy.generate_vibration_signal(
        label="X",
        total_time=10.0,
        sampling_rate=100,
        frequency_inputs=[5.0, 10.0],
        amplitude_inputs=[1.0, 0.5],
        noise_amplitude=0.1
    )
    
    # Process the signal
    processor = sigepy.utils.SignalProcessor(df, labels=["X"], lowcut=1, highcut=20)
    filtered_df = processor.execute_preparing_signal()
    
    # Plot the result
    sigepy.utils.plot_acceleration(filtered_df, label="X")
