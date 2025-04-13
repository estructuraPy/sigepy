Utils Module
=============

This module provides general-purpose utilities for data handling, signal generation, and preprocessing.

.. automodule:: sigepy.utils
   :members:
   :undoc-members:
   :show-inheritance:

File Handling
-------------

.. autofunction:: sigepy.utils.get_tests_files_location
.. autofunction:: sigepy.utils.get_results_files_location
.. autofunction:: sigepy.utils.get_data_files_location

Data Import
-----------

.. autofunction:: sigepy.utils.import_sts_acceleration_txt
.. autofunction:: sigepy.utils.import_csv_acceleration
.. autofunction:: sigepy.utils.import_cscr_fed

Signal Generation
-----------------

.. autofunction:: sigepy.utils.generate_vibration_signal
.. autofunction:: sigepy.utils.generate_vibration_signals

Signal Processing
-----------------

.. autofunction:: sigepy.utils.estimate_power_of_two
.. autoclass:: sigepy.utils.SignalProcessor
   :members:
   :undoc-members:

Visualization
-------------

.. autofunction:: sigepy.utils.plot_acceleration
.. autofunction:: sigepy.utils.plotly_acceleration
