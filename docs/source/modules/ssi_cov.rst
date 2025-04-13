Stochastic Subspace Identification (SSI-COV)
===========================================

.. automodule:: sigepy.ssi_cov
   :members:
   :undoc-members:
   :show-inheritance:

This module implements Covariance-driven Stochastic Subspace Identification for modal analysis of structural vibration data.

Utility Functions
---------------

.. autofunction:: sigepy.ssi_cov.construct_and_svd_block_toeplitz

SSI-COV Implementation
--------------------

.. autoclass:: sigepy.ssi_cov.SSICov
   :members:
   :undoc-members:
   :special-members: __post_init__

Core Analysis Methods
-------------------

.. automethod:: sigepy.ssi_cov.SSICov.execute_ssicov_analysis
.. automethod:: sigepy.ssi_cov.SSICov.compute_impulse_response_function
.. automethod:: sigepy.ssi_cov.SSICov.identify_modal_parameters
.. automethod:: sigepy.ssi_cov.SSICov.perform_stability_analysis

Visualization Methods
-------------------

.. automethod:: sigepy.ssi_cov.SSICov.plot_stability_diagram
.. automethod:: sigepy.ssi_cov.SSICov.plotly_stability_diagram

