Stochastic Subspace Identification (SSI-COV)
===========================================

.. automodule:: sigepy.ssi_cov
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``ssi_cov`` module implements Covariance-driven Stochastic Subspace Identification for modal analysis:

* Extraction of modal parameters (natural frequencies, damping ratios, mode shapes)
* Creation of stabilization diagrams
* Automated model order selection
* Robust handling of noisy data

Example Usage
------------

.. code-block:: python

    import sigepy
    import pandas as pd
    
    # Load data
    df = pd.DataFrame({"Time": [0.0, 0.1, 0.2], "X Acceleration": [0.1, 0.2, 0.3]})
    
    # Create SSI-COV analyzer
    ssi = sigepy.SSICov(df, acceleration_labels=["X"], min_model_order=2, max_model_order=20)
    
    # Generate stability diagram
    ssi.plot_stability_diagram()
