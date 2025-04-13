# Configuration file for the Sphinx documentation builder.

import os
import sys
import warnings

# Add the source directory to the Python path so Sphinx can import modules
sys.path.insert(0, os.path.abspath('../../src'))

# Enable better handling of imports
autodoc_mock_imports = ['numpy', 'pandas', 'scipy', 'matplotlib', 'plotly', 'pywt', 'tqdm', 'PIL', 'ipywidgets']

# Silence warnings about missing references (usually from external libs)
warnings.filterwarnings('ignore', message='.*Duplicate C++ declaration.*')

# -- Project information -----------------------------------------------------
project = 'SigePy'
author = 'Angel Navarro-Mora'
copyright = '2025, estructuraPy'
release = '0.1.5'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',  
    'sphinx.ext.napoleon',  # For parsing Google/NumPy-style docstrings
    'sphinx.ext.viewcode',  # Shows source code links in docs
    'sphinx.ext.autosummary',  # Creates summary tables
]

# Configure autodoc
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}

# Enable autosummary
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = 'https://github.com/estructuraPy/sigepy/raw/main/estructurapy.png'
