# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:

import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))
# -- Project information -----------------------------------------------------

project = 'SigePy'
copyright = '2025, estructuraPy'
author = 'Angel Navarro-Mora'
release = '0.1.4'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Ensure that the module documentation can be found
html_extra_path = ['modules']
