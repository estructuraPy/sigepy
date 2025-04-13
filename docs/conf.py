# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'SigePy'
author = 'Your Name or Organization'
release = '0.1.4'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']
html_logo = '_static/estructurapy.png'
