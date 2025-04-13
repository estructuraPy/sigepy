# Configuration file for the Sphinx documentation builder.

import os
import sys
import warnings

# Asegurar que Sphinx pueda encontrar los módulos
sys.path.insert(0, os.path.abspath('../../src'))

# Mock imports para evitar errores con dependencias externas
autodoc_mock_imports = ['numpy', 'pandas', 'scipy', 'matplotlib', 'plotly', 
                        'pywt', 'tqdm', 'PIL', 'ipywidgets']

# Configuración del proyecto
project = 'SigePy'
author = 'Angel Navarro-Mora'
copyright = '2025, estructuraPy'
release = '0.1.5'

# Extensiones necesarias
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

# Opciones de autodoc
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__, __post_init__',
}

# Genera resúmenes automáticos
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []

# Opciones HTML
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = 'https://github.com/estructuraPy/sigepy/raw/main/estructurapy.png'

# Para debugging
keep_warnings = True
nitpicky = True
