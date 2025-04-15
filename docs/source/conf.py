# Configuration file for the Sphinx documentation builder.

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath('../../../src/sigepy'))

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

# Add intersphinx extension
extensions.append('sphinx.ext.intersphinx')

# Configure intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'plotly': ('https://plotly.com/python-api-reference', None),
}

# Add nitpick ignore patterns for common external references
nitpick_ignore = [
    ('py:class', 'pandas.DataFrame'),
    ('py:class', 'numpy.ndarray'),
    ('py:class', 'numpy.typing.NDArray'),
    ('py:class', 'plotly.graph_objects.Figure'),
    ('py:class', 'plotly.graph_objs.Figure'),
    ('py:class', '_io._BufferedIOBase'),
    ('py:class', 'pathlib.PurePath'),
    ('py:class', 'callable'),
]

# Opciones de autodoc - ajustar para mostrar más contenido
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__, __post_init__',
    'imported-members': True,
}

# Ensure that the __init__ method is documented
autoclass_content = 'both'

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
