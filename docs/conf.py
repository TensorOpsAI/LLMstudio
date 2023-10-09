# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LLMstudio'
copyright = '2023, TensorOps'
author = 'TensorOps'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(0, os.path.abspath('../api'))
# sys.path.insert(0, os.path.abspath('../llmstudio'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # if you're using Google- or NumPy-style docstrings
    'sphinx.ext.viewcode',  # if you want to include linked source code in your docs
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
