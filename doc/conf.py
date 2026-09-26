# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'silt'
copyright = '2026, Nicholas McDonald, erosiv Studio'
author = 'Nicholas McDonald, erosiv Studio'

import pathlib as _pathlib
release = (_pathlib.Path(__file__).resolve().parent.parent
           / 'VERSION').read_text(encoding='utf-8').strip()
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'breathe',      # renders Doxygen XML (below) as Sphinx pages -- api_cpp.rst
    'myst_parser',  # lets .rst files pull in fragments of README.md -- see index.rst
]

templates_path = ['_templates']
exclude_patterns = []

# -- C++ API reference (Doxygen + Breathe) ------------------------------------
#
#  Breathe itself only reads Doxygen's XML output; it doesn't run Doxygen.
#  Running it here, once, keeps `sphinx-build doc build/html` a single command
#  instead of a documented two-step "run doxygen, then sphinx-build" -- if
#  Doxygen isn't installed/on PATH this raises loudly at build time rather
#  than silently leaving api_cpp.rst empty.

import subprocess

_doc_dir = _pathlib.Path(__file__).resolve().parent
subprocess.run(['doxygen', 'Doxyfile'], cwd=_doc_dir, check=True)

breathe_projects = {'silt': str(_doc_dir / '_build' / 'doxygen' / 'xml')}
breathe_default_project = 'silt'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
