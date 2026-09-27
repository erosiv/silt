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
    'breathe',            # renders Doxygen XML (below) as Sphinx pages -- api_cpp.rst
    'myst_parser',        # lets .rst files pull in fragments of README.md -- see index.rst
    'sphinx.ext.autodoc', # generates api_python.rst's reference from the built `silt` module
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

import shutil
import subprocess

_doc_dir = _pathlib.Path(__file__).resolve().parent
_doxygen = shutil.which('doxygen')
if _doxygen is None:
    raise RuntimeError(
        "doxygen not found on PATH -- required to build the C++ API reference "
        "(see doc/Doxyfile, and the `docs` extra in pyproject.toml for the rest "
        "of the toolchain). If you just installed it, reopen your terminal: "
        "PATH changes don't reach a shell that was already running."
    )
# Doxygen can create one missing OUTPUT_DIRECTORY level itself, but not
# nested ones -- `_build/doxygen` is two levels deep on a clean checkout
# (`_build` doesn't exist either), so it refuses outright instead of
# creating both. Pre-create it here rather than relying on Doxygen to.
(_doc_dir / '_build' / 'doxygen').mkdir(parents=True, exist_ok=True)

_doxygen_result = subprocess.run(
    [_doxygen, 'Doxyfile'], cwd=_doc_dir,
    capture_output=True, text=True,
)
if _doxygen_result.returncode != 0:
    raise RuntimeError(
        f"doxygen exited with status {_doxygen_result.returncode}:\n"
        f"{_doxygen_result.stdout}{_doxygen_result.stderr}"
    )

breathe_projects = {'silt': str(_doc_dir / '_build' / 'doxygen' / 'xml')}
breathe_default_project = 'silt'

# silt::typedesc<T> is explicitly specialized per dtype (float/int/rng) --
# see design.rst's static-polymorphism section. Breathe renders each
# specialization's signature without its template argument, so Sphinx's
# C++ domain sees three identical `template<> typedesc` declarations and
# warns. Harmless (all three still render); this is the documented way to
# silence that specific, expected warning class rather than every warning.
suppress_warnings = ['duplicate_declaration.cpp']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']

# Matches https://erosiv.studio's palette/type (see doc/_static/erosiv.css);
# loaded after the theme's own stylesheet so its rules win on equal specificity.
html_css_files = ['erosiv.css']
