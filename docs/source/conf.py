# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

import pytorch_sphinx_theme
from typing import List

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "SESEMI"
copyright = "2021, Flyreel AI"
author = "Vu Tran, Tigist Diriba, et al."


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinxcontrib.programoutput",
    "m2r2",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: List[str] = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]
html_theme_options = {
    "logo_url": "https://sesemi.readthedocs.io/en/latest/",
    "menu": [
        {
            "name": "Installation",
            "url": "https://sesemi.readthedocs.io/en/latest/installation.html",
        },
        {
            "name": "Quickstart",
            "url": "https://sesemi.readthedocs.io/en/latest/quickstart.html",
        },
        # {"name": "Blogs", "url": "https://github.com/open-mmlab/"},
        {
            "name": "Tutorials",
            "url": "https://sesemi.readthedocs.io/en/latest/tutorials/project_setup.html",
        },
        {
            "name": "Methods",
            "url": "https://sesemi.readthedocs.io/en/latest/methods/rotation_prediction.html",
        },
        {
            "name": "Ops",
            "url": "https://sesemi.readthedocs.io/en/latest/ops/inference.html",
        },
        {
            "name": "API Reference",
            "url": "https://sesemi.readthedocs.io/en/latest/api/sesemi.html",
        },
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/readthedocs.css"]

source_suffix = [".rst", ".md"]
