# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

import toml

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "bio_embeddings"
html_title = "bio_embeddings"
copyright = (
    "2020, Christian Dallago, Konstantin Schütze, Michael Heinzinger, Tobias Olenyi"
)
author = "Christian Dallago <christian.dallago@tum.de>, Konstantin Schütze <schuetze@in.tum.de>, Michael Heinzinger <mheinzinger@rostlab.org>, Tobias Olenyi <olenyi@rostlab.org>"

html_baseurl = "https://docs.bioembeddings.com"

# The full version, including alpha/beta/rc tags
release = toml.loads(
    Path(__file__).parent.parent.joinpath("pyproject.toml").read_text()
)["tool"]["poetry"]["version"]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    # "If you are using MyST-NB in your documentation, do not activate myst-parser.
    # It will be automatically activated by myst-nb."
    "myst_nb",
    "sphinx_copybutton",
]

jupyter_execute_notebooks = "off"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

autodoc_default_options = {
    'special-members': '__init__',
}