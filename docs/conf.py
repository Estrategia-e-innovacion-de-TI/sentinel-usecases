# Sphinx configuration for Sentinel documentation

import os
import sys

# Add source directory to path so autodoc can find the package
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "Sentinel"
copyright = "2026, Bancolombia"
author = "José Manuel Vergara Álvarez et al."
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True

# Napoleon settings (NumPy / Google style docstrings)
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "Sentinel"
html_logo = "images/sentinel icon with text.png"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
}

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- MyST parser (for .md files) ---------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Mock imports for optional dependencies -----------------------------------
autodoc_mock_imports = [
    "torch",
    "ncps",
    "rrcf",
    "plotly",
    "shap",
]
