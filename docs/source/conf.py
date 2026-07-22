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
import sys,os,datetime
from importlib.metadata import version
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = "YARP<sup>again</sup>"
copyright = f'{datetime.date.today().year}, SRG'
copyright = '2025, Savoie Research Group'
author = 'Savoie Research Group'

# The full version, including alpha/beta/rc tags
release = '3.0.1'
#release = version('yarp_again')
# for example take major/minor
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'autoapi.extension',
    'sphinx_rtd_theme',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
]


# Point autoapi to the yarp source
autoapi_type = 'python'
#autoapi_dirs = ['../yarp', '../test']  # path from docs/source/
import os

autoapi_dirs = [os.path.abspath(os.path.join(os.path.dirname(__file__), '../../yarp'))]

autoapi_add_toctree_entry = True
autoapi_keep_files = True  # Optional: lets you inspect the generated .rst



# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']

# -- Extension configuration -------------------------------------------------

autosummary_generate = True
autodoc_default_flags = ['members', ]
add_module_names = False  # hide full dotted names in function headers



typehints_defaults = "comma"

napoleon_google_docstring = False
napoleon_include_init_with_doc = True

rst_prolog = """
.. |autoapi-title| replace:: YARP Internal Module Reference
"""

autoapi_python_class_content = "both"  # include both class docstring and __init__ doc
autoapi_member_order = "groupwise"     # groups functions, classes, etc.
