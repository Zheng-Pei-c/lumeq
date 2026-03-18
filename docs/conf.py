# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
#sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = 'lumeq'
copyright = '2025, Zheng Pei'
author = 'Zheng Pei'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.napoleon',
        'sphinx.ext.viewcode',
        #'sphinx.ext.linkcode',
        'sphinx.ext.autosummary',
        'sphinx.ext.autodoc',
        'sphinx.ext.apidoc',
        'sphinx.ext.extlinks',
        'sphinx.ext.githubpages',
        'sphinx.ext.intersphinx',
        ]

templates_path = ['_templates']
exclude_patterns = ['_build']

add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

try:
    import sphinx_rtd_theme  # noqa: F401
except ImportError:
    html_theme = 'alabaster'
    html_theme_options = {}
else:
    html_theme = 'sphinx_rtd_theme'
    html_theme_options = {
            'vcs_pageview_mode': 'edit',
            }
html_static_path = ['_static']
html_context = {
    'display_github': True,
    'github_user': 'Zheng-Pei-c',
    'github_repo': 'lumeq',
    'github_version': 'main',
    'conf_py_path': '/docs/',  # Path in the checkout to the docs root
}
