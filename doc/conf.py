import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xmodmap import __version__


# -- Project information -----------------------------------------------------

project = 'xmodmap'

# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

copyright = '2026, INRAE, École Normale Supérieure Paris-Saclay, Johns Hopkins University.'
author = 'xmodmap dev team'

# -- General configuration ---------------------------------------------------


extensions = [
    "sphinx_prompt",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",
    "sphinxcontrib.httpdomain",
    "sphinx_design",
    "sphinx.ext.napoleon",
    #"sphinx.ext.linkcode",
    "sphinx_gallery.gen_gallery",
    "pyvista.ext.viewer_directive",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_title = project + " " + release

html_context = {
    "display_github": True,  # Integrate Github
    "github_user": "bcharlier",  # Username
    "github_repo": "xIV-LDDMM-Particle",  # Repo name
    "github_version": "unpack",  # Version
    "conf_py_path": "/doc/",  # Path in the checkout to the docs root
}

# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = False
autodoc_member_order = "bysource"


# -- Options for Gallery  -------------------------------------------------

from sphinx_gallery.sorting import FileNameSortKey
import pyvista as pv
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper

pv.BUILDING_GALLERY = True
pv.OFF_SCREEN = True


sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": ["../examples",],
    # path where to save gallery generated examples
    "gallery_dirs": ["./_auto_examples",],
    # order of the Gallery
    "within_subsection_order": FileNameSortKey,
    # Add patterns
    "ignore_pattern": r"__init__\.py|G2T\.py",
    "parallel": int(os.environ.get("SPHINX_GALLERY_JOBS", "1")),
    "image_scrapers": (DynamicScraper(),),

}
