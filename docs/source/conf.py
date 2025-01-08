import sys
import os
import types

# Yukarı dizine yol ekle
sys.path.insert(0, os.path.abspath("../.."))

# obliquetree.src.base için mock modül oluştur
mock_base_module = types.ModuleType("obliquetree.src.base")


class TreeClassifier:
    pass


# TreeClassifier sınıfını mock modüle ekle
mock_base_module.TreeClassifier = TreeClassifier

# Mock modülü sys.modules'a ekle
sys.modules["obliquetree.src.base"] = mock_base_module

# obliquetree.src.utils için mock modül oluştur
mock_utils_module = types.ModuleType("obliquetree.src.utils")


# export_tree fonksiyonunu mock olarak oluştur
def mock_export_tree(*args, **kwargs):
    pass


# Mock modüle export_tree fonksiyonunu ekle
mock_utils_module.export_tree = mock_export_tree

# Mock modülü sys.modules'a ekle
sys.modules["obliquetree.src.utils"] = mock_utils_module

# Sphinx yapılandırmaları
project = "obliquetree"
copyright = "2025, Samet Çopur"
author = "Samet Çopur"
version = "1.0"
release = "1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "myst_parser",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]


napoleon_use_param = False
napoleon_use_ivar = False
napoleon_use_rtype = False


autodoc_typehints = 'description' 
autodoc_member_order = "bysource"

templates_path = ["_templates"]
html_theme = "furo"
html_static_path = ["_static"]
