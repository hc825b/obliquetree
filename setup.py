from setuptools import setup
from Cython.Build import cythonize
import numpy as np
from setuptools.extension import Extension
import sys

if sys.platform == "win32":
    extra_compile_args = [
        "/O2",
        "/fp:fast",
        "/Ot",
        "/Ox",
        "/Oi",
        "/GT",
    ]
    extra_link_args = ["/OPT:REF", "/OPT:ICF"]

elif sys.platform == "linux":
    extra_compile_args = [
        "-O3",
        "-ffast-math",
        "-funroll-loops",
        "-ftree-vectorize",
        "-fstrict-aliasing",
        "-fstack-protector-strong",
    ]
    extra_link_args = []

elif sys.platform == "darwin":  # macOS
    extra_compile_args = [
        "-O3",
        #"-ffast-math",
        "-funroll-loops",
        "-ftree-vectorize",
        "-fstrict-aliasing",
        "-fstack-protector-strong",
        "-Wno-unreachable-code-fallthrough",
        "-std=c++17",
    ]
    extra_link_args = []

extensions = [
    Extension(
        "obliquetree.src.tree",
        ["obliquetree/src/tree.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
        ],  # Disable deprecated API
        extra_compile_args=extra_compile_args,  # Optimization flag
        language="c++",  # Use C++ language
    ),
    Extension(
        "obliquetree.src.oblique",
        ["obliquetree/src/oblique.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
        ],  # Disable deprecated API
        extra_compile_args=extra_compile_args,  # Optimization flag
        language="c++",  # Use C++ language
    ),
    Extension(
        "obliquetree.src.base",
        ["obliquetree/src/base.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
        ],  # Disable deprecated API
        extra_compile_args=extra_compile_args,  # Optimization flag
        language="c++",  # Use C++ language
    ),
    Extension(
        "obliquetree.src.utils",
        ["obliquetree/src/utils.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
        ],  # Disable deprecated API
        extra_compile_args=extra_compile_args,  # Optimization flag
        language="c++",  # Use C++ language
    ),
    Extension(
        "obliquetree.src.metric",
        ["obliquetree/src/metric.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
        ],  # Disable deprecated API
        extra_compile_args=extra_compile_args,  # Optimization flag
        language="c++",  # Use C++ language
    ),
    Extension(
        "obliquetree.src.ccp",
        ["obliquetree/src/ccp.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
        ],  # Disable deprecated API
        extra_compile_args=extra_compile_args,  # Optimization flag
        language="c++",  # Use C++ language
    ),
]

setup(
    name="obliquetree",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,  # Python 3 dil seviyesi
            "boundscheck": False,  # Dizi sınırı kontrolü kapalı
            "wraparound": False,  # Negatif indeks kontrolü kapalı
            "initializedcheck": False,  # Başlatılmamış değişken kontrolü kapalı
            "cdivision": True,  # C tarzı bölme işlemleri
            "nonecheck": False,
            "overflowcheck":False,
            #"infer_types":True,
        },
    ),
)
