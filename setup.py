from setuptools import setup
from Cython.Build import cythonize
import numpy as np
from setuptools.extension import Extension
import sys
if sys.platform in ("win32", "linux", "darwin"):  # All platforms
    if sys.platform == "win32":
        extra_compile_args = [
            "/O2",            # Equivalent to -O3
            "/fp:fast",       # Fast floating point model
            "/Ot",           # Favor fast code
            "/Ox",           # Maximum optimization
            "/Oi",           # Enable intrinsic functions
            "/GT",           # Fiber-safe optimizations
            "/std:c++17"     # C++17 standard
        ]
        extra_link_args = ["/OPT:REF", "/OPT:ICF"]
    else:  # linux and darwin (macOS)
        extra_compile_args = [
            "-O3",                             # Maximum optimization
            "-funroll-loops",                  # Loop unrolling
            "-ftree-vectorize",                # Enable vectorization
            "-fstrict-aliasing",               # Enable strict aliasing
            "-fstack-protector-strong",        # Stack protection
            "-Wno-unreachable-code-fallthrough",  # Ignore unreachable code warnings
            "-std=c++17"                       # C++17 standard
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
