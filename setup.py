from setuptools import setup
from Cython.Build import cythonize
import numpy as np
from setuptools.extension import Extension
import sys

if sys.platform in ("win32", "linux", "darwin"):  # All platforms
    if sys.platform == "win32":
        extra_compile_args = [
            "/O2",  # Equivalent to -O3
            "/fp:fast",  # Fast floating point model
            "/Ot",  # Favor fast code
            "/Ox",  # Maximum optimization
            "/Oi",  # Enable intrinsic functions
            "/GT",  # Fiber-safe optimizations
            "/std:c++17",  # C++17 standard
        ]
        extra_link_args = ["/OPT:REF", "/OPT:ICF"]
    else:  # linux and darwin (macOS)
        extra_compile_args = [
            "-O3",  # Maximum optimization
            "-funroll-loops",  # Loop unrolling
            "-ftree-vectorize",  # Enable vectorization
            "-fstrict-aliasing",  # Enable strict aliasing
            "-fstack-protector-strong",  # Stack protection
            "-Wno-unreachable-code-fallthrough",  # Ignore unreachable code warnings
            "-std=c++17",  # C++17 standard
        ]
        extra_link_args = []

define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
include_dirs = [np.get_include()]


extensions = [
    Extension(
        "obliquetree.src.tree",
        ["obliquetree/src/tree.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    Extension(
        "obliquetree.src.oblique",
        ["obliquetree/src/oblique.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    Extension(
        "obliquetree.src.base",
        ["obliquetree/src/base.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    Extension(
        "obliquetree.src.utils",
        ["obliquetree/src/utils.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    Extension(
        "obliquetree.src.metric",
        ["obliquetree/src/metric.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    Extension(
        "obliquetree.src.ccp",
        ["obliquetree/src/ccp.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
]

setup(
    name="obliquetree",
    packages=["obliquetree", "obliquetree.src"],  # Explicitly list packages
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
            "nonecheck": False,
            "overflowcheck": False,
        },
    ),
)
