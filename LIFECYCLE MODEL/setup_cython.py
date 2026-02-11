"""
Setup script for building Cython extensions for both Linux and Windows.
Run: python setup_cython.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys
import os


setup_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(setup_dir)


extensions = [
    Extension(
        "lrs_cython",
        sources=[os.path.join(setup_dir, "lrs_cython.pyx")],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"] if sys.platform != "win32" else ["/O2"],
        extra_link_args=["-O3"] if sys.platform != "win32" else ["/O2"],
        language="c"
    ),
    Extension(
        "bates_mle_cython",
        sources=[os.path.join(setup_dir, "bates_mle_cython.pyx")],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"] if sys.platform != "win32" else ["/O2"],
        extra_link_args=["-O3"] if sys.platform != "win32" else ["/O2"],
        language="c"
    ),
    Extension(
        "regime_bates",
        sources=[os.path.join(setup_dir, "regime_bates.pyx")],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"] if sys.platform != "win32" else ["/O2"],
        extra_link_args=["-O3"] if sys.platform != "win32" else ["/O2"],
        language="c"
    ),
]

setup(
    name="lifecycle_cython_extensions",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'language_level': "3"
        }
    ),
    zip_safe=False,
)
