"""
setup_cy.py — Build the SensorWF Cython extensions.

Usage
-----
    pip install cython numpy setuptools
    python setup_cy.py build_ext --inplace

Produces
    scripts/_core_cy.<platform>.so   (Linux / macOS)
    scripts/_core_cy.<platform>.pyd  (Windows)

pipeline_core.py imports the compiled module automatically and falls back to
pure-Python equivalents if the .so is absent (e.g. on a new machine before
building).

Compiler flags
--------------
  -O3           aggressive optimisation
  -march=native target CPU instruction set (remove for portable binaries)
  -ffast-math   allow non-IEEE float reordering (safe for anomaly scoring)
"""

from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
except ImportError:
    print(
        "Cython is required to build the extension.\n"
        "Install it with:  pip install cython\n"
        "Then re-run:      python setup_cy.py build_ext --inplace"
    )
    raise SystemExit(1)

ext = Extension(
    name    = "scripts._core_cy",
    sources = ["scripts/_core_cy.pyx"],
    include_dirs = [np.get_include()],
    extra_compile_args = ["-O3", "-ffast-math"],
    language = "c",
)

setup(
    name = "sensorwf_core",
    ext_modules = cythonize(
        [ext],
        compiler_directives = {
            "language_level": "3",
            "boundscheck":    False,
            "wraparound":     False,
            "cdivision":      True,
            "nonecheck":      False,
        },
        annotate = False,
    ),
)
