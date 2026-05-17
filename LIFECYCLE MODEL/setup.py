"""
Setup script for lifecycle_model package (for editable install)
This allows the package to be properly importable for multiprocessing
"""
from setuptools import setup

setup(
    name='lifecycle_model',
    version='7.0',
    packages=['lifecycle_model'],
    package_dir={'lifecycle_model': '.'},
    description='Lifecycle Retirement Simulation Package',
    python_requires='>=3.7',
)

