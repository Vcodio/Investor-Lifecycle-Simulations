"""
Setup script for Lifecycle Retirement Simulation Package
Installs the lifecycle_model package for multiprocessing compatibility
"""
from setuptools import setup
import os


requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
requirements = []
if os.path.exists(requirements_path):
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]


package_dir_with_space = 'LIFECYCLE MODEL'
if os.path.exists(package_dir_with_space):
    setup(
        name='lifecycle-model',
        version='7.0',
        description='Lifecycle Retirement Simulation with GKOS earnings, block bootstrap, and utility calculations',
        packages=['lifecycle_model'],
        package_dir={'lifecycle_model': package_dir_with_space},
        install_requires=requirements,
        python_requires='>=3.7',
        zip_safe=False,
    )
else:
    raise FileNotFoundError(f"Package directory '{package_dir_with_space}' not found")

