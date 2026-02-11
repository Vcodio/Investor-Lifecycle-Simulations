"""
Lifecycle Retirement Simulation Package

This package provides a modular implementation of the lifecycle retirement simulation
with GKOS earnings dynamics, Amortization Based Withdrawal strategy, block bootstrap, and utility based consumption and bequest utility evaluation. 

Config.py to set the simulation parameters.
run_simulation.py to run the simulation.
Cython_wrapper.py to provide the Cython wrapper for the simulation.
"""

import sys
import os
import importlib.util
import importlib



_package_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_package_dir)


if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

_build_dir = os.path.join(_parent_dir, 'build')
if os.path.exists(_build_dir) and _build_dir not in sys.path:
    sys.path.insert(0, _build_dir)


if _package_dir not in sys.path:
    sys.path.insert(0, _package_dir)



class PackageImportFinder:
    """Import finder that makes dynamically loaded modules importable"""
    def __init__(self, package_dir):
        self.package_dir = package_dir
        self.package_name = 'lifecycle_model'
    
    def find_spec(self, name, path, target=None):
        if name.startswith(f'{self.package_name}.'):
            module_name = name[len(self.package_name) + 1:]
            module_file = os.path.join(self.package_dir, f'{module_name}.py')
            if os.path.exists(module_file):
                spec = importlib.util.spec_from_file_location(name, module_file)
                if spec:
                    return spec
        return None


if 'lifecycle_model' not in [f.name for f in sys.meta_path if hasattr(f, 'name')]:
    finder = PackageImportFinder(_package_dir)
    finder.package_name = 'lifecycle_model'
    sys.meta_path.insert(0, finder)



import types
if 'lifecycle_model' not in sys.modules:
    package_module = types.ModuleType('lifecycle_model')
    package_module.__package__ = 'lifecycle_model'
    package_module.__path__ = [_package_dir]
    sys.modules['lifecycle_model'] = package_module

__version__ = "7.1"

