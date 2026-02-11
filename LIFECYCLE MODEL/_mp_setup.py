"""
Multiprocessing setup script

This module is imported by worker processes to set up the package structure
before any unpickling happens. It must be importable as a top-level module.
"""

import sys
import os
import importlib.util
import importlib


_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_build_dir = os.path.join(_parent_dir, 'build')


if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
if os.path.exists(_build_dir) and _build_dir not in sys.path:
    sys.path.insert(0, _build_dir)


class PackageImportFinder:
    """Import finder that makes dynamically loaded modules importable"""
    def __init__(self, package_dir, pkg_name):
        self.package_dir = package_dir
        self.package_name = pkg_name
    
    def find_spec(self, name, path, target=None):
        if name.startswith(f'{self.package_name}.'):
            module_name = name[len(self.package_name) + 1:]
            module_file = os.path.join(self.package_dir, f'{module_name}.py')
            if os.path.exists(module_file):
                spec = importlib.util.spec_from_file_location(name, module_file)
                if spec:
                    return spec
        return None

package_name = 'lifecycle_model'


finder_installed = any(
    isinstance(f, PackageImportFinder) and f.package_name == package_name
    for f in sys.meta_path
)
if not finder_installed:
    finder = PackageImportFinder(_current_dir, package_name)
    sys.meta_path.insert(0, finder)


if package_name not in sys.modules:
    sys.modules[package_name] = type(sys)(package_name)
    

    init_path = os.path.join(_current_dir, '__init__.py')
    if os.path.exists(init_path):
        try:
            spec = importlib.util.spec_from_file_location(package_name, init_path)
            if spec and spec.loader:
                init_module = importlib.util.module_from_spec(spec)
                sys.modules[package_name] = init_module
                spec.loader.exec_module(init_module)
        except Exception:
            pass
    

    modules_to_load = [
        ('config', 'config.py'),
        ('cython_wrapper', 'cython_wrapper.py'),
        ('bootstrap', 'bootstrap.py'),
        ('earnings', 'earnings.py'),
        ('utils', 'utils.py'),
        ('utility', 'utility.py'),
        ('simulation', 'simulation.py'),
    ]
    
    for mod_name, mod_file in modules_to_load:
        mod_path = os.path.join(_current_dir, mod_file)
        if os.path.exists(mod_path):
            try:
                full_name = f"{package_name}.{mod_name}"
                if full_name not in sys.modules:
                    spec = importlib.util.spec_from_file_location(full_name, mod_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        module.__package__ = package_name
                        module.__name__ = full_name
                        sys.modules[full_name] = module
                        spec.loader.exec_module(module)
            except Exception:
                pass

