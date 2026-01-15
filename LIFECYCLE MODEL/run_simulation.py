"""
Simple entry point script to run the simulation

This script can be run directly: python modular/run_simulation.py
Or as a module: python -m modular.run_simulation
"""

import sys
import os

# Add parent directory to path so we can import modular
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

if __name__ == "__main__":
    from modular.main import main
    main()

