"""
Simple entry point script to run the simulation

This script can be run directly: python "LIFECYCLE MODEL/run_simulation.py"
Or from the parent directory: python -m "LIFECYCLE MODEL.run_simulation"
"""

import sys
import os
import importlib.util

# Get the current directory and parent directory
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)

# Add parent directory to path so we can import the package
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Add build directory for Cython modules
_build_dir = os.path.join(_parent_dir, 'build')
if os.path.exists(_build_dir) and _build_dir not in sys.path:
    sys.path.insert(0, _build_dir)

if __name__ == "__main__":
    # #region agent log
    try:
        import json
        import time
        import os
        log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'debug.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'id': 'log_run_entry', 'timestamp': time.time() * 1000, 'location': 'run_simulation.py:25', 'message': 'run_simulation.py entry point', 'data': {}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'E'}) + '\n')
    except Exception as log_err: pass
    # #endregion
    # CRITICAL: Import the package __init__ FIRST to set up import finder
    # This must happen before any multiprocessing to ensure worker processes can import modules
    import importlib.util
    import importlib
    
    init_path = os.path.join(_current_dir, '__init__.py')
    if os.path.exists(init_path):
        spec = importlib.util.spec_from_file_location("lifecycle_model", init_path)
        if spec and spec.loader:
            init_module = importlib.util.module_from_spec(spec)
            sys.modules['lifecycle_model'] = init_module
            spec.loader.exec_module(init_module)
    
    # Set up package structure for relative imports
    package_name = 'lifecycle_model'
    if package_name not in sys.modules:
        sys.modules[package_name] = type(sys)(package_name)
    
    # Add current directory to Python path so modules can be imported
    # This is critical for multiprocessing workers
    if _current_dir not in sys.path:
        sys.path.insert(0, _current_dir)
    
    # Ensure import finder from __init__.py is active
    # This allows Python's pickle system to import lifecycle_model.* modules
    try:
        import lifecycle_model
        # Force the import finder to be registered
        if hasattr(lifecycle_model, '__file__'):
            # The import finder should already be installed by __init__.py
            pass
    except Exception:
        pass
    
    # Helper function to load a module with proper package setup
    def load_module(mod_name, mod_file):
        """Load a module and set it up for relative imports"""
        mod_path = os.path.join(_current_dir, mod_file)
        if not os.path.exists(mod_path):
            raise FileNotFoundError(f"Module file not found: {mod_path}")
        
        full_name = f"{package_name}.{mod_name}"
        spec = importlib.util.spec_from_file_location(full_name, mod_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {mod_name} from {mod_path}")
        
        module = importlib.util.module_from_spec(spec)
        module.__package__ = package_name
        module.__name__ = full_name
        sys.modules[full_name] = module
        
        # Execute the module (this will trigger relative imports)
        spec.loader.exec_module(module)
        return module
    
    # Load modules in dependency order
    # 1. Base modules (no local dependencies)
    # #region agent log
    try:
        import json
        import time
        import os
        log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'debug.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'id': 'log_before_module_load', 'timestamp': time.time() * 1000, 'location': 'run_simulation.py:82', 'message': 'Starting to load modules', 'data': {}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'A'}) + '\n')
    except Exception as log_err: pass
    # #endregion
    try:
        config_module = load_module('config', 'config.py')
        # #region agent log
        try:
            import json
            import time
            import os
            log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'log_config_loaded', 'timestamp': time.time() * 1000, 'location': 'run_simulation.py:85', 'message': 'config module loaded', 'data': {}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'A'}) + '\n')
        except Exception as log_err: pass
        # #endregion
        cython_module = load_module('cython_wrapper', 'cython_wrapper.py')
        # #region agent log
        try:
            import json
            import time
            import os
            log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'log_cython_loaded', 'timestamp': time.time() * 1000, 'location': 'run_simulation.py:87', 'message': 'cython_wrapper module loaded', 'data': {}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'A'}) + '\n')
        except Exception as log_err: pass
        # #endregion
        bootstrap_module = load_module('bootstrap', 'bootstrap.py')
        utils_module = load_module('utils', 'utils.py')
        utility_module = load_module('utility', 'utility.py')
        
        # 2. Modules that depend on base modules
        earnings_module = load_module('earnings', 'earnings.py')
        simulation_module = load_module('simulation', 'simulation.py')
        visualization_module = load_module('visualization', 'visualization.py')
        
        # 3. Main module (depends on all others)
        main_module = load_module('main', 'main.py')
        # #region agent log
        try:
            import json
            import time
            import os
            log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'log_all_modules_loaded', 'timestamp': time.time() * 1000, 'location': 'run_simulation.py:98', 'message': 'All modules loaded successfully', 'data': {}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'A'}) + '\n')
        except Exception as log_err: pass
        # #endregion
    except Exception as module_err:
        # #region agent log
        try:
            import json
            import time
            import os
            import traceback
            log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'log_module_load_error', 'timestamp': time.time() * 1000, 'location': 'run_simulation.py:99', 'message': 'Exception loading modules', 'data': {'error': str(module_err), 'error_type': type(module_err).__name__, 'traceback': traceback.format_exc()}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'A'}) + '\n')
        except Exception as log_err: pass
        # #endregion
        raise
    
    # Set environment variable for multiprocessing workers
    # This ensures worker processes can find the modules
    os.environ['PYTHONPATH'] = os.pathsep.join([
        _parent_dir,
        _current_dir,
        _build_dir if os.path.exists(_build_dir) else '',
        os.environ.get('PYTHONPATH', '')
    ]).strip(os.pathsep)
    
    # Call the main function with profiling
    import cProfile
    import pstats
    import io
    from datetime import datetime
    
    # Create profiler
    profiler = cProfile.Profile()
    
    try:
        # Start profiling
        profiler.enable()
        main_module.main()
        profiler.disable()
        
        # Generate profiling report
        print("\n" + "="*70)
        print("PROFILING RESULTS")
        print("="*70)
        
        stats = pstats.Stats(profiler)
        
        # Print summary statistics
        total_time = stats.total_tt
        print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Total function calls: {stats.total_calls:,}")
        print(f"Primitive calls: {stats.prim_calls:,}")
        
        # Print top 30 functions by cumulative time (includes subcalls)
        print("\n" + "-"*70)
        print("Top 30 Functions by Cumulative Time (includes subcalls)")
        print("-"*70)
        stats.sort_stats('cumulative')
        stats.print_stats(30)
        
        # Print top 30 functions by total time (excludes subcalls - shows where time is actually spent)
        print("\n" + "-"*70)
        print("Top 30 Functions by Total Time (excludes subcalls - actual work)")
        print("-"*70)
        stats.sort_stats('tottime')
        stats.print_stats(30)
        
        # Print top 30 functions by number of calls
        print("\n" + "-"*70)
        print("Top 30 Functions by Number of Calls")
        print("-"*70)
        stats.sort_stats('ncalls')
        stats.print_stats(30)
        
        # Save detailed profile to file
        profile_filename = f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
        profile_path = os.path.join(_parent_dir, profile_filename)
        profiler.dump_stats(profile_path)
        print("\n" + "-"*70)
        print(f"[INFO] Detailed profile saved to: {profile_path}")
        print(f"       View with: python -m pstats {profile_filename}")
        print(f"       Or use: snakeviz {profile_filename} (if snakeviz is installed)")
        print("="*70)
        
    except Exception as e:
        profiler.disable()
        # #region agent log
        try:
            import json
            import traceback
            import time
            import os
            log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'log_run_error', 'timestamp': time.time() * 1000, 'location': 'run_simulation.py:107', 'message': 'Exception calling main()', 'data': {'error': str(e), 'traceback': traceback.format_exc(), 'error_type': type(e).__name__}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'D'}) + '\n')
        except Exception as log_err: pass
        # #endregion
        raise

