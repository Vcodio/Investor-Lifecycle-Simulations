"""
Core Simulation Module

This module contains the core simulation functions for the lifecycle retirement model,
including withdrawal phase simulations and accumulation phase simulations with GKOS earnings.
"""

import numpy as np
import logging
import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
# Check if running in Streamlit and disable tqdm if so
try:
    import streamlit as st
    _IN_STREAMLIT = True
    # Create a no-op tqdm for Streamlit that works as both function and context manager
    class _NoOpTqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable if iterable is not None else []
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])
        def update(self, *args):
            pass
        def close(self):
            pass
    tqdm = _NoOpTqdm
except ImportError:
    _IN_STREAMLIT = False
    from tqdm import tqdm

# Set up package paths for multiprocessing workers
# This ensures worker processes can import modules
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_build_dir = os.path.join(_parent_dir, 'build')

if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
if os.path.exists(_build_dir) and _build_dir not in sys.path:
    sys.path.insert(0, _build_dir)

# Import __init__ to set up import finder for multiprocessing
# This must happen before any functions are defined that will be pickled
try:
    import importlib.util
    init_path = os.path.join(_current_dir, '__init__.py')
    if os.path.exists(init_path):
        package_name = 'lifecycle_model'
        if package_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(package_name, init_path)
            if spec and spec.loader:
                init_module = importlib.util.module_from_spec(spec)
                sys.modules[package_name] = init_module
                spec.loader.exec_module(init_module)
except Exception:
    pass  # Continue if init fails

from .cython_wrapper import simulate_monthly_return_svj, calculate_max_drawdown
from .bootstrap import create_block_bootstrap_sampler
from .earnings import simulate_earnings_path_with_inflation

logger = logging.getLogger(__name__)


def _init_worker_process():
    """Initialize worker process for multiprocessing - sets up package structure
    
    This runs when each worker process starts. It imports the setup module
    which sets up the package structure so Python's pickle system can import modules.
    """
    # CRITICAL: Set up paths FIRST before trying to import anything
    worker_current_dir = os.path.dirname(os.path.abspath(__file__))
    worker_parent_dir = os.path.dirname(worker_current_dir)
    worker_build_dir = os.path.join(worker_parent_dir, 'build')
    
    if worker_parent_dir not in sys.path:
        sys.path.insert(0, worker_parent_dir)
    if worker_current_dir not in sys.path:
        sys.path.insert(0, worker_current_dir)
    if os.path.exists(worker_build_dir) and worker_build_dir not in sys.path:
        sys.path.insert(0, worker_build_dir)
    
    # Create package stub IMMEDIATELY so pickle can find it
    package_name = 'lifecycle_model'
    if package_name not in sys.modules:
        sys.modules[package_name] = type(sys)(package_name)
        sys.modules[package_name].__package__ = package_name
        sys.modules[package_name].__path__ = [worker_current_dir]
    
    # Create simulation module stub so pickle can find it
    sim_module_name = f'{package_name}.simulation'
    if sim_module_name not in sys.modules:
        sys.modules[sim_module_name] = type(sys)(sim_module_name)
        sys.modules[sim_module_name].__package__ = package_name
        sys.modules[sim_module_name].__name__ = sim_module_name
    
    # Now try to import the setup module which sets up everything
    # This must be importable as a top-level module
    try:
        import _mp_setup
    except ImportError:
        # Fallback: set up manually if _mp_setup can't be imported
        # Package stub is already created above, so this is just for additional setup
        pass


def calculate_expected_return_from_data(monthly_returns, monthly_inflation=None, params=None, mean_inflation=None):
    """
    Calculate expected real return from historical data or model parameters.
    
    Parameters:
    -----------
    monthly_returns : array-like or None
        Historical monthly returns (if using bootstrap)
    monthly_inflation : array-like or None
        Historical monthly inflation (if using bootstrap)
    params : dict or None
        Model parameters dict with 'mu' key (if using parametric model)
    mean_inflation : float or None
        Expected inflation rate (annual, geometric mean)
        
    Returns:
    --------
    expected_real_return : float
        Expected real return (annual, as decimal)
    """
    if monthly_returns is not None and len(monthly_returns) > 0:
        # Calculate from historical data
        # Convert monthly returns to annual (geometric mean)
        # Annual return = (1 + r1) * (1 + r2) * ... * (1 + r12) - 1 for one year
        # Geometric mean monthly = (prod(1 + r))^(1/n) - 1
        # Annual = (geometric_mean_monthly + 1)^12 - 1
        
        # Calculate geometric mean of monthly returns
        monthly_returns = np.array(monthly_returns)
        # Handle any negative returns that would cause issues
        monthly_returns = np.clip(monthly_returns, -0.99, np.inf)
        geometric_mean_monthly = np.exp(np.mean(np.log(1.0 + monthly_returns))) - 1.0
        annual_nominal_return = (1.0 + geometric_mean_monthly) ** 12 - 1.0
        
        if monthly_inflation is not None and len(monthly_inflation) > 0:
            # Calculate real return: (1 + nominal) / (1 + inflation) - 1
            monthly_inflation = np.array(monthly_inflation)
            monthly_inflation = np.clip(monthly_inflation, -0.99, np.inf)
            geometric_mean_monthly_inflation = np.exp(np.mean(np.log(1.0 + monthly_inflation))) - 1.0
            annual_inflation = (1.0 + geometric_mean_monthly_inflation) ** 12 - 1.0
            expected_real_return = (1.0 + annual_nominal_return) / (1.0 + annual_inflation) - 1.0
        else:
            # Use provided mean_inflation if available
            if mean_inflation is not None:
                expected_real_return = (1.0 + annual_nominal_return) / (1.0 + mean_inflation) - 1.0
            else:
                # Fallback: assume real return ≈ nominal - 2.5% inflation
                expected_real_return = annual_nominal_return - 0.025
        
        logger.info(f"[AMORTIZATION] Calculated expected real return from historical data: "
                   f"Nominal annual return: {annual_nominal_return*100:.2f}%, "
                   f"Inflation: {annual_inflation*100:.2f}%, "
                   f"Real return: {expected_real_return*100:.2f}%")
        return expected_real_return
    
    elif params is not None and 'mu' in params:
        # Calculate from model parameters
        # mu is the expected return (annual, nominal)
        mu = params['mu']
        
        if mean_inflation is not None:
            # Real return = (1 + mu) / (1 + inflation) - 1
            expected_real_return = (1.0 + mu) / (1.0 + mean_inflation) - 1.0
        else:
            # Fallback: approximate real return = mu - 2.5%
            expected_real_return = mu - 0.025
        
        logger.info(f"[AMORTIZATION] Calculated expected real return from model parameters: "
                   f"mu (nominal): {mu*100:.2f}%, "
                   f"Inflation: {mean_inflation*100:.2f}%, "
                   f"Real return: {expected_real_return*100:.2f}%")
        return expected_real_return
    
    else:
        # Fallback: use conservative default
        logger.warning("[AMORTIZATION] No data available for expected return calculation, using default 3% real return")
        return 0.03


def calculate_amortized_withdrawal(principal, remaining_years, real_rate, desired_bequest=0.0):
    """
    Calculate annual withdrawal amount using fixed amortization formula.
    
    Formula (without bequest): W = P * [r(1+r)^n] / [(1+r)^n - 1]
    Formula (with bequest B): W = (P - B/(1+r)^n) * [r(1+r)^n] / [(1+r)^n - 1]
    Where:
        W = annual withdrawal amount
        P = current principal
        r = real rate of return (annual)
        n = remaining years
        B = desired bequest at death
    
    Parameters:
    -----------
    principal : float
        Current portfolio value (real or nominal, depending on rate)
    remaining_years : float
        Number of years remaining in retirement
    real_rate : float
        Fixed real rate of return (annual, as decimal, e.g., 0.03 for 3%)
    desired_bequest : float, optional
        Desired portfolio value at death (default 0.0, meaning consume to zero)
        
    Returns:
    --------
    annual_withdrawal : float
        Annual withdrawal amount in same units as principal
    """
    if remaining_years <= 0:
        # No years remaining - withdraw everything except desired bequest
        return max(0.0, principal - desired_bequest)
    
    if remaining_years <= 1:
        # One year or less remaining - withdraw principal minus present value of bequest
        # Present value of bequest: B / (1+r)
        pv_bequest = desired_bequest / (1.0 + real_rate) if remaining_years > 0 else desired_bequest
        available_principal = max(0.0, principal - pv_bequest)
        return available_principal / max(remaining_years, 0.01)
    
    # Modified amortization formula with desired bequest
    # Effective principal = P - PV(B), where PV(B) = B / (1+r)^n
    r = real_rate
    n = remaining_years
    
    # Calculate (1+r)^n
    one_plus_r_to_n = (1.0 + r) ** n
    
    # Calculate present value of desired bequest
    pv_bequest = desired_bequest / one_plus_r_to_n
    
    # Effective principal (principal minus present value of bequest)
    effective_principal = max(0.0, principal - pv_bequest)
    
    # If effective principal is zero or negative, no withdrawal
    if effective_principal <= 0:
        return 0.0
    
    # Calculate numerator: r * (1+r)^n
    numerator = r * one_plus_r_to_n
    
    # Calculate denominator: (1+r)^n - 1
    denominator = one_plus_r_to_n - 1.0
    
    # Handle edge case where denominator is very close to zero (shouldn't happen for n > 1)
    if abs(denominator) < 1e-10:
        # Fallback: use simple division
        return effective_principal / n
    
    # Calculate annual withdrawal using effective principal
    annual_withdrawal = effective_principal * (numerator / denominator)
    
    return annual_withdrawal


def simulate_withdrawals(start_portfolio, start_age, rng_local, params_annual,
                         spending_annual_real, include_social_security,
                         social_security_real, social_security_start_age,
                         config, bootstrap_sampler=None):
    """
    Simulate withdrawal phase (retirement).
    
    Returns:
        tuple: (success, mean_growth, std_dev, max_drawdown, portfolio_history)
    """
    portfolio = float(start_portfolio)
    age_in_months = int(start_age * 12)
    # Cache integer month values to avoid division in hot loops
    start_age_months = age_in_months
    death_age_months = int(config.death_age * 12)
    social_security_start_age_months = int(social_security_start_age * 12) if include_social_security else 0
    # Pre-allocate array for better performance (estimate max months)
    max_months = death_age_months - start_age_months + 1
    portfolio_history = np.empty(max_months, dtype=np.float64)
    portfolio_history[0] = portfolio
    history_idx = 1
    
    current_variance = params_annual["v0"]

    current_annual_spending_nominal = spending_annual_real
    current_monthly_spending_nominal = current_annual_spending_nominal / 12.0
    current_social_security_nominal = social_security_real
    current_monthly_social_security_nominal = current_social_security_nominal / 12.0

    # Pre-sample all returns and inflation if using block bootstrap
    use_bootstrap = False
    if config.use_block_bootstrap and bootstrap_sampler is not None:
        try:
            total_months = int((config.death_age - start_age) * 12)
            bootstrap_returns, bootstrap_inflation = bootstrap_sampler.sample_sequence(total_months)
            bootstrap_month_idx = 0
            use_bootstrap = True
        except Exception as e:
            logger.warning(f"Block bootstrap failed, falling back to parametric model: {e}")
            bootstrap_returns = None
            bootstrap_inflation = None
            bootstrap_month_idx = None
            use_bootstrap = False
    else:
        bootstrap_returns = None
        bootstrap_inflation = None
        bootstrap_month_idx = None
        use_bootstrap = False

    while age_in_months < death_age_months:
        if (age_in_months % 12) == 0 and age_in_months > start_age_months:
            if config.use_block_bootstrap:
                # When block bootstrap is enabled, inflation MUST come from CSV/bootstrap data
                if use_bootstrap and bootstrap_inflation is not None:
                    year_start_idx = max(0, bootstrap_month_idx - 12)
                    year_end_idx = bootstrap_month_idx
                    if year_end_idx > year_start_idx and year_end_idx - year_start_idx == 12:
                        monthly_inflations = bootstrap_inflation[year_start_idx:year_end_idx]
                        annual_inflation = np.prod(1.0 + monthly_inflations) - 1.0
                    else:
                        # If we don't have exactly 12 months yet, use what we have or raise error
                        # This should only happen at the very beginning, use available months
                        available_months = bootstrap_inflation[:bootstrap_month_idx] if bootstrap_month_idx > 0 else bootstrap_inflation[:12]
                        if len(available_months) > 0:
                            annual_inflation = np.prod(1.0 + available_months) - 1.0
                        else:
                            logger.error(f"Block bootstrap enabled but no inflation data available at month {bootstrap_month_idx}")
                            raise ValueError("Block bootstrap enabled but inflation data not available")
                else:
                    logger.error("Block bootstrap enabled but bootstrap data not available")
                    raise ValueError("Block bootstrap enabled but bootstrap data not available")
            else:
                # When block bootstrap is NOT enabled, use config values
                annual_inflation = rng_local.normal(config.mean_inflation_geometric,
                                                   config.std_inflation)
            annual_inflation = max(annual_inflation, -0.99)

            current_annual_spending_nominal *= (1.0 + annual_inflation)
            current_monthly_spending_nominal = current_annual_spending_nominal / 12.0
            current_social_security_nominal *= (1.0 + annual_inflation)
            current_monthly_social_security_nominal = current_social_security_nominal / 12.0

        net_withdrawal_nominal = current_monthly_spending_nominal
        if include_social_security and age_in_months >= social_security_start_age_months:
            net_withdrawal_nominal = max(0.0, net_withdrawal_nominal -
                                        current_monthly_social_security_nominal)

        portfolio -= net_withdrawal_nominal
        if portfolio <= 0:
            return False, 0.0, 0.0, 0.0, []

        # Get market return from bootstrap or parametric model
        if use_bootstrap and bootstrap_returns is not None and bootstrap_month_idx < len(bootstrap_returns):
            market_return = bootstrap_returns[bootstrap_month_idx]
            bootstrap_month_idx += 1
        else:
            market_return, current_variance = simulate_monthly_return_svj(
                rng_local, params_annual, current_variance)
        
        portfolio *= (1.0 + market_return)
        if history_idx < max_months:
            portfolio_history[history_idx] = portfolio
            history_idx += 1

        age_in_months += 1

    # Trim to actual size and calculate statistics
    portfolio_history = portfolio_history[:history_idx]
    
    # Vectorized calculation for better performance (already a numpy array)
    if len(portfolio_history) > 1:
        growth_rates = np.diff(portfolio_history) / portfolio_history[:-1]
        mean_growth = np.mean(growth_rates)
        std_dev = np.std(growth_rates)
    else:
        mean_growth = 0.0
        std_dev = 0.0
    max_drawdown = calculate_max_drawdown(portfolio_history)
    
    # #region agent log
    t_end = time.perf_counter()
    try:
        import json
        log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'debug.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'id': 'log_sim_withdrawals_time', 'timestamp': time.time() * 1000, 'location': 'simulation.py:333', 'message': 'simulate_withdrawals timing', 'data': {'duration_ms': (t_end - t_start) * 1000, 'months': history_idx}, 'sessionId': 'debug-session', 'runId': 'profile', 'hypothesisId': 'PERF'}) + '\n')
    except: pass
    # #endregion

    return True, mean_growth, std_dev, max_drawdown, portfolio_history


def check_success_rate_worker(principal, retirement_age, num_sims, seed_offset,
                              config, params, bootstrap_data=None):
    """Worker function for parallel success rate checking"""
    
    if config.seed is None:
        nested_rng = np.random.default_rng()
    else:
        nested_rng = np.random.default_rng(seed=(config.seed + seed_offset + 1))

    bootstrap_sampler = None
    if config.use_block_bootstrap:
        try:
            monthly_returns = None
            monthly_inflation = None
            if bootstrap_data is not None:
                monthly_returns, monthly_inflation = bootstrap_data
            bootstrap_sampler = create_block_bootstrap_sampler(
                config, nested_rng, monthly_returns, monthly_inflation)
        except Exception as e:
            logger.error(f"Worker {seed_offset} failed to create bootstrap sampler: {e}")
            bootstrap_sampler = None

    successes = 0
    metrics = {'mean_growth': [], 'std_dev': [], 'max_drawdown': []}

    for _ in range(num_sims):
        is_success, mg, sd, mdd, _ = simulate_withdrawals(
            principal, retirement_age, nested_rng, params, config.spending_real,
            config.include_social_security, config.social_security_real,
            config.social_security_start_age, config, bootstrap_sampler
        )
        if is_success:
            successes += 1
            metrics['mean_growth'].append(mg)
            metrics['std_dev'].append(sd)
            metrics['max_drawdown'].append(mdd)

    return {'successes': successes, 'metrics': metrics}


def check_success_rate(principal, retirement_age, num_nested_sims, config, params, bootstrap_data=None):
    """Check success rate for a given principal and retirement age"""
    # Skip multiprocessing entirely if disabled (avoids setup overhead)
    if config.num_workers <= 1 or num_nested_sims < 100:
        res = check_success_rate_worker(principal, retirement_age, num_nested_sims,
                                       0, config, params, bootstrap_data)
        success_rate = res['successes'] / max(1, num_nested_sims)
        combined_metrics = {k: np.array(v) for k, v in res['metrics'].items()}
        return success_rate, combined_metrics
    
    sims_per_worker = num_nested_sims // config.num_workers
    remaining = num_nested_sims % config.num_workers
    futures = []

    # CRITICAL: On Windows, multiprocessing uses 'spawn' which starts fresh Python processes.
    # When unpickling functions, Python needs to import lifecycle_model.simulation BEFORE
    # any code runs. We patch multiprocessing.spawn._main to run setup code before unpickling.
    
    import multiprocessing
    import multiprocessing.spawn as mp_spawn
    import multiprocessing.reduction as mp_reduction
    
    # CRITICAL FIX: Ensure PYTHONPATH includes our directories so child processes can import
    pythonpath = os.environ.get('PYTHONPATH', '')
    pythonpath_parts = pythonpath.split(os.pathsep) if pythonpath else []
    if _parent_dir not in pythonpath_parts:
        pythonpath_parts.insert(0, _parent_dir)
    if _current_dir not in pythonpath_parts:
        pythonpath_parts.insert(0, _current_dir)
    if os.path.exists(_build_dir) and _build_dir not in pythonpath_parts:
        pythonpath_parts.insert(0, _build_dir)
    os.environ['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)
    
    # Clear debug file at start
    debug_file = os.path.join(_parent_dir, 'mp_debug.log')
    try:
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(f"=== MAIN PROCESS START ===\\n")
            f.write(f"Setting up multiprocessing with paths:\\n")
            f.write(f"  _parent_dir: {_parent_dir}\\n")
            f.write(f"  _current_dir: {_current_dir}\\n")
            f.write(f"  _build_dir: {_build_dir}\\n")
    except Exception:
        pass
    
    # Set PYTHONPATH for worker processes so _mp_setup.py can be found
    original_pythonpath = os.environ.get('PYTHONPATH', '')
    pythonpath_parts = []
    if _parent_dir not in pythonpath_parts:
        pythonpath_parts.append(_parent_dir)
    if _current_dir not in pythonpath_parts:
        pythonpath_parts.append(_current_dir)
    if os.path.exists(_build_dir) and _build_dir not in pythonpath_parts:
        pythonpath_parts.append(_build_dir)
    if original_pythonpath:
        pythonpath_parts.append(original_pythonpath)
    os.environ['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)
    
    # Create setup code that will run in child process before unpickling
    # Escape backslashes for Windows paths in the f-string
    current_dir_escaped = _current_dir.replace('\\', '\\\\')
    parent_dir_escaped = _parent_dir.replace('\\', '\\\\')
    build_dir_escaped = _build_dir.replace('\\', '\\\\')
    
    # CRITICAL: This code MUST run in child process BEFORE any unpickling
    # We patch __import__ and pickle.Unpickler.find_class IMMEDIATELY
    # Create setup code with detailed debugging
    # This creates stub modules in sys.modules so pickle can find them
    debug_file = os.path.join(_parent_dir, 'mp_debug.log')
    setup_code_str = f'''
# THIS CODE RUNS IN CHILD PROCESS - MUST RUN BEFORE unpickling
import sys
import os
import traceback

# Debug logging
debug_file = r"{debug_file.replace(chr(92), chr(92)+chr(92))}"
try:
    with open(debug_file, 'a', encoding='utf-8') as f:
        f.write("=== CHILD PROCESS START ===\\n")
        f.write(f"Python: {{sys.version}}\\n")
        f.write(f"sys.path: {{sys.path[:3]}}\\n")
        f.write(f"sys.modules has lifecycle_model: {{'lifecycle_model' in sys.modules}}\\n")
except:
    pass

# CRITICAL: Create package and module stubs FIRST, before anything else
# This ensures pickle can find them even if the rest of the code fails
package_name = 'lifecycle_model'
try:
    if package_name not in sys.modules:
        sys.modules[package_name] = type(sys)(package_name)
        sys.modules[package_name].__package__ = package_name
        sys.modules[package_name].__path__ = [r"{current_dir_escaped}"]
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write(f"Created package: {{package_name}}\\n")
    else:
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write(f"Package already exists: {{package_name}}\\n")
except Exception as e:
    with open(debug_file, 'a', encoding='utf-8') as f:
        f.write(f"ERROR creating package: {{e}}\\n")
        f.write(traceback.format_exc() + "\\n")

sim_module_name = package_name + '.simulation'
try:
    if sim_module_name not in sys.modules:
        sys.modules[sim_module_name] = type(sys)(sim_module_name)
        sys.modules[sim_module_name].__package__ = package_name
        sys.modules[sim_module_name].__name__ = sim_module_name
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write(f"Created module stub: {{sim_module_name}}\\n")
except Exception as e:
    with open(debug_file, 'a', encoding='utf-8') as f:
        f.write(f"ERROR creating module stub: {{e}}\\n")
        f.write(traceback.format_exc() + "\\n")

# Now set up paths
setup_dir = r"{current_dir_escaped}"
setup_parent = r"{parent_dir_escaped}"
setup_build = r"{build_dir_escaped}"

if setup_parent not in sys.path:
    sys.path.insert(0, setup_parent)
if setup_dir not in sys.path:
    sys.path.insert(0, setup_dir)
if os.path.exists(setup_build) and setup_build not in sys.path:
    sys.path.insert(0, setup_build)

# Import modules AFTER paths and stubs are set
import importlib.util
import builtins
import pickle

# Set up import finder
class PackageImportFinder:
    def __init__(self, package_dir, pkg_name):
        self.package_dir = package_dir
        self.package_name = pkg_name
    
    def find_spec(self, name, path, target=None):
        if name.startswith(self.package_name + '.'):
            module_name = name[len(self.package_name) + 1:]
            module_file = os.path.join(self.package_dir, module_name + '.py')
            if os.path.exists(module_file):
                spec = importlib.util.spec_from_file_location(name, module_file)
                if spec:
                    return spec
        return None

# Install import finder
if not any(isinstance(f, PackageImportFinder) and getattr(f, 'package_name', None) == package_name for f in sys.meta_path):
    finder = PackageImportFinder(setup_dir, package_name)
    sys.meta_path.insert(0, finder)

# PATCH __import__ to handle lifecycle_model imports
_original_import = builtins.__import__
def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.startswith('lifecycle_model'):
        if name not in sys.modules:
            try:
                return _original_import(name, globals, locals, fromlist, level)
            except ImportError:
                if sys.meta_path and isinstance(sys.meta_path[0], PackageImportFinder):
                    try:
                        spec = sys.meta_path[0].find_spec(name)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            module.__package__ = package_name
                            module.__name__ = name
                            sys.modules[name] = module
                            spec.loader.exec_module(module)
                            return module
                    except:
                        pass
                raise
    return _original_import(name, globals, locals, fromlist, level)
builtins.__import__ = _patched_import

# PATCH pickle.Unpickler.find_class
try:
    _original_find_class = pickle.Unpickler.find_class
    def _find_class_with_setup(self, module, name):
        if module.startswith('lifecycle_model'):
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write(f"pickle.find_class called for: {{module}}.{{name}}\\n")
            if module not in sys.modules:
                try:
                    __import__(module, globals(), locals(), [], 0)
                    with open(debug_file, 'a', encoding='utf-8') as f:
                        f.write(f"Successfully imported: {{module}}\\n")
                except Exception as e:
                    with open(debug_file, 'a', encoding='utf-8') as f:
                        f.write(f"Failed to import {{module}}: {{e}}\\n")
        try:
            return _original_find_class(self, module, name)
        except Exception as e:
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write(f"ERROR in _original_find_class: {{e}}\\n")
                f.write(traceback.format_exc() + "\\n")
            raise
    pickle.Unpickler.find_class = _find_class_with_setup
    with open(debug_file, 'a', encoding='utf-8') as f:
        f.write("Patched pickle.Unpickler.find_class\\n")
except Exception as e:
    with open(debug_file, 'a', encoding='utf-8') as f:
        f.write(f"ERROR patching pickle: {{e}}\\n")
        f.write(traceback.format_exc() + "\\n")

try:
    with open(debug_file, 'a', encoding='utf-8') as f:
        f.write(f"Setup complete. sys.modules has: {{', '.join([m for m in sys.modules.keys() if 'lifecycle' in m])}}\\n")
        f.write("=== CHILD PROCESS SETUP DONE ===\\n")
except:
    pass
'''
    
    # Save original get_preparation_data
    _original_get_preparation_data = mp_spawn.get_preparation_data
    
    def _get_preparation_data_with_setup(name):
        """Inject setup code that runs in child before unpickling"""
        prep_data = _original_get_preparation_data(name)
        
        # CRITICAL: Add paths to sys_path so they're available in child BEFORE unpickling
        # sys_path is set in prepare() which runs before the main pickle.load()
        sys_path = prep_data.get('sys_path', [])
        if _parent_dir not in sys_path:
            sys_path.insert(0, _parent_dir)
        if _current_dir not in sys_path:
            sys_path.insert(0, _current_dir)
        if os.path.exists(_build_dir) and _build_dir not in sys_path:
            sys_path.insert(0, _build_dir)
        prep_data['sys_path'] = sys_path
        
        try:
            debug_file = os.path.join(_parent_dir, 'mp_debug.log')
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write(f"_get_preparation_data_with_setup called for: {name}\n")
                f.write(f"Added paths to sys_path: {sys_path[:3]}\n")
        except:
            pass
        
        return prep_data
    
    # CRITICAL: The setup code needs to run in the child process BEFORE prepare() or pickle.load()
    # Since we can't patch functions in the child (it starts fresh), we inject code into
    # the preparation data that will patch _main() itself to run our setup FIRST
    # This is done by creating a custom prepare() that runs setup code before calling original prepare()
    
    # CRITICAL: Our patched prepare() won't exist in the child process (it starts fresh)
    # So we need to ensure the setup code executes when multiprocessing.spawn is imported in child
    # The setup code patches pickle.Unpickler.find_class to ensure the package can be imported
    # But it needs to execute BEFORE pickle tries to import the module
    # Since prepare() doesn't execute _mp_setup_code, we need a different mechanism
    
    # SOLUTION: Execute setup code when multiprocessing.spawn is imported in child
    # We do this by modifying sys_path to include a path that, when Python starts, imports
    # a module that sets up the package. But that's complex.
    
    # ACTUALLY: The simplest solution is to ensure the package exists in sys.modules
    # BEFORE pickle tries to import it. We can do this by patching pickle.Unpickler.find_class
    # to create the package stub if it doesn't exist. But that patch needs to exist when
    # pickle is imported, which happens very early in the child process.
    
    # The setup code patches pickle.Unpickler.find_class, but it needs to execute FIRST
    # Since prepare() doesn't execute _mp_setup_code, we need to ensure it runs another way
    
    # For now, let's patch prepare() in parent - it won't help, but at least we try
    # DON'T patch prepare() - patches don't exist in child process
    # Instead, ensure __init__.py is imported via initializer which runs early
    
    # Setup code will execute when prepare() is called
    # Since prepare() runs BEFORE the second pickle.load() (where we need the package),
    # this should be sufficient
    setup_code_with_patch = setup_code_str + '''

# Setup code complete - package should now be ready for unpickling
'''
    
    # Don't patch prepare() - it won't work because child starts fresh
    
    # CRITICAL FIX: On Windows spawn, unpickling happens BEFORE initializer runs
    # We need to patch pickle.Unpickler.find_class to create package stubs on demand
    # The simplest way: monkey-patch multiprocessing.spawn._main to set up package before unpickling
    
    # Save original spawn._main
    original_spawn_main = mp_spawn._main
    
    def patched_spawn_main(fd, parent_sentinel):
        """Patched _main that sets up package BEFORE unpickling"""
        import sys
        import os
        import types
        
        # Set up paths first (use parent's paths from environment)
        parent_dir = os.environ.get('LIFECYCLE_PARENT_DIR', _parent_dir)
        current_dir = os.environ.get('LIFECYCLE_CURRENT_DIR', _current_dir)
        build_dir = os.environ.get('LIFECYCLE_BUILD_DIR', _build_dir)
        
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        if os.path.exists(build_dir) and build_dir not in sys.path:
            sys.path.insert(0, build_dir)
        
        # CRITICAL: Import _mp_setup to set up package structure
        # This must happen BEFORE pickle.load() tries to import anything
        try:
            import _mp_setup
        except ImportError:
            # Fallback: create package stubs manually
            package_name = 'lifecycle_model'
            if package_name not in sys.modules:
                sys.modules[package_name] = types.ModuleType(package_name)
                sys.modules[package_name].__package__ = package_name
                sys.modules[package_name].__path__ = [current_dir]
            
            sim_module_name = f'{package_name}.simulation'
            if sim_module_name not in sys.modules:
                sys.modules[sim_module_name] = types.ModuleType(sim_module_name)
                sys.modules[sim_module_name].__package__ = package_name
                sys.modules[sim_module_name].__name__ = sim_module_name
        
        # Now call original _main which will unpickle (and pickle will find our stubs)
        return original_spawn_main(fd, parent_sentinel)
    
    # Patch it
    mp_spawn._main = patched_spawn_main
    
    # Set environment variables so child processes know the paths
    os.environ['LIFECYCLE_PARENT_DIR'] = _parent_dir
    os.environ['LIFECYCLE_CURRENT_DIR'] = _current_dir
    os.environ['LIFECYCLE_BUILD_DIR'] = _build_dir
    
    # Patch get_preparation_data to inject paths
    mp_spawn.get_preparation_data = _get_preparation_data_with_setup
    
    # Multiprocessing setup (only reached if num_workers > 1)
    use_multiprocessing = True
    if use_multiprocessing:
        try:
            with ProcessPoolExecutor(max_workers=config.num_workers, initializer=_init_worker_process) as executor:
                for i in range(config.num_workers):
                    sims_this_worker = sims_per_worker + (1 if i < remaining else 0)
                    if sims_this_worker > 0:
                        futures.append(executor.submit(check_success_rate_worker,
                                                         principal, retirement_age,
                                                         sims_this_worker, i, config, params, None))

                results = [f.result() for f in futures]
        except (BrokenProcessPool, ModuleNotFoundError, ImportError) as e:
            # Fallback to single process if multiprocessing fails (e.g., Windows spawn issues)
            logger.warning(f"Multiprocessing failed ({e}), falling back to single process")
            use_multiprocessing = False
            futures = []
    else:
        use_multiprocessing = False
    
    if not use_multiprocessing:
        # Run sequentially
        res = check_success_rate_worker(principal, retirement_age, num_nested_sims,
                                       0, config, params, bootstrap_data)
        success_rate = res['successes'] / max(1, num_nested_sims)
        combined_metrics = {k: np.array(v) for k, v in res['metrics'].items()}
        # Clean up environment variables
        os.environ.pop('LIFECYCLE_PARENT_DIR', None)
        os.environ.pop('LIFECYCLE_CURRENT_DIR', None)
        os.environ.pop('LIFECYCLE_BUILD_DIR', None)
        # #region agent log
        t_check_end = time.perf_counter()
        try:
            import json
            log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'log_check_success_time', 'timestamp': time.time() * 1000, 'location': 'simulation.py:809', 'message': 'check_success_rate timing (single process)', 'data': {'duration_ms': (t_check_end - t_check_start) * 1000, 'num_nested_sims': num_nested_sims, 'num_workers': 1}, 'sessionId': 'debug-session', 'runId': 'profile', 'hypothesisId': 'PERF'}) + '\n')
        except: pass
        # #endregion
        return success_rate, combined_metrics
    
    # Process multiprocessing results
    try:
        total_successes = sum(r['successes'] for r in results)
        all_mg = [np.array(r['metrics']['mean_growth']) for r in results
                  if r['metrics']['mean_growth']]
        all_sd = [np.array(r['metrics']['std_dev']) for r in results
                  if r['metrics']['std_dev']]
        all_mdd = [np.array(r['metrics']['max_drawdown']) for r in results
                   if r['metrics']['max_drawdown']]

        combined_metrics = {
            'mean_growth': np.concatenate(all_mg) if all_mg else np.array([]),
            'std_dev': np.concatenate(all_sd) if all_sd else np.array([]),
            'max_drawdown': np.concatenate(all_mdd) if all_mdd else np.array([]),
        }

        success_rate = total_successes / max(1, num_nested_sims)
        # #region agent log
        t_check_end = time.perf_counter()
        try:
            import json
            log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'log_check_success_time', 'timestamp': time.time() * 1000, 'location': 'simulation.py:864', 'message': 'check_success_rate timing (multiprocess)', 'data': {'duration_ms': (t_check_end - t_check_start) * 1000, 'num_nested_sims': num_nested_sims, 'num_workers': config.num_workers}, 'sessionId': 'debug-session', 'runId': 'profile', 'hypothesisId': 'PERF'}) + '\n')
        except: pass
        # #endregion
        return success_rate, combined_metrics
    finally:
        # Restore original functions
        mp_spawn.get_preparation_data = _original_get_preparation_data
        mp_spawn._main = original_spawn_main
        # Clean up environment variables
        os.environ.pop('LIFECYCLE_PARENT_DIR', None)
        os.environ.pop('LIFECYCLE_CURRENT_DIR', None)
        os.environ.pop('LIFECYCLE_BUILD_DIR', None)


def find_required_principal(target_age, success_target, num_nested_sims, config, params,
                            warm_start_principal=None, bootstrap_data=None):
    """Find required principal with optional warm start from previous age"""
    if warm_start_principal is not None and warm_start_principal > 0:
        search_range_factor = 0.5
        low_principal = max(10_000.0, warm_start_principal * (1 - search_range_factor))
        high_principal = min(20_000_000.0, warm_start_principal * (1 + search_range_factor))
        
        min_range = warm_start_principal * 0.2
        current_range = high_principal - low_principal
        if current_range < min_range:
            center = (low_principal + high_principal) / 2.0
            low_principal = max(10_000.0, center - min_range / 2.0)
            high_principal = min(20_000_000.0, center + min_range / 2.0)
    else:
        low_principal = 10_000.0
        high_principal = 20_000_000.0
    
    tolerance = 1000.0
    principal_cache = {}
    max_iterations = 30
    
    iteration = 0
    while high_principal - low_principal > tolerance and iteration < max_iterations:
        mid_principal = (low_principal + high_principal) / 2.0
        cache_key = round(mid_principal, 2)

        if cache_key in principal_cache:
            success_rate = principal_cache[cache_key]
        else:
            success_rate, _ = check_success_rate(mid_principal, target_age,
                                                 num_nested_sims, config, params, bootstrap_data)
            principal_cache[cache_key] = success_rate

        if success_rate >= success_target:
            high_principal = mid_principal
        else:
            low_principal = mid_principal
        
        iteration += 1

    return high_principal


def create_simulation_record(sim, age_in_months, is_retired, portfolio,
                            current_variance, principal_lookup, current_age_years,
                            current_monthly_spending_nominal, current_annual_ss_nominal,
                            annual_inflation_draw, cumulative_inflation_since_start,
                            savings_rate_for_month, current_monthly_income_real,
                            dollars_saved_nominal, market_return,
                            portfolio_growth_factor, config):
    """Create a record for a simulation time step"""
    return {
        'SIM_ID': sim,
        'AGE': age_in_months / 12.0,
        'RETIRED?': is_retired,
        'PORTFOLIO_VALUE': portfolio,
        'VOLATILITY': current_variance,
        'REQUIRED_REAL_PRINCIPAL': principal_lookup.get(
            current_age_years, {}).get('principal_real', np.nan),
        'WITHDRAWAL_RATE': principal_lookup.get(
            current_age_years, {}).get('swr', np.nan),
        'REQUIRED_NOMINAL_PRINCIPAL': principal_lookup.get(
            current_age_years, {}).get('principal_nominal', np.nan),
        'NOMINAL_DESIRED_CONSUMPTION': current_monthly_spending_nominal * 12.0,
        'REAL_DESIRED_CONSUMPTION': config.spending_real,
        'ANNUAL_INFLATION': (annual_inflation_draw
                            if (age_in_months % 12) == 0 else np.nan),
        'CUMULATIVE_INFLATION': cumulative_inflation_since_start,
        'REAL_SOCIAL_SECURITY_BENEFIT': (
            config.social_security_real
            if is_retired and current_age_years >= config.social_security_start_age
            else np.nan),
        'NOMINAL_SOCIAL_SECURITY_BENEFIT': (
            current_annual_ss_nominal
            if is_retired and current_age_years >= config.social_security_start_age
            else np.nan),
        'SAVINGS_RATE': savings_rate_for_month * 12.0 if not is_retired else np.nan,
        'SALARY_REAL': (current_monthly_income_real * 12.0
                       if not is_retired else np.nan),
        'SALARY_NOMINAL': (
            (current_monthly_income_real * cumulative_inflation_since_start) * 12.0
            if not is_retired else np.nan),
        'DOLLARS_SAVED': dollars_saved_nominal * 12.0 if not is_retired else np.nan,
        'MONTHLY_PORTFOLIO_RETURN': market_return,
        'CUMULATIVE_PORTFOLIO_RETURN': portfolio_growth_factor - 1.0,
    }


def run_single_accumulation_simulation(sim, config, params, principal_lookup, rng,
                                       savings_rate, bootstrap_data=None):
    """
    Run a single accumulation simulation with GKOS earnings model.
    
    Note: Unemployment has been removed - earnings are now modeled using GKOS dynamics.
    """
    portfolio = float(config.initial_portfolio)
    age_in_months = int(config.initial_age * 12)
    retirement_age = np.nan
    is_retired = False
    portfolio_growth_factor = 1.0
    current_sim_record = []
    current_variance = params["v0"]
    cumulative_inflation_since_start = 1.0
    annual_inflation_draw = 0.0

    # Initialize spending variables
    if config.use_amortization:
        # For amortization, we'll calculate withdrawal at start of each retirement year
        # Initialize with a placeholder (will be calculated when retirement starts)
        current_annual_spending_nominal = config.spending_real  # Initial placeholder
    else:
        # Fixed real spending (traditional method)
        current_annual_spending_nominal = config.spending_real
    current_monthly_spending_nominal = current_annual_spending_nominal / 12.0
    current_annual_ss_nominal = config.social_security_real
    current_monthly_ss_nominal = current_annual_ss_nominal / 12.0
    
    # Track amortization statistics
    amortization_stats = {
        'initial_spending_real': config.spending_real,  # For threshold comparison
        'withdrawals': [],  # Annual withdrawal amounts (real)
        'withdrawals_nominal': [],  # Annual withdrawal amounts (nominal)
        'principal_at_year_start': [],  # Principal at start of each retirement year
        'remaining_years': [],  # Remaining years at start of each retirement year
        'below_threshold_count': 0,  # Count of years below minimum threshold
        'total_years': 0,  # Total retirement years
    }

    # Pre-sample inflation rates for GKOS earnings model
    total_years = int(config.death_age - config.initial_age)
    annual_inflation_rates = []
    
    # Pre-sample all returns and inflation if using block bootstrap
    use_bootstrap = False
    if config.use_block_bootstrap:
        try:
            monthly_returns = None
            monthly_inflation = None
            if bootstrap_data is not None:
                monthly_returns, monthly_inflation = bootstrap_data
            bootstrap_sampler = create_block_bootstrap_sampler(
                config, rng, monthly_returns, monthly_inflation)
            total_months = int((config.death_age - config.initial_age) * 12)
            bootstrap_returns, bootstrap_inflation = bootstrap_sampler.sample_sequence(total_months)
            bootstrap_month_idx = 0
            use_bootstrap = True
        except Exception as e:
            logger.warning(f"Simulation {sim} failed to create bootstrap sampler, falling back to parametric model: {e}")
            bootstrap_returns = None
            bootstrap_inflation = None
            bootstrap_month_idx = None
            use_bootstrap = False
    else:
        bootstrap_returns = None
        bootstrap_inflation = None
        bootstrap_month_idx = None
        use_bootstrap = False

    # Generate GKOS earnings path if we have initial income
    earnings_nominal = None
    earnings_real = None
    if config.annual_income_real > 0:
        # Pre-generate annual inflation rates for GKOS earnings
        if config.use_block_bootstrap:
            # When block bootstrap is enabled, use bootstrap inflation data
            if use_bootstrap and bootstrap_inflation is not None:
                # Calculate annual inflation rates from bootstrap monthly inflation
                annual_inflation_rates = []
                for year in range(total_years):
                    year_start_month = year * 12
                    year_end_month = min((year + 1) * 12, len(bootstrap_inflation))
                    if year_end_month > year_start_month:
                        monthly_inflations = bootstrap_inflation[year_start_month:year_end_month]
                        annual_inflation = np.prod(1.0 + monthly_inflations) - 1.0
                        annual_inflation_rates.append(annual_inflation)
                    else:
                        # If we don't have enough data, use geometric mean of available data as fallback
                        if len(bootstrap_inflation) > 0:
                            geometric_mean_monthly = np.exp(np.mean(np.log(1.0 + bootstrap_inflation))) - 1.0
                            annual_inflation_rates.append((1.0 + geometric_mean_monthly) ** 12 - 1.0)
                        else:
                            logger.error("Block bootstrap enabled but no inflation data available for earnings generation")
                            raise ValueError("Block bootstrap enabled but inflation data not available")
                inflation_array = annual_inflation_rates
            else:
                logger.error("Block bootstrap enabled but bootstrap data not available for earnings generation")
                raise ValueError("Block bootstrap enabled but bootstrap data not available")
        else:
            # When block bootstrap is NOT enabled, use config values
            annual_inflation_rates = []
            for year in range(total_years):
                annual_inflation_rates.append(rng.normal(config.mean_inflation_geometric,
                                                        config.std_inflation))
            inflation_array = annual_inflation_rates
        
        # Generate GKOS earnings path (real earnings)
        baseline_earnings = config.annual_income_real
        
        earnings_nominal, earnings_real, _ = simulate_earnings_path_with_inflation(
            int(config.initial_age),
            int(config.death_age),
            baseline_earnings,
            inflation_array if inflation_array else [config.mean_inflation_geometric] * total_years,
            config.gkos_params,
            rng
        )
    
    # Track consumption for utility calculation (only during retirement/withdrawal phase)
    # Only track if utility calculations are enabled (saves memory when disabled)
    consumption_stream = [] if config.enable_utility_calculations else None  # Only append during retirement phase
    
    # Cache integer month values to avoid division in hot loops
    death_age_months_acc = int(config.death_age * 12)
    initial_age_months = int(config.initial_age * 12)
    initial_age_int = int(config.initial_age)

    while age_in_months <= death_age_months_acc:
        current_age_years = age_in_months // 12  # Integer division (faster)
        current_year_idx = current_age_years - initial_age_int  # Years since start

        # Check if we can retire (at start of each year)
        if (age_in_months % 12) == 0:
            if (not is_retired) and (current_age_years in principal_lookup):
                req = principal_lookup[current_age_years]
                required_principal_real = req.get('principal_real', np.nan)
                if not np.isnan(required_principal_real):
                    required_principal_nominal = (required_principal_real *
                                                 cumulative_inflation_since_start)
                    if portfolio >= required_principal_nominal:
                        retirement_age = current_age_years
                        is_retired = True

        # Annual inflation adjustment (at start of each year)
        if age_in_months > initial_age_months and (age_in_months % 12) == 0:
            if config.use_block_bootstrap:
                # When block bootstrap is enabled, inflation MUST come from CSV/bootstrap data
                if use_bootstrap and bootstrap_inflation is not None:
                    year_start_idx = max(0, bootstrap_month_idx - 12)
                    year_end_idx = bootstrap_month_idx
                    if year_end_idx > year_start_idx and year_end_idx - year_start_idx == 12:
                        monthly_inflations = bootstrap_inflation[year_start_idx:year_end_idx]
                        annual_inflation_draw = np.prod(1.0 + monthly_inflations) - 1.0
                    else:
                        # If we don't have exactly 12 months yet, use what we have or raise error
                        # This should only happen at the very beginning, use available months
                        available_months = bootstrap_inflation[:bootstrap_month_idx] if bootstrap_month_idx > 0 else bootstrap_inflation[:12]
                        if len(available_months) > 0:
                            annual_inflation_draw = np.prod(1.0 + available_months) - 1.0
                        else:
                            logger.error(f"Block bootstrap enabled but no inflation data available at month {bootstrap_month_idx}")
                            raise ValueError("Block bootstrap enabled but inflation data not available")
                else:
                    logger.error("Block bootstrap enabled but bootstrap data not available")
                    raise ValueError("Block bootstrap enabled but bootstrap data not available")
            else:
                # When block bootstrap is NOT enabled, use config values
                annual_inflation_draw = rng.normal(config.mean_inflation_geometric,
                                                  config.std_inflation)
            annual_inflation_draw = max(annual_inflation_draw, -0.99)
            cumulative_inflation_since_start *= (1.0 + annual_inflation_draw)
            
            # Update social security for inflation
            current_annual_ss_nominal *= (1.0 + annual_inflation_draw)
            current_monthly_ss_nominal = current_annual_ss_nominal / 12.0

            # Calculate spending for this year
            # NOTE: This block runs ONCE per year (when age_in_months % 12 == 0)
            # For amortization, withdrawals are VARIABLE and recalculated annually
            # based on current principal and remaining years
            if is_retired:
                if config.use_amortization:
                    # Calculate amortized withdrawal based on current principal and remaining years
                    # This happens ONCE per year at the start of each retirement year
                    remaining_years = config.death_age - current_age_years
                    principal_at_year_start = portfolio  # Principal before this year's withdrawal
                    
                    # Calculate annual withdrawal in REAL terms
                    # Note: principal is in nominal terms, so we need to convert to real
                    principal_real = principal_at_year_start / cumulative_inflation_since_start
                    # Use expected return (calculated or user-specified)
                    # If not set, use a conservative fallback (shouldn't happen if amortization is enabled)
                    expected_return = getattr(config, 'amortization_expected_return', None)
                    if expected_return is None:
                        logger.warning(f"[AMORTIZATION] Expected return not calculated, using fallback 3%")
                        expected_return = 0.03
                    # Get desired bequest (in real terms)
                    desired_bequest_real = getattr(config, 'amortization_desired_bequest', 0.0)
                    annual_withdrawal_real = calculate_amortized_withdrawal(
                        principal_real, remaining_years, expected_return, desired_bequest_real
                    )
                    
                    # Convert to nominal for this year
                    # This annual amount is then divided by 12 for monthly withdrawals throughout the year
                    annual_withdrawal_nominal = annual_withdrawal_real * cumulative_inflation_since_start
                    current_annual_spending_nominal = annual_withdrawal_nominal
                    current_monthly_spending_nominal = current_annual_spending_nominal / 12.0
                    
                    # Track statistics
                    amortization_stats['withdrawals'].append(annual_withdrawal_real)
                    amortization_stats['withdrawals_nominal'].append(annual_withdrawal_nominal)
                    amortization_stats['principal_at_year_start'].append(principal_at_year_start)
                    amortization_stats['remaining_years'].append(remaining_years)
                    amortization_stats['total_years'] += 1
                    
                    # Check if below minimum threshold
                    min_threshold = config.amortization_min_spending_threshold * amortization_stats['initial_spending_real']
                    if annual_withdrawal_real < min_threshold:
                        amortization_stats['below_threshold_count'] += 1
                else:
                    # Traditional: adjust fixed real spending for inflation
                    current_annual_spending_nominal *= (1.0 + annual_inflation_draw)
                    current_monthly_spending_nominal = current_annual_spending_nominal / 12.0
            else:
                # Not retired yet - keep spending at initial level (will be set when retirement starts)
                if not config.use_amortization:
                    current_annual_spending_nominal = config.spending_real * cumulative_inflation_since_start
                    current_monthly_spending_nominal = current_annual_spending_nominal / 12.0

        # Get market return
        if use_bootstrap and bootstrap_returns is not None and bootstrap_month_idx < len(bootstrap_returns):
            market_return = bootstrap_returns[bootstrap_month_idx]
            bootstrap_month_idx += 1
        else:
            market_return, current_variance = simulate_monthly_return_svj(
                rng, params, current_variance)
        portfolio_growth_factor *= (1.0 + market_return)

        dollars_saved_nominal = 0.0
        savings_rate_for_month = savings_rate
        current_monthly_income_real = 0.0

        if not is_retired:
            # Use GKOS earnings if available
            # Earnings continue until retirement flag is true (not fixed at age 65)
            # Use current_year_idx to index into annual earnings array
            if earnings_nominal is not None and current_year_idx >= 0 and current_year_idx < len(earnings_nominal):
                # Get annual earnings for current year, convert to monthly
                annual_earnings_nominal = earnings_nominal[current_year_idx]
                current_monthly_income_nominal = annual_earnings_nominal / 12.0
                current_monthly_income_real = annual_earnings_nominal / (12.0 * cumulative_inflation_since_start)
            elif earnings_nominal is not None and current_year_idx >= len(earnings_nominal):
                # If we've exceeded the pre-generated earnings path, use the last year's earnings
                # (This can happen if retirement happens later than expected)
                annual_earnings_nominal = earnings_nominal[-1]
                current_monthly_income_nominal = annual_earnings_nominal / 12.0
                current_monthly_income_real = annual_earnings_nominal / (12.0 * cumulative_inflation_since_start)
            else:
                current_monthly_income_nominal = 0.0
                current_monthly_income_real = 0.0
            
            # Calculate savings
            dollars_saved_nominal = current_monthly_income_nominal * savings_rate_for_month
            portfolio += dollars_saved_nominal
            
            # During accumulation: Don't track consumption for utility (matches Cederburg - they only track retirement consumption)
            # Consumption happens but we don't store it for utility calculation
        else:
            # In retirement: consumption = spending - social security
            net_withdrawal = current_monthly_spending_nominal
            if (config.include_social_security and
                current_age_years >= config.social_security_start_age):
                net_withdrawal = max(0.0, net_withdrawal - current_monthly_ss_nominal)
            portfolio -= net_withdrawal
            # Track consumption for utility calculation (only if enabled)
            if config.enable_utility_calculations:
                # Convert withdrawal to real consumption (deflate by cumulative inflation)
                consumption_real = net_withdrawal / cumulative_inflation_since_start
                # Scale for utility calculation (divide by sqrt of household size) - matches Cederburg approach
                consumption = consumption_real / np.sqrt(config.household_size)
                consumption_stream.append(consumption)

        portfolio *= (1.0 + market_return)

        if is_retired and portfolio <= 0.0:
            portfolio = 0.0

        if portfolio <= 0.0:
            break

        # Create record for exported simulations
        if sim < config.num_sims_to_export:
            record_dict = create_simulation_record(
                sim, age_in_months, is_retired, portfolio, current_variance,
                principal_lookup, current_age_years, current_monthly_spending_nominal,
                current_annual_ss_nominal, annual_inflation_draw,
                cumulative_inflation_since_start, savings_rate_for_month,
                current_monthly_income_real, dollars_saved_nominal,
                market_return, portfolio_growth_factor, config
            )
            current_sim_record.append(record_dict)

        age_in_months += 1

    final_bequest_nominal = portfolio
    final_bequest_real = final_bequest_nominal / cumulative_inflation_since_start

    return {
        'retirement_age': retirement_age,
        'final_bequest_nominal': final_bequest_nominal,
        'final_bequest_real': final_bequest_real,
        'simulation_record': current_sim_record,
        'consumption_stream': consumption_stream if consumption_stream is not None else [],
        'amortization_stats': amortization_stats if config.use_amortization else None
    }


def run_accumulation_simulations(config, params, principal_lookup, rng, bootstrap_data=None, progress_callback=None):
    """Run all accumulation simulations
    
    Args:
        progress_callback: Optional callback function(sim_num, total) called during simulation
    """
    # #region agent log
    t_accum_start = time.perf_counter()
    # #endregion
    retirement_ages = np.full(config.num_outer, np.nan)
    ever_retired = np.zeros(config.num_outer, dtype=bool)
    detailed_simulations_to_export = []
    all_final_bequest_nominal = []
    all_final_bequest_real = []
    all_consumption_streams = []
    all_amortization_stats = []  # Collect amortization statistics if enabled

    # Get savings rate from config (default to 0.25 if not set)
    savings_rate = getattr(config, 'savings_rate', 0.25)

    for sim in tqdm(range(config.num_outer), desc="Running Simulations"):
        result = run_single_accumulation_simulation(
            sim, config, params, principal_lookup, rng, savings_rate, bootstrap_data
        )

        retirement_ages[sim] = result['retirement_age']
        if not np.isnan(result['retirement_age']):
            ever_retired[sim] = True

        all_final_bequest_nominal.append(result['final_bequest_nominal'])
        all_final_bequest_real.append(result['final_bequest_real'])
        # Only collect consumption streams if utility calculations are enabled
        if config.enable_utility_calculations:
            all_consumption_streams.append(result['consumption_stream'])
        else:
            all_consumption_streams.append([])  # Empty list to maintain list length
        
        # Collect amortization statistics if enabled
        if config.use_amortization:
            all_amortization_stats.append(result.get('amortization_stats', None))
        else:
            all_amortization_stats.append(None)

        if sim < config.num_sims_to_export:
            detailed_simulations_to_export.append(result['simulation_record'])
        
        # Call progress callback if provided
        if progress_callback:
            try:
                progress_callback(sim + 1, config.num_outer)
            except Exception:
                pass  # Don't let progress callback errors break the simulation

    # #region agent log
    t_accum_end = time.perf_counter()
    try:
        import json
        log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'debug.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'id': 'log_accum_total_time', 'timestamp': time.time() * 1000, 'location': 'simulation.py:1307', 'message': 'run_accumulation_simulations total time', 'data': {'duration_ms': (t_accum_end - t_accum_start) * 1000, 'num_outer': config.num_outer}, 'sessionId': 'debug-session', 'runId': 'profile', 'hypothesisId': 'PERF'}) + '\n')
    except: pass
    # #endregion
    return (retirement_ages, ever_retired, detailed_simulations_to_export,
            all_final_bequest_nominal, all_final_bequest_real, all_consumption_streams,
            all_amortization_stats)

