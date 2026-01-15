"""
Cython Wrapper Module

This module handles Cython imports and provides fallback Python implementations
for performance-critical functions.
"""

import sys
import os

# Add parent directories to path to find compiled Cython module
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Add build directory to path (where compiled Cython modules are located)
_build_dir = os.path.join(_parent_dir, 'build')
if os.path.exists(_build_dir) and _build_dir not in sys.path:
    sys.path.insert(0, _build_dir)

# Also check lib.win-amd64-cpython-XXX subdirectory (Windows build structure)
_build_lib_dir = os.path.join(_build_dir, f'lib.win-amd64-cpython-{sys.version_info.major}{sys.version_info.minor}')
if os.path.exists(_build_lib_dir) and _build_lib_dir not in sys.path:
    sys.path.insert(0, _build_lib_dir)

CYTHON_AVAILABLE = False
try:
    from lrs_cython import (
        simulate_monthly_return_svj_cython,
        calculate_max_drawdown_cython
    )
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


def simulate_monthly_return_svj(rng_local, params_annual, current_variance):
    """Wrapper that uses Cython version if available, else Python fallback"""
    if CYTHON_AVAILABLE:
        return simulate_monthly_return_svj_cython(rng_local, params_annual, current_variance)
    else:
        # Python fallback
        import numpy as np
        dt = 1.0 / 12.0
        kappa = params_annual["kappa"]
        theta = params_annual["theta"]
        nu = params_annual["nu"]
        rho = params_annual["rho"]
        jump_intensity = params_annual["lam"]
        jump_mean = params_annual["mu_J"]
        jump_std_dev = params_annual["sigma_J"]
        mu_annual = params_annual["mu"]

        z1 = rng_local.normal(0.0, 1.0)
        z2 = rng_local.normal(0.0, 1.0)
        z_v = z1
        z_s = rho * z1 + np.sqrt(max(0.0, 1.0 - rho**2)) * z2

        v = current_variance
        dv = kappa * (theta - v) * dt + nu * np.sqrt(max(v, 0.0)) * np.sqrt(dt) * z_v
        v_new = v + dv
        v_new = max(v_new, 1e-8)

        num_jumps = rng_local.poisson(jump_intensity * dt)
        jump_component = 0.0
        if num_jumps > 0:
            jump_component = rng_local.normal(jump_mean, jump_std_dev, size=num_jumps).sum()

        jump_drift_correction = jump_intensity * (np.exp(jump_mean + 0.5 * jump_std_dev**2) - 1.0)
        drift_component = (mu_annual - jump_drift_correction - 0.5 * v) * dt
        diffusion_component = np.sqrt(max(v, 0.0)) * np.sqrt(dt) * z_s
        total_log_return = drift_component + diffusion_component + jump_component
        simple_return = np.exp(total_log_return) - 1.0

        return simple_return, v_new


def calculate_max_drawdown(series):
    """Wrapper for max drawdown calculation"""
    import numpy as np
    
    if len(series) == 0:
        return 0.0

    if CYTHON_AVAILABLE and isinstance(series, np.ndarray):
        return calculate_max_drawdown_cython(series.astype(np.float64))
    else:
        # Python fallback
        series = np.array(series)
        peak_series = np.maximum.accumulate(series)
        drawdowns = (series - peak_series) / peak_series
        return np.min(drawdowns) if drawdowns.size > 0 else 0.0

