"""
Cython module for Lifecycle Retirement Simulation performance-critical functions.
Compiled version of simulate_monthly_return_svj and calculate_max_drawdown.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, fmax
from libc.stdlib cimport rand, srand, RAND_MAX
cimport cython

# Numpy must be initialized
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def simulate_monthly_return_svj_cython(object rng_local, dict params_annual, double current_variance):
    """
    Cython-accelerated Bates stochastic volatility jump-diffusion model.
    
    Args:
        rng_local: NumPy random number generator (Python object)
        params_annual: Dictionary with annual parameters (kappa, theta, nu, rho, lam, mu_J, sigma_J, mu)
        current_variance: Current variance level
        
    Returns:
        tuple: (simple_return, new_variance)
    """
    cdef double dt = 1.0 / 12.0
    cdef double kappa = params_annual["kappa"]
    cdef double theta = params_annual["theta"]
    cdef double nu = params_annual["nu"]
    cdef double rho = params_annual["rho"]
    cdef double jump_intensity = params_annual["lam"]
    cdef double jump_mean = params_annual["mu_J"]
    cdef double jump_std_dev = params_annual["sigma_J"]
    cdef double mu_annual = params_annual["mu"]
    
    # Generate correlated random numbers (Python calls)
    cdef double z1 = rng_local.normal(0.0, 1.0)
    cdef double z2 = rng_local.normal(0.0, 1.0)
    cdef double z_v = z1
    cdef double rho_sq = rho * rho
    cdef double sqrt_one_minus_rho_sq = sqrt(fmax(0.0, 1.0 - rho_sq))
    cdef double z_s = rho * z1 + sqrt_one_minus_rho_sq * z2
    
    # Update variance using Heston model
    cdef double v = current_variance
    cdef double sqrt_v = sqrt(fmax(v, 0.0))
    cdef double sqrt_dt = sqrt(dt)
    cdef double dv = kappa * (theta - v) * dt + nu * sqrt_v * sqrt_dt * z_v
    cdef double v_new = fmax(v + dv, 1e-8)
    
    # Generate jumps (Python call)
    cdef int num_jumps = rng_local.poisson(jump_intensity * dt)
    cdef double jump_component = 0.0
    cdef int i
    if num_jumps > 0:
        # Use numpy array for multiple jumps (more efficient)
        jump_array = rng_local.normal(jump_mean, jump_std_dev, size=num_jumps)
        for i in range(num_jumps):
            jump_component += jump_array[i]
    
    # Calculate return components
    cdef double jump_drift_correction = jump_intensity * (exp(jump_mean + 0.5 * jump_std_dev * jump_std_dev) - 1.0)
    cdef double drift_component = (mu_annual - jump_drift_correction - 0.5 * v) * dt
    cdef double diffusion_component = sqrt_v * sqrt_dt * z_s
    cdef double total_log_return = drift_component + diffusion_component + jump_component
    cdef double simple_return = exp(total_log_return) - 1.0
    
    return simple_return, v_new


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calculate_max_drawdown_cython(np.ndarray[np.float64_t, ndim=1] series):
    """
    Cython-accelerated maximum drawdown calculation.
    
    Args:
        series: NumPy array of portfolio values (float64)
        
    Returns:
        double: Maximum drawdown (negative value)
    """
    cdef int n = series.shape[0]
    if n == 0:
        return 0.0
    
    cdef double peak = series[0]
    cdef double max_dd = 0.0
    cdef double current_dd
    cdef int i
    
    for i in range(1, n):
        if series[i] > peak:
            peak = series[i]
        current_dd = (series[i] - peak) / peak
        if current_dd < max_dd:
            max_dd = current_dd
    
    return max_dd
