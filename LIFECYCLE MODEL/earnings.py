"""
GKOS Earnings Model - Integrated Module

This module provides functions to simulate lifecycle earnings using the
Guvenen, Karahan, Ozkan, and Song (GKOS, 2019) model with non-Gaussian shocks.
"""

import numpy as np


def generate_persistent_shock_mixture(n_samples, params, rng):
    """Generate persistent shocks using a Mixture of Normals."""
    component_indicator = rng.binomial(1, params['MIXTURE_PROB_TAIL'], size=n_samples)
    
    normal_shocks = rng.normal(0, params['SIGMA_ETA'], size=n_samples)
    
    tail_std = params['SIGMA_ETA'] * np.sqrt(params['TAIL_VARIANCE_MULTIPLIER'])
    tail_mean = params['SKEWNESS_PARAM'] * params['SIGMA_ETA']
    tail_shocks = rng.normal(tail_mean, tail_std, size=n_samples)
    
    eta = np.where(component_indicator == 0, normal_shocks, tail_shocks)
    return eta


def generate_transitory_shock_laplace(n_samples, params, rng):
    """Generate transitory shocks using Laplace distribution."""
    scale = params['SIGMA_EPSILON'] / np.sqrt(2)
    epsilon = rng.laplace(0, scale, size=n_samples)
    return epsilon


def compute_age_profile(ages, params):
    """Compute deterministic age-earnings profile (quadratic in age)."""
    age_centered = ages - params['AGE_PEAK']
    profile = params['AGE_PROFILE_A'] * age_centered**2 + params['AGE_PROFILE_B'] * age_centered
    profile = profile - np.mean(profile)
    return profile


def simulate_single_earnings_path(age_start, age_end, baseline_earnings, params, rng):
    """
    Simulate a single individual's earnings path using GKOS model.
    
    Parameters:
    -----------
    age_start : int
        Starting age (e.g., 25)
    age_end : int
        Ending age (e.g., 65)
    baseline_earnings : float
        Baseline earnings level at age_start (real dollars)
    params : dict
        GKOS model parameters
    rng : numpy.random.Generator
        Random number generator
        
    Returns:
    --------
    earnings : array
        Annual earnings (real dollars) for each age from age_start to age_end
    ages : array
        Array of ages
    """
    ages = np.arange(age_start, age_end + 1)
    years = len(ages)
    
    # Compute age profile
    age_profile = compute_age_profile(ages, params)
    
    # Initialize persistent component
    z = np.zeros(years)
    z[0] = rng.normal(0, params['SIGMA_Z0'])
    
    # Generate shocks
    persistent_shocks = generate_persistent_shock_mixture(years - 1, params, rng)
    transitory_shocks = generate_transitory_shock_laplace(years, params, rng)
    
    # Simulate AR(1) process
    for t in range(1, years):
        z[t] = params['RHO'] * z[t-1] + persistent_shocks[t-1]
    
    # Compute log earnings
    log_earnings = z + age_profile + transitory_shocks
    
    # Convert to actual earnings (real dollars)
    earnings = np.exp(log_earnings) * baseline_earnings
    
    return earnings, ages


def simulate_earnings_path_with_inflation(age_start, age_end, baseline_earnings, inflation_rates, params, rng):
    """
    Simulate earnings path with inflation adjustment.
    
    This function generates real earnings using GKOS model, then converts to nominal
    by applying cumulative inflation.
    
    Parameters:
    -----------
    age_start : int
        Starting age
    age_end : int
        Ending age
    baseline_earnings : float
        Real earnings at age_start
    inflation_rates : array
        Annual inflation rates (must have length = age_end - age_start, i.e., one per year)
    params : dict
        GKOS model parameters
    rng : numpy.random.Generator
        Random number generator
        
    Returns:
    --------
    nominal_earnings : array
        Nominal earnings in each year
    real_earnings : array
        Real earnings (baseline-adjusted) in each year
    ages : array
        Array of ages
    """
    # Get real earnings path from GKOS model
    real_earnings, ages = simulate_single_earnings_path(
        age_start, age_end, baseline_earnings=baseline_earnings, params=params, rng=rng
    )
    
    # Apply inflation to convert real to nominal
    nominal_earnings = np.zeros(len(ages))
    nominal_earnings[0] = real_earnings[0]  # First year, no inflation
    cumulative_inflation = 1.0
    
    # Apply inflation year by year
    n_years = len(ages)
    n_inflation_rates = len(inflation_rates)
    
    for i in range(1, n_years):
        if i-1 < n_inflation_rates:
            cumulative_inflation *= (1 + inflation_rates[i-1])
        # real_earnings[i] is already in real dollars, apply cumulative inflation
        nominal_earnings[i] = real_earnings[i] * cumulative_inflation
    
    return nominal_earnings, real_earnings, ages

