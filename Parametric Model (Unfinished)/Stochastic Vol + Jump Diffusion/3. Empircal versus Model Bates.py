"""
Monte Carlo Simulation: Regime-Switching Bates Model

This script runs Monte Carlo simulations using:
1. Bates model parameters fitted to each regime (from moment matching)
2. HMM transition probabilities for regime switching

It compares empirical data statistics to simulated model statistics:
- Median CAGR (Compound Annual Growth Rate)
- Median Annual Volatility
- Median Skewness
- Median Kurtosis
- Median Max Drawdown

Usage:
    python Sim.py [--n-simulations N] [--n-periods N] [--seed N] [--csv PATH]
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from scipy import stats
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Constants
PERIODS_PER_YEAR = 12
DT = 1.0 / PERIODS_PER_YEAR

# Crisis regime tail boost: (lam_floor, sigma_J_floor) - ensures minimum jump intensity/vol in crisis
# to better capture tail risk (empirical max DD ~-86% vs model ~-65% without boost)
CRISIS_JUMP_BOOST = (1.8, 0.24)

# Parameter indices (consistent with moment matching script)
MU_IDX = 0
KAPPA_IDX = 1
THETA_IDX = 2
NU_IDX = 3
RHO_IDX = 4
V0_IDX = 5
LAM_IDX = 6
MU_J_IDX = 7
SIGMA_J_IDX = 8

IDX = {
    "mu": MU_IDX, "kappa": KAPPA_IDX, "theta": THETA_IDX, "nu": NU_IDX,
    "rho": RHO_IDX, "v0": V0_IDX, "lam": LAM_IDX, "mu_J": MU_J_IDX,
    "sigma_J": SIGMA_J_IDX
}

console = Console(width=120)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_bates_parameters(csv_path):
    """Load Bates model parameters for each regime from CSV.
    
    Returns:
        dict: {regime_id: params_array} where params_array is 9-element array
        dict: {regime_id: moment_matching_info} with fitting quality metrics
    """
    df = pd.read_csv(csv_path)
    params_dict = {}
    moment_matching_info = {}
    
    for _, row in df.iterrows():
        regime_id = float(row['name'].split('_')[1])
        params = np.array([
            row['mu_annual'],
            row['kappa'],
            row['theta_annual'],
            row['nu'],
            row['rho'],
            row['v0'],
            row['lambda_per_year'],
            row['mu_J'],
            row['sigma_J']
        ])
        params_dict[regime_id] = params
        
        # Store moment matching quality metrics
        moment_matching_info[regime_id] = {
            'n_obs': int(row['N_obs']),
            'objective': row['objective'],
            'emp_mean': row['emp_mean'],
            'emp_std': row['emp_std'],
            'emp_skew': row['emp_skew'],
            'emp_kurt': row['emp_kurt'],
            'emp_max_dd': row['emp_max_dd'],
            'model_mean': row['model_mean'],
            'model_std': row['model_std'],
            'model_skew': row['model_skew'],
            'model_kurt': row['model_kurt'],
            'model_max_dd': row['model_max_dd'],
            'mean_error_pct': row['mean_error_pct'],
            'std_error_pct': row['std_error_pct'],
            'skew_error_pct': row['skew_error_pct'],
            'kurt_error_pct': row['kurt_error_pct'],
            'max_dd_error_pct': row['max_dd_error_pct']
        }
    
    return params_dict, moment_matching_info

def load_transition_matrix(csv_path):
    """Load HMM transition matrix from CSV.
    
    Returns:
        np.array: Transition matrix (n_regimes x n_regimes)
        list: Regime IDs in order
    """
    df = pd.read_csv(csv_path, index_col=0)
    
    # Extract regime IDs from column names
    regime_ids = []
    for col in df.columns:
        # Extract number from "0: V-1: Lowest Volatility" format
        regime_id = float(col.split(':')[0].strip())
        regime_ids.append(regime_id)
    
    # Build transition matrix - ensure rows and columns are in same order
    n_regimes = len(regime_ids)
    transition_matrix = np.zeros((n_regimes, n_regimes))
    
    # Create mapping from regime ID to index
    regime_to_idx = {regime_id: idx for idx, regime_id in enumerate(regime_ids)}
    
    for i, row_name in enumerate(df.index):
        # Extract regime ID from row name
        from_regime = float(row_name.split(':')[0].strip())
        from_idx = regime_to_idx[from_regime]
        
        for j, col_name in enumerate(df.columns):
            to_regime = float(col_name.split(':')[0].strip())
            to_idx = regime_to_idx[to_regime]
            transition_matrix[from_idx, to_idx] = df.iloc[i, j]
    
    return transition_matrix, regime_ids

def load_empirical_data(csv_path):
    """Load empirical returns data.
    
    Returns:
        np.array: Log returns
        np.array: Regime IDs (for reference)
    """
    df = pd.read_csv(csv_path)
    
    # Get returns column
    if 'Total Nominal Return (%)' in df.columns:
        returns = df['Total Nominal Return (%)'].values / 100.0
    elif 'returns' in df.columns:
        returns = df['returns'].values
    else:
        raise ValueError("Could not find returns column")
    
    # Convert to log returns: "Total Nominal Return (%)" is simple return (decimal).
    # For consistency with Bates model (which outputs log returns), convert.
    returns = np.log(1.0 + returns)
    
    # Get regime IDs if available
    regime_col = None
    for col in df.columns:
        if 'regime' in col.lower() or 'Regime' in col:
            regime_col = col
            break
    
    regime_ids = df[regime_col].values if regime_col else None
    
    dates = df['Monthly Start Date'].values if 'Monthly Start Date' in df.columns else None
    
    return returns, regime_ids, dates

# ============================================================================
# BATES MODEL SIMULATION
# ============================================================================

def simulate_bates_period(params, v_prev, rng, use_expected_variance=False):
    """Simulate one period of Bates model using proper Euler discretization.
    
    IMPORTANT: Standard Euler scheme uses V_t (previous variance) for both drift and diffusion,
    then updates variance for next period. Using V_{t+1} introduces look-ahead bias!
    
    Args:
        params: 9-element array of Bates parameters
        v_prev: Previous variance value (V_t)
        rng: Random number generator
        use_expected_variance: If True, use expected variance in mean calculation
    
    Returns:
        log_return: Log return for this period
        v_new: New variance value (V_{t+1})
    """
    mu = params[IDX['mu']]
    kappa = params[IDX['kappa']]
    theta = params[IDX['theta']]
    nu = params[IDX['nu']]
    rho = params[IDX['rho']]
    lam = params[IDX['lam']]
    mu_J = params[IDX['mu_J']]
    sigma_J = params[IDX['sigma_J']]
    
    # Use V_t (previous variance) for this period's calculations
    v_t = max(v_prev, 1e-12)
    sqrt_v_t = math.sqrt(v_t)
    
    # Correlated Wiener processes
    z1 = rng.standard_normal()
    z2 = rho * z1 + math.sqrt(1.0 - rho**2) * rng.standard_normal()
    
    # Merton jumps
    num_jumps = rng.poisson(lam * DT)
    if num_jumps > 0:
        jump_size = np.sum(rng.normal(mu_J, sigma_J, num_jumps))
    else:
        jump_size = 0.0
    
    # Bates price process - use V_t (previous variance) for drift and diffusion
    # This is the correct Euler discretization order
    if use_expected_variance:
        # Calculate expected variance over this period
        if kappa > 1e-6:
            v_end_expected = theta + (v_t - theta) * math.exp(-kappa * DT)
            expected_v = (v_t + v_end_expected) / 2.0
        else:
            expected_v = theta
        variance_for_mean = expected_v
    else:
        variance_for_mean = v_t  # FIXED: Use V_t (previous variance), not V_{t+1}
    
    log_return = (mu - 0.5 * variance_for_mean) * DT + sqrt_v_t * math.sqrt(DT) * z1 + jump_size
    
    # NOW update variance for next period (V_{t+1})
    v_new = v_t + kappa * (theta - v_t) * DT + nu * sqrt_v_t * math.sqrt(DT) * z2
    v_new = max(v_new, 1e-12)
    
    return log_return, v_new

def simulate_regime_switching_bates(params_dict, transition_matrix, regime_ids, n_periods, 
                                     initial_regime=None, initial_variance=None, seed=None,
                                     burn_in_periods=0, reset_variance_on_switch=False,
                                     stress_variance_on_crisis_entry=True,
                                     crisis_jump_boost=CRISIS_JUMP_BOOST,
                                     use_expected_variance=False):
    """Simulate regime-switching Bates model.
    
    Args:
        params_dict: {regime_id: params_array} for each regime
        transition_matrix: Transition probability matrix (n_regimes x n_regimes)
        regime_ids: List of regime IDs in order matching transition_matrix
        n_periods: Number of periods to simulate
        initial_regime: Initial regime ID (default: sample from stationary distribution)
        initial_variance: Initial variance (default: use v0 from initial regime)
        seed: Random seed
        burn_in_periods: Number of periods to simulate before starting to record (for equilibrium)
        reset_variance_on_switch: If True, reset variance to v0 when switching regimes
        stress_variance_on_crisis_entry: If True, when entering crisis regime (highest theta),
            reset variance to v0 to avoid understating tail risk from low-vol entry
        crisis_jump_boost: (lam_floor, sigma_J_floor) - when in crisis, use at least these values
            for jump intensity and jump vol to capture tail risk (crises often have jump-like crashes).
            Set to (0,0) to disable.
    
    Returns:
        returns: Array of log returns (n_periods,)
        regimes: Array of regime IDs (n_periods,)
    """
    rng = np.random.default_rng(seed)
    
    # Crisis regime = highest long-run variance (theta)
    crisis_regime = max(regime_ids, key=lambda r: params_dict[r][IDX['theta']]) if stress_variance_on_crisis_entry else None
    
    # Initialize regime
    if initial_regime is None:
        # Sample from stationary distribution
        # Compute stationary distribution as left eigenvector of transition matrix
        eigenvals, eigenvecs = np.linalg.eig(transition_matrix.T)
        stationary_idx = np.argmax(np.real(eigenvals))
        stationary_dist = np.real(eigenvecs[:, stationary_idx])
        stationary_dist = stationary_dist / np.sum(stationary_dist)
        initial_regime_idx = rng.choice(len(regime_ids), p=stationary_dist)
        current_regime = regime_ids[initial_regime_idx]
    else:
        current_regime = initial_regime
        initial_regime_idx = regime_ids.index(initial_regime)
    
    # Initialize variance - use equilibrium value (theta) for better convergence
    if initial_variance is None:
        # Use theta (equilibrium) instead of v0 for better matching with analytical formulas
        initial_variance = params_dict[current_regime][IDX['theta']]
    
    # Initialize arrays
    returns = np.zeros(n_periods)
    regimes = np.zeros(n_periods)
    variance = initial_variance
    previous_regime = current_regime
    
    # Burn-in period (simulate but don't record)
    for _ in range(burn_in_periods):
        params = params_dict[current_regime]
        params_use = params
        if crisis_jump_boost and len(crisis_jump_boost) >= 2 and current_regime == crisis_regime:
            lam_floor, sigma_J_floor = crisis_jump_boost[0], crisis_jump_boost[1]
            if lam_floor > 0 or sigma_J_floor > 0:
                params_use = np.array(params, dtype=float)
                if lam_floor > 0:
                    params_use[IDX['lam']] = max(params[IDX['lam']], lam_floor)
                if sigma_J_floor > 0:
                    params_use[IDX['sigma_J']] = max(params[IDX['sigma_J']], sigma_J_floor)
        _, variance = simulate_bates_period(params_use, variance, rng, use_expected_variance=use_expected_variance)
        
        # Transition to next regime
        current_regime_idx = regime_ids.index(current_regime)
        transition_probs = transition_matrix[current_regime_idx, :]
        next_regime_idx = rng.choice(len(regime_ids), p=transition_probs)
        current_regime = regime_ids[next_regime_idx]
    
    # Crisis regime for logging (always; crisis_regime may be None when stress_variance_on_crisis_entry is False)
    crisis_regime_for_log = max(regime_ids, key=lambda r: params_dict[r][IDX['theta']])
    _debug_log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.debug', 'debug.log')
    _entered_crisis_logged = 0
    # Main simulation
    for t in range(n_periods):
        # Store current regime
        regimes[t] = current_regime
        
        # Get parameters for current regime
        params = params_dict[current_regime]
        
        # Reset variance if regime switched and reset_variance_on_switch is True
        if reset_variance_on_switch and current_regime != previous_regime:
            variance = params[IDX['v0']]  # Reset to new regime's v0
        # Stress: when entering crisis regime, reset to v0 to avoid understating tail risk
        # (carrying low variance from calm regime into crisis understates severity)
        elif stress_variance_on_crisis_entry and current_regime == crisis_regime and current_regime != previous_regime:
            variance = params[IDX['v0']]
        
        # Crisis jump boost: ensure minimum jump intensity/vol in crisis (captures crash dynamics)
        params_use = params
        if crisis_jump_boost and len(crisis_jump_boost) >= 2 and current_regime == crisis_regime:
            lam_floor, sigma_J_floor = crisis_jump_boost[0], crisis_jump_boost[1]
            if lam_floor > 0 or sigma_J_floor > 0:
                params_use = np.array(params, dtype=float)
                if lam_floor > 0:
                    params_use[IDX['lam']] = max(params[IDX['lam']], lam_floor)
                if sigma_J_floor > 0:
                    params_use[IDX['sigma_J']] = max(params[IDX['sigma_J']], sigma_J_floor)
        
        # Simulate one period
        log_return, variance = simulate_bates_period(params_use, variance, rng, use_expected_variance=use_expected_variance)
        returns[t] = log_return
        
        # Transition to next regime
        previous_regime = current_regime
        current_regime_idx = regime_ids.index(current_regime)
        transition_probs = transition_matrix[current_regime_idx, :]
        next_regime_idx = rng.choice(len(regime_ids), p=transition_probs)
        current_regime = regime_ids[next_regime_idx]
    
    return returns, regimes

def simulate_with_historical_regimes(params_dict, empirical_regimes, n_periods, n_simulations=500,
                                     seed=None, stress_variance_on_crisis_entry=True,
                                     crisis_jump_boost=CRISIS_JUMP_BOOST, use_expected_variance=False):
    """Simulate returns using the EXACT historical HMM regime sequence.
    
    Preserves time-series dependence of regime timing. For each period t we use
    empirical_regimes[t] to select params and draw one return from Bates.
    Variance carries across periods (same as standard simulation).
    
    This diagnostic tests: if we keep the regime path fixed, does the model
    produce tail risk similar to empirical? If yes, the gap is from random
    regime switching. If no, the gap is from within-regime misspecification.
    
    Returns:
        list of (returns, regimes) - regimes are identical (historical) in all paths
    """
    rng = np.random.default_rng(seed)
    regime_ids = sorted(params_dict.keys())
    crisis_regime = max(regime_ids, key=lambda r: params_dict[r][IDX['theta']]) if stress_variance_on_crisis_entry else None
    empirical_regimes_int = np.array([int(r) for r in empirical_regimes[:n_periods]])
    
    paths = []
    for sim in range(n_simulations):
        sim_rng = np.random.default_rng(seed + sim if seed is not None else None)
        returns = np.zeros(n_periods)
        # Start variance at theta of first regime
        first_regime = empirical_regimes_int[0]
        variance = params_dict[first_regime][IDX['theta']]
        prev_regime = first_regime
        
        for t in range(n_periods):
            regime_id = empirical_regimes_int[t]
            params = params_dict[regime_id]
            # Stress: when entering crisis regime, reset variance to v0
            if stress_variance_on_crisis_entry and regime_id == crisis_regime and regime_id != prev_regime:
                variance = params[IDX['v0']]
            prev_regime = regime_id
            
            params_use = params
            if crisis_jump_boost and len(crisis_jump_boost) >= 2 and regime_id == crisis_regime:
                lam_floor, sigma_J_floor = crisis_jump_boost[0], crisis_jump_boost[1]
                if lam_floor > 0 or sigma_J_floor > 0:
                    params_use = np.array(params, dtype=float)
                    if lam_floor > 0:
                        params_use[IDX['lam']] = max(params[IDX['lam']], lam_floor)
                    if sigma_J_floor > 0:
                        params_use[IDX['sigma_J']] = max(params[IDX['sigma_J']], sigma_J_floor)
            
            log_return, variance = simulate_bates_period(
                params_use, variance, sim_rng, use_expected_variance=use_expected_variance
            )
            returns[t] = log_return
        
        paths.append((returns.copy(), empirical_regimes_int.copy()))
    
    return paths

# ============================================================================
# SIMULATION-BASED MOMENT COMPUTATION
# ============================================================================

def compute_moments_via_simulation(params, n_simulations=10000, n_periods=1000, 
                                    burn_in_periods=100, seed=None, use_expected_variance=False):
    """Compute Bates model moments via Monte Carlo simulation.
    
    This is an alternative to analytical formulas that ensures exact matching
    between parameter fitting and simulation behavior.
    
    Args:
        params: 9-element array of Bates parameters
        n_simulations: Number of simulation paths
        n_periods: Number of periods per path
        burn_in_periods: Burn-in periods to reach equilibrium
        seed: Random seed
        use_expected_variance: Use expected variance in mean calculation
    
    Returns:
        dict with keys: 'mean', 'std', 'skew', 'kurt', 'max_dd'
    """
    rng = np.random.default_rng(seed)
    all_returns = []
    
    for sim in range(n_simulations):
        # Single regime simulation (no switching)
        returns = []
        variance = params[IDX['theta']]  # Start at equilibrium
        
        # Burn-in
        for _ in range(burn_in_periods):
            _, variance = simulate_bates_period(params, variance, rng, use_expected_variance=use_expected_variance)
        
        # Main simulation
        for _ in range(n_periods):
            log_return, variance = simulate_bates_period(params, variance, rng, use_expected_variance=use_expected_variance)
            returns.append(log_return)
        
        all_returns.extend(returns)
    
    all_returns = np.array(all_returns)
    
    # Compute statistics
    mean_period = np.mean(all_returns)
    std_period = np.std(all_returns, ddof=0)
    skew = stats.skew(all_returns, bias=False) if len(all_returns) > 2 else 0.0
    kurt = stats.kurtosis(all_returns, fisher=True, bias=False) if len(all_returns) > 3 else 0.0
    
    # Max drawdown (compute on a per-path basis, then average)
    max_dds = []
    for sim in range(n_simulations):
        path_returns = all_returns[sim * n_periods:(sim + 1) * n_periods]
        if len(path_returns) > 0:
            cumulative_price = np.exp(np.cumsum(path_returns))
            peak_series = np.maximum.accumulate(cumulative_price)
            drawdowns = (cumulative_price - peak_series) / peak_series
            max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
            max_dds.append(max_dd)
    
    max_dd = np.mean(max_dds) if len(max_dds) > 0 else 0.0
    
    return {
        'mean': mean_period,
        'std': std_period,
        'skew': skew,
        'kurt': kurt,
        'max_dd': max_dd
    }

# ============================================================================
# STATISTICS COMPUTATION
# ============================================================================

def compute_cagr(returns):
    """Compute Compound Annual Growth Rate from log returns.
    
    Args:
        returns: Array of log returns
    
    Returns:
        CAGR: Annualized return
    """
    if len(returns) == 0:
        return 0.0
    total_return = np.sum(returns)
    n_years = len(returns) / PERIODS_PER_YEAR
    if n_years > 0:
        cagr = total_return / n_years
    else:
        cagr = 0.0
    return cagr

def compute_annual_volatility(returns):
    """Compute annualized volatility from log returns.
    
    Args:
        returns: Array of log returns
    
    Returns:
        Annualized volatility
    """
    if len(returns) < 2:
        return 0.0
    period_vol = np.std(returns, ddof=0)
    annual_vol = period_vol * math.sqrt(PERIODS_PER_YEAR)
    return annual_vol

def compute_skewness(returns):
    """Compute skewness from log returns.
    
    Args:
        returns: Array of log returns
    
    Returns:
        Skewness
    """
    if len(returns) < 3:
        return 0.0
    return stats.skew(returns, bias=False)

def compute_kurtosis(returns):
    """Compute excess kurtosis from log returns.
    
    Args:
        returns: Array of log returns
    
    Returns:
        Excess kurtosis
    """
    if len(returns) < 4:
        return 0.0
    return stats.kurtosis(returns, fisher=True, bias=False)

def get_drawdown_path(returns):
    """Return running drawdown from peak at each time step. Used to find max DD window."""
    if len(returns) == 0:
        return np.array([])
    cumulative_price = np.exp(np.cumsum(returns))
    peak_series = np.maximum.accumulate(cumulative_price)
    drawdowns = (cumulative_price - peak_series) / peak_series
    return drawdowns

def compute_max_drawdown(returns):
    """Compute maximum drawdown from log returns.
    
    Args:
        returns: Array of log returns
    
    Returns:
        Maximum drawdown (negative value, e.g. -0.5 = 50% drawdown)
    """
    if len(returns) == 0:
        return 0.0
    
    # Log returns: P_t/P_0 = exp(cumsum(returns)). cumprod(1+r) is WRONG for log returns.
    cumulative_price = np.exp(np.cumsum(returns))
    peak_series = np.maximum.accumulate(cumulative_price)
    drawdowns = (cumulative_price - peak_series) / peak_series
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
    return max_dd

def compute_statistics(returns):
    """Compute all statistics for a returns series.
    
    Returns:
        dict with keys: 'cagr', 'vol', 'skew', 'kurt', 'max_dd'
    """
    return {
        'cagr': compute_cagr(returns),
        'vol': compute_annual_volatility(returns),
        'skew': compute_skewness(returns),
        'kurt': compute_kurtosis(returns),
        'max_dd': compute_max_drawdown(returns)
    }

# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_monte_carlo_simulation(params_dict, transition_matrix, regime_ids, 
                               n_simulations=1000, n_periods=None, seed=None, 
                               return_paths=False, use_expected_variance=False,
                               crisis_jump_boost=None):
    """Run Monte Carlo simulation with regime switching.
    
    Args:
        params_dict: {regime_id: params_array} for each regime
        transition_matrix: Transition probability matrix
        regime_ids: List of regime IDs
        n_simulations: Number of simulation paths
        n_periods: Number of periods per simulation (if None, use empirical length)
        seed: Random seed
        return_paths: If True, also return returns and regimes arrays
    
    Returns:
        list of dicts: Each dict contains statistics for one simulation
        (if return_paths=True, also returns list of (returns, regimes) tuples)
    """
    if n_periods is None:
        # Use average regime duration to estimate length
        # For now, use a reasonable default (e.g., 1000 periods = ~83 years)
        n_periods = 1000
    
    all_stats = []
    all_paths = [] if return_paths else None
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[current_sim]}/{task.fields[total_sims]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            "[bold cyan]Running Monte Carlo simulations...",
            total=n_simulations,
            current_sim=0,
            total_sims=n_simulations
        )
        
        for sim in range(n_simulations):
            # Use different seed for each simulation
            sim_seed = seed + sim if seed is not None else None
            boost = crisis_jump_boost if crisis_jump_boost is not None else CRISIS_JUMP_BOOST
            returns, regimes = simulate_regime_switching_bates(
                params_dict, transition_matrix, regime_ids, n_periods,
                seed=sim_seed,
                crisis_jump_boost=boost,
                use_expected_variance=use_expected_variance
            )
            
            stats_dict = compute_statistics(returns)
            all_stats.append(stats_dict)
            
            if return_paths:
                all_paths.append((returns, regimes))
            
            progress.update(task, advance=1, current_sim=sim + 1)
    
    if return_paths:
        return all_stats, all_paths
    return all_stats

# ============================================================================
# DEEP INVESTIGATION
# ============================================================================

def _find_max_drawdown_window(returns):
    """Find (peak_idx, trough_idx) for the maximum drawdown. Returns (0,0) if empty."""
    if len(returns) == 0:
        return 0, 0
    cumulative_price = np.exp(np.cumsum(returns))
    peak_series = np.maximum.accumulate(cumulative_price)
    dd_path = (cumulative_price - peak_series) / peak_series
    trough_idx = int(np.argmin(dd_path))
    # Peak is where price was highest before or at the start of the drawdown (last high before trough)
    peak_idx = int(np.argmax(cumulative_price[:trough_idx + 1])) if trough_idx >= 0 else 0
    return peak_idx, trough_idx

def investigate_max_drawdown_timing(empirical_returns, empirical_regimes, dates=None):
    """When and where did the empirical max drawdown occur?"""
    peak_idx, trough_idx = _find_max_drawdown_window(empirical_returns)
    n = len(empirical_returns)
    if n == 0:
        return {}
    regimes_during = empirical_regimes[peak_idx:trough_idx + 1] if empirical_regimes is not None else None
    dd_window_returns = empirical_returns[peak_idx:trough_idx + 1]
    result = {
        'peak_idx': peak_idx,
        'trough_idx': trough_idx,
        'duration_months': trough_idx - peak_idx + 1,
        'max_dd': compute_max_drawdown(empirical_returns),
        'window_cumulative_return': np.sum(dd_window_returns),
        'window_mean_monthly': np.mean(dd_window_returns) if len(dd_window_returns) > 0 else 0,
        'window_std': np.std(dd_window_returns) if len(dd_window_returns) > 1 else 0,
        'regime_counts': dict(zip(*np.unique(regimes_during, return_counts=True))) if regimes_during is not None else None,
        'regime_pct': {int(k): v / len(regimes_during) * 100 for k, v in (dict(zip(*np.unique(regimes_during, return_counts=True))).items() if regimes_during is not None else [])} or None,
        'peak_date': dates[peak_idx] if dates is not None and peak_idx < len(dates) else None,
        'trough_date': dates[trough_idx] if dates is not None and trough_idx < len(dates) else None,
        'worst_single_month_idx': peak_idx + int(np.argmin(dd_window_returns)) if len(dd_window_returns) > 0 else peak_idx,
        'worst_single_month_return': np.min(dd_window_returns) if len(dd_window_returns) > 0 else 0,
    }
    return result

def investigate_regime_duration_vs_drawdown(simulated_paths, regime_ids):
    """Correlate time spent in high-vol regime with max drawdown severity."""
    regime_2_id = max(regime_ids)  # Assume highest ID is crisis regime
    data = []
    for returns, regimes in simulated_paths:
        max_dd = compute_max_drawdown(returns)
        pct_in_r2 = np.mean(regimes == regime_2_id) * 100
        n_periods_r2 = np.sum(regimes == regime_2_id)
        # Longest consecutive stint in R2
        in_r2 = (regimes == regime_2_id).astype(int)
        streaks = []
        current = 0
        for x in in_r2:
            current = current * x + x
            streaks.append(current)
        max_streak_r2 = max(streaks) if streaks else 0
        data.append({'max_dd': max_dd, 'pct_in_r2': pct_in_r2, 'n_r2': n_periods_r2, 'max_streak_r2': max_streak_r2})
    df = pd.DataFrame(data)
    corr_pct = np.corrcoef(df['max_dd'], df['pct_in_r2'])[0, 1] if len(df) > 1 else 0
    corr_streak = np.corrcoef(df['max_dd'], df['max_streak_r2'])[0, 1] if len(df) > 1 else 0
    return {
        'corr_max_dd_pct_r2': corr_pct,
        'corr_max_dd_max_streak_r2': corr_streak,
        'mean_pct_r2': df['pct_in_r2'].mean(),
        'mean_max_streak_r2': df['max_streak_r2'].mean(),
        'paths_with_dd_below_80pct': int(np.sum(df['max_dd'] < -0.80)),
        'paths_with_dd_below_86pct': int(np.sum(df['max_dd'] < -0.86)),
    }

def investigate_worst_periods(empirical_returns, empirical_regimes, params_dict, regime_ids, n_worst=24):
    """Compare empirical worst months to model's distribution in same regime."""
    worst_idx = np.argsort(empirical_returns)[:n_worst]  # Most negative returns
    emp_worst_returns = empirical_returns[worst_idx]
    emp_worst_regimes = empirical_regimes[worst_idx]
    regime_counts = dict(zip(*np.unique(emp_worst_regimes, return_counts=True)))
    regime_2_id = max(regime_ids)
    # Simulate many returns from Regime 2 (crisis) and check if empirical worst is in the tail
    params_r2 = params_dict[regime_2_id]
    n_sim = 50000
    rng = np.random.default_rng(42)
    v = params_r2[IDX['theta']]
    sim_r2_returns = []
    for _ in range(n_sim):
        r, v = simulate_bates_period(params_r2, v, rng)
        sim_r2_returns.append(r)
    sim_r2_returns = np.array(sim_r2_returns)
    emp_worst_single = np.min(empirical_returns)  # Single worst month
    percentile_of_worst = stats.percentileofscore(sim_r2_returns, emp_worst_single)
    return {
        'emp_worst_single_month': emp_worst_single,
        'model_p1_r2': np.percentile(sim_r2_returns, 1),
        'model_p01_r2': np.percentile(sim_r2_returns, 0.1),
        'emp_worst_in_model_percentile': percentile_of_worst,
        'worst_24_regime_dist': regime_counts,
        'n_worst_in_regime_2': int(regime_counts.get(regime_2_id, regime_counts.get(float(regime_2_id), regime_counts.get(int(regime_2_id), 0)))),
    }

def simulate_with_variance_tracking(params_dict, transition_matrix, regime_ids, empirical_regimes, n_periods, use_historical_path=True, seed=42):
    """Track variance when entering Regime 2. With historical path, run once through and record at each 1->2 or 0->2 transition."""
    regime_2_id = max(regime_ids)
    variances_at_r2_entry = []
    rng = np.random.default_rng(seed)
    if use_historical_path and empirical_regimes is not None:
        regime_seq = np.array([int(r) for r in empirical_regimes[:n_periods]])
        variance = params_dict[regime_seq[0]][IDX['theta']]
        prev_r = int(regime_seq[0])
        for t in range(1, n_periods):
            curr_r = int(regime_seq[t])
            if curr_r == regime_2_id and curr_r != prev_r:
                variances_at_r2_entry.append(variance)
            params = params_dict[curr_r]
            _, variance = simulate_bates_period(params, variance, rng)
            prev_r = curr_r
    else:
        current_idx = rng.integers(len(regime_ids))
        current_regime = regime_ids[current_idx]
        variance = params_dict[current_regime][IDX['theta']]
        for _ in range(n_periods):
            next_idx = rng.choice(len(regime_ids), p=transition_matrix[current_idx, :])
            next_regime = regime_ids[next_idx]
            if next_regime == regime_2_id and next_regime != current_regime:
                variances_at_r2_entry.append(variance)
            params = params_dict[next_regime]
            _, variance = simulate_bates_period(params, variance, rng)
            current_regime = next_regime
            current_idx = next_idx
    return np.array(variances_at_r2_entry) if variances_at_r2_entry else np.array([float('nan')])

def run_deep_investigation(params_dict, transition_matrix, regime_ids, empirical_returns,
                           empirical_regimes, simulated_paths, dates=None):
    """Run all investigation steps and print report."""
    from rich.panel import Panel
    console.print("\n[bold magenta]═══════════════════════════════════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]  DEEP INVESTIGATION: Why does the model understate tail risk?[/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════════════════════════════════════════[/bold magenta]\n")

    # 1. Max drawdown timing
    dd_timing = investigate_max_drawdown_timing(empirical_returns, empirical_regimes, dates)
    console.print("[bold cyan]1. When did the empirical max drawdown occur?[/bold cyan]")
    console.print(f"   Peak index: {dd_timing['peak_idx']} → Trough index: {dd_timing['trough_idx']}")
    console.print(f"   Duration: {dd_timing['duration_months']} months")
    if dd_timing.get('peak_date'):
        console.print(f"   Approx dates: {dd_timing['peak_date']} to {dd_timing['trough_date']}")
    console.print(f"   Max DD: {dd_timing['max_dd']:.4f} ({dd_timing['max_dd']*100:.2f}%)")
    console.print(f"   Cumulative return over drawdown window: {np.exp(dd_timing['window_cumulative_return'])-1:.2%}")
    console.print(f"   Mean monthly return in window: {dd_timing['window_mean_monthly']:.4f} ({dd_timing['window_mean_monthly']*100:.2f}%)")
    console.print(f"   Std in window: {dd_timing['window_std']:.4f}")
    if dd_timing.get('regime_pct'):
        regime_pct_str = {int(k): f"{float(v):.1f}%" for k, v in dd_timing['regime_pct'].items()}
        console.print(f"   Regime mix during drawdown: {regime_pct_str}")
    console.print(f"   Worst single month in window: {dd_timing['worst_single_month_return']:.4f} ({dd_timing['worst_single_month_return']*100:.2f}%) at index {dd_timing['worst_single_month_idx']}")
    console.print()

    # 2. Regime duration vs drawdown
    reg_dd = investigate_regime_duration_vs_drawdown(simulated_paths, regime_ids)
    console.print("[bold cyan]2. Does time in Regime 2 (crisis) predict max drawdown?[/bold cyan]")
    console.print(f"   Correlation(max_dd, % time in Regime 2): {reg_dd['corr_max_dd_pct_r2']:.3f}")
    console.print(f"   Correlation(max_dd, max consecutive months in R2): {reg_dd['corr_max_dd_max_streak_r2']:.3f}")
    console.print(f"   Simulated paths: mean {reg_dd['mean_pct_r2']:.1f}% in R2, mean max streak {reg_dd['mean_max_streak_r2']:.1f} months")
    console.print(f"   Paths with max DD < -80%: {reg_dd['paths_with_dd_below_80pct']} | < -86%: {reg_dd['paths_with_dd_below_86pct']}")
    console.print()

    # 3. Worst periods
    worst = investigate_worst_periods(empirical_returns, empirical_regimes, params_dict, regime_ids)
    console.print("[bold cyan]3. How extreme were the empirical worst months vs model?[/bold cyan]")
    console.print(f"   Empirical worst single month: {worst['emp_worst_single_month']:.4f} ({worst['emp_worst_single_month']*100:.2f}%)")
    console.print(f"   Model Regime 2: 1st percentile = {worst['model_p1_r2']:.4f}, 0.1st = {worst['model_p01_r2']:.4f}")
    console.print(f"   Empirical worst month falls at model's {worst['emp_worst_in_model_percentile']:.2f}th percentile (lower = more extreme)")
    dist_str = {int(k): int(v) for k, v in worst['worst_24_regime_dist'].items()}
    console.print(f"   Of worst 24 months, {worst['n_worst_in_regime_2']} were in Regime 2. Full dist: {dist_str}")
    console.print()

    # 4. Historical regime sequence - how much time in R2 during empirical DD?
    emp_r2_during_dd = 0
    if dd_timing.get('regime_counts') and empirical_regimes is not None:
        peak, trough = dd_timing['peak_idx'], dd_timing['trough_idx']
        r2_id = max(regime_ids)
        during = empirical_regimes[peak:trough+1]
        emp_r2_during_dd = np.sum(during == r2_id)
    console.print("[bold cyan]4. Empirical drawdown: regime composition[/bold cyan]")
    console.print(f"   Months in Regime 2 during empirical max DD window: {emp_r2_during_dd} of {dd_timing['duration_months']}")
    console.print()

    # 5. Variance at Regime 2 entry (with historical path)
    console.print("[bold cyan]5. Variance level when entering Regime 2 (historical path)[/bold cyan]")
    var_at_entry = simulate_with_variance_tracking(
        params_dict, transition_matrix, regime_ids, empirical_regimes, len(empirical_returns),
        use_historical_path=True, seed=42
    )
    theta_r2 = params_dict[max(regime_ids)][IDX['theta']]
    v0_r2 = params_dict[max(regime_ids)][IDX['v0']]
    if len(var_at_entry) > 0 and not np.all(np.isnan(var_at_entry)):
        var_valid = var_at_entry[~np.isnan(var_at_entry)]
        if len(var_valid) > 0:
            console.print(f"   Variance at R2 entry (historical path): mean={np.mean(var_valid):.4f}, min={np.min(var_valid):.4f}, max={np.max(var_valid):.4f}, n_entries={len(var_valid)}")
    else:
        console.print("   No transitions into Regime 2 in historical path (or path too short)")
    console.print(f"   Regime 2 θ (long-run): {theta_r2:.4f}, v0: {v0_r2:.4f}")
    console.print()

    # 6. Median vs distribution: "aggregation bias" and summary-stat choice
    sim_max_dds = np.array([compute_max_drawdown(r) for r, _ in simulated_paths])
    emp_max_dd = compute_max_drawdown(empirical_returns)
    pct_below_emp = np.mean(sim_max_dds <= emp_max_dd) * 100  # % of paths with max DD <= empirical
    console.print("[bold cyan]6. Median vs distribution: summary-stat and bounded-range[/bold cyan]")
    console.print("   [dim]Hypothesis: Using median could understate severity. Max DD is bounded [-100%, 0%].[/dim]")
    console.print(f"   Model max DD: mean={np.mean(sim_max_dds):.4f}, median={np.median(sim_max_dds):.4f}, std={np.std(sim_max_dds):.4f}")
    console.print(f"   Model max DD: min={np.min(sim_max_dds):.4f}, 1st pctl={np.percentile(sim_max_dds, 1):.4f}, 5th={np.percentile(sim_max_dds, 5):.4f}, 95th={np.percentile(sim_max_dds, 95):.4f}")
    console.print(f"   Skewness of max DD distribution (across paths): {stats.skew(sim_max_dds):.3f} (negative = left tail)")
    console.print(f"   Empirical -86.5%: {pct_below_emp:.1f}% of simulated paths have max DD <= empirical (i.e. as bad or worse)")
    n_severe = np.sum((sim_max_dds >= -1) & (sim_max_dds < -0.9))
    console.print(f"   Paths with max DD in [-100%, -90%): {n_severe} (worst single path: {np.min(sim_max_dds)*100:.1f}%)")
    if np.mean(sim_max_dds) < np.median(sim_max_dds) - 0.02:
        console.print("   [yellow]→ Mean << median: heavy left tail. Median understates 'typical' severity.[/yellow]")
    else:
        console.print("   [dim]→ Mean ≈ median: distribution not heavily skewed. Median is reasonable.[/dim]")
    if np.min(sim_max_dds) > -0.98:
        console.print("   [dim]→ Worst path is {:.1f}% from -100% bound. Bound not compressing.[/dim]".format((np.min(sim_max_dds) + 1) * 100))
    else:
        console.print("   [yellow]→ Paths very close to -100% bound; floor may be compressing tail.[/yellow]")
    console.print()

    # 7. Synthesis
    console.print(Panel(
        "[bold]Synthesis[/bold]\n\n"
        "• If empirical max DD occurred mostly in Regime 2 and the model's R2 distribution doesn't "
        "reach that severity → within-regime misspecification (jump/vol params may understate crises).\n\n"
        "• If correlation(pct in R2, max_dd) is strong → random switching under-samples crisis paths; "
        "historical regime sequence helps.\n\n"
        "• If empirical worst month is far in the model's tail → model may need heavier jumps or "
        "higher vol-of-vol to capture black swan months.\n\n"
        "• Median vs mean: If mean << median, the median understates typical severity (aggregation-like). "
        "Use mean or a lower percentile for stress-testing.\n\n"
        "• Bounded [-100%, 0%]: The bound only matters if model would produce <-100%; we'd see pile-up "
        "at -100%. If min >> -1, the bound is not the culprit.",
        title="Interpretation",
        border_style="yellow"
    ))
    console.print()

# ============================================================================
# COMPARISON AND OUTPUT
# ============================================================================

def compare_empirical_vs_model(empirical_returns, simulated_stats):
    """Compare empirical statistics to simulated model statistics.
    
    Args:
        empirical_returns: Array of empirical log returns
        simulated_stats: List of dicts with statistics from simulations
    
    Returns:
        dict with comparison results
    """
    # Compute empirical statistics
    emp_stats = compute_statistics(empirical_returns)
    
    # Extract simulated statistics
    sim_cagrs = [s['cagr'] for s in simulated_stats]
    sim_vols = [s['vol'] for s in simulated_stats]
    sim_skews = [s['skew'] for s in simulated_stats]
    sim_kurts = [s['kurt'] for s in simulated_stats]
    sim_max_dds = [s['max_dd'] for s in simulated_stats]
    
    # Compute medians
    medians = {
        'cagr': np.median(sim_cagrs),
        'vol': np.median(sim_vols),
        'skew': np.median(sim_skews),
        'kurt': np.median(sim_kurts),
        'max_dd': np.median(sim_max_dds)
    }
    
    # Compute percentiles for confidence intervals
    percentiles = {
        'cagr': (np.percentile(sim_cagrs, 5), np.percentile(sim_cagrs, 95)),
        'vol': (np.percentile(sim_vols, 5), np.percentile(sim_vols, 95)),
        'skew': (np.percentile(sim_skews, 5), np.percentile(sim_skews, 95)),
        'kurt': (np.percentile(sim_kurts, 5), np.percentile(sim_kurts, 95)),
        'max_dd': (np.percentile(sim_max_dds, 5), np.percentile(sim_max_dds, 95))
    }
    
    return {
        'empirical': emp_stats,
        'model_median': medians,
        'model_percentiles': percentiles,
        'all_simulations': simulated_stats
    }

def compute_per_regime_statistics(returns, regime_ids):
    """Compute statistics for each regime separately.
    
    Args:
        returns: Array of returns
        regime_ids: Array of regime IDs (same length as returns)
    
    Returns:
        dict: {regime_id: stats_dict} for each regime
    """
    regime_stats = {}
    
    # Normalize regime IDs to integers
    regime_ids_normalized = np.array([int(rid) for rid in regime_ids])
    
    for regime_id in np.unique(regime_ids_normalized):
        mask = regime_ids_normalized == regime_id
        regime_returns = returns[mask]
        
        if len(regime_returns) > 0:
            regime_stats[int(regime_id)] = compute_statistics(regime_returns)
    
    return regime_stats

def compare_per_regime(empirical_returns, empirical_regimes, simulated_paths, regime_ids):
    """Compare empirical vs model statistics per regime.
    
    Args:
        empirical_returns: Array of empirical returns
        empirical_regimes: Array of empirical regime IDs
        simulated_paths: List of (returns, regimes) tuples from simulations
        regime_ids: List of all regime IDs
    
    Returns:
        dict: {regime_id: comparison_dict} for each regime
    """
    # Normalize empirical regime IDs
    empirical_regimes_normalized = np.array([int(rid) for rid in empirical_regimes])
    
    # Compute empirical statistics per regime
    emp_regime_stats = compute_per_regime_statistics(empirical_returns, empirical_regimes_normalized)
    
    # Compute simulated statistics per regime
    sim_regime_stats = {rid: [] for rid in regime_ids}
    
    for returns, regimes in simulated_paths:
        # Normalize simulated regime IDs
        regimes_normalized = np.array([int(rid) for rid in regimes])
        sim_stats = compute_per_regime_statistics(returns, regimes_normalized)
        for rid, stats in sim_stats.items():
            if rid in sim_regime_stats:
                sim_regime_stats[rid].append(stats)
    
    # Compute medians and percentiles per regime
    regime_comparisons = {}
    
    for rid in regime_ids:
        if rid not in emp_regime_stats:
            continue
        
        emp_stats = emp_regime_stats[rid]
        sim_stats_list = sim_regime_stats.get(rid, [])
        
        if len(sim_stats_list) == 0:
            continue
        
        # Extract statistics
        sim_cagrs = [s['cagr'] for s in sim_stats_list]
        sim_vols = [s['vol'] for s in sim_stats_list]
        sim_skews = [s['skew'] for s in sim_stats_list]
        sim_kurts = [s['kurt'] for s in sim_stats_list]
        sim_max_dds = [s['max_dd'] for s in sim_stats_list]
        
        medians = {
            'cagr': np.median(sim_cagrs),
            'vol': np.median(sim_vols),
            'skew': np.median(sim_skews),
            'kurt': np.median(sim_kurts),
            'max_dd': np.median(sim_max_dds)
        }
        
        percentiles = {
            'cagr': (np.percentile(sim_cagrs, 5), np.percentile(sim_cagrs, 95)),
            'vol': (np.percentile(sim_vols, 5), np.percentile(sim_vols, 95)),
            'skew': (np.percentile(sim_skews, 5), np.percentile(sim_skews, 95)),
            'kurt': (np.percentile(sim_kurts, 5), np.percentile(sim_kurts, 95)),
            'max_dd': (np.percentile(sim_max_dds, 5), np.percentile(sim_max_dds, 95))
        }
        
        regime_comparisons[rid] = {
            'empirical': emp_stats,
            'model_median': medians,
            'model_percentiles': percentiles,
            'n_empirical': np.sum(empirical_regimes_normalized == rid),
            'n_simulations': len(sim_stats_list)
        }
    
    return regime_comparisons

def display_comparison_table(comparison_results, title="Empirical vs Model Statistics Comparison"):
    """Display comparison table of empirical vs model statistics."""
    emp = comparison_results['empirical']
    med = comparison_results['model_median']
    pct = comparison_results['model_percentiles']
    
    table = Table(title=title)
    table.add_column("Statistic", style="cyan", width=20)
    table.add_column("Empirical", justify="right", style="green", width=15)
    table.add_column("Model Median", justify="right", style="blue", width=15)
    table.add_column("5th Percentile", justify="right", style="dim", width=15)
    table.add_column("95th Percentile", justify="right", style="dim", width=15)
    table.add_column("Difference", justify="right", style="yellow", width=15)
    table.add_column("% Error", justify="right", style="red", width=12)
    
    stats_names = {
        'cagr': 'CAGR (annual)',
        'vol': 'Annual Volatility',
        'skew': 'Skewness',
        'kurt': 'Kurtosis (excess)',
        'max_dd': 'Max Drawdown'
    }
    
    for stat_key, stat_name in stats_names.items():
        emp_val = emp[stat_key]
        med_val = med[stat_key]
        pct_5 = pct[stat_key][0]
        pct_95 = pct[stat_key][1]
        diff = med_val - emp_val
        
        # Compute percentage error
        if abs(emp_val) > 1e-6:
            pct_error = (diff / abs(emp_val)) * 100
        else:
            pct_error = diff * 100 if abs(diff) > 1e-6 else 0.0
        
        # Format values
        if stat_key == 'cagr' or stat_key == 'vol':
            emp_str = f"{emp_val:.4f}"
            med_str = f"{med_val:.4f}"
            pct_5_str = f"{pct_5:.4f}"
            pct_95_str = f"{pct_95:.4f}"
            diff_str = f"{diff:.4f}"
        elif stat_key == 'max_dd':
            emp_str = f"{emp_val:.4f}"
            med_str = f"{med_val:.4f}"
            pct_5_str = f"{pct_5:.4f}"
            pct_95_str = f"{pct_95:.4f}"
            diff_str = f"{diff:.4f}"
        else:
            emp_str = f"{emp_val:.3f}"
            med_str = f"{med_val:.3f}"
            pct_5_str = f"{pct_5:.3f}"
            pct_95_str = f"{pct_95:.3f}"
            diff_str = f"{diff:.3f}"
        
        pct_error_str = f"{pct_error:.2f}%"
        
        # Color code based on error
        if abs(pct_error) < 5:
            error_style = "green"
        elif abs(pct_error) < 15:
            error_style = "yellow"
        else:
            error_style = "red"
        
        table.add_row(
            stat_name,
            emp_str,
            med_str,
            pct_5_str,
            pct_95_str,
            f"[{error_style}]{diff_str}[/{error_style}]",
            f"[{error_style}]{pct_error_str}[/{error_style}]"
        )
    
    console.print(table)

def display_per_regime_tables(regime_comparisons):
    """Display comparison tables for each regime."""
    for regime_id in sorted(regime_comparisons.keys()):
        comp = regime_comparisons[regime_id]
        title = f"Regime {regime_id} Comparison (Empirical n={comp['n_empirical']}, Simulations n={comp['n_simulations']})"
        display_comparison_table(comp, title=title)
        console.print()  # Blank line between regimes

def display_moment_matching_quality(moment_matching_info, regime_ids):
    """Display moment matching quality diagnostics from the fitting process.
    
    Args:
        moment_matching_info: dict with moment matching results per regime
        regime_ids: List of regime IDs
    """
    console.print("\n[bold cyan]Moment Matching Quality (from Parameter Fitting)[/bold cyan]")
    console.print("=" * 80)
    
    table = Table(title="Moment Matching Errors from Fitting Process")
    table.add_column("Regime", style="cyan", width=8)
    table.add_column("N Obs", justify="right", width=8)
    table.add_column("Mean Error %", justify="right", width=12)
    table.add_column("Std Error %", justify="right", width=12)
    table.add_column("Skew Error %", justify="right", width=12)
    table.add_column("Kurt Error %", justify="right", width=12)
    table.add_column("Max DD Error %", justify="right", width=15)
    table.add_column("Objective", justify="right", width=12)
    
    for regime_id in sorted(regime_ids):
        if regime_id not in moment_matching_info:
            continue
        
        info = moment_matching_info[regime_id]
        
        # Color code based on error magnitude
        def get_error_style(pct_error):
            if abs(pct_error) < 1:
                return "green"
            elif abs(pct_error) < 5:
                return "yellow"
            else:
                return "red"
        
        mean_style = get_error_style(info['mean_error_pct'])
        std_style = get_error_style(info['std_error_pct'])
        skew_style = get_error_style(info['skew_error_pct'])
        kurt_style = get_error_style(info['kurt_error_pct'])
        dd_style = get_error_style(info['max_dd_error_pct'])
        
        table.add_row(
            f"Regime {int(regime_id)}",
            str(info['n_obs']),
            f"[{mean_style}]{info['mean_error_pct']:.2f}%[/{mean_style}]",
            f"[{std_style}]{info['std_error_pct']:.2f}%[/{std_style}]",
            f"[{skew_style}]{info['skew_error_pct']:.2f}%[/{skew_style}]",
            f"[{kurt_style}]{info['kurt_error_pct']:.2f}%[/{kurt_style}]",
            f"[{dd_style}]{info['max_dd_error_pct']:.2f}%[/{dd_style}]",
            f"{info['objective']:.2e}"
        )
    
    console.print(table)
    
    # Check for potential issues
    console.print("\n[bold cyan]Moment Matching Quality Assessment:[/bold cyan]")
    issues = []
    
    for regime_id in sorted(regime_ids):
        if regime_id not in moment_matching_info:
            continue
        
        info = moment_matching_info[regime_id]
        
        # Check for large errors
        if abs(info['mean_error_pct']) > 5:
            issues.append(f"Regime {int(regime_id)}: Mean error {info['mean_error_pct']:.2f}% > 5%")
        if abs(info['std_error_pct']) > 5:
            issues.append(f"Regime {int(regime_id)}: Std error {info['std_error_pct']:.2f}% > 5%")
        if abs(info['skew_error_pct']) > 10:
            issues.append(f"Regime {int(regime_id)}: Skew error {info['skew_error_pct']:.2f}% > 10%")
        if abs(info['kurt_error_pct']) > 10:
            issues.append(f"Regime {int(regime_id)}: Kurt error {info['kurt_error_pct']:.2f}% > 10%")
        if abs(info['max_dd_error_pct']) > 20:
            issues.append(f"Regime {int(regime_id)}: Max DD error {info['max_dd_error_pct']:.2f}% > 20%")
    
    if issues:
        console.print("[yellow]⚠ Potential Issues Detected:[/yellow]")
        for issue in issues:
            console.print(f"  • {issue}")
        console.print("\n[dim]Note: Large moment matching errors may indicate parameter estimation issues.[/dim]")
        console.print("[dim]Consider re-running moment matching with different settings or bounds.[/dim]")
    else:
        console.print("[green]✓ Moment matching quality looks good (all errors within acceptable ranges)[/green]")
    
    console.print()

def display_parameter_estimates(params_dict, regime_ids):
    """Display Bates model parameter estimates for each regime.
    
    Args:
        params_dict: {regime_id: params_array} for each regime
        regime_ids: List of regime IDs
    """
    table = Table(title="Bates Model Parameter Estimates by Regime")
    table.add_column("Regime", style="cyan", width=8)
    table.add_column("μ (annual)", justify="right", style="green", width=12)
    table.add_column("κ", justify="right", width=10)
    table.add_column("θ (annual)", justify="right", width=12)
    table.add_column("ν", justify="right", width=10)
    table.add_column("ρ", justify="right", width=10)
    table.add_column("v₀", justify="right", width=10)
    table.add_column("λ (annual)", justify="right", width=12)
    table.add_column("μ_J", justify="right", width=10)
    table.add_column("σ_J", justify="right", width=10)
    
    param_names = ['mu', 'kappa', 'theta', 'nu', 'rho', 'v0', 'lam', 'mu_J', 'sigma_J']
    
    for regime_id in sorted(regime_ids):
        if regime_id not in params_dict:
            continue
        
        params = params_dict[regime_id]
        row_data = [f"Regime {regime_id}"]
        
        for param_name in param_names:
            param_idx = IDX[param_name]
            param_val = params[param_idx]
            
            # Format based on parameter type
            if param_name in ['mu', 'theta', 'v0', 'mu_J', 'sigma_J']:
                row_data.append(f"{param_val:.4f}")
            elif param_name in ['kappa', 'nu', 'lam']:
                row_data.append(f"{param_val:.3f}")
            elif param_name == 'rho':
                row_data.append(f"{param_val:.3f}")
            else:
                row_data.append(f"{param_val:.4f}")
        
        table.add_row(*row_data)
    
    console.print(table)
    
    # Also display parameter interpretation
    console.print("\n[bold cyan]Parameter Interpretation:[/bold cyan]")
    console.print("  μ (mu): Annual drift rate")
    console.print("  κ (kappa): Mean reversion speed of variance")
    console.print("  θ (theta): Long-run variance (annual)")
    console.print("  ν (nu): Volatility of volatility")
    console.print("  ρ (rho): Correlation between price and variance processes")
    console.print("  v₀: Initial variance")
    console.print("  λ (lambda): Jump intensity per year")
    console.print("  μ_J: Mean jump size")
    console.print("  σ_J: Jump size volatility")
    console.print()
    
    # Parameter comparison analysis
    if len(params_dict) > 1:
        console.print("[bold cyan]Parameter Comparison Across Regimes:[/bold cyan]")
        console.print("=" * 80)
        
        param_names = ['mu', 'kappa', 'theta', 'nu', 'rho', 'v0', 'lam', 'mu_J', 'sigma_J']
        param_labels = {
            'mu': 'Drift (μ)',
            'kappa': 'Mean Reversion (κ)',
            'theta': 'Long-run Variance (θ)',
            'nu': 'Vol-of-Vol (ν)',
            'rho': 'Correlation (ρ)',
            'v0': 'Initial Variance (v₀)',
            'lam': 'Jump Intensity (λ)',
            'mu_J': 'Jump Mean (μ_J)',
            'sigma_J': 'Jump Vol (σ_J)'
        }
        
        for param_name in param_names:
            param_idx = IDX[param_name]
            values = [params_dict[rid][param_idx] for rid in sorted(regime_ids) if rid in params_dict]
            regimes_list = [rid for rid in sorted(regime_ids) if rid in params_dict]
            
            if len(values) > 1:
                min_val = min(values)
                max_val = max(values)
                min_regime = regimes_list[values.index(min_val)]
                max_regime = regimes_list[values.index(max_val)]
                range_val = max_val - min_val
                
                # Format based on parameter type
                if param_name in ['mu', 'theta', 'v0', 'mu_J', 'sigma_J']:
                    fmt = ".4f"
                else:
                    fmt = ".3f"
                
                console.print(f"  {param_labels[param_name]}:")
                console.print(f"    Range: [{min_val:{fmt}}, {max_val:{fmt}}] (span: {range_val:{fmt}})")
                console.print(f"    Min: Regime {min_regime} = {min_val:{fmt}}")
                console.print(f"    Max: Regime {max_regime} = {max_val:{fmt}}")
        
        console.print()

# ============================================================================
# PLOTTING
# ============================================================================

def plot_historical_vs_simulations(empirical_returns, simulated_paths, output_path, n_sample_paths=50):
    """Plot historical returns vs simulated paths.
    
    Args:
        empirical_returns: Array of empirical returns
        simulated_paths: List of (returns, regimes) tuples
        output_path: Path to save the plot
        n_sample_paths: Number of simulation paths to plot (for clarity)
    """
    # Set dark background style
    plt.style.use('dark_background')
    
    fig = plt.figure(figsize=(16, 10), facecolor='black')
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.3, height_ratios=[2, 1, 1])
    
    # Convert log returns to cumulative prices: P_t = exp(cumsum(returns))
    empirical_prices = np.exp(np.cumsum(empirical_returns)) * 100  # Start at 100
    
    # Plot 1: Cumulative price paths
    ax1 = fig.add_subplot(gs[0])
    
    # Plot historical
    time_empirical = np.arange(len(empirical_prices)) / PERIODS_PER_YEAR
    ax1.plot(time_empirical, empirical_prices, linewidth=2.5, color='cyan', 
             label='Historical', alpha=0.9, zorder=10)
    
    # Plot sample of simulated paths
    n_paths_to_plot = min(n_sample_paths, len(simulated_paths))
    indices = np.linspace(0, len(simulated_paths) - 1, n_paths_to_plot, dtype=int)
    
    for idx in indices:
        returns, _ = simulated_paths[idx]
        prices = np.exp(np.cumsum(returns)) * 100
        time_sim = np.arange(len(prices)) / PERIODS_PER_YEAR
        ax1.plot(time_sim, prices, linewidth=0.8, color='orange', alpha=0.15, zorder=1)
    
    # Plot median and percentiles of all simulations
    all_prices = []
    min_length = min(len(empirical_returns), min(len(r) for r, _ in simulated_paths))
    
    for returns, _ in simulated_paths:
        prices = np.exp(np.cumsum(returns[:min_length])) * 100
        all_prices.append(prices)
    
    all_prices = np.array(all_prices)
    median_prices = np.median(all_prices, axis=0)
    pct_5_prices = np.percentile(all_prices, 5, axis=0)
    pct_95_prices = np.percentile(all_prices, 95, axis=0)
    
    time_median = np.arange(len(median_prices)) / PERIODS_PER_YEAR
    ax1.plot(time_median, median_prices, linewidth=2, color='yellow', 
             label='Model Median', linestyle='--', alpha=0.8, zorder=5)
    ax1.fill_between(time_median, pct_5_prices, pct_95_prices, 
                     color='orange', alpha=0.2, label='Model 5th-95th Percentile', zorder=2)
    
    ax1.set_xlabel('Years', fontsize=12, color='white')
    ax1.set_ylabel('Cumulative Price (Index = 100, Log Scale)', fontsize=12, color='white')
    ax1.set_title('Historical vs Simulated Price Paths', fontsize=14, fontweight='bold', color='white')
    ax1.set_yscale('log')  # Use log scale for better visualization of growth
    ax1.legend(fontsize=11, facecolor='black', edgecolor='white', labelcolor='white', loc='upper left')
    ax1.grid(True, alpha=0.3, color='gray')
    ax1.set_facecolor('black')
    ax1.tick_params(colors='white')
    
    # Plot 2: Returns distribution comparison
    ax2 = fig.add_subplot(gs[1])
    
    # Flatten all simulated returns
    all_sim_returns = np.concatenate([r for r, _ in simulated_paths])
    
    # Histogram comparison
    bins = np.linspace(min(np.min(empirical_returns), np.min(all_sim_returns)),
                      max(np.max(empirical_returns), np.max(all_sim_returns)), 50)
    
    ax2.hist(empirical_returns, bins=bins, alpha=0.6, label='Historical', 
            density=True, color='cyan', edgecolor='white', linewidth=0.5)
    ax2.hist(all_sim_returns, bins=bins, alpha=0.6, label='Simulated (all paths)', 
            density=True, color='orange', edgecolor='white', linewidth=0.5)
    
    ax2.set_xlabel('Log Return', fontsize=12, color='white')
    ax2.set_ylabel('Density', fontsize=12, color='white')
    ax2.set_title('Returns Distribution Comparison', fontsize=13, fontweight='bold', color='white')
    ax2.legend(fontsize=11, facecolor='black', edgecolor='white', labelcolor='white')
    ax2.grid(True, alpha=0.3, color='gray')
    ax2.set_facecolor('black')
    ax2.tick_params(colors='white')
    
    # Plot 3: Drawdown comparison
    ax3 = fig.add_subplot(gs[2])
    
    # Compute drawdowns for historical
    emp_peak = np.maximum.accumulate(empirical_prices)
    emp_dd = (empirical_prices - emp_peak) / emp_peak * 100
    time_emp_dd = np.arange(len(emp_dd)) / PERIODS_PER_YEAR
    
    ax3.plot(time_emp_dd, emp_dd, linewidth=2, color='cyan', label='Historical Drawdown', alpha=0.9)
    
    # Compute drawdowns for median simulation
    sim_peak = np.maximum.accumulate(median_prices)
    sim_dd = (median_prices - sim_peak) / sim_peak * 100
    
    ax3.plot(time_median, sim_dd, linewidth=2, color='yellow', 
             label='Model Median Drawdown', linestyle='--', alpha=0.8)
    
    # Fill percentile range
    sim_pct_5_peak = np.maximum.accumulate(pct_5_prices)
    sim_pct_5_dd = (pct_5_prices - sim_pct_5_peak) / sim_pct_5_peak * 100
    sim_pct_95_peak = np.maximum.accumulate(pct_95_prices)
    sim_pct_95_dd = (pct_95_prices - sim_pct_95_peak) / sim_pct_95_peak * 100
    
    ax3.fill_between(time_median, sim_pct_5_dd, sim_pct_95_dd, 
                     color='orange', alpha=0.2, label='Model 5th-95th Percentile')
    
    ax3.set_xlabel('Years', fontsize=12, color='white')
    ax3.set_ylabel('Drawdown (%)', fontsize=12, color='white')
    ax3.set_title('Drawdown Comparison', fontsize=13, fontweight='bold', color='white')
    ax3.legend(fontsize=11, facecolor='black', edgecolor='white', labelcolor='white', loc='lower left')
    ax3.grid(True, alpha=0.3, color='gray')
    ax3.set_facecolor('black')
    ax3.tick_params(colors='white')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    console.print(f"[green]✓ Plot saved to: {output_path}[/green]")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run Monte Carlo simulation and comparison."""
    parser = argparse.ArgumentParser(description='Monte Carlo simulation with regime-switching Bates model')
    parser.add_argument('--n-simulations', type=int, default=1000,
                       help='Number of Monte Carlo simulations (default: 1000)')
    parser.add_argument('--n-periods', type=int, default=None,
                       help='Number of periods per simulation (default: match empirical data length)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--bates-csv', type=str, 
                       default='data/Bates_per_regime_Moment_Matching_results.csv',
                       help='Path to Bates parameters CSV')
    parser.add_argument('--transition-csv', type=str,
                       default='data/HMM_transition_matrix.csv',
                       help='Path to HMM transition matrix CSV')
    parser.add_argument('--returns-csv', type=str,
                       default='data/regime_classification_nominal_returns.csv',
                       help='Path to empirical returns CSV')
    parser.add_argument('--plot', action='store_true',
                       help='Generate and save comparison plots')
    parser.add_argument('--n-sample-paths', type=int, default=50,
                       help='Number of simulation paths to plot (default: 50)')
    parser.add_argument('--use-expected-variance', action='store_true',
                       help='Use expected variance in mean calculation (for consistency with analytical formulas)')
    parser.add_argument('--simulation-based-validation', action='store_true',
                       help='Use simulation-based moment computation for validation (slower but more accurate)')
    parser.add_argument('--historical-regimes', action='store_true',
                       help='Run diagnostic: use exact historical regime sequence to test if regime timing explains tail risk gap')
    parser.add_argument('--investigate', action='store_true',
                       help='Deep investigation: timing of max DD, regime duration correlation, worst-period comparison, variance at regime switches')
    parser.add_argument('--no-crisis-boost', action='store_true',
                       help='Disable crisis tail boost (use raw fitted params; will understate tail risk)')
    args = parser.parse_args()
    
    console.print("[bold cyan]Monte Carlo Simulation: Regime-Switching Bates Model[/bold cyan]")
    console.print("=" * 80)
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    
    bates_path = args.bates_csv
    if not os.path.exists(bates_path):
        bates_path = os.path.join(grandparent_dir, args.bates_csv)
    
    transition_path = args.transition_csv
    if not os.path.exists(transition_path):
        transition_path = os.path.join(grandparent_dir, args.transition_csv)
    
    returns_path = args.returns_csv
    if not os.path.exists(returns_path):
        returns_path = os.path.join(grandparent_dir, args.returns_csv)
    
    # Load data
    console.print("\n[cyan]Loading data...[/cyan]")
    console.print(f"  Bates parameters: {bates_path}")
    console.print(f"  Transition matrix: {transition_path}")
    console.print(f"  Empirical returns: {returns_path}")
    
    params_dict, moment_matching_info = load_bates_parameters(bates_path)
    transition_matrix, regime_ids = load_transition_matrix(transition_path)
    empirical_returns, empirical_regimes, empirical_dates = load_empirical_data(returns_path)
    
    # Normalize regime IDs to integers for consistency
    params_dict_normalized = {}
    for regime_id, params in params_dict.items():
        normalized_id = int(regime_id)
        params_dict_normalized[normalized_id] = params
    
    regime_ids_normalized = [int(rid) for rid in regime_ids]
    params_dict = params_dict_normalized
    regime_ids = regime_ids_normalized
    
    # Validate that all regimes in params_dict are in transition matrix
    missing_regimes = set(params_dict.keys()) - set(regime_ids)
    if missing_regimes:
        console.print(f"[red]Error: Regimes {missing_regimes} in parameters but not in transition matrix[/red]")
        return
    
    console.print(f"[green]✓ Loaded {len(params_dict)} regimes[/green]")
    console.print(f"[green]✓ Transition matrix: {transition_matrix.shape}[/green]")
    console.print(f"[green]✓ Empirical data: {len(empirical_returns)} periods[/green]")
    console.print(f"[dim]  Regime IDs: {sorted(regime_ids)}[/dim]")
    
    # Display moment matching quality first (to check if parameters are well-fitted)
    display_moment_matching_quality(moment_matching_info, regime_ids)
    
    # Display parameter estimates
    console.print("\n[bold cyan]Bates Model Parameter Estimates[/bold cyan]")
    console.print("=" * 80)
    display_parameter_estimates(params_dict, regime_ids)
    
    # Display transition matrix
    console.print("\n[bold cyan]HMM Transition Matrix[/bold cyan]")
    console.print("=" * 80)
    trans_table = Table(title="Transition Probabilities")
    trans_table.add_column("From → To", style="cyan", width=15)
    for rid in regime_ids:
        trans_table.add_column(f"Regime {rid}", justify="right", width=12)
    
    for i, from_rid in enumerate(regime_ids):
        row_data = [f"Regime {from_rid}"]
        for j, to_rid in enumerate(regime_ids):
            prob = transition_matrix[i, j]
            row_data.append(f"{prob:.4f}")
        trans_table.add_row(*row_data)
    
    console.print(trans_table)
    console.print()
    
    # Set n_periods if not specified
    n_periods = args.n_periods if args.n_periods is not None else len(empirical_returns)
    
    # Remind user about plot flag
    if not args.plot:
        console.print("[dim]💡 Tip: Use --plot flag to generate comparison plots[/dim]")
        console.print()
    
    # Run Monte Carlo simulation
    console.print(f"\n[cyan]Running {args.n_simulations} Monte Carlo simulations...[/cyan]")
    console.print(f"  Periods per simulation: {n_periods}")
    console.print(f"  Random seed: {args.seed}")
    
    # Get paths if plotting or per-regime analysis is needed
    return_paths = args.plot or True  # Always return paths for per-regime analysis
    crisis_boost = (0, 0) if args.no_crisis_boost else None  # None = use CRISIS_JUMP_BOOST default
    result = run_monte_carlo_simulation(
        params_dict, transition_matrix, regime_ids,
        n_simulations=args.n_simulations,
        n_periods=n_periods,
        seed=args.seed,
        return_paths=return_paths,
        use_expected_variance=args.use_expected_variance,
        crisis_jump_boost=crisis_boost
    )
    
    if return_paths:
        simulated_stats, simulated_paths = result
    else:
        simulated_stats = result
        simulated_paths = None
    
    # Compare results (overall)
    console.print("\n[cyan]Comparing empirical vs model statistics (overall)...[/cyan]")
    comparison_results = compare_empirical_vs_model(empirical_returns, simulated_stats)
    
    # Display overall results
    console.print("\n")
    display_comparison_table(comparison_results)
    
    # Historical regime path diagnostic (tests "breaking time-series dependence" hypothesis)
    if args.historical_regimes and empirical_regimes is not None:
        console.print("\n[bold cyan]Historical Regime Path Diagnostic[/bold cyan]")
        console.print("=" * 80)
        console.print("[dim]Using EXACT historical HMM regime sequence; drawing returns from Bates per regime.[/dim]")
        console.print("[dim]Tests: Is the tail risk gap from (a) random regime switching or (b) within-regime misspecification?[/dim]")
        console.print()
        hist_boost = (0, 0) if args.no_crisis_boost else CRISIS_JUMP_BOOST
        hist_paths = simulate_with_historical_regimes(
            params_dict, empirical_regimes, n_periods,
            n_simulations=500, seed=args.seed, crisis_jump_boost=hist_boost,
            use_expected_variance=args.use_expected_variance
        )
        hist_stats = [compute_statistics(r) for r, _ in hist_paths]
        hist_max_dds = [s['max_dd'] for s in hist_stats]
        hist_cagrs = [s['cagr'] for s in hist_stats]
        hist_kurts = [s['kurt'] for s in hist_stats]
        emp_max_dd = comparison_results['empirical']['max_dd']
        rand_median_dd = comparison_results['model_median']['max_dd']
        hist_median_dd = np.median(hist_max_dds)
        hist_p5_dd = np.percentile(hist_max_dds, 5)
        hist_p95_dd = np.percentile(hist_max_dds, 95)
        hist_table = Table(title="Historical Regime Sequence vs Random Regime Switching")
        hist_table.add_column("Metric", style="cyan", width=25)
        hist_table.add_column("Empirical", justify="right", width=14)
        hist_table.add_column("Model (random regimes)", justify="right", width=18)
        hist_table.add_column("Model (historical regimes)", justify="right", width=22)
        hist_table.add_row("Max Drawdown", f"{emp_max_dd:.4f}", f"{rand_median_dd:.4f}", f"{hist_median_dd:.4f} (5th-95th: {hist_p5_dd:.3f} to {hist_p95_dd:.3f})")
        hist_table.add_row("CAGR (annual)", f"{comparison_results['empirical']['cagr']:.4f}", f"{comparison_results['model_median']['cagr']:.4f}", f"{np.median(hist_cagrs):.4f}")
        hist_table.add_row("Kurtosis (excess)", f"{comparison_results['empirical']['kurt']:.2f}", f"{comparison_results['model_median']['kurt']:.2f}", f"{np.median(hist_kurts):.2f}")
        console.print(hist_table)
        console.print()
        if abs(hist_median_dd - emp_max_dd) < abs(rand_median_dd - emp_max_dd):
            console.print("[yellow]→ Historical regime path gets CLOSER to empirical. Regime timing / sequence matters;[/yellow]")
            console.print("[yellow]  random switching breaks time-series dependence and understates tail risk.[/yellow]")
        else:
            console.print("[yellow]→ Historical regime path does NOT close the gap. The issue is likely within-regime[/yellow]")
            console.print("[yellow]  model misspecification (Bates may not capture crisis dynamics well enough).[/yellow]")
        console.print()
    
    # Deep investigation
    if args.investigate and simulated_paths is not None:
        run_deep_investigation(
            params_dict, transition_matrix, regime_ids, empirical_returns,
            empirical_regimes, simulated_paths, dates=empirical_dates
        )
    
    # Per-regime comparison
    if empirical_regimes is not None and simulated_paths is not None and len(empirical_regimes) == len(empirical_returns):
        console.print("\n[cyan]Comparing empirical vs model statistics (per regime)...[/cyan]")
        regime_comparisons = compare_per_regime(
            empirical_returns, empirical_regimes, simulated_paths, regime_ids
        )
        
        if regime_comparisons:
            console.print("\n")
            display_per_regime_tables(regime_comparisons)
            
            # Compare moment matching predictions vs simulation results
            console.print("\n[bold cyan]Moment Matching Predictions vs Simulation Results[/bold cyan]")
            console.print("=" * 80)
            console.print("[dim]Comparing what moment matching predicted vs what we get in full simulations[/dim]")
            console.print()
            
            diagnostic_table = Table(title="Moment Matching Fit vs Simulation Reality")
            diagnostic_table.add_column("Regime", style="cyan", width=8)
            diagnostic_table.add_column("Metric", style="yellow", width=20)
            diagnostic_table.add_column("Moment Match Predicted", justify="right", style="blue", width=18)
            diagnostic_table.add_column("Simulation Median", justify="right", style="green", width=18)
            diagnostic_table.add_column("Difference", justify="right", style="red", width=15)
            
            for regime_id in sorted(regime_comparisons.keys()):
                if regime_id not in moment_matching_info:
                    continue
                
                mm_info = moment_matching_info[regime_id]
                sim_comp = regime_comparisons[regime_id]
                
                # Compare CAGR (annualized mean)
                # Moment matching CSV has monthly mean, convert to annual
                mm_cagr = mm_info['model_mean'] * PERIODS_PER_YEAR
                sim_cagr = sim_comp['model_median']['cagr']
                cagr_diff = sim_cagr - mm_cagr
                cagr_pct_diff = (cagr_diff / abs(mm_cagr) * 100) if abs(mm_cagr) > 1e-6 else 0.0
                diagnostic_table.add_row(
                    f"Regime {regime_id}",
                    "CAGR (annual)",
                    f"{mm_cagr:.4f}",
                    f"{sim_cagr:.4f}",
                    f"{cagr_diff:.4f} ({cagr_pct_diff:+.1f}%)"
                )
                
                # Compare Volatility
                # Moment matching CSV has monthly std, convert to annual
                mm_vol = mm_info['model_std'] * math.sqrt(PERIODS_PER_YEAR)
                sim_vol = sim_comp['model_median']['vol']
                vol_diff = sim_vol - mm_vol
                vol_pct_diff = (vol_diff / abs(mm_vol) * 100) if abs(mm_vol) > 1e-6 else 0.0
                diagnostic_table.add_row(
                    "",
                    "Volatility (annual)",
                    f"{mm_vol:.4f}",
                    f"{sim_vol:.4f}",
                    f"{vol_diff:.4f} ({vol_pct_diff:+.1f}%)"
                )
                
                # Compare Skewness (same for monthly/annual)
                mm_skew = mm_info['model_skew']
                sim_skew = sim_comp['model_median']['skew']
                skew_diff = sim_skew - mm_skew
                skew_pct_diff = (skew_diff / abs(mm_skew) * 100) if abs(mm_skew) > 1e-6 else 0.0
                diagnostic_table.add_row(
                    "",
                    "Skewness",
                    f"{mm_skew:.3f}",
                    f"{sim_skew:.3f}",
                    f"{skew_diff:.3f} ({skew_pct_diff:+.1f}%)"
                )
                
                # Compare Kurtosis (same for monthly/annual)
                mm_kurt = mm_info['model_kurt']
                sim_kurt = sim_comp['model_median']['kurt']
                kurt_diff = sim_kurt - mm_kurt
                kurt_pct_diff = (kurt_diff / abs(mm_kurt) * 100) if abs(mm_kurt) > 1e-6 else 0.0
                diagnostic_table.add_row(
                    "",
                    "Kurtosis (excess)",
                    f"{mm_kurt:.3f}",
                    f"{sim_kurt:.3f}",
                    f"{kurt_diff:.3f} ({kurt_pct_diff:+.1f}%)"
                )
                
                # Compare Max Drawdown (same for monthly/annual)
                mm_dd = mm_info['model_max_dd']
                sim_dd = sim_comp['model_median']['max_dd']
                dd_diff = sim_dd - mm_dd
                dd_pct_diff = (dd_diff / abs(mm_dd) * 100) if abs(mm_dd) > 1e-6 else 0.0
                diagnostic_table.add_row(
                    "",
                    "Max Drawdown",
                    f"{mm_dd:.4f}",
                    f"{sim_dd:.4f}",
                    f"{dd_diff:.4f} ({dd_pct_diff:+.1f}%)"
                )
                
                diagnostic_table.add_row("", "", "", "", "")  # Spacer
            
            console.print(diagnostic_table)
            console.print("\n[dim]Note: Large differences indicate that regime-switching behavior may differ from[/dim]")
            console.print("[dim]isolated regime behavior, or that moment matching analytical formulas may not[/dim]")
            console.print("[dim]perfectly match simulated behavior in the full regime-switching context.[/dim]")
            console.print()
            
            # Additional diagnostic: Test isolated regime simulations
            console.print("\n[bold cyan]Isolated Regime Test (No Switching)[/bold cyan]")
            console.print("=" * 80)
            console.print("[dim]Simulating each regime in isolation to see if parameters match expectations[/dim]")
            console.print()
            
            isolated_test_table = Table(title="Isolated Regime Simulations (2 paths × 500 periods = 1000 total, with 100-period burn-in)")
            isolated_test_table.add_column("Regime", style="cyan", width=8)
            isolated_test_table.add_column("Metric", style="yellow", width=20)
            isolated_test_table.add_column("Moment Match Predicted", justify="right", style="blue", width=18)
            isolated_test_table.add_column("Isolated Simulation", justify="right", style="green", width=18)
            isolated_test_table.add_column("Difference", justify="right", style="red", width=15)
            
            for regime_id in sorted(regime_comparisons.keys()):
                if regime_id not in moment_matching_info or regime_id not in params_dict:
                    continue
                
                # Simulate this regime in isolation (no switching)
                params = params_dict[regime_id]
                mm_info = moment_matching_info[regime_id]
                
                # Run isolated simulation with burn-in to reach equilibrium
                # Use longer paths and burn-in period for better convergence
                # Try with expected variance first (should match analytical formulas better)
                isolated_returns = []
                isolated_returns_expected = []  # With expected variance
                burn_in = 100  # Increased burn-in for better equilibrium
                path_length = 500  # Longer paths for better statistics
                
                for path_idx in range(2):  # 2 paths, 500 periods each = 1000 total periods
                    # Standard simulation (actual variance)
                    returns, _ = simulate_regime_switching_bates(
                        {regime_id: params},  # Only one regime
                        np.array([[1.0]]),  # Stay in same regime (100% probability)
                        [regime_id],
                        n_periods=path_length,
                        initial_regime=regime_id,
                        initial_variance=params[IDX['theta']],  # Start at equilibrium
                        seed=path_idx,  # Different seed per path
                        burn_in_periods=burn_in,
                        reset_variance_on_switch=False,
                        use_expected_variance=False
                    )
                    isolated_returns.extend(returns)
                    
                    # With expected variance (should match analytical formulas)
                    returns_exp, _ = simulate_regime_switching_bates(
                        {regime_id: params},
                        np.array([[1.0]]),
                        [regime_id],
                        n_periods=path_length,
                        initial_regime=regime_id,
                        initial_variance=params[IDX['theta']],
                        seed=path_idx,
                        burn_in_periods=burn_in,
                        reset_variance_on_switch=False,
                        use_expected_variance=True  # Use expected variance for consistency
                    )
                    isolated_returns_expected.extend(returns_exp)
                
                isolated_returns = np.array(isolated_returns)
                isolated_returns_expected = np.array(isolated_returns_expected)
                isolated_stats = compute_statistics(isolated_returns)
                isolated_stats_exp = compute_statistics(isolated_returns_expected)
                
                # Compare to moment matching predictions (use expected variance version)
                mm_cagr = mm_info['model_mean'] * PERIODS_PER_YEAR
                iso_cagr = isolated_stats['cagr']
                iso_cagr_exp = isolated_stats_exp['cagr']
                cagr_diff = iso_cagr - mm_cagr
                cagr_diff_exp = iso_cagr_exp - mm_cagr
                cagr_pct = (cagr_diff / abs(mm_cagr) * 100) if abs(mm_cagr) > 1e-6 else 0.0
                cagr_pct_exp = (cagr_diff_exp / abs(mm_cagr) * 100) if abs(mm_cagr) > 1e-6 else 0.0
                
                isolated_test_table.add_row(
                    f"Regime {regime_id}",
                    "CAGR (actual var)",
                    f"{mm_cagr:.4f}",
                    f"{iso_cagr:.4f}",
                    f"{cagr_diff:.4f} ({cagr_pct:+.1f}%)"
                )
                
                isolated_test_table.add_row(
                    "",
                    "CAGR (expected var)",
                    f"{mm_cagr:.4f}",
                    f"{iso_cagr_exp:.4f}",
                    f"{cagr_diff_exp:.4f} ({cagr_pct_exp:+.1f}%)"
                )
                
                mm_vol = mm_info['model_std'] * math.sqrt(PERIODS_PER_YEAR)
                iso_vol = isolated_stats['vol']
                iso_vol_exp = isolated_stats_exp['vol']
                vol_diff = iso_vol - mm_vol
                vol_diff_exp = iso_vol_exp - mm_vol
                vol_pct = (vol_diff / abs(mm_vol) * 100) if abs(mm_vol) > 1e-6 else 0.0
                vol_pct_exp = (vol_diff_exp / abs(mm_vol) * 100) if abs(mm_vol) > 1e-6 else 0.0
                
                isolated_test_table.add_row(
                    "",
                    "Volatility (actual)",
                    f"{mm_vol:.4f}",
                    f"{iso_vol:.4f}",
                    f"{vol_diff:.4f} ({vol_pct:+.1f}%)"
                )
                
                isolated_test_table.add_row(
                    "",
                    "Volatility (expected)",
                    f"{mm_vol:.4f}",
                    f"{iso_vol_exp:.4f}",
                    f"{vol_diff_exp:.4f} ({vol_pct_exp:+.1f}%)"
                )
                
                mm_skew = mm_info['model_skew']
                iso_skew = isolated_stats['skew']
                iso_skew_exp = isolated_stats_exp['skew']
                skew_diff = iso_skew - mm_skew
                skew_diff_exp = iso_skew_exp - mm_skew
                skew_pct = (skew_diff / abs(mm_skew) * 100) if abs(mm_skew) > 1e-6 else 0.0
                skew_pct_exp = (skew_diff_exp / abs(mm_skew) * 100) if abs(mm_skew) > 1e-6 else 0.0
                
                isolated_test_table.add_row(
                    "",
                    "Skewness (actual)",
                    f"{mm_skew:.3f}",
                    f"{iso_skew:.3f}",
                    f"{skew_diff:.3f} ({skew_pct:+.1f}%)"
                )
                
                isolated_test_table.add_row(
                    "",
                    "Skewness (expected)",
                    f"{mm_skew:.3f}",
                    f"{iso_skew_exp:.3f}",
                    f"{skew_diff_exp:.3f} ({skew_pct_exp:+.1f}%)"
                )
                
                isolated_test_table.add_row("", "", "", "", "")  # Spacer
            
            console.print(isolated_test_table)
            console.print("\n[dim]Note: Isolated simulations use equilibrium variance (θ) as starting point and 100-period burn-in[/dim]")
            console.print("[dim]to ensure variance reaches steady state before measuring statistics.[/dim]")
            console.print("[dim]Two versions are shown: 'actual var' uses realized variance, 'expected var' uses expected variance[/dim]")
            console.print("[dim]for consistency with analytical formulas. Expected variance version should match better.[/dim]")
            console.print("\n[dim]If isolated simulations match moment matching but regime-switching doesn't,[/dim]")
            console.print("[dim]the issue is likely variance continuity across regime switches or transition probabilities.[/dim]")
            console.print()
            
            # Simulation-based validation (if requested)
            if args.simulation_based_validation:
                console.print("\n[bold cyan]Simulation-Based Moment Validation[/bold cyan]")
                console.print("=" * 80)
                console.print("[dim]Computing moments via simulation (this may take a while)...[/dim]")
                
                sim_validation_table = Table(title="Simulation-Based Moment Computation (10,000 paths × 1,000 periods)")
                sim_validation_table.add_column("Regime", style="cyan", width=8)
                sim_validation_table.add_column("Metric", style="yellow", width=20)
                sim_validation_table.add_column("Analytical", justify="right", style="blue", width=15)
                sim_validation_table.add_column("Simulation", justify="right", style="green", width=15)
                sim_validation_table.add_column("Difference", justify="right", style="red", width=15)
                
                for regime_id in sorted(regime_comparisons.keys()):
                    if regime_id not in moment_matching_info or regime_id not in params_dict:
                        continue
                    
                    params = params_dict[regime_id]
                    mm_info = moment_matching_info[regime_id]
                    
                    # Compute via simulation
                    sim_moments = compute_moments_via_simulation(
                        params, 
                        n_simulations=10000, 
                        n_periods=1000,
                        burn_in_periods=100,
                        seed=42,
                        use_expected_variance=args.use_expected_variance
                    )
                    
                    # Compare
                    mm_mean = mm_info['model_mean']
                    sim_mean = sim_moments['mean']
                    mean_diff = sim_mean - mm_mean
                    mean_pct = (mean_diff / abs(mm_mean) * 100) if abs(mm_mean) > 1e-6 else 0.0
                    
                    sim_validation_table.add_row(
                        f"Regime {regime_id}",
                        "Mean (monthly)",
                        f"{mm_mean:.6f}",
                        f"{sim_mean:.6f}",
                        f"{mean_diff:.6f} ({mean_pct:+.2f}%)"
                    )
                    
                    mm_std = mm_info['model_std']
                    sim_std = sim_moments['std']
                    std_diff = sim_std - mm_std
                    std_pct = (std_diff / abs(mm_std) * 100) if abs(mm_std) > 1e-6 else 0.0
                    
                    sim_validation_table.add_row(
                        "",
                        "Std (monthly)",
                        f"{mm_std:.6f}",
                        f"{sim_std:.6f}",
                        f"{std_diff:.6f} ({std_pct:+.2f}%)"
                    )
                    
                    mm_skew = mm_info['model_skew']
                    sim_skew = sim_moments['skew']
                    skew_diff = sim_skew - mm_skew
                    skew_pct = (skew_diff / abs(mm_skew) * 100) if abs(mm_skew) > 1e-6 else abs(skew_diff)
                    
                    sim_validation_table.add_row(
                        "",
                        "Skewness",
                        f"{mm_skew:.3f}",
                        f"{sim_skew:.3f}",
                        f"{skew_diff:.3f} ({skew_pct:+.2f}%)"
                    )
                    
                    mm_kurt = mm_info['model_kurt']
                    sim_kurt = sim_moments['kurt']
                    kurt_diff = sim_kurt - mm_kurt
                    kurt_pct = (kurt_diff / abs(mm_kurt) * 100) if abs(mm_kurt) > 1e-6 else abs(kurt_diff)
                    
                    sim_validation_table.add_row(
                        "",
                        "Kurtosis",
                        f"{mm_kurt:.3f}",
                        f"{sim_kurt:.3f}",
                        f"{kurt_diff:.3f} ({kurt_pct:+.2f}%)"
                    )
                    
                    sim_validation_table.add_row("", "", "", "", "")  # Spacer
                
                console.print(sim_validation_table)
                console.print("\n[dim]This shows what the analytical formulas predict vs what simulations actually produce.[/dim]")
                console.print("[dim]Large differences indicate analytical formula issues.[/dim]")
                console.print()
            
            # Detailed analysis of analytical formula issues
            console.print("\n[bold cyan]Analytical Formula Validation[/bold cyan]")
            console.print("=" * 80)
            
            # Check which formulas are problematic
            formula_issues = []
            for regime_id in sorted(regime_comparisons.keys()):
                if regime_id not in moment_matching_info or regime_id not in params_dict:
                    continue
                
                mm_info = moment_matching_info[regime_id]
                params = params_dict[regime_id]
                
                # Get isolated simulation results with burn-in
                isolated_returns = []
                isolated_returns_expected = []
                burn_in = 100
                path_length = 500
                
                for path_idx in range(2):
                    # Actual variance
                    returns, _ = simulate_regime_switching_bates(
                        {regime_id: params},
                        np.array([[1.0]]),
                        [regime_id],
                        n_periods=path_length,
                        initial_regime=regime_id,
                        initial_variance=params[IDX['theta']],
                        seed=path_idx,
                        burn_in_periods=burn_in,
                        reset_variance_on_switch=False,
                        use_expected_variance=False
                    )
                    isolated_returns.extend(returns)
                    
                    # Expected variance
                    returns_exp, _ = simulate_regime_switching_bates(
                        {regime_id: params},
                        np.array([[1.0]]),
                        [regime_id],
                        n_periods=path_length,
                        initial_regime=regime_id,
                        initial_variance=params[IDX['theta']],
                        seed=path_idx,
                        burn_in_periods=burn_in,
                        reset_variance_on_switch=False,
                        use_expected_variance=True
                    )
                    isolated_returns_expected.extend(returns_exp)
                
                isolated_returns = np.array(isolated_returns)
                isolated_returns_expected = np.array(isolated_returns_expected)
                isolated_stats = compute_statistics(isolated_returns)
                isolated_stats_exp = compute_statistics(isolated_returns_expected)
                
                # Compare mean (use expected variance version - should match better)
                mm_mean = mm_info['model_mean']
                iso_mean = isolated_stats['cagr'] / PERIODS_PER_YEAR  # Convert to monthly
                iso_mean_exp = isolated_stats_exp['cagr'] / PERIODS_PER_YEAR
                mean_error = abs(iso_mean - mm_mean) / abs(mm_mean) * 100 if abs(mm_mean) > 1e-6 else 0.0
                mean_error_exp = abs(iso_mean_exp - mm_mean) / abs(mm_mean) * 100 if abs(mm_mean) > 1e-6 else 0.0
                
                # Compare std (should be exact)
                mm_std = mm_info['model_std']
                iso_std = isolated_stats['vol'] / math.sqrt(PERIODS_PER_YEAR)  # Convert to monthly
                iso_std_exp = isolated_stats_exp['vol'] / math.sqrt(PERIODS_PER_YEAR)
                std_error = abs(iso_std - mm_std) / abs(mm_std) * 100 if abs(mm_std) > 1e-6 else 0.0
                std_error_exp = abs(iso_std_exp - mm_std) / abs(mm_std) * 100 if abs(mm_std) > 1e-6 else 0.0
                
                # Compare skew (approximate formula)
                mm_skew = mm_info['model_skew']
                iso_skew = isolated_stats['skew']
                iso_skew_exp = isolated_stats_exp['skew']
                skew_error = abs(iso_skew - mm_skew) / abs(mm_skew) * 100 if abs(mm_skew) > 1e-6 else abs(iso_skew - mm_skew)
                skew_error_exp = abs(iso_skew_exp - mm_skew) / abs(mm_skew) * 100 if abs(mm_skew) > 1e-6 else abs(iso_skew_exp - mm_skew)
                
                # Compare kurt (approximate formula)
                mm_kurt = mm_info['model_kurt']
                iso_kurt = isolated_stats['kurt']
                iso_kurt_exp = isolated_stats_exp['kurt']
                kurt_error = abs(iso_kurt - mm_kurt) / abs(mm_kurt) * 100 if abs(mm_kurt) > 1e-6 else abs(iso_kurt - mm_kurt)
                kurt_error_exp = abs(iso_kurt_exp - mm_kurt) / abs(mm_kurt) * 100 if abs(mm_kurt) > 1e-6 else abs(iso_kurt_exp - mm_kurt)
                
                # Use expected variance errors for reporting (should be better)
                mean_error = mean_error_exp
                std_error = std_error_exp
                skew_error = skew_error_exp
                kurt_error = kurt_error_exp
                
                if mean_error > 5:
                    formula_issues.append(f"Regime {regime_id}: Mean formula error {mean_error:.1f}% (should be exact!)")
                if std_error > 5:
                    formula_issues.append(f"Regime {regime_id}: Std formula error {std_error:.1f}% (should be exact!)")
                if abs(skew_error) > 50:
                    formula_issues.append(f"Regime {regime_id}: Skew approximation error {skew_error:.1f}% (approximate formula)")
                if abs(kurt_error) > 50:
                    formula_issues.append(f"Regime {regime_id}: Kurt approximation error {kurt_error:.1f}% (approximate formula)")
            
            if formula_issues:
                console.print("[red]⚠ Analytical Formula Issues Detected:[/red]")
                for issue in formula_issues:
                    console.print(f"  • {issue}")
                console.print()
                console.print("[yellow]The analytical formulas used in moment matching do not accurately predict[/yellow]")
                console.print("[yellow]simulated behavior. This explains why parameters fit well but simulations differ.[/yellow]")
                console.print()
                console.print("[dim]Note: Isolated tests use equilibrium variance initialization and burn-in periods.[/dim]")
                console.print("[dim]If errors persist, the analytical formulas themselves may need correction.[/dim]")
            else:
                console.print("[green]✓ Analytical formulas appear to match simulations reasonably well[/green]")
                console.print("[dim]  (Isolated tests use equilibrium variance and burn-in for fair comparison)[/dim]")
            
            console.print()
    elif empirical_regimes is None:
        console.print("\n[yellow]⚠ Per-regime comparison skipped: No regime IDs in empirical data[/yellow]")
    
    # Summary and Root Cause Analysis
    console.print("\n[bold cyan]Summary (Overall)[/bold cyan]")
    console.print("=" * 80)
    emp = comparison_results['empirical']
    med = comparison_results['model_median']
    
    console.print(f"Empirical CAGR: {emp['cagr']:.4f} | Model Median: {med['cagr']:.4f}")
    console.print(f"Empirical Volatility: {emp['vol']:.4f} | Model Median: {med['vol']:.4f}")
    console.print(f"Empirical Skewness: {emp['skew']:.3f} | Model Median: {med['skew']:.3f}")
    console.print(f"Empirical Kurtosis: {emp['kurt']:.3f} | Model Median: {med['kurt']:.3f}")
    console.print(f"Empirical Max DD: {emp['max_dd']:.4f} | Model Median: {med['max_dd']:.4f}")
    
    # Root cause analysis
    console.print("\n[bold cyan]Root Cause Analysis[/bold cyan]")
    console.print("=" * 80)
    
    # Check if moment matching quality is good but simulations differ
    mm_quality_good = True
    for regime_id in sorted(regime_ids):
        if regime_id in moment_matching_info:
            mm_info = moment_matching_info[regime_id]
            if (abs(mm_info['mean_error_pct']) > 5 or 
                abs(mm_info['std_error_pct']) > 5 or
                abs(mm_info['skew_error_pct']) > 10 or
                abs(mm_info['kurt_error_pct']) > 10):
                mm_quality_good = False
                break
    
    if mm_quality_good:
        console.print("[green]✓ Moment matching quality is good (parameters fit well during estimation)[/green]")
        console.print()
        console.print("[yellow]⚠ However, large discrepancies exist between predicted and simulated moments.[/yellow]")
        console.print()
        console.print("[bold]Likely Root Causes:[/bold]")
        console.print("1. [cyan]Variance Continuity Issue:[/cyan] When regimes switch, variance continues from")
        console.print("   previous regime but parameters (κ, θ, ν) change. Variance may be far from")
        console.print("   equilibrium (θ) for the new regime, causing different behavior than analytical formulas assume.")
        console.print()
        console.print("2. [cyan]Analytical Formula Limitations:[/cyan] Moment matching uses analytical formulas")
        console.print("   that assume single-regime, equilibrium conditions. Regime-switching breaks these assumptions.")
        console.print()
        console.print("3. [cyan]Regime Duration Effects:[/cyan] Analytical formulas assume infinite horizon,")
        console.print("   but actual regimes have finite durations, affecting convergence to equilibrium.")
        console.print()
        console.print("[bold]Potential Solutions:[/bold]")
        console.print()
        console.print("[bold cyan]Option 1 (RECOMMENDED):[/bold cyan] Use simulation-based moment matching")
        console.print("  Instead of analytical formulas, use Monte Carlo simulations during parameter fitting.")
        console.print("  This ensures parameters match actual simulated behavior, not approximate formulas.")
        console.print()
        console.print("[bold cyan]Option 2:[/bold cyan] Fix/improve analytical formulas")
        console.print("  The current skewness/kurtosis formulas are approximations. Consider using exact")
        console.print("  characteristic function methods or better approximations that account for:")
        console.print("  - Non-equilibrium variance states")
        console.print("  - Finite horizon effects")
        console.print("  - Regime-specific parameter interactions")
        console.print()
        console.print("Option 3: Reset variance to v₀ when switching regimes")
        console.print("  This may be unrealistic but would make analytical formulas more applicable.")
        console.print()
        console.print("Option 4: Adjust parameters post-hoc")
        console.print("  Calibrate parameters to match regime-switching simulation results rather than")
        console.print("  isolated regime analytical formulas.")
        console.print()
        console.print("Option 5: Use hybrid approach")
        console.print("  Fit to analytical formulas first, then fine-tune using simulation-based optimization")
        console.print("  to account for regime-switching effects.")
        console.print()
        console.print("[bold yellow]Immediate Action Items:[/bold yellow]")
        console.print("0. Run with --historical-regimes to test if regime timing (vs random switching) explains tail risk gap")
        console.print("1. For Regime 2 (worst case): Consider re-fitting with simulation-based moment matching")
        console.print("2. Review analytical formula approximations in 2B. Moment_Matching.py (lines 354-423)")
        console.print("3. Consider using exact characteristic function methods instead of approximations")
        console.print("4. Test if increasing simulation length in isolated test improves convergence")
        console.print()
    else:
        console.print("[red]✗ Moment matching quality is poor - parameters may not be well-estimated[/red]")
        console.print("[dim]Consider re-running moment matching with:[/dim]")
        console.print("  • More optimization restarts")
        console.print("  • Different parameter bounds")
        console.print("  • Higher precision tolerances")
        console.print("  • --match-max-dd flag for better tail risk matching")
        console.print()
    
    # Generate plot if requested
    if args.plot and simulated_paths is not None:
        console.print("\n[cyan]Generating comparison plot...[/cyan]")
        os.makedirs('output', exist_ok=True)
        output_path = os.path.join('output', 'historical_vs_simulations.png')
        plot_historical_vs_simulations(
            empirical_returns, simulated_paths, output_path, 
            n_sample_paths=args.n_sample_paths
        )
    elif not args.plot:
        console.print("\n[yellow]⚠ Plots not generated. Use --plot flag to create comparison plots.[/yellow]")
    
    console.print(f"\n[bold green]✓ Simulation complete![/bold green]")

if __name__ == "__main__":
    main()

