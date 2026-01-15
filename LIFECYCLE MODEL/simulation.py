"""
Core Simulation Module

This module contains the core simulation functions for the lifecycle retirement model,
including withdrawal phase simulations and accumulation phase simulations with GKOS earnings.
"""

import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from .cython_wrapper import simulate_monthly_return_svj, calculate_max_drawdown
from .bootstrap import create_block_bootstrap_sampler
from .earnings import simulate_earnings_path_with_inflation

logger = logging.getLogger(__name__)


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


def calculate_amortized_withdrawal(principal, remaining_years, real_rate):
    """
    Calculate annual withdrawal amount using fixed amortization formula.
    
    Formula: W = P * [r(1+r)^n] / [(1+r)^n - 1]
    Where:
        W = annual withdrawal amount
        P = current principal
        r = real rate of return (annual)
        n = remaining years
    
    Parameters:
    -----------
    principal : float
        Current portfolio value (real or nominal, depending on rate)
    remaining_years : float
        Number of years remaining in retirement
    real_rate : float
        Fixed real rate of return (annual, as decimal, e.g., 0.03 for 3%)
        
    Returns:
    --------
    annual_withdrawal : float
        Annual withdrawal amount in same units as principal
    """
    if remaining_years <= 0:
        # No years remaining - withdraw everything
        return principal
    
    if remaining_years <= 1:
        # One year or less remaining - withdraw all (or use simple formula)
        # For n=1: W = P * r / (1 - 1/(1+r)) = P * (1+r)
        # But more conservatively, just withdraw principal if less than a year
        return principal / max(remaining_years, 0.01)
    
    # Standard amortization formula: W = P * [r(1+r)^n] / [(1+r)^n - 1]
    r = real_rate
    n = remaining_years
    
    # Calculate (1+r)^n
    one_plus_r_to_n = (1.0 + r) ** n
    
    # Calculate numerator: r * (1+r)^n
    numerator = r * one_plus_r_to_n
    
    # Calculate denominator: (1+r)^n - 1
    denominator = one_plus_r_to_n - 1.0
    
    # Handle edge case where denominator is very close to zero (shouldn't happen for n > 1)
    if abs(denominator) < 1e-10:
        # Fallback: use simple division
        return principal / n
    
    # Calculate annual withdrawal
    annual_withdrawal = principal * (numerator / denominator)
    
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
    portfolio_history = [portfolio]
    
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

    while (age_in_months / 12.0) < config.death_age:
        if (age_in_months % 12) == 0 and age_in_months > int(start_age * 12):
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
        if include_social_security and (age_in_months / 12.0) >= social_security_start_age:
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
        portfolio_history.append(portfolio)

        age_in_months += 1

    growth_rates = [portfolio_history[i] / portfolio_history[i-1] - 1.0
                   for i in range(1, len(portfolio_history))]
    mean_growth = np.mean(growth_rates) if growth_rates else 0.0
    std_dev = np.std(growth_rates) if growth_rates else 0.0
    max_drawdown = calculate_max_drawdown(np.array(portfolio_history))

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
    if config.num_workers <= 1 or num_nested_sims < 100:
        res = check_success_rate_worker(principal, retirement_age, num_nested_sims,
                                       0, config, params, bootstrap_data)
        success_rate = res['successes'] / max(1, num_nested_sims)
        combined_metrics = {k: np.array(v) for k, v in res['metrics'].items()}
        return success_rate, combined_metrics

    sims_per_worker = num_nested_sims // config.num_workers
    remaining = num_nested_sims % config.num_workers
    futures = []

    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        for i in range(config.num_workers):
            sims_this_worker = sims_per_worker + (1 if i < remaining else 0)
            if sims_this_worker > 0:
                futures.append(executor.submit(check_success_rate_worker,
                                             principal, retirement_age,
                                             sims_this_worker, i, config, params, None))

        results = [f.result() for f in futures]

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
    return success_rate, combined_metrics


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

    while (age_in_months / 12.0) <= config.death_age:
        current_age_years = int(age_in_months // 12)
        current_year_idx = current_age_years - int(config.initial_age)  # Years since start

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
        if (age_in_months > int(config.initial_age * 12)) and ((age_in_months % 12) == 0):
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
                    annual_withdrawal_real = calculate_amortized_withdrawal(
                        principal_real, remaining_years, expected_return
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


def run_accumulation_simulations(config, params, principal_lookup, rng, bootstrap_data=None):
    """Run all accumulation simulations"""
    retirement_ages = np.full(config.num_outer, np.nan)
    ever_retired = np.zeros(config.num_outer, dtype=bool)
    detailed_simulations_to_export = []
    all_final_bequest_nominal = []
    all_final_bequest_real = []
    all_consumption_streams = []
    all_amortization_stats = []  # Collect amortization statistics if enabled

    # Fixed savings rate (can be made configurable)
    savings_rate = 0.25  # Default savings rate

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

    return (retirement_ages, ever_retired, detailed_simulations_to_export,
            all_final_bequest_nominal, all_final_bequest_real, all_consumption_streams,
            all_amortization_stats)

