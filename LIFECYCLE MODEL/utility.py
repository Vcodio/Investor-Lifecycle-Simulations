"""
Utility Model for Consumption and Bequest

This module provides functions for calculating utility using CRRA (Constant Relative Risk Aversion)
utility function, including utility from consumption streams and bequests.

Uses EX-ANTE approach: Calculates utility and CE based on consumption distribution ACROSS simulations
at each time t, which properly captures CRRA risk aversion effects.

Note: Consumption streams are MONTHLY periods (from retirement/withdrawal phase only).
We use monthly discounting: beta_monthly^t where t is in months (0, 1, 2, ...).
This matches Cederburg's annual approach exactly, but adapted for monthly steps.

Implementation matches Cederburg Lifecycle Simulation.py exactly, but with:
- Monthly discounting: beta_monthly = beta^(1/12), discount_factor = beta_monthly^t (t in months)
- Monthly consumption values
- CE output as monthly (multiply by 12 for annual display)
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def crra_utility(c, gamma):
    """
    CRRA utility function with handling for non-positive consumption.
    Matches Cederburg implementation exactly.
    
    Parameters:
    -----------
    c : float or array
        Consumption level(s)
    gamma : float
        CRRA risk aversion parameter (gamma=1 corresponds to log utility)
        
    Returns:
    --------
    utility : float or array
        Utility value(s)
    """
    if c <= 0:
        return -1e10  # Heavy penalty for zero/negative consumption (matches Cederburg)
    if gamma == 1:
        return np.log(c)
    else:
        return (c**(1 - gamma)) / (1 - gamma)


def calculate_total_utility_ex_ante(consumption_streams_dict, bequests_dict, gamma_param, beta, k_bequest, theta, household_size):
    """
    Calculate total expected utility (consumption + bequest) for all portfolios using EX-ANTE approach.
    
    Matches Cederburg Lifecycle Simulation.py exactly, but adapted for monthly steps.
    Uses EX-ANTE approach: Calculate utility based on distribution of consumption ACROSS simulations
    at each time t, including bequest utility.
    
    For monthly adaptation:
    - Use beta_monthly = beta^(1/12) for monthly discounting
    - t represents months (0, 1, 2, ...)
    - discount_factor = beta_monthly^t (instead of beta^t for years)
    
    Parameters:
    -----------
    consumption_streams_dict : dict
        Dictionary mapping portfolio/strategy names to lists of consumption streams.
        Each consumption stream is a list/array of MONTHLY consumption values.
        Format: {name: [stream1, stream2, ..., streamN]} where each stream is [c_0, c_1, ...]
    bequests_dict : dict
        Dictionary mapping portfolio/strategy names to lists of bequest values.
        Format: {name: [b1, b2, ..., bN]}
    gamma_param : float
        CRRA risk aversion parameter
    beta : float
        Time discount factor (ANNUAL)
    k_bequest : float
        Bequest threshold (minimum bequest value for utility)
    theta : float
        Bequest weight parameter
    household_size : float
        Household size (affects utility scaling via sqrt(household_size))
        
    Returns:
    --------
    total_utilities : dict
        Dictionary mapping portfolio names to total expected utility values
    """
    total_utilities = {}
    gamma = gamma_param
    
    # Monthly discount factor: beta_monthly = beta^(1/12)
    beta_monthly = beta**(1.0/12.0)
    
    for name in consumption_streams_dict.keys():
        # Collect consumption values at each time t across all simulations
        consumption_by_time = {}  # t -> list of consumption values at time t (MONTHLY periods)
        bequest_values = []
        
        # Handle both dict format (from multi-portfolio) and list format (single portfolio)
        consumption_streams_list = consumption_streams_dict[name]
        bequests_list = bequests_dict[name] if name in bequests_dict else []
        
        logger.info(f"  Number of simulations: {len(consumption_streams_list)}")
        logger.info(f"  Number of bequests: {len(bequests_list)}")
        
        for sim_idx, sim_data in enumerate(consumption_streams_list):
            # Handle both dict format (with 'consumption' key) and direct list/array format
            if isinstance(sim_data, dict) and 'consumption' in sim_data:
                consumption_stream = sim_data['consumption']
                bequest = sim_data.get('bequest', 0.0)
            else:
                # Direct consumption stream (list/array)
                consumption_stream = sim_data
                bequest = bequests_list[sim_idx] if sim_idx < len(bequests_list) else 0.0
            
            bequest_values.append(bequest)
            
            for t, c in enumerate(consumption_stream):
                if t not in consumption_by_time:
                    consumption_by_time[t] = []
                consumption_by_time[t].append(c)
            
            # Debug first few simulations
            if sim_idx < 3:
                logger.info(f"  Sim {sim_idx}: stream_length={len(consumption_stream)}, "
                           f"bequest={bequest:.2f}, "
                           f"first_5_consumption={[f'{c:.2f}' for c in consumption_stream[:5]]}")
        
        logger.info(f"  Total time periods (months): {len(consumption_by_time)}")
        logger.info(f"  Time period range: {min(consumption_by_time.keys())} to {max(consumption_by_time.keys())}")
        
        # Debug consumption statistics
        if len(consumption_by_time) > 0:
            sample_t = sorted(consumption_by_time.keys())[0]
            sample_values = consumption_by_time[sample_t]
            logger.info(f"  Sample t={sample_t}: n_values={len(sample_values)}, "
                       f"mean={np.mean(sample_values):.2f}, "
                       f"min={min(sample_values):.2f}, max={max(sample_values):.2f}")
        
        # Debug bequest statistics
        if len(bequest_values) > 0:
            logger.info(f"  Bequest stats: n={len(bequest_values)}, "
                       f"mean={np.mean(bequest_values):.2f}, "
                       f"min={min(bequest_values):.2f}, max={max(bequest_values):.2f}")
        
        # Calculate expected utility at each time t
        # E[U(c_t)] = (1/N) * sum_i U(c_t^i)
        # Matches Cederburg exactly: uses log or c^(1-γ)/(1-γ) directly
        expected_utility_by_time = {}
        for t, consumption_values in consumption_by_time.items():
            if gamma == 1:
                expected_utility_values = [np.log(c) for c in consumption_values]
            else:
                expected_utility_values = [c**(1 - gamma) / (1 - gamma) for c in consumption_values]
            expected_utility_by_time[t] = np.mean(expected_utility_values)
            
            # Debug first few time periods
            if t < 5 or (t % 12 == 0 and t < 60):  # First 5 months and first few years
                logger.info(f"    t={t} (month): n_values={len(consumption_values)}, "
                           f"mean_c={np.mean(consumption_values):.2f}, "
                           f"E[U]={expected_utility_by_time[t]:.6e}")
        
        # Calculate expected bequest utility
        if len(bequest_values) > 0:
            if gamma == 1:
                bequest_utilities = [np.log(b + k_bequest) for b in bequest_values]
            else:
                bequest_utilities = [(b + k_bequest)**(1 - gamma) / (1 - gamma) for b in bequest_values]
            expected_bequest_utility = np.mean(bequest_utilities)
        else:
            expected_bequest_utility = 0.0
        
        # Calculate total expected utility: sum_t β_monthly^t * E[U(c_t)] + β_monthly^T * θ * E[U(bequest)]
        # For monthly: t is in months, use beta_monthly^t instead of beta^t
        total_expected_utility_c = 0.0
        discount_sum_check = 0.0
        for t in sorted(consumption_by_time.keys()):
            discount_factor = beta_monthly**t  # Monthly discounting (matches Cederburg but with monthly periods)
            discount_sum_check += discount_factor
            utility_contribution = expected_utility_by_time[t] * discount_factor
            total_expected_utility_c += utility_contribution
            
            # Debug first few contributions
            if t < 5 or (t % 12 == 0 and t < 60):
                logger.info(f"    t={t}: discount={discount_factor:.6f}, "
                           f"E[U]={expected_utility_by_time[t]:.6e}, "
                           f"contribution={utility_contribution:.6e}")
        
        # Average retirement length (for bequest discount) - in months
        avg_retirement_months = np.mean([len(sim_data['consumption']) if isinstance(sim_data, dict) and 'consumption' in sim_data 
                                        else len(sim_data) for sim_data in consumption_streams_list])
        bequest_discount = beta_monthly**avg_retirement_months  # Monthly discounting
        bequest_utility_contribution = bequest_discount * theta * expected_bequest_utility
        total_expected_utility = total_expected_utility_c + bequest_utility_contribution
        
        logger.info(f"\n  [DEBUG UTILITY] Total utility calculation summary:")
        logger.info(f"    Total consumption utility: {total_expected_utility_c:.6e}")
        logger.info(f"    Expected bequest utility: {expected_bequest_utility:.6e}")
        logger.info(f"    Avg retirement months: {avg_retirement_months:.1f}")
        logger.info(f"    Bequest discount (beta_monthly^{avg_retirement_months:.1f}): {bequest_discount:.6f}")
        logger.info(f"    Bequest contribution (discount * theta * E[U]): {bequest_utility_contribution:.6e}")
        logger.info(f"    Total expected utility: {total_expected_utility:.6e}")
        logger.info(f"    Discount sum check: {discount_sum_check:.6f}")
        
        total_utilities[name] = total_expected_utility
    
    return total_utilities


def calculate_ce_for_crra(consumption_streams_dict, bequests_dict, gamma_param, beta, k_bequest, theta, household_size):
    """
    Calculate CE values for all portfolios given a CRRA value.
    
    Matches Cederburg Lifecycle Simulation.py calculate_ce_for_crra exactly, but adapted for monthly steps.
    
    Uses EX-ANTE approach: Calculate CE based on distribution of consumption ACROSS simulations
    at each time t, not per simulation. This captures uncertainty about which simulation will be experienced.
    
    For monthly adaptation:
    - Use beta_monthly = beta^(1/12) for monthly discounting
    - t represents months (0, 1, 2, ...)
    - discount_factor = beta_monthly^t (instead of beta^t for years)
    
    Parameters:
    -----------
    consumption_streams_dict : dict
        Dictionary mapping portfolio/strategy names to lists of consumption streams.
        Format: {name: [stream1, stream2, ..., streamN]} where each stream is [c_0, c_1, ...]
    bequests_dict : dict
        Dictionary mapping portfolio/strategy names to lists of bequest values.
        Format: {name: [b1, b2, ..., bN]}
    gamma_param : float
        The CRRA coefficient (risk aversion parameter)
    beta : float
        Time discount factor (ANNUAL)
    k_bequest : float
        Bequest threshold
    theta : float
        Bequest weight parameter
    household_size : float
        Household size
        
    Returns:
    --------
    ce_values : dict
        Dictionary mapping portfolio names to lists of CE values (one value per portfolio, ex-ante)
        Format: {name: [ce_value, ce_value, ..., ce_value]} (all values same for ex-ante)
    """
    ce_values = {name: [] for name in consumption_streams_dict.keys()}
    
    # Use gamma_param explicitly
    gamma = gamma_param
    
    # Monthly discount factor: beta_monthly = beta^(1/12)
    beta_monthly = beta**(1.0/12.0)
    
    logger.info(f"\n[DEBUG CE] calculate_ce_for_crra called with:")
    logger.info(f"  gamma={gamma_param}, beta={beta}, beta_monthly={beta_monthly:.6f}")
    logger.info(f"  k_bequest={k_bequest}, theta={theta}, household_size={household_size}")
    logger.info(f"  Portfolios: {list(consumption_streams_dict.keys())}")
    
    for name in consumption_streams_dict.keys():
        logger.info(f"\n[DEBUG CE] Processing portfolio: {name}")
        
        # Collect consumption values at each time t across all simulations
        max_years = 0
        consumption_streams_list = consumption_streams_dict[name]
        bequests_list = bequests_dict[name] if name in bequests_dict else []
        
        logger.info(f"  Number of simulations: {len(consumption_streams_list)}")
        logger.info(f"  Number of bequests: {len(bequests_list)}")
        
        for sim_data in consumption_streams_list:
            if isinstance(sim_data, dict) and 'consumption' in sim_data:
                max_years = max(max_years, len(sim_data['consumption']))
            else:
                max_years = max(max_years, len(sim_data))
        
        # For each time t, collect consumption values across simulations
        consumption_by_time = {}  # t -> list of consumption values at time t (MONTHLY periods)
        bequest_values = []
        
        for sim_idx, sim_data in enumerate(consumption_streams_list):
            # Handle both dict format and direct list format
            if isinstance(sim_data, dict) and 'consumption' in sim_data:
                consumption_stream = sim_data['consumption']
                bequest = sim_data.get('bequest', 0.0)
            else:
                consumption_stream = sim_data
                bequest = bequests_list[sim_idx] if sim_idx < len(bequests_list) else 0.0
            
            bequest_values.append(bequest)
            
            for t, c in enumerate(consumption_stream):
                if t not in consumption_by_time:
                    consumption_by_time[t] = []
                consumption_by_time[t].append(c)
            
            # Debug first few simulations
            if sim_idx < 3:
                logger.info(f"  Sim {sim_idx}: stream_length={len(consumption_stream)}, "
                           f"bequest={bequest:.2f}, "
                           f"first_5_consumption={[f'{c:.2f}' for c in consumption_stream[:5]]}")
        
        logger.info(f"  Total time periods (months): {len(consumption_by_time)}")
        if len(consumption_by_time) > 0:
            logger.info(f"  Time period range: {min(consumption_by_time.keys())} to {max(consumption_by_time.keys())}")
            # Debug consumption statistics
            sample_t = sorted(consumption_by_time.keys())[0]
            sample_values = consumption_by_time[sample_t]
            logger.info(f"  Sample t={sample_t}: n_values={len(sample_values)}, "
                       f"mean={np.mean(sample_values):.2f}, "
                       f"min={min(sample_values):.2f}, max={max(sample_values):.2f}")
        
        # Debug bequest statistics
        if len(bequest_values) > 0:
            logger.info(f"  Bequest stats: n={len(bequest_values)}, "
                       f"mean={np.mean(bequest_values):.2f}, "
                       f"min={min(bequest_values):.2f}, max={max(bequest_values):.2f}")
        
        # Calculate expected utility at each time t
        # E[U(c_t)] = (1/N) * sum_i U(c_t^i)
        # Matches Cederburg exactly
        expected_utility_by_time = {}
        for t, consumption_values in consumption_by_time.items():
            # Calculate expected value of c^(1-γ) at time t
            if gamma == 1:
                # For γ=1, U(c) = log(c)
                expected_utility_values = [np.log(c) for c in consumption_values]
            else:
                expected_utility_values = [c**(1 - gamma) / (1 - gamma) for c in consumption_values]
            expected_utility_by_time[t] = np.mean(expected_utility_values)
            
            # Debug first few time periods
            if t < 5 or (t % 12 == 0 and t < 60):
                logger.info(f"    t={t} (month): n_values={len(consumption_values)}, "
                           f"mean_c={np.mean(consumption_values):.2f}, "
                           f"E[U]={expected_utility_by_time[t]:.6e}")
        
        # Calculate expected bequest utility
        # E[U(bequest)] = (1/N) * sum_i U(bequest_i + k_bequest)
        if len(bequest_values) > 0:
            if gamma == 1:
                bequest_utilities = [np.log(b + k_bequest) for b in bequest_values]
            else:
                bequest_utilities = [(b + k_bequest)**(1 - gamma) / (1 - gamma) for b in bequest_values]
            expected_bequest_utility = np.mean(bequest_utilities)
            logger.info(f"  Expected bequest utility: {expected_bequest_utility:.6e}")
        else:
            expected_bequest_utility = 0.0
            logger.info(f"  Expected bequest utility: 0.0 (no bequests)")
        
        # Calculate total expected utility: sum_t β_monthly^t * E[U(c_t)] + β_monthly^T * θ * E[U(bequest)]
        # For monthly: use beta_monthly^t where t is in months
        discount_sum = 0.0
        total_expected_utility_c = 0.0
        for t in sorted(consumption_by_time.keys()):
            discount_factor = beta_monthly**t  # Monthly discounting
            discount_sum += discount_factor
            utility_contribution = expected_utility_by_time[t] * discount_factor
            total_expected_utility_c += utility_contribution
            
            # Debug first few contributions
            if t < 5 or (t % 12 == 0 and t < 60):
                logger.info(f"    t={t}: discount={discount_factor:.6f}, "
                           f"E[U]={expected_utility_by_time[t]:.6e}, "
                           f"contribution={utility_contribution:.6e}")
        
        # Average retirement length (for bequest discount) - in months
        avg_retirement_months = np.mean([len(sim_data['consumption']) if isinstance(sim_data, dict) and 'consumption' in sim_data 
                                        else len(sim_data) for sim_data in consumption_streams_list])
        bequest_discount = beta_monthly**avg_retirement_months  # Monthly discounting
        bequest_utility_contribution = bequest_discount * theta * expected_bequest_utility
        total_expected_utility = total_expected_utility_c + bequest_utility_contribution
        
        logger.info(f"\n  [DEBUG CE] Total utility calculation summary:")
        logger.info(f"    Total consumption utility: {total_expected_utility_c:.6e}")
        logger.info(f"    Expected bequest utility: {expected_bequest_utility:.6e}")
        logger.info(f"    Avg retirement months: {avg_retirement_months:.1f}")
        logger.info(f"    Bequest discount (beta_monthly^{avg_retirement_months:.1f}): {bequest_discount:.6f}")
        logger.info(f"    Bequest contribution: {bequest_utility_contribution:.6e}")
        logger.info(f"    Total expected utility: {total_expected_utility:.6e}")
        logger.info(f"    Discount sum: {discount_sum:.6f}")
        
        # Calculate CE: sum_t β_monthly^t * U(CE) = sum_t β_monthly^t * E[U(c_t)]
        # For CRRA: CE^(1-γ)/(1-γ) * sum_t β_monthly^t = sum_t β_monthly^t * E[U(c_t)]
        # Solving: CE^(1-γ) = sum_t β_monthly^t * E[U(c_t)] * (1-γ) / sum_t β_monthly^t
        # CE = [sum_t β_monthly^t * E[U(c_t)] * (1-γ) / sum_t β_monthly^t]^(1/(1-γ))
        
        if discount_sum > 0 and total_expected_utility_c != 0:
            if gamma == 1:
                # For γ=1, U(CE) = log(CE)
                # log(CE) * sum_t β_monthly^t = sum_t β_monthly^t * E[log(c_t)]
                # log(CE) = sum_t β_monthly^t * E[log(c_t)] / sum_t β_monthly^t
                # CE = exp(sum_t β_monthly^t * E[log(c_t)] / sum_t β_monthly^t)
                logger.info(f"  [DEBUG CE] Gamma=1 case (log utility)")
                expected_log_by_time = {}
                for t, consumption_values in consumption_by_time.items():
                    log_values = [np.log(c) for c in consumption_values]
                    expected_log_by_time[t] = np.mean(log_values)
                
                weighted_log_sum = 0.0
                for t in sorted(expected_log_by_time.keys()):
                    discount_factor = beta_monthly**t  # Monthly discounting
                    weighted_log_sum += expected_log_by_time[t] * discount_factor
                    if t < 5 or (t % 12 == 0 and t < 60):
                        logger.info(f"    t={t}: E[log(c)]={expected_log_by_time[t]:.6f}, "
                                   f"discount={discount_factor:.6f}, "
                                   f"weighted={expected_log_by_time[t] * discount_factor:.6f}")
                
                ce_consumption = np.exp(weighted_log_sum / discount_sum) if discount_sum > 0 else 0.0
                logger.info(f"  [DEBUG CE] weighted_log_sum={weighted_log_sum:.6f}, "
                           f"discount_sum={discount_sum:.6f}, "
                           f"CE (monthly, before household scaling)={ce_consumption:.2f}")
            else:
                # For γ > 1: utility is negative
                if total_expected_utility_c < 0:  # Should be negative for γ > 1
                    logger.info(f"  [DEBUG CE] Gamma > 1 case (gamma={gamma}), utility is negative")
                    numerator = total_expected_utility_c * (1 - gamma)  # Both negative, so positive
                    base = numerator / discount_sum  # Positive
                    exponent = 1 / (1 - gamma)  # Negative for γ > 1
                    
                    logger.info(f"    total_expected_utility_c={total_expected_utility_c:.6e}")
                    logger.info(f"    numerator = {total_expected_utility_c:.6e} * {1 - gamma:.2f} = {numerator:.6e}")
                    logger.info(f"    base = {numerator:.6e} / {discount_sum:.6f} = {base:.6e}")
                    logger.info(f"    exponent = 1/(1-{gamma:.2f}) = {exponent:.6f}")
                    
                    if base > 0:
                        # Use log for numerical stability: CE = exp(exponent * log(base))
                        # Since exponent is negative, this is equivalent to 1/base^(1/(γ-1))
                        log_base = np.log(base)
                        log_ce = exponent * log_base
                        ce_consumption = np.exp(log_ce)
                        logger.info(f"    log_base={log_base:.6f}, log_ce={log_ce:.6f}, CE={ce_consumption:.2f}")
                        # Verify: should be positive and finite
                        if not np.isfinite(ce_consumption) or ce_consumption <= 0:
                            logger.warning(f"    WARNING: CE is not finite or <= 0, setting to 0")
                            ce_consumption = 0.0
                    else:
                        logger.warning(f"    WARNING: base <= 0, setting CE to 0")
                        ce_consumption = 0.0
                else:
                    # For γ < 1: utility is positive, use similar approach
                    logger.info(f"  [DEBUG CE] Gamma < 1 case (gamma={gamma}), utility is positive")
                    numerator = total_expected_utility_c * (1 - gamma)  # Both positive, so positive
                    base = numerator / discount_sum  # Positive
                    exponent = 1 / (1 - gamma)  # Positive for γ < 1
                    
                    logger.info(f"    total_expected_utility_c={total_expected_utility_c:.6e}")
                    logger.info(f"    numerator={numerator:.6e}, base={base:.6e}, exponent={exponent:.6f}")
                    
                    if base > 0:
                        log_base = np.log(base)
                        log_ce = exponent * log_base
                        ce_consumption = np.exp(log_ce)
                        logger.info(f"    log_base={log_base:.6f}, log_ce={log_ce:.6f}, CE={ce_consumption:.2f}")
                        if not np.isfinite(ce_consumption) or ce_consumption <= 0:
                            logger.warning(f"    WARNING: CE is not finite or <= 0, setting to 0")
                            ce_consumption = 0.0
                    else:
                        logger.warning(f"    WARNING: base <= 0, setting CE to 0")
                        ce_consumption = 0.0
        else:
            logger.warning(f"  [DEBUG CE] WARNING: discount_sum={discount_sum:.6f} <= 0 or total_expected_utility_c={total_expected_utility_c:.6e} == 0")
            ce_consumption = 0.0
        
        # Scale by household_size (matches Cederburg)
        ce_consumption_total = ce_consumption * np.sqrt(household_size)
        logger.info(f"  [DEBUG CE] Final CE calculation:")
        logger.info(f"    CE (monthly, before household scaling): {ce_consumption:.2f}")
        logger.info(f"    household_size: {household_size}, sqrt(household_size): {np.sqrt(household_size):.3f}")
        logger.info(f"    CE_total (monthly, scaled): {ce_consumption_total:.2f}")
        logger.info(f"    CE_total (annual, scaled): ${ce_consumption_total * 12:,.2f}")
        
        # Store this single CE value for this portfolio (calculated across all simulations)
        # Note: We return a list with one value per simulation for compatibility,
        # but it's the same value for all (calculated ex-ante)
        num_sims = len(consumption_streams_list)
        ce_values[name] = [ce_consumption_total] * num_sims
        logger.info(f"  [DEBUG CE] Stored CE value for {num_sims} simulations: {ce_consumption_total:.2f} (monthly)")
    
    logger.info(f"\n[DEBUG CE] Completed calculate_ce_for_crra. Returning CE values for {len(ce_values)} portfolios.")
    return ce_values


# Backward compatibility function - converts list format to dict format
def calculate_certainty_equivalent_ex_ante(consumption_streams, bequests, gamma, beta, k_bequest, theta, household_size):
    """
    Backward compatibility wrapper for single portfolio.
    
    Converts list format to dict format and calls calculate_ce_for_crra.
    """
    # Convert to dict format
    consumption_streams_dict = {'Portfolio': consumption_streams}
    bequests_dict = {'Portfolio': bequests}
    
    ce_values_dict = calculate_ce_for_crra(
        consumption_streams_dict, bequests_dict, gamma, beta, k_bequest, theta, household_size
    )
    
    # Return first (and only) portfolio's CE value
    if 'Portfolio' in ce_values_dict and len(ce_values_dict['Portfolio']) > 0:
        return ce_values_dict['Portfolio'][0]
    else:
        return 0.0


# Legacy functions for backward compatibility (per-simulation approach)
# These are kept but not recommended - use EX-ANTE functions above

def calculate_total_utility(consumption_stream, bequest, gamma, beta, k_bequest, theta, household_size):
    """
    Calculate total utility for a single simulation path (LEGACY - use calculate_total_utility_ex_ante).
    
    Note: This per-simulation approach does not properly capture CRRA effects across the
    distribution of outcomes. Use calculate_total_utility_ex_ante instead.
    """
    beta_monthly = beta**(1.0/12.0)
    
    # Calculate consumption utility
    consumption_utility = 0.0
    for t, c in enumerate(consumption_stream):
        discount_factor = beta_monthly**t  # Monthly discounting
        if gamma == 1:
            consumption_utility += np.log(max(c, 1e-10)) * discount_factor
        else:
            consumption_utility += (max(c, 1e-10)**(1 - gamma) / (1 - gamma)) * discount_factor
    
    # Scale by household size
    consumption_utility *= np.sqrt(household_size)
    
    # Calculate bequest utility
    bequest_value = max(0.0, bequest + k_bequest)
    if gamma == 1:
        bequest_utility = np.log(bequest_value) if bequest_value > 0 else -1e10
    else:
        bequest_utility = (bequest_value**(1 - gamma) / (1 - gamma)) if bequest_value > 0 else (-1e10 if gamma > 1 else 0.0)
    
    # Scale bequest utility by household size
    bequest_utility *= np.sqrt(household_size)
    
    # Discount bequest utility by final period (months)
    T_months = len(consumption_stream)
    bequest_discount = beta_monthly**T_months
    
    # Total utility
    total_utility = consumption_utility + bequest_discount * theta * bequest_utility
    
    return total_utility


def calculate_certainty_equivalent(consumption_stream, bequest, gamma, beta, k_bequest, theta, household_size):
    """
    Calculate certainty equivalent for a single simulation path (LEGACY - use calculate_certainty_equivalent_ex_ante).
    
    Note: This per-simulation approach does not properly capture CRRA effects across the
    distribution of outcomes. Use calculate_certainty_equivalent_ex_ante instead.
    """
    beta_monthly = beta**(1.0/12.0)
    
    # Calculate total utility
    total_utility = calculate_total_utility(
        consumption_stream, bequest, gamma, beta, k_bequest, theta, household_size
    )
    
    # Calculate discount sum (monthly periods)
    T_months = len(consumption_stream)
    discount_sum = sum(beta_monthly**t for t in range(T_months))
    
    if discount_sum > 0:
        if gamma == 1:
            consumption_utility = 0.0
            for t, c in enumerate(consumption_stream):
                discount_factor = beta_monthly**t
                consumption_utility += np.log(max(c, 1e-10)) * discount_factor * np.sqrt(household_size)
            
            ce_consumption = np.exp(consumption_utility / (discount_sum * np.sqrt(household_size)))
        else:
            consumption_utility = 0.0
            for t, c in enumerate(consumption_stream):
                discount_factor = beta_monthly**t
                consumption_utility += (max(c, 1e-10)**(1 - gamma) / (1 - gamma)) * discount_factor * np.sqrt(household_size)
            
            # Solve for CE
            if consumption_utility < 0:  # gamma > 1
                numerator = consumption_utility * (1 - gamma)
                base = numerator / (discount_sum * np.sqrt(household_size))
                if base > 0:
                    exponent = 1 / (1 - gamma)
                    ce_consumption = np.exp(exponent * np.log(base))
                else:
                    ce_consumption = 0.0
            else:  # gamma < 1
                numerator = consumption_utility * (1 - gamma)
                base = numerator / (discount_sum * np.sqrt(household_size))
                if base > 0:
                    exponent = 1 / (1 - gamma)
                    ce_consumption = np.exp(exponent * np.log(base))
                else:
                    ce_consumption = 0.0
    else:
        ce_consumption = 0.0
    
    # Scale back up (consumption was divided by sqrt(household_size) when stored)
    ce_consumption *= np.sqrt(household_size)
    
    # Note: ce_consumption is MONTHLY
    return ce_consumption
