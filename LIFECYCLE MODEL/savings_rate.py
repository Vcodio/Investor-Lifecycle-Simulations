"""
Equivalent Savings Rate Module

This module provides functions to calculate equivalent savings rates that produce
the same utility as a baseline scenario using EX-ANTE utility calculation.
"""

import numpy as np
import logging

from .utility import calculate_total_utility_ex_ante

logger = logging.getLogger(__name__)


def calculate_equivalent_savings_rate_scaling(target_utility, current_utility, current_savings_rate, gamma):
    """
    Calculate equivalent savings rate using utility scaling approach.
    
    This uses the insight that for CRRA utility, if we scale consumption/bequest by factor k,
    utility scales by k^(1-γ). So to match a target utility ratio, we need a wealth ratio
    k = (utility_ratio)^(1/(1-γ)), and savings rate scales roughly proportionally with wealth.
    
    Parameters:
    -----------
    target_utility : float
        Target utility value to match
    current_utility : float
        Current utility value
    current_savings_rate : float
        Current savings rate
    gamma : float
        CRRA risk aversion parameter
        
    Returns:
    --------
    equivalent_savings_rate : float
        Equivalent savings rate that would produce target utility
    """
    if current_utility == 0:
        return current_savings_rate
    
    # Calculate utility ratio
    utility_ratio = target_utility / current_utility
    
    # For CRRA: if we scale consumption/bequest by factor k, utility scales by k^(1-γ)
    # So to get utility ratio, we need wealth ratio: k = (utility_ratio)^(1/(1-γ))
    if gamma == 1:
        # For log utility: U(kc) = log(k) + log(c), so k = exp(utility_ratio - 1)
        wealth_scale_factor = np.exp(utility_ratio - 1.0) if utility_ratio > 0 else 1.0
    else:
        # For CRRA: k = (utility_ratio)^(1/(1-γ))
        if utility_ratio > 0:
            wealth_scale_factor = utility_ratio ** (1.0 / (1.0 - gamma))
        else:
            wealth_scale_factor = 1.0
    
    # Savings rate scales roughly proportionally with wealth (first-order approximation)
    # More precisely: if savings rate doubles, final wealth roughly doubles (all else equal)
    # So: s_equiv ≈ s_current * wealth_scale_factor
    equiv_rate = current_savings_rate * wealth_scale_factor
    
    # Clamp to reasonable range
    equiv_rate = max(0.01, min(0.50, equiv_rate))
    
    return equiv_rate


def find_equivalent_savings_rate(target_utility, consumption_streams_baseline, bequests_baseline,
                                 consumption_streams_test, bequests_test,
                                 baseline_savings_rate, gamma, beta, k_bequest, theta, household_size):
    """
    Find the equivalent savings rate for a test scenario that produces the same utility as baseline.
    
    Uses scaling approach based on utility ratios and CRRA properties, similar to Cederburg approach.
    
    Parameters:
    -----------
    target_utility : float
        Target utility value (baseline utility)
    consumption_streams_baseline : list of arrays
        Baseline consumption streams (for reference, not used in calculation)
    bequests_baseline : array
        Baseline bequest values (for reference, not used in calculation)
    consumption_streams_test : list of arrays
        Test consumption streams (scenario to find equivalent savings rate for)
    bequests_test : array
        Test bequest values (scenario to find equivalent savings rate for)
    baseline_savings_rate : float
        Baseline savings rate
    gamma : float
        CRRA risk aversion parameter
    beta : float
        Time discount factor
    k_bequest : float
        Bequest threshold
    theta : float
        Bequest weight parameter
    household_size : float
        Household size
        
    Returns:
    --------
    equivalent_savings_rate : float
        Equivalent savings rate for test scenario
    """
    # Calculate current utility for test scenario using EX-ANTE approach
    current_utility = calculate_total_utility_ex_ante(
        consumption_streams_test, bequests_test, gamma, beta, k_bequest, theta, household_size
    )
    
    # Use scaling approach to find equivalent savings rate
    equiv_rate = calculate_equivalent_savings_rate_scaling(
        target_utility, current_utility, baseline_savings_rate, gamma
    )
    
    return equiv_rate
