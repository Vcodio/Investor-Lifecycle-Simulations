"""
Main Execution Module

This module coordinates the execution of the lifecycle retirement simulation
using the modular components.
"""

import numpy as np
import logging

try:
    import streamlit as st
    _IN_STREAMLIT = True

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

from .config import SimulationConfig
from .bootstrap import load_bootstrap_data
from .simulation import (
    find_required_principal,
    check_success_rate,
    run_accumulation_simulations
)
from .visualization import create_all_plots
from .utils import (
    convert_geometric_to_arithmetic,
    calculate_nominal_value,
    export_to_csv,
    export_detailed_simulations_to_csv,
    print_rich_table
)
from .cython_wrapper import CYTHON_AVAILABLE
from .utility import calculate_total_utility_ex_ante, calculate_certainty_equivalent_ex_ante

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    pd = None
    logger.warning("pandas not available - some features may be limited")


def build_required_principal_table(config, params, bootstrap_data=None):
    """Build required principal lookup table for different retirement ages"""
    success_pct_label = f"{int(config.success_target * 100)}%"
    print(f"\n--- Stage 1: Building Required Principal Lookup Table "
          f"({success_pct_label} success) ---")

    target_ages = np.arange(config.retirement_age_min, config.retirement_age_max + 1)
    required_principal_table = {}
    mean_inflation_arithmetic = convert_geometric_to_arithmetic(
        config.mean_inflation_geometric, config.std_inflation)

    previous_principal = None

    import time
    t_table_start = time.perf_counter()

    for age in tqdm(target_ages, desc="Calculating required principal per age"):

        t_age_start = time.perf_counter()

        principal = find_required_principal(age, config.success_target,
                                           config.num_nested, config, params,
                                           warm_start_principal=previous_principal,
                                           bootstrap_data=bootstrap_data)

        t_age_end = time.perf_counter()
        if age == target_ages[0] or age == target_ages[len(target_ages)//2]:
            try:
                import json
                import os
                log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, 'debug.log')
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'id': 'log_find_principal_time', 'timestamp': time.time() * 1000, 'location': 'main.py:54', 'message': 'find_required_principal timing', 'data': {'duration_ms': (t_age_end - t_age_start) * 1000, 'age': age}, 'sessionId': 'debug-session', 'runId': 'profile', 'hypothesisId': 'PERF'}) + '\n')
            except: pass

        
        if config.use_principal_deviation_threshold and previous_principal is not None:
            max_change = previous_principal * config.principal_deviation_threshold
            min_allowed = previous_principal - max_change
            max_allowed = previous_principal + max_change
            
            if principal < min_allowed:
                principal = min_allowed
                logger.info(
                    f"Age {age}: Principal constrained to minimum allowed "
                    f"(${principal:,.2f} due to {config.principal_deviation_threshold*100:.2f}% threshold)"
                )
            elif principal > max_allowed:
                principal = max_allowed
                logger.info(
                    f"Age {age}: Principal constrained to maximum allowed "
                    f"(${principal:,.2f} due to {config.principal_deviation_threshold*100:.2f}% threshold)"
                )
        
        required_principal_table[int(age)] = principal
        previous_principal = principal

    required_principal_data = []
    for age, principal_real in required_principal_table.items():
        principal_nominal = calculate_nominal_value(
            principal_real, config.initial_age, age, mean_inflation_arithmetic)

        net_withdrawal_real = config.spending_real
        if config.include_social_security and age >= config.social_security_start_age:
            net_withdrawal_real = max(0.0, config.spending_real -
                                     config.social_security_real)

        swr_val = ((net_withdrawal_real / principal_real) * 100.0
                  if principal_real > 0 else np.nan)

        nominal_spending = calculate_nominal_value(
            config.spending_real, config.initial_age, age, mean_inflation_arithmetic)
        nominal_ss = 0.0
        if config.include_social_security:
            nominal_ss = calculate_nominal_value(
                config.social_security_real, config.initial_age, age,
                mean_inflation_arithmetic)

        net_withdrawal_nominal = nominal_spending
        if config.include_social_security and age >= config.social_security_start_age:
            net_withdrawal_nominal = max(0.0, nominal_spending - nominal_ss)

        required_principal_data.append({
            'age': age,
            'principal_real': principal_real,
            'principal_nominal': principal_nominal,
            'spending_real': config.spending_real,
            'spending_nominal': nominal_spending,
            'net_withdrawal_real': net_withdrawal_real,
            'net_withdrawal_nominal': net_withdrawal_nominal,
            'swr': swr_val
        })

    if config.generate_csv_summary:
        export_to_csv(required_principal_data, 'required_principal_table.csv', config.output_directory, 
                     subdirectory='Principal Requirements')

    display_principal_table(required_principal_data, success_pct_label)

    principal_lookup = {
        int(row['age']): {
            'principal_real': row['principal_real'],
            'principal_nominal': row['principal_nominal'],
            'swr': row['swr']
        } for row in required_principal_data
    }

    return required_principal_data, principal_lookup


def display_principal_table(required_principal_data, success_pct_label):
    """Display the required principal table"""
    if pd is None:
        logger.warning("pandas not available - skipping table display")
        return
        
    df_table = pd.DataFrame(required_principal_data)
    df_table_display = df_table.copy()

    for col in ['principal_real', 'principal_nominal', 'spending_real',
                'spending_nominal', 'net_withdrawal_real', 'net_withdrawal_nominal']:
        df_table_display[col] = df_table_display[col].apply(
            lambda x: f"${x:,.2f}")

    df_table_display['swr'] = df_table_display['swr'].apply(
        lambda x: f"{x:.2f}%" if not np.isnan(x) else "NaN")

    df_table_display.rename(columns={
        'age': 'Retirement Age',
        'principal_real': 'Principal (Real $)',
        'principal_nominal': 'Principal (Nominal $)',
        'spending_real': 'Spending (Real $)',
        'spending_nominal': 'Spending (Nominal $)',
        'net_withdrawal_real': 'Net Withdrawal (Real $)',
        'net_withdrawal_nominal': 'Net Withdrawal (Nominal $)',
        'swr': 'Withdrawal Rate'
    }, inplace=True)

    print_rich_table(df_table_display,
                    f"Required Principal & Withdrawal Rate for {success_pct_label} "
                    f"Success (Real principals)")


def display_final_results(retirement_ages, config, utility_ex_ante=None, ce_annual=None):
    """Display final simulation results including utility metrics"""
    print("\n--- Final Results ---")

    valid_ages = retirement_ages[~np.isnan(retirement_ages)]
    num_retired = len(valid_ages)
    pct_ever_retired = 100.0 * num_retired / max(1, config.num_outer)


    median_age = np.nan if valid_ages.size == 0 else float(np.nanmedian(retirement_ages))
    p10 = np.nan if valid_ages.size == 0 else np.percentile(valid_ages, 10)
    p25 = np.nan if valid_ages.size == 0 else np.percentile(valid_ages, 25)
    p75 = np.nan if valid_ages.size == 0 else np.percentile(valid_ages, 75)
    p90 = np.nan if valid_ages.size == 0 else np.percentile(valid_ages, 90)

    def prob_retire_before(age_limit):
        if valid_ages.size == 0:
            return 0.0
        return 100.0 * np.sum(valid_ages <= age_limit) / config.num_outer

    prob_before_50 = prob_retire_before(50)
    prob_before_55 = prob_retire_before(55)
    prob_before_60 = prob_retire_before(60)

    print(f"\nSimulations run: {config.num_outer}")
    print(f"Ever retire with >= {config.success_target*100:.0f}% success: "
          f"{pct_ever_retired:.2f}%")
    print(f"Median retirement age: {median_age:.1f}")
    print(f"10th percentile retirement age: {p10:.1f}")
    print(f"25th percentile retirement age: {p25:.1f}")
    print(f"75th percentile retirement age: {p75:.1f}")
    print(f"90th percentile retirement age: {p90:.1f}")
    print(f"Probability retire before age 50: {prob_before_50:.2f}%")
    print(f"Probability retire before age 55: {prob_before_55:.2f}%")
    print(f"Probability retire before age 60: {prob_before_60:.2f}%")
    

    if utility_ex_ante is not None:
        print("\n--- Utility Metrics (EX-ANTE) ---")
        print(f"Total expected utility: {utility_ex_ante:.2e}")
        if ce_annual is not None:
            print(f"Certainty equivalent (annual, real dollars): ${ce_annual:,.2f}")
        print(f"Note: Utility and CE calculated using EX-ANTE approach")
        print(f"      (consumption distribution across simulations at each time t)")
        print(f"      This properly captures CRRA risk aversion effects")
        print(f"      CE is in REAL dollars (today's purchasing power)")

    return median_age


def display_amortization_stats(amortization_stats_list, final_bequest_nominal, final_bequest_real, config):
    """Display amortization method statistics and comparisons"""
    if not amortization_stats_list or not config.use_amortization:
        return
    
    print("\n" + "="*80)
    print("AMORTIZATION-BASED WITHDRAWAL STATISTICS")
    print("="*80)
    

    all_withdrawals = []
    all_final_balances_nominal = final_bequest_nominal
    all_final_balances_real = final_bequest_real
    total_below_threshold = 0
    total_years = 0
    initial_spending = None
    
    for stats in amortization_stats_list:
        if stats is None:
            continue
        
        if initial_spending is None:
            initial_spending = stats.get('initial_spending_real', config.spending_real)
        
        withdrawals = stats.get('withdrawals', [])
        all_withdrawals.extend(withdrawals)
        total_below_threshold += stats.get('below_threshold_count', 0)
        total_years += stats.get('total_years', 0)
    
    if not all_withdrawals:
        print("No amortization withdrawal data available")
        return
    
    all_withdrawals = np.array(all_withdrawals)
    threshold = config.amortization_min_spending_threshold * initial_spending
    

    median_withdrawal = np.median(all_withdrawals)
    mean_withdrawal = np.mean(all_withdrawals)
    p10_withdrawal = np.percentile(all_withdrawals, 10)
    p25_withdrawal = np.percentile(all_withdrawals, 25)
    p75_withdrawal = np.percentile(all_withdrawals, 75)
    p90_withdrawal = np.percentile(all_withdrawals, 90)
    min_withdrawal = np.min(all_withdrawals)
    max_withdrawal = np.max(all_withdrawals)
    
    pct_below_threshold = (total_below_threshold / total_years * 100) if total_years > 0 else 0
    

    valid_balances_nominal = [b for b in all_final_balances_nominal if not np.isnan(b) and b >= 0]
    valid_balances_real = [b for b in all_final_balances_real if not np.isnan(b) and b >= 0]
    
    median_balance_nominal = np.median(valid_balances_nominal) if valid_balances_nominal else 0
    median_balance_real = np.median(valid_balances_real) if valid_balances_real else 0
    mean_balance_nominal = np.mean(valid_balances_nominal) if valid_balances_nominal else 0
    mean_balance_real = np.mean(valid_balances_real) if valid_balances_real else 0
    
    print(f"\nWithdrawal Statistics (Real Dollars):")
    print(f"  Initial Spending Target: ${initial_spending:,.2f}")
    print(f"  Minimum Threshold ({config.amortization_min_spending_threshold*100:.0f}% of initial): ${threshold:,.2f}")
    print(f"  Mean Annual Withdrawal: ${mean_withdrawal:,.2f}")
    print(f"  Median Annual Withdrawal: ${median_withdrawal:,.2f}")
    print(f"  10th Percentile: ${p10_withdrawal:,.2f}")
    print(f"  25th Percentile: ${p25_withdrawal:,.2f}")
    print(f"  75th Percentile: ${p75_withdrawal:,.2f}")
    print(f"  90th Percentile: ${p90_withdrawal:,.2f}")
    print(f"  Minimum Withdrawal: ${min_withdrawal:,.2f}")
    print(f"  Maximum Withdrawal: ${max_withdrawal:,.2f}")
    
    print(f"\nBelow-Threshold Statistics:")
    print(f"  Total Retirement Years (across all sims): {total_years}")
    print(f"  Years Below Threshold: {total_below_threshold}")
    print(f"  Percentage Below Threshold: {pct_below_threshold:.2f}%")
    
    print(f"\nFinal Balance Statistics:")
    print(f"  Median Final Balance (Nominal): ${median_balance_nominal:,.2f}")
    print(f"  Median Final Balance (Real): ${median_balance_real:,.2f}")
    print(f"  Mean Final Balance (Nominal): ${mean_balance_nominal:,.2f}")
    print(f"  Mean Final Balance (Real): ${mean_balance_real:,.2f}")
    print(f"  Simulations with Positive Balance: {len(valid_balances_nominal)}/{len(all_final_balances_nominal)}")
    
    print(f"\nAmortization Parameters:")
    expected_return = getattr(config, 'amortization_expected_return', 0.03)
    if expected_return is None:
        expected_return = 0.03
    print(f"  Expected Real Return Used: {expected_return*100:.2f}%")
    print(f"  Method: Fixed Amortization (W = P * [r(1+r)^n] / [(1+r)^n - 1])")
    print(f"  Note: Withdrawal recalculated each year based on current principal")
    print(f"        and remaining years until death age ({config.death_age})")
    print(f"        Expected return is calculated from historical data when available,")
    print(f"        or from model parameters, or can be user-specified in config")


def main():
    """Main execution function - run this to start the simulation"""

    import time
    t_main_start = time.perf_counter()
    try:
        import json
        import os
        log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'debug.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'id': 'log_main_start', 'timestamp': time.time() * 1000, 'location': 'main.py:330', 'message': 'main() function entry', 'data': {}, 'sessionId': 'debug-session', 'runId': 'profile', 'hypothesisId': 'PERF'}) + '\n')
    except Exception as log_err: pass

    print("\n" + "="*70)
    print("MODULAR LIFECYCLE RETIREMENT SIMULATION v7.0")
    print("="*70)


    if CYTHON_AVAILABLE:
        print("[OK] Cython module imported successfully (compiled extension)")
        print("[OK] Running with Cython acceleration (10-50x faster!)")
    else:
        print("[WARNING] Running in pure Python mode (no Cython acceleration)")
        print("  To enable Cython acceleration, compile the module:")
        print("  Run: cython/build_cython_fixed.bat (or python setup.py build_ext --inplace)")

    print("="*70 + "\n")


    try:
        import json
        import time
        import os
        log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'debug.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'id': 'log_before_try', 'timestamp': time.time() * 1000, 'location': 'main.py:337', 'message': 'Before try block in main()', 'data': {}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'E'}) + '\n')
    except Exception as log_err: pass


    try:

        try:
            import json
            import time
            import os
            log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'log_config_init', 'timestamp': time.time() * 1000, 'location': 'main.py:338', 'message': 'Creating SimulationConfig', 'data': {}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'C'}) + '\n')
        except Exception as log_err: pass

        config = SimulationConfig()

        try:
            import json
            import time
            import os
            log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'log_config_created', 'timestamp': time.time() * 1000, 'location': 'main.py:384', 'message': 'SimulationConfig created', 'data': {'num_workers': config.num_workers, 'num_nested': config.num_nested, 'num_outer': config.num_outer}, 'sessionId': 'debug-session', 'runId': 'profile', 'hypothesisId': 'PERF'}) + '\n')
        except Exception as log_err: pass

        config.validate()

        try:
            import json
            import time
            import os
            log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'log_config_validated', 'timestamp': time.time() * 1000, 'location': 'main.py:340', 'message': 'Config validated successfully', 'data': {}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'C'}) + '\n')
        except Exception as log_err: pass

        params = config.params


        if config.use_block_bootstrap:
            print(f"[INFO] Block Bootstrap ENABLED")
            print(f"  CSV Path: {config.bootstrap_csv_path}")
            print(f"  Portfolio Column: {config.portfolio_column_name}")
            print(f"  Block Length: {config.block_length_years} years ({config.block_length_years * 12} months)")
            print(f"  Block Type: {'Overlapping' if config.block_overlapping else 'Non-overlapping'}")
        else:
            print(f"[INFO] Using Parametric Model (Bates/Heston)")
            print(f"  Model Parameters: mu={params['mu']:.4f}, kappa={params['kappa']:.4f}, "
                  f"theta={params['theta']:.4f}, nu={params['nu']:.4f}")
        print()
        
        print(f"[INFO] GKOS Earnings Model ENABLED")
        print(f"  Using GKOS earnings dynamics (unemployment model removed)")
        print()
        
        if config.enable_utility_calculations:
            print(f"[INFO] Utility Model Configuration")
            print(f"  CRRA (gamma): {config.gamma}")
            print(f"  Time discount (beta): {config.beta}")
            print(f"  Bequest weight (theta): {config.theta}")
            print(f"  Household size: {config.household_size}")
        else:
            print(f"[INFO] Utility calculations: DISABLED")
        print()


        bootstrap_data = None
        if config.use_block_bootstrap:

            try:
                import json
                import time
                import os
                log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, 'debug.log')
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'id': 'log_bootstrap_start', 'timestamp': time.time() * 1000, 'location': 'main.py:372', 'message': 'Loading bootstrap data', 'data': {'csv_path': config.bootstrap_csv_path}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'B'}) + '\n')
            except Exception as log_err: pass

            print("[INFO] Loading bootstrap data from CSV (one-time operation)...")
            bootstrap_data = load_bootstrap_data(config)

            try:
                import json
                import time
                import os
                log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, 'debug.log')
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'id': 'log_bootstrap_done', 'timestamp': time.time() * 1000, 'location': 'main.py:374', 'message': 'Bootstrap data loaded', 'data': {'success': bootstrap_data is not None, 'data_length': len(bootstrap_data[0]) if bootstrap_data else 0}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'B'}) + '\n')
            except Exception as log_err: pass

            if bootstrap_data is None:
                logger.warning("Failed to load bootstrap data, falling back to parametric model")
                config.use_block_bootstrap = False
            else:
                print(f"[OK] Bootstrap data loaded: {len(bootstrap_data[0])} monthly returns")
        print()

        if config.seed is not None:
            rng = np.random.default_rng(seed=config.seed)
        else:
            rng = np.random.default_rng()
        

        if config.use_amortization:
            if config.amortization_expected_return is None:

                from .simulation import calculate_expected_return_from_data
                monthly_returns = bootstrap_data[0] if bootstrap_data is not None else None
                monthly_inflation = bootstrap_data[1] if bootstrap_data is not None else None
                config.amortization_expected_return = calculate_expected_return_from_data(
                    monthly_returns=monthly_returns,
                    monthly_inflation=monthly_inflation,
                    params=params if not config.use_block_bootstrap else None,
                    mean_inflation=config.mean_inflation_geometric,
                    expected_real_annual_override=getattr(config, 'bootstrap_geometric_mean_override', None),
                )
                print(f"[INFO] Amortization expected real return (calculated): {config.amortization_expected_return*100:.2f}%")
            else:

                print(f"[INFO] Amortization expected real return (user-specified): {config.amortization_expected_return*100:.2f}%")

        required_principal_data, principal_lookup = build_required_principal_table(
            config, params, bootstrap_data)

        print("\n--- Stage 2: Running Accumulation Simulations "
              "(monthly returns, annual inflation adjustments, GKOS earnings) ---")

        (retirement_ages, ever_retired, detailed_simulations,
         final_bequest_nominal, final_bequest_real, consumption_streams,
         amortization_stats_list, earnings_nominal_list, earnings_real_list) = run_accumulation_simulations(
            config, params, principal_lookup, rng, bootstrap_data)




        ce_values_dict = {}
        total_utilities_dict = {}
        
        if config.enable_utility_calculations:

            if not consumption_streams or not final_bequest_real:
                logger.warning("[WARNING] Utility calculations enabled but no consumption/bequest data available")
                utilities = None
                certainty_equivalents_annual = None
                ce_values_dict = {}
                total_utilities_dict = {}
            else:

                logger.info("\n" + "="*80)
                logger.info("[DEBUG MAIN] UTILITY CALCULATION INPUT DATA")
                logger.info("="*80)
                logger.info(f"Number of consumption streams: {len(consumption_streams)}")
                logger.info(f"Number of bequests: {len(final_bequest_real)}")
                

                non_empty_streams = [cs for cs in consumption_streams if len(cs) > 0]
                logger.info(f"Non-empty consumption streams: {len(non_empty_streams)}/{len(consumption_streams)}")
                if len(non_empty_streams) > 0:
                    stream_lengths = [len(cs) for cs in non_empty_streams]
                    logger.info(f"Stream lengths: min={min(stream_lengths)}, max={max(stream_lengths)}, "
                               f"mean={np.mean(stream_lengths):.1f}, median={np.median(stream_lengths):.1f}")
                    

                    for i in range(min(3, len(non_empty_streams))):
                        cs = non_empty_streams[i]
                        logger.info(f"  Stream {i}: length={len(cs)}, "
                                   f"first_5={[f'{c:.2f}' for c in cs[:5]]}, "
                                   f"mean={np.mean(cs):.2f}")
                

                logger.info(f"Bequest stats: min={min(final_bequest_real):.2f}, "
                           f"max={max(final_bequest_real):.2f}, "
                           f"mean={np.mean(final_bequest_real):.2f}, "
                           f"median={np.median(final_bequest_real):.2f}")
                
                logger.info(f"\n[DEBUG MAIN] Utility parameters:")
                logger.info(f"  gamma={config.gamma}, beta={config.beta}")
                logger.info(f"  k_bequest={config.k_bequest}, theta={config.theta}, household_size={config.household_size}")
                logger.info("="*80 + "\n")
                

                portfolio_name = "Portfolio"
                consumption_streams_dict = {portfolio_name: consumption_streams}
                bequests_dict = {portfolio_name: final_bequest_real}
                

                from .utility import calculate_ce_for_crra
                ce_values_dict = calculate_ce_for_crra(
                    consumption_streams_dict, bequests_dict, config.gamma, config.beta, 
                    config.k_bequest, config.theta, config.household_size
                )
                

                from .utility import calculate_total_utility_ex_ante
                total_utilities_dict = calculate_total_utility_ex_ante(
                    consumption_streams_dict, bequests_dict, config.gamma, config.beta, 
                    config.k_bequest, config.theta, config.household_size
                )
                

                ce_ex_ante_monthly = ce_values_dict[portfolio_name][0] if portfolio_name in ce_values_dict else 0.0
                total_utility_ex_ante = total_utilities_dict[portfolio_name] if portfolio_name in total_utilities_dict else 0.0
                
                logger.info("\n" + "="*80)
                logger.info("[DEBUG MAIN] UTILITY CALCULATION RESULTS")
                logger.info("="*80)
                logger.info(f"Total expected utility (EX-ANTE): {total_utility_ex_ante:.6e}")
                logger.info(f"CE (monthly, real dollars): ${ce_ex_ante_monthly:,.2f}")
                logger.info(f"CE (annual, real dollars): ${ce_ex_ante_monthly * 12.0:,.2f}")
                logger.info("="*80 + "\n")
                

                utilities = [total_utility_ex_ante]

                certainty_equivalents_annual = [ce_ex_ante_monthly * 12.0]
                

                utility_for_display = total_utility_ex_ante
                ce_annual_for_display = ce_ex_ante_monthly * 12.0
        else:

            utilities = None
            certainty_equivalents_annual = None
            utility_for_display = None
            ce_annual_for_display = None
        
        median_age = display_final_results(
            retirement_ages, config, 
            utility_ex_ante=utility_for_display,
            ce_annual=ce_annual_for_display)
        

        if config.use_amortization:
            display_amortization_stats(amortization_stats_list, final_bequest_nominal, 
                                      final_bequest_real, config)

        if config.generate_csv_summary:
            export_detailed_simulations_to_csv(
                detailed_simulations, 'detailed_lifecycle_paths.csv', config.output_directory,
                subdirectory='Simulation Data')
            

            if utilities and certainty_equivalents_annual:
                utility_data = {
                    'Utility': utilities,
                    'Certainty_Equivalent_Annual': certainty_equivalents_annual
                }
                export_to_csv(utility_data, 'utility_metrics.csv', config.output_directory,
                             subdirectory='Utility Analysis')

        create_all_plots(required_principal_data, retirement_ages,
                        detailed_simulations, config, median_age, config.num_outer,
                        utilities=utilities if utilities else None,
                        certainty_equivalents_annual=certainty_equivalents_annual if certainty_equivalents_annual else None,
                        amortization_stats_list=amortization_stats_list if config.use_amortization else None)


        if config.enable_utility_calculations:
            print("\n" + "=" * 80)
            print("CALCULATING CERTAINTY EQUIVALENT CONSUMPTION")
            print("=" * 80)
            
            if ce_values_dict:

                print("\nCertainty Equivalent Consumption (Annual Real $ - today's purchasing power):")
                print(f"{'Portfolio':<50} {'CE ($)':>20}")
                print("-" * 72)
                for name, ce_dollars_list in ce_values_dict.items():
                    ce_value_annual = ce_dollars_list[0] * 12.0 if len(ce_dollars_list) > 0 else 0.0
                    print(f"{name:<50} ${ce_value_annual:>19,.2f}")
            



        else:
            print("\n[INFO] Utility calculations are DISABLED (enable_utility_calculations=False)")
            utilities = None
            certainty_equivalents_annual = None

        logger.info("[OK] Simulation completed successfully")

        if CYTHON_AVAILABLE:
            print("\n[SUCCESS] Cython acceleration was used - simulation ran much faster!")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

        try:
            import json
            import time
            import os
            log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'log_main_error', 'timestamp': time.time() * 1000, 'location': 'main.py:555', 'message': 'Exception in main()', 'data': {'error': str(e), 'traceback': traceback.format_exc(), 'error_type': type(e).__name__}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'D'}) + '\n')
        except Exception as log_err: pass


    t_main_end = time.perf_counter()
    try:
        import json
        import os
        log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'debug.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'id': 'log_main_total_time', 'timestamp': time.time() * 1000, 'location': 'main.py:677', 'message': 'main() total execution time', 'data': {'duration_ms': (t_main_end - t_main_start) * 1000}, 'sessionId': 'debug-session', 'runId': 'profile', 'hypothesisId': 'PERF'}) + '\n')
    except: pass



if __name__ == "__main__":
    main()

