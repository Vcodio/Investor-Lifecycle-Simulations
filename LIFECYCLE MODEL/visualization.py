"""
Visualization Module

This module provides plotting and visualization functions for simulation results.
"""

import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def detect_bimodality(ages, bins=50):
    """
    Detect if retirement age distribution is bimodal.
    
    Returns:
        dict with keys: 'is_bimodal', 'mode1', 'mode2', 'dip_depth', 'dip_location'
    """
    if len(ages) < 10:
        return {'is_bimodal': False, 'mode1': None, 'mode2': None, 
                'dip_depth': None, 'dip_location': None}
    
    hist, bin_edges = np.histogram(ages, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    

    try:
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(hist, height=np.max(hist) * 0.1, distance=len(hist)//10)
        
        if len(peaks) >= 2:

            peak_heights = hist[peaks]
            sorted_indices = np.argsort(peak_heights)[::-1]
            mode1_idx = peaks[sorted_indices[0]]
            mode2_idx = peaks[sorted_indices[1]]
            

            start_idx = min(mode1_idx, mode2_idx)
            end_idx = max(mode1_idx, mode2_idx)
            dip_idx = start_idx + np.argmin(hist[start_idx:end_idx+1])
            dip_depth = (hist[mode1_idx] + hist[mode2_idx]) / 2 - hist[dip_idx]
            

            avg_peak_height = (hist[mode1_idx] + hist[mode2_idx]) / 2
            is_bimodal = dip_depth > avg_peak_height * 0.2
            
            return {
                'is_bimodal': is_bimodal,
                'mode1': bin_centers[mode1_idx],
                'mode2': bin_centers[mode2_idx],
                'dip_depth': dip_depth,
                'dip_location': bin_centers[dip_idx]
            }
    except (ImportError, Exception):

        pass
    
    return {'is_bimodal': False, 'mode1': None, 'mode2': None, 
            'dip_depth': None, 'dip_location': None}


def plot_required_principal_nominal(ages, principals_nominal, swr, config, median_age):
    """Plot required principal in nominal terms"""
    if not HAS_MATPLOTLIB:
        return

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel('Retirement Age', color='white', fontsize=12)
    ax1.set_ylabel('Required Principal ($)', color='white', fontsize=12)
    ax1.plot(ages, principals_nominal, color='cyan', marker='o',
            label='Required Principal (Nominal $)', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='white')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    ax2 = ax1.twinx()
    ax2.set_ylabel('Withdrawal Rate (%)', color='white', fontsize=12)
    ax2.plot(ages, np.array(swr), color='magenta', marker='x',
            linestyle='--', label='Withdrawal Rate', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='white')
    ax2.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{x:.2f}%'))

    if config.include_social_security:
        ax1.axvline(x=config.social_security_start_age, color='lime',
                   linestyle=':',
                   label=f'Social Security (Age {config.social_security_start_age})',
                   linewidth=2)

    if not np.isnan(median_age):
        ax1.axvline(x=median_age, color='yellow', linestyle='--',
                   label=f'Median Retirement Age ({median_age:.1f})', linewidth=2)

    success_pct = int(config.success_target * 100)
    fig.suptitle(f"Required Principal & Withdrawal Rate for {success_pct}% Success (Nominal)",
                fontsize=14, color='white')
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    import os
    os.makedirs(config.output_directory, exist_ok=True)
    filepath = os.path.join(config.output_directory, 'required_principal_and_swr_nominal.png')
    plt.savefig(filepath, bbox_inches='tight', format='png', dpi=150)
    plt.show()
    plt.close(fig)


def plot_required_principal_real(ages, principals_real, swr, config, median_age):
    """Plot required principal in real terms"""
    if not HAS_MATPLOTLIB:
        return

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel('Retirement Age', color='white', fontsize=12)
    ax1.set_ylabel('Required Principal ($)', color='white', fontsize=12)
    ax1.plot(ages, principals_real, color='orange', marker='o',
            label='Required Principal (Real $)', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='white')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    ax2 = ax1.twinx()
    ax2.set_ylabel('Withdrawal Rate (%)', color='white', fontsize=12)
    ax2.plot(ages, np.array(swr), color='lime', marker='x',
            linestyle='--', label='Withdrawal Rate', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='white')
    ax2.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{x:.2f}%'))

    if config.include_social_security:
        ax1.axvline(x=config.social_security_start_age, color='cyan',
                   linestyle=':',
                   label=f'Social Security (Age {config.social_security_start_age})',
                   linewidth=2)

    if not np.isnan(median_age):
        ax1.axvline(x=median_age, color='yellow', linestyle='--',
                   label=f'Median Retirement Age ({median_age:.1f})', linewidth=2)

    success_pct = int(config.success_target * 100)
    fig.suptitle(f"Required Principal & Withdrawal Rate for {success_pct}% Success (Real)",
                fontsize=14, color='white')
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    import os
    os.makedirs(config.output_directory, exist_ok=True)
    filepath = os.path.join(config.output_directory, 'required_principal_and_swr_real.png')
    plt.savefig(filepath, bbox_inches='tight', format='png', dpi=150)
    plt.show()
    plt.close(fig)


def plot_retirement_age_distribution(valid_ages, median_age, output_dir='output'):
    """Plot retirement age distribution"""
    if not HAS_MATPLOTLIB or valid_ages.size == 0:
        return



    valid_ages_int = np.round(valid_ages).astype(int)
    


    
    plt.figure(figsize=(12, 7))


    min_age = int(np.min(valid_ages_int))
    max_age = int(np.max(valid_ages_int)) + 1
    n, bins, patches = plt.hist(valid_ages_int, bins=range(min_age, max_age + 1), 
                                 color='cyan', alpha=0.7, align='left', 
                                 edgecolor='cyan', linewidth=0)
    

    for patch in patches:
        patch.set_edgecolor(patch.get_facecolor())
        patch.set_linewidth(0)
    
    plt.title('Distribution of Retirement Ages', fontsize=14, color='white')
    
    plt.axvline(median_age, color='red', linestyle='--', linewidth=2,
               label=f'Median: {median_age:.1f}')
    plt.xlabel('Retirement Age', color='white', fontsize=12)
    plt.ylabel('Frequency', color='white', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    import os
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'retirement_age_distribution.png')
    plt.savefig(filepath, bbox_inches='tight', format='png', dpi=150)
    plt.show()
    plt.close()
    









def plot_cumulative_retirement_probability(retirement_ages, config, median_age, num_outer):
    """Plot cumulative probability of retiring by age"""
    if not HAS_MATPLOTLIB:
        return

    valid_retirement_ages = retirement_ages[~np.isnan(retirement_ages)]
    if valid_retirement_ages.size == 0:
        return

    sorted_ages = np.sort(valid_retirement_ages)
    cumulative_prob = np.arange(1, len(sorted_ages) + 1) / num_outer * 100

    plt.figure(figsize=(12, 7))
    plt.plot(sorted_ages, cumulative_prob, color='lime', marker='o',
            linestyle='-', markersize=4, alpha=0.8, linewidth=2)

    if config.include_social_security:
        plt.axvline(x=config.social_security_start_age, color='white',
                   linestyle=':',
                   label=f'Social Security (Age {config.social_security_start_age})',
                   linewidth=2)

    if not np.isnan(median_age):
        plt.axvline(x=median_age, color='white', linestyle='--',
                   label=f'Median Retirement Age ({median_age:.1f})', linewidth=2)

    plt.title("Cumulative Probability of Retiring by Age", fontsize=14, color='white')
    plt.xlabel("Age", color='white', fontsize=12)
    plt.ylabel("Cumulative Probability (%)", color='white', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.legend(facecolor='black', edgecolor='white', framealpha=0.6)
    plt.tight_layout()
    import os
    os.makedirs(config.output_directory, exist_ok=True)
    filepath = os.path.join(config.output_directory, 'cumulative_prob_retiring_by_age.png')
    plt.savefig(filepath, bbox_inches='tight', format='png', dpi=150)
    plt.show()
    plt.close()


def plot_certainty_equivalent_distribution(certainty_equivalents_annual, output_dir):
    """Plot distribution of certainty equivalent consumption values"""
    if not HAS_MATPLOTLIB or len(certainty_equivalents_annual) == 0:
        return
    
    plt.figure(figsize=(12, 7))
    n, bins, patches = plt.hist(certainty_equivalents_annual, bins=50, color='lime', edgecolor='black', alpha=0.7)
    plt.yscale('log')
    plt.title('Certainty Equivalent Consumption Distribution', fontsize=14, color='white', fontweight='bold')
    plt.xlabel('Certainty Equivalent Annual Consumption ($)', color='white', fontsize=12)
    plt.ylabel('Frequency (log scale)', color='white', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    

    mean_ce = np.mean(certainty_equivalents_annual)
    median_ce = np.median(certainty_equivalents_annual)
    plt.axvline(mean_ce, color='yellow', linestyle='--', linewidth=2, label=f'Mean: ${mean_ce:,.0f}')
    plt.axvline(median_ce, color='cyan', linestyle='--', linewidth=2, label=f'Median: ${median_ce:,.0f}')
    plt.legend()
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'certainty_equivalent_distribution.png')
    plt.savefig(filepath, bbox_inches='tight', format='png', dpi=150, facecolor='black')
    plt.show()
    plt.close()


def plot_utility_distribution(utilities, output_dir):
    """Plot distribution of utility values"""
    if not HAS_MATPLOTLIB:
        return
    

    if utilities is None or len(utilities) == 0:
        return
    

    valid_utilities = [u for u in utilities if np.isfinite(u) and not np.isnan(u)]
    

    if len(valid_utilities) <= 1:


        return
    

    unique_values = len(set(valid_utilities))
    if unique_values < 2:
        return
    

    data_range = max(valid_utilities) - min(valid_utilities)
    if data_range <= 0:
        return
    

    n_bins = min(50, max(10, len(valid_utilities) // 10))
    
    plt.figure(figsize=(12, 7))
    n, bins, patches = plt.hist(valid_utilities, bins=n_bins, color='cyan', edgecolor='black', alpha=0.7)
    plt.title('Total Utility Distribution', fontsize=14, color='white', fontweight='bold')
    plt.xlabel('Total Utility', color='white', fontsize=12)
    plt.ylabel('Frequency', color='white', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    

    mean_util = np.mean(valid_utilities)
    median_util = np.median(valid_utilities)
    plt.axvline(mean_util, color='yellow', linestyle='--', linewidth=2, label=f'Mean: {mean_util:.2e}')
    plt.axvline(median_util, color='lime', linestyle='--', linewidth=2, label=f'Median: {median_util:.2e}')
    plt.legend()
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'utility_distribution.png')
    plt.savefig(filepath, bbox_inches='tight', format='png', dpi=150, facecolor='black')
    plt.show()
    plt.close()


def plot_certainty_equivalent_bar(certainty_equivalents_annual, output_dir, title_suffix=""):
    """Plot certainty equivalent consumption as a bar chart"""
    if not HAS_MATPLOTLIB or len(certainty_equivalents_annual) == 0:
        return
    
    plt.figure(figsize=(10, 6))
    mean_ce = np.mean(certainty_equivalents_annual)
    bars = plt.bar(['Mean CE'], [mean_ce], width=0.6, color='lime', alpha=0.8)
    plt.title(f'Certainty Equivalent Consumption{title_suffix}', color='white', fontsize=14)
    plt.ylabel('Annual Consumption ($)', color='white', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.gca().get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'${int(x/1000)}K'))
    plt.tight_layout()
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'certainty_equivalent_bar.png')
    plt.savefig(filepath, bbox_inches='tight', format='png', dpi=150, facecolor='black')
    plt.show()
    plt.close()


def create_all_plots(required_principal_data, retirement_ages,
                     detailed_simulations, config, median_age, num_outer,
                     utilities=None, certainty_equivalents_annual=None,
                     amortization_stats_list=None):
    """Create all visualization plots"""
    import os
    

    base_dir = config.output_directory
    principal_dir = os.path.join(base_dir, 'Principal Requirements')
    retirement_dir = os.path.join(base_dir, 'Retirement Analysis')
    utility_dir = os.path.join(base_dir, 'Utility Analysis')
    
    os.makedirs(principal_dir, exist_ok=True)
    os.makedirs(retirement_dir, exist_ok=True)
    os.makedirs(utility_dir, exist_ok=True)
    
    ages = [row['age'] for row in required_principal_data]
    principals_nominal = [row['principal_nominal'] for row in required_principal_data]
    principals_real = [row['principal_real'] for row in required_principal_data]
    swr = [row['swr'] for row in required_principal_data]


    original_output = config.output_directory
    config.output_directory = principal_dir
    plot_required_principal_nominal(ages, principals_nominal, swr, config, median_age)
    plot_required_principal_real(ages, principals_real, swr, config, median_age)
    config.output_directory = retirement_dir
    valid_ages = retirement_ages[~np.isnan(retirement_ages)]
    if valid_ages.size > 0:
        plot_retirement_age_distribution(valid_ages, median_age, retirement_dir)
    plot_cumulative_retirement_probability(retirement_ages, config, median_age, num_outer)
    

    if utilities is not None and len(utilities) > 0:
        plot_utility_distribution(utilities, utility_dir)
    if certainty_equivalents_annual is not None and len(certainty_equivalents_annual) > 0:
        plot_certainty_equivalent_distribution(certainty_equivalents_annual, utility_dir)
        plot_certainty_equivalent_bar(certainty_equivalents_annual, utility_dir)
    

    if amortization_stats_list is not None and config.use_amortization:
        create_amortization_plots(amortization_stats_list, config, base_dir)
    

    config.output_directory = original_output


def plot_amortization_withdrawal_over_time(amortization_stats_list, config, output_dir):
    """Plot withdrawal amounts over time for amortization method"""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping amortization withdrawal plot")
        return
    
    if not amortization_stats_list or all(stats is None for stats in amortization_stats_list):
        logger.warning("No amortization statistics available for plotting")
        return
    

    all_withdrawals_by_year = {}
    initial_spending = None
    
    for stats in amortization_stats_list:
        if stats is None or not stats.get('withdrawals'):
            continue
        
        if initial_spending is None:
            initial_spending = stats.get('initial_spending_real', config.spending_real)
        
        withdrawals = stats['withdrawals']
        for year_idx, withdrawal in enumerate(withdrawals):
            if year_idx not in all_withdrawals_by_year:
                all_withdrawals_by_year[year_idx] = []
            all_withdrawals_by_year[year_idx].append(withdrawal)
    
    if not all_withdrawals_by_year:
        logger.warning("No withdrawal data found for plotting")
        return
    

    years = sorted(all_withdrawals_by_year.keys())
    medians = [np.median(all_withdrawals_by_year[y]) for y in years]
    p25 = [np.percentile(all_withdrawals_by_year[y], 25) for y in years]
    p75 = [np.percentile(all_withdrawals_by_year[y], 75) for y in years]
    p10 = [np.percentile(all_withdrawals_by_year[y], 10) for y in years]
    p90 = [np.percentile(all_withdrawals_by_year[y], 90) for y in years]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    

    ax.fill_between(years, p10, p90, alpha=0.2, color='cyan', label='10th-90th Percentile')
    ax.fill_between(years, p25, p75, alpha=0.3, color='blue', label='25th-75th Percentile')
    ax.plot(years, medians, 'o-', color='yellow', linewidth=2, markersize=6, label='Median')
    

    if initial_spending:
        ax.axhline(y=initial_spending, color='red', linestyle='--', linewidth=2, 
                  label=f'Initial Spending (${initial_spending:,.0f})')

        threshold = config.amortization_min_spending_threshold * initial_spending
        ax.axhline(y=threshold, color='orange', linestyle=':', linewidth=2,
                  label=f'Minimum Threshold ({config.amortization_min_spending_threshold*100:.0f}% = ${threshold:,.0f})')
    
    ax.set_xlabel('Years in Retirement', fontsize=12)
    ax.set_ylabel('Annual Withdrawal (Real $)', fontsize=12)
    ax.set_title('Amortization-Based Withdrawals Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'amortization_withdrawals_over_time.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved amortization withdrawal plot to {output_path}")


def plot_amortization_below_threshold_percentage(amortization_stats_list, config, output_dir):
    """Plot percentage of simulations with spending below threshold by retirement year"""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping below threshold plot")
        return
    
    if not amortization_stats_list or all(stats is None for stats in amortization_stats_list):
        logger.warning("No amortization statistics available for plotting")
        return
    



    initial_spending = None
    threshold_counts_by_year = {}
    total_counts_by_year = {}
    
    total_sims = len([s for s in amortization_stats_list if s is not None and s.get('withdrawals')])
    
    for stats in amortization_stats_list:
        if stats is None or not stats.get('withdrawals'):
            continue
        
        if initial_spending is None:
            initial_spending = stats.get('initial_spending_real', config.spending_real)
        
        threshold = config.amortization_min_spending_threshold * initial_spending
        withdrawals = stats['withdrawals']
        
        for year_idx, withdrawal in enumerate(withdrawals):
            if year_idx not in threshold_counts_by_year:
                threshold_counts_by_year[year_idx] = 0
                total_counts_by_year[year_idx] = 0
            
            total_counts_by_year[year_idx] += 1
            if withdrawal < threshold:
                threshold_counts_by_year[year_idx] += 1
    
    if not threshold_counts_by_year:
        logger.warning("No data for below-threshold percentage plot")
        return
    

    min_sims_threshold = max(10, total_sims * 0.5)
    years_with_sufficient_data = {y: total_counts_by_year[y] >= min_sims_threshold 
                                  for y in total_counts_by_year.keys()}
    

    years = sorted([y for y in threshold_counts_by_year.keys() if years_with_sufficient_data[y]])
    percentages = [(threshold_counts_by_year[y] / total_counts_by_year[y] * 100) 
                   if total_counts_by_year[y] > 0 else 0 for y in years]
    num_sims_per_year = [total_counts_by_year[y] for y in years]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    

    max_year_shown = max(years) if years else 0
    min_year_shown = min(years) if years else 0
    avg_sims_in_shown_years = np.mean(num_sims_per_year) if num_sims_per_year else 0
    coverage_pct = (avg_sims_in_shown_years / total_sims * 100) if total_sims > 0 else 0
    

    ax1.bar(years, percentages, color='coral', alpha=0.7, edgecolor='red', linewidth=1.5)
    ax1.axhline(y=50, color='orange', linestyle='--', linewidth=2, label='50% Reference')
    ax1.axhline(y=25, color='yellow', linestyle='--', linewidth=2, label='25% Reference')

    ax1.set_xlabel('Years Since Retirement Started', fontsize=12)
    ax1.set_ylabel('% of Simulations Below Minimum Threshold', fontsize=12)
    ax1.set_title(
        f'Percentage of Simulations with Spending Below {config.amortization_min_spending_threshold*100:.0f}% of Initial Spending\n'
        f'Years {min_year_shown}-{max_year_shown} shown (avg {coverage_pct:.1f}% data coverage, {total_sims:,} total simulations)',
        fontsize=14,
        fontweight='bold',
    )



    if percentages:
        max_pct = max(percentages)

        y_max = max_pct * 1.2

        y_max = max(y_max, 55.0)

        y_max = min(y_max, 100.0)
    else:
        y_max = 100.0
    ax1.set_ylim(0, y_max)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    

    ax2.bar(years, num_sims_per_year, color='gray', alpha=0.6, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=min_sims_threshold, color='red', linestyle='--', linewidth=1, 
                label=f'Threshold ({min_sims_threshold:.0f} sims, {min_sims_threshold/total_sims*100:.0f}%)')
    ax2.set_xlabel('Years Since Retirement Started', fontsize=12)
    ax2.set_ylabel('Number of Simulations', fontsize=12)
    ax2.set_title(f'Data Coverage (Total Simulations: {total_sims})', fontsize=11)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'amortization_below_threshold_percentage.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved below-threshold percentage plot to {output_path}")
    logger.info(f"Below-threshold plot: showing {len(years)} years (filtered to years with >= {min_sims_threshold:.0f} simulations)")


def plot_amortization_principal_trajectory(amortization_stats_list, config, output_dir):
    """Plot principal value over time for amortization method"""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping principal trajectory plot")
        return
    
    if not amortization_stats_list or all(stats is None for stats in amortization_stats_list):
        logger.warning("No amortization statistics available for plotting")
        return
    



    all_principal_by_year = {}
    total_sims = len([s for s in amortization_stats_list if s is not None and s.get('principal_at_year_start')])
    
    for stats in amortization_stats_list:
        if stats is None or not stats.get('principal_at_year_start'):
            continue
        
        principals = stats['principal_at_year_start']
        for year_idx, principal in enumerate(principals):
            if year_idx not in all_principal_by_year:
                all_principal_by_year[year_idx] = []
            all_principal_by_year[year_idx].append(principal)
    
    if not all_principal_by_year:
        logger.warning("No principal data found for plotting")
        return
    

    min_sims_threshold = max(10, total_sims * 0.5)
    years_with_sufficient_data = {y: len(all_principal_by_year[y]) >= min_sims_threshold 
                                   for y in all_principal_by_year.keys()}
    

    years = sorted([y for y in all_principal_by_year.keys() if years_with_sufficient_data[y]])
    medians = [np.median(all_principal_by_year[y]) for y in years]
    p25 = [np.percentile(all_principal_by_year[y], 25) for y in years]
    p75 = [np.percentile(all_principal_by_year[y], 75) for y in years]
    num_sims_per_year = [len(all_principal_by_year[y]) for y in years]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    

    ax1.fill_between(years, p25, p75, alpha=0.3, color='blue', label='25th-75th Percentile')
    ax1.plot(years, medians, 'o-', color='yellow', linewidth=2, markersize=4, label='Median Principal')
    
    ax1.set_xlabel('Years Since Retirement Started', fontsize=12)
    ax1.set_ylabel('Portfolio Value (Nominal $)', fontsize=12)
    ax1.set_title('Portfolio Value Trajectory (Amortization Method)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')
    

    ax2.bar(years, num_sims_per_year, color='gray', alpha=0.6, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=min_sims_threshold, color='red', linestyle='--', linewidth=1, 
                label=f'Threshold ({min_sims_threshold:.0f} sims, {min_sims_threshold/total_sims*100:.0f}%)')
    ax2.set_xlabel('Years Since Retirement Started', fontsize=12)
    ax2.set_ylabel('Number of Simulations', fontsize=12)
    ax2.set_title(f'Data Coverage (Total Simulations: {total_sims})', fontsize=11)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'amortization_principal_trajectory.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved principal trajectory plot to {output_path}")
    logger.info(f"Principal trajectory: showing {len(years)} years (filtered to years with >= {min_sims_threshold:.0f} simulations)")


def plot_amortization_withdrawal_distribution(amortization_stats_list, config, output_dir):
    """Plot distribution of withdrawal amounts (all years combined)"""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping withdrawal distribution plot")
        return
    
    if not amortization_stats_list or all(stats is None for stats in amortization_stats_list):
        logger.warning("No amortization statistics available for plotting")
        return
    

    all_withdrawals = []
    initial_spending = None
    
    for stats in amortization_stats_list:
        if stats is None or not stats.get('withdrawals'):
            continue
        
        if initial_spending is None:
            initial_spending = stats.get('initial_spending_real', config.spending_real)
        
        all_withdrawals.extend(stats['withdrawals'])
    
    if not all_withdrawals:
        logger.warning("No withdrawal data for distribution plot")
        return
    
    all_withdrawals = np.array(all_withdrawals)
    

    median_w = np.median(all_withdrawals)
    mean_w = np.mean(all_withdrawals)
    p10_w = np.percentile(all_withdrawals, 10)
    p25_w = np.percentile(all_withdrawals, 25)
    p75_w = np.percentile(all_withdrawals, 75)
    p90_w = np.percentile(all_withdrawals, 90)
    
    fig = plt.figure(figsize=(16, 10))
    

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35, left=0.08, right=0.95, top=0.93, bottom=0.08)
    

    ax1 = fig.add_subplot(gs[0, 0])
    n_bins = min(80, len(np.unique(all_withdrawals)))
    counts, bins, patches = ax1.hist(all_withdrawals, bins=n_bins, color='skyblue', edgecolor='none', alpha=0.7)
    if initial_spending:
        ax1.axvline(x=initial_spending, color='red', linestyle='--', linewidth=2,
                   label=f'Initial Spending: ${initial_spending:,.0f}')
        threshold = config.amortization_min_spending_threshold * initial_spending
        ax1.axvline(x=threshold, color='orange', linestyle=':', linewidth=2,
                   label=f'Threshold ({config.amortization_min_spending_threshold*100:.0f}%): ${threshold:,.0f}')
    ax1.axvline(x=median_w, color='green', linestyle='-', linewidth=2, alpha=0.7,
               label=f'Median: ${median_w:,.0f}')
    ax1.set_xlabel('Annual Withdrawal (Real $)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Annual Withdrawals (Linear Scale)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xlim(left=0)
    

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    stats_text = f"""
Withdrawal Distribution Statistics
{'='*52}

Total Withdrawals: {len(all_withdrawals):,}

Mean:       ${mean_w:>14,.2f}
Median:     ${median_w:>14,.2f}

10th %ile:  ${p10_w:>14,.2f}
25th %ile:  ${p25_w:>14,.2f}
75th %ile:  ${p75_w:>14,.2f}
90th %ile:  ${p90_w:>14,.2f}

Min:        ${np.min(all_withdrawals):>14,.2f}
Max:        ${np.max(all_withdrawals):>14,.2f}

Initial Target: ${initial_spending:>12,.2f}
Threshold ({config.amortization_min_spending_threshold*100:.0f}%): ${config.amortization_min_spending_threshold * initial_spending:>12,.2f}


    """
    ax2.text(0.05, 0.5, stats_text, fontsize=10.5, family='monospace',
            verticalalignment='center', transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    

    ax3 = fig.add_subplot(gs[1, 0])
    bp = ax3.boxplot(
        [all_withdrawals],
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', alpha=0.7, linewidth=1.5),
        medianprops=dict(color='red', linewidth=2.5),

        whiskerprops=dict(color='white', linewidth=1.5),
        capprops=dict(color='white', linewidth=1.5),
        widths=0.6,
        showfliers=False,
    )
    if initial_spending:
        ax3.axhline(y=initial_spending, color='red', linestyle='--', linewidth=2,
                   label=f'Initial: ${initial_spending:,.0f}')
        threshold = config.amortization_min_spending_threshold * initial_spending
        ax3.axhline(y=threshold, color='orange', linestyle=':', linewidth=2,
                   label=f'Threshold: ${threshold:,.0f}')
    ax3.set_ylabel('Annual Withdrawal (Real $)', fontsize=12)
    ax3.set_title('Withdrawal Distribution (Box Plot, Outliers Removed)', fontsize=13, fontweight='bold')
    ax3.set_xticklabels(['All Withdrawals'], fontsize=11)
    ax3.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(bottom=0)
    

    ax4 = fig.add_subplot(gs[1, 1])

    p99_w = np.percentile(all_withdrawals, 99)
    capped_withdrawals = np.clip(all_withdrawals, 0, p99_w)
    n_bins2 = min(60, len(np.unique(capped_withdrawals)))
    ax4.hist(capped_withdrawals, bins=n_bins2, color='lightcoral', edgecolor='none', alpha=0.7)
    if initial_spending:
        ax4.axvline(x=initial_spending, color='red', linestyle='--', linewidth=2,
                   label=f'Initial: ${initial_spending:,.0f}')
        threshold = config.amortization_min_spending_threshold * initial_spending
        ax4.axvline(x=threshold, color='orange', linestyle=':', linewidth=2,
                   label=f'Threshold: ${threshold:,.0f}')
    ax4.axvline(x=median_w, color='green', linestyle='-', linewidth=2, alpha=0.7,
               label=f'Median: ${median_w:,.0f}')
    ax4.set_xlabel('Annual Withdrawal (Real $, capped at 99th %ile)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title(f'Distribution (Focus on Main Range, Max=${p99_w:,.0f})', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xlim(left=0)
    
    plt.suptitle('Amortization-Based Withdrawal Distribution Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = os.path.join(output_dir, 'amortization_withdrawal_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved withdrawal distribution plot to {output_path}")


def create_amortization_plots(amortization_stats_list, config, output_dir):
    """Create all amortization-related plots"""
    if not amortization_stats_list or not config.use_amortization:
        return
    
    import os
    amortization_dir = os.path.join(output_dir, 'Amortization Analysis')
    os.makedirs(amortization_dir, exist_ok=True)
    
    plot_amortization_withdrawal_over_time(amortization_stats_list, config, amortization_dir)
    plot_amortization_below_threshold_percentage(amortization_stats_list, config, amortization_dir)
    plot_amortization_principal_trajectory(amortization_stats_list, config, amortization_dir)
    plot_amortization_withdrawal_distribution(amortization_stats_list, config, amortization_dir)

