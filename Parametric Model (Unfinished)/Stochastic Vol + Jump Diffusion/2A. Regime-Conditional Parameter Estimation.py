"""
Regime-Conditional Parameter Estimation using HMM

This script combines HMM regime inference with parameter estimation to make
each model parameter conditional on the inferred regime state.

Workflow:
1. Use HMM to infer market regimes from historical returns
2. Split data by inferred regime
3. For each regime, estimate model parameters separately using moment matching or QMLE
4. Return a dictionary of parameters for each regime

This approach allows the model parameters to vary based on market conditions,
capturing regime-dependent behavior in returns and volatility.

Usage:
    python "2A. Regime-Conditional Parameter Estimation.py" [--method moment|qmle] [--hmm-features vol|skew|both]
    
    --method: Parameter estimation method (moment matching or QMLE)
    --hmm-features: Features to use for HMM regime detection (volatility, skewness, or both)
"""

import os
import sys
import numpy as np
import pandas as pd
from hmmlearn import hmm
from rich.console import Console
from rich.table import Table
from rich.progress import track, Progress
import argparse

# Add parent directory to path to import moment matching functions
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import parameter estimation from moment matching module
try:
    # Import from the moment matching script
    sys.path.insert(0, _current_dir)
    from importlib.util import spec_from_file_location, module_from_spec
    
    moment_matching_path = os.path.join(_current_dir, "2B. Moment Matching.py")
    spec = spec_from_file_location("moment_matching", moment_matching_path)
    moment_matching = module_from_spec(spec)
    spec.loader.exec_module(moment_matching)
    
    # Import key functions
    fit_bates_moment_matching = moment_matching.fit_bates_moment_matching
    compute_empirical_moments = moment_matching.compute_empirical_moments
    IDX = moment_matching.IDX
    DEFAULT_BOUNDS = moment_matching.DEFAULT_BOUNDS
except ImportError as e:
    console = Console()
    console.print(f"[bold red]Error importing moment matching module: {e}[/bold red]")
    console.print("[yellow]Please ensure '2B. Moment Matching.py' is in the same directory[/yellow]")
    sys.exit(1)

console = Console()

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_REGIMES = 3  # Number of HMM regimes
LOOKBACK_PERIOD = 21  # Days for non-overlapping monthly steps
VOLATILITY_WINDOW = 21  # Rolling window for volatility (trading days)
SKEWNESS_WINDOW = 252  # Rolling window for skewness (12 months)
RANDOM_SEED = 42
DEBUG_MODE = False

# ============================================================================
# HMM REGIME INFERENCE
# ============================================================================

def infer_regimes_hmm(returns, features='vol&skew', num_regimes=3, random_seed=42):
    """
    Infer market regimes using Hidden Markov Model.
    
    Args:
        returns: pandas Series of log returns (daily)
        features: 'vol', 'skew', or 'vol&skew' - which features to use for HMM
        num_regimes: Number of regimes to infer
        random_seed: Random seed for HMM
        
    Returns:
        tuple: (regime_labels, transition_matrix, hmm_model)
            - regime_labels: array of regime assignments for each period
            - transition_matrix: HMM transition matrix
            - hmm_model: fitted HMM model
    """
    console.print(f"\n[bold yellow]Step 1: Inferring {num_regimes} regimes using HMM with features='{features}'[/bold yellow]")
    
    # Calculate features for HMM
    df = pd.DataFrame({'returns': returns}, index=returns.index)
    
    # Calculate monthly non-overlapping volatility (optimized)
    df['monthly_vol'] = df['returns'].rolling(window=VOLATILITY_WINDOW, min_periods=1).std() * np.sqrt(252)
    
    # Calculate 12-month rolling skewness (optimized - use raw=True for better performance)
    df['skewness'] = df['returns'].rolling(window=SKEWNESS_WINDOW, min_periods=1).skew()
    
    # Create non-overlapping monthly indices
    df['month_index'] = (df.index - df.index[0]).days // LOOKBACK_PERIOD
    
    # Aggregate to monthly (take last value in each month)
    # groupby('month_index') makes month_index the index, so reset_index() converts it back to a column
    df_monthly = df.groupby('month_index').last().reset_index()  # Keep month_index as column
    df_monthly = df_monthly.dropna()  # Remove months with insufficient data
    
    # Prepare HMM input features
    if features == 'vol':
        X_data = df_monthly[['monthly_vol']].values
        covariance_type = 'diag'
    elif features == 'skew':
        X_data = df_monthly[['skewness']].values
        covariance_type = 'diag'
    elif features == 'both' or features == 'vol&skew':
        X_data = df_monthly[['monthly_vol', 'skewness']].values
        covariance_type = 'diag'
    else:
        raise ValueError(f"Unknown features option: {features}")
    
    console.print(f"  HMM input shape: {X_data.shape}")
    
    if X_data.shape[0] < num_regimes:
        console.print(f"[bold red]ERROR: Not enough data points ({X_data.shape[0]}) for {num_regimes} regimes[/bold red]")
        return None, None, None
    
    # Fit HMM (reduced iterations for speed - typically converges much earlier)
    model_inferred = hmm.GaussianHMM(
        n_components=num_regimes,
        covariance_type=covariance_type,
        n_iter=500,  # Reduced from 2000 for speed - usually converges in <200 iterations
        tol=1e-4,    # Slightly relaxed tolerance for faster convergence
        random_state=random_seed
    )
    
    try:
        model_inferred.fit(X_data)
        console.print(f"[green]✓ HMM fit successful[/green]")
    except ValueError as e:
        console.print(f"[bold red]Error during HMM fit: {e}[/bold red]")
        return None, None, None
    
    # Predict regimes
    regime_labels_monthly = model_inferred.predict(X_data)
    transition_matrix = model_inferred.transmat_
    
    # Map monthly regimes back to daily data (forward fill)
    df_monthly['regime'] = regime_labels_monthly
    # Create regime map from month_index to regime
    regime_map = df_monthly.set_index('month_index')['regime'].to_dict()
    
    # Map regimes to daily data (df already has month_index column from above)
    # Calculate month_index for daily data (in case it wasn't calculated yet, but it should be)
    if 'month_index' not in df.columns:
        df['month_index'] = (df.index - df.index[0]).days // LOOKBACK_PERIOD
    df['regime'] = df['month_index'].map(regime_map).ffill()
    
    # Map regime indices to daily returns
    regime_labels = df['regime'].values
    
    if DEBUG_MODE:
        console.print(f"\n[dim]Regime distribution:[/dim]")
        for i in range(num_regimes):
            count = np.sum(regime_labels == i)
            pct = count / len(regime_labels) * 100
            console.print(f"  Regime {i}: {count} observations ({pct:.1f}%)")
    
    return regime_labels, transition_matrix, model_inferred

# ============================================================================
# REGIME-CONDITIONAL PARAMETER ESTIMATION
# ============================================================================

def estimate_regime_conditional_parameters(returns, regime_labels, num_regimes=3,
                                          method='moment', bounds=DEFAULT_BOUNDS):
    """
    Estimate model parameters separately for each regime.
    
    Args:
        returns: pandas Series of log returns (daily)
        regime_labels: array of regime assignments (same length as returns)
        num_regimes: Number of regimes
        method: 'moment' for moment matching, 'qmle' for quasi-maximum likelihood
        bounds: Parameter bounds for optimization
        
    Returns:
        dict: {regime_id: params_array} where params_array is 9-element array for Bates model
        dict: {regime_id: moment_info} with fitting quality metrics
    """
    console.print(f"\n[bold yellow]Step 2: Estimating parameters for each regime using {method} method[/bold yellow]")
    
    params_dict = {}
    moment_info_dict = {}
    
    for regime_id in range(num_regimes):
        # Extract returns for this regime
        regime_mask = regime_labels == regime_id
        regime_returns = returns[regime_mask].values
        
        if len(regime_returns) < 50:  # Need minimum observations
            console.print(f"[yellow]⚠ Regime {regime_id}: Only {len(regime_returns)} observations, skipping[/yellow]")
            continue
        
        console.print(f"\n[cyan]Fitting Regime {regime_id} ({len(regime_returns)} observations)[/cyan]")
        
        # Calculate empirical moments for this regime
        emp_moments = compute_empirical_moments(regime_returns)
        console.print(f"  Empirical: mean={emp_moments['mean']:.6f}, std={emp_moments['std']:.6f}, "
                     f"skew={emp_moments['skew']:.3f}, kurt={emp_moments['kurt']:.3f}")
        
        if method == 'moment':
            # Use moment matching
            result = fit_bates_moment_matching(
                regime_returns,
                name=f"Regime_{regime_id}",
                restarts=5,  # Fewer restarts for speed
                maxiter=1000,
                bounds_vec=bounds,
                use_bounds=True,
                match_max_dd=False
            )
        elif method == 'qmle':
            # TODO: Implement QMLE estimation
            console.print(f"[yellow]⚠ QMLE method not yet implemented, using moment matching[/yellow]")
            result = fit_bates_moment_matching(
                regime_returns,
                name=f"Regime_{regime_id}",
                restarts=5,
                maxiter=1000,
                bounds_vec=bounds,
                use_bounds=True,
                match_max_dd=False
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if result is not None:
            params_dict[regime_id] = result['params']
            # Extract model moments from result dict (model_moments is a dict)
            model_moments = result.get('model_moments', {})
            moment_info_dict[regime_id] = {
                'n_obs': len(regime_returns),
                'objective': result.get('objective', np.nan),
                'emp_mean': emp_moments['mean'],
                'emp_std': emp_moments['std'],
                'emp_skew': emp_moments['skew'],
                'emp_kurt': emp_moments['kurt'],
                'model_mean': model_moments.get('mean', np.nan),
                'model_std': model_moments.get('std', np.nan),
                'model_skew': model_moments.get('skew', np.nan),
                'model_kurt': model_moments.get('kurt', np.nan),
            }
            console.print(f"[green]✓ Regime {regime_id} parameters estimated[/green]")
        else:
            console.print(f"[red]✗ Regime {regime_id} parameter estimation failed[/red]")
    
    return params_dict, moment_info_dict

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def estimate_regime_conditional_bates_params(returns, features='vol&skew', num_regimes=3,
                                             method='moment', random_seed=42):
    """
    Complete workflow: HMM regime inference + regime-conditional parameter estimation.
    
    Args:
        returns: pandas Series of log returns (daily)
        features: 'vol', 'skew', or 'vol&skew' for HMM features
        num_regimes: Number of regimes to infer
        method: 'moment' or 'qmle' for parameter estimation
        random_seed: Random seed for reproducibility
        
    Returns:
        dict: Results including params_dict, regime_labels, transition_matrix, etc.
    """
    console.print("[bold cyan]=" * 60)
    console.print("[bold cyan]Regime-Conditional Parameter Estimation[/bold cyan]")
    console.print("[bold cyan]=" * 60)
    
    # Step 1: Infer regimes using HMM
    regime_labels, transition_matrix, hmm_model = infer_regimes_hmm(
        returns, features=features, num_regimes=num_regimes, random_seed=random_seed
    )
    
    if regime_labels is None:
        console.print("[bold red]HMM inference failed, cannot proceed[/bold red]")
        return None
    
    # Step 2: Estimate parameters for each regime
    params_dict, moment_info_dict = estimate_regime_conditional_parameters(
        returns, regime_labels, num_regimes=num_regimes, method=method
    )
    
    # Prepare results
    results = {
        'params_dict': params_dict,
        'moment_info_dict': moment_info_dict,
        'regime_labels': regime_labels,
        'transition_matrix': transition_matrix,
        'hmm_model': hmm_model,
        'num_regimes': num_regimes,
        'features': features,
        'method': method
    }
    
    return results

# ============================================================================
# OUTPUT AND VISUALIZATION
# ============================================================================

def display_regime_conditional_results(results):
    """Display estimated parameters for each regime."""
    if results is None:
        console.print("[bold red]No results to display[/bold red]")
        return
    
    params_dict = results['params_dict']
    moment_info_dict = results['moment_info_dict']
    
    console.print("\n[bold cyan]=" * 80)
    console.print("[bold cyan]Regime-Conditional Parameter Estimates[/bold cyan]")
    console.print("[bold cyan]=" * 80)
    
    # Create parameter table
    table = Table(title="Bates Model Parameters by Regime")
    table.add_column("Parameter", style="cyan")
    for regime_id in sorted(params_dict.keys()):
        table.add_column(f"Regime {regime_id}", justify="right")
    
    param_names = ['mu', 'kappa', 'theta', 'nu', 'rho', 'v0', 'lam', 'mu_J', 'sigma_J']
    param_labels = {
        'mu': 'μ (drift)',
        'kappa': 'κ (mean reversion)',
        'theta': 'θ (long-run var)',
        'nu': 'ν (vol-of-vol)',
        'rho': 'ρ (correlation)',
        'v0': 'v₀ (initial var)',
        'lam': 'λ (jump intensity)',
        'mu_J': 'μ_J (jump mean)',
        'sigma_J': 'σ_J (jump vol)'
    }
    
    for param_name in param_names:
        row = [param_labels[param_name]]
        for regime_id in sorted(params_dict.keys()):
            val = params_dict[regime_id][IDX[param_name]]
            row.append(f"{val:.6f}")
        table.add_row(*row)
    
    console.print(table)
    
    # Display moment matching quality
    console.print("\n[bold cyan]Moment Matching Quality by Regime[/bold cyan]")
    quality_table = Table()
    quality_table.add_column("Regime", style="cyan")
    quality_table.add_column("N Obs", justify="right")
    quality_table.add_column("Emp Mean", justify="right")
    quality_table.add_column("Model Mean", justify="right")
    quality_table.add_column("Emp Std", justify="right")
    quality_table.add_column("Model Std", justify="right")
    
    for regime_id in sorted(params_dict.keys()):
        info = moment_info_dict[regime_id]
        quality_table.add_row(
            str(regime_id),
            str(info['n_obs']),
            f"{info['emp_mean']:.6f}",
            f"{info['model_mean']:.6f}",
            f"{info['emp_std']:.6f}",
            f"{info['model_std']:.6f}"
        )
    
    console.print(quality_table)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regime-Conditional Parameter Estimation")
    parser.add_argument('--method', choices=['moment', 'qmle'], default='moment',
                       help='Parameter estimation method')
    parser.add_argument('--hmm-features', choices=['vol', 'skew', 'vol&skew'], default='vol&skew',
                       help='Features for HMM regime detection')
    parser.add_argument('--num-regimes', type=int, default=3,
                       help='Number of regimes to infer')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    DEBUG_MODE = args.debug
    
    console.print("[yellow]Note: This script requires input data (returns series)[/yellow]")
    console.print("[yellow]To use this module, import and call estimate_regime_conditional_bates_params()[/yellow]")
    console.print("[yellow]Example usage will be added when integrated with data loading[/yellow]")

