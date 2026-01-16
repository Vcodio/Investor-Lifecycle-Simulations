"""
Streamlit App for Lifecycle Retirement Simulation
"""
import streamlit as st
import sys
import os

# IMPORTANT: Suppress tqdm BEFORE importing any modules that use it
# This prevents tqdm progress bars from interfering with Streamlit's output
os.environ['TQDM_DISABLE'] = '1'

import numpy as np
import pandas as pd
import logging
from io import StringIO
import io
import contextlib

# Try to import plotly for better visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    # Fallback to matplotlib
    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False

# Add parent directory to path so we can import LIFECYCLE MODEL as a package
parent_dir = os.path.dirname(os.path.abspath(__file__))
lifecycle_model_dir = os.path.join(parent_dir, 'LIFECYCLE MODEL')

# Import modules from LIFECYCLE MODEL
# Since the modules use relative imports, we need to set up the package structure properly
try:
    import importlib.util
    import importlib
    
    # First, set up a package structure for relative imports to work
    # Create a package name that Python can handle (replace space with underscore)
    package_name = 'lifecycle_model'
    
    # Create the package module
    package_module = type(sys)('lifecycle_model')
    sys.modules['lifecycle_model'] = package_module
    
    # Helper function to load a module and handle relative imports
    def load_module(module_name, file_path):
        """Load a module from a file path, setting up parent package for relative imports"""
        full_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(full_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {module_name} from {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        
        # Set __package__ for relative imports to work
        # This tells Python that this module is part of the package
        module.__package__ = package_name
        module.__name__ = full_name
        
        # Execute the module (this will trigger relative imports)
        spec.loader.exec_module(module)
        return module
    
    # Load modules in dependency order
    # 1. config (no dependencies on other local modules)
    config_module = load_module('config', os.path.join(lifecycle_model_dir, 'config.py'))
    SimulationConfig = config_module.SimulationConfig
    
    # 2. cython_wrapper (no dependencies on other local modules)
    # Ensure build directory is in path before loading cython_wrapper
    build_dir = os.path.join(parent_dir, 'build')
    if os.path.exists(build_dir) and build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    
    cython_module = load_module('cython_wrapper', os.path.join(lifecycle_model_dir, 'cython_wrapper.py'))
    CYTHON_AVAILABLE = cython_module.CYTHON_AVAILABLE
    
    # Verify Cython is actually working by checking if the functions exist
    if CYTHON_AVAILABLE:
        try:
            # Try to access the actual Cython functions to verify they work
            has_cython_func = hasattr(cython_module, 'simulate_monthly_return_svj_cython') or \
                             'lrs_cython' in sys.modules
            if not has_cython_func:
                # Double-check by trying to import directly
                try:
                    import lrs_cython
                    CYTHON_AVAILABLE = True
                except ImportError:
                    CYTHON_AVAILABLE = False
        except Exception:
            CYTHON_AVAILABLE = False
    
    # 3. bootstrap (no dependencies on other local modules)
    bootstrap_module = load_module('bootstrap', os.path.join(lifecycle_model_dir, 'bootstrap.py'))
    load_bootstrap_data = bootstrap_module.load_bootstrap_data
    
    # 4. earnings (may depend on cython_wrapper)
    earnings_module = load_module('earnings', os.path.join(lifecycle_model_dir, 'earnings.py'))
    
    # 5. utils (no dependencies on other local modules)
    utils_module = load_module('utils', os.path.join(lifecycle_model_dir, 'utils.py'))
    convert_geometric_to_arithmetic = utils_module.convert_geometric_to_arithmetic
    calculate_nominal_value = utils_module.calculate_nominal_value
    
    # 6. utility (no dependencies on other local modules)
    utility_module = load_module('utility', os.path.join(lifecycle_model_dir, 'utility.py'))
    calculate_total_utility_ex_ante = utility_module.calculate_total_utility_ex_ante
    calculate_ce_for_crra = utility_module.calculate_ce_for_crra
    
    # 6b. savings_rate (depends on utility)
    savings_rate_module = load_module('savings_rate', os.path.join(lifecycle_model_dir, 'savings_rate.py'))
    calculate_equivalent_savings_rate_scaling = savings_rate_module.calculate_equivalent_savings_rate_scaling
    
    # 7. simulation (depends on cython_wrapper, bootstrap, earnings)
    simulation_module = load_module('simulation', os.path.join(lifecycle_model_dir, 'simulation.py'))
    find_required_principal = simulation_module.find_required_principal
    run_accumulation_simulations = simulation_module.run_accumulation_simulations
    calculate_expected_return_from_data = simulation_module.calculate_expected_return_from_data
    
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)
    import traceback
    IMPORT_ERROR += f"\n\nTraceback:\n{traceback.format_exc()}"
    # #region agent log
    try:
        import json
        import time
        import os
        log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'debug.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'id': 'log_import_error', 'timestamp': time.time() * 1000, 'location': 'app.py:106', 'message': 'ImportError during module loading', 'data': {'error': str(e), 'traceback': traceback.format_exc()}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'A'}) + '\n')
    except Exception as log_err: pass
    # #endregion
except Exception as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = f"Error loading modules: {str(e)}"
    import traceback
    IMPORT_ERROR += f"\n\nTraceback:\n{traceback.format_exc()}"
    # #region agent log
    try:
        import json
        import time
        import os
        log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'debug.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'id': 'log_module_error', 'timestamp': time.time() * 1000, 'location': 'app.py:111', 'message': 'Exception during module loading', 'data': {'error': str(e), 'traceback': traceback.format_exc()}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'A'}) + '\n')
    except Exception as log_err: pass
    # #endregion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Lifecycle Retirement Simulation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'config' not in st.session_state:
    st.session_state.config = None

def create_config_from_sidebar():
    """Create SimulationConfig from sidebar inputs"""
    config = SimulationConfig()
    
    # Basic Parameters
    with st.sidebar.expander("📋 Basic Parameters", expanded=True):
        config.initial_age = st.number_input("Initial Age", min_value=18, max_value=100, value=config.initial_age, step=1)
        config.death_age = st.number_input("Death Age", min_value=50, max_value=120, value=config.death_age, step=1)
        config.initial_portfolio = st.number_input("Initial Portfolio ($)", min_value=0.0, value=float(config.initial_portfolio), step=10000.0, format="%.0f")
        config.annual_income_real = st.number_input("Annual Income (Real $)", min_value=0.0, value=float(config.annual_income_real), step=1000.0, format="%.0f")
        config.savings_rate = st.slider("Savings Rate", min_value=0.0, max_value=1.0, value=float(config.savings_rate), step=0.01, format="%.2f", help="Fraction of income saved during accumulation phase")
        config.spending_real = st.number_input("Target Annual Spending (Real $)", min_value=0.0, value=float(config.spending_real), step=1000.0, format="%.0f")
    
    # Social Security
    with st.sidebar.expander("💰 Social Security", expanded=False):
        config.include_social_security = st.checkbox("Include Social Security", value=config.include_social_security)
        if config.include_social_security:
            config.social_security_real = st.number_input("Social Security Benefit (Real $)", min_value=0.0, value=float(config.social_security_real), step=1000.0, format="%.0f")
            config.social_security_start_age = st.number_input("Social Security Start Age", min_value=62, max_value=70, value=config.social_security_start_age, step=1)
    
    # Simulation Settings
    with st.sidebar.expander("⚙️ Simulation Settings", expanded=False):
        # Ensure default value is within valid range
        default_outer = max(10, config.num_outer) if config.num_outer >= 10 else 10
        config.num_outer = st.number_input("Number of Outer Simulations", min_value=10, max_value=100000, value=default_outer, step=10, help="More simulations = more accurate but slower")
        
        # Ensure default value is within valid range
        default_nested = max(50, config.num_nested) if config.num_nested >= 50 else 50
        config.num_nested = st.number_input("Number of Nested Simulations", min_value=50, max_value=10000, value=default_nested, step=50, help="Used for principal calculation")
        config.success_target = st.slider("Success Target (%)", min_value=0.5, max_value=1.0, value=config.success_target, step=0.01, format="%.2f")
        seed_input = st.number_input("Random Seed (optional)", min_value=0, value=None, step=1, help="Leave empty for random")
        config.seed = int(seed_input) if seed_input is not None else None
    
    # Retirement Age Range
    with st.sidebar.expander("🎯 Retirement Age Range", expanded=False):
        config.retirement_age_min = st.number_input("Min Retirement Age", min_value=18, max_value=100, value=config.retirement_age_min, step=1)
        config.retirement_age_max = st.number_input("Max Retirement Age", min_value=18, max_value=100, value=config.retirement_age_max, step=1)
    
    # Model Selection
    with st.sidebar.expander("📈 Model Selection", expanded=False):
        config.use_block_bootstrap = st.checkbox("Use Block Bootstrap", value=config.use_block_bootstrap, help="Use historical data blocks instead of parametric model")
        
        if config.use_block_bootstrap:
            config.bootstrap_csv_path = st.text_input("Bootstrap CSV Path", value=config.bootstrap_csv_path)
            config.portfolio_column_name = st.text_input("Portfolio Column Name", value=config.portfolio_column_name)
            config.inflation_column_name = st.text_input("Inflation Column Name", value=config.inflation_column_name)
            config.block_length_years = st.number_input("Block Length (Years)", min_value=1, max_value=20, value=config.block_length_years, step=1)
            config.block_overlapping = st.checkbox("Overlapping Blocks", value=config.block_overlapping)
        else:
            # Parametric Model Parameters (shown when not using bootstrap)
            st.write("**Bates/Heston Model Parameters**")
            config.params['mu'] = st.number_input("mu (drift)", value=float(config.params['mu']), step=0.001, format="%.4f")
            config.params['kappa'] = st.number_input("kappa (mean reversion)", value=float(config.params['kappa']), step=0.01, format="%.4f")
            config.params['theta'] = st.number_input("theta (long-term variance)", value=float(config.params['theta']), step=0.001, format="%.4f")
            config.params['nu'] = st.number_input("nu (volatility of variance)", value=float(config.params['nu']), step=0.001, format="%.4f")
            config.params['rho'] = st.number_input("rho (correlation)", value=float(config.params['rho']), step=0.01, format="%.4f")
            config.params['v0'] = st.number_input("v0 (initial variance)", value=float(config.params['v0']), step=0.001, format="%.4f")
            config.params['lam'] = st.number_input("lam (jump intensity)", value=float(config.params['lam']), step=0.01, format="%.4f")
            config.params['mu_J'] = st.number_input("mu_J (jump mean)", value=float(config.params['mu_J']), step=0.001, format="%.4f")
            config.params['sigma_J'] = st.number_input("sigma_J (jump std dev)", value=float(config.params['sigma_J']), step=0.001, format="%.4f")
            
            st.write("**Inflation Parameters**")
            config.mean_inflation_geometric = st.number_input("Mean Inflation (Geometric)", value=float(config.mean_inflation_geometric), step=0.001, format="%.4f")
            config.std_inflation = st.number_input("Std Inflation", value=float(config.std_inflation), step=0.001, format="%.4f")
    
    # GKOS Earnings Parameters
    with st.sidebar.expander("💼 GKOS Earnings Parameters", expanded=False):
        config.gkos_params['RHO'] = st.number_input("RHO (persistence)", value=float(config.gkos_params['RHO']), step=0.01, format="%.4f")
        config.gkos_params['SIGMA_ETA'] = st.number_input("SIGMA_ETA (persistent shock std)", value=float(config.gkos_params['SIGMA_ETA']), step=0.01, format="%.4f")
        config.gkos_params['MIXTURE_PROB_NORMAL'] = st.number_input("MIXTURE_PROB_NORMAL", value=float(config.gkos_params['MIXTURE_PROB_NORMAL']), step=0.01, format="%.4f")
        config.gkos_params['MIXTURE_PROB_TAIL'] = st.number_input("MIXTURE_PROB_TAIL", value=float(config.gkos_params['MIXTURE_PROB_TAIL']), step=0.01, format="%.4f")
        config.gkos_params['SKEWNESS_PARAM'] = st.number_input("SKEWNESS_PARAM", value=float(config.gkos_params['SKEWNESS_PARAM']), step=0.1, format="%.2f")
        config.gkos_params['TAIL_VARIANCE_MULTIPLIER'] = st.number_input("TAIL_VARIANCE_MULTIPLIER", value=float(config.gkos_params['TAIL_VARIANCE_MULTIPLIER']), step=1.0, format="%.1f")
        config.gkos_params['SIGMA_EPSILON'] = st.number_input("SIGMA_EPSILON (transitory shock)", value=float(config.gkos_params['SIGMA_EPSILON']), step=0.01, format="%.4f")
        config.gkos_params['SIGMA_Z0'] = st.number_input("SIGMA_Z0 (initial heterogeneity)", value=float(config.gkos_params['SIGMA_Z0']), step=0.01, format="%.4f")
        config.gkos_params['AGE_PROFILE_A'] = st.number_input("AGE_PROFILE_A (quadratic)", value=float(config.gkos_params['AGE_PROFILE_A']), step=0.0001, format="%.4f")
        config.gkos_params['AGE_PROFILE_B'] = st.number_input("AGE_PROFILE_B (linear)", value=float(config.gkos_params['AGE_PROFILE_B']), step=0.01, format="%.4f")
        config.gkos_params['AGE_PEAK'] = st.number_input("AGE_PEAK", value=float(config.gkos_params['AGE_PEAK']), step=0.5, format="%.1f")
    
    # Utility Parameters
    with st.sidebar.expander("📊 Utility Parameters", expanded=False):
        st.info("⚠️ **Note:** Utility calculations are in development. This feature will be used to compare different portfolios.")
        config.enable_utility_calculations = st.checkbox("Enable Utility Calculations", value=config.enable_utility_calculations)
        if config.enable_utility_calculations:
            config.gamma = st.number_input("gamma (CRRA risk aversion)", min_value=0.1, max_value=10.0, value=float(config.gamma), step=0.1, format="%.2f")
            config.beta = st.number_input("beta (time discount)", min_value=0.8, max_value=1.0, value=float(config.beta), step=0.01, format="%.3f")
            config.k_bequest = st.number_input("k_bequest (bequest threshold)", min_value=0.0, value=float(config.k_bequest), step=1000.0, format="%.0f")
            config.theta = st.number_input("theta (bequest weight)", min_value=0.0, max_value=1.0, value=float(config.theta), step=0.1, format="%.2f")
            config.household_size = st.number_input("household_size", min_value=0.1, max_value=10.0, value=float(config.household_size), step=0.1, format="%.1f")
    
    # Withdrawal Strategy
    with st.sidebar.expander("💸 Withdrawal Strategy", expanded=False):
        config.use_amortization = st.checkbox("Use Amortization-Based Withdrawal", value=config.use_amortization)
        if config.use_amortization:
            amort_return_input = st.number_input("Amortization Expected Return (leave empty for auto)", value=None, step=0.001, format="%.4f", help="Auto-calculated from data if empty")
            config.amortization_expected_return = float(amort_return_input) if amort_return_input is not None else None
            config.amortization_min_spending_threshold = st.slider("Min Spending Threshold (fraction)", min_value=0.0, max_value=1.0, value=config.amortization_min_spending_threshold, step=0.05, format="%.2f")
            config.amortization_desired_bequest = st.number_input("Desired Bequest (Real $)", min_value=0.0, value=float(config.amortization_desired_bequest), step=1000.0, format="%.0f", help="Target portfolio value at death. If 0, portfolio is consumed to zero.")
    
    # Output Settings
    with st.sidebar.expander("💾 Output Settings", expanded=False):
        config.generate_csv_summary = st.checkbox("Generate CSV Summary", value=config.generate_csv_summary)
        if config.generate_csv_summary:
            config.num_sims_to_export = st.number_input("Number of Sims to Export", min_value=1, max_value=1000, value=config.num_sims_to_export, step=1)
    
    # Advanced Settings
    with st.sidebar.expander("🔧 Advanced Settings", expanded=False):
        config.use_principal_deviation_threshold = st.checkbox("Use Principal Deviation Threshold", value=config.use_principal_deviation_threshold)
        if config.use_principal_deviation_threshold:
            config.principal_deviation_threshold = st.slider("Principal Deviation Threshold", min_value=0.01, max_value=0.2, value=config.principal_deviation_threshold, step=0.01, format="%.3f")
        config.num_workers = st.number_input("Number of Workers (multiprocessing)", min_value=1, max_value=os.cpu_count() or 1, value=config.num_workers, step=1)
    
    return config

def estimate_runtime(config):
    """Estimate simulation runtime in seconds"""
    # More accurate estimates based on actual performance
    # Stage 1: Binary search for each age
    # - Each age: ~20 iterations * num_nested sims * ~0.5s per sim (with Cython)
    num_ages = config.retirement_age_max - config.retirement_age_min + 1
    time_per_sim_stage1 = 0.5 if CYTHON_AVAILABLE else 2.0  # seconds per nested sim
    binary_search_iterations = 20  # Average iterations per age
    stage1_time = num_ages * binary_search_iterations * config.num_nested * time_per_sim_stage1
    
    # Stage 2: Accumulation simulations
    # - Each outer sim: ~0.1s (with Cython) to ~0.5s (without)
    time_per_outer_sim = 0.1 if CYTHON_AVAILABLE else 0.5
    stage2_time = config.num_outer * time_per_outer_sim
    
    total_seconds = stage1_time + stage2_time
    
    return total_seconds

def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        return f"{seconds/3600:.1f} hours"

def create_plot_required_principal_nominal(ages, principals_nominal, swr, config, median_age):
    """Create plot for required principal in nominal terms - returns figure"""
    if HAS_PLOTLY:
        from plotly.subplots import make_subplots
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Principal line
        fig.add_trace(
            go.Scatter(x=ages, y=principals_nominal, mode='lines+markers',
                      name='Required Principal (Nominal $)', 
                      line=dict(color='#00ffff', width=3),
                      marker=dict(size=8, color='#00ffff', line=dict(width=1, color='white'))),
            secondary_y=False
        )
        
        # Withdrawal rate line
        fig.add_trace(
            go.Scatter(x=ages, y=np.array(swr), mode='lines+markers',
                      name='Withdrawal Rate (%)',
                      line=dict(color='#ff00ff', width=3, dash='dash'),
                      marker=dict(size=8, symbol='x', color='#ff00ff')),
            secondary_y=True
        )
        
        # Reference lines
        if config.include_social_security:
            fig.add_vline(x=config.social_security_start_age, line_dash="dot", line_color="lime",
                         annotation_text=f"SS Age {config.social_security_start_age}", 
                         annotation_position="top")
        
        if not np.isnan(median_age):
            fig.add_vline(x=median_age, line_dash="dash", line_color="yellow",
                         annotation_text=f"Median: {median_age:.1f}", 
                         annotation_position="top")
        
        success_pct = int(config.success_target * 100)
        fig.update_layout(
            title=f"Required Principal & Withdrawal Rate for {success_pct}% Success (Nominal)",
            template='plotly_white',
            height=600,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent to match Streamlit background
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent to match Streamlit background
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=0,
                bgcolor='rgba(0,0,0,0)',  # Transparent to match Streamlit background
                bordercolor='rgba(0,0,0,0)',  # Transparent border
                borderwidth=0
            )
        )
        
        fig.update_xaxes(title_text="Retirement Age", showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(title_text="Required Principal ($)", secondary_y=False, 
                       showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
                       tickformat='$,.0f')
        fig.update_yaxes(title_text="Withdrawal Rate (%)", secondary_y=True,
                       showgrid=False, tickformat='.2f')
        
        return fig
    else:
        # Fallback to matplotlib
        try:
            import matplotlib.pyplot as plt
            plt.style.use('dark_background')
            fig, ax1 = plt.subplots(figsize=(12, 7), facecolor='black')
            fig.patch.set_facecolor('black')
            ax1.set_facecolor('black')
            ax1.set_xlabel('Retirement Age', fontsize=12, color='white')
            ax1.set_ylabel('Required Principal ($)', fontsize=12, color='white')
            ax1.tick_params(axis='both', colors='white')
            ax1.plot(ages, principals_nominal, color='#00ffff', marker='o', markersize=6,
                    label='Required Principal (Nominal $)', linewidth=2.5, 
                    markerfacecolor='#00ffff', markeredgecolor='white', markeredgewidth=1,
                    alpha=0.9)
            ax1.grid(True, linestyle='--', alpha=0.2, color='white', linewidth=0.8)
            ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            ax2 = ax1.twinx()
            ax2.set_facecolor('black')
            ax2.set_ylabel('Withdrawal Rate (%)', fontsize=12, color='white')
            ax2.plot(ages, np.array(swr), color='#ff00ff', marker='x', markersize=7,
                    linestyle='--', label='Withdrawal Rate', linewidth=2.5,
                    markeredgewidth=2, alpha=0.9)
            ax2.tick_params(axis='y', colors='white')
            ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}%'))
            
            if config.include_social_security:
                ax1.axvline(x=config.social_security_start_age, color='lime',
                           linestyle=':', label=f'Social Security (Age {config.social_security_start_age})',
                           linewidth=2)
            
            if not np.isnan(median_age):
                ax1.axvline(x=median_age, color='yellow', linestyle='--',
                           label=f'Median Retirement Age ({median_age:.1f})', linewidth=2)
            
            success_pct = int(config.success_target * 100)
            fig.suptitle(f"Required Principal & Withdrawal Rate for {success_pct}% Success (Nominal)",
                        fontsize=15, color='white', fontweight='bold')
            fig.tight_layout()
            fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes,
                      facecolor='black', edgecolor='white', labelcolor='white', framealpha=0.8, fontsize=10)
            return fig
        except ImportError:
            return None

def create_plot_required_principal_real(ages, principals_real, swr, config, median_age):
    """Create plot for required principal in real terms - returns figure"""
    if HAS_PLOTLY:
        from plotly.subplots import make_subplots
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Principal line
        fig.add_trace(
            go.Scatter(x=ages, y=principals_real, mode='lines+markers',
                      name='Required Principal (Real $)', 
                      line=dict(color='#ff8800', width=3),
                      marker=dict(size=8, color='#ff8800', line=dict(width=1, color='white'))),
            secondary_y=False
        )
        
        # Withdrawal rate line
        fig.add_trace(
            go.Scatter(x=ages, y=np.array(swr), mode='lines+markers',
                      name='Withdrawal Rate (%)',
                      line=dict(color='#ff00ff', width=3, dash='dash'),
                      marker=dict(size=8, symbol='x', color='#ff00ff')),
            secondary_y=True
        )
        
        # Reference lines
        if config.include_social_security:
            fig.add_vline(x=config.social_security_start_age, line_dash="dot", line_color="cyan",
                         annotation_text=f"SS Age {config.social_security_start_age}", 
                         annotation_position="top")
        
        if not np.isnan(median_age):
            fig.add_vline(x=median_age, line_dash="dash", line_color="yellow",
                         annotation_text=f"Median: {median_age:.1f}", 
                         annotation_position="top")
        
        success_pct = int(config.success_target * 100)
        fig.update_layout(
            title=f"Required Principal & Withdrawal Rate for {success_pct}% Success (Real)",
            template='plotly_white',
            height=600,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent to match Streamlit background
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent to match Streamlit background
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=0,
                bgcolor='rgba(0,0,0,0)',  # Transparent to match Streamlit background
                bordercolor='rgba(0,0,0,0)',  # Transparent border
                borderwidth=0
            )
        )
        
        fig.update_xaxes(title_text="Retirement Age", showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(title_text="Required Principal ($)", secondary_y=False, 
                       showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
                       tickformat='$,.0f')
        fig.update_yaxes(title_text="Withdrawal Rate (%)", secondary_y=True,
                       showgrid=False, tickformat='.2f')
        
        return fig
    else:
        # Fallback to matplotlib
        try:
            import matplotlib.pyplot as plt
            plt.style.use('dark_background')
            fig, ax1 = plt.subplots(figsize=(12, 7), facecolor='black')
            fig.patch.set_facecolor('black')
            ax1.set_facecolor('black')
            ax1.set_xlabel('Retirement Age', fontsize=12, color='white')
            ax1.set_ylabel('Required Principal ($)', fontsize=12, color='white')
            ax1.tick_params(axis='both', colors='white')
            ax1.plot(ages, principals_real, color='#ff8800', marker='o', markersize=6,
                    label='Required Principal (Real $)', linewidth=2.5,
                    markerfacecolor='#ff8800', markeredgecolor='white', markeredgewidth=1,
                    alpha=0.9)
            ax1.grid(True, linestyle='--', alpha=0.2, color='white', linewidth=0.8)
            ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            ax2 = ax1.twinx()
            ax2.set_facecolor('black')
            ax2.set_ylabel('Withdrawal Rate (%)', fontsize=12, color='white')
            ax2.plot(ages, np.array(swr), color='#ff00ff', marker='x', markersize=7,
                    linestyle='--', label='Withdrawal Rate', linewidth=2.5,
                    markeredgewidth=2, alpha=0.9)
            ax2.tick_params(axis='y', colors='white')
            ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}%'))
            
            if config.include_social_security:
                ax1.axvline(x=config.social_security_start_age, color='cyan',
                           linestyle=':', label=f'Social Security (Age {config.social_security_start_age})',
                           linewidth=2)
            
            if not np.isnan(median_age):
                ax1.axvline(x=median_age, color='yellow', linestyle='--',
                           label=f'Median Retirement Age ({median_age:.1f})', linewidth=2)
            
            success_pct = int(config.success_target * 100)
            fig.suptitle(f"Required Principal & Withdrawal Rate for {success_pct}% Success (Real)",
                        fontsize=15, color='white', fontweight='bold')
            fig.tight_layout()
            fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes,
                      facecolor='black', edgecolor='white', labelcolor='white', framealpha=0.8, fontsize=10)
            return fig
        except ImportError:
            return None

def create_plot_retirement_age_distribution(valid_ages, median_age):
    """Create plot for retirement age distribution - returns figure"""
    if valid_ages.size == 0:
        return None
    
    if HAS_PLOTLY:
        valid_ages_int = np.round(valid_ages).astype(int)
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=valid_ages_int,
            name='Retirement Age',
            nbinsx=len(np.unique(valid_ages_int)),
            marker_color='#00ffff',
            opacity=0.8,
            hovertemplate='Age: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add median line
        fig.add_vline(x=median_age, line_dash="dash", line_color="red", line_width=3,
                     annotation_text=f"Median: {median_age:.1f}", annotation_position="top")
        
        fig.update_layout(
            title='Distribution of Retirement Ages',
            xaxis_title='Retirement Age',
            yaxis_title='Frequency',
            template='plotly_white',
            height=500,
            showlegend=False,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent to match Streamlit background
            paper_bgcolor='rgba(0,0,0,0)'  # Transparent to match Streamlit background
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        
        return fig
    else:
        # Fallback to matplotlib
        try:
            import matplotlib.pyplot as plt
            plt.style.use('dark_background')
            valid_ages_int = np.round(valid_ages).astype(int)
            min_age = int(np.min(valid_ages_int))
            max_age = int(np.max(valid_ages_int)) + 1
            
            fig, ax = plt.subplots(figsize=(12, 7), facecolor='black')
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            n, bins, patches = ax.hist(valid_ages_int, bins=range(min_age, max_age + 1), 
                                         color='#00ffff', alpha=0.8, align='left', 
                                         edgecolor='#00ffff', linewidth=1.5)
            
            for i, patch in enumerate(patches):
                intensity = i / len(patches) if len(patches) > 0 else 0
                patch.set_facecolor(plt.cm.viridis(intensity))
                patch.set_edgecolor('#00ffff')
                patch.set_linewidth(1.5)
            
            ax.set_title('Distribution of Retirement Ages', fontsize=15, color='white', fontweight='bold')
            ax.axvline(median_age, color='#ff0000', linestyle='--', linewidth=3,
                       label=f'Median: {median_age:.1f}', alpha=0.9)
            ax.set_xlabel('Retirement Age', fontsize=13, color='white', fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=13, color='white', fontweight='bold')
            ax.tick_params(axis='both', colors='white', labelsize=11)
            ax.legend(facecolor='black', edgecolor='white', labelcolor='white', framealpha=0.8, fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.2, color='white', linewidth=0.8)
            fig.tight_layout()
            return fig
        except ImportError:
            return None

def create_amortization_visualizations(amortization_stats_list, config, final_bequest_nominal, final_bequest_real):
    """Create amortization visualization plots for Streamlit"""
    try:
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
        
        if not amortization_stats_list or all(stats is None for stats in amortization_stats_list):
            st.warning("No amortization statistics available for visualization")
            return
        
        # Collect all withdrawals by retirement year
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
            st.warning("No withdrawal data available")
            return
        
        # Filter out years beyond reasonable retirement length
        # Maximum reasonable years = death_age - min_retirement_age
        max_reasonable_years = config.death_age - config.retirement_age_min
        all_years = sorted(all_withdrawals_by_year.keys())
        years = [y for y in all_years if y <= max_reasonable_years]
        
        if not years:
            st.warning("No valid withdrawal data available")
            return
        
        medians = [np.median(all_withdrawals_by_year[y]) for y in years]
        p25 = [np.percentile(all_withdrawals_by_year[y], 25) for y in years]
        p75 = [np.percentile(all_withdrawals_by_year[y], 75) for y in years]
        p10 = [np.percentile(all_withdrawals_by_year[y], 10) for y in years]
        p90 = [np.percentile(all_withdrawals_by_year[y], 90) for y in years]
        
        # Use Plotly for better-looking interactive plots
        if HAS_PLOTLY:
            fig1 = go.Figure()
            
            # Add percentile bands
            fig1.add_trace(go.Scatter(
                x=years + years[::-1],
                y=p90 + p10[::-1],
                fill='toself',
                fillcolor='rgba(0, 255, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='10th-90th Percentile',
                showlegend=True,
                hoverinfo='skip'
            ))
            
            fig1.add_trace(go.Scatter(
                x=years + years[::-1],
                y=p75 + p25[::-1],
                fill='toself',
                fillcolor='rgba(0, 100, 255, 0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                name='25th-75th Percentile',
                showlegend=True,
                hoverinfo='skip'
            ))
            
            # Add median line
            fig1.add_trace(go.Scatter(
                x=years,
                y=medians,
                mode='lines+markers',
                name='Median',
                line=dict(color='yellow', width=3),
                marker=dict(size=8, color='yellow'),
                hovertemplate='Year: %{x}<br>Median: $%{y:,.0f}<extra></extra>'
            ))
            
            # Add reference lines
            if initial_spending:
                threshold = config.amortization_min_spending_threshold * initial_spending
                fig1.add_hline(y=initial_spending, line_dash="dash", line_color="red", line_width=2,
                              annotation_text=f"Initial: ${initial_spending:,.0f}", annotation_position="right")
                fig1.add_hline(y=threshold, line_dash="dot", line_color="orange", line_width=2,
                              annotation_text=f"Threshold: ${threshold:,.0f}", annotation_position="right")
            
            fig1.update_layout(
                title='Amortization-Based Withdrawals Over Time',
                xaxis_title='Years in Retirement',
                yaxis_title='Annual Withdrawal (Real $)',
                template='plotly_white',
                height=600,
                font=dict(size=12),
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent to match Streamlit background
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent to match Streamlit background
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='gray',
                    borderwidth=1
                )
            )
            fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
            fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
                            tickformat='$,.0f')
            
            st.plotly_chart(fig1, use_container_width=True)
        else:
            # Fallback to matplotlib
            import matplotlib.pyplot as plt
            fig1, ax1 = plt.subplots(figsize=(12, 7), facecolor='black')
            fig1.patch.set_facecolor('black')
            ax1.set_facecolor('black')
            
            ax1.fill_between(years, p10, p90, alpha=0.2, color='cyan', label='10th-90th Percentile')
            ax1.fill_between(years, p25, p75, alpha=0.3, color='blue', label='25th-75th Percentile')
            ax1.plot(years, medians, 'o-', color='yellow', linewidth=2.5, markersize=6, label='Median', alpha=0.9)
            
            if initial_spending:
                threshold = config.amortization_min_spending_threshold * initial_spending
                ax1.axhline(y=initial_spending, color='red', linestyle='--', linewidth=2, 
                           label=f'Initial Spending (${initial_spending:,.0f})')
                ax1.axhline(y=threshold, color='orange', linestyle=':', linewidth=2,
                           label=f'Min Threshold ({config.amortization_min_spending_threshold*100:.0f}% = ${threshold:,.0f})')
            
            ax1.set_xlabel('Years in Retirement', fontsize=13, color='white', fontweight='bold')
            ax1.set_ylabel('Annual Withdrawal (Real $)', fontsize=13, color='white', fontweight='bold')
            ax1.set_title('Amortization-Based Withdrawals Over Time', fontsize=15, color='white', fontweight='bold')
            ax1.tick_params(axis='both', colors='white', labelsize=11)
            ax1.legend(facecolor='black', edgecolor='white', labelcolor='white', framealpha=0.8, fontsize=10)
            ax1.grid(True, alpha=0.2, color='white', linewidth=0.8)
            ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
            
            st.pyplot(fig1)
        
        # Plot 2: Withdrawal distribution
        all_withdrawals = []
        for stats in amortization_stats_list:
            if stats and stats.get('withdrawals'):
                all_withdrawals.extend(stats['withdrawals'])
        
        if all_withdrawals:
            all_withdrawals = np.array(all_withdrawals)
            
            if HAS_PLOTLY:
                fig2 = go.Figure()
                
                fig2.add_trace(go.Histogram(
                    x=all_withdrawals,
                    name='Withdrawals',
                    nbinsx=50,
                    marker_color='#00ffff',
                    opacity=0.7,
                    hovertemplate='Withdrawal: $%{x:,.0f}<br>Count: %{y}<extra></extra>'
                ))
                
                # Add reference lines
                if initial_spending:
                    threshold = config.amortization_min_spending_threshold * initial_spending
                    fig2.add_vline(x=initial_spending, line_dash="dash", line_color="red", line_width=2.5,
                                  annotation_text=f"Initial: ${initial_spending:,.0f}", annotation_position="top")
                    fig2.add_vline(x=threshold, line_dash="dot", line_color="orange", line_width=2.5,
                                  annotation_text=f"Threshold: ${threshold:,.0f}", annotation_position="top")
                
                median_w = np.median(all_withdrawals)
                fig2.add_vline(x=median_w, line_dash="solid", line_color="yellow", line_width=2.5,
                              annotation_text=f"Median: ${median_w:,.0f}", annotation_position="top")
                
                fig2.update_layout(
                    title='Distribution of Annual Withdrawals',
                    xaxis_title='Annual Withdrawal (Real $)',
                    yaxis_title='Frequency',
                    template='plotly_white',
                    height=500,
                    showlegend=False,
                    font=dict(size=12),
                    plot_bgcolor='rgba(0,0,0,0)',  # Transparent to match Streamlit background
                    paper_bgcolor='rgba(0,0,0,0)'  # Transparent to match Streamlit background
                )
                fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)', tickformat='$,.0f')
                fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
                
                st.plotly_chart(fig2, use_container_width=True)
            else:
                # Fallback to matplotlib
                import matplotlib.pyplot as plt
                fig2, ax2 = plt.subplots(figsize=(12, 7), facecolor='black')
                fig2.patch.set_facecolor('black')
                ax2.set_facecolor('black')
                
                n_bins = min(50, len(np.unique(all_withdrawals)))
                counts, bins, patches = ax2.hist(all_withdrawals, bins=n_bins, color='#00ffff', 
                                                edgecolor='#00ffff', linewidth=1.5, alpha=0.7)
                
                # Add gradient to histogram
                for i, patch in enumerate(patches):
                    intensity = i / len(patches) if len(patches) > 0 else 0
                    patch.set_facecolor(plt.cm.viridis(intensity))
                
                if initial_spending:
                    threshold = config.amortization_min_spending_threshold * initial_spending
                    ax2.axvline(x=initial_spending, color='red', linestyle='--', linewidth=2.5,
                               label=f'Initial: ${initial_spending:,.0f}')
                    ax2.axvline(x=threshold, color='orange', linestyle=':', linewidth=2.5,
                               label=f'Threshold: ${threshold:,.0f}')
                
                median_w = np.median(all_withdrawals)
                ax2.axvline(x=median_w, color='yellow', linestyle='-', linewidth=2.5,
                           label=f'Median: ${median_w:,.0f}')
                
                ax2.set_xlabel('Annual Withdrawal (Real $)', fontsize=13, color='white', fontweight='bold')
                ax2.set_ylabel('Frequency', fontsize=13, color='white', fontweight='bold')
                ax2.set_title('Distribution of Annual Withdrawals', fontsize=15, color='white', fontweight='bold')
                ax2.tick_params(axis='both', colors='white', labelsize=11)
                ax2.legend(facecolor='black', edgecolor='white', labelcolor='white', framealpha=0.8, fontsize=10)
                ax2.grid(True, alpha=0.2, color='white', linewidth=0.8, axis='y')
                ax2.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
                
                st.pyplot(fig2)
        
    except Exception as e:
        st.error(f"Error creating amortization visualizations: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def create_plot_cumulative_retirement_probability(retirement_ages, config, median_age, num_outer):
    """Create plot for cumulative retirement probability - returns figure"""
    valid_retirement_ages = retirement_ages[~np.isnan(retirement_ages)]
    if valid_retirement_ages.size == 0:
        return None
    
    if HAS_PLOTLY:
        sorted_ages = np.sort(valid_retirement_ages)
        cumulative_prob = np.arange(1, len(sorted_ages) + 1) / num_outer * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sorted_ages,
            y=cumulative_prob,
            mode='lines+markers',
            name='Cumulative Probability',
            line=dict(color='#00ff00', width=3),
            marker=dict(size=6, color='#00ff00', line=dict(width=1, color='white')),
            hovertemplate='Age: %{x}<br>Probability: %{y:.2f}%<extra></extra>'
        ))
        
        # Reference lines
        if config.include_social_security:
            fig.add_vline(x=config.social_security_start_age, line_dash="dot", line_color="gray",
                         annotation_text=f"SS Age {config.social_security_start_age}", 
                         annotation_position="top")
        
        if not np.isnan(median_age):
            fig.add_vline(x=median_age, line_dash="dash", line_color="gray",
                         annotation_text=f"Median: {median_age:.1f}", 
                         annotation_position="top")
        
        fig.update_layout(
            title="Cumulative Probability of Retiring by Age",
            xaxis_title="Age",
            yaxis_title="Cumulative Probability (%)",
            template='plotly_white',
            height=500,
            showlegend=False,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent to match Streamlit background
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent to match Streamlit background
            yaxis=dict(range=[0, 100])
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        
        return fig
    else:
        # Fallback to matplotlib
        try:
            import matplotlib.pyplot as plt
            plt.style.use('dark_background')
            sorted_ages = np.sort(valid_retirement_ages)
            cumulative_prob = np.arange(1, len(sorted_ages) + 1) / num_outer * 100
            
            fig, ax = plt.subplots(figsize=(12, 7), facecolor='black')
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            ax.plot(sorted_ages, cumulative_prob, color='#00ff00', marker='o', markersize=5,
                    linestyle='-', alpha=0.9, linewidth=2.5,
                    markerfacecolor='#00ff00', markeredgecolor='white', markeredgewidth=1)
            
            if config.include_social_security:
                ax.axvline(x=config.social_security_start_age, color='gray',
                           linestyle=':', label=f'Social Security (Age {config.social_security_start_age})',
                           linewidth=2)
            
            if not np.isnan(median_age):
                ax.axvline(x=median_age, color='gray', linestyle='--',
                           label=f'Median Retirement Age ({median_age:.1f})', linewidth=2)
            
            ax.set_title("Cumulative Probability of Retiring by Age", fontsize=15, color='white', fontweight='bold')
            ax.set_xlabel("Age", fontsize=13, color='white', fontweight='bold')
            ax.set_ylabel("Cumulative Probability (%)", fontsize=13, color='white', fontweight='bold')
            ax.tick_params(axis='both', colors='white', labelsize=11)
            ax.set_ylim(0, 100)
            ax.grid(True, which='both', linestyle='--', linewidth=0.8, color='white', alpha=0.2)
            ax.legend(facecolor='black', edgecolor='white', labelcolor='white', framealpha=0.8, fontsize=11)
            fig.tight_layout()
            return fig
        except ImportError:
            return None

def main():
    """Main Streamlit app"""
    # Check if imports were successful
    if not IMPORTS_SUCCESSFUL:
        st.error(f"❌ Failed to import required modules from LIFECYCLE MODEL: {IMPORT_ERROR}")
        st.error("Please ensure the LIFECYCLE MODEL directory exists and contains all required modules.")
        return
    
    st.title("📊 Lifecycle Retirement Simulation")
    st.markdown("**Version 7.0** - Simulate retirement outcomes with uncertain labor income and market returns")
    
    # Display Cython status with diagnostics
    col1, col2 = st.columns(2)
    with col1:
        if CYTHON_AVAILABLE:
            st.success("✅ Cython acceleration enabled (10-50x faster!)")
        else:
            st.warning("⚠️ Running in pure Python mode (slower)")
            with st.expander("🔍 Cython Diagnostics", expanded=False):
                st.write("**Why Cython might not be available:**")
                st.write("1. Cython modules not compiled")
                st.write("2. Build directory not found")
                st.write("3. Python version mismatch")
                
                # Check build directory
                build_dir = os.path.join(parent_dir, 'build')
                if os.path.exists(build_dir):
                    st.write(f"✅ Build directory exists: `{build_dir}`")
                    build_files = [f for f in os.listdir(build_dir) if f.endswith('.pyd')]
                    if build_files:
                        st.write(f"✅ Found {len(build_files)} compiled module(s):")
                        for f in build_files[:5]:  # Show first 5
                            st.code(f)
                    else:
                        st.write("❌ No .pyd files found in build directory")
                else:
                    st.write(f"❌ Build directory not found: `{build_dir}`")
                
                st.write(f"**Python version:** {sys.version_info.major}.{sys.version_info.minor}")
                st.write(f"**Expected build path:** `build/lib.win-amd64-cpython-{sys.version_info.major}{sys.version_info.minor}/`")
    
    # Create configuration from sidebar
    config = create_config_from_sidebar()
    
    # Validate configuration
    try:
        config.validate()
        validation_ok = True
    except ValueError as e:
        st.sidebar.error(f"Configuration Error: {str(e)}")
        validation_ok = False
    
    # Removed estimated runtime - will show ETA during execution instead
    
    # Run Simulation Button
    if validation_ok:
        col1, col2 = st.columns([3, 1])
        with col1:
            run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
        with col2:
            if st.session_state.simulation_results is not None:
                if st.button("🔄 Clear Results", use_container_width=True):
                    st.session_state.simulation_results = None
                    st.session_state.config = None
                    st.rerun()
        
        if run_button:
            run_simulation(config)
    
    # Display results if available
    if st.session_state.simulation_results is not None:
        display_results(st.session_state.simulation_results, st.session_state.config)

def run_simulation(config):
    """Run the simulation with progress tracking"""
    # #region agent log
    try:
        import json
        import time
        import os
        log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'debug.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'id': 'log_run_sim_start', 'timestamp': time.time() * 1000, 'location': 'app.py:505', 'message': 'run_simulation started', 'data': {'config_validated': True}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'C'}) + '\n')
    except Exception as log_err: pass
    # #endregion
    st.session_state.config = config
    
    # Store results in session state
    results = {
        'required_principal_data': None,
        'principal_lookup': None,
        'retirement_ages': None,
        'ever_retired': None,
        'detailed_simulations': None,
        'final_bequest_nominal': None,
        'final_bequest_real': None,
        'consumption_streams': None,
        'amortization_stats_list': None,
        'utilities': None,
        'certainty_equivalents_annual': None,
        'utility_ex_ante': None,
        'ce_annual': None,
        'ce_values_dict': None,
        'total_utilities_dict': None
    }
    
    try:
        # Validate configuration first
        config.validate()
        params = config.params
        
        # Display configuration summary
        with st.expander("📋 Configuration Summary", expanded=False):
            st.json({
                "Basic": {
                    "Initial Age": config.initial_age,
                    "Death Age": config.death_age,
                    "Initial Portfolio": f"${config.initial_portfolio:,.0f}",
                    "Annual Income": f"${config.annual_income_real:,.0f}",
                    "Spending": f"${config.spending_real:,.0f}"
                },
                "Simulation": {
                    "Outer Simulations": config.num_outer,
                    "Nested Simulations": config.num_nested,
                    "Success Target": f"{config.success_target*100:.0f}%"
                },
                "Model": "Block Bootstrap" if config.use_block_bootstrap else "Parametric (Bates/Heston)"
            })
        
        # Load bootstrap data
        bootstrap_data = None
        if config.use_block_bootstrap:
            with st.status("Loading bootstrap data...", expanded=True) as status:
                try:
                    bootstrap_data = load_bootstrap_data(config)
                    if bootstrap_data is None:
                        st.warning("Failed to load bootstrap data, falling back to parametric model")
                        config.use_block_bootstrap = False
                    else:
                        status.update(label=f"✅ Bootstrap data loaded: {len(bootstrap_data[0])} monthly returns", state="complete")
                except Exception as e:
                    st.error(f"Error loading bootstrap data: {str(e)}")
                    config.use_block_bootstrap = False
        
        # Initialize RNG
        if config.seed is not None:
            rng = np.random.default_rng(seed=config.seed)
        else:
            rng = np.random.default_rng()
        
        # Calculate expected return for amortization
        if config.use_amortization:
            if config.amortization_expected_return is None:
                with st.status("Calculating expected return for amortization...", expanded=False) as status:
                    monthly_returns = bootstrap_data[0] if bootstrap_data is not None else None
                    monthly_inflation = bootstrap_data[1] if bootstrap_data is not None else None
                    config.amortization_expected_return = calculate_expected_return_from_data(
                        monthly_returns=monthly_returns,
                        monthly_inflation=monthly_inflation,
                        params=params if not config.use_block_bootstrap else None,
                        mean_inflation=config.mean_inflation_geometric
                    )
                    # Only show this message when expected return was auto-calculated (was None/blank)
                    status.update(label=f"✅ Expected return: {config.amortization_expected_return*100:.2f}%", state="complete")
        
        # Stage 1: Build required principal table
        st.header("Stage 1: Building Required Principal Lookup Table")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        target_ages = np.arange(config.retirement_age_min, config.retirement_age_max + 1)
        required_principal_table = {}
        mean_inflation_arithmetic = convert_geometric_to_arithmetic(
            config.mean_inflation_geometric, config.std_inflation)
        
        previous_principal = None
        import time as time_module
        stage1_start_time = time_module.time()
        stage1_times = []
        
        for idx, age in enumerate(target_ages):
            age_start_time = time_module.time()
            principal = find_required_principal(age, config.success_target,
                                                       config.num_nested, config, params,
                                                       warm_start_principal=previous_principal,
                                                       bootstrap_data=bootstrap_data)
            age_elapsed = time_module.time() - age_start_time
            stage1_times.append(age_elapsed)
            
            # Calculate ETA based on average time per age
            if len(stage1_times) > 0:
                avg_time_per_age = np.mean(stage1_times)
                remaining_ages = len(target_ages) - (idx + 1)
                eta_seconds = avg_time_per_age * remaining_ages
                eta_str = format_time(eta_seconds)
                status_text.text(f"Age {int(age)} ({idx + 1}/{len(target_ages)}) | ETA: {eta_str}")
            else:
                status_text.text(f"Calculating principal for age {int(age)}... ({idx + 1}/{len(target_ages)})")
            
            progress_bar.progress((idx + 1) / len(target_ages))
            
            if config.use_principal_deviation_threshold and previous_principal is not None:
                max_change = previous_principal * config.principal_deviation_threshold
                min_allowed = previous_principal - max_change
                max_allowed = previous_principal + max_change
                
                if principal < min_allowed:
                    principal = min_allowed
                elif principal > max_allowed:
                    principal = max_allowed
            
            required_principal_table[int(age)] = principal
            previous_principal = principal
        
        # Build required_principal_data
        required_principal_data = []
        for age, principal_real in required_principal_table.items():
            principal_nominal = calculate_nominal_value(
                principal_real, config.initial_age, age, mean_inflation_arithmetic)
            
            net_withdrawal_real = config.spending_real
            if config.include_social_security and age >= config.social_security_start_age:
                net_withdrawal_real = max(0.0, config.spending_real - config.social_security_real)
            
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
        
        results['required_principal_data'] = required_principal_data
        results['principal_lookup'] = {
            int(row['age']): {
                'principal_real': row['principal_real'],
                'principal_nominal': row['principal_nominal'],
                'swr': row['swr']
            } for row in required_principal_data
        }
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Principal lookup table complete for ages {config.retirement_age_min}-{config.retirement_age_max}")
        
        # Stage 2: Run accumulation simulations
        st.header("Stage 2: Running Accumulation Simulations")
        
        # Create progress bar and status for accumulation simulations
        accum_progress_bar = st.progress(0)
        accum_status_text = st.empty()
        
        # Track timing for ETA calculation
        import time as time_module
        stage2_start_time = time_module.time()
        stage2_times = []
        
        # Progress callback function with ETA
        def update_accumulation_progress(current, total):
            progress_pct = current / total
            accum_progress_bar.progress(progress_pct)
            
            # Calculate ETA
            if current > 1:
                elapsed = time_module.time() - stage2_start_time
                avg_time_per_sim = elapsed / current
                remaining = total - current
                eta_seconds = avg_time_per_sim * remaining
                eta_str = format_time(eta_seconds)
                accum_status_text.text(f"Simulation {current}/{total} ({progress_pct*100:.1f}%) | ETA: {eta_str}")
            else:
                accum_status_text.text(f"Running simulation {current}/{total} ({progress_pct*100:.1f}%)")
        
        # Run accumulation simulations with progress callback
        (retirement_ages, ever_retired, detailed_simulations,
         final_bequest_nominal, final_bequest_real, consumption_streams,
         amortization_stats_list) = run_accumulation_simulations(
            config, params, results['principal_lookup'], rng, bootstrap_data,
            progress_callback=update_accumulation_progress)
        
        # Clear progress indicators
        accum_progress_bar.empty()
        accum_status_text.empty()
        
        results['retirement_ages'] = retirement_ages
        results['ever_retired'] = ever_retired
        results['detailed_simulations'] = detailed_simulations
        results['final_bequest_nominal'] = final_bequest_nominal
        results['final_bequest_real'] = final_bequest_real
        results['consumption_streams'] = consumption_streams
        results['amortization_stats_list'] = amortization_stats_list
        
        # Clear progress indicators and show success
        accum_progress_bar.empty()
        accum_status_text.empty()
        st.success("✅ Accumulation simulations complete!")
        
        # Calculate utility metrics
        if config.enable_utility_calculations:
            with st.status("Calculating utility metrics...", expanded=False) as status:
                if consumption_streams and final_bequest_real:
                    portfolio_name = "Portfolio"
                    consumption_streams_dict = {portfolio_name: consumption_streams}
                    bequests_dict = {portfolio_name: final_bequest_real}
                    
                    results['ce_values_dict'] = calculate_ce_for_crra(
                        consumption_streams_dict, bequests_dict, config.gamma, config.beta,
                        config.k_bequest, config.theta, config.household_size
                    )
                    
                    results['total_utilities_dict'] = calculate_total_utility_ex_ante(
                        consumption_streams_dict, bequests_dict, config.gamma, config.beta,
                        config.k_bequest, config.theta, config.household_size
                    )
                    
                    ce_ex_ante_monthly = results['ce_values_dict'][portfolio_name][0] if portfolio_name in results['ce_values_dict'] else 0.0
                    total_utility_ex_ante = results['total_utilities_dict'][portfolio_name] if portfolio_name in results['total_utilities_dict'] else 0.0
                    
                    results['utility_ex_ante'] = total_utility_ex_ante
                    results['ce_annual'] = ce_ex_ante_monthly * 12.0
                    results['utilities'] = [total_utility_ex_ante]
                    results['certainty_equivalents_annual'] = [ce_ex_ante_monthly * 12.0]
                    
                    status.update(label="✅ Utility metrics calculated", state="complete")
        
        st.session_state.simulation_results = results
        st.success("🎉 Simulation complete! Scroll down to see results.")
        
    except ValueError as e:
        st.error(f"❌ Configuration Error: {str(e)}")
        logger.exception("Configuration validation error")
        st.session_state.simulation_results = None
        # #region agent log
        try:
            import json
            import traceback
            import time
            import os
            log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'log_config_error', 'timestamp': time.time() * 1000, 'location': 'app.py:729', 'message': 'Configuration validation error', 'data': {'error': str(e), 'traceback': traceback.format_exc()}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'C'}) + '\n')
        except Exception as log_err: pass
        # #endregion
    except FileNotFoundError as e:
        st.error(f"❌ File Not Found: {str(e)}\n\nPlease check that the bootstrap CSV file exists at the specified path.")
        logger.exception("File not found error")
        st.session_state.simulation_results = None
        # #region agent log
        try:
            import json
            import traceback
            import time
            import os
            log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'log_file_not_found', 'timestamp': time.time() * 1000, 'location': 'app.py:733', 'message': 'FileNotFoundError', 'data': {'error': str(e), 'traceback': traceback.format_exc()}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'B'}) + '\n')
        except Exception as log_err: pass
        # #endregion
    except Exception as e:
        st.error(f"❌ Error during simulation: {str(e)}\n\nPlease check the logs for more details.")
        logger.exception("Simulation error")
        st.session_state.simulation_results = None
        import traceback
        with st.expander("🔍 Detailed Error Information"):
            st.code(traceback.format_exc())
        # #region agent log
        try:
            import json
            import time
            import os
            log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'log_sim_error', 'timestamp': time.time() * 1000, 'location': 'app.py:737', 'message': 'Exception during simulation', 'data': {'error': str(e), 'traceback': traceback.format_exc(), 'error_type': type(e).__name__}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'D'}) + '\n')
        except Exception as log_err: pass
        # #endregion

def display_results(results, config):
    """Display simulation results"""
    st.header("📊 Simulation Results")
    
    # Principal Lookup Table
    st.subheader("Required Principal by Retirement Age")
    if results['required_principal_data']:
        df_principal = pd.DataFrame(results['required_principal_data'])
        df_display = df_principal.copy()
        
        # Format currency columns
        for col in ['principal_real', 'principal_nominal', 'spending_real',
                   'spending_nominal', 'net_withdrawal_real', 'net_withdrawal_nominal']:
            df_display[col] = df_display[col].apply(lambda x: f"${x:,.2f}")
        
        df_display['swr'] = df_display['swr'].apply(
            lambda x: f"{x:.2f}%" if not np.isnan(x) else "NaN")
        
        df_display.columns = ['Age', 'Principal (Real $)', 'Principal (Nominal $)',
                             'Spending (Real $)', 'Spending (Nominal $)',
                             'Net Withdrawal (Real $)', 'Net Withdrawal (Nominal $)', 'Withdrawal Rate']
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Download button for principal table
        csv_principal = df_principal.to_csv(index=False)
        st.download_button(
            label="📥 Download Principal Table (CSV)",
            data=csv_principal,
            file_name="required_principal_table.csv",
            mime="text/csv"
        )
    
    # Final Results Statistics
    st.subheader("Retirement Age Statistics")
    if results['retirement_ages'] is not None:
        retirement_ages = results['retirement_ages']
        valid_ages = retirement_ages[~np.isnan(retirement_ages)]
        num_retired = len(valid_ages)
        pct_ever_retired = 100.0 * num_retired / max(1, config.num_outer)
        
        if valid_ages.size > 0:
            median_age = np.median(valid_ages)
            p10 = np.percentile(valid_ages, 10)
            p25 = np.percentile(valid_ages, 25)
            p75 = np.percentile(valid_ages, 75)
            p90 = np.percentile(valid_ages, 90)
            
            def prob_retire_before(age_limit):
                return 100.0 * np.sum(valid_ages <= age_limit) / config.num_outer
            
            prob_before_50 = prob_retire_before(50)
            prob_before_55 = prob_retire_before(55)
            prob_before_60 = prob_retire_before(60)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Median Retirement Age", f"{median_age:.1f}")
            with col2:
                st.metric("Ever Retired", f"{pct_ever_retired:.2f}%")
            with col3:
                st.metric("10th Percentile", f"{p10:.1f}")
            with col4:
                st.metric("90th Percentile", f"{p90:.1f}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("25th Percentile", f"{p25:.1f}")
            with col2:
                st.metric("75th Percentile", f"{p75:.1f}")
            with col3:
                st.metric("Simulations Run", f"{config.num_outer:,}")
            
            st.write("**Retirement Probabilities:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Before Age 50", f"{prob_before_50:.2f}%")
            with col2:
                st.metric("Before Age 55", f"{prob_before_55:.2f}%")
            with col3:
                st.metric("Before Age 60", f"{prob_before_60:.2f}%")
        else:
            st.warning("No successful retirements in simulations")
    
    # Utility Metrics - Show Equivalent Savings Rate instead of Total Utility
    if config.enable_utility_calculations and results.get('utility_ex_ante') is not None:
        st.subheader("Utility Metrics")
        st.warning("⚠️ **Note:** Utility calculations are in development. This feature will be used to compare different portfolios. Current implementation shows metrics for a single portfolio.")
        
        # The equivalent savings rate is the savings rate used in the simulation
        # This is displayed to show what savings rate achieved this utility
        baseline_savings_rate = config.savings_rate  # Savings rate used in simulation
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Equivalent Savings Rate", f"{baseline_savings_rate*100:.2f}%")
        with col2:
            if results.get('ce_annual') is not None:
                st.metric("Certainty Equivalent (Annual, Real $)", f"${results['ce_annual']:,.2f}")
        
        st.info("Note: Equivalent Savings Rate shows the savings rate used in the simulation. Utility calculated using EX-ANTE approach.")
    
    # Amortization Statistics
    if config.use_amortization and results.get('amortization_stats_list'):
        st.subheader("Amortization-Based Withdrawal Statistics")
        try:
            amortization_stats_list = results['amortization_stats_list']
            final_bequest_nominal = results.get('final_bequest_nominal', [])
            final_bequest_real = results.get('final_bequest_real', [])
            
            if amortization_stats_list and any(stats is not None for stats in amortization_stats_list):
                # Collect all statistics
                all_withdrawals = []
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
                
                if all_withdrawals:
                    all_withdrawals = np.array(all_withdrawals)
                    threshold = config.amortization_min_spending_threshold * initial_spending
                    
                    # Calculate statistics
                    median_withdrawal = np.median(all_withdrawals)
                    mean_withdrawal = np.mean(all_withdrawals)
                    p10_withdrawal = np.percentile(all_withdrawals, 10)
                    p25_withdrawal = np.percentile(all_withdrawals, 25)
                    p75_withdrawal = np.percentile(all_withdrawals, 75)
                    p90_withdrawal = np.percentile(all_withdrawals, 90)
                    min_withdrawal = np.min(all_withdrawals)
                    max_withdrawal = np.max(all_withdrawals)
                    
                    pct_below_threshold = (total_below_threshold / total_years * 100) if total_years > 0 else 0
                    
                    # Final balance statistics
                    valid_balances_nominal = [b for b in final_bequest_nominal if not np.isnan(b) and b >= 0]
                    valid_balances_real = [b for b in final_bequest_real if not np.isnan(b) and b >= 0]
                    
                    median_balance_nominal = np.median(valid_balances_nominal) if valid_balances_nominal else 0
                    median_balance_real = np.median(valid_balances_real) if valid_balances_real else 0
                    mean_balance_nominal = np.mean(valid_balances_nominal) if valid_balances_nominal else 0
                    mean_balance_real = np.mean(valid_balances_real) if valid_balances_real else 0
                    
                    # Display metrics
                    st.write("**Withdrawal Statistics (Real Dollars):**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Annual Withdrawal", f"${mean_withdrawal:,.2f}")
                    with col2:
                        st.metric("Median Annual Withdrawal", f"${median_withdrawal:,.2f}")
                    with col3:
                        st.metric("Initial Spending Target", f"${initial_spending:,.2f}")
                    with col4:
                        st.metric("Min Threshold", f"${threshold:,.2f}")
                    
                    st.write("**Percentiles:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("10th Percentile", f"${p10_withdrawal:,.2f}")
                    with col2:
                        st.metric("25th Percentile", f"${p25_withdrawal:,.2f}")
                    with col3:
                        st.metric("75th Percentile", f"${p75_withdrawal:,.2f}")
                    with col4:
                        st.metric("90th Percentile", f"${p90_withdrawal:,.2f}")
                    
                    st.write("**Range:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Minimum Withdrawal", f"${min_withdrawal:,.2f}")
                    with col2:
                        st.metric("Maximum Withdrawal", f"${max_withdrawal:,.2f}")
                    
                    st.write("**Below-Threshold Statistics:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Percentage Below Threshold", f"{pct_below_threshold:.2f}%")
                    with col2:
                        st.metric("Years Below Threshold", f"{total_below_threshold:,}")
                    
                    st.write("**Final Balance Statistics:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Median Final Balance (Nominal)", f"${median_balance_nominal:,.2f}")
                    with col2:
                        st.metric("Median Final Balance (Real)", f"${median_balance_real:,.2f}")
                    with col3:
                        st.metric("Mean Final Balance (Nominal)", f"${mean_balance_nominal:,.2f}")
                    with col4:
                        st.metric("Mean Final Balance (Real)", f"${mean_balance_real:,.2f}")
                    
                    # (Removed confusing/incorrect expected real return note)
        except Exception as e:
            st.warning(f"Could not display amortization statistics: {str(e)}")
    
    # End-of-Simulation Wealth Distribution (always show, regardless of ABW)
    if results.get('final_bequest_nominal') and results.get('final_bequest_real'):
        st.subheader("End-of-Simulation Wealth Distribution")
        final_bequest_nominal = results.get('final_bequest_nominal', [])
        final_bequest_real = results.get('final_bequest_real', [])
        
        # Filter valid values
        valid_balances_nominal = [b for b in final_bequest_nominal if not np.isnan(b) and b >= 0]
        valid_balances_real = [b for b in final_bequest_real if not np.isnan(b) and b >= 0]
        
        if valid_balances_nominal and valid_balances_real:
            # Calculate statistics
            median_balance_nominal = np.median(valid_balances_nominal)
            median_balance_real = np.median(valid_balances_real)
            mean_balance_nominal = np.mean(valid_balances_nominal)
            mean_balance_real = np.mean(valid_balances_real)
            p10_nominal = np.percentile(valid_balances_nominal, 10)
            p25_nominal = np.percentile(valid_balances_nominal, 25)
            p75_nominal = np.percentile(valid_balances_nominal, 75)
            p90_nominal = np.percentile(valid_balances_nominal, 90)
            p10_real = np.percentile(valid_balances_real, 10)
            p25_real = np.percentile(valid_balances_real, 25)
            p75_real = np.percentile(valid_balances_real, 75)
            p90_real = np.percentile(valid_balances_real, 90)
            
            # Display metrics
            st.write("**Final Wealth Statistics (Nominal $):**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Median", f"${median_balance_nominal:,.2f}")
            with col2:
                st.metric("Mean", f"${mean_balance_nominal:,.2f}")
            with col3:
                st.metric("10th Percentile", f"${p10_nominal:,.2f}")
            with col4:
                st.metric("90th Percentile", f"${p90_nominal:,.2f}")
            
            st.write("**Final Wealth Statistics (Real $):**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Median", f"${median_balance_real:,.2f}")
            with col2:
                st.metric("Mean", f"${mean_balance_real:,.2f}")
            with col3:
                st.metric("10th Percentile", f"${p10_real:,.2f}")
            with col4:
                st.metric("90th Percentile", f"${p90_real:,.2f}")
            
            # Create wealth distribution plot
            if HAS_PLOTLY:
                fig_wealth = go.Figure()
                
                # Real wealth distribution
                fig_wealth.add_trace(go.Histogram(
                    x=valid_balances_real,
                    name='Final Wealth (Real $)',
                    nbinsx=50,
                    marker_color='#00ffff',
                    opacity=0.7,
                    hovertemplate='Wealth: $%{x:,.0f}<br>Count: %{y}<extra></extra>'
                ))
                
                # Add vertical lines for statistics
                fig_wealth.add_vline(x=median_balance_real, line_dash="dash", line_color="yellow",
                                    annotation_text=f"Median: ${median_balance_real:,.0f}", 
                                    annotation_position="top")
                fig_wealth.add_vline(x=mean_balance_real, line_dash="dot", line_color="lime",
                                    annotation_text=f"Mean: ${mean_balance_real:,.0f}", 
                                    annotation_position="top")
                
                fig_wealth.update_layout(
                    title='Distribution of Final Wealth at Death (Real $)',
                    xaxis_title='Final Wealth (Real $)',
                    yaxis_title='Frequency',
                    template='plotly_white',
                    height=500,
                    showlegend=False,
                    font=dict(size=12),
                    plot_bgcolor='rgba(0,0,0,0)',  # Transparent to match Streamlit background
                    paper_bgcolor='rgba(0,0,0,0)'  # Transparent to match Streamlit background
                )
                fig_wealth.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
                fig_wealth.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
                
                st.plotly_chart(fig_wealth, use_container_width=True)
            else:
                # Fallback to matplotlib
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(12, 6), facecolor='black')
                    fig.patch.set_facecolor('black')
                    ax.set_facecolor('black')
                    
                    ax.hist(valid_balances_real, bins=50, color='#00ffff', alpha=0.7, edgecolor='#00ffff', linewidth=1.5)
                    ax.axvline(median_balance_real, color='yellow', linestyle='--', linewidth=2.5, label=f'Median: ${median_balance_real:,.0f}')
                    ax.axvline(mean_balance_real, color='lime', linestyle=':', linewidth=2.5, label=f'Mean: ${mean_balance_real:,.0f}')
                    
                    ax.set_xlabel('Final Wealth (Real $)', fontsize=13, color='white', fontweight='bold')
                    ax.set_ylabel('Frequency', fontsize=13, color='white', fontweight='bold')
                    ax.set_title('Distribution of Final Wealth at Death (Real $)', fontsize=15, color='white', fontweight='bold')
                    ax.tick_params(axis='both', colors='white', labelsize=11)
                    ax.legend(facecolor='black', edgecolor='white', labelcolor='white', framealpha=0.8, fontsize=10)
                    ax.grid(True, alpha=0.2, color='white', linewidth=0.8, axis='y')
                    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
                    
                    st.pyplot(fig)
                except ImportError:
                    st.info("No plotting library available")
    
    # Visualizations
    st.subheader("Visualizations")
    if results['required_principal_data'] and results['retirement_ages'] is not None:
        # Calculate median age for plots
        retirement_ages = results['retirement_ages']
        valid_ages = retirement_ages[~np.isnan(retirement_ages)]
        median_age = np.median(valid_ages) if valid_ages.size > 0 else np.nan
        
        # Principal plots - add amortization tab if enabled
        if config.use_amortization and results.get('amortization_stats_list'):
            tab1, tab2, tab3, tab4 = st.tabs(["Principal Requirements", "Retirement Age Distribution", "Cumulative Probability", "Amortization Analysis"])
        else:
            tab1, tab2, tab3 = st.tabs(["Principal Requirements", "Retirement Age Distribution", "Cumulative Probability"])
        
        with tab1:
            st.write("**Principal Requirements by Retirement Age**")
            ages = [row['age'] for row in results['required_principal_data']]
            principals_nominal = [row['principal_nominal'] for row in results['required_principal_data']]
            principals_real = [row['principal_real'] for row in results['required_principal_data']]
            swr = [row['swr'] for row in results['required_principal_data']]
            
            col1, col2 = st.columns(2)
            with col1:
                fig_nominal = create_plot_required_principal_nominal(ages, principals_nominal, swr, config, median_age)
                if fig_nominal:
                    if HAS_PLOTLY and isinstance(fig_nominal, go.Figure):
                        st.plotly_chart(fig_nominal, use_container_width=True)
                    else:
                        st.pyplot(fig_nominal)
            with col2:
                fig_real = create_plot_required_principal_real(ages, principals_real, swr, config, median_age)
                if fig_real:
                    if HAS_PLOTLY and isinstance(fig_real, go.Figure):
                        st.plotly_chart(fig_real, use_container_width=True)
                    else:
                        st.pyplot(fig_real)
        
        with tab2:
            if valid_ages.size > 0:
                fig_dist = create_plot_retirement_age_distribution(valid_ages, median_age)
                if fig_dist:
                    if HAS_PLOTLY and isinstance(fig_dist, go.Figure):
                        st.plotly_chart(fig_dist, use_container_width=True)
                    else:
                        st.pyplot(fig_dist)
            else:
                st.warning("No retirement ages to plot")
        
        with tab3:
            if valid_ages.size > 0:
                fig_cum = create_plot_cumulative_retirement_probability(retirement_ages, config, median_age, config.num_outer)
                if fig_cum:
                    if HAS_PLOTLY and isinstance(fig_cum, go.Figure):
                        st.plotly_chart(fig_cum, use_container_width=True)
                    else:
                        st.pyplot(fig_cum)
            else:
                st.warning("No retirement ages to plot")
        
        # Amortization tab
        if config.use_amortization and results.get('amortization_stats_list'):
            with tab4:
                create_amortization_visualizations(results['amortization_stats_list'], config, 
                                                   results.get('final_bequest_nominal', []), 
                                                   results.get('final_bequest_real', []))
    
    # Download detailed simulations
    if results.get('detailed_simulations') and config.generate_csv_summary:
        st.subheader("Download Detailed Results")
        try:
            # Convert detailed_simulations to DataFrame and CSV
            if isinstance(results['detailed_simulations'], list) and len(results['detailed_simulations']) > 0:
                # Try to convert to DataFrame
                df_detailed = pd.DataFrame(results['detailed_simulations'][:config.num_sims_to_export])
                csv_detailed = df_detailed.to_csv(index=False)
                st.download_button(
                    label="📥 Download Detailed Simulation Data (CSV)",
                    data=csv_detailed,
                    file_name="detailed_lifecycle_paths.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.warning(f"Could not prepare detailed simulation data for download: {str(e)}")
    
    # Download utility metrics
    if config.enable_utility_calculations and results.get('utilities') and results.get('certainty_equivalents_annual'):
        st.subheader("Download Utility Metrics")
        utility_df = pd.DataFrame({
            'Utility': results['utilities'],
            'Certainty_Equivalent_Annual': results['certainty_equivalents_annual']
        })
        csv_utility = utility_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Utility Metrics (CSV)",
            data=csv_utility,
            file_name="utility_metrics.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()

