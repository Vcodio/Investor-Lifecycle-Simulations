"""
Streamlit App for Lifecycle Retirement Simulation
"""
import streamlit as st
import sys
import os



os.environ['TQDM_DISABLE'] = '1'

import numpy as np
import pandas as pd
import logging
from io import StringIO
import io
import contextlib


try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False


parent_dir = os.path.dirname(os.path.abspath(__file__))
lifecycle_model_dir = os.path.join(parent_dir, 'LIFECYCLE MODEL')



try:
    import importlib.util
    import importlib
    


    import types

    package_name = 'lifecycle_model'

    if package_name not in sys.modules:
        package_module = types.ModuleType(package_name)
        package_module.__path__ = [lifecycle_model_dir]
        sys.modules[package_name] = package_module

    _MODULE_REQUIRED_ATTRS = {
        'simulation': (
            'find_required_principal',
            'run_accumulation_simulations',
        ),
    }

    def load_module(module_name, file_path):
        """Load a module from a file path, setting up parent package for relative imports.
        Returns cached module if already loaded - critical for Streamlit which re-runs
        the entire script on every interaction. Without this, 8+ modules get re-executed
        on every click, causing the app to grind to a halt."""
        full_name = f"{package_name}.{module_name}"
        file_path = os.path.abspath(file_path)
        if full_name in sys.modules:
            cached = sys.modules[full_name]
            cached_file = getattr(cached, '__file__', None)
            required = _MODULE_REQUIRED_ATTRS.get(module_name, ())
            attrs_ok = not required or all(hasattr(cached, a) for a in required)
            if (cached_file
                    and os.path.abspath(cached_file) == file_path
                    and attrs_ok):
                return cached
            del sys.modules[full_name]
        spec = importlib.util.spec_from_file_location(full_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {module_name} from {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        

        module.__package__ = package_name
        module.__name__ = full_name
        

        spec.loader.exec_module(module)
        return module
    


    config_module = load_module('config', os.path.join(lifecycle_model_dir, 'config.py'))
    SimulationConfig = config_module.SimulationConfig
    


    build_dir = os.path.join(parent_dir, 'build')
    if os.path.exists(build_dir) and build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    
    cython_module = load_module('cython_wrapper', os.path.join(lifecycle_model_dir, 'cython_wrapper.py'))
    CYTHON_AVAILABLE = cython_module.CYTHON_AVAILABLE
    

    if CYTHON_AVAILABLE:
        try:

            has_cython_func = hasattr(cython_module, 'simulate_monthly_return_svj_cython') or \
                             'lrs_cython' in sys.modules
            if not has_cython_func:

                try:
                    import lrs_cython
                    CYTHON_AVAILABLE = True
                except ImportError:
                    CYTHON_AVAILABLE = False
        except Exception:
            CYTHON_AVAILABLE = False
    

    bootstrap_module = load_module('bootstrap', os.path.join(lifecycle_model_dir, 'bootstrap.py'))
    load_bootstrap_data = bootstrap_module.load_bootstrap_data
    

    earnings_module = load_module('earnings', os.path.join(lifecycle_model_dir, 'earnings.py'))
    

    utils_module = load_module('utils', os.path.join(lifecycle_model_dir, 'utils.py'))
    convert_geometric_to_arithmetic = utils_module.convert_geometric_to_arithmetic
    calculate_nominal_value = utils_module.calculate_nominal_value
    

    utility_module = load_module('utility', os.path.join(lifecycle_model_dir, 'utility.py'))
    calculate_total_utility_ex_ante = utility_module.calculate_total_utility_ex_ante
    calculate_ce_for_crra = utility_module.calculate_ce_for_crra
    

    savings_rate_module = load_module('savings_rate', os.path.join(lifecycle_model_dir, 'savings_rate.py'))
    calculate_equivalent_savings_rate_scaling = savings_rate_module.calculate_equivalent_savings_rate_scaling
    

    simulation_module = load_module('simulation', os.path.join(lifecycle_model_dir, 'simulation.py'))
    find_required_principal = simulation_module.find_required_principal
    run_accumulation_simulations = simulation_module.run_accumulation_simulations
    calculate_expected_return_from_data = simulation_module.calculate_expected_return_from_data
    ensure_mortality_loaded = simulation_module.ensure_mortality_loaded
    calculate_amortized_withdrawal = simulation_module.calculate_amortized_withdrawal
    
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)
    import traceback
    IMPORT_ERROR += f"\n\nTraceback:\n{traceback.format_exc()}"

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

except Exception as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = f"Error loading modules: {str(e)}"
    import traceback
    IMPORT_ERROR += f"\n\nTraceback:\n{traceback.format_exc()}"

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



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Lifecycle Retirement Simulation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'config' not in st.session_state:
    st.session_state.config = None

def create_config_from_sidebar():
    """Create SimulationConfig from sidebar inputs"""
    config = SimulationConfig()
    config.enable_utility_calculations = False
    config.use_amortization = False

    with st.expander("📋 Basic Parameters", expanded=True):
        config.initial_age = st.number_input("Initial Age", min_value=18, max_value=100, value=config.initial_age, step=1)
        config.use_stochastic_mortality = st.checkbox(
            "Use stochastic mortality (SSA)",
            value=config.use_stochastic_mortality,
            help="When on, death each period is random from the SSA table; Death Age is disabled and not used."
        )
        config.death_age = st.number_input(
            "Death Age",
            min_value=50,
            max_value=120,
            value=config.death_age,
            step=1,
            disabled=config.use_stochastic_mortality,
            help="Fixed age at which the simulation ends. Disabled when stochastic mortality is on."
        )
        config.initial_portfolio = st.number_input("Initial Portfolio ($)", min_value=0.0, value=float(config.initial_portfolio), step=10000.0, format="%.0f")
        config.annual_income_real = st.number_input("Annual Income (Real $)", min_value=0.0, value=float(config.annual_income_real), step=1000.0, format="%.0f")
        config.savings_rate = st.slider("Savings Rate", min_value=0.0, max_value=1.0, value=float(config.savings_rate), step=0.01, format="%.2f", help="Fraction of income saved during accumulation phase")
        config.spending_real = st.number_input("Target Annual Spending (Real $)", min_value=0.0, value=float(config.spending_real), step=1000.0, format="%.0f")
    

    with st.expander("💰 Social Security", expanded=False):
        config.include_social_security = st.checkbox("Include Social Security", value=config.include_social_security)
        if config.include_social_security:
            config.social_security_real = st.number_input("Social Security Benefit (Real $)", min_value=0.0, value=float(config.social_security_real), step=1000.0, format="%.0f")
            config.social_security_start_age = st.number_input("Social Security Start Age", min_value=62, max_value=70, value=config.social_security_start_age, step=1)
    

    with st.expander("⚙️ Simulation Settings", expanded=False):
        st.caption("Defaults are set for interactive runs. Increase simulation counts for final analysis.")
        try:
            _no = int(config.num_outer) if config.num_outer is not None else 10
        except (TypeError, ValueError):
            _no = 10
        default_outer = max(10, min(_no, 100000))
        config.num_outer = st.number_input("Number of Outer Simulations", min_value=10, max_value=100000, value=default_outer, step=10, help="Interactive: 1,000-2,000. Final analysis: 10,000+.", key="cfg_sim_num_outer")
        
        try:
            _nn = int(config.num_nested) if config.num_nested is not None else 50
        except (TypeError, ValueError):
            _nn = 50
        default_nested = max(50, min(_nn, 10000))
        config.num_nested = st.number_input("Number of Nested Simulations", min_value=50, max_value=10000, value=default_nested, step=50, help="Used repeatedly during the required-principal search. This is usually the biggest runtime driver.", key="cfg_sim_num_nested")
        try:
            _st = float(config.success_target) if config.success_target is not None else 0.95
        except (TypeError, ValueError):
            _st = 0.95
        _st = max(0.5, min(1.0, _st))
        config.success_target = st.slider("Success Target (%)", min_value=0.5, max_value=1.0, value=_st, step=0.01, format="%.2f", key="cfg_sim_success_target")
        seed_display = int(config.seed) if config.seed is not None else 0
        seed_input = st.number_input("Random Seed (0 = random)", min_value=0, value=seed_display, step=1, help="Use 0 for random seed", key="cfg_sim_seed")
        config.seed = None if seed_input == 0 else int(seed_input)
    

    with st.expander("🎯 Retirement Age Range", expanded=False):
        config.retirement_age_min = st.number_input("Min Retirement Age", min_value=18, max_value=100, value=config.retirement_age_min, step=1)
        config.retirement_age_max = st.number_input("Max Retirement Age", min_value=18, max_value=100, value=config.retirement_age_max, step=1)
    

    with st.expander("📈 Model Selection", expanded=False):
        config.use_block_bootstrap = st.checkbox("Use Block Bootstrap", value=config.use_block_bootstrap, help="Use historical data blocks instead of parametric model")
        
        if config.use_block_bootstrap:
            config.bootstrap_csv_path = st.text_input("Bootstrap CSV Path", value=st.session_state.get("_lifecycle_autofill_path", config.bootstrap_csv_path))
            config.portfolio_column_name = st.text_input("Portfolio Column Name", value=st.session_state.get("_lifecycle_autofill_portfolio_col", config.portfolio_column_name))
            config.inflation_column_name = st.text_input("Inflation Column Name", value=st.session_state.get("_lifecycle_autofill_inflation_col", config.inflation_column_name))
            st.session_state._lifecycle_autofill_path = config.bootstrap_csv_path
            st.session_state._lifecycle_autofill_portfolio_col = config.portfolio_column_name
            st.session_state._lifecycle_autofill_inflation_col = config.inflation_column_name
            config.block_length_years = st.number_input("Block Length (Years)", min_value=1, max_value=20, value=config.block_length_years, step=1)
            config.block_overlapping = st.checkbox("Overlapping Blocks", value=config.block_overlapping)
            use_mean_override = st.checkbox(
                "Override expected geometric mean (annual, real)",
                value=(getattr(config, 'bootstrap_geometric_mean_override', None) is not None),
                help="Shift bootstrapped returns so the expected real geometric mean matches your target. Volatility and correlation from history are preserved. Amortization uses this same real return when set to auto."
            )
            if use_mean_override:
                config.bootstrap_geometric_mean_override = st.number_input(
                    "Target geometric mean (annual, real)",
                    value=float(config.bootstrap_geometric_mean_override if getattr(config, 'bootstrap_geometric_mean_override', None) is not None else 0.04),
                    min_value=-0.1, max_value=0.2, step=0.005, format="%.3f",
                    help="e.g. 0.04 = 4% annual real. Simulation converts to nominal internally using inflation."
                )
            else:
                config.bootstrap_geometric_mean_override = None
        else:

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
    

    with st.expander("🔧 Advanced", expanded=False):
        with st.expander("💼 Earnings Dynamics (GKOS)", expanded=False):
            st.caption("Table IV Model (8) — Guvenen, Karahan, Ozkan, Song (2019). typical_worker=True sets α=β=z₀=0.")
            p = config.gkos_params
            def _g(k, d): return float(p.get(k, d))
            def _gb(k, d): return bool(p.get(k, d))

            config.gkos_params['typical_worker'] = st.checkbox(
                "Typical worker (α=β=z₀=0)",
                value=_gb('typical_worker', True),
                help="Base case from Anarkulova, Cederburg, O'Doherty (2023). Disabling adds HIP heterogeneity."
            )
            config.gkos_params['additional_log_intercept'] = st.number_input(
                "Log intercept (level scale)", value=_g('additional_log_intercept', 7.427),
                step=0.01, format="%.4f",
                help="Additive shift to log earnings. 7.427 calibrates P90 at age 47 ≈ $160k (2010 USD)."
            )

            st.write("*Persistent shock η — mixture of two normals*")
            col1, col2 = st.columns(2)
            with col1:
                config.gkos_params['rho'] = st.number_input("ρ (AR persistence)", value=_g('rho', 0.958), step=0.001, format="%.4f")
                config.gkos_params['p_z'] = st.number_input("p_z (unfavorable prob)", value=_g('p_z', 0.219), step=0.01, format="%.4f")
                config.gkos_params['mu_eta1'] = st.number_input("μ_η1 (unfavorable mean)", value=_g('mu_eta1', -0.147), step=0.01, format="%.4f")
                config.gkos_params['sigma_eta1'] = st.number_input("σ_η1 (unfavorable std)", value=_g('sigma_eta1', 0.457), step=0.01, format="%.4f")
            with col2:
                config.gkos_params['sigma_eta2'] = st.number_input("σ_η2 (favorable std)", value=_g('sigma_eta2', 0.139), step=0.01, format="%.4f")
                config.gkos_params['sigma_z0'] = st.number_input("σ_z0 (initial z std)", value=_g('sigma_z0', 0.667), step=0.01, format="%.4f")

            st.write("*Transitory shock ε — mixture of two normals*")
            col1, col2 = st.columns(2)
            with col1:
                config.gkos_params['p_eps'] = st.number_input("p_ε (unfavorable prob)", value=_g('p_eps', 0.126), step=0.01, format="%.4f")
                config.gkos_params['mu_eps1'] = st.number_input("μ_ε1 (unfavorable mean)", value=_g('mu_eps1', 0.236), step=0.01, format="%.4f")
                config.gkos_params['sigma_eps1'] = st.number_input("σ_ε1 (unfavorable std)", value=_g('sigma_eps1', 0.343), step=0.01, format="%.4f")
            with col2:
                config.gkos_params['sigma_eps2'] = st.number_input("σ_ε2 (favorable std)", value=_g('sigma_eps2', 0.063), step=0.01, format="%.4f")

            st.write("*HIP heterogeneity (α, β) — active when typical_worker=False*")
            col1, col2 = st.columns(2)
            with col1:
                config.gkos_params['sigma_alpha'] = st.number_input("σ_α", value=_g('sigma_alpha', 0.298), step=0.01, format="%.4f")
                config.gkos_params['sigma_beta'] = st.number_input("σ_β", value=_g('sigma_beta', 0.185), step=0.01, format="%.4f")
            with col2:
                config.gkos_params['corr_alpha_beta'] = st.number_input("corr(α,β)", value=_g('corr_alpha_beta', 0.976), step=0.01, format="%.4f")

            st.write("*Age earnings profile g(t) — cubic polynomial in (t−24)/10*")
            col1, col2, col3 = st.columns(3)
            with col1:
                config.gkos_params['g_a0'] = st.number_input("g_a0", value=_g('g_a0', 2.6291), step=0.01, format="%.4f")
            with col2:
                config.gkos_params['g_a1'] = st.number_input("g_a1", value=_g('g_a1', 0.7300), step=0.01, format="%.4f")
            with col3:
                config.gkos_params['g_a2'] = st.number_input("g_a2", value=_g('g_a2', -0.1692), step=0.01, format="%.4f")

            st.write("*Nonemployment probability logit: ξ = a + b·t̃ + c·z + d·z·t̃*")
            col1, col2 = st.columns(2)
            with col1:
                config.gkos_params['a_nu'] = st.number_input("a_ν", value=_g('a_nu', -3.2131), step=0.1, format="%.4f")
                config.gkos_params['b_nu'] = st.number_input("b_ν", value=_g('b_nu', -1.0235), step=0.1, format="%.4f")
            with col2:
                config.gkos_params['c_nu'] = st.number_input("c_ν", value=_g('c_nu', -3.2602), step=0.1, format="%.4f")
                config.gkos_params['d_nu'] = st.number_input("d_ν", value=_g('d_nu', -2.1656), step=0.1, format="%.4f")
            config.gkos_params['lambda_nu'] = st.number_input(
                "λ_ν (exp scale for income loss)", value=_g('lambda_nu', 0.001),
                step=0.0001, format="%.4f",
                help="Scale parameter of the exponential income loss during nonemployment."
            )

        with st.expander("⚙️ Advanced Settings", expanded=False):
            if config.use_stochastic_mortality:
                config.mortality_table_path = st.text_input(
                    "Mortality table path (SSA)",
                    value=config.mortality_table_path,
                    help="CSV with columns age, qx_male, qx_female (or age, qx)."
                )
                config.mortality_sex = st.selectbox(
                    "Mortality table sex",
                    options=["male", "female", "average"],
                    index=["male", "female", "average"].index(config.mortality_sex) if config.mortality_sex in ("male", "female", "average") else 2,
                    help="Which death probabilities to use when table has male/female columns."
                )
            config.use_principal_deviation_threshold = st.checkbox("Use Principal Deviation Threshold", value=config.use_principal_deviation_threshold)
            if config.use_principal_deviation_threshold:
                config.principal_deviation_threshold = st.slider("Principal Deviation Threshold", min_value=0.01, max_value=0.2, value=config.principal_deviation_threshold, step=0.01, format="%.3f")
            config.num_workers = st.number_input("Number of Workers (multiprocessing)", min_value=1, max_value=os.cpu_count() or 1, value=config.num_workers, step=1)

    with st.expander("🚧 Unfinished", expanded=False):
        st.caption("Work-in-progress features. All are off unless you enable them here.")
        with st.expander("📊 Utility", expanded=False):
            st.info("⚠️ **Note:** Utility calculations are in development.")
            config.enable_utility_calculations = st.checkbox(
                "Enable Utility Calculations",
                value=False,
                key="cfg_enable_utility",
            )
            if config.enable_utility_calculations:
                config.gamma = st.number_input("gamma (CRRA risk aversion)", min_value=0.1, max_value=10.0, value=float(config.gamma), step=0.1, format="%.2f")
                config.beta = st.number_input("beta (time discount)", min_value=0.8, max_value=1.0, value=float(config.beta), step=0.01, format="%.3f")
                config.k_bequest = st.number_input("k_bequest (bequest threshold)", min_value=0.0, value=float(config.k_bequest), step=1000.0, format="%.0f")
                config.theta = st.number_input("theta (bequest weight)", min_value=0.0, max_value=1.0, value=float(config.theta), step=0.1, format="%.2f")
                config.household_size = st.number_input("household_size", min_value=0.1, max_value=10.0, value=float(config.household_size), step=0.1, format="%.1f")

        with st.expander("💸 Amortization-Based Withdrawals", expanded=False):
            config.use_amortization = st.checkbox(
                "Use Amortization-Based Withdrawal",
                value=False,
                key="cfg_use_amortization",
            )
            if config.use_amortization:
                _ar = config.amortization_expected_return
                amort_display = float(_ar) if _ar is not None else 0.0
                amort_return_input = st.number_input(
                    "Amortization expected real return (annual). 0 = auto",
                    value=amort_display, step=0.001, format="%.4f",
                    help="Expected real (inflation-adjusted) return for amortization. 0 = auto-calculated from data.",
                    key="cfg_amort_return"
                )
                config.amortization_expected_return = None if (amort_return_input == 0.0 or amort_return_input is None) else float(amort_return_input)
                config.amortization_min_spending_threshold = st.slider("Min Spending Threshold (fraction of initial real spending)", min_value=0.0, max_value=1.0, value=config.amortization_min_spending_threshold, step=0.05, format="%.2f")
                config.amortization_desired_bequest = st.number_input("Desired Bequest (Real $)", min_value=0.0, value=float(config.amortization_desired_bequest), step=1000.0, format="%.0f", help="Target portfolio value at death (real dollars). If 0, portfolio is consumed to zero.")
                config.debug_amortization = st.checkbox("🔍 Debug Amortization (writes to .cursor/amortization_debug.json)", value=config.debug_amortization, help="Enable detailed debugging output for amortization calculations")

    with st.expander("💾 Output Settings", expanded=False):
        config.generate_csv_summary = st.checkbox("Generate CSV Summary", value=config.generate_csv_summary)
        if config.generate_csv_summary:
            config.num_sims_to_export = st.number_input("Number of Sims to Export", min_value=1, max_value=1000, value=config.num_sims_to_export, step=1)

    return config

def estimate_runtime(config):
    """Estimate simulation runtime in seconds"""



    num_ages = config.retirement_age_max - config.retirement_age_min + 1
    time_per_sim_stage1 = 0.5 if CYTHON_AVAILABLE else 2.0
    binary_search_iterations = 20
    stage1_time = num_ages * binary_search_iterations * config.num_nested * time_per_sim_stage1
    


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
    """Create plot for required principal in nominal terms returns figure"""
    if HAS_PLOTLY:
        from plotly.subplots import make_subplots
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        

        fig.add_trace(
            go.Scatter(x=ages, y=principals_nominal, mode='lines+markers',
                      name='Required Principal (Nominal $)', 
                      line=dict(color='#00ffff', width=3),
                      marker=dict(size=8, color='#00ffff', line=dict(width=1, color='white'))),
            secondary_y=False
        )
        

        fig.add_trace(
            go.Scatter(x=ages, y=np.array(swr), mode='lines+markers',
                      name='Withdrawal Rate (%)',
                      line=dict(color='#ff00ff', width=3, dash='dash'),
                      marker=dict(size=8, symbol='x', color='#ff00ff')),
            secondary_y=True
        )
        

        if config.include_social_security:
            fig.add_vline(x=config.social_security_start_age, line_dash="dot", line_color="lime",
                         annotation_text=f"SS Age {config.social_security_start_age}", 
                         annotation_position="top")
        
        if not np.isnan(median_age):
            fig.add_vline(x=median_age, line_dash="dash", line_color="yellow",
                         annotation_text=f"Median {median_age:.1f}", 
                         annotation_position="top")
        
        success_pct = int(config.success_target * 100)
        fig.update_layout(
            title=f"Required Principal & Withdrawal Rate for {success_pct}% Success (Nominal)",
            template='plotly_white',
            height=600,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=0,
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)',
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
    """Create plot for required principal in real terms returns figure"""
    if HAS_PLOTLY:
        from plotly.subplots import make_subplots
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        

        fig.add_trace(
            go.Scatter(x=ages, y=principals_real, mode='lines+markers',
                      name='Required Principal (Real $)', 
                      line=dict(color='#ff8800', width=3),
                      marker=dict(size=8, color='#ff8800', line=dict(width=1, color='white'))),
            secondary_y=False
        )
        

        fig.add_trace(
            go.Scatter(x=ages, y=np.array(swr), mode='lines+markers',
                      name='Withdrawal Rate (%)',
                      line=dict(color='#ff00ff', width=3, dash='dash'),
                      marker=dict(size=8, symbol='x', color='#ff00ff')),
            secondary_y=True
        )
        

        if config.include_social_security:
            fig.add_vline(x=config.social_security_start_age, line_dash="dot", line_color="cyan",
                         annotation_text=f"SS Age {config.social_security_start_age}", 
                         annotation_position="top")
        
        if not np.isnan(median_age):
            fig.add_vline(x=median_age, line_dash="dash", line_color="yellow",
                         annotation_text=f"Median {median_age:.1f}", 
                         annotation_position="top")
        
        success_pct = int(config.success_target * 100)
        fig.update_layout(
            title=f"Required Principal & Withdrawal Rate for {success_pct}% Success (Real)",
            template='plotly_white',
            height=600,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=0,
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)',
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
    """Create plot for retirement age distribution returns figure"""
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
        

        fig.add_vline(x=median_age, line_dash="dash", line_color="red", line_width=3,
                     annotation_text=f"Median {median_age:.1f}", annotation_position="top")
        
        fig.update_layout(
            title='Distribution of Retirement Ages',
            xaxis_title='Retirement Age',
            yaxis_title='Frequency',
            template='plotly_white',
            height=500,
            showlegend=False,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        
        return fig
    else:

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
        

        effective_max_age = 120 if getattr(config, 'use_stochastic_mortality', False) else config.death_age
        max_reasonable_years = effective_max_age - config.retirement_age_min
        total_sims = len([s for s in amortization_stats_list if s and s.get('withdrawals')])
        min_obs_frac = 0.30
        all_years = sorted(all_withdrawals_by_year.keys())
        years = []
        for y in all_years:
            if y > max_reasonable_years:
                break
            n_obs = len(all_withdrawals_by_year[y])
            if n_obs >= min_obs_frac * total_sims:
                years.append(y)
            else:
                break
        if not years:
            st.warning("No valid withdrawal data available")
            return
        
        medians = [np.median(all_withdrawals_by_year[y]) for y in years]
        p25 = [np.percentile(all_withdrawals_by_year[y], 25) for y in years]
        p75 = [np.percentile(all_withdrawals_by_year[y], 75) for y in years]
        p10 = [np.percentile(all_withdrawals_by_year[y], 10) for y in years]
        p90 = [np.percentile(all_withdrawals_by_year[y], 90) for y in years]
        

        if HAS_PLOTLY:
            fig1 = go.Figure()
            

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
            

            fig1.add_trace(go.Scatter(
                x=years,
                y=medians,
                mode='lines+markers',
                name='Median',
                line=dict(color='yellow', width=3),
                marker=dict(size=8, color='yellow'),
                hovertemplate='Year: %{x}<br>Median: $%{y:,.0f}<extra></extra>'
            ))
            

            if initial_spending:
                threshold = config.amortization_min_spending_threshold * initial_spending
                fig1.add_hline(y=initial_spending, line_dash="dash", line_color="red", line_width=2,
                              annotation_text=f"Initial ${initial_spending:,.0f}", annotation_position="right")
                fig1.add_hline(y=threshold, line_dash="dot", line_color="orange", line_width=2,
                              annotation_text=f"Threshold ${threshold:,.0f}", annotation_position="right")
            
            fig1.update_layout(
                title='Amortization-Based Withdrawals Over Time',
                xaxis_title='Years in Retirement',
                yaxis_title='Annual Withdrawal (Real $)',
                template='plotly_white',
                height=600,
                font=dict(size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=0,
                    bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(0,0,0,0)',
                    borderwidth=0
                )
            )
            fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
            fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
                            tickformat='$,.0f')
            
            st.plotly_chart(fig1, width='stretch')
        else:

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
        

        all_withdrawals = []
        for stats in amortization_stats_list:
            if stats and stats.get('withdrawals'):
                all_withdrawals.extend(stats['withdrawals'])
        
        if all_withdrawals:
            all_withdrawals = np.array(all_withdrawals)
            

            p1 = np.percentile(all_withdrawals, 1)
            p95 = np.percentile(all_withdrawals, 95)
            median_w = np.median(all_withdrawals)
            mean_w = np.mean(all_withdrawals)
            


            x_range = [max(0, p1 * 0.95), p95 * 1.05]
            
            if HAS_PLOTLY:
                fig2 = go.Figure()
                

                fig2.add_trace(go.Histogram(
                    x=all_withdrawals,
                    name='Withdrawals',
                    nbinsx=min(75, max(50, len(np.unique(all_withdrawals))//10)),
                    marker_color='#00ffff',
                    opacity=0.7,
                    hovertemplate='Withdrawal: $%{x:,.0f}<br>Count: %{y}<extra></extra>'
                ))
                

                if initial_spending:
                    threshold = config.amortization_min_spending_threshold * initial_spending
                    if x_range[0] <= initial_spending <= x_range[1]:
                        fig2.add_vline(x=initial_spending, line_dash="dash", line_color="red", line_width=2.5,
                                      annotation_text=f"Initial ${initial_spending:,.0f}", annotation_position="top")
                    if x_range[0] <= threshold <= x_range[1]:
                        fig2.add_vline(x=threshold, line_dash="dot", line_color="orange", line_width=2.5,
                                      annotation_text=f"Threshold ${threshold:,.0f}", annotation_position="top")
                

                if x_range[0] <= median_w <= x_range[1]:
                    fig2.add_vline(x=median_w, line_dash="solid", line_color="yellow", line_width=2.5,
                                  annotation_text=f"Median ${median_w:,.0f}", annotation_position="top")
                if x_range[0] <= mean_w <= x_range[1]:
                    fig2.add_vline(x=mean_w, line_dash="solid", line_color="lime", line_width=2.5,
                                  annotation_text=f"Mean ${mean_w:,.0f}", annotation_position="top")
                

                data_min = np.min(all_withdrawals)
                data_max = np.max(all_withdrawals)
                range_note = ""
                if data_min < x_range[0] or data_max > x_range[1]:
                    range_note = f" (Showing 1st-95th percentile, range: ${x_range[0]:,.0f} - ${x_range[1]:,.0f})"
                
                fig2.update_layout(
                    title=f'Distribution of Annual Withdrawals{range_note}',
                    xaxis_title='Annual Withdrawal (Real $)',
                    yaxis_title='Frequency',
                    template='plotly_white',
                    height=600,
                    showlegend=False,
                    font=dict(size=12),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )

                fig2.update_xaxes(
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='rgba(0,0,0,0.1)', 
                    tickformat='$,.0f',
                    range=x_range
                )

                fig2.update_yaxes(
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='rgba(0,0,0,0.1)',
                    tickformat='.2s'
                )
                
                st.plotly_chart(fig2, width='stretch')
                

                fig_box = go.Figure()
                
                fig_box.add_trace(go.Box(
                    y=all_withdrawals,
                    name='Withdrawals',
                    boxmean='sd',
                    marker_color='#00ffff',
                    line=dict(color='#00ffff', width=2),
                    fillcolor='rgba(0, 255, 255, 0.1)',
                    hovertemplate='Withdrawal: $%{y:,.0f}<extra></extra>'
                ))
                

                if initial_spending:
                    threshold = config.amortization_min_spending_threshold * initial_spending
                    fig_box.add_hline(y=initial_spending, line_dash="dash", line_color="red", line_width=2.5,
                                     annotation_text=f"Initial ${initial_spending:,.0f}", annotation_position="right")
                    fig_box.add_hline(y=threshold, line_dash="dot", line_color="orange", line_width=2.5,
                                     annotation_text=f"Threshold ${threshold:,.0f}", annotation_position="right")
                

                fig_box.add_hline(y=median_w, line_dash="solid", line_color="yellow", line_width=2.5,
                                 annotation_text=f"Median ${median_w:,.0f}", annotation_position="right")
                

                q1 = np.percentile(all_withdrawals, 25)
                q3 = np.percentile(all_withdrawals, 75)
                iqr = q3 - q1
                y_max = min(q3 + 2.5 * iqr, p95 * 1.1)
                y_min = max(0, q1 - 2.5 * iqr)
                
                fig_box.update_layout(
                    title='Withdrawal Distribution (Box Plot)',
                    yaxis_title='Annual Withdrawal (Real $)',
                    template='plotly_white',
                    height=1200,
                    showlegend=False,
                    font=dict(size=12),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                fig_box.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(0,0,0,0.1)',
                    tickformat='$,.0f',
                    range=[y_min, y_max]
                )
                fig_box.update_xaxes(showticklabels=False)
                
                st.plotly_chart(fig_box, width='stretch')
            else:

                import matplotlib.pyplot as plt
                fig2, ax2 = plt.subplots(figsize=(12, 7), facecolor='black')
                fig2.patch.set_facecolor('black')
                ax2.set_facecolor('black')
                
                n_bins = min(50, len(np.unique(all_withdrawals)))
                counts, bins, patches = ax2.hist(all_withdrawals, bins=n_bins, color='#00ffff', 
                                                edgecolor='#00ffff', linewidth=1.5, alpha=0.7)
                

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

def create_plot_log_earnings_dynamics(earnings_real_list, config):
    """Create plot for earnings dynamics across simulations (regular scale, not log)"""
    if not earnings_real_list or all(e is None for e in earnings_real_list):
        return None
    

    valid_earnings = [e for e in earnings_real_list if e is not None and len(e) > 0]
    if not valid_earnings:
        return None
    max_paths = min(len(valid_earnings), 5000)
    valid_earnings = valid_earnings[:max_paths]
    

    try:
        import time as _time
        import json as _json
        _lp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cursor', 'debug.log')
        _lens = [len(e) for e in valid_earnings[:5]]
        _d = {'n_paths': len(valid_earnings), 'lengths_first5': _lens, 'initial_age': config.initial_age, 'hypothesisId': 'H3'}
        with open(_lp, 'a', encoding='utf-8') as _f:
            _f.write(_json.dumps({'id': 'earnings_dynamics_input', 'timestamp': _time.time()*1000, 'location': 'app.py:create_plot_log_earnings_dynamics', 'message': 'earnings_dynamics input', 'data': _d, 'runId': 'debug'}) + '\n')
    except Exception:
        pass



    max_length = max(len(e) for e in valid_earnings)
    ages = np.arange(config.initial_age, config.initial_age + max_length)
    earnings_matrix = np.full((len(valid_earnings), max_length), np.nan)
    for i, earnings_path in enumerate(valid_earnings):
        path = np.asarray(earnings_path, dtype=float)
        earnings_matrix[i, :len(path)] = path
    
    if HAS_PLOTLY:
        fig = go.Figure()
        

        medians = np.nanpercentile(earnings_matrix, 50, axis=0)
        p25 = np.nanpercentile(earnings_matrix, 25, axis=0)
        p75 = np.nanpercentile(earnings_matrix, 75, axis=0)
        p10 = np.nanpercentile(earnings_matrix, 10, axis=0)
        p90 = np.nanpercentile(earnings_matrix, 90, axis=0)
        

        valid_indices = ~np.isnan(medians)
        ages_plot = ages[valid_indices]
        medians_plot = medians[valid_indices]
        p25_plot = p25[valid_indices]
        p75_plot = p75[valid_indices]
        p10_plot = p10[valid_indices]
        p90_plot = p90[valid_indices]
        

        try:
            import time as _time
            import json as _json
            _lp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cursor', 'debug.log')
            _idx0 = 0
            _idx10 = min(10, len(medians)-1) if len(medians) else 0
            _n0 = sum(1 for ep in valid_earnings if 0 < len(ep))
            _n10 = sum(1 for ep in valid_earnings if _idx10 < len(ep))
            _d = {'max_length': max_length, 'n_ages_plot': len(ages_plot), 'at_age0_n': _n0, 'median0': float(medians[_idx0]) if _idx0 < len(medians) else None, 'p10_0': float(p10[_idx0]) if _idx0 < len(p10) else None, 'p90_0': float(p90[_idx0]) if _idx0 < len(p90) else None, 'at_age10_n': _n10, 'median10': float(medians[_idx10]) if _idx10 < len(medians) else None, 'any_nan': bool(np.any(np.isnan(medians))), 'hypothesisId': 'H3_H5'}
            with open(_lp, 'a', encoding='utf-8') as _f:
                _f.write(_json.dumps({'id': 'earnings_dynamics_percentiles', 'timestamp': _time.time()*1000, 'location': 'app.py:earnings_dynamics', 'message': 'percentiles computed', 'data': _d, 'runId': 'debug'}) + '\n')
        except Exception:
            pass

        

        fig.add_trace(go.Scatter(
            x=list(ages_plot) + list(ages_plot[::-1]),
            y=list(p90_plot) + list(p10_plot[::-1]),
            fill='toself',
            fillcolor='rgba(0, 255, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='10th-90th Percentile',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=list(ages_plot) + list(ages_plot[::-1]),
            y=list(p75_plot) + list(p25_plot[::-1]),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            name='25th-75th Percentile',
            showlegend=True,
            hoverinfo='skip'
        ))
        

        fig.add_trace(go.Scatter(
            x=ages_plot,
            y=medians_plot,
            mode='lines+markers',
            name='Median',
            line=dict(color='yellow', width=3),
            marker=dict(size=6, color='yellow'),
            hovertemplate='Age: %{x}<br>Earnings: $%{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Earnings Dynamics Over Lifecycle',
            xaxis_title='Age',
            yaxis_title='Earnings (Real $)',
            template='plotly_dark',
            height=600,
            font=dict(size=12),
            plot_bgcolor='rgba(20,20,20,1)',
            paper_bgcolor='rgba(20,20,20,1)',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=0,
                bgcolor='rgba(30,30,30,0.9)',
                bordercolor='rgba(100,100,100,0.5)',
                borderwidth=1
            )
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(100,100,100,0.3)', tickfont=dict(color='white'))
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(100,100,100,0.3)', tickformat='$,.0f', tickfont=dict(color='white'))
        
        return fig
    else:

        try:
            import matplotlib.pyplot as plt
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 7), facecolor='black')
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            

            medians = np.nanpercentile(earnings_matrix, 50, axis=0)
            p25 = np.nanpercentile(earnings_matrix, 25, axis=0)
            p75 = np.nanpercentile(earnings_matrix, 75, axis=0)
            p10 = np.nanpercentile(earnings_matrix, 10, axis=0)
            p90 = np.nanpercentile(earnings_matrix, 90, axis=0)
            
            valid_indices = ~np.isnan(medians)
            ages_plot = ages[valid_indices]
            medians_plot = medians[valid_indices]
            p25_plot = p25[valid_indices]
            p75_plot = p75[valid_indices]
            p10_plot = p10[valid_indices]
            p90_plot = p90[valid_indices]
            
            ax.fill_between(ages_plot, p10_plot, p90_plot, alpha=0.2, color='cyan', label='10th-90th Percentile')
            ax.fill_between(ages_plot, p25_plot, p75_plot, alpha=0.3, color='blue', label='25th-75th Percentile')
            ax.plot(ages_plot, medians_plot, 'o-', color='yellow', linewidth=2.5, markersize=6, label='Median', alpha=0.9)
            
            ax.set_xlabel('Age', fontsize=13, color='white', fontweight='bold')
            ax.set_ylabel('Earnings (Real $)', fontsize=13, color='white', fontweight='bold')
            ax.set_title('Earnings Dynamics Over Lifecycle', fontsize=15, color='white', fontweight='bold')
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
            ax.tick_params(axis='both', colors='white', labelsize=11)
            ax.legend(facecolor='black', edgecolor='white', labelcolor='white', framealpha=0.9, fontsize=10)
            ax.grid(True, alpha=0.2, color='white', linewidth=0.8)
            
            return fig
        except ImportError:
            return None

def create_earnings_moment_summary(earnings_real_list, config):
    """Summarize lifecycle earnings distribution by age bucket (GKOS-style diagnostics)."""
    if not earnings_real_list:
        return pd.DataFrame()

    valid_earnings = [
        np.asarray(e, dtype=float)
        for e in earnings_real_list
        if e is not None and len(e) > 1
    ]
    if not valid_earnings:
        return pd.DataFrame()
    valid_earnings = valid_earnings[:5000]
    max_length = max(len(e) for e in valid_earnings)
    earnings_matrix = np.full((len(valid_earnings), max_length), np.nan)
    for i, ep in enumerate(valid_earnings):
        earnings_matrix[i, :len(ep)] = ep
    ages = np.arange(config.initial_age, config.initial_age + max_length)

    buckets = [("25-34", 25, 34), ("35-44", 35, 44), ("45-54", 45, 54), ("55-64", 55, 64)]
    rows = []
    for label, lo, hi in buckets:
        mask = (ages >= lo) & (ages <= hi)
        if not np.any(mask):
            continue
        block = earnings_matrix[:, mask]  # (sims, years_in_bucket)

        all_vals = block.ravel()
        finite_mask = np.isfinite(all_vals)
        zero_frac = float(np.mean(all_vals[finite_mask] <= 0)) if finite_mask.any() else np.nan

        pos_vals = all_vals[finite_mask & (all_vals > 0)]
        if pos_vals.size < 10:
            continue

        rows.append({
            "Age Bucket": label,
            "% Non-Employed": f"{zero_frac * 100:.1f}%",
            "P10": f"${np.percentile(pos_vals, 10):,.0f}",
            "P25": f"${np.percentile(pos_vals, 25):,.0f}",
            "Median": f"${np.percentile(pos_vals, 50):,.0f}",
            "P75": f"${np.percentile(pos_vals, 75):,.0f}",
            "P90": f"${np.percentile(pos_vals, 90):,.0f}",
            "Mean": f"${np.mean(pos_vals):,.0f}",
        })

    # Peak earnings row
    medians_by_age = np.array([
        np.nanmedian(earnings_matrix[:, t][earnings_matrix[:, t] > 0])
        if np.any(earnings_matrix[:, t] > 0) else np.nan
        for t in range(max_length)
    ])
    valid_peak = np.where(np.isfinite(medians_by_age))[0]
    if valid_peak.size:
        peak_idx = valid_peak[np.argmax(medians_by_age[valid_peak])]
        peak_age = int(ages[peak_idx])
        peak_median = medians_by_age[peak_idx]
    else:
        peak_age, peak_median = None, None

    return pd.DataFrame(rows), peak_age, peak_median


def create_plot_gkos_benchmark(config, n_workers: int = 20_000):
    """
    Run simulate_gkos with the current config GKOS params, scaled so the
    median starting earnings matches config.annual_income_real, starting at
    config.initial_age.  Returns (fig, stats) where stats is a dict of
    diagnostics matching the standalone benchmark script output.
    """
    try:
        import sys as _sys, importlib.util, os as _os
        _gbm = (_sys.modules.get('lifecycle_model.gkos_benchmark')
                or _sys.modules.get('gkos_benchmark'))
        if _gbm is None:
            _gbm_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                      'LIFECYCLE MODEL', 'gkos_benchmark.py')
            _spec = importlib.util.spec_from_file_location('gkos_benchmark', _gbm_path)
            _gbm = importlib.util.module_from_spec(_spec)
            _sys.modules.setdefault('gkos_benchmark', _gbm)
            _spec.loader.exec_module(_gbm)
    except Exception as e:
        return None, f"Could not load gkos_benchmark: {e}"

    params = _gbm.params_from_dict(config.gkos_params)
    age_start = int(config.initial_age)
    age_end   = int(config.retirement_age_max) + 1
    t_years   = max(age_end - age_start, 2)

    try:
        _, y, ages = _gbm.simulate_gkos(params=params, n_workers=n_workers,
                                         t_years=t_years, age_start=age_start, seed=42)
    except Exception as e:
        return None, f"simulate_gkos failed: {e}"

    # Scale so median first-year positive earnings = config.annual_income_real
    try:
        med_y0 = _gbm.median_first_year_positive(params, age_start, t_years=1)
        scale = float(config.annual_income_real) / med_y0 if med_y0 > 0 else 1.0
    except Exception:
        scale = 1.0
    y = y * scale

    # Percentiles conditioned on strictly positive earners
    pctiles = (10, 25, 50, 75, 90)
    bands = {p: np.full(t_years, np.nan) for p in pctiles}
    frac_zero = np.full(t_years, np.nan)
    for t in range(t_years):
        col = y[:, t]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        frac_zero[t] = float(np.mean(finite <= 0))
        pos = finite[finite > 0]
        if pos.size < 50:
            continue
        for p in pctiles:
            bands[p][t] = float(np.percentile(pos, p))

    peak_idx = int(np.nanargmax(bands[50])) if np.any(np.isfinite(bands[50])) else None
    peak_age = int(ages[peak_idx]) if peak_idx is not None else None
    peak_p90 = float(bands[90][peak_idx]) if peak_idx is not None else None
    avg_zero_frac = float(np.nanmean(frac_zero))
    age_start_idx = 0
    age55_idx = int(np.argmin(np.abs(ages - 55)))
    med_start = float(bands[50][age_start_idx]) if np.isfinite(bands[50][age_start_idx]) else None
    med55 = float(bands[50][age55_idx]) if np.isfinite(bands[50][age55_idx]) else None

    stats = dict(peak_age=peak_age, peak_p90=peak_p90,
                 avg_zero_frac=avg_zero_frac,
                 med_start=med_start, age_start=age_start, med55=med55)

    if not HAS_PLOTLY:
        return None, "Plotly not available"

    p10, p25, p50, p75, p90 = (bands[p] for p in pctiles)
    valid = np.isfinite(p50)
    av = ages[valid]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([av, av[::-1]]),
        y=np.concatenate([p90[valid], p10[valid][::-1]]),
        fill='toself', fillcolor='rgba(61,143,110,0.22)',
        line=dict(color='rgba(0,0,0,0)'), name='P10–P90', hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([av, av[::-1]]),
        y=np.concatenate([p75[valid], p25[valid][::-1]]),
        fill='toself', fillcolor='rgba(126,207,174,0.40)',
        line=dict(color='rgba(0,0,0,0)'), name='P25–P75', hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=av, y=p50[valid], mode='lines',
        line=dict(color='#fff4e0', width=2.5), name='Median',
        hovertemplate='Age %{x}: $%{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title='Earnings Levels by Age',
        xaxis_title='Age', yaxis_title='Earnings (Real $)',
        template='plotly_dark',
        plot_bgcolor='#0b0f14', paper_bgcolor='#0b0f14',
        height=520,
        yaxis=dict(tickformat='$,.0f'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        font=dict(color='#c8d0da'),
    )
    fig.update_xaxes(gridcolor='rgba(60,74,92,0.6)')
    fig.update_yaxes(gridcolor='rgba(60,74,92,0.6)')

    return fig, stats


def create_plot_cumulative_retirement_probability(retirement_ages, config, median_age, num_outer):
    """Create plot for cumulative retirement probability returns figure"""
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
        

        if config.include_social_security:
            fig.add_vline(x=config.social_security_start_age, line_dash="dot", line_color="gray",
                         annotation_text=f"SS Age {config.social_security_start_age}", 
                         annotation_position="top")
        
        if not np.isnan(median_age):
            fig.add_vline(x=median_age, line_dash="dash", line_color="gray",
                         annotation_text=f"Median {median_age:.1f}", 
                         annotation_position="top")
        
        fig.update_layout(
            title="Cumulative Probability of Retiring by Age",
            xaxis_title="Age",
            yaxis_title="Cumulative Probability (%)",
            template='plotly_white',
            height=500,
            showlegend=False,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, 100])
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        
        return fig
    else:

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
    if not IMPORTS_SUCCESSFUL:
        st.error(f"❌ Failed to import required modules from LIFECYCLE MODEL: {IMPORT_ERROR}")
        st.error("Please ensure the LIFECYCLE MODEL directory exists and contains all required modules.")
        return
    
    st.title("📊 Lifecycle Retirement Simulation")
    st.markdown("**Version 10.0.7** - Simulate retirement outcomes with uncertain labor income and market returns")
    st.caption("All returns and dollar amounts in this app are in **real** (inflation-adjusted) terms.")
    

    tab_sim, tab_params = st.tabs(["🏠 Lifecycle Simulation", "📈 Parametric Model Estimation"])
    
    with tab_sim:
        lifecycle_simulation_tab()
    
    with tab_params:
        parametric_model_estimation_tab()

def lifecycle_simulation_tab():
    """Main lifecycle simulation tab"""
    

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
                

                build_dir = os.path.join(parent_dir, 'build')
                if os.path.exists(build_dir):
                    st.write(f"✅ Build directory exists: `{build_dir}`")

                    import platform
                    is_windows = platform.system().lower() == 'windows'
                    ext = '.pyd' if is_windows else '.so'
                    build_files = [f for f in os.listdir(build_dir) if f.endswith(('.pyd', '.so'))]
                    if build_files:
                        st.write(f"✅ Found {len(build_files)} compiled module(s):")
                        for f in build_files[:5]:
                            st.code(f)

                        matching_files = [f for f in build_files if f.endswith(ext)]
                        if not matching_files:
                            st.warning(f"⚠️ No {ext} files found for current platform ({platform.system()})")
                            st.write(f"   Found: {', '.join(set([f.split('.')[-1] for f in build_files]))}")
                    else:
                        st.write(f"❌ No compiled modules (.pyd or .so) found in build directory")
                        st.write("   Run `./build_cython.sh` (Linux/macOS) or `build_cython.bat` (Windows) to compile")
                else:
                    st.write(f"❌ Build directory not found: `{build_dir}`")
                    st.write("   Run `./build_cython.sh` (Linux/macOS) or `build_cython.bat` (Windows) to create and compile")
                
                st.write(f"**Python version:** {sys.version_info.major}.{sys.version_info.minor}")
                st.write(f"**Platform:** {platform.system()} {platform.machine()}")
                python_ver = f"{sys.version_info.major}{sys.version_info.minor}"
                if is_windows:
                    st.write(f"**Expected build path:** `build/lib.win-amd64-cpython-{python_ver}/`")
                else:
                    st.write(f"**Expected build path:** `build/lib.linux-{platform.machine()}-cpython-{python_ver}/` or `build/` (inplace)")
    

    # Auto-fill block bootstrap parameters from Parametric Model data source (plan item 2)
    _default_bootstrap_path = os.path.join(parent_dir, "data", "TFP - Block Bootstrap.csv")
    if st.button("Auto-fill block bootstrap from Parametric Model", key="lifecycle_autofill_btn"):
        _path = st.session_state.get("_param_data_path", _default_bootstrap_path)
        _port = st.session_state.get("_param_portfolio_col", "Three Fund Portfolio")
        _inf = st.session_state.get("_param_inflation_col", "Inflation")
        st.session_state._lifecycle_autofill_path = _path
        st.session_state._lifecycle_autofill_portfolio_col = _port
        st.session_state._lifecycle_autofill_inflation_col = _inf
        st.rerun()

    config = create_config_from_sidebar()
    



    validation_ok = True
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

            st.session_state.simulation_results = None
            st.session_state.config = None
            run_simulation(config)
    

    if st.session_state.simulation_results is not None:
        display_results(st.session_state.simulation_results, st.session_state.config)

def parametric_model_estimation_tab():
    """Parametric model estimation tab with separate HMM and parameter estimation sections"""
    st.header("📈 Parametric Model Analysis")
    st.markdown("HMM regime detection and regime-conditional parameter estimation")
    

    if 'hmm_results' not in st.session_state:
        st.session_state.hmm_results = None
    if 'param_estimation_results' not in st.session_state:
        st.session_state.param_estimation_results = None
    

    st.subheader("📊 Data Input")
    data_option = st.radio(
        "Data Source",
        options=["Use Default Dataset", "Upload CSV"],
        horizontal=True
    )
    
    returns_data = None
    
    if data_option == "Use Default Dataset":
        default_dataset = st.selectbox(
            "Select Default Dataset",
            options=["S&P 500", "3-Fund Portfolio"],
            help="Choose from pre-loaded datasets"
        )
        
        try:
            if default_dataset == "S&P 500":
                data_path = os.path.join(parent_dir, "data", "Market and Inflation - Regime Testing.csv")
                if os.path.exists(data_path):
                    _cache_key = ('default', 'S&P 500')
                    if getattr(st.session_state, '_returns_data_cache_key', None) == _cache_key and '_returns_data_cache' in st.session_state:
                        returns_data = st.session_state._returns_data_cache
                    else:
                        df = pd.read_csv(data_path)
                        date_col = "Date" if "Date" in df.columns else df.columns[0]
                        returns_col = "S&P 500" if "S&P 500" in df.columns else [c for c in df.columns if c != date_col][0]
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        df = df.dropna(subset=[date_col])
                        df = df.set_index(date_col).sort_index()
                        prices = df[returns_col].dropna()
                        returns_data = np.log(prices / prices.shift(1)).dropna() * 100 if prices.min() > 100 else prices
                        st.session_state._returns_data_cache = returns_data
                        st.session_state._returns_data_cache_key = _cache_key
                        if prices.min() > 100:
                            st.info(f"📊 Converted price data to log returns (percentage)")
                    st.success(f"✅ Loaded S&P 500 data: {len(returns_data)} observations")
                    st.session_state._param_data_path = os.path.abspath(data_path)
                    st.session_state._param_portfolio_col = "S&P 500"
                    st.session_state._param_inflation_col = "Inflation"
                else:
                    st.error(f"❌ Could not find S&P 500 data file at: {data_path}")
            
            elif default_dataset == "3-Fund Portfolio":
                data_path = os.path.join(parent_dir, "data", "TFP - Block Bootstrap.csv")
                if os.path.exists(data_path):
                    _cache_key = ('default', '3-Fund Portfolio')
                    if getattr(st.session_state, '_returns_data_cache_key', None) == _cache_key and '_returns_data_cache' in st.session_state:
                        returns_data = st.session_state._returns_data_cache
                    else:
                        df = pd.read_csv(data_path)
                        date_col = "Date" if "Date" in df.columns else df.columns[0]
                        returns_col = "Three Fund Portfolio" if "Three Fund Portfolio" in df.columns else [c for c in df.columns if c != date_col][0]
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        df = df.dropna(subset=[date_col])
                        df = df.set_index(date_col).sort_index()
                        prices = df[returns_col].dropna()
                        returns_data = np.log(prices / prices.shift(1)).dropna() * 100 if prices.min() > 100 else prices
                        st.session_state._returns_data_cache = returns_data
                        st.session_state._returns_data_cache_key = _cache_key
                        if prices.min() > 100:
                            st.info(f"📊 Converted price data to log returns (percentage)")
                    st.success(f"✅ Loaded 3-Fund Portfolio data: {len(returns_data)} observations")
                    st.session_state._param_data_path = os.path.abspath(data_path)
                    st.session_state._param_portfolio_col = "Three Fund Portfolio"
                    st.session_state._param_inflation_col = "Inflation"
                else:
                    st.error(f"❌ Could not find 3-Fund Portfolio data file at: {data_path}")
        except Exception as e:
            st.error(f"Error loading default dataset: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    else:
        uploaded_file = st.file_uploader(
            "Upload Returns Data (CSV)",
            type=['csv'],
            help="CSV file with date index and returns column"
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                date_col = st.selectbox("Select Date Column", df.columns.tolist(), key="upload_date")
                returns_col = st.selectbox("Select Returns Column", [c for c in df.columns if c != date_col], key="upload_returns")
                
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col])
                df = df.set_index(date_col)
                returns_data = df[returns_col].dropna()
                st.success(f"✅ Loaded {len(returns_data)} observations")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    if returns_data is None or len(returns_data) < 100:
        if returns_data is None:
            st.warning("⚠️ Please load returns data first")
        else:
            st.warning(f"⚠️ Insufficient data: {len(returns_data)} observations. Need at least 100.")
        return

    use_standalone_scripts = st.checkbox(
        "Use standalone scripts",
        value=True,
        help="Run HMM then Moment Matching. Requires S&P 500 dataset."
    )
    if use_standalone_scripts and data_option == "Upload CSV":
        st.warning("⚠️ Standalone scripts require **S&P 500**. Use 'Use Default Dataset' → S&P 500.")
    script_input_path = None
    if use_standalone_scripts and data_option == "Use Default Dataset" and default_dataset == "S&P 500":
        script_input_path = os.path.join(parent_dir, "data", "Market and Inflation - Regime Testing.csv")
        if not os.path.exists(script_input_path):
            st.error(f"❌ Data file not found: {script_input_path}")
            script_input_path = None
        else:
            st.caption("📜 HMM and Param Estimation will run the standalone scripts.")
    elif use_standalone_scripts and data_option == "Use Default Dataset" and default_dataset == "3-Fund Portfolio":
        st.warning("⚠️ Scripts expect 'S&P 500' and 'Inflation' columns. Use S&P 500 for scripts.")
        script_input_path = None

    st.session_state._use_standalone_scripts = use_standalone_scripts
    st.session_state._script_input_path = script_input_path

    # Keep daily data for HMM (daily vol, 21-day subsample). No calendar resampling.
    returns_for_regime = returns_data
    resampling_applied = False
    # Do not clear _returns_original_for_plot here: it is set when HMM runs and is used by the HMM tab
    # to show daily-resolution plots. Clearing it on every rerun caused the HMM plot to switch to
    # monthly resolution when returning from Parameter Estimation.

    hmm_tab, param_tab, dist_tab = st.tabs(["🔍 HMM Regime Detection", "🔬 Parameter Estimation", "📊 Distribution Comparison"])

    with hmm_tab:
        hmm_regime_detection_section(returns_for_regime)

    with param_tab:
        parameter_estimation_section(returns_for_regime)

    with dist_tab:
        distribution_comparison_section(returns_for_regime)

def _infer_regimes_1a_style(returns_series, lookback_period=21, num_regimes=3, random_seed=1):
    """Daily returns -> 21-day annualized vol -> subsample every 21 -> HMM on vol. Returns (regime_labels_monthly, monthly_returns_series, transition_matrix, hmm_model)."""
    from hmmlearn import hmm
    df = pd.DataFrame({'returns': returns_series}, index=returns_series.index)
    # Annualized vol: rolling(21).std() * sqrt(252)
    df['Volatility'] = df['returns'].rolling(window=lookback_period).std() * np.sqrt(252)
    df = df.dropna(subset=['Volatility'])
    if len(df) < lookback_period * num_regimes:
        return None, None, None, None
    df_monthly = df.iloc[::lookback_period].copy()
    X = df_monthly[['Volatility']].values
    if len(X) < num_regimes:
        return None, None, None, None
    model = hmm.GaussianHMM(n_components=num_regimes, covariance_type='diag', n_iter=2000, tol=1e-5, random_state=random_seed)
    model.fit(X)
    raw_labels = model.predict(X)
    # Sort regimes by mean vol (low -> high), map to 0,1,2
    regime_data = [{'original_index': i, 'mean_vol': model.means_[i, 0]} for i in range(num_regimes)]
    regime_data.sort(key=lambda x: x['mean_vol'], reverse=False)
    index_map = {d['original_index']: i for i, d in enumerate(regime_data)}
    regime_labels = np.array([index_map.get(int(r), 0) if np.isfinite(r) else 0 for r in raw_labels])
    sorted_indices = [d['original_index'] for d in regime_data]
    transmat_sorted = model.transmat_[np.ix_(sorted_indices, sorted_indices)]
    # Monthly returns: sum of daily log returns in each 21-day block
    monthly_returns_list = []
    for idx in df_monthly.index:
        loc = df.index.get_loc(idx)
        end_loc = min(loc + lookback_period, len(df))
        block = df.iloc[loc:end_loc]['returns']
        monthly_returns_list.append(block.sum())
    monthly_returns = pd.Series(monthly_returns_list, index=df_monthly.index)
    return regime_labels, monthly_returns, transmat_sorted, model


def hmm_regime_detection_section(returns_data):
    """HMM regime detection (daily vol, 21-day subsample when data is daily)."""
    import os
    st.subheader("🔍 Hidden Markov Model Regime Detection")
    st.markdown("Infer market regimes from returns (daily vol, 21-day steps).")
    

    with st.expander("⚙️ HMM Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            num_regimes = st.number_input(
                "Number of Regimes",
                min_value=2,
                max_value=6,
                value=3,
                help="Number of market regimes to infer"
            )
        with col2:
            random_seed = st.number_input(
                "Random Seed",
                min_value=0,
                value=1,
                help="Random seed for HMM inference"
            )
            lookback_period = st.number_input(
                "Lookback Period (days)",
                min_value=5,
                max_value=252,
                value=21,
                help="Non-overlapping step for HMM (trading days)"
            )
    

    if st.button("🔍 Run HMM Regime Detection", type="primary", use_container_width=True):
        with st.spinner("Running HMM..."):
            try:
                # Run HMM script when standalone scripts and S&P 500 path are set
                run_script_1a = (
                    getattr(st.session_state, '_use_standalone_scripts', False)
                    and getattr(st.session_state, '_script_input_path', None)
                )
                if run_script_1a:
                    import subprocess
                    script_1a = os.path.join(parent_dir, "Parametric Model (Unfinished)", "Hidden Markov Models", "1A. HMM_Vol.py")
                    data_dir = os.path.join(parent_dir, "data")
                    proc = subprocess.run(
                        [sys.executable, script_1a, st.session_state._script_input_path, data_dir],
                        cwd=parent_dir,
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if proc.returncode != 0:
                        st.error(f"❌ HMM script failed: {proc.stderr[:500] if proc.stderr else proc.stdout[:500]}")
                    else:
                        regime_csv = os.path.join(data_dir, "regime_classification_nominal_returns.csv")
                        trans_csv = os.path.join(data_dir, "HMM_transition_matrix.csv")
                        means_json = os.path.join(data_dir, "hmm_means.json")
                        if not os.path.exists(regime_csv):
                            st.error("❌ HMM did not write regime_classification_nominal_returns.csv")
                        else:
                            df_reg = pd.read_csv(regime_csv)
                            regime_col = [c for c in df_reg.columns if 'regime' in c.lower() or 'Regime' in c][0]
                            regime_labels = np.array(df_reg[regime_col].values, dtype=float)
                            ret_col = [c for c in df_reg.columns if 'Nominal' in c or 'Return' in c or c == 'returns'][0]
                            monthly_returns = pd.Series(df_reg[ret_col].values)
                            if 'Monthly Start Date' in df_reg.columns:
                                monthly_returns.index = pd.to_datetime(df_reg['Monthly Start Date'])
                            transition_matrix = np.eye(3)
                            if os.path.exists(trans_csv):
                                tm_df = pd.read_csv(trans_csv, index_col=0)
                                transition_matrix = tm_df.values.astype(float)
                            class _FakeHMM:
                                means_ = np.array([[0.0]])
                            num_regimes_1a = 3
                            if os.path.exists(means_json):
                                import json
                                with open(means_json) as f:
                                    j = json.load(f)
                                _FakeHMM.means_ = np.array(j['means'])
                                num_regimes_1a = int(j.get('num_regimes', 3))
                            st.session_state._returns_original_for_plot = returns_data
                            st.session_state.hmm_results = {
                                'regime_labels': regime_labels,
                                'transition_matrix': transition_matrix,
                                'hmm_model': _FakeHMM(),
                                'num_regimes': num_regimes_1a,
                                'features': 'vol',
                                'returns_data': monthly_returns,
                                'lookback_period': 21,
                                'regime_index_map': {i: i for i in range(num_regimes_1a)},
                                'portfolio_name': 'S&P 500'
                            }
                            st.session_state._hmm_from_script = True
                            st.success("✅ HMM complete. Results match standalone script.")
                else:
                    st.session_state._hmm_from_script = False
                    # Use daily-vol style when we have daily-like data
                    use_1a = (
                        hasattr(returns_data, 'index') and isinstance(returns_data.index, pd.DatetimeIndex)
                        and len(returns_data) >= 2500
                    )
                    if use_1a:
                        regime_labels, monthly_returns, transition_matrix, hmm_model = _infer_regimes_1a_style(
                            returns_data, lookback_period=int(lookback_period), num_regimes=int(num_regimes), random_seed=int(random_seed)
                        )
                        if regime_labels is not None:
                            st.session_state._returns_original_for_plot = returns_data
                            portfolio_name = "Portfolio"
                            if hasattr(st.session_state, 'config') and st.session_state.config is not None:
                                if hasattr(st.session_state.config, 'portfolio_column_name') and st.session_state.config.portfolio_column_name:
                                    portfolio_name = st.session_state.config.portfolio_column_name
                            st.session_state.hmm_results = {
                                'regime_labels': regime_labels,
                                'transition_matrix': transition_matrix,
                                'hmm_model': hmm_model,
                                'num_regimes': num_regimes,
                                'features': 'vol',
                                'returns_data': monthly_returns,
                                'lookback_period': lookback_period,
                                'regime_index_map': {i: i for i in range(int(num_regimes))},
                                'portfolio_name': portfolio_name
                            }
                            try:
                                out_dir = os.path.join(parent_dir, "output")
                                os.makedirs(out_dir, exist_ok=True)
                                hmm_csv_path = os.path.join(out_dir, "hmm_regime_results.csv")
                                n = len(regime_labels)
                                dates = list(monthly_returns.index[:n]) if hasattr(monthly_returns, 'index') else list(range(n))
                                pd.DataFrame({'date': dates, 'regime': regime_labels}).to_csv(hmm_csv_path, index=False)
                                st.caption(f"💾 HMM results saved to `output/hmm_regime_results.csv`")
                            except Exception as save_err:
                                st.caption(f"Could not save HMM CSV: {save_err}")
                            st.success("✅ HMM complete (daily vol, 21-day steps). Plots use daily resolution.")
                        else:
                            st.error("❌ Not enough data for daily-vol HMM (need enough daily obs after 21-day rolling).")
                    else:
                        # Fallback: monthly or short series — use 2A if available
                        param_model_path = os.path.join(
                            parent_dir,
                            "Parametric Model (Unfinished)",
                            "Stochastic Vol + Jump Diffusion",
                            "2A. Regime-Conditional Parameter Estimation.py"
                        )
                        if os.path.exists(param_model_path):
                            from importlib.util import spec_from_file_location, module_from_spec
                            spec = spec_from_file_location("regime_estimation", param_model_path)
                            regime_module = module_from_spec(spec)
                            spec.loader.exec_module(regime_module)
                            regime_labels, transition_matrix, hmm_model = regime_module.infer_regimes_hmm(
                                returns_data, features='vol', num_regimes=int(num_regimes),
                                random_seed=int(random_seed), n_iter=100, tol=1e-3
                            )
                            if regime_labels is not None:
                                hmm_means = hmm_model.means_
                                regime_data = [{'original_index': i, 'feature_mean': hmm_means[i, 0]} for i in range(int(num_regimes))]
                                regime_data.sort(key=lambda x: x['feature_mean'], reverse=False)
                                index_map = {d['original_index']: i for i, d in enumerate(regime_data)}
                                def _safe_regime_label(r):
                                    if not np.isfinite(r):
                                        return index_map.get(0, 0)
                                    return index_map.get(int(r), r)
                                regime_labels_sorted = np.array([_safe_regime_label(r) for r in regime_labels])
                                sorted_indices = [d['original_index'] for d in regime_data]
                                transition_matrix_sorted = transition_matrix[np.ix_(sorted_indices, sorted_indices)]
                                portfolio_name = "Portfolio"
                                if hasattr(st.session_state, 'config') and st.session_state.config is not None:
                                    if hasattr(st.session_state.config, 'portfolio_column_name') and st.session_state.config.portfolio_column_name:
                                        portfolio_name = st.session_state.config.portfolio_column_name
                                st.session_state._returns_original_for_plot = None
                                st.session_state.hmm_results = {
                                    'regime_labels': regime_labels_sorted,
                                    'transition_matrix': transition_matrix_sorted,
                                    'hmm_model': hmm_model,
                                    'num_regimes': num_regimes,
                                    'features': 'vol',
                                    'returns_data': returns_data,
                                    'lookback_period': lookback_period,
                                    'regime_index_map': index_map,
                                    'portfolio_name': portfolio_name
                                }
                                try:
                                    out_dir = os.path.join(parent_dir, "output")
                                    os.makedirs(out_dir, exist_ok=True)
                                    hmm_csv_path = os.path.join(out_dir, "hmm_regime_results.csv")
                                    n = len(regime_labels_sorted)
                                    dates = (list(returns_data.index[:n]) if hasattr(returns_data, 'index') else list(range(n)))
                                    pd.DataFrame({'date': dates, 'regime': regime_labels_sorted}).to_csv(hmm_csv_path, index=False)
                                    st.caption(f"💾 HMM results saved to `output/hmm_regime_results.csv`")
                                except Exception as save_err:
                                    st.caption(f"Could not save HMM CSV: {save_err}")
                                st.success("✅ HMM regime detection complete (2A fallback for monthly/short data).")
                            else:
                                st.error("❌ HMM inference failed")
                        else:
                            st.warning("⚠️ Load daily returns (2500+ obs with dates) for daily-vol HMM, or use fallback.")
            except Exception as e:
                st.error(f"❌ Error during HMM inference: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    

    if st.session_state.hmm_results is not None:
        try:
            display_hmm_results(st.session_state.hmm_results)
        except Exception as e:
            st.error(f"❌ Error displaying HMM results: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

def parameter_estimation_section(returns_data):
    """Parameter estimation section (can use HMM results or run its own)"""
    st.subheader("🔬 Regime-Conditional Parameter Estimation")
    st.markdown("Estimate Bates model parameters conditional on regimes")
    

    use_existing_hmm = st.checkbox(
        "Use existing HMM results",
        value=st.session_state.hmm_results is not None,
        help="If checked, uses HMM results from the HMM tab. Otherwise, runs HMM first."
    )
    

    with st.expander("⚙️ Parameter Estimation Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox(
                "Model Type",
                options=['Bates Model (Jump Diffusion + Stochastic Vol)', 
                        'Heston Stochastic Volatility',
                        'Merton Jump Diffusion',
                        'Geometric Brownian Motion'],
                index=0,
                help="Select the model to fit to the data"
            )
            estimation_method = st.selectbox(
                "Estimation Method",
                options=['moment', 'qmle'],
                index=0,
                help="Parameter estimation method: moment matching or quasi-maximum likelihood"
            )
        with col2:
            if not use_existing_hmm:

                hmm_features = st.selectbox(
                    "HMM Features (if running new HMM)",  
                    options=['vol', 'skew', 'vol&skew'],
                    index=0,
                    help="Features for HMM if running new inference"
                )
                num_regimes = st.number_input(
                    "Number of Regimes (if running new HMM)",
                    min_value=1,
                    max_value=6,
                    value=1,
                    help="Number of regimes if running new HMM (use 1 to fit to entire historical data without regimes)"
                )
                random_seed = st.number_input(
                    "Random Seed (if running new HMM)",
                    min_value=0,
                    value=1,
                    help="Random seed for HMM if running new inference"
                )
            else:
                st.info("Using HMM results from HMM tab")
                if st.session_state.hmm_results:
                    st.write(f"**Current HMM:** {st.session_state.hmm_results['num_regimes']} regimes, "
                            f"features='{st.session_state.hmm_results['features']}'")
        

        with st.expander("🔧 Parameter Bounds (Optional)", expanded=False):
            st.markdown("Set optimization bounds for Bates model parameters. Leave defaults for standard bounds.")
            

            default_bounds = [
                (-0.3, 0.3),
                (0.1, 50.0),
                (0.001, 0.25),
                (0.1, 10.0),
                (-0.99, 0.0),
                (0.001, 0.25),
                (0.0, 5.0),
                (-0.3, 0.0),
                (0.01, 0.5),
            ]
            
            param_names = ['mu', 'kappa', 'theta', 'nu', 'rho', 'v0', 'lam', 'mu_J', 'sigma_J']
            param_descriptions = {
                'mu': 'Drift (annual)',
                'kappa': 'Mean reversion speed',
                'theta': 'Long-run variance',
                'nu': 'Volatility of variance',
                'rho': 'Correlation',
                'v0': 'Initial variance',
                'lam': 'Jump intensity (per year)',
                'mu_J': 'Jump mean',
                'sigma_J': 'Jump volatility'
            }
            

            col_reset, col_use_bounds = st.columns(2)
            with col_reset:
                if st.button("🔄 Reset to Default Bounds", key="reset_bounds_btn", use_container_width=True):

                    for i, param_name in enumerate(param_names):
                        st.session_state[f"bound_lower_{param_name}"] = float(default_bounds[i][0])
                        st.session_state[f"bound_upper_{param_name}"] = float(default_bounds[i][1])
                    st.rerun()
            
            with col_use_bounds:
                use_bounds = st.checkbox("Enable Bounds", value=True, key="use_bounds_checkbox",
                                       help="Uncheck to disable bounds entirely (unbounded optimization)")
            
            custom_bounds = {}
            for i, param_name in enumerate(param_names):
                col_lb, col_ub = st.columns(2)
                with col_lb:
                    lower_bound = st.number_input(
                        f"{param_descriptions[param_name]} (Lower)",
                        value=float(default_bounds[i][0]),
                        step=0.01,
                        format="%.4f",
                        key=f"bound_lower_{param_name}"
                    )
                with col_ub:
                    upper_bound = st.number_input(
                        f"{param_descriptions[param_name]} (Upper)",
                        value=float(default_bounds[i][1]),
                        step=0.01,
                        format="%.4f",
                        key=f"bound_upper_{param_name}"
                    )
                custom_bounds[param_name] = (lower_bound, upper_bound)
            

            use_custom_bounds = st.checkbox("Use Custom Bounds", value=False, help="Enable to use the bounds above instead of defaults")
            
            if not use_bounds:

                st.session_state.param_bounds = None
                st.session_state.param_use_bounds = False
            elif use_custom_bounds:

                st.session_state.param_bounds = [
                    custom_bounds['mu'],
                    custom_bounds['kappa'],
                    custom_bounds['theta'],
                    custom_bounds['nu'],
                    custom_bounds['rho'],
                    custom_bounds['v0'],
                    custom_bounds['lam'],
                    custom_bounds['mu_J'],
                    custom_bounds['sigma_J']
                ]
                st.session_state.param_use_bounds = True
            else:

                st.session_state.param_bounds = None
                st.session_state.param_use_bounds = True
        

        if 'param_bounds' not in st.session_state:
            st.session_state.param_bounds = None
        if 'param_use_bounds' not in st.session_state:
            st.session_state.param_use_bounds = True
            

        st.write("**Data Sampling Options (for large datasets):**")
        use_sampling = st.checkbox("Use Data Sampling for Large Datasets", value=False, 
                                  key="use_sampling_estimation",
                                  help="Sample data for estimation (faster, more stable for large datasets). Optional for datasets > 10000 observations.")
        
        if use_sampling:
            sample_size = st.number_input("Sample Size", min_value=500, max_value=10000, 
                                         value=5000, step=500, key="estimation_sample_size",
                                         help="Number of observations to use for estimation")
        

        st.write("**Optimization Settings:**")
        n_restarts = st.number_input("Number of Restarts", min_value=1, max_value=10000, value=15, step=1,
                                    key="param_estimation_restarts",
                                    help="Number of optimization restarts (more restarts = better chance of finding global optimum, but slower). Each restart uses a different random starting point.")
        max_iterations = st.number_input("Max Iterations per Restart", min_value=100, max_value=50000, value=10000, step=500,
                                        key="param_estimation_maxiter",
                                        help="Maximum iterations per optimization restart (matches standalone script default)")
        

        _col_viz, _col_nsim, _col_debug = st.columns(3)
        with _col_viz:
            st.checkbox("Show distribution visualizations", value=True, key="param_show_viz",
                        help="Show distribution plots in the results section below.")
        with _col_nsim:
            st.number_input(
                "Number of simulations for visuals",
                min_value=500,
                max_value=50000,
                value=5000,
                step=500,
                key="param_viz_n_simulations",
                help="Monte Carlo paths for distribution plots (higher = smoother, slower)."
            )
        with _col_debug:
            enable_debug = st.checkbox("Enable Debug Mode", value=False, key="enable_debug_mode",
                                      help="Enable detailed debug output during parameter estimation (slower)")
    

    if st.button("🔬 Run Parameter Estimation", type="primary", use_container_width=True):
        with st.spinner("Running parameter estimation..."):
            try:
                # Run moment-matching script when HMM was from standalone script
                run_script_2b = (
                    use_existing_hmm
                    and getattr(st.session_state, '_hmm_from_script', False)
                    and getattr(st.session_state, '_use_standalone_scripts', False)
                    and st.session_state.hmm_results is not None
                )
                if run_script_2b:
                    import subprocess
                    import sys as _sys
                    script_2b = os.path.join(parent_dir, "Parametric Model (Unfinished)", "Stochastic Vol + Jump Diffusion", "2B. Moment Matching.py")
                    regime_csv = os.path.join(parent_dir, "data", "regime_classification_nominal_returns.csv")
                    if not os.path.exists(regime_csv):
                        st.error("❌ Run HMM first to create regime_classification_nominal_returns.csv")
                    else:
                        proc = subprocess.run(
                            [_sys.executable, script_2b, "--csv", regime_csv, "--no-plots"],
                            cwd=parent_dir,
                            capture_output=True,
                            text=True,
                            timeout=600,
                        )
                        if proc.returncode != 0:
                            st.error(f"❌ Moment matching script failed: {proc.stderr[:800] if proc.stderr else proc.stdout[:800]}")
                        else:
                            bates_csv = os.path.join(parent_dir, "data", "Bates_per_regime_Moment_Matching_results.csv")
                            if not os.path.exists(bates_csv):
                                st.error("❌ Moment matching did not write Bates_per_regime_Moment_Matching_results.csv")
                            else:
                                bates_df = pd.read_csv(bates_csv)
                                params_dict = {}
                                moment_info_dict = {}
                                results_list = []
                                for _, row in bates_df.iterrows():
                                    name = str(row.get('name', ''))
                                    if not name.startswith('regime_'):
                                        continue
                                    regime_id = float(name.split('_')[1])
                                    params_dict[regime_id] = [
                                        float(row['mu_annual']), float(row['kappa']), float(row['theta_annual']),
                                        float(row['nu']), float(row['rho']), float(row['v0']),
                                        float(row['lambda_per_year']), float(row['mu_J']), float(row['sigma_J'])
                                    ]
                                    moment_info_dict[regime_id] = {
                                        'n_obs': int(row['N_obs']),
                                        'emp_mean': float(row['emp_mean']), 'model_mean': float(row['model_mean']),
                                        'emp_std': float(row['emp_std']), 'model_std': float(row['model_std']),
                                        'emp_skew': float(row['emp_skew']), 'model_skew': float(row['model_skew']),
                                        'emp_kurt': float(row['emp_kurt']), 'model_kurt': float(row['model_kurt']),
                                    }
                                    results_list.append({
                                        'name': name, 'params': params_dict[regime_id], 'n_obs': int(row['N_obs']),
                                        'emp_moments': {'mean': float(row['emp_mean']), 'std': float(row['emp_std']), 'skew': float(row['emp_skew']), 'kurt': float(row['emp_kurt'])},
                                        'model_moments': {'mean': float(row['model_mean']), 'std': float(row['model_std']), 'skew': float(row['model_skew']), 'kurt': float(row['model_kurt'])},
                                    })
                                hmm_results = st.session_state.hmm_results
                                df_reg = pd.read_csv(regime_csv)
                                ret_col = [c for c in df_reg.columns if 'Nominal' in c or 'Return' in c][0]
                                regime_col = [c for c in df_reg.columns if 'regime' in c.lower() or 'Regime' in c][0]
                                df = pd.DataFrame({
                                    'returns': np.log(1.0 + df_reg[ret_col].values / 100.0),
                                    'regime': df_reg[regime_col].values.astype(float)
                                })
                                results = {
                                    'params_dict': params_dict,
                                    'moment_info_dict': moment_info_dict,
                                    'regime_labels': df_reg[regime_col].values,
                                    'transition_matrix': hmm_results.get('transition_matrix'),
                                    'hmm_model': hmm_results.get('hmm_model'),
                                    'num_regimes': len(params_dict),
                                    'features': 'vol',
                                    'method': 'moment',
                                    'results_list': results_list,
                                    'df': df
                                }
                                st.session_state.param_estimation_results = results
                                st.success("✅ Parameter estimation complete. Results match standalone script.")
                else:
                    param_model_path = os.path.join(
                        parent_dir,
                        "Parametric Model (Unfinished)",
                        "Stochastic Vol + Jump Diffusion",
                        "2A. Regime-Conditional Parameter Estimation.py"
                    )
                
                    if os.path.exists(param_model_path):
                        # Use cached module if available, otherwise load and cache it
                        module_cache_key = "regime_estimation_module"
                        if module_cache_key not in st.session_state:
                            from importlib.util import spec_from_file_location, module_from_spec
                            spec = spec_from_file_location("regime_estimation", param_model_path)
                            regime_module = module_from_spec(spec)
                            
                            # Suppress Rich console output in Streamlit to prevent blocking
                            import sys
                            from io import StringIO
                            old_stdout = sys.stdout
                            old_stderr = sys.stderr
                            sys.stdout = StringIO()
                            sys.stderr = StringIO()
                            try:
                                spec.loader.exec_module(regime_module)
                                # Patch the console object to suppress output
                                if hasattr(regime_module, 'console'):
                                    # Create a no-op console that doesn't output anything
                                    class NoOpConsole:
                                        def print(self, *args, **kwargs):
                                            pass
                                    regime_module.console = NoOpConsole()
                            finally:
                                sys.stdout = old_stdout
                                sys.stderr = old_stderr
                            
                            st.session_state[module_cache_key] = regime_module
                        else:
                            regime_module = st.session_state[module_cache_key]
                        

                        use_bounds_flag = st.session_state.get('param_use_bounds', True)
                        bounds_to_use = st.session_state.get('param_bounds', None)
                        if bounds_to_use is None and use_bounds_flag:
                            bounds_to_use = regime_module.DEFAULT_BOUNDS
                        elif not use_bounds_flag:


                            bounds_to_use = regime_module.DEFAULT_BOUNDS
                        



                        import sys
                        if enable_debug:

                            debug_set = False

                            if hasattr(regime_module, 'fit_bates_moment_matching'):
                                fit_func = regime_module.fit_bates_moment_matching
                                if hasattr(fit_func, '__module__'):
                                    mod_name = fit_func.__module__
                                    if mod_name in sys.modules:
                                        mod = sys.modules[mod_name]
                                        if hasattr(mod, 'DEBUG'):
                                            mod.DEBUG = True
                                            debug_set = True

                            if not debug_set:
                                for mod_name in list(sys.modules.keys()):
                                    if 'moment_matching' in mod_name.lower():
                                        mod = sys.modules[mod_name]
                                        if hasattr(mod, 'DEBUG'):
                                            mod.DEBUG = True
                                            debug_set = True
                                            break
                            if debug_set:
                                st.info("✅ Debug mode enabled")
                        else:

                            if hasattr(regime_module, 'fit_bates_moment_matching'):
                                fit_func = regime_module.fit_bates_moment_matching
                                if hasattr(fit_func, '__module__'):
                                    mod_name = fit_func.__module__
                                    if mod_name in sys.modules:
                                        mod = sys.modules[mod_name]
                                        if hasattr(mod, 'DEBUG'):
                                            mod.DEBUG = False
                            for mod_name in list(sys.modules.keys()):
                                if 'moment_matching' in mod_name.lower():
                                    mod = sys.modules[mod_name]
                                    if hasattr(mod, 'DEBUG'):
                                        mod.DEBUG = False
                                        break
                        
                        if use_existing_hmm and st.session_state.hmm_results is not None:

                            hmm_results = st.session_state.hmm_results
                            regime_labels = hmm_results['regime_labels']
                            num_regimes = hmm_results['num_regimes']
                            # Use HMM returns (monthly) so length matches regime_labels
                            _ret = hmm_results['returns_data']
                            if hasattr(_ret, 'values'):
                                returns_for_estimation = _ret.values.copy()
                            else:
                                returns_for_estimation = np.array(_ret).copy()
                            if np.any(np.abs(returns_for_estimation) > 1.0):
                                returns_for_estimation = returns_for_estimation / 100.0
                            if not isinstance(returns_for_estimation, pd.Series):
                                returns_for_estimation = pd.Series(returns_for_estimation, index=_ret.index if hasattr(_ret, 'index') else range(len(returns_for_estimation)))

                            regime_progress = st.progress(0)
                            regime_status = st.empty()
                            regime_status.text(f"Estimating parameters for {num_regimes} regime(s)...")
                            
                            params_dict, moment_info_dict = regime_module.estimate_regime_conditional_parameters(
                                returns_for_estimation,
                                regime_labels,
                                num_regimes=num_regimes,
                                method=estimation_method,
                                bounds=bounds_to_use
                            )
                            

                            regime_progress.progress(1.0)
                            regime_status.text(f"✅ Completed estimation for {len(params_dict)} regime(s)")
                            regime_progress.empty()
                            regime_status.empty()
                            
                            results = {
                                'params_dict': params_dict,
                                'moment_info_dict': moment_info_dict,
                                'regime_labels': regime_labels,
                                'transition_matrix': hmm_results['transition_matrix'],
                                'hmm_model': hmm_results['hmm_model'],
                                'num_regimes': num_regimes,
                                'features': hmm_results['features'],
                                'method': estimation_method
                            }
                        else:

                            if num_regimes == 1:

                                regime_labels = np.zeros(len(returns_data), dtype=int)
                                transition_matrix = np.array([[1.0]])
                            


                                regime_mask = regime_labels == 0
                                if hasattr(returns_data, 'values'):
                                    regime_returns = returns_data[regime_mask].values.copy()
                                else:
                                    regime_returns = np.array(returns_data[regime_mask]).copy()
                                
                                # CRITICAL: Convert percentage log returns to simple log returns if needed
                                # The estimation function expects simple log returns (not multiplied by 100)
                                if np.any(np.abs(regime_returns) > 1.0):
                                    # Likely percentage log returns, divide by 100
                                    regime_returns = regime_returns / 100.0
                            


                                use_sampling = st.session_state.get('use_sampling_estimation', False)
                                sample_size = st.session_state.get('estimation_sample_size', 5000)
                                
                                if use_sampling and len(regime_returns) > sample_size:
                                    np.random.seed(42)
                                    sample_indices = np.random.choice(len(regime_returns), size=sample_size, replace=False)
                                    regime_returns_sample = regime_returns[sample_indices]
                                    st.info(f"📊 Using {sample_size} randomly sampled observations from {len(regime_returns)} total")
                                else:
                                    regime_returns_sample = regime_returns
                            

                            log_file = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor\debug.log'
                            import json
                            import time
                            from scipy import stats
                            try:
                                emp_mean = np.mean(regime_returns_sample)
                                emp_std = np.std(regime_returns_sample)
                                emp_skew = stats.skew(regime_returns_sample)
                                emp_kurt = stats.kurtosis(regime_returns_sample, fisher=True) + 3
                                log_entry = {
                                    "timestamp": int(time.time() * 1000),
                                    "sessionId": "debug-session",
                                    "runId": "pre-opt",
                                    "hypothesisId": "A",
                                    "location": "app.py:1839",
                                    "message": "Before optimization call",
                                    "data": {
                                        "dataset_size": len(regime_returns_sample),
                                        "use_bounds": use_bounds_flag,
                                        "bounds_enabled": use_bounds_flag,
                                        "restarts": 15,
                                        "maxiter": 3000,
                                        "emp_mean": float(emp_mean),
                                        "emp_std": float(emp_std),
                                        "emp_skew": float(emp_skew),
                                        "emp_kurt": float(emp_kurt),
                                        "bounds_lower": [float(b[0]) for b in bounds_to_use] if bounds_to_use else None,
                                        "bounds_upper": [float(b[1]) for b in bounds_to_use] if bounds_to_use else None
                                    }
                                }
                                with open(log_file, 'a', encoding='utf-8') as f:
                                    f.write(json.dumps(log_entry) + '\n')
                            except Exception as log_err:
                                pass

                            

                            try:
                                import sys
                                from io import StringIO
                                

                                old_stdout = sys.stdout
                                old_stderr = sys.stderr
                                captured_output = StringIO()
                                
                                try:

                                    sys.stdout = captured_output
                                    sys.stderr = captured_output
                                    

                                    fit_func = regime_module.fit_bates_moment_matching
                                    


                                    from rich.console import Console
                                    try:
                                        if hasattr(fit_func, '__module__'):
                                            mod_name = fit_func.__module__
                                            if mod_name in sys.modules:
                                                mod = sys.modules[mod_name]
                                                if hasattr(mod, 'console'):

                                                    mod.console = Console(file=captured_output, width=200, force_terminal=False, legacy_windows=False)
                                                    if enable_debug:

                                                        if hasattr(mod, 'DEBUG'):
                                                            mod.DEBUG = True
                                    except Exception as console_err:

                                        captured_output.write(f"Warning: Could not redirect Rich Console: {console_err}\n")
                                    

                                    n_restarts_use = st.session_state.get('param_estimation_restarts', 15)
                                    max_iter_use = st.session_state.get('param_estimation_maxiter', 10000)  # Match standalone script: MAXITER = 10000
                                    


                                    opt_status = st.empty()
                                    opt_status.text(f"Running optimization with {n_restarts_use} restarts (up to {max_iter_use} iterations each)...")
                                    

                                    if enable_debug:
                                        captured_output.write(f"=== DEBUG: Starting parameter estimation ===\n")
                                        captured_output.write(f"Dataset size: {len(regime_returns_sample)}\n")
                                        captured_output.write(f"Restarts: {n_restarts_use}\n")
                                        captured_output.write(f"Max iterations per restart: {max_iter_use}\n")
                                        captured_output.write(f"Use bounds: {use_bounds_flag}\n")

                                        try:
                                            if hasattr(fit_func, '__module__'):
                                                mod_name = fit_func.__module__
                                                if mod_name in sys.modules:
                                                    mod = sys.modules[mod_name]
                                                    if hasattr(mod, 'DEBUG'):
                                                        captured_output.write(f"Moment matching module DEBUG flag: {mod.DEBUG}\n")
                                        except Exception:
                                            pass
                                        captured_output.write("============================================\n\n")
                                    
                                    direct_result = fit_func(
                                        regime_returns_sample,
                                        name="Regime_0",
                                        restarts=n_restarts_use,
                                        maxiter=max_iter_use,
                                        bounds_vec=bounds_to_use,
                                        use_bounds=use_bounds_flag,
                                        match_max_dd=False
                                    )
                                    

                                    opt_status.empty()
                                    
                                    if direct_result is not None:

                                        params_dict = {0: direct_result['params']}
                                        

                                        emp_moments = direct_result.get('emp_moments', {})
                                        model_moments = direct_result.get('model_moments', {})
                                        
                                        moment_info_dict = {
                                            0: {
                                                'n_obs': direct_result.get('n_obs', len(regime_returns_sample)),
                                                'objective': direct_result.get('objective', np.nan),
                                                'emp_mean': emp_moments.get('mean', np.nan),
                                                'emp_std': emp_moments.get('std', np.nan),
                                                'emp_skew': emp_moments.get('skew', np.nan),
                                                'emp_kurt': emp_moments.get('kurt', np.nan),
                                                'model_mean': model_moments.get('mean', np.nan),
                                                'model_std': model_moments.get('std', np.nan),
                                                'model_skew': model_moments.get('skew', np.nan),
                                                'model_kurt': model_moments.get('kurt', np.nan),
                                            }
                                        }
                                        st.success("✅ Parameter estimation succeeded!")
                                    else:

                                        try:
                                            log_entry = {
                                                "timestamp": int(time.time() * 1000),
                                                "sessionId": "debug-session",
                                                "runId": "post-opt-fail",
                                                "hypothesisId": "B",
                                                "location": "app.py:1893",
                                                "message": "Optimization returned None",
                                                "data": {
                                                    "dataset_size": len(regime_returns_sample),
                                                    "result_is_none": direct_result is None,
                                                    "use_bounds": use_bounds_flag
                                                }
                                            }
                                            with open(log_file, 'a', encoding='utf-8') as f:
                                                f.write(json.dumps(log_entry) + '\n')
                                        except Exception:
                                            pass

                                        
                                        st.error("❌ Parameter estimation failed - optimization returned None")
                                        st.write("**Possible reasons:**")
                                        st.write("1. Optimization failed after all restarts")
                                        st.write("2. Data characteristics incompatible with Bates model")
                                        st.write("3. Bounds too restrictive (try disabling bounds)")
                                        st.write("4. Try enabling data sampling for large datasets")
                                        params_dict = {}
                                        moment_info_dict = {}
                                        
                                finally:
                                    sys.stdout = old_stdout
                                    sys.stderr = old_stderr
                                    output_text = captured_output.getvalue()
                                    

                                    import re
                                    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                                    output_text_clean = ansi_escape.sub('', output_text) if output_text else ""
                                    

                                    if output_text_clean and (enable_debug or direct_result is None):
                                        with st.expander("🔍 Estimation Console Output", expanded=enable_debug or direct_result is None):
                                            st.code(output_text_clean, language=None)
                                    elif enable_debug and not output_text_clean:

                                        debug_status = "unknown"
                                        try:
                                            if hasattr(regime_module, 'fit_bates_moment_matching'):
                                                fit_func = regime_module.fit_bates_moment_matching
                                                if hasattr(fit_func, '__module__'):
                                                    mod_name = fit_func.__module__
                                                    if mod_name in sys.modules:
                                                        mod = sys.modules[mod_name]
                                                        if hasattr(mod, 'DEBUG'):
                                                            debug_status = str(mod.DEBUG)
                                        except Exception:
                                            pass
                                        st.warning(f"⚠️ Debug mode enabled but no console output captured. DEBUG flag status: {debug_status}. Rich console output may not be captured by StringIO.")
                                    
                            except Exception as est_err:
                                st.error(f"❌ Error during 1-regime parameter estimation: {str(est_err)}")
                                import traceback
                                with st.expander("Estimation Error Details"):
                                    st.code(traceback.format_exc())
                                params_dict = {}
                                moment_info_dict = {}
                            

                                class DummyHMM:
                                    def __init__(self):
                                        self.means_ = np.array([[0.0]])
                                
                                results = {
                                    'params_dict': params_dict,
                                    'moment_info_dict': moment_info_dict,
                                    'regime_labels': regime_labels,
                                    'transition_matrix': transition_matrix,
                                    'hmm_model': DummyHMM(),
                                    'num_regimes': 1,
                                    'features': hmm_features,
                                    'method': estimation_method
                                }
                            else:
                                # Replicate standalone: log returns, group by regime, fit per regime.
                                # Use HMM results' returns when available (monthly 21-day bucket). Else tab returns.
                                if use_existing_hmm and st.session_state.hmm_results is not None:
                                    _returns_param = st.session_state.hmm_results['returns_data']
                                else:
                                    _returns_param = returns_data
                            # Get returns and convert to log decimal (percent → divide by 100)
                            if hasattr(_returns_param, 'values'):
                                all_returns = _returns_param.values.copy()
                            else:
                                all_returns = np.array(_returns_param).copy()
                            # Percentage log returns → log decimal
                            if np.any(np.abs(all_returns) > 1.0):
                                log_returns = all_returns / 100.0
                            else:
                                log_returns = all_returns
                            
                            # Get regime labels from HMM results (matching standalone script's CSV regime column)
                            if use_existing_hmm and st.session_state.hmm_results is not None:
                                regime_labels = st.session_state.hmm_results['regime_labels']
                            else:
                                # If not using existing HMM, we need to run HMM first (but this path shouldn't be used if use_existing_hmm is True)
                                st.error("❌ Cannot proceed without HMM results. Please run HMM detection first.")
                                return
                            
                            # Align regime labels with returns
                            regime_labels_aligned = regime_labels[:len(log_returns)]
                            
                            # Create DataFrame exactly like standalone script (line 1248-1279)
                            df = pd.DataFrame({
                                'returns': log_returns,
                                'regime': np.array(regime_labels_aligned, dtype=float)
                            })
                            
                            # Group by regime (matching standalone script line 1280)
                            regimes = df.groupby('regime')
                            
                            # Get bounds (matching standalone script lines 1210-1216)
                            bounds_to_use = regime_module.DEFAULT_BOUNDS  # Use default bounds (not option pricing)
                            
                            # Fit each regime using fit_bates_moment_matching directly (matching standalone script lines 1319-1416)
                            results_list = []
                            for regime_id, regime_df in regimes:
                                regime_returns = regime_df['returns'].values.copy()

                                # Regime 3 (extreme crisis): same filtering as standalone
                                if regime_id == 3.0 or regime_id == 3:
                                    cumulative_price = 100.0 * np.cumprod(1 + regime_returns)
                                    running_peak = np.maximum.accumulate(cumulative_price)
                                    drawdown_from_peak = (running_peak - cumulative_price) / running_peak
                                    extreme_threshold = 0.50
                                    extreme_mask = drawdown_from_peak > extreme_threshold
                                    if np.sum(extreme_mask) < 20:
                                        extreme_threshold = 0.30
                                        extreme_mask = drawdown_from_peak > extreme_threshold
                                    if np.sum(extreme_mask) >= 10:
                                        regime_returns = regime_returns[extreme_mask]

                                # Skip if too few observations (matching standalone script line 1366)
                                if len(regime_returns) < 10:
                                    continue

                                # Check if already log returns (matching standalone script line 1372-1379)
                                # NOTE: regime_returns from df['returns'] are already log returns (we set them above)
                                # The standalone script checks again (line 1376-1379) but that's defensive - the DataFrame
                                # already has log returns from line 1262, so this check should never trigger
                                conversion_triggered = False
                                if np.any(np.abs(regime_returns) > 0.5):
                                    # Likely simple returns, convert to log
                                    conversion_triggered = True
                                    regime_returns = np.log(1.0 + regime_returns)
                                
                                # Call fit_bates_moment_matching with same parameters as standalone
                                result = regime_module.fit_bates_moment_matching(
                                    regime_returns,
                                    name=f'regime_{regime_id}',
                                    restarts=regime_module.NUM_RESTARTS,
                                    maxiter=regime_module.MAXITER,
                                    bounds_vec=bounds_to_use,
                                    match_max_dd=False
                                )
                                
                                # Only add if fitting succeeded (matching standalone script line 1398-1399)
                                if result is not None:
                                    results_list.append(result)
                            
                            # Convert results_list to params_dict and moment_info_dict format (for compatibility with display function)
                            params_dict = {}
                            moment_info_dict = {}
                            for r in results_list:
                                regime_id = float(r['name'].split('_')[1])
                                params_dict[regime_id] = r['params']
                                moment_info_dict[regime_id] = {
                                    'n_obs': r['n_obs'],
                                    'emp_mean': r['emp_moments']['mean'],
                                    'model_mean': r['model_moments']['mean'],
                                    'emp_std': r['emp_moments']['std'],
                                    'model_std': r['model_moments']['std'],
                                    'emp_skew': r['emp_moments']['skew'],
                                    'model_skew': r['model_moments']['skew'],
                                    'emp_kurt': r['emp_moments']['kurt'],
                                    'model_kurt': r['model_moments']['kurt'],
                                }

                            # Create results dict matching the format expected by display function
                            results = {
                                'params_dict': params_dict,
                                'moment_info_dict': moment_info_dict,
                                'regime_labels': regime_labels_aligned,
                                'transition_matrix': st.session_state.hmm_results.get('transition_matrix') if use_existing_hmm else None,
                                'hmm_model': st.session_state.hmm_results.get('hmm_model') if use_existing_hmm else None,
                                'num_regimes': len(params_dict),
                                'features': hmm_features if not use_existing_hmm else st.session_state.hmm_results.get('features', 'vol&skew'),
                                'method': estimation_method,
                                'results_list': results_list,  # Store original results_list for visualization
                                'df': df  # Store DataFrame for visualization
                            }
                        
                        st.session_state.param_estimation_results = results
                        st.success("✅ Parameter estimation complete!")
                    else:
                        st.error(f"❌ Could not find estimation module at: {param_model_path}")
            except Exception as e:
                st.error(f"❌ Error during parameter estimation: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    

    if st.session_state.param_estimation_results is not None:
        display_param_estimation_results(st.session_state.param_estimation_results, returns_data)

def compute_regime_stability(regime_labels, transition_matrix, returns_data, num_regimes):
    """
    Compute regime stability metrics:
    - Regime persistence (average duration)
    - Transition stability (consistency of transition probabilities)
    - Feature distribution stability (volatility and skewness within each regime)
    - Regime balance (how evenly distributed regimes are)
    
    Returns:
        dict with stability metrics
    """
    results = {}
    

    regime_series = pd.Series(regime_labels)
    regime_changes = (regime_series != regime_series.shift()).cumsum()
    regime_durations = regime_series.groupby(regime_changes).size()
    

    regime_duration_dict = {i: [] for i in range(num_regimes)}
    current_regime = regime_series.iloc[0]
    current_duration = 0
    
    for regime in regime_series:
        if regime == current_regime:
            current_duration += 1
        else:
            if current_regime < num_regimes:
                regime_duration_dict[current_regime].append(current_duration)
            current_regime = regime
            current_duration = 1

    if current_regime < num_regimes:
        regime_duration_dict[current_regime].append(current_duration)
    

    regime_avg_durations = []
    regime_min_durations = []
    regime_max_durations = []
    regime_std_durations = []
    
    for i in range(num_regimes):
        durations = regime_duration_dict[i] if regime_duration_dict[i] else [0]
        regime_avg_durations.append(np.mean(durations))
        regime_min_durations.append(np.min(durations))
        regime_max_durations.append(np.max(durations))
        regime_std_durations.append(np.std(durations) if len(durations) > 1 else 0.0)
    
    results['regime_avg_durations'] = regime_avg_durations
    results['regime_min_durations'] = regime_min_durations
    results['regime_max_durations'] = regime_max_durations
    results['regime_std_durations'] = regime_std_durations
    results['avg_duration'] = np.mean(regime_avg_durations)
    


    total_days = len(regime_labels)
    expected_random_duration = total_days / num_regimes
    results['persistence_score'] = min(results['avg_duration'] / expected_random_duration, 2.0)
    


    diagonal_sum = np.trace(transition_matrix)
    results['transition_stability'] = diagonal_sum / num_regimes
    

    if hasattr(returns_data, 'index') and len(returns_data) > 0:

        import time, json
        t0_features = time.time()

        returns_df = pd.DataFrame({'returns': returns_data}, index=returns_data.index)
        returns_df['regime'] = regime_labels[:len(returns_df)]
        

        volatility_cvs = []
        skewness_cvs = []
        
        for i in range(num_regimes):

            t0_regime = time.time()

            regime_mask = returns_df['regime'] == i
            regime_returns = returns_df.loc[regime_mask, 'returns']
            
            if len(regime_returns) > 21:

                regime_vol = regime_returns.rolling(window=21, min_periods=1).std() * np.sqrt(252) * 100
                regime_vol = regime_vol.dropna()
                
                if len(regime_vol) > 0 and regime_vol.mean() > 0:
                    vol_cv = regime_vol.std() / regime_vol.mean()
                else:
                    vol_cv = np.nan
                

                if len(regime_returns) > 252:
                    regime_skew = regime_returns.rolling(window=252, min_periods=1).skew()
                    regime_skew = regime_skew.dropna()
                    
                    if len(regime_skew) > 0 and regime_skew.abs().mean() > 0:
                        skew_cv = regime_skew.std() / (regime_skew.abs().mean() + 1e-6)
                    else:
                        skew_cv = np.nan
                else:
                    skew_cv = np.nan
                
                volatility_cvs.append(vol_cv if not np.isnan(vol_cv) else 0.0)
                skewness_cvs.append(skew_cv if not np.isnan(skew_cv) else 0.0)
            else:
                volatility_cvs.append(0.0)
                skewness_cvs.append(0.0)

            t1_regime = time.time()
            try:
                log_path = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor\debug.log'
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'id': 'perf_regime_features', 'timestamp': time.time() * 1000, 'location': 'app.py:1695', 'message': f'Regime {i} feature calculation', 'data': {'regime': i, 'duration_sec': t1_regime - t0_regime, 'regime_length': len(regime_returns)}, 'sessionId': 'debug-session', 'runId': 'perf-1', 'hypothesisId': 'D'}) + '\n')
            except: pass


        t1_features = time.time()
        try:
            log_path = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor\debug.log'
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'perf_features_total', 'timestamp': time.time() * 1000, 'location': 'app.py:1726', 'message': 'Total feature calculation timing', 'data': {'duration_sec': t1_features - t0_features, 'num_regimes': num_regimes}, 'sessionId': 'debug-session', 'runId': 'perf-1', 'hypothesisId': 'D'}) + '\n')
        except: pass

        
        results['volatility_cvs'] = volatility_cvs
        results['skewness_cvs'] = skewness_cvs
    else:
        results['volatility_cvs'] = [0.0] * num_regimes
        results['skewness_cvs'] = [0.0] * num_regimes
    


    regime_counts = pd.Series(regime_labels).value_counts().sort_index()
    regime_proportions = regime_counts / len(regime_labels)
    

    entropy = -np.sum(regime_proportions * np.log(regime_proportions + 1e-10))
    max_entropy = np.log(num_regimes)
    results['balance_score'] = entropy / max_entropy
    
    return results

def display_hmm_results(hmm_results):
    """Display HMM regime detection results"""
    st.subheader("📋 HMM Results")

    regime_labels = hmm_results['regime_labels']
    transition_matrix = hmm_results['transition_matrix']
    num_regimes = hmm_results['num_regimes']
    returns_data = hmm_results['returns_data']  # monthly (used for HMM)

    # Plot at daily resolution when we have original (daily) data: expand monthly regime to daily index
    returns_data_display = returns_data
    regime_labels_display = regime_labels
    original = getattr(st.session_state, '_returns_original_for_plot', None)
    if (original is not None and hasattr(original, 'index') and isinstance(original.index, pd.DatetimeIndex)
        and len(original) > len(regime_labels) and hasattr(returns_data, 'index') and isinstance(returns_data.index, pd.DatetimeIndex)):
        # Map (year, month) -> regime from monthly data
        month_to_regime = {}
        for i, t in enumerate(returns_data.index):
            month_to_regime[(t.year, t.month)] = regime_labels[i]
        # For each daily date, use regime of its month (or nearest prior month)
        daily_regimes = []
        for d in original.index:
            key = (d.year, d.month)
            if key in month_to_regime:
                daily_regimes.append(month_to_regime[key])
            else:
                # Fallback: use first regime if date is before our monthly index
                daily_regimes.append(regime_labels[0] if len(regime_labels) else 0)
        regime_labels_display = np.array(daily_regimes)
        returns_data_display = original
        st.caption("📈 **Plots use daily resolution** (analysis was done on monthly returns).")

    portfolio_name = "Portfolio"
    if hasattr(st.session_state, 'config') and st.session_state.config is not None:
        if hasattr(st.session_state.config, 'portfolio_column_name') and st.session_state.config.portfolio_column_name:
            portfolio_name = st.session_state.config.portfolio_column_name
    

    if hasattr(returns_data_display, 'index') and len(returns_data_display) > 0:
        st.caption(f"📅 Date range: {returns_data_display.index.min()} to {returns_data_display.index.max()}")
    

    st.write("**Regime Distribution:**")
    regime_counts = pd.Series(regime_labels).value_counts().sort_index()
    regime_pcts = (regime_counts / len(regime_labels) * 100).round(2)
    
    dist_data = {
        'Regime': [f"Regime {i}" for i in range(num_regimes)],
        'Count': [regime_counts.get(i, 0) for i in range(num_regimes)],
        'Percentage': [f"{regime_pcts.get(i, 0):.2f}%" for i in range(num_regimes)]
    }
    dist_df = pd.DataFrame(dist_data)
    st.dataframe(dist_df, use_container_width=True, hide_index=True)
    

    st.write("**Transition Matrix:**")
    trans_df = pd.DataFrame(
        transition_matrix,
        index=[f"From Regime {i}" for i in range(num_regimes)],
        columns=[f"To Regime {j}" for j in range(num_regimes)]
    )

    trans_df = trans_df.round(2)
    st.dataframe(trans_df, use_container_width=True)
    

    st.write("**📊 Regime Stability Analysis:**")

    import time, json
    t0_stability = time.time()

    if '_stability_cache' not in hmm_results:
        hmm_results['_stability_cache'] = compute_regime_stability(regime_labels, transition_matrix, returns_data, num_regimes)
    stability_results = hmm_results['_stability_cache']

    t1_stability = time.time()
    try:
        log_path = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor\debug.log'
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'id': 'perf_stability', 'timestamp': time.time() * 1000, 'location': 'app.py:1784', 'message': 'compute_regime_stability timing', 'data': {'duration_sec': t1_stability - t0_stability, 'num_regimes': num_regimes, 'data_length': len(regime_labels)}, 'sessionId': 'debug-session', 'runId': 'perf-1', 'hypothesisId': 'A'}) + '\n')
    except: pass

    

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Regime Duration", f"{stability_results['avg_duration']:.1f} days")
    with col2:
        st.metric("Regime Persistence Score", f"{stability_results['persistence_score']:.2f}")
    with col3:
        st.metric("Transition Stability", f"{stability_results['transition_stability']:.2f}")
    

    with st.expander("📈 Detailed Stability Metrics", expanded=False):
        st.write("**Regime Duration Statistics (days):**")
        duration_df = pd.DataFrame({
            'Regime': [f"Regime {i}" for i in range(num_regimes)],
            'Avg Duration': [f"{d:.1f}" for d in stability_results['regime_avg_durations']],
            'Min Duration': [f"{d:.1f}" for d in stability_results['regime_min_durations']],
            'Max Duration': [f"{d:.1f}" for d in stability_results['regime_max_durations']],
            'Std Duration': [f"{d:.1f}" for d in stability_results['regime_std_durations']]
        })
        st.dataframe(duration_df, use_container_width=True, hide_index=True)
        
        st.write("**Feature Distribution Stability (within each regime):**")
        feature_stability_df = pd.DataFrame({
            'Regime': [f"Regime {i}" for i in range(num_regimes)],
            'Volatility CV': [f"{cv:.3f}" for cv in stability_results['volatility_cvs']],
            'Skewness CV': [f"{cv:.3f}" for cv in stability_results['skewness_cvs']]
        })
        st.dataframe(feature_stability_df, use_container_width=True, hide_index=True)
        st.caption("CV = Coefficient of Variation (std/mean). Lower CV indicates more stable distribution.")
        
        st.write("**Regime Balance:**")
        balance_score = stability_results['balance_score']
        st.progress(balance_score, text=f"Balance Score: {balance_score:.2f} (1.0 = perfectly balanced)")
        st.caption("Measures how evenly distributed regimes are. Higher is better.")
    

    st.write("**📊 Regime Characteristics (Means and Risk Metrics):**")
    if 'hmm_model' in hmm_results and hmm_results['hmm_model'] is not None:
        hmm_model = hmm_results['hmm_model']
        features_used = hmm_results.get('features', 'vol')
        

        def generate_gradient_colors(n, start_color=(0, 255, 0), end_color=(255, 0, 0)):
            """Generate n colors in a gradient from start_color (green) to end_color (red)"""
            if n == 1:
                return ['#00ff00']
            colors = []
            for i in range(n):

                ratio = i / (n - 1) if n > 1 else 0
                r = int(start_color[0] + (end_color[0] - start_color[0]) * ratio)
                g = int(start_color[1] + (end_color[1] - start_color[1]) * ratio)
                b = int(start_color[2] + (end_color[2] - start_color[2]) * ratio)

                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                colors.append(f'#{r:02x}{g:02x}{b:02x}')
            return colors
        
        regime_colors = generate_gradient_colors(num_regimes)
        

        def calculate_max_drawdown(daily_returns_series):
            """Calculates Max Drawdown from a series of daily returns (in percent)."""
            if len(daily_returns_series) == 0:
                return 0.0

            simple_returns = np.exp(daily_returns_series / 100) - 1 if daily_returns_series.abs().max() > 1 else daily_returns_series / 100

            cumulative_returns = (1 + simple_returns).cumprod()

            running_max = cumulative_returns.cummax()

            drawdown = (cumulative_returns / running_max) - 1

            return drawdown.min() * 100
        

        hmm_means = hmm_model.means_
        TRADING_DAYS_YEAR = 252
        

        index_map = hmm_results.get('regime_index_map', {i: i for i in range(num_regimes)})

        reverse_map = {v: k for k, v in index_map.items()}
        

        regime_stats = []
        returns_df = pd.DataFrame({'returns': returns_data_display}, index=returns_data_display.index)
        returns_df['regime'] = regime_labels_display[:len(returns_df)]
        
        for regime_idx in range(num_regimes):

            original_idx = reverse_map.get(regime_idx, regime_idx)
            regime_mask = returns_df['regime'] == regime_idx
            regime_returns = returns_df.loc[regime_mask, 'returns']
            
            if len(regime_returns) > 0:

                annual_nominal_return = regime_returns.mean() * TRADING_DAYS_YEAR
                

                max_drawdown = calculate_max_drawdown(regime_returns)
            else:
                annual_nominal_return = 0.0
                max_drawdown = 0.0
            

            if features_used == 'vol':
                feature_mean = hmm_means[original_idx, 0]

                percentile = int((regime_idx / num_regimes) * 100) if num_regimes > 1 else 0
                regime_label = f"Regime {regime_idx}: Vol={feature_mean:.1f}% (Rank {regime_idx+1}/{num_regimes})"
                stats_row = {
                    'Index': regime_idx,
                    'Regime Label': regime_label,
                    'FEATURE: Annual Volatility (%) [HMM MEAN]': f"{feature_mean:.2f}",
                    'Annual Nominal Return (%) (Derived)': f"{annual_nominal_return:.2f}",
                    'Max Drawdown (%)': f"{max_drawdown:.2f}"
                }
            elif features_used == 'skew':
                feature_mean = hmm_means[original_idx, 0]
                skew_window_days = 252
                regime_label = f"Regime {regime_idx}: Skew={feature_mean:.2f} (Rank {regime_idx+1}/{num_regimes})"
                stats_row = {
                    'Index': regime_idx,
                    'Regime Label': regime_label,
                    f'FEATURE: {skew_window_days}-Day Skewness [HMM MEAN]': f"{feature_mean:.2f}",
                    'Annual Nominal Return (%) (Derived)': f"{annual_nominal_return:.2f}",
                    'Max Drawdown (%)': f"{max_drawdown:.2f}"
                }
            else:
                vol_mean = hmm_means[original_idx, 0]
                skew_mean = hmm_means[original_idx, 1]
                regime_label = f"Regime {regime_idx}: Vol={vol_mean:.1f}%, Skew={skew_mean:.2f} (Rank {regime_idx+1}/{num_regimes})"
                stats_row = {
                    'Index': regime_idx,
                    'Regime Label': regime_label,
                    'FEATURE: Volatility [HMM MEAN]': f"{vol_mean:.2f}",
                    'FEATURE: Skewness [HMM MEAN]': f"{skew_mean:.2f}",
                    'Annual Nominal Return (%) (Derived)': f"{annual_nominal_return:.2f}",
                    'Max Drawdown (%)': f"{max_drawdown:.2f}"
                }
            
            regime_stats.append(stats_row)
        

        stats_df = pd.DataFrame(regime_stats)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    else:
        st.info("HMM model not available for regime characteristics calculation.")
    

    # Show matplotlib version of volatility/skewness and price plots (fast, always shown)
    if hasattr(returns_data, 'index') and HAS_MATPLOTLIB:
        features_used = hmm_results.get('features', 'vol')
        
        # Prepare data for plotting
        returns_df_daily = pd.DataFrame({'returns': returns_data_display}, index=returns_data_display.index)
        
        # Calculate volatility if needed
        if features_used == 'vol' or features_used == 'vol&skew' or features_used == 'both':
            returns_df_daily['volatility'] = returns_df_daily['returns'].rolling(window=21).std() * np.sqrt(252)
        
        # Calculate skewness if needed
        if features_used == 'skew' or features_used == 'vol&skew' or features_used == 'both':
            returns_df_daily['skewness'] = returns_df_daily['returns'].rolling(window=252, min_periods=1).skew()
        
        # Calculate price from returns
        returns_decimal = returns_df_daily['returns'] / 100.0 if returns_df_daily['returns'].abs().max() > 1 else returns_df_daily['returns']
        daily_price = 10000 * (1 + returns_decimal).cumprod()
        returns_df_daily['price'] = daily_price
        
        # Align regime labels with data
        returns_df_daily['regime'] = regime_labels_display[:len(returns_df_daily)]
        
        # Determine which features to plot and check if we have data
        plot_df = None
        has_vol = False
        has_skew = False
        
        if features_used == 'skew':
            if 'skewness' in returns_df_daily.columns and returns_df_daily['skewness'].notna().sum() > 0:
                has_skew = True
                plot_df = returns_df_daily[returns_df_daily['skewness'].notna()].copy()
        elif features_used == 'vol':
            if 'volatility' in returns_df_daily.columns and returns_df_daily['volatility'].notna().sum() > 0:
                has_vol = True
                plot_df = returns_df_daily[returns_df_daily['volatility'].notna()].copy()
        elif features_used == 'vol&skew' or features_used == 'both':
            # Need both features for vol&skew
            has_vol = 'volatility' in returns_df_daily.columns and returns_df_daily['volatility'].notna().sum() > 0
            has_skew = 'skewness' in returns_df_daily.columns and returns_df_daily['skewness'].notna().sum() > 0
            if has_vol and has_skew:
                # Filter to where both are not null
                plot_df = returns_df_daily[
                    returns_df_daily['volatility'].notna() & returns_df_daily['skewness'].notna()
                ].copy()
            elif has_vol:
                plot_df = returns_df_daily[returns_df_daily['volatility'].notna()].copy()
                has_skew = False
            elif has_skew:
                plot_df = returns_df_daily[returns_df_daily['skewness'].notna()].copy()
                has_vol = False
        
        # Only plot if we have valid feature data
        if plot_df is not None and len(plot_df) > 0:
            # Determine section title
            if features_used == 'skew':
                st.write("**Skewness & Asset Price with Inferred Regimes:**")
            elif features_used == 'vol':
                st.write("**Volatility & Asset Price with Inferred Regimes:**")
            else:
                st.write("**Volatility & Skewness & Asset Price with Inferred Regimes:**")
            
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
            from matplotlib.lines import Line2D
            import matplotlib.dates as mdates
            # Avoid mathtext ParseException with log scale (unicode minus / font issues)
            _rc_unicode = plt.rcParams.get('axes.unicode_minus', True)
            plt.rcParams['axes.unicode_minus'] = False
            
            # Define colors for regimes (matching the standalone scripts)
            regime_colors = ['#00BFFF', '#FFD700', '#FF4500']  # Blue, Gold, Orange
            if num_regimes > 3:
                regime_colors.extend(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
            
            # Ensure regime labels are aligned with filtered data and are integers
            if 'regime' in plot_df.columns:
                plot_df['regime'] = plot_df['regime'].fillna(0).astype(int)
            
            # Determine number of subplots
            if has_vol and has_skew:
                # 3 plots: vol, skew, price
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), facecolor='black', sharex=True)
                ax1.set_facecolor('black')
                ax2.set_facecolor('black')
                ax3.set_facecolor('black')
            else:
                # 2 plots: feature, price
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor='black', sharex=True)
                ax1.set_facecolor('black')
                ax2.set_facecolor('black')
                ax3 = None
            
            # Prepare x-axis as numeric for LineCollection
            x_numeric = np.arange(len(plot_df))
            
            # Helper function to plot colored line segments by regime
            def plot_colored_line(ax, x_data, y_data, regime_data, linewidth=1.5):
                """Plot a line with segments colored by regime."""
                # Create line segments
                points = np.array([x_data, y_data]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Create color array based on regime
                segment_colors = []
                for i in range(len(regime_data) - 1):
                    regime = regime_data.iloc[i]
                    if pd.notna(regime):
                        regime_index = int(regime)
                        if 0 <= regime_index < len(regime_colors):
                            segment_colors.append(regime_colors[regime_index])
                        else:
                            segment_colors.append('gray')
                    else:
                        segment_colors.append('gray')
                
                # Create LineCollection
                lc = LineCollection(segments, colors=segment_colors, linewidth=linewidth, alpha=0.9)
                ax.add_collection(lc)
                ax.autoscale()
            
            # Set x-axis ticks to show dates (used by all plots)
            n_ticks = min(10, len(plot_df))
            tick_indices = np.linspace(0, len(plot_df) - 1, n_ticks, dtype=int)
            plot_idx = 0
            
            # --- PLOT 1: Volatility (if available) ---
            if has_vol:
                plot_colored_line(ax1, x_numeric, plot_df['volatility'].values, plot_df['regime'], linewidth=1.5)
                ax1.set_ylabel('Annualized Volatility (%)', color='white', fontsize=12)
                ax1.tick_params(axis='y', labelcolor='white')
                ax1.grid(True, alpha=0.3, color='gray')
                ax1.set_title('Volatility Over Time with Inferred Regimes', color='white', fontsize=14, fontweight='bold')
                ax1.set_xticks(tick_indices)
                ax1.set_xticklabels([plot_df.index[i].strftime('%Y') for i in tick_indices], rotation=45, color='white')
                plot_idx += 1
            
            # --- PLOT 2: Skewness (if available, or feature if only one) ---
            if has_skew:
                target_ax = ax2 if has_vol else ax1
                plot_colored_line(target_ax, x_numeric, plot_df['skewness'].values, plot_df['regime'], linewidth=1.5)
                target_ax.set_ylabel('12-Month Rolling Skewness', color='white', fontsize=12)
                target_ax.tick_params(axis='y', labelcolor='white')
                target_ax.grid(True, alpha=0.3, color='gray')
                target_ax.set_title('Skewness Over Time with Inferred Regimes', color='white', fontsize=14, fontweight='bold')
                target_ax.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
                target_ax.set_xticks(tick_indices)
                target_ax.set_xticklabels([plot_df.index[i].strftime('%Y') for i in tick_indices], rotation=45, color='white')
                plot_idx += 1
            elif not has_vol:
                # Only volatility, already plotted in ax1
                pass
            
            # --- PLOT 3: Price (Log Scale) ---
            price_ax = ax3 if (has_vol and has_skew) else ax2
            plot_colored_line(price_ax, x_numeric, plot_df['price'].values, plot_df['regime'], linewidth=1.5)
            price_ax.set_yscale('log')
            # Use ScalarFormatter so log axis tick labels are plain numbers (avoids mathtext ParseException)
            try:
                from matplotlib.ticker import ScalarFormatter
                _fmt = ScalarFormatter()
                _fmt.set_scientific(False)
                price_ax.yaxis.set_major_formatter(_fmt)
            except Exception: pass
            price_ax.set_ylabel('Price (Log Scale)', color='white', fontsize=12)
            price_ax.set_xlabel('Date', color='white', fontsize=12)
            price_ax.tick_params(axis='y', labelcolor='white')
            price_ax.tick_params(axis='x', labelcolor='white')
            price_ax.grid(True, alpha=0.3, color='gray', which='both')
            price_ax.set_title('Asset Price Over Time with Inferred Regimes (Log Scale)', color='white', fontsize=14, fontweight='bold')
            price_ax.set_xticks(tick_indices)
            price_ax.set_xticklabels([plot_df.index[i].strftime('%Y') for i in tick_indices], rotation=45, color='white')
            
            # Add regime legend to the price plot
            legend_elements = [Line2D([0], [0], color=regime_colors[i % len(regime_colors)], linewidth=3,
                                     label=f"Regime {i}")
                              for i in range(num_regimes)]
            price_ax.legend(handles=legend_elements, loc='upper left', facecolor='black', 
                          edgecolor='white', framealpha=0.8, labelcolor='white')
            try:
                plt.tight_layout()
                st.pyplot(fig)
            finally:
                plt.close(fig)
                plt.rcParams['axes.unicode_minus'] = _rc_unicode
    
    # Optional Plotly charts (slower, checkbox-controlled)
    if hasattr(returns_data, 'index') and HAS_PLOTLY:
        show_charts = st.checkbox(
            "Load interactive regime charts (may be slow with long series)",
            value=False,
            key="hmm_show_charts",
            help="Toggle to load the interactive Plotly regime overlay and rolling metric charts. Kept off by default for faster app response."
        )
        if not show_charts:
            st.caption("Check the box above to load interactive regime overlay and rolling metric charts.")
        else:
            import time, json
            t0_viz_section = time.time()

            features_used = hmm_results.get('features', 'vol')
            
            feature_labels = {
                'vol': 'Volatility',
                'skew': 'Skewness',
                'vol&skew': 'Volatility & Skewness',
                'both': 'Volatility & Skewness'
            }
            feature_label = feature_labels.get(features_used, features_used.capitalize() if isinstance(features_used, str) else str(features_used))
            
            st.write(f"**{feature_label} & Asset Price with Inferred Regimes:**")
            
            import hashlib
            rl_hash = hashlib.md5(np.asarray(regime_labels).tobytes()).hexdigest()[:12]
            cache_version = "v2_colors"
            cache_key = f"hmm_viz_{len(returns_data)}_{rl_hash}_{cache_version}"
        

            def create_hmm_plot(use_regime_colors, use_regime_shading, use_log_scale, use_vol_log_scale, returns_df, dates, regime_array, segment_starts, num_regimes, regime_colors, features_used):
                price_scale_label = "Log Scale" if use_log_scale else "Cumulative Scale"
                feature_scale_label = "Log Scale" if use_vol_log_scale else "Linear Scale"
            

                use_secondary_y = False

                portfolio_name = hmm_results.get('portfolio_name', 'Portfolio')
                if features_used == 'skew':
                    top_title = f'{portfolio_name} - Skewness Over Time with Inferred Regimes ({feature_scale_label})'
                    feature_col = 'skewness'
                    feature_label = '252-Day Rolling Skewness'
                    secondary_feature_col = None
                    secondary_feature_label = None
                elif features_used == 'vol':
                    top_title = f'{portfolio_name} - Volatility Over Time with Inferred Regimes ({feature_scale_label})'
                    feature_col = 'volatility'
                    feature_label = 'Annualized Volatility (%)'
                    secondary_feature_col = None
                    secondary_feature_label = None
                else:
                    top_title = f'{portfolio_name} - Volatility & Skewness Over Time with Inferred Regimes ({feature_scale_label})'
                    feature_col = 'volatility'
                    feature_label = 'Annualized Volatility (%)'
                    secondary_feature_col = 'skewness'
                    secondary_feature_label = '252-Day Rolling Skewness'
                    use_secondary_y = True

                from plotly.subplots import make_subplots
                if use_secondary_y:
                    portfolio_name = hmm_results.get('portfolio_name', 'Portfolio')
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=(top_title, 
                                      f'{portfolio_name} - Asset Price Over Time with Inferred Regimes ({price_scale_label})'),
                        vertical_spacing=0.08,
                        row_heights=[0.5, 0.5],
                        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
                    )
                else:
                    portfolio_name = hmm_results.get('portfolio_name', 'Portfolio')
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=(top_title, 
                                      f'{portfolio_name} - Asset Price Over Time with Inferred Regimes ({price_scale_label})'),
                        vertical_spacing=0.08,
                        row_heights=[0.5, 0.5]
                    )
            


            

                num_segments = len(segment_starts) - 1
                segment_regimes = []
                if use_regime_colors or use_regime_shading:
                    segment_durations = []
                    for seg_idx in range(num_segments):
                        start_idx = segment_starts[seg_idx]
                        end_idx = segment_starts[seg_idx + 1] if seg_idx + 1 < len(segment_starts) else len(dates)
                        duration = end_idx - start_idx
                        segment_durations.append(duration)
                        if start_idx < len(regime_array):
                            regime_val = regime_array[start_idx]
                            segment_regimes.append(int(regime_val) if not pd.isna(regime_val) else None)
                        else:
                            segment_regimes.append(None)
            

                if use_regime_shading:


                    feature_max = returns_df[feature_col].max() * 1.1 if feature_col in returns_df.columns else 100
                    price_min = returns_df['price'].min() * 0.5
                    price_max = returns_df['price'].max() * 2
                

                    combined_regions = []
                    current_regime = None
                    current_start = None
                
                    for seg_idx in range(num_segments):
                        regime = segment_regimes[seg_idx]
                        start_idx = segment_starts[seg_idx]
                        end_idx = segment_starts[seg_idx + 1] if seg_idx + 1 < len(segment_starts) else len(dates)
                    
                        if regime is not None:
                            regime = int(regime)
                            if regime == current_regime:

                                pass
                            else:

                                if current_regime is not None and current_start is not None:

                                    combined_regions.append((current_regime, current_start, start_idx))
                                current_regime = regime
                                current_start = start_idx
                        else:

                            if current_regime is not None and current_start is not None:
                                combined_regions.append((current_regime, current_start, start_idx))
                                current_regime = None
                                current_start = None
                

                    if current_regime is not None and current_start is not None:
                        combined_regions.append((current_regime, current_start, len(dates)))
                

                    for regime, start_idx, end_idx in combined_regions:
                        if 0 <= regime < num_regimes and start_idx < len(dates) and end_idx <= len(dates):
                            color = regime_colors[regime % len(regime_colors)]
                            x0, x1 = dates[start_idx], dates[end_idx-1] if end_idx > start_idx else dates[start_idx]
                        
                            fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=feature_max,
                                         fillcolor=color, opacity=0.15, layer="below", line_width=0, row=1, col=1)
                            fig.add_shape(type="rect", x0=x0, x1=x1, y0=price_min, y1=price_max,
                                         fillcolor=color, opacity=0.15, layer="below", line_width=0, row=2, col=1)
            

                if use_regime_colors:



                    regime_traces_feature = {r: {'x': [], 'y': [], 'last_idx': -1} for r in range(num_regimes)}
                    regime_traces_secondary = {r: {'x': [], 'y': [], 'last_idx': -1} for r in range(num_regimes)} if use_secondary_y else {}
                    regime_traces_price = {r: {'x': [], 'y': [], 'last_idx': -1} for r in range(num_regimes)}
                
                    for i in range(len(dates)):
                        regime = regime_array[i] if i < len(regime_array) else None
                        if regime is not None and 0 <= regime < num_regimes and not np.isnan(regime):

                            if regime_traces_feature[regime]['last_idx'] >= 0 and i > regime_traces_feature[regime]['last_idx'] + 1:

                                regime_traces_feature[regime]['x'].append(None)
                                regime_traces_feature[regime]['y'].append(None)
                                if use_secondary_y:
                                    regime_traces_secondary[regime]['x'].append(None)
                                    regime_traces_secondary[regime]['y'].append(None)
                                regime_traces_price[regime]['x'].append(None)
                                regime_traces_price[regime]['y'].append(None)
                        
                            regime_traces_feature[regime]['x'].append(dates[i])
                            regime_traces_feature[regime]['y'].append(returns_df[feature_col].iloc[i] if feature_col in returns_df.columns else 0)
                            if use_secondary_y and secondary_feature_col in returns_df.columns:
                                regime_traces_secondary[regime]['x'].append(dates[i])
                                regime_traces_secondary[regime]['y'].append(returns_df[secondary_feature_col].iloc[i])
                            regime_traces_price[regime]['x'].append(dates[i])
                            regime_traces_price[regime]['y'].append(returns_df['price'].iloc[i])
                            regime_traces_feature[regime]['last_idx'] = i
                            if use_secondary_y:
                                regime_traces_secondary[regime]['last_idx'] = i
                            regime_traces_price[regime]['last_idx'] = i
                

                    for regime in range(num_regimes):
                        if len(regime_traces_feature[regime]['x']) > 0:
                            color = regime_colors[regime % len(regime_colors)]
                            hover_label = 'Volatility' if features_used != 'skew' else 'Skewness'
                            hover_unit = '%' if features_used != 'skew' else ''
                            fig.add_trace(
                                go.Scatter(x=regime_traces_feature[regime]['x'], y=regime_traces_feature[regime]['y'], 
                                          mode='lines', name=f'Regime {regime}',
                                          line=dict(color=color, width=2), showlegend=False,
                                          legendgroup='feature_regimes',
                                          hovertemplate=f'Date: %{{x}}<br>{hover_label}: %{{y:.2f}}{hover_unit}<extra></extra>'),
                                row=1, col=1, secondary_y=False
                            )
                        if use_secondary_y and len(regime_traces_secondary[regime]['x']) > 0:
                            color = regime_colors[regime % len(regime_colors)]
                            fig.add_trace(
                                go.Scatter(x=regime_traces_secondary[regime]['x'], y=regime_traces_secondary[regime]['y'], 
                                          mode='lines', name=f'Regime {regime} Skew',
                                          line=dict(color=color, width=2, dash='dash'), showlegend=False,
                                          legendgroup='secondary_regimes',
                                          hovertemplate=f'Date: %{{x}}<br>Skewness: %{{y:.2f}}<extra></extra>'),
                                row=1, col=1, secondary_y=True
                            )
                        if len(regime_traces_price[regime]['x']) > 0:
                            color = regime_colors[regime % len(regime_colors)]
                            fig.add_trace(
                                go.Scatter(x=regime_traces_price[regime]['x'], y=regime_traces_price[regime]['y'], 
                                          mode='lines', name=f'Regime {regime}',
                                          line=dict(color=color, width=2), showlegend=False,
                                          legendgroup='price_regimes',
                                          hovertemplate='Date: %{x}<br>Price: %{y:,.0f}<extra></extra>'),
                                row=2, col=1
                            )
                else:

                    hover_label = 'Volatility' if features_used != 'skew' else 'Skewness'
                    hover_unit = '%' if features_used != 'skew' else ''
                    fig.add_trace(
                        go.Scatter(x=dates, y=returns_df[feature_col] if feature_col in returns_df.columns else [0]*len(dates), mode='lines',
                                  name=feature_label,
                                  line=dict(color='cyan', width=1.5),
                                  hovertemplate=f'Date: %{{x}}<br>{hover_label}: %{{y:.2f}}{hover_unit}<extra></extra>'),
                        row=1, col=1, secondary_y=False
                    )
                    if use_secondary_y and secondary_feature_col in returns_df.columns:
                        fig.add_trace(
                            go.Scatter(x=dates, y=returns_df[secondary_feature_col], mode='lines',
                                      name=secondary_feature_label,
                                      line=dict(color='magenta', width=1.5, dash='dash'),
                                      hovertemplate='Date: %{x}<br>Skewness: %{y:.2f}<extra></extra>'),
                            row=1, col=1, secondary_y=True
                        )
                    fig.add_trace(
                        go.Scatter(x=dates, y=returns_df['price'], mode='lines',
                                  name='Asset Price',
                                  line=dict(color='white', width=1.5),
                                  hovertemplate='Date: %{x}<br>Price: %{y:,.0f}<extra></extra>'),
                        row=2, col=1
                    )
            
            

                if use_regime_colors:
                    for i in range(num_regimes):
                        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                                marker=dict(size=12, color=regime_colors[i % len(regime_colors)], 
                                                           symbol='circle', line=dict(width=0)),
                                                name=f"Regime {i}", showlegend=True, legendgroup="regimes"))
            

                fig.update_layout(
                    title=f'HMM Regime Detection - {num_regimes} Regimes',
                    height=800, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=12),
                    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=0,
                               bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', borderwidth=0)
                )
            

                fig.update_xaxes(title_text="Date", row=2, col=1)
                if use_vol_log_scale:
                    fig.update_yaxes(title_text=f"{feature_label} - Log Scale", type="log", row=1, col=1, secondary_y=False)
                else:
                    fig.update_yaxes(title_text=feature_label, row=1, col=1, secondary_y=False)
                if use_secondary_y:
                    fig.update_yaxes(title_text=secondary_feature_label, row=1, col=1, secondary_y=True)
                if use_log_scale:
                    fig.update_yaxes(title_text="Price (Log Scale)", type="log", row=2, col=1)
                else:
                    fig.update_yaxes(title_text="Price (Cumulative)", row=2, col=1)
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)', row=1, col=1)
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)', row=2, col=1)
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)', row=1, col=1)
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)', row=2, col=1)
            
                return fig
        

            plots_cache_key = f"{cache_key}_plots"
        

            if plots_cache_key not in st.session_state:

                returns_df_daily = pd.DataFrame({'returns': returns_data_display}, index=returns_data_display.index)
            

                if features_used == 'vol' or features_used == 'vol&skew' or features_used == 'both':

                    returns_df_daily['volatility'] = returns_df_daily['returns'].rolling(window=21).std() * np.sqrt(252)
            
                if features_used == 'skew' or features_used == 'vol&skew' or features_used == 'both':

                    returns_df_daily['skewness'] = returns_df_daily['returns'].rolling(window=252, min_periods=1).skew()
            


                returns_decimal = returns_df_daily['returns'] / 100.0 if returns_df_daily['returns'].abs().max() > 1 else returns_df_daily['returns']
            

                daily_price = 10000 * (1 + returns_decimal).cumprod()
            

                daily_returns_pct = returns_decimal * 100 if returns_df_daily['returns'].abs().max() > 1 else returns_decimal
            

                df_dict = {
                    'returns': daily_returns_pct,
                    'price': daily_price
                }
                if 'volatility' in returns_df_daily.columns:
                    df_dict['volatility'] = returns_df_daily['volatility']
                if 'skewness' in returns_df_daily.columns:
                    df_dict['skewness'] = returns_df_daily['skewness']
            
                returns_df = pd.DataFrame(df_dict, index=returns_df_daily.index)
            


                dates = returns_df.index
                regime_array = np.array(regime_labels_display[:len(dates)]) if len(regime_labels_display) >= len(dates) else np.array(list(regime_labels_display) + [regime_labels_display[-1]] * (len(dates) - len(regime_labels_display)))
                regime_changes = np.diff(regime_array, prepend=regime_array[0]) != 0
                segment_starts = np.where(regime_changes)[0].tolist()
                segment_starts.insert(0, 0)
                segment_starts.append(len(dates))
            


                def generate_gradient_colors(n, start_color=(0, 255, 0), end_color=(255, 0, 0)):
                    """Generate n colors in a gradient from start_color (green) to end_color (red)"""
                    if n == 1:
                        return ['#00ff00']
                    colors = []
                    for i in range(n):

                        ratio = i / (n - 1) if n > 1 else 0
                        r = int(start_color[0] + (end_color[0] - start_color[0]) * ratio)
                        g = int(start_color[1] + (end_color[1] - start_color[1]) * ratio)
                        b = int(start_color[2] + (end_color[2] - start_color[2]) * ratio)

                        r = max(0, min(255, r))
                        g = max(0, min(255, g))
                        b = max(0, min(255, b))
                        colors.append(f'#{r:02x}{g:02x}{b:02x}')
                    return colors
            
                regime_colors = generate_gradient_colors(num_regimes)
            

                try:
                    log_path = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor\debug.log'
                    import json, time
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({'id': 'debug_regime_colors_set', 'timestamp': time.time() * 1000, 'location': 'app.py:2259', 'message': 'Regime colors set', 'data': {'num_regimes': num_regimes, 'regime_colors': regime_colors}, 'sessionId': 'debug-session', 'runId': 'perf-1', 'hypothesisId': 'D'}) + '\n')
                except: pass

            

                st.session_state[plots_cache_key] = {
                    'returns_df': returns_df,
                    'dates': dates,
                    'regime_array': regime_array,
                    'segment_starts': segment_starts,
                    'num_regimes': num_regimes,
                    'regime_colors': regime_colors
                }

                t1_data_prep = time.time()
                try:
                    log_path = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor\debug.log'
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({'id': 'perf_data_prep_total', 'timestamp': time.time() * 1000, 'location': 'app.py:1998', 'message': 'Total data preparation', 'data': {'duration_sec': t1_data_prep - t0_data_prep, 'data_length': len(returns_data)}, 'sessionId': 'debug-session', 'runId': 'perf-1', 'hypothesisId': 'B'}) + '\n')
                except: pass

        

            use_regime_colors = st.session_state.get(f"regime_colors_{cache_key}", True)
            use_regime_shading = st.session_state.get(f"regime_shading_{cache_key}", False)
            use_log_scale = st.session_state.get(f"log_scale_{cache_key}", True)
            use_vol_log_scale = st.session_state.get(f"vol_log_scale_{cache_key}", False)
        

            cached_data = st.session_state[plots_cache_key]

            import time, json
            t0_plot_create = time.time()

            fig = create_hmm_plot(use_regime_colors, use_regime_shading, use_log_scale, use_vol_log_scale,
                                 cached_data['returns_df'], cached_data['dates'], 
                                 cached_data['regime_array'], cached_data['segment_starts'],
                                 cached_data['num_regimes'], cached_data['regime_colors'], features_used)

            t1_plot_create = time.time()
            try:
                log_path = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor\debug.log'
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'id': 'perf_plot_create', 'timestamp': time.time() * 1000, 'location': 'app.py:2013', 'message': 'Plotly figure creation', 'data': {'duration_sec': t1_plot_create - t0_plot_create, 'use_regime_colors': use_regime_colors, 'num_segments': len(cached_data['segment_starts'])}, 'sessionId': 'debug-session', 'runId': 'perf-1', 'hypothesisId': 'D'}) + '\n')
            except: pass

        


            t0_render = time.time()

            st.plotly_chart(fig, width='stretch')

            t1_render = time.time()
            try:
                log_path = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor\debug.log'
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'id': 'perf_plot_render', 'timestamp': time.time() * 1000, 'location': 'app.py:2019', 'message': 'Plotly chart rendering', 'data': {'duration_sec': t1_render - t0_render}, 'sessionId': 'debug-session', 'runId': 'perf-1', 'hypothesisId': 'E'}) + '\n')
            except: pass

        

            col_toggle1, col_toggle2, col_toggle3, col_toggle4 = st.columns(4)
            with col_toggle1:
                st.checkbox("Show Colored Lines", value=use_regime_colors, help="Show regime colors in line plots (fast)", key=f"regime_colors_{cache_key}")
            with col_toggle2:
                st.checkbox("Show Background Shading", value=use_regime_shading, help="Show colored background shading (slower, may take time)", key=f"regime_shading_{cache_key}")
            with col_toggle3:
                st.checkbox("Log Scale for Price", value=use_log_scale, help="Toggle between cumulative (linear) and log scale for price", key=f"log_scale_{cache_key}")
            with col_toggle4:
                st.checkbox("Log Scale for Volatility", value=use_vol_log_scale, help="Toggle between linear and log scale for volatility", key=f"vol_log_scale_{cache_key}")

            t1_viz_section = time.time()
            try:
                log_path = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor\debug.log'
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'id': 'perf_viz_section_total', 'timestamp': time.time() * 1000, 'location': 'app.py:2026', 'message': 'Total visualization section', 'data': {'duration_sec': t1_viz_section - t0_viz_section}, 'sessionId': 'debug-session', 'runId': 'perf-1', 'hypothesisId': 'ALL'}) + '\n')
            except: pass

        

            st.write("**📈 Rolling Metrics Over Time by Regime:**")
            if hasattr(returns_data, 'index') and len(returns_data) > 21:
                returns_df = pd.DataFrame({'returns': returns_data_display}, index=returns_data_display.index)
                returns_df['regime'] = regime_labels_display[:len(returns_df)]
            

                col_param1, col_param2 = st.columns(2)
                with col_param1:
                    rolling_window_days_metrics = st.number_input(
                        "Rolling Window (trading days)",
                        min_value=21,
                        max_value=min(2520, len(returns_data)),
                        value=252,
                        step=21,
                        key="rolling_window_metrics",
                        help="Number of trading days for rolling calculation (252 = 1 year)"
                    )
                with col_param2:
                    metric_choice_metrics = st.selectbox(
                        "Metric",
                        options=['CAGR', 'Volatility', 'Skewness', 'Kurtosis'],
                        index=0,
                        key="metric_choice_metrics",
                        help="Select the metric to analyze over time"
                    )
            

                if len(returns_data) >= rolling_window_days_metrics:
                    if metric_choice_metrics == 'CAGR':

                        rolling_sum = returns_df['returns'].rolling(window=rolling_window_days_metrics, min_periods=1).sum()
                        returns_df['rolling_metric_ts'] = ((1 + rolling_sum / 100) ** (252 / rolling_window_days_metrics) - 1) * 100
                        metric_label_ts = 'CAGR (%)'
                    elif metric_choice_metrics == 'Volatility':
                        returns_df['rolling_metric_ts'] = returns_df['returns'].rolling(window=rolling_window_days_metrics, min_periods=1).std() * np.sqrt(252)
                        metric_label_ts = 'Volatility (%)'
                    elif metric_choice_metrics == 'Skewness':
                        returns_df['rolling_metric_ts'] = returns_df['returns'].rolling(window=rolling_window_days_metrics, min_periods=1).skew()
                        metric_label_ts = 'Skewness'
                    else:
                        def calc_kurtosis(x):
                            if len(x) < 4:
                                return np.nan
                            mean = np.mean(x)
                            std = np.std(x, ddof=0)
                            if std == 0:
                                return np.nan
                            n = len(x)
                            return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((x - mean) / std) ** 4) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
                        returns_df['rolling_metric_ts'] = returns_df['returns'].rolling(window=rolling_window_days_metrics, min_periods=4).apply(calc_kurtosis, raw=True)
                        metric_label_ts = 'Kurtosis'
                else:
                    returns_df['rolling_metric_ts'] = np.nan
            
                if HAS_PLOTLY:

                    fig_metrics = go.Figure()
                

                    regime_traces = {r: {'x': [], 'y': [], 'last_idx': -1} for r in range(num_regimes)}
                

                    for i in range(len(returns_df)):
                        regime = int(returns_df['regime'].iloc[i]) if not pd.isna(returns_df['regime'].iloc[i]) else None
                    
                        if regime is not None and regime < num_regimes:

                            if regime_traces[regime]['last_idx'] >= 0 and i > regime_traces[regime]['last_idx'] + 1:
                                regime_traces[regime]['x'].append(None)
                                regime_traces[regime]['y'].append(None)
                        

                            regime_traces[regime]['x'].append(returns_df.index[i])
                            metric_val = returns_df['rolling_metric_ts'].iloc[i]
                            regime_traces[regime]['y'].append(metric_val if not pd.isna(metric_val) else None)
                            regime_traces[regime]['last_idx'] = i
                

                    for regime_idx in range(num_regimes):
                        if len(regime_traces[regime_idx]['x']) > 0:
                            color = regime_colors[regime_idx % len(regime_colors)]
                            fig_metrics.add_trace(
                                go.Scatter(
                                    x=regime_traces[regime_idx]['x'],
                                    y=regime_traces[regime_idx]['y'],
                                    mode='lines',
                                    name=f'Regime {regime_idx}',
                                    line=dict(color=color, width=2),
                                    showlegend=True,
                                    hovertemplate=f'Date: %{{x}}<br>{metric_label_ts}: %{{y:.2f}}<extra></extra>'
                                )
                            )
                
                    window_label_ts = f"{rolling_window_days_metrics}-Day" if rolling_window_days_metrics != 252 else "1-Year"
                    portfolio_name = hmm_results.get('portfolio_name', 'Portfolio')
                    fig_metrics.update_layout(
                        title=f'{portfolio_name} - Rolling {window_label_ts} {metric_choice_metrics} Over Time by Regime',
                        xaxis_title='Date',
                        yaxis_title=f'{metric_label_ts}',
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=12),
                        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)'),
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_metrics, width='stretch')
                else:

                    import matplotlib.pyplot as plt
                    plt.style.use('dark_background')
                    fig, ax = plt.subplots(figsize=(12, 6), facecolor='black')
                

                    current_regime = None
                    segment_start_idx = None
                
                    for i in range(len(returns_df)):
                        regime = int(returns_df['regime'].iloc[i]) if not pd.isna(returns_df['regime'].iloc[i]) else None
                    
                        if regime != current_regime:

                            if current_regime is not None and segment_start_idx is not None:
                                segment_mask = (returns_df.index >= returns_df.index[segment_start_idx]) & (returns_df.index < returns_df.index[i])
                                segment_data = returns_df.loc[segment_mask, 'rolling_metric_ts']
                                if len(segment_data.dropna()) > 0:
                                    hex_color = regime_colors[current_regime % len(regime_colors)]
                                    rgb_color = tuple(int(hex_color[j:j+2], 16) / 255.0 for j in (1, 3, 5))
                                    ax.plot(returns_df.index[segment_mask], segment_data, 
                                           color=rgb_color, linewidth=2, alpha=0.8)
                        

                            current_regime = regime
                            segment_start_idx = i
                

                    if current_regime is not None and segment_start_idx is not None:
                        segment_mask = returns_df.index >= returns_df.index[segment_start_idx]
                        segment_data = returns_df.loc[segment_mask, 'rolling_metric_ts']
                        if len(segment_data.dropna()) > 0:
                            hex_color = regime_colors[current_regime % len(regime_colors)]
                            rgb_color = tuple(int(hex_color[j:j+2], 16) / 255.0 for j in (1, 3, 5))
                            ax.plot(returns_df.index[segment_mask], segment_data, 
                                   color=rgb_color, linewidth=2, alpha=0.8, label=f'Regime {current_regime}')
                
                    window_label_ts = f"{rolling_window_days_metrics}-Day" if rolling_window_days_metrics != 252 else "1-Year"
                    portfolio_name = hmm_results.get('portfolio_name', 'Portfolio')
                    ax.set_xlabel('Date', color='white', fontsize=12)
                    ax.set_ylabel(f'{metric_label_ts}', color='white', fontsize=12)
                    ax.set_title(f'{portfolio_name} - Rolling {window_label_ts} {metric_choice_metrics} Over Time by Regime', 
                                color='white', fontsize=14, fontweight='bold')
                    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
                    ax.grid(True, alpha=0.3, color='gray')
                    ax.set_facecolor('black')
                    ax.tick_params(colors='white')
                    st.pyplot(fig)
            else:
                st.info("Insufficient data for rolling metrics analysis (need at least 21 days).")
        

            st.write("**📈 Sequential Regime Backtest (Separate Lines, No Gaps):**")
            if hasattr(returns_data, 'index') and len(returns_data) > 21:

                returns_df_sequential = pd.DataFrame({'returns': returns_data_display}, index=returns_data_display.index)
                returns_df_sequential['regime'] = regime_labels_display[:len(returns_df_sequential)]
            

                if len(returns_data) >= rolling_window_days_metrics:
                    if metric_choice_metrics == 'CAGR':
                        rolling_sum = returns_df_sequential['returns'].rolling(window=rolling_window_days_metrics, min_periods=1).sum()
                        returns_df_sequential['rolling_metric_seq'] = ((1 + rolling_sum / 100) ** (252 / rolling_window_days_metrics) - 1) * 100
                    elif metric_choice_metrics == 'Volatility':
                        returns_df_sequential['rolling_metric_seq'] = returns_df_sequential['returns'].rolling(window=rolling_window_days_metrics, min_periods=1).std() * np.sqrt(252)
                    elif metric_choice_metrics == 'Skewness':
                        returns_df_sequential['rolling_metric_seq'] = returns_df_sequential['returns'].rolling(window=rolling_window_days_metrics, min_periods=1).skew()
                    else:
                        def calc_kurtosis_seq(x):
                            if len(x) < 4:
                                return np.nan
                            mean = np.mean(x)
                            std = np.std(x, ddof=0)
                            if std == 0:
                                return np.nan
                            n = len(x)
                            return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((x - mean) / std) ** 4) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
                        returns_df_sequential['rolling_metric_seq'] = returns_df_sequential['returns'].rolling(window=rolling_window_days_metrics, min_periods=4).apply(calc_kurtosis_seq, raw=True)
                else:
                    returns_df_sequential['rolling_metric_seq'] = np.nan
            

                regime_options = [f'Regime {i}' for i in range(num_regimes)]
                selected_regimes = st.multiselect(
                    "Select Regimes to Display",
                    options=regime_options,
                    default=regime_options,
                    key="sequential_regime_selector",
                    help="Choose which regimes to show in the sequential backtest plot"
                )
                selected_regime_indices = [int(r.split()[1]) for r in selected_regimes]
            
                if HAS_PLOTLY and len(selected_regime_indices) > 0:
                    fig_seq = go.Figure()
                

                    for regime_idx in selected_regime_indices:
                        regime_mask = returns_df_sequential['regime'] == regime_idx
                        regime_data = returns_df_sequential.loc[regime_mask, 'rolling_metric_seq'].dropna()
                    
                        if len(regime_data) > 0:

                            sequential_x = list(range(len(regime_data)))
                            color = regime_colors[regime_idx % len(regime_colors)]
                        
                            fig_seq.add_trace(
                                go.Scatter(
                                    x=sequential_x,
                                    y=regime_data.values,
                                    mode='lines',
                                    name=f'Regime {regime_idx}',
                                    line=dict(color=color, width=2),
                                    showlegend=True,
                                    hovertemplate=f'Regime {regime_idx}<br>Observation: %{{x}}<br>{metric_label_ts}: %{{y:.2f}}<extra></extra>'
                                )
                            )
                
                    window_label_seq = f"{rolling_window_days_metrics}-Day" if rolling_window_days_metrics != 252 else "1-Year"
                    portfolio_name = hmm_results.get('portfolio_name', 'Portfolio')
                
                    fig_seq.update_layout(
                        title=f'{portfolio_name} - Sequential {window_label_seq} {metric_choice_metrics} Backtest by Regime',
                        xaxis_title='Observation Index (within each regime)',
                        yaxis_title=f'{metric_label_ts}',
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=12),
                        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', x=1.02, y=1),
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_seq, width='stretch')
                elif len(selected_regime_indices) == 0:
                    st.info("Please select at least one regime to display.")
                else:

                    import matplotlib.pyplot as plt
                    plt.style.use('dark_background')
                    fig, ax = plt.subplots(figsize=(12, 6), facecolor='black')
                
                    for regime_idx in selected_regime_indices:
                        regime_mask = returns_df_sequential['regime'] == regime_idx
                        regime_data = returns_df_sequential.loc[regime_mask, 'rolling_metric_seq'].dropna()
                    
                        if len(regime_data) > 0:
                            sequential_x = list(range(len(regime_data)))
                            hex_color = regime_colors[regime_idx % len(regime_colors)]
                            rgb_color = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5))
                            ax.plot(sequential_x, regime_data.values, 
                                   label=f'Regime {regime_idx}', color=rgb_color, linewidth=2)
                
                    window_label_seq = f"{rolling_window_days_metrics}-Day" if rolling_window_days_metrics != 252 else "1-Year"
                    portfolio_name = hmm_results.get('portfolio_name', 'Portfolio')
                    ax.set_xlabel('Observation Index (within each regime)', color='white', fontsize=12)
                    ax.set_ylabel(f'{metric_label_ts}', color='white', fontsize=12)
                    ax.set_title(f'{portfolio_name} - Sequential {window_label_seq} {metric_choice_metrics} Backtest by Regime', 
                                color='white', fontsize=14, fontweight='bold')
                    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
                    ax.grid(True, alpha=0.3, color='gray')
                    ax.set_facecolor('black')
                    ax.tick_params(colors='white')
                    st.pyplot(fig)
            else:
                st.info("Insufficient data for sequential regime backtest (need at least 21 days).")
        

            st.write("**📊 Rolling Distribution by Regime:**")
            if hasattr(returns_data, 'index') and len(returns_data) > 21:
                returns_df = pd.DataFrame({'returns': returns_data_display}, index=returns_data_display.index)
                returns_df['regime'] = regime_labels_display[:len(returns_df)]
            

                col_param1, col_param2 = st.columns(2)
                with col_param1:
                    rolling_window_days = st.number_input(
                        "Rolling Window (trading days)",
                        min_value=21,
                        max_value=min(2520, len(returns_data)),
                        value=252,
                        step=21,
                        key="rolling_window_dist",
                        help="Number of trading days for rolling calculation (252 = 1 year)"
                    )
                with col_param2:
                    metric_choice = st.selectbox(
                        "Metric",
                        options=['Return', 'Volatility', 'Skewness', 'Kurtosis'],
                        index=0,
                        key="metric_choice_dist",
                        help="Select the metric to analyze"
                    )
            

                if len(returns_data) >= rolling_window_days:
                    if metric_choice == 'Return':
                        returns_df['rolling_metric'] = returns_df['returns'].rolling(window=rolling_window_days, min_periods=1).sum()
                        metric_label = 'Return (%)'
                    elif metric_choice == 'Volatility':
                        returns_df['rolling_metric'] = returns_df['returns'].rolling(window=rolling_window_days, min_periods=1).std() * np.sqrt(252)
                        metric_label = 'Volatility (%)'
                    elif metric_choice == 'Skewness':
                        returns_df['rolling_metric'] = returns_df['returns'].rolling(window=rolling_window_days, min_periods=1).skew()
                        metric_label = 'Skewness'
                    else:

                        def calc_kurtosis(x):
                            if len(x) < 4:
                                return np.nan
                            mean = np.mean(x)
                            std = np.std(x, ddof=0)
                            if std == 0:
                                return np.nan
                            n = len(x)
                            return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((x - mean) / std) ** 4) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
                        returns_df['rolling_metric'] = returns_df['returns'].rolling(window=rolling_window_days, min_periods=4).apply(calc_kurtosis, raw=True)
                        metric_label = 'Kurtosis'
                else:
                    returns_df['rolling_metric'] = np.nan
            
                if HAS_PLOTLY:

                    fig_dist = go.Figure()
                
                    for regime_idx in range(num_regimes):
                        regime_mask = returns_df['regime'] == regime_idx
                        regime_values = returns_df.loc[regime_mask, 'rolling_metric'].dropna()
                    
                        if len(regime_values) > 0:
                            color = regime_colors[regime_idx % len(regime_colors)]
                        

                            fig_dist.add_trace(go.Histogram(
                                x=regime_values,
                                name=f'Regime {regime_idx}',
                                opacity=0.7,
                                marker_color=color,
                                nbinsx=50,
                                histnorm='probability density'
                            ))
                
                    window_label = f"{rolling_window_days}-Day" if rolling_window_days != 252 else "1-Year"
                    portfolio_name = hmm_results.get('portfolio_name', 'Portfolio')
                    fig_dist.update_layout(
                        title=f'{portfolio_name} - Rolling {window_label} {metric_choice} Distribution by Regime',
                        xaxis_title=f'{rolling_window_days}-Day {metric_label}',
                        yaxis_title='Density',
                        barmode='overlay',
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=12),
                        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)')
                    )
                    st.plotly_chart(fig_dist, width='stretch')
                else:

                    import matplotlib.pyplot as plt
                    plt.style.use('dark_background')
                    fig, ax = plt.subplots(figsize=(12, 6), facecolor='black')
                
                    for regime_idx in range(num_regimes):
                        regime_mask = returns_df['regime'] == regime_idx
                        regime_values = returns_df.loc[regime_mask, 'rolling_metric'].dropna()
                    
                        if len(regime_values) > 0:

                            hex_color = regime_colors[regime_idx % len(regime_colors)]
                            rgb_color = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5))
                            color = rgb_color
                            ax.hist(regime_values, bins=50, alpha=0.6, label=f'Regime {regime_idx}', 
                                   color=color, edgecolor='white', linewidth=0.5, density=True)
                
                    window_label = f"{rolling_window_days}-Day" if rolling_window_days != 252 else "1-Year"
                    portfolio_name = hmm_results.get('portfolio_name', 'Portfolio')
                    ax.set_xlabel(f'{rolling_window_days}-Day {metric_label}', color='white', fontsize=12)
                    ax.set_ylabel('Density', color='white', fontsize=12)
                    ax.set_title(f'{portfolio_name} - Rolling {window_label} {metric_choice} Distribution by Regime', color='white', fontsize=14, fontweight='bold')
                    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
                    ax.grid(True, alpha=0.3, color='gray')
                    ax.set_facecolor('black')
                    ax.tick_params(colors='white')
                    st.pyplot(fig)
            else:
                st.info("Insufficient data for rolling analysis (need at least 21 days).")
    

    if hasattr(returns_data_display, 'index'):
        st.write("**Regime Labels Over Time:**")
        regime_series = pd.Series(regime_labels_display, index=returns_data_display.index)
        
        if HAS_MATPLOTLIB:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            fig, ax = plt.subplots(figsize=(12, 4), facecolor='black')
            ax.set_facecolor('black')
            
            # Use step plot for discrete regime values - more efficient and appropriate
            ax.step(regime_series.index, regime_series.values, where='post', 
                   linewidth=1, color='cyan', label='Regime')
            
            ax.set_title('Regime Labels Over Time', color='white', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', color='white', fontsize=12)
            ax.set_ylabel('Regime', color='white', fontsize=12)
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3, color='gray')
            
            # Set y-axis to show integer regime values
            unique_regimes = np.unique(regime_series.values)
            ax.set_yticks(unique_regimes)
            ax.set_ylim(unique_regimes.min() - 0.5, unique_regimes.max() + 0.5)
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        elif HAS_PLOTLY:
            # Fallback to Plotly if matplotlib not available
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=regime_series.index,
                y=regime_series.values,
                mode='lines',
                name='Regime',
                line=dict(width=1, shape='hv'),  # 'hv' creates step plot
                hovertemplate='Date: %{x}<br>Regime: %{y}<extra></extra>',
                connectgaps=False
            ))
            fig.update_layout(
                title='Regime Labels Over Time',
                xaxis_title='Date',
                yaxis_title='Regime',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_yaxes(tickmode='linear', tick0=0, dtick=1)
            st.plotly_chart(fig, width='stretch')
        else:
            st.line_chart(regime_series)
    

    st.download_button(
        label="📥 Download HMM Results (CSV)",
        data=pd.DataFrame({
            'date': returns_data.index,
            'regime': regime_labels
        }).to_csv(index=False),
        file_name="hmm_regime_results.csv",
        mime="text/csv",
        help="Monthly dates and regimes (matches analysis). Use 'Save to output' for daily expansion."
    )

def display_param_estimation_results(results, returns_data):
    """Display regime-conditional parameter estimation results"""
    st.subheader("📋 Results")

    # Debug: show recent Bates-related log entries so we can see data prep, fit, and viz under the hood
    _debug_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cursor', 'debug.log')
    if os.path.exists(_debug_log_path):
        try:
            with open(_debug_log_path, 'r', encoding='utf-8') as _f:
                _lines = _f.readlines()
            _bates_lines = [ln.strip() for ln in _lines[-200:] if ln.strip() and ('bates_data_prep' in ln or 'bates_post_fit_summary' in ln or 'bates_fit_exit' in ln or 'bates_viz_entry' in ln)]
            _recent = _bates_lines[-15:]  # last 15 Bates-related entries
            if _recent:
                with st.expander("🔧 Debug: recent Bates pipeline log (data prep → fit → viz)", expanded=False):
                    for _ln in _recent:
                        try:
                            _j = __import__('json').loads(_ln)
                            _id = _j.get('id', '')
                            _loc = _j.get('location', '')
                            _msg = _j.get('message', '')
                            _data = _j.get('data', {})
                            st.code(f"{_id} | {_loc}\n  {_msg}\n  data: {_data}", language=None)
                        except Exception:
                            st.text(_ln[:500])
        except Exception:
            pass
    
    params_dict = results['params_dict']
    moment_info_dict = results['moment_info_dict']
    
    if not params_dict:
        st.warning("No parameters estimated. Check estimation logs above.")
        return
    

    st.write("**Estimated Parameters by Regime:**")
    

    IDX = {
        'mu': 0, 'kappa': 1, 'theta': 2, 'nu': 3, 'rho': 4,
        'v0': 5, 'lam': 6, 'mu_J': 7, 'sigma_J': 8
    }
    
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
    

    # Rounding: 5 decimals for parameters and moment levels, 4 for skew/kurt
    FMT_PARAM = ".5f"
    FMT_MOMENT = ".5f"
    FMT_SKEW_KURT = ".4f"

    def _fmt(val, fmt=FMT_MOMENT):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        return f"{float(val):{fmt}}"

    param_data = []
    for param_name in param_names:
        row = {"Parameter": param_labels[param_name]}
        for regime_id in sorted(params_dict.keys()):
            val = params_dict[regime_id][IDX[param_name]]
            row[f"Regime {regime_id}"] = _fmt(val, FMT_PARAM)
        param_data.append(row)

    param_df = pd.DataFrame(param_data)
    st.dataframe(param_df, use_container_width=True, hide_index=True)

    st.write("**Moment Matching Quality:**")
    quality_data = []
    for regime_id in sorted(params_dict.keys()):
        info = moment_info_dict[regime_id]
        quality_data.append({
            "Regime": regime_id,
            "N Obs": info["n_obs"],
            "Emp Mean": _fmt(info["emp_mean"]),
            "Model Mean": _fmt(info["model_mean"]),
            "Emp Std": _fmt(info["emp_std"]),
            "Model Std": _fmt(info["model_std"]),
            "Emp Skew": _fmt(info["emp_skew"], FMT_SKEW_KURT),
            "Model Skew": _fmt(info["model_skew"], FMT_SKEW_KURT),
            "Emp Kurt": _fmt(info["emp_kurt"], FMT_SKEW_KURT),
            "Model Kurt": _fmt(info["model_kurt"], FMT_SKEW_KURT),
        })

    quality_df = pd.DataFrame(quality_data)
    st.dataframe(quality_df, use_container_width=True, hide_index=True)

    # Warnings: bounds hitting and optimization not converging (when results_list available)
    results_list = results.get("results_list") or []
    if results_list:
        warnings_any = []
        for r in results_list:
            name = r.get("name", "")
            opt_ok = r.get("optimization_success", True)
            diag = r.get("boundary_diagnostics") or {}
            hitting = diag.get("hitting_boundaries") or []
            if not opt_ok:
                warnings_any.append(f"**{name}**: Optimization did not report convergence (consider more restarts or relaxing bounds).")
            if hitting:
                warnings_any.append(f"**{name}**: Parameters hitting bounds: {', '.join(hitting)}. Bounds may be too tight.")
        if warnings_any:
            st.write("**⚠️ Estimation Warnings**")
            for w in warnings_any:
                st.warning(w)

    if results.get("transition_matrix") is not None:
        st.write("**HMM Transition Matrix:**")
        trans_mat = np.asarray(results["transition_matrix"], dtype=float)
        trans_rounded = np.round(trans_mat, 5)
        trans_df = pd.DataFrame(
            trans_rounded,
            index=[f"Regime {i}" for i in range(len(trans_rounded))],
            columns=[f"Regime {j}" for j in range(trans_rounded.shape[1])]
        )
        st.dataframe(trans_df, use_container_width=True)

    # Show/hide controlled by "Show distribution visualizations" checkbox at top of Parameter Estimation
    if st.session_state.get("param_show_viz", True) and HAS_MATPLOTLIB and results.get("regime_labels") is not None and hasattr(returns_data, "__len__"):
        try:
            project_root = os.path.dirname(os.path.abspath(__file__))
            moment_matching_path = os.path.join(
                project_root,
                "Parametric Model (Unfinished)",
                "Stochastic Vol + Jump Diffusion",
                "2B. Moment Matching.py"
            )
            
            if os.path.exists(moment_matching_path):
                from importlib.util import spec_from_file_location, module_from_spec
                spec = spec_from_file_location("bates_moment_matching", moment_matching_path)
                moment_module = module_from_spec(spec)
                spec.loader.exec_module(moment_module)
                
                # REPLICATE STANDALONE SCRIPT EXACTLY for visualization
                # The standalone script uses results_list and df directly from the fitting loop
                # If we have results_list and df stored in results, use those (new workflow)
                # Otherwise, reconstruct from params_dict (old workflow for backward compatibility)
                
                if 'results_list' in results and 'df' in results:
                    # New workflow: Use results_list and df directly (matches standalone script exactly)
                    results_list = results['results_list']
                    df = results['df']
                else:
                    # Old workflow: Reconstruct from params_dict (for backward compatibility)
                    pass

                regime_labels = results['regime_labels']
                n_regime = len(regime_labels)
                # Use returns that match regime_labels length (monthly; use HMM monthly returns when available)
                hmm_ret = None
                if getattr(st.session_state, 'hmm_results', None) is not None:
                    hr = st.session_state.hmm_results.get('returns_data')
                    if hr is not None and len(hr) == n_regime:
                        hmm_ret = hr
                if hmm_ret is not None:
                    all_emp_returns = hmm_ret.values.copy() if hasattr(hmm_ret, 'values') else np.array(hmm_ret).copy()
                else:
                    if hasattr(returns_data, 'values'):
                        all_emp_returns = returns_data.values.copy()
                    else:
                        all_emp_returns = np.array(returns_data).copy()
                    # Truncate to match regime length if tab returns are longer (e.g. daily vs monthly)
                    if len(all_emp_returns) > n_regime:
                        all_emp_returns = all_emp_returns[:n_regime]
                    elif len(all_emp_returns) < n_regime:
                        regime_labels = regime_labels[:len(all_emp_returns)]
                
                # Convert to log decimal: if percentage log returns (|x|>1), divide by 100
                if np.any(np.abs(all_emp_returns) > 1.0):
                    all_emp_returns = all_emp_returns / 100.0
                    conversion_applied = True
                else:
                    conversion_applied = False
                
                # Align: same length by construction; ensure regime_labels_aligned matches all_emp_returns
                regime_labels_aligned = np.asarray(regime_labels, dtype=float)[:len(all_emp_returns)]
                
                # Create DataFrame EXACTLY like standalone script (line 1248-1262)
                # The standalone script creates df with 'returns' column as log returns
                df = pd.DataFrame({
                    'returns': all_emp_returns,  # Simple log returns (matching standalone script line 1262)
                    'regime': np.array(regime_labels_aligned, dtype=float)
                })
                
                # Create results_list from already-estimated params_dict (matching standalone script format)
                # The standalone script's results list has 'name', 'params', 'n_obs' keys (line 1398-1399)
                results_list = []
                for regime_id in sorted(params_dict.keys()):
                    if regime_id not in moment_info_dict:
                        continue
                    info = moment_info_dict[regime_id]
                    if params_dict[regime_id] is None:
                        continue
                    results_list.append({
                        'name': f'regime_{regime_id}',  # Match standalone script format (line 1392)
                        'params': params_dict[regime_id],
                        'n_obs': info.get('n_obs', 0)
                    })
                
                # #region agent log
                try:
                    import json
                    import time
                    log_path_debug = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cursor', 'debug.log')
                    os.makedirs(os.path.dirname(log_path_debug), exist_ok=True)
                    with open(log_path_debug, 'a', encoding='utf-8') as f:
                        df_grouped = df.groupby('regime')
                        regimes_in_df = {float(k): len(v) for k, v in df_grouped}
                        f.write(json.dumps({'id': 'log_df_created', 'timestamp': time.time() * 1000, 'location': 'app.py:display_param_estimation_results', 'message': 'DataFrame created', 'data': {'df_shape': list(df.shape), 'df_columns': list(df.columns), 'regime_counts': regimes_in_df, 'regime_keys_types': [type(k).__name__ for k in list(regimes_in_df.keys())[:5]], 'returns_min': float(np.min(df['returns'])), 'returns_max': float(np.max(df['returns'])), 'returns_mean': float(np.mean(df['returns']))}, 'runId': 'initial', 'hypothesisId': 'B'}) + '\n')
                except Exception as log_err: pass
                # #endregion
                
                # results_list is already created above by replicating the standalone script's fitting loop
                
                portfolio_name = results.get('portfolio_name', 'Portfolio')
                
                # Show/hide controlled by "Show distribution visualizations" checkbox at top of Parameter Estimation
                if st.session_state.get("param_show_viz", True):
                    # Check if visualizations already exist (do not reuse old images from previous runs)
                    import re
                    portfolio_slug = re.sub(r'[^a-zA-Z0-9]+', '_', portfolio_name.lower()).strip('_')
                    output_dir = os.path.join(project_root, 'output')
                    png_files = [
                        f'{portfolio_slug}_bates_regime_distributions.png',
                        f'{portfolio_slug}_bates_regime_metrics.png',
                        f'{portfolio_slug}_bates_joint_distributions.png',
                        f'{portfolio_slug}_bates_comprehensive_comparison.png'
                    ]
                    
                    # Create a cache key based on parameters
                    import hashlib
                    params_hash = hashlib.md5(str(sorted(params_dict.items())).encode()).hexdigest()[:8]
                    cache_key = f"viz_generated_{portfolio_slug}_{params_hash}"
                    
                    # Check if all files exist
                    all_exist = all(os.path.exists(os.path.join(output_dir, png_file)) for png_file in png_files)
                    
                    # Number of simulations comes from top-level control (param_viz_n_simulations)
                    param_viz_n_sim = int(st.session_state.get("param_viz_n_simulations", 5000))
                    
                    # Regenerate unless we already generated this session (do not reuse old images)
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        force_regenerate = st.button("🔄 Regenerate", key="force_regen_viz", help="Force regeneration of visualizations")
                    
                    should_generate = force_regenerate or (cache_key not in st.session_state)
                    
                    if should_generate:
                        with st.spinner("Generating distribution visualizations (this may take 1-2 minutes due to Monte Carlo simulations)..."):
                            # CRITICAL: Change to output directory before calling generate_distribution_plots
                            # The function uses relative paths 'output/' which must be relative to current working directory
                            original_cwd = os.getcwd()
                            try:
                                os.chdir(project_root)
                            
                                # Suppress console output from the module to avoid blocking
                                import sys
                                from io import StringIO
                                old_stdout = sys.stdout
                                old_stderr = sys.stderr
                                sys.stdout = StringIO()
                                sys.stderr = StringIO()
                                try:
                                    # Call generate_distribution_plots directly (no wrapper) to match standalone script behavior
                                    moment_module.generate_distribution_plots(
                                        results_list, 
                                        df, 
                                        'regime', 
                                        portfolio_name=portfolio_name,
                                        n_simulations_for_plots=int(param_viz_n_sim)
                                    )
                                    
                                except Exception as gen_err:
                                    raise
                                finally:
                                    sys.stdout = old_stdout
                                    sys.stderr = old_stderr
                            finally:
                                # Restore original working directory
                                os.chdir(original_cwd)
                    
                        st.session_state[cache_key] = True
                        st.success("✅ All visualizations generated and saved to output/ folder")
                        # Display the generated images in the UI
                        st.write("**📊 Visualizations:**")
                        for png_file in png_files:
                            png_path = os.path.join(output_dir, png_file)
                            if os.path.exists(png_path):
                                title = png_file.replace(f'{portfolio_slug}_bates_', '').replace('.png', '').replace('_', ' ').title()
                                st.write(f"**{title}:**")
                                st.image(png_path, use_container_width=True)
                            else:
                                st.warning(f"Visualization file not found: {png_file}")
                    elif all_exist and cache_key in st.session_state:
                        # Only treat as cache when we generated for these exact params in this session
                        st.info("ℹ️ Using cached visualizations for current parameters. Click 'Regenerate' to refresh.")

                        # Display the saved PNG files only when we generated this session
                        st.write("**📊 Visualizations:**")
                        try:
                            with open(log_path, 'a', encoding='utf-8') as f:
                                f.write(json.dumps({'id': 'log_check_png_files', 'timestamp': time.time() * 1000, 'location': 'app.py:display_param_estimation_results', 'message': 'Checking PNG files', 'data': {'output_dir': output_dir, 'png_files': png_files}, 'runId': 'initial', 'hypothesisId': 'F'}) + '\n')
                        except Exception as log_err: pass
                        for png_file in png_files:
                            png_path = os.path.join(output_dir, png_file)
                            if os.path.exists(png_path):
                                file_size = os.path.getsize(png_path)
                                title = png_file.replace(f'{portfolio_slug}_bates_', '').replace('.png', '').replace('_', ' ').title()
                                st.write(f"**{title}:** (Size: {file_size:,} bytes)")
                                st.image(png_path, use_container_width=True)
                            else:
                                st.warning(f"Visualization file not found: {png_file}")
                    else:
                        # Files may exist from a previous run; do not display them
                        st.write("**📊 Visualizations:**")
                        st.caption("Click 'Regenerate' to generate visualizations for current parameters.")
                    
            else:
                st.warning("Could not find moment matching script to generate visualizations.")
        except Exception as e:
            st.warning(f"Could not generate visualizations: {str(e)}")
            import traceback
            with st.expander("Visualization Error Details"):
                st.code(traceback.format_exc())

def distribution_comparison_section(returns_data):
    """Display distribution comparison between empirical data and simulated model"""
    st.subheader("📊 Distribution Comparison: Empirical vs Model")
    

    if 'param_estimation_results' not in st.session_state or st.session_state.param_estimation_results is None:
        st.info("💡 Run parameter estimation first to enable distribution comparison.")
        return
    
    results = st.session_state.param_estimation_results
    params_dict = results.get('params_dict', {})
    regime_labels = results.get('regime_labels')
    transition_matrix = results.get('transition_matrix')
    
    if not params_dict:
        st.warning("No parameters available for comparison. Run parameter estimation first.")
        return
    

    col1, col2 = st.columns(2)
    with col1:
        n_simulations = st.number_input("Number of Simulations", min_value=1, max_value=10000, value=1000, step=100, key="dist_n_sim")
    with col2:

        default_periods = len(returns_data) if hasattr(returns_data, '__len__') else 1000
        default_periods = max(100, default_periods)
        n_periods = st.number_input("Periods per Simulation", min_value=100, max_value=None, value=default_periods, step=100, key="dist_n_periods")
    
    if st.button("🔄 Run Distribution Comparison", type="primary", key="dist_run_btn"):
        with st.spinner("Running Monte Carlo simulations and generating comparisons..."):
            try:


                project_root = os.path.dirname(os.path.abspath(__file__))
                sim_script_path = os.path.join(
                    project_root,
                    "Parametric Model (Unfinished)",
                    "Stochastic Vol + Jump Diffusion",
                    "3. Empircal versus Model Bates.py"
                )
                
                if os.path.exists(sim_script_path):
                    from importlib.util import spec_from_file_location, module_from_spec
                    spec = spec_from_file_location("bates_simulation", sim_script_path)
                    sim_module = module_from_spec(spec)
                    spec.loader.exec_module(sim_module)
                    

                    regime_ids = sorted(params_dict.keys())
                    
                    simulated_returns_list = []
                    first_path_regimes = None
                    
                    np.random.seed(42)
                    seeds = np.random.randint(0, 1000000, size=n_simulations)
                    

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    crisis_boost = (0, 0)
                    for i in range(n_simulations):
                        sim_returns, _regimes = sim_module.simulate_regime_switching_bates(
                            params_dict,
                            transition_matrix,
                            regime_ids,
                            n_periods,
                            seed=int(seeds[i]),
                            crisis_jump_boost=crisis_boost,
                            stress_variance_on_crisis_entry=False,
                        )
                        simulated_returns_list.append(sim_returns)
                        if i == 0:
                            first_path_regimes = _regimes
                        

                        progress = (i + 1) / n_simulations
                        progress_bar.progress(progress)
                        status_text.text(f"Simulation {i + 1}/{n_simulations} ({progress*100:.1f}%)")
                    

                    progress_bar.empty()
                    status_text.empty()
                    
                    simulated_returns_array = np.array(simulated_returns_list)
                    

                    if hasattr(returns_data, 'values'):
                        empirical_returns = returns_data.values
                    else:
                        empirical_returns = np.array(returns_data)
                    

                    from scipy import stats
                    emp_mean = np.mean(empirical_returns)
                    emp_std = np.std(empirical_returns)
                    emp_skew = stats.skew(empirical_returns)
                    emp_kurt = stats.kurtosis(empirical_returns)
                    

                    all_sim_returns = simulated_returns_array.flatten()
                    sim_mean = np.mean(all_sim_returns)
                    sim_std = np.std(all_sim_returns)
                    sim_skew = stats.skew(all_sim_returns)
                    sim_kurt = stats.kurtosis(all_sim_returns)
                    

                    st.session_state.dist_comparison_data = {
                        'empirical_returns': empirical_returns,
                        'simulated_returns': all_sim_returns,
                        'empirical_stats': {
                            'mean': emp_mean,
                            'std': emp_std,
                            'skew': emp_skew,
                            'kurt': emp_kurt
                        },
                        'simulated_stats': {
                            'mean': sim_mean,
                            'std': sim_std,
                            'skew': sim_skew,
                            'kurt': sim_kurt
                        }
                    }
                    
                    st.success("✅ Simulation complete!")
                    
                else:
                    st.error(f"Could not find simulation script at: {sim_script_path}")
                    
            except Exception as e:
                st.error(f"Error during distribution comparison: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    

    if 'dist_comparison_data' in st.session_state and st.session_state.dist_comparison_data is not None:
        comp_data = st.session_state.dist_comparison_data
        

        st.write("**Statistical Comparison:**")
        stats_data = {
            'Metric': ['Mean', 'Std Dev', 'Skewness', 'Kurtosis'],
            'Empirical': [
                f"{comp_data['empirical_stats']['mean']:.6f}",
                f"{comp_data['empirical_stats']['std']:.6f}",
                f"{comp_data['empirical_stats']['skew']:.3f}",
                f"{comp_data['empirical_stats']['kurt']:.3f}"
            ],
            'Simulated': [
                f"{comp_data['simulated_stats']['mean']:.6f}",
                f"{comp_data['simulated_stats']['std']:.6f}",
                f"{comp_data['simulated_stats']['skew']:.3f}",
                f"{comp_data['simulated_stats']['kurt']:.3f}"
            ],
            'Difference': [
                f"{comp_data['simulated_stats']['mean'] - comp_data['empirical_stats']['mean']:.6f}",
                f"{comp_data['simulated_stats']['std'] - comp_data['empirical_stats']['std']:.6f}",
                f"{comp_data['simulated_stats']['skew'] - comp_data['empirical_stats']['skew']:.3f}",
                f"{comp_data['simulated_stats']['kurt'] - comp_data['empirical_stats']['kurt']:.3f}"
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        

        # Use matplotlib for all visualizations
        if HAS_MATPLOTLIB:
            import matplotlib.pyplot as plt
            plt.style.use('dark_background')

        st.write("**Return Distribution Comparison:**")

        all_returns = np.concatenate([comp_data['empirical_returns'], comp_data['simulated_returns']])
        min_val = np.min(all_returns)
        max_val = np.max(all_returns)
        bins = np.linspace(min_val, max_val, 100)
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='black')
        ax.set_facecolor('black')
        
        ax.hist(comp_data['empirical_returns'], bins=bins, alpha=0.6, label='Empirical', 
               density=True, color='cyan', edgecolor='white', linewidth=0.5)
        ax.hist(comp_data['simulated_returns'], bins=bins, alpha=0.6, label='Simulated (Model)', 
               density=True, color='orange', edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('Log Return', fontsize=12, color='white')
        ax.set_ylabel('Density', fontsize=12, color='white')
        ax.set_title('Return Distribution: Empirical vs Simulated Model', 
                    fontsize=14, fontweight='bold', color='white')
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax.grid(True, alpha=0.3, color='gray')
        ax.tick_params(colors='white')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.write("**Q-Q Plot (Quantile-Quantile Comparison):**")

        n_points = min(len(comp_data['empirical_returns']), len(comp_data['simulated_returns']), 10000)
        emp_quantiles = np.linspace(0, 1, n_points)
        
        emp_values = np.quantile(comp_data['empirical_returns'], emp_quantiles)
        sim_values = np.quantile(comp_data['simulated_returns'], emp_quantiles)
        
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        ax.set_facecolor('black')
        
        ax.scatter(emp_values, sim_values, alpha=0.5, s=3, color='cyan', edgecolors='white', linewidths=0.1)

        min_qq = min(np.min(emp_values), np.min(sim_values))
        max_qq = max(np.max(emp_values), np.max(sim_values))
        ax.plot([min_qq, max_qq], [min_qq, max_qq], 'r--', linewidth=2, label='Perfect Match (y=x)')
        
        ax.set_xlabel('Empirical Quantiles', fontsize=12, color='white')
        ax.set_ylabel('Simulated Quantiles', fontsize=12, color='white')
        ax.set_title('Q-Q Plot: Empirical vs Simulated Model', 
                    fontsize=14, fontweight='bold', color='white')
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax.grid(True, alpha=0.3, color='gray')
        ax.tick_params(colors='white')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.write("**Tail Behavior Comparison:**")

        tail_percentiles = np.arange(1, 100, 1)
        emp_tail = np.percentile(comp_data['empirical_returns'], tail_percentiles)
        sim_tail = np.percentile(comp_data['simulated_returns'], tail_percentiles)
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='black')
        ax.set_facecolor('black')
        
        ax.plot(tail_percentiles, emp_tail, linewidth=2, label='Empirical', color='cyan')
        ax.plot(tail_percentiles, sim_tail, linewidth=2, label='Simulated', color='orange', linestyle='--')
        
        ax.set_xlabel('Percentile', fontsize=12, color='white')
        ax.set_ylabel('Return Value', fontsize=12, color='white')
        ax.set_title('Percentile Comparison: Empirical vs Simulated Model', 
                    fontsize=14, fontweight='bold', color='white')
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax.grid(True, alpha=0.3, color='gray')
        ax.tick_params(colors='white')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Matplotlib not available. Please install matplotlib to view visualizations.")

def display_results(results, config):
    """Display simulation results in Streamlit"""
    if results is None or config is None:
        st.warning("No results to display")
        return
    
    st.header("📊 Simulation Results")
    
    retirement_ages = results.get('retirement_ages')
    if retirement_ages is None:
        st.warning("No retirement age data available")
        return
    
    retirement_ages = np.array(retirement_ages)
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
    
    # Principal Lookup Table
    st.subheader("💰 Required Principal by Retirement Age")
    required_principal_data = results.get('required_principal_data')
    if required_principal_data:
        df_principal = pd.DataFrame(required_principal_data)
        df_display = df_principal.copy()
        
        # Format currency columns
        for col in ['principal_real', 'principal_nominal', 'spending_real',
                   'spending_nominal', 'net_withdrawal_real', 'net_withdrawal_nominal']:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"${x:,.2f}")
        
        if 'swr' in df_display.columns:
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
    st.subheader("📈 Retirement Age Statistics")
    if valid_ages.size > 0:
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Median Retirement Age", f"{median_age:.1f}" if not np.isnan(median_age) else "N/A")
        with col2:
            st.metric("Ever Retired", f"{pct_ever_retired:.2f}%")
        with col3:
            st.metric("10th Percentile", f"{p10:.1f}" if not np.isnan(p10) else "N/A")
        with col4:
            st.metric("90th Percentile", f"{p90:.1f}" if not np.isnan(p90) else "N/A")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("25th Percentile", f"{p25:.1f}" if not np.isnan(p25) else "N/A")
        with col2:
            st.metric("75th Percentile", f"{p75:.1f}" if not np.isnan(p75) else "N/A")
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
    
    # Utility Metrics
    if config.enable_utility_calculations and results.get('utility_ex_ante') is not None:
        st.subheader("💡 Utility Metrics")
        baseline_savings_rate = config.savings_rate
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Equivalent Savings Rate", f"{baseline_savings_rate*100:.2f}%")
        with col2:
            if results.get('ce_annual') is not None:
                st.metric("Certainty Equivalent (Annual, Real $)", f"${results['ce_annual']:,.2f}")
        
        st.info("Note: Equivalent Savings Rate shows the savings rate used in the simulation. Utility calculated using EX-ANTE approach.")
    
    # Amortization Statistics
    if config.use_amortization and results.get('amortization_stats_list'):
        st.subheader("📉 Amortization-Based Withdrawal Statistics")
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
        except Exception as e:
            st.warning(f"Error displaying amortization statistics: {str(e)}")
    
    # Principal plots - add labor income and amortization tabs when available
    has_earnings_results = bool(results.get('earnings_real_list')) and config.annual_income_real > 0
    tab_labels = ["Principal Requirements", "Retirement Age Distribution", "Cumulative Probability"]
    if has_earnings_results:
        tab_labels.append("Labor Income Dynamics")
    if config.use_amortization and results.get('amortization_stats_list'):
        tab_labels.append("Amortization Analysis")
    tabs = st.tabs(tab_labels)
    tab1, tab2, tab3 = tabs[:3]
    next_tab_idx = 3
    tab_earnings = tabs[next_tab_idx] if has_earnings_results else None
    if has_earnings_results:
        next_tab_idx += 1
    tab_amortization = tabs[next_tab_idx] if config.use_amortization and results.get('amortization_stats_list') else None
    
    with tab1:
        st.write("**Principal Requirements by Retirement Age**")
        if required_principal_data:
            ages = [row['age'] for row in required_principal_data]
            principals_nominal = [row['principal_nominal'] for row in required_principal_data]
            principals_real = [row['principal_real'] for row in required_principal_data]
            swr = [row['swr'] for row in required_principal_data]
            
            col1, col2 = st.columns(2)
            with col1:
                fig_nominal = create_plot_required_principal_nominal(ages, principals_nominal, swr, config, median_age)
                if fig_nominal:
                    if HAS_PLOTLY:
                        st.plotly_chart(fig_nominal, width='stretch')
                    else:
                        st.pyplot(fig_nominal)
                        plt.close(fig_nominal)
            with col2:
                fig_real = create_plot_required_principal_real(ages, principals_real, swr, config, median_age)
                if fig_real:
                    if HAS_PLOTLY:
                        st.plotly_chart(fig_real, width='stretch')
                    else:
                        st.pyplot(fig_real)
                        plt.close(fig_real)
        else:
            st.warning("No principal data available")
    
    with tab2:
        if valid_ages.size > 0:
            fig_dist = create_plot_retirement_age_distribution(valid_ages, median_age)
            if fig_dist:
                if HAS_PLOTLY:
                    st.plotly_chart(fig_dist, width='stretch')
                else:
                    st.pyplot(fig_dist)
                    plt.close(fig_dist)
        else:
            st.warning("No retirement ages to plot")
    
    with tab3:
        if valid_ages.size > 0:
            fig_cum = create_plot_cumulative_retirement_probability(retirement_ages, config, median_age, config.num_outer)
            if fig_cum:
                if HAS_PLOTLY:
                    st.plotly_chart(fig_cum, width='stretch')
                else:
                    st.pyplot(fig_cum)
                    plt.close(fig_cum)
        else:
            st.warning("No retirement ages to plot")

    if tab_earnings is not None:
        with tab_earnings:
            with st.spinner("Running GKOS earnings simulation…"):
                bench_fig, bench_stats = create_plot_gkos_benchmark(config, n_workers=20_000)

            if bench_fig is not None:
                st.plotly_chart(bench_fig, width='stretch')
                if isinstance(bench_stats, dict):
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Peak Age", str(bench_stats.get('peak_age', '—')))
                    with c2:
                        v = bench_stats.get('peak_p90')
                        st.metric("P90 at Peak", f"${v:,.0f}" if v else '—')
                    with c3:
                        age_s = bench_stats.get('age_start', config.initial_age)
                        v = bench_stats.get('med_start')
                        st.metric(f"Median Age {int(age_s)}", f"${v:,.0f}" if v else '—')
                    with c4:
                        v = bench_stats.get('med55')
                        st.metric("Median Age 55", f"${v:,.0f}" if v else '—')
                    fz = bench_stats.get('avg_zero_frac')
                    if fz is not None:
                        st.caption(f"Non-employment rate: {fz*100:.1f}%  ·  "
                                   f"Percentiles among positive earners  ·  n=20,000 workers")
            elif isinstance(bench_stats, str):
                st.warning(f"Could not generate earnings chart: {bench_stats}")

            moment_result = create_earnings_moment_summary(results.get('earnings_real_list'), config)
            if isinstance(moment_result, tuple) and not moment_result[0].empty:
                earnings_moments, _, _ = moment_result
                st.write("**Distribution by Age Bucket**")
                st.dataframe(earnings_moments, use_container_width=True, hide_index=True)
                st.caption("Percentiles among positive earners per age bucket.")
    
    # Amortization tab
    if tab_amortization is not None:
        with tab_amortization:
            create_amortization_visualizations(results['amortization_stats_list'], config, 
                                               results.get('final_bequest_nominal', []), 
                                               results.get('final_bequest_real', []))
    
    # Download detailed simulations
    if results.get('detailed_simulations') and config.generate_csv_summary:
        st.subheader("📥 Download Detailed Results")
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
        st.subheader("📥 Download Utility Metrics")
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
    
    # Bequest statistics
    final_bequest_real = results.get('final_bequest_real')
    if final_bequest_real is not None and len(final_bequest_real) > 0:
        st.subheader("💎 Final Bequest Statistics")
        bequest_array = np.array(final_bequest_real)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Bequest (Real $)", f"${np.mean(bequest_array):,.2f}")
        with col2:
            st.metric("Median Bequest (Real $)", f"${np.median(bequest_array):,.2f}")
        with col3:
            st.metric("10th Percentile", f"${np.percentile(bequest_array, 10):,.2f}")

def run_simulation(config):
    """Run the simulation with progress tracking"""

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

    st.session_state.config = config
    

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

        config.validate()
        params = config.params


        ensure_mortality_loaded(config)
        if config.use_stochastic_mortality:
            qx = getattr(config, '_mortality_qx', None)
            n_ages = len(qx) if qx is not None else 0
            st.info(f"**Stochastic mortality on:** Death each period is drawn from the SSA table ({n_ages} ages). Death Age is not used.")
        

        with st.expander("📋 Configuration Summary", expanded=False):
            st.json({
                "Basic": {
                    "Initial Age": config.initial_age,
                    "Death Age": config.death_age,
                    "Stochastic mortality (SSA)": config.use_stochastic_mortality,
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
        

        if config.seed is not None:
            rng = np.random.default_rng(seed=config.seed)
        else:
            rng = np.random.default_rng()
        

        if config.use_amortization:
            if config.amortization_expected_return is None:
                with st.status("Calculating expected return for amortization...", expanded=False) as status:
                    monthly_returns = bootstrap_data[0] if bootstrap_data is not None else None
                    monthly_inflation = bootstrap_data[1] if bootstrap_data is not None else None
                    config.amortization_expected_return = calculate_expected_return_from_data(
                        monthly_returns=monthly_returns,
                        monthly_inflation=monthly_inflation,
                        params=params if not config.use_block_bootstrap else None,
                        mean_inflation=config.mean_inflation_geometric,
                        expected_real_annual_override=getattr(config, 'bootstrap_geometric_mean_override', None),
                    )

                    status.update(label=f"✅ Expected real return (auto from data): {config.amortization_expected_return*100:.2f}%", state="complete")
        

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
        

        required_principal_data = []
        for age, principal_real in required_principal_table.items():
            principal_nominal = calculate_nominal_value(
                principal_real, config.initial_age, age, mean_inflation_arithmetic)
            
            # Calculate actual withdrawal amount - use amortization if enabled
            if config.use_amortization:
                # Calculate remaining years
                remaining_years = config.death_age - age
                
                # Get expected return for amortization
                expected_return = getattr(config, 'amortization_expected_return', None)
                if expected_return is None:
                    expected_return = 0.03  # Fallback
                
                # Get desired bequest
                desired_bequest_real = getattr(config, 'amortization_desired_bequest', 0.0)
                
                # Calculate amortized withdrawal for first year
                annual_withdrawal_real = calculate_amortized_withdrawal(
                    principal_real, remaining_years, expected_return, desired_bequest_real
                )
                net_withdrawal_real = annual_withdrawal_real
            else:
                net_withdrawal_real = config.spending_real
            
            if config.include_social_security and age >= config.social_security_start_age:
                net_withdrawal_real = max(0.0, net_withdrawal_real - config.social_security_real)
            
            swr_val = ((net_withdrawal_real / principal_real) * 100.0
                      if principal_real > 0 else np.nan)
            
            # Calculate nominal spending - use amortized withdrawal if enabled
            if config.use_amortization:
                # Use the amortized withdrawal amount (before SS adjustment)
                nominal_spending = calculate_nominal_value(
                    annual_withdrawal_real, config.initial_age, age, mean_inflation_arithmetic)
            else:
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
        

        st.header("Stage 2: Running Accumulation Simulations")
        

        accum_progress_bar = st.progress(0.0)
        accum_status_text = st.empty()
        

        accum_status_text.text("Starting accumulation simulations...")
        

        import time as time_module
        stage2_start_time = time_module.time()
        stage2_times = []
        

        def update_accumulation_progress(current, total):

            if total <= 0:
                return
            current = max(0, min(current, total))
            update_every = max(1, total // 200)
            if current not in (1, total) and current % update_every != 0:
                return
            progress_pct = float(current) / float(total)

            progress_pct = max(0.0, min(1.0, progress_pct))
            

            accum_progress_bar.progress(progress_pct)
            

            if current > 1:
                elapsed = time_module.time() - stage2_start_time
                avg_time_per_sim = elapsed / current
                remaining = total - current
                eta_seconds = avg_time_per_sim * remaining
                eta_str = format_time(eta_seconds)
                accum_status_text.text(f"Simulation {current}/{total} ({progress_pct*100:.1f}%) | ETA: {eta_str}")
            else:
                accum_status_text.text(f"Running simulation {current}/{total} ({progress_pct*100:.1f}%)")
        

        (retirement_ages, ever_retired, detailed_simulations,
         final_bequest_nominal, final_bequest_real, consumption_streams,
         amortization_stats_list, earnings_nominal_list, earnings_real_list) = run_accumulation_simulations(
            config, params, results['principal_lookup'], rng, bootstrap_data,
            progress_callback=update_accumulation_progress)
        

        accum_progress_bar.empty()
        accum_status_text.empty()
        
        results['retirement_ages'] = retirement_ages
        results['ever_retired'] = ever_retired
        results['detailed_simulations'] = detailed_simulations
        results['final_bequest_nominal'] = final_bequest_nominal
        results['final_bequest_real'] = final_bequest_real
        results['consumption_streams'] = consumption_streams
        results['amortization_stats_list'] = amortization_stats_list
        results['earnings_nominal_list'] = earnings_nominal_list
        results['earnings_real_list'] = earnings_real_list
        

        accum_progress_bar.empty()
        accum_status_text.empty()
        st.success("✅ Accumulation simulations complete!")
        

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

    except FileNotFoundError as e:
        st.error(f"❌ File Not Found: {str(e)}\n\nPlease check that the bootstrap CSV file exists at the specified path.")
        logger.exception("File not found error")
        st.session_state.simulation_results = None

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

    except Exception as e:
        st.error(f"❌ Error during simulation: {str(e)}\n\nPlease check the logs for more details.")
        logger.exception("Simulation error")
        st.session_state.simulation_results = None
        import traceback
        with st.expander("🔍 Detailed Error Information"):
            st.code(traceback.format_exc())

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

# Call main() for Streamlit (Streamlit runs the entire script)
main()
