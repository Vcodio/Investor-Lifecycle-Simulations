"""
Configuration classes for Lifecycle Retirement Simulation
"""
import multiprocessing as mp


class SimulationConfig:
    """Configuration class for simulation parameters"""
    def __init__(self):
        self.initial_age = 20
        self.death_age = 100
        self.initial_portfolio = 100_000
        self.annual_income_real = 50_000
        self.spending_real = 50_000
        self.savings_rate = 0.25  # Savings rate during accumulation phase (fraction of income)
        self.social_security_real = 25_000.0
        self.social_security_start_age = 67
        self.include_social_security = False

        # Monte Carlo simulation sizes:
        # - num_outer: number of full lifecycle paths (outer simulations)
        # - num_nested: number of retirement-phase re-simulations per candidate
        #   retirement age when checking success rates.
        # Larger values improve stability of estimates at the cost of runtime.
        self.num_outer = 10 # number of full lifecycle paths (outer simulations)
        self.num_nested = 50 # number of retirement-phase re-simulations per candidate retirement age when checking success rates.
        self.success_target = 0.95 # target success rate for retirement age candidates
        self.generate_csv_summary = False # toggle to generate a CSV summary of the simulation results
        self.num_sims_to_export = 50 # number of simulations to export to a CSV file
        self.seed = None # random seed for the simulation   
        self.num_workers = max(1, mp.cpu_count() - 1) # number of workers for parallel simulations
        self.output_directory = 'Lifecycle Outputs' # directory to save the simulation results
        self.use_principal_deviation_threshold = True # toggle to use a principal deviation threshold to determine if a simulation is successful
        self.principal_deviation_threshold = 0.07  # 7% deviation threshold (as fraction)
        self.retirement_age_min = 30  # Must be >= initial_age
        self.retirement_age_max = 70
        # Block bootstrap configuration (historical returns + inflation from CSV)
        self.use_block_bootstrap = True
        self.bootstrap_csv_path = 'data/TFP - Block Bootstrap.csv'
        self.portfolio_column_name = "Three Fund Portfolio"
        self.inflation_column_name = 'Inflation'
        self.block_length_years = 5  # Block length in years (default 10 years = 120 months)
        self.block_overlapping = True
        
        # Parametric model configuration (used when NOT using block bootstrap,
        # or as a fallback if bootstrap data cannot be loaded).
        # Inflation assumptions (annual, geometric mean and std dev):
        self.mean_inflation_geometric = 0.025
        self.std_inflation = 0.03

        # Bates/Heston model parameters (nominal return process)
        self.params = {
            "mu": 0.0928,
            "kappa": 1.189,
            "theta": 0.0201,
            "nu": 0.0219,
            "rho": -0.714,
            "v0": 0.0201,
            "lam": 0.353,
            "mu_J": -0.007,
            "sigma_J": 0.0328,
        }
        
        # GKOS earnings model parameters
        self.gkos_params = {
            'RHO': 0.97,  # Persistence parameter (AR(1) coefficient)
            'SIGMA_ETA': 0.15,  # Standard deviation of persistent shock
            'MIXTURE_PROB_NORMAL': 0.95,  # Probability of "normal" state
            'MIXTURE_PROB_TAIL': 0.05,  # Probability of "tail event" state
            'SKEWNESS_PARAM': -2.5,  # Negative mean for tail events
            'TAIL_VARIANCE_MULTIPLIER': 25.0,  # Tail events variance multiplier
            'SIGMA_EPSILON': 0.05,  # Standard deviation of transitory shock (Laplace)
            'SIGMA_Z0': 0.3,  # Initial heterogeneity variance
            'AGE_PROFILE_A': -0.0005,  # Quadratic coefficient
            'AGE_PROFILE_B': 0.04,  # Linear coefficient
            'AGE_PEAK': 47.5,  # Age at which earnings peak
        }
        
        # Utility model parameters
        self.enable_utility_calculations = True  # Toggle to enable/disable utility and CE calculations
        self.gamma = 2.0  # CRRA risk aversion parameter
        self.beta = 0.98  # Time discount factor
        self.k_bequest = 10000.0  # Bequest threshold (minimum bequest value for utility)
        self.theta = 0.5  # Bequest weight parameter
        self.household_size = 1.0  # Household size (affects utility scaling)
        
        # Withdrawal strategy parameters
        self.use_amortization = True  # Toggle between fixed real spending and amortization-based withdrawal
        self.amortization_expected_return = 0.06  # Expected real return for amortization calculation (None = auto-calculate from historical/model)
        self.amortization_min_spending_threshold = 0.5  # Minimum spending as fraction of initial spending (for statistics)
        self.amortization_desired_bequest = 0.0  # Desired bequest amount (real $) at death. If 0, portfolio is consumed to zero.

    def validate(self):
        """Validate configuration parameters"""
        import logging
        logger = logging.getLogger(__name__)
        
        errors = []
        if not (0 < self.initial_age < self.death_age):
            errors.append("Initial age must be between 0 and death age")
        if self.initial_portfolio <= 0:
            errors.append("Initial portfolio must be positive")
        if not (self.initial_age <= self.retirement_age_min <= self.retirement_age_max <= self.death_age):
            errors.append(f"Retirement age range ({self.retirement_age_min}-{self.retirement_age_max}) must be within initial_age ({self.initial_age}) and death_age ({self.death_age})")
        if self.retirement_age_min >= self.retirement_age_max:
            errors.append(f"retirement_age_min ({self.retirement_age_min}) must be less than retirement_age_max ({self.retirement_age_max})")
        if errors:
            raise ValueError("Parameter validation failed:\n" + "\n".join(errors))
        logger.info("All parameters validated successfully")

