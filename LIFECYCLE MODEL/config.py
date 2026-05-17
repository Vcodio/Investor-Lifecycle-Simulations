"""
Configuration classes for Lifecycle Retirement Simulation
"""
import multiprocessing as mp


class SimulationConfig:
    """Configuration class for simulation parameters"""
    def __init__(self):
        self.initial_age = 20
        self.death_age = 100

        self.use_stochastic_mortality = False
        self.mortality_table_path = 'data/ssa_period_life_table.csv'
        self.mortality_sex = 'average'
        self.initial_portfolio = 100_000
        self.annual_income_real = 30_000
        self.spending_real = 50_000
        self.savings_rate = 0.25
        self.social_security_real = 25_000.0
        self.social_security_start_age = 67
        self.include_social_security = False






        self.num_outer = 1000
        self.num_nested = 500
        self.success_target = 0.95
        self.generate_csv_summary = False
        self.num_sims_to_export = 50
        self.seed = None
        self.num_workers = max(1, mp.cpu_count() - 1)
        self.output_directory = 'Lifecycle Outputs'
        self.use_principal_deviation_threshold = True
        self.principal_deviation_threshold = 0.07
        self.retirement_age_min = 30
        self.retirement_age_max = 70

        self.use_block_bootstrap = True
        self.bootstrap_csv_path = 'data/TFP - Block Bootstrap.csv'
        self.portfolio_column_name = "Three Fund Portfolio"
        self.inflation_column_name = 'Inflation'
        self.block_length_years = 5
        self.block_overlapping = True


        self.bootstrap_geometric_mean_override = None




        self.mean_inflation_geometric = 0.025
        self.std_inflation = 0.03


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
        


        # GKOSBenchmarkParams field names and Table IV Model (8) values.
        # typical_worker=True → alpha=beta=z0=0 (Anarkulova, Cederburg, O'Doherty 2023 base case).
        # additional_log_intercept calibrated so P90 at age 47 ≈ $160k (2010 USD, CPI-adjusted).
        self.gkos_params = {
            "typical_worker": True,
            "additional_log_intercept": 7.427,
            "rho": 0.958,
            "p_z": 0.219,
            "mu_eta1": -0.147,
            "sigma_eta1": 0.457,
            "sigma_eta2": 0.139,
            "sigma_z0": 0.667,
            "lambda_nu": 0.001,
            "p_eps": 0.126,
            "mu_eps1": 0.236,
            "sigma_eps1": 0.343,
            "sigma_eps2": 0.063,
            "sigma_alpha": 0.298,
            "sigma_beta": 0.185,
            "corr_alpha_beta": 0.976,
            "g_a0": 2.6291,
            "g_a1": 0.7300,
            "g_a2": -0.1692,
            "a_nu": -3.2131,
            "b_nu": -1.0235,
            "c_nu": -3.2602,
            "d_nu": -2.1656,
        }
        

        self.enable_utility_calculations = False
        self.gamma = 2.0
        self.beta = 0.98
        self.k_bequest = 10000.0
        self.theta = 0.5
        self.household_size = 1.0
        

        self.use_amortization = False
        self.amortization_expected_return = None
        self.amortization_min_spending_threshold = 0.5
        self.amortization_desired_bequest = 0.0
        
        # Debug flag for amortization calculations
        self.debug_amortization = False

    def validate(self):
        """Validate configuration parameters"""
        import logging
        logger = logging.getLogger(__name__)
        
        errors = []
        effective_max_age = 120 if getattr(self, 'use_stochastic_mortality', False) else self.death_age
        if not (0 < self.initial_age < effective_max_age):
            errors.append("Initial age must be between 0 and death age (or 120 when stochastic mortality is on)")
        if self.initial_portfolio <= 0:
            errors.append("Initial portfolio must be positive")
        if not (self.initial_age <= self.retirement_age_min <= self.retirement_age_max <= effective_max_age):
            errors.append(f"Retirement age range ({self.retirement_age_min}-{self.retirement_age_max}) must be within initial_age ({self.initial_age}) and max age ({effective_max_age})")
        if self.retirement_age_min >= self.retirement_age_max:
            errors.append(f"retirement_age_min ({self.retirement_age_min}) must be less than retirement_age_max ({self.retirement_age_max})")
        if errors:
            raise ValueError("Parameter validation failed:\n" + "\n".join(errors))
        logger.info("All parameters validated successfully")

