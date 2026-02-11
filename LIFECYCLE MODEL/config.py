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
        self.annual_income_real = 50_000
        self.spending_real = 50_000
        self.savings_rate = 0.25
        self.social_security_real = 25_000.0
        self.social_security_start_age = 67
        self.include_social_security = False






        self.num_outer = 10
        self.num_nested = 50
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
        


        self.gkos_params = {
            'RHO': 0.958,

            'ETA_P1': 0.219,
            'ETA_MU1': -0.147,
            'ETA_SIGMA1': 0.463,
            'ETA_MU2': 0.041,
            'ETA_SIGMA2': 0.148,

            'EPS_P1': 0.118,
            'EPS_MU1': -0.554,
            'EPS_SIGMA1': 1.433,
            'EPS_MU2': 0.074,
            'EPS_SIGMA2': 0.116,
            'SIGMA_Z0': 0.272,

            'AGE_PROFILE_A': -0.0005,
            'AGE_PEAK': 47.5,

            'SIGMA_ALPHA': 0.189,
            'SIGMA_BETA': 0.013,
            'HIP_CORR_AB': -0.01,

            'NU_A': -3.12,
            'NU_B': 0.005,
            'NU_C': -1.15,
            'NU_D': -0.015,
        }
        

        self.enable_utility_calculations = True
        self.gamma = 2.0
        self.beta = 0.98
        self.k_bequest = 10000.0
        self.theta = 0.5
        self.household_size = 1.0
        

        self.use_amortization = True
        self.amortization_expected_return = None
        self.amortization_min_spending_threshold = 0.5
        self.amortization_desired_bequest = 0.0

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

