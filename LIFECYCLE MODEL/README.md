# Modular Lifecycle Retirement Simulation

This is a modular version of the Lifecycle Retirement Simulation with the following enhancements:

1. **Modular Architecture**: Code is organized into separate modules for better maintainability
2. **GKOS Earnings Model**: Integrated Guvenen, Karahan, Ozkan, and Song (GKOS, 2019) earnings dynamics
3. **Utility-Based Evaluation**: Includes CRRA utility function for consumption and bequest evaluation
4. **Removed Unemployment**: Unemployment model has been removed in favor of GKOS earnings dynamics

## Module Structure

- `config.py`: Configuration classes and parameters
- `bootstrap.py`: Block bootstrap sampling functionality
- `earnings.py`: GKOS earnings model implementation
- `utility.py`: Utility functions for consumption and bequest evaluation
- `savings_rate.py`: Equivalent savings rate calculation (find savings rate that produces target utility)
- `simulation.py`: Core simulation functions (withdrawal phase and accumulation phase)
- `cython_wrapper.py`: Cython wrapper for performance-critical functions
- `visualization.py`: Plotting and visualization functions
- `utils.py`: Helper functions for data export and display
- `main.py`: Main execution module that coordinates all components

## Usage

### Basic Usage

```python
from modular.main import main

# Run the simulation
main()
```

### Custom Configuration

```python
from modular.config import SimulationConfig
from modular.main import main, build_required_principal_table, run_accumulation_simulations

# Create custom configuration
config = SimulationConfig()
config.initial_age = 30
config.retirement_age_min = 45
config.retirement_age_max = 70
config.gamma = 5.0  # CRRA risk aversion
config.beta = 0.98  # Time discount factor
config.gkos_params['RHO'] = 0.97  # GKOS persistence parameter

# Validate and run
config.validate()
main()
```

## Key Features

### GKOS Earnings Model

The GKOS earnings model simulates realistic lifecycle earnings with:
- Persistent shocks (mixture of normals with tail events)
- Transitory shocks (Laplace distribution)
- Age-earnings profile (quadratic in age)
- Correlation with inflation

### Utility Model

The utility model evaluates outcomes using:
- CRRA (Constant Relative Risk Aversion) utility function
- Time discounting (beta)
- Bequest utility with weight parameter (theta)
- Household size scaling

**Note on Certainty Equivalent**: The certainty equivalent consumption is calculated from monthly consumption streams but is displayed and exported as **annual** consumption (multiplied by 12).

### Block Bootstrap

Historical returns and inflation can be sampled using block bootstrap to preserve:
- Historical correlations between returns and inflation
- Autocorrelation within blocks
- Non-parametric approach (alternative to parametric Bates/Heston model)

### Equivalent Savings Rate

The `savings_rate.py` module provides `find_equivalent_savings_rate()` function that finds the savings rate producing a target utility level. This can be used to compare different scenarios or find optimal savings rates.

## Configuration Parameters

### Simulation Parameters
- `initial_age`: Starting age for simulation
- `death_age`: Terminal age
- `initial_portfolio`: Starting portfolio value
- `spending_real`: Real spending amount per year
- `num_outer`: Number of outer loop simulations
- `num_nested`: Number of nested simulations for success rate calculation

### GKOS Earnings Parameters
- `RHO`: Persistence parameter (AR(1) coefficient)
- `SIGMA_ETA`: Standard deviation of persistent shock
- `MIXTURE_PROB_TAIL`: Probability of tail event
- `SKEWNESS_PARAM`: Mean for tail events
- `AGE_PEAK`: Age at which earnings peak

### Utility Parameters
- `gamma`: CRRA risk aversion parameter
- `beta`: Time discount factor
- `k_bequest`: Bequest threshold
- `theta`: Bequest weight parameter
- `household_size`: Household size (affects utility scaling)

## Output Structure

All outputs are saved in the `Lifecycle Outputs/` folder, organized into subdirectories:

```
Lifecycle Outputs/
├── Principal Requirements/          # Required principal tables and plots
│   ├── required_principal_table.csv
│   ├── required_principal_and_swr_nominal.png
│   └── required_principal_and_swr_real.png
├── Retirement Analysis/             # Retirement age analysis
│   ├── retirement_age_distribution.png
│   └── cumulative_prob_retiring_by_age.png
├── Utility Analysis/                # Utility metrics and visualizations
│   ├── utility_metrics.csv
│   ├── utility_distribution.png
│   ├── certainty_equivalent_distribution.png
│   └── certainty_equivalent_bar.png
└── Simulation Data/                 # Detailed simulation paths
    └── detailed_lifecycle_paths.csv
```

## Output

The simulation generates:
1. Required principal lookup table (for different retirement ages)
2. Retirement age distribution statistics
3. Utility metrics (mean/median utility, certainty equivalent - **displayed as annual**)
4. Visualization plots organized by category:
   - Principal requirements plots
   - Retirement age analysis plots
   - Utility analysis plots (distribution, bar charts)
5. CSV exports in organized subdirectories

## Dependencies

- numpy
- pandas
- matplotlib (for visualization)
- tqdm (for progress bars)
- rich (optional, for better table display)
- scipy (optional, for bimodality detection and optimization)
- cython (optional, for performance acceleration)

## Performance

The modular version maintains the same performance characteristics as the original:
- Cython acceleration (10-50x speedup) when available
- Multiprocessing support for parallel simulations
- Efficient block bootstrap implementation
- Warm start binary search for principal calculation

## Differences from Original

1. **Modular structure**: Code is split into logical modules
2. **GKOS earnings**: Replaces simple salary growth + unemployment model
3. **Utility evaluation**: Adds utility-based metrics (utility, certainty equivalent displayed as annual)
4. **No unemployment**: Unemployment model completely removed
5. **Better organization**: Easier to modify and extend individual components
6. **Organized outputs**: All outputs saved in organized folder structure
