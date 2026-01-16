# Streamlit App for Lifecycle Retirement Simulation

This is a Streamlit web application version of the Lifecycle Retirement Simulation.

## Running the App

1. **Install Streamlit** (if not already installed):
   ```bash
   pip install streamlit
   ```

2. **Run the app**:
   ```bash
   python -m streamlit run app.py
   ```
   
   Or if `streamlit` is in your PATH:
   ```bash
   streamlit run app.py
   ```

   The app will open in your default web browser, typically at `http://localhost:8501`
   
   **Note**: On Windows, if the `streamlit` command is not recognized, use `python -m streamlit run app.py` instead.

## Features

- **Interactive Configuration**: All simulation parameters can be configured through the sidebar UI
- **Real-time Progress**: Progress bars and status updates during simulation
- **Visual Results**: Interactive plots and charts showing simulation results
- **Data Export**: Download results as CSV files
- **Comprehensive Statistics**: Detailed statistics on retirement age, utility metrics, and amortization

## Configuration Sections

The sidebar is organized into the following sections:

1. **Basic Parameters**: Age, portfolio, income, and spending settings
2. **Social Security**: Social Security benefit configuration
3. **Simulation Settings**: Number of simulations, success target, random seed
4. **Retirement Age Range**: Minimum and maximum retirement ages to consider
5. **Model Selection**: Choose between block bootstrap (historical data) or parametric model
6. **Parametric Model Parameters**: Bates/Heston model parameters (when not using bootstrap)
7. **GKOS Earnings Parameters**: Earnings model configuration
8. **Utility Parameters**: CRRA utility function settings
9. **Withdrawal Strategy**: Fixed spending vs. amortization-based withdrawal
10. **Output Settings**: CSV export options
11. **Advanced Settings**: Principal deviation threshold, multiprocessing workers

## Usage Tips

- **Start with smaller simulations**: Use lower `num_outer` and `num_nested` values initially to test configurations quickly
- **Check estimated runtime**: The app shows an estimated runtime before running the simulation
- **Use Clear Results**: Click "Clear Results" to remove previous simulation results and start fresh
- **Download results**: Use the download buttons to save CSV files with your simulation data

## Requirements

All dependencies from the original simulation are required, plus:
- `streamlit` (for the web interface)

## Notes

- The app uses the same simulation engine as the command-line version
- Cython acceleration is automatically used if available
- Results are stored in session state and persist until you clear them or restart the app
- Large simulations may take several minutes to complete

