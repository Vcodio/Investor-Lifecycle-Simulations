"""
Utility Functions Module

This module provides helper functions for data conversion, CSV export, and display.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False


def convert_geometric_to_arithmetic(mean_geometric, std_dev):
    """Convert geometric mean to arithmetic mean"""
    return mean_geometric + 0.5 * (std_dev**2)


def calculate_nominal_value(real_value, current_age_in_years, target_age_in_years,
                           mean_inflation):
    """Calculate nominal value from real value given inflation"""
    years = target_age_in_years - current_age_in_years
    return real_value * (1 + mean_inflation)**years


def print_rich_table(df, title):
    """Print a pandas DataFrame as a rich table"""
    if HAS_RICH:
        table = Table(title=title, title_style="bold magenta",
                     header_style="bold cyan")
        for col in df.columns:
            table.add_column(str(col), justify="right")
        for _, row in df.iterrows():
            table.add_row(*[str(item) for item in row])
        console.print(table)
    else:
        print(f"\n--- {title} ---")
        print(df.to_string())


def export_to_csv(data, filename, output_dir='Lifecycle Outputs', subdirectory=None):
    """Export data to CSV file"""
    try:
        df = pd.DataFrame(data)
        import os
        if subdirectory:
            output_dir = os.path.join(output_dir, subdirectory)
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Data successfully exported to '{filepath}'")
    except Exception as e:
        logger.error(f"Error exporting to CSV '{filepath}': {e}")


def export_detailed_simulations_to_csv(sim_data, filename, output_dir='Lifecycle Outputs', subdirectory=None):
    """Export detailed simulation paths to CSV"""
    if not sim_data:
        return
    try:
        flat_data = [entry for sim in sim_data for entry in sim]
        df = pd.DataFrame(flat_data)
        all_expected_cols = [
            'SIM_ID', 'AGE', 'RETIRED?', 'PORTFOLIO_VALUE', 'VOLATILITY',
            'REQUIRED_REAL_PRINCIPAL', 'WITHDRAWAL_RATE', 'REQUIRED_NOMINAL_PRINCIPAL',
            'NOMINAL_DESIRED_CONSUMPTION', 'REAL_DESIRED_CONSUMPTION',
            'ANNUAL_INFLATION', 'CUMULATIVE_INFLATION',
            'REAL_SOCIAL_SECURITY_BENEFIT', 'NOMINAL_SOCIAL_SECURITY_BENEFIT',
            'SAVINGS_RATE', 'SALARY_REAL', 'SALARY_NOMINAL',
            'DOLLARS_SAVED', 'MONTHLY_PORTFOLIO_RETURN', 'CUMULATIVE_PORTFOLIO_RETURN'
        ]
        for col in all_expected_cols:
            if col not in df.columns:
                df[col] = None
        import os
        if subdirectory:
            output_dir = os.path.join(output_dir, subdirectory)
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False, columns=all_expected_cols)
        logger.info(f"Detailed simulations exported to '{filepath}'")
    except Exception as e:
        logger.error(f"Error exporting detailed simulations: {e}")

