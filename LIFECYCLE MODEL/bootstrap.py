"""
Block Bootstrap Module

This module provides functions for block bootstrap sampling of historical returns and inflation data.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_and_convert_to_monthly_returns(csv_path, portfolio_column_name, inflation_column_name):
    """
    Load CSV file and convert daily returns to monthly returns.
    
    Args:
        csv_path: Path to CSV file
        portfolio_column_name: Name of column containing portfolio values
        inflation_column_name: Name of column containing inflation values
    
    Returns:
        tuple: (monthly_returns, monthly_inflation, monthly_dates)
    """
    try:
        logger.info(f"[DEBUG] Loading CSV file: {csv_path}")
        logger.info(f"[DEBUG] Looking for portfolio column: '{portfolio_column_name}'")
        logger.info(f"[DEBUG] Looking for inflation column: '{inflation_column_name}'")
        
        df = pd.read_csv(csv_path)
        logger.info(f"[DEBUG] CSV loaded successfully. Shape: {df.shape}, Columns: {list(df.columns)}")
        

        if portfolio_column_name not in df.columns:
            logger.error(f"[DEBUG] Portfolio column '{portfolio_column_name}' not found!")
            logger.error(f"[DEBUG] Available columns: {list(df.columns)}")
            raise ValueError(f"Column '{portfolio_column_name}' not found in CSV. Available columns: {list(df.columns)}")
        if inflation_column_name not in df.columns:
            logger.error(f"[DEBUG] Inflation column '{inflation_column_name}' not found!")
            logger.error(f"[DEBUG] Available columns: {list(df.columns)}")
            raise ValueError(f"Column '{inflation_column_name}' not found in CSV. Available columns: {list(df.columns)}")
        
        logger.info(f"[DEBUG] Both columns found. Portfolio column has {len(df[portfolio_column_name])} values, "
                   f"Inflation column has {len(df[inflation_column_name])} values")
        

        logger.info(f"[DEBUG] Converting Date column to datetime...")
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        

        na_count = df['Date'].isna().sum()
        if na_count > 0:
            logger.warning(f"[DEBUG] {na_count} dates could not be parsed. Dropping rows with invalid dates.")
            df = df.dropna(subset=['Date'])
            logger.info(f"[DEBUG] After dropping invalid dates, DataFrame shape: {df.shape}")
        
        logger.info(f"[DEBUG] Date range: {df['Date'].min()} to {df['Date'].max()}")
        


        portfolio_values = df[portfolio_column_name].values
        logger.info(f"[DEBUG] Portfolio values: min={np.min(portfolio_values):.2f}, max={np.max(portfolio_values):.2f}, "
                   f"first={portfolio_values[0]:.2f}, last={portfolio_values[-1]:.2f}")
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        logger.info(f"[DEBUG] Calculated {len(daily_returns)} daily returns. "
                   f"Mean={np.mean(daily_returns):.6f}, Std={np.std(daily_returns):.6f}, "
                   f"Min={np.min(daily_returns):.6f}, Max={np.max(daily_returns):.6f}")
        

        inflation_values = df[inflation_column_name].values
        logger.info(f"[DEBUG] Inflation values: min={np.min(inflation_values):.6f}, max={np.max(inflation_values):.6f}, "
                   f"first={inflation_values[0]:.6f}, last={inflation_values[-1]:.6f}")
        daily_inflation_changes = np.diff(inflation_values) / inflation_values[:-1]
        logger.info(f"[DEBUG] Calculated {len(daily_inflation_changes)} daily inflation changes. "
                   f"Mean={np.mean(daily_inflation_changes):.6f}, Std={np.std(daily_inflation_changes):.6f}")
        

        dates = df['Date'].values[1:]
        logger.info(f"[DEBUG] Using {len(dates)} dates (excluded first date due to diff)")
        

        df_returns = pd.DataFrame({
            'Date': dates,
            'DailyReturn': daily_returns,
            'DailyInflation': daily_inflation_changes
        })
        df_returns['YearMonth'] = df_returns['Date'].dt.to_period('M')
        logger.info(f"[DEBUG] Created YearMonth periods. Unique months: {df_returns['YearMonth'].nunique()}")
        



        logger.info(f"[DEBUG] Aggregating daily data to monthly returns...")
        monthly_returns_data = df_returns.groupby('YearMonth').agg({
            'DailyReturn': lambda x: np.prod(1 + x) - 1.0,
            'DailyInflation': lambda x: np.prod(1 + x) - 1.0,
            'Date': 'first'
        }).reset_index()
        logger.info(f"[DEBUG] Aggregated to {len(monthly_returns_data)} monthly periods")
        
        monthly_returns = monthly_returns_data['DailyReturn'].values
        monthly_inflation = monthly_returns_data['DailyInflation'].values
        monthly_dates = monthly_returns_data['Date'].values
        
        logger.info(f"[DEBUG] Monthly returns: len={len(monthly_returns)}, "
                   f"mean={np.mean(monthly_returns):.6f}, std={np.std(monthly_returns):.6f}")
        logger.info(f"[DEBUG] Monthly inflation: len={len(monthly_inflation)}, "
                   f"mean={np.mean(monthly_inflation):.6f}, std={np.std(monthly_inflation):.6f}")
        logger.info(f"[DEBUG] Monthly dates range: {monthly_dates[0]} to {monthly_dates[-1]}")
        

        if len(monthly_returns) != len(monthly_inflation):
            logger.error(f"[DEBUG] ALIGNMENT ERROR: monthly_returns length={len(monthly_returns)}, "
                        f"monthly_inflation length={len(monthly_inflation)}")
            raise ValueError(f"Alignment error: monthly_returns and monthly_inflation must have same length. "
                           f"Got {len(monthly_returns)} and {len(monthly_inflation)}")
        
        logger.info(f"[DEBUG] Successfully loaded and converted to monthly returns. "
                   f"Returning {len(monthly_returns)} monthly data points")
        return monthly_returns, monthly_inflation, monthly_dates
        
    except Exception as e:
        logger.error(f"Error loading CSV file '{csv_path}': {e}")
        raise


class BlockBootstrap:
    """
    Block bootstrap sampler for monthly returns.
    Supports both overlapping and non-overlapping blocks.
    Optional: shift returns in log space so expected geometric mean matches a target (annual rate).
    """
    def __init__(self, monthly_returns, monthly_inflation, block_length_months,
                 overlapping=True, rng=None, target_geometric_mean_annual=None):
        """
        Initialize block bootstrap.

        Args:
            monthly_returns: Array of monthly returns
            monthly_inflation: Array of monthly inflation values
            block_length_months: Length of each block in months (e.g., 120 for 10 years)
            overlapping: If True, use overlapping blocks; if False, use non-overlapping
            rng: Random number generator (default: new generator)
            target_geometric_mean_annual: If set (e.g. 0.06 for 6%), each bootstrapped
                return is shifted in log space so the expected geometric mean (annual) equals
                this value. Historical volatility and correlation structure are preserved.
        """
        self.monthly_returns = np.array(monthly_returns)
        self.monthly_inflation = np.array(monthly_inflation)


        if len(self.monthly_returns) != len(self.monthly_inflation):
            raise ValueError(f"monthly_returns and monthly_inflation must have the same length. "
                             f"Got {len(self.monthly_returns)} and {len(self.monthly_inflation)}")
        self.block_length_months = int(block_length_months)
        self.overlapping = overlapping
        self.rng = rng if rng is not None else np.random.default_rng()
        self.target_geometric_mean_annual = target_geometric_mean_annual


        self._hist_mean_log_return_monthly = np.mean(np.log(1.0 + self.monthly_returns))


        self._create_blocks(verbose=False)
        
    def _create_blocks(self, verbose=False):
        """Create blocks from monthly data."""
        n_months = len(self.monthly_returns)
        
        if self.overlapping:
            self.num_blocks = n_months - self.block_length_months + 1
            self.block_starts = np.arange(self.num_blocks)
        else:
            self.num_blocks = n_months // self.block_length_months
            self.block_starts = np.arange(self.num_blocks) * self.block_length_months
        
        if self.num_blocks == 0:
            raise ValueError(f"Not enough data for block length {self.block_length_months}. "
                           f"Need at least {self.block_length_months} months, got {n_months}")
        
        if verbose:
            step_size = 1 if self.overlapping else self.block_length_months
            logger.info(f"Created {self.num_blocks} {'overlapping' if self.overlapping else 'non-overlapping'} "
                       f"blocks of length {self.block_length_months} months "
                       f"(step size: {step_size} month{'s' if step_size > 1 else ''} between block starts)")
    
    def sample_block(self):
        """
        Sample a random block from the data.
        
        Returns:
            tuple: (returns_block, inflation_block) - arrays of length block_length_months
        """
        block_idx = self.rng.integers(0, self.num_blocks)
        start_idx = self.block_starts[block_idx]
        end_idx = start_idx + self.block_length_months
        
        returns_block = self.monthly_returns[start_idx:end_idx].copy()
        inflation_block = self.monthly_inflation[start_idx:end_idx].copy()
        
        return returns_block, inflation_block
    
    def sample_sequence(self, num_months, verbose=False):
        """
        Sample a sequence of returns of specified length using block bootstrap.
        
        Args:
            num_months: Number of months to sample
            verbose: If True, log when blocks switch
        
        Returns:
            tuple: (returns_sequence, inflation_sequence) - arrays of length num_months
        """
        returns_sequence = []
        inflation_sequence = []
        
        remaining_months = num_months
        block_number = 0
        while remaining_months > 0:
            returns_block, inflation_block = self.sample_block()
            block_number += 1
            
            if verbose:
                logger.info(f"Block {block_number}: Sampling {len(returns_block)} months "
                          f"({remaining_months} months remaining)")
            
            take_months = min(remaining_months, len(returns_block))
            returns_sequence.extend(returns_block[:take_months])
            inflation_sequence.extend(inflation_block[:take_months])
            
            remaining_months -= take_months
        
        if verbose:
            logger.info(f"Sampled {block_number} blocks for {num_months} total months")

        returns_out = np.array(returns_sequence[:num_months])

        if self.target_geometric_mean_annual is not None:

            target_log_monthly = np.log(1.0 + self.target_geometric_mean_annual) / 12.0
            delta = target_log_monthly - self._hist_mean_log_return_monthly

            returns_out = (1.0 + returns_out) * np.exp(delta) - 1.0

        return returns_out, np.array(inflation_sequence[:num_months])



_bootstrap_data_cache = None

def load_bootstrap_data(config):
    """
    Load bootstrap data once and cache it globally.
    
    Args:
        config: SimulationConfig instance
    
    Returns:
        tuple: (monthly_returns, monthly_inflation) or None if bootstrap disabled
    """
    global _bootstrap_data_cache
    
    if not config.use_block_bootstrap:
        return None
    

    if _bootstrap_data_cache is not None:
        return _bootstrap_data_cache
    
    try:

        try:
            import json
            import time
            import os
            log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'log_bootstrap_load_start', 'timestamp': time.time() * 1000, 'location': 'bootstrap.py:254', 'message': 'Starting load_and_convert_to_monthly_returns', 'data': {'csv_path': config.bootstrap_csv_path, 'portfolio_col': config.portfolio_column_name, 'inflation_col': config.inflation_column_name}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'B'}) + '\n')
        except Exception as log_err: pass

        monthly_returns, monthly_inflation, _ = load_and_convert_to_monthly_returns(
            config.bootstrap_csv_path,
            config.portfolio_column_name,
            config.inflation_column_name
        )
        
        _bootstrap_data_cache = (monthly_returns, monthly_inflation)

        try:
            import json
            import time
            import os
            log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'log_bootstrap_load_success', 'timestamp': time.time() * 1000, 'location': 'bootstrap.py:261', 'message': 'Bootstrap data loaded successfully', 'data': {'returns_len': len(monthly_returns), 'inflation_len': len(monthly_inflation)}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'B'}) + '\n')
        except Exception as log_err: pass

        return _bootstrap_data_cache
        
    except Exception as e:
        logger.error(f"Failed to load bootstrap data: {e}")

        try:
            import json
            import time
            import os
            import traceback
            log_dir = r'd:\Finance\Scripting\Lifecycle-Retirement-Simulation-main\.cursor'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id': 'log_bootstrap_load_error', 'timestamp': time.time() * 1000, 'location': 'bootstrap.py:264', 'message': 'Exception in load_bootstrap_data', 'data': {'error': str(e), 'error_type': type(e).__name__, 'traceback': traceback.format_exc()}, 'sessionId': 'debug-session', 'runId': 'initial', 'hypothesisId': 'B'}) + '\n')
        except Exception as log_err: pass

        raise


def create_block_bootstrap_sampler(config, rng, monthly_returns=None, monthly_inflation=None):
    """
    Create a block bootstrap sampler from configuration.
    
    Args:
        config: SimulationConfig instance
        rng: Random number generator
        monthly_returns: Pre-loaded monthly returns (optional, will load if None)
        monthly_inflation: Pre-loaded monthly inflation (optional, will load if None)
    
    Returns:
        BlockBootstrap instance or None if block bootstrap is disabled
    """
    if not config.use_block_bootstrap:
        return None
    
    try:

        if monthly_returns is None or monthly_inflation is None:
            data = load_bootstrap_data(config)
            if data is None:
                return None
            monthly_returns, monthly_inflation = data
        
        block_length_months = int(config.block_length_years * 12)

        real_override = getattr(config, 'bootstrap_geometric_mean_override', None)
        if real_override is not None:
            monthly_inflation_arr = np.array(monthly_inflation)
            monthly_inflation_arr = np.clip(monthly_inflation_arr, -0.99, np.inf)
            geom_mean_monthly_inflation = np.exp(np.mean(np.log(1.0 + monthly_inflation_arr))) - 1.0
            annual_inflation = (1.0 + geom_mean_monthly_inflation) ** 12 - 1.0
            nominal_target = (1.0 + real_override) * (1.0 + annual_inflation) - 1.0
        else:
            nominal_target = None

        bootstrap_sampler = BlockBootstrap(
            monthly_returns,
            monthly_inflation,
            block_length_months,
            overlapping=config.block_overlapping,
            rng=rng,
            target_geometric_mean_annual=nominal_target
        )
        
        return bootstrap_sampler
        
    except Exception as e:
        logger.error(f"Failed to create block bootstrap sampler: {e}")
        raise

