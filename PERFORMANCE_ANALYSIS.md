# Performance Bottleneck Analysis

## Problem Summary
The script is extremely slow at "Stage 1: Building Required Principal Lookup Table" because it performs a massive number of simulations.

## The Math

### Execution Flow:
1. **Loop through 41 retirement ages** (30 to 70)
2. **For each age, binary search** to find required principal:
   - Up to **30 iterations** per age
   - Each iteration runs **500 simulations** (`num_nested=500`)
   - Each simulation runs **monthly from retirement_age to death_age (100)**
   
### Total Computation:
- **41 ages** × **30 binary search iterations** × **500 simulations** = **615,000 total simulations**
- Average simulation length: ~600 months (50 years)
- **Total months simulated: ~369,000,000 months**

### Time per Simulation:
- For retirement age 30: 70 years × 12 = **840 months** per simulation
- For retirement age 70: 30 years × 12 = **360 months** per simulation
- Each month involves: market return calculation, portfolio update, inflation adjustment

## Identified Bottlenecks

### 1. **Excessive `num_nested` for Binary Search** (CRITICAL)
- **Location**: `config.py` line 25: `self.num_nested = 500`
- **Problem**: This is used in `find_required_principal()` which does binary search
- **Impact**: Each binary search iteration runs 500 full retirement simulations
- **Solution**: Use a lower value (e.g., 100-200) for the binary search phase, then use 500 for final validation

### 2. **Binary Search Iterations** (HIGH)
- **Location**: `simulation.py` line 910: `max_iterations = 30`
- **Problem**: Up to 30 iterations per age, even with caching
- **Impact**: 30 × 500 = 15,000 simulations per age
- **Solution**: Reduce to 20 iterations or improve convergence

### 3. **Multiprocessing May Not Be Working** (MEDIUM)
- **Location**: `simulation.py` line 457: `if config.num_workers <= 1 or num_nested_sims < 100:`
- **Problem**: Complex Windows multiprocessing setup with lots of overhead
- **Impact**: If multiprocessing fails, runs single-threaded (500 sims sequentially)
- **Solution**: Verify multiprocessing is actually being used, simplify setup

### 4. **No Early Termination in Binary Search** (MEDIUM)
- **Location**: `simulation.py` line 913: Binary search continues even when close
- **Problem**: Runs full 30 iterations even when tolerance is met early
- **Impact**: Wastes computation on unnecessary iterations
- **Solution**: Check tolerance more frequently or use adaptive iteration count

### 5. **Inefficient Bootstrap Sampling** (LOW)
- **Location**: `simulation.py` line 267: `bootstrap_sampler.sample_sequence(total_months)`
- **Problem**: Pre-samples entire sequence upfront
- **Impact**: Memory allocation and potential inefficiency
- **Solution**: Already optimized, but could be improved

## Recommended Fixes (Priority Order)

### Fix 1: Reduce `num_nested` for Binary Search (IMMEDIATE - 5-10x speedup)
```python
# In find_required_principal, use fewer simulations for binary search
# Then validate with full num_nested at the end
binary_search_sims = min(100, num_nested_sims)  # Use 100 for search
```

### Fix 2: Reduce Binary Search Iterations (IMMEDIATE - 1.5x speedup)
```python
max_iterations = 20  # Instead of 30
```

### Fix 3: Verify Multiprocessing (HIGH - 4-8x speedup if working)
- Check if `config.num_workers > 1` and `num_nested_sims >= 100`
- Verify workers are actually being spawned
- Consider using `concurrent.futures` instead of complex multiprocessing setup

### Fix 4: Add Progress Logging (MEDIUM)
- Log time per age calculation
- Show estimated remaining time
- Help identify if specific ages are slower

### Fix 5: Optimize Tolerance (LOW)
```python
tolerance = 5000.0  # Increase from 1000.0 to reduce iterations
```

## Expected Performance Improvement

With Fixes 1-3:
- **Before**: 615,000 simulations × ~600 months = ~369M month-simulations
- **After**: 41 ages × 20 iterations × 100 sims = 82,000 simulations
- **Speedup**: ~7.5x reduction in simulations
- **With multiprocessing**: Additional 4-8x speedup = **30-60x total speedup**

## Quick Test
To verify the bottleneck, temporarily set:
```python
self.num_nested = 50  # In config.py
self.retirement_age_max = 35  # Test with fewer ages
```
This should complete in seconds instead of hours.

