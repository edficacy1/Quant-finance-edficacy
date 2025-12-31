import numpy as np
from scipy.stats import norm
import time

# Exchange Option Pricing

def black_scholes_exchange(S1_0, S2_0, T, r, sigma_1, sigma_2, rho):
    """
    Price exchange option using Black-Scholes formula under S_2 measure.
    
    """
    # Ratio Y = S_1/S_2
    Y_0 = S1_0 / S2_0
    K = 1.0  # Strike for the ratio
    
    # Volatility of the ratio (combined volatility)
    sigma_Y = np.sqrt(sigma_1**2 + sigma_2**2 - 2 * rho * sigma_1 * sigma_2)
       
    d1 = (np.log(Y_0 / K) + 0.5 * sigma_Y**2 * T) / (sigma_Y * np.sqrt(T))
    d2 = d1 - sigma_Y * np.sqrt(T)
    
    call_value_per_unit = Y_0 * norm.cdf(d1) - K * norm.cdf(d2)
    
    exchange_value = S2_0 * call_value_per_unit
    
    return exchange_value


def monte_carlo_exchange(S1_0, S2_0, T, r, sigma_1, sigma_2, rho, n_sims, n_steps=252):
    """
    Price exchange option using Monte Carlo under risk-neutral measure Q.
    
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Initialize stock price arrays
    S1 = S1_0 * np.ones(n_sims)
    S2 = S2_0 * np.ones(n_sims)
    
    # Simulate paths under Q measure
    for _ in range(n_steps):
        # Generate correlated random variables
        Z1 = np.random.standard_normal(n_sims)
        Z2 = np.random.standard_normal(n_sims)
        
        # Create correlated Brownian increments
        dW1 = Z1 * sqrt_dt
        dW2 = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * sqrt_dt
        
        # Update S_1 under Q
        S1 = S1 * np.exp((r - 0.5 * sigma_1**2) * dt + sigma_1 * dW1)
        
        # Update S_2 under Q
        S2 = S2 * np.exp((r - 0.5 * sigma_2**2) * dt + sigma_2 * dW2)
    
    # Payoff: max(S_1(T) - S_2(T), 0)
    payoffs = np.maximum(S1 - S2, 0)
    
    # Discount to present value
    option_value = np.exp(-r * T) * np.mean(payoffs)
    
    return option_value


# Parameters
S1_0 = 100.0       # Initial price of asset 1
S2_0 = 95.0        # Initial price of asset 2
T = 1.0            # Time to maturity (1 year)
r = 0.05           # Risk-free rate
sigma_1 = 0.30     # Volatility of asset 1
sigma_2 = 0.25     # Volatility of asset 2
rho = 0.5          # Correlation between assets

# Method 1: Black-Scholes with S_2 numeraire
print("\n1. BLACK-SCHOLES FORMULA (S_2 Numeraire Method)")
print("-" * 70)
start = time.time()
bs_price = black_scholes_exchange(S1_0, S2_0, T, r, sigma_1, sigma_2, rho)
bs_time = time.time() - start
print(f"   Price: {bs_price:.6f}")
print(f"   Computation time: {bs_time:.6f} seconds")

# Method 2: Monte Carlo under Q measure
print("\n2. MONTE CARLO SIMULATION (Q Measure)")
print("-" * 70)

simulation_sizes = [10000, 50000, 100000, 500000]

for n_sims in simulation_sizes:
    np.random.seed(42)  # For reproducibility
    start = time.time()
    mc_price = monte_carlo_exchange(S1_0, S2_0, T, r, sigma_1, sigma_2, rho, n_sims)
    mc_time = time.time() - start
    
    error = abs(mc_price - bs_price)
    error_pct = (error / bs_price) * 100 if bs_price > 0 else 0
    
    print(f"\n   Simulations: {n_sims:,}")
    print(f"   Price: {mc_price:.6f}")
    print(f"   Computation time: {mc_time:.6f} seconds")
    print(f"   Absolute error vs BS: {error:.6f}")
    print(f"   Relative error: {error_pct:.4f}%")
