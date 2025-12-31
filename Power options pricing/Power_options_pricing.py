import numpy as np
from scipy.stats import norm
import time

# Power Option Pricing: Two Methods Comparison

def power_option_analytical(S_0, K, T, r, sigma):
    """
    Analytical pricing using change of numeraire to S-measure.
    
    """
    # Adjusted drift under S-measure
    r_adjusted = r + 0.5 * sigma**2
    
    # Calculate d1 and d2
    d1 = (np.log(S_0 / K) + (r_adjusted + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Black-Scholes call under S-measure
    # E^{Q^S}[(S(T) - K)^+] = S(0)*exp(r_adjusted*T)*N(d1) - K*N(d2)
    bs_call = S_0 * np.exp((r + sigma**2) * T) * norm.cdf(d1) - K * norm.cdf(d2)
    
    # Multiply by S(0) to get V(0)
    V_0 = S_0 * bs_call
    
    return V_0

def power_option_monte_carlo(S_0, K, T, r, sigma, n_sims, seed=42):
    """
    Monte Carlo pricing under risk-neutral measure Q.

    """
    np.random.seed(seed)
    
    # Generate standard normal random variables
    Z = np.random.standard_normal(n_sims)
    
    # Simulate S(T) under risk-neutral measure Q
    S_T = S_0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Payoff: (S(T)^2 - K*S(T))^+ = S(T) * max(S(T) - K, 0)
    payoff = S_T * np.maximum(S_T - K, 0)
    
    # Discount and average
    V_0 = np.exp(-r * T) * np.mean(payoff)
    
    # Standard error
    std_error = np.exp(-r * T) * np.std(payoff) / np.sqrt(n_sims)
    
    return V_0, std_error

# Parameters
S_0 = 100.0        # Initial stock price
K = 110.0          # Strike price
T = 1.0            # Time to maturity (1 year)
r = 0.05           # Risk-free rate
sigma = 0.30       # Volatility

# Method 1: Analytical (Black-Scholes with S-measure)
print("\n" + "=" * 70)
print("METHOD 1: Analytical (Change to S-measure)")
print("=" * 70)

start = time.time()
price_analytical = power_option_analytical(S_0, K, T, r, sigma)
time_analytical = time.time() - start

print(f"\nPrice: ${price_analytical:.6f}")
print(f"Time: {time_analytical*1000:.4f} ms")

# Method 2: Monte Carlo with different number of simulations
print("\n" + "=" * 70)
print("METHOD 2: Monte Carlo Simulation (under Q measure)")
print("=" * 70)

simulation_sizes = [10000, 50000, 100000]

for n_sims in simulation_sizes:
    start = time.time()
    price_mc, std_error = power_option_monte_carlo(S_0, K, T, r, sigma, n_sims)
    time_mc = time.time() - start
    
    error = price_mc - price_analytical
    error_pct = (error / price_analytical) * 100
    
    print(f"Simulations: {n_sims:,}")
    print(f"  Price: ${price_mc:.6f}")
    print(f"  Std Error: ${std_error:.6f}")
    print(f"  Error vs Analytical: ${error:.6f} ({error_pct:+.3f}%)")
    print(f"  Time: {time_mc:.4f} seconds")

