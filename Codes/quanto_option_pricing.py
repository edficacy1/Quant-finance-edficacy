import numpy as np
from scipy.stats import norm
import time

# Quanto Option Pricing: Two Methods Comparison

def quanto_option_analytical(S_d_0, S_f_0, K, T, r_d, r_f, sigma_d, sigma_f, 
                             sigma_X, rho, rho_X):
    """
    Analytical pricing using change of numeraire to S_d measure.
    
    Starting from: V(0) = E^Q[e^{-r_d*T} * S_d(T) * max(S_f(T) - K, 0)]
    
    Change to S_d-numeraire measure:
    V(0) = S_d(0) * E^{S_d}[max(S_f(T) - K, 0)]
    
    Under Q^{S_d}, the Radon-Nikodym derivative is:
    dQ^{S_d}/dQ = S_d(T)/(S_d(0)*e^{r_d*T})
    
    By Girsanov theorem:
    dW_f^{S_d} = dW_f^Q - sigma_d * rho * dt
    dW_d^{S_d} = dW_d^Q - sigma_d * dt
    
    So S_f under Q^{S_d} has drift:
    (r_f - rho_X * sigma_X * sigma_f) + sigma_f * sigma_d * rho
    
    This gives a Black-Scholes formula with adjusted drift.
    """
    # Adjusted drift under S_d measure
    # Original drift under Q: r_f - rho_X * sigma_X * sigma_f
    # Adjustment from numeraire change: + rho * sigma_d * sigma_f
    r_adjusted = r_f - rho_X * sigma_X * sigma_f + rho * sigma_d * sigma_f
    
    # Black-Scholes call with adjusted parameters
    d1 = (np.log(S_f_0 / K) + (r_adjusted + 0.5 * sigma_f**2) * T) / (sigma_f * np.sqrt(T))
    d2 = d1 - sigma_f * np.sqrt(T)
    
    bs_call = S_f_0 * np.exp(r_adjusted * T) * norm.cdf(d1) - K * norm.cdf(d2)
    
    # Multiply by domestic asset price
    V_0 = S_d_0 * bs_call
    
    return V_0

def quanto_option_monte_carlo(S_d_0, S_f_0, K, T, r_d, r_f, sigma_d, sigma_f, 
                              sigma_X, rho, rho_X, n_sims, n_steps=252, seed=42):
    """
    Monte Carlo pricing under domestic risk-neutral measure Q.
    
    V(0) = E^Q[e^{-r_d * T} * S_d(T) * max(S_f(T) - K, 0)]
    
    Under Q (using exact simulation for log-normal processes):
    S_d(T) = S_d(0) * exp((r_d - 0.5*sigma_d^2)*T + sigma_d*sqrt(T)*Z_d)
    S_f(T) = S_f(0) * exp((r_f - rho_X*sigma_X*sigma_f - 0.5*sigma_f^2)*T + sigma_f*sqrt(T)*Z_f)
    
    where Z_d and Z_f are correlated standard normals with correlation rho
    """
    np.random.seed(seed)
    
    # Generate correlated random variables
    Z1 = np.random.standard_normal(n_sims)
    Z2 = np.random.standard_normal(n_sims)
    
    Z_d = Z1
    Z_f = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    # Simulate terminal values using exact solution
    S_d_T = S_d_0 * np.exp((r_d - 0.5 * sigma_d**2) * T + sigma_d * np.sqrt(T) * Z_d)
    S_f_T = S_f_0 * np.exp((r_f - rho_X * sigma_X * sigma_f - 0.5 * sigma_f**2) * T 
                           + sigma_f * np.sqrt(T) * Z_f)
    
    # Payoff: S_d(T) * max(S_f(T) - K, 0)
    payoff = S_d_T * np.maximum(S_f_T - K, 0)
    
    # Discount and average
    V_0 = np.exp(-r_d * T) * np.mean(payoff)
    
    # Standard error
    std_error = np.exp(-r_d * T) * np.std(payoff) / np.sqrt(n_sims)
    
    return V_0, std_error

# Parameters
S_d_0 = 100.0      # Initial domestic asset price
S_f_0 = 50.0       # Initial foreign asset price
K = 55.0           # Strike price
T = 1.0            # Time to maturity (1 year)
r_d = 0.05         # Domestic risk-free rate
r_f = 0.03         # Foreign risk-free rate
sigma_d = 0.20     # Domestic asset volatility
sigma_f = 0.30     # Foreign asset volatility
sigma_X = 0.15     # Exchange rate volatility
rho = 0.4          # Correlation between S_d and S_f
rho_X = -0.3       # Correlation between exchange rate and S_f

# Method 1: Analytical (Black-Scholes with change of numeraire)
print("\n" + "=" * 70)
print("METHOD 1: Analytical (Black-Scholes with S_d numeraire)")
print("=" * 70)
start = time.time()
price_analytical = quanto_option_analytical(S_d_0, S_f_0, K, T, r_d, r_f, 
                                            sigma_d, sigma_f, sigma_X, rho, rho_X)
time_analytical = time.time() - start
print(f"Price: ${price_analytical:.6f}")
print(f"Time: {time_analytical*1000:.4f} ms")

# Method 2: Monte Carlo with different number of simulations
print("\n" + "=" * 70)
print("METHOD 2: Monte Carlo Simulation (under Q measure)")
print("=" * 70)

simulation_sizes = [10000, 50000, 100000, 500000, 1000000]

for n_sims in simulation_sizes:
    start = time.time()
    price_mc, std_error = quanto_option_monte_carlo(S_d_0, S_f_0, K, T, r_d, r_f, 
                                                     sigma_d, sigma_f, sigma_X, 
                                                     rho, rho_X, n_sims)
    time_mc = time.time() - start
    
    error = price_mc - price_analytical
    error_pct = (error / price_analytical) * 100
    
    print(f"\nSimulations: {n_sims:,}")
    print(f"  Price: ${price_mc:.6f}")
    print(f"  Std Error: ${std_error:.6f}")
    print(f"  Error vs Analytical: ${error:.6f} ({error_pct:+.3f}%)")
    print(f"  Time: {time_mc:.4f} seconds")

