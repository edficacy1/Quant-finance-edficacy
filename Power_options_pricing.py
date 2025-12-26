import numpy as np
from scipy.stats import norm
import time

# Power Option Pricing: Two Methods Comparison

def power_option_analytical(S_0, K, T, r, sigma):
    """
    Analytical pricing using change of numeraire to S-measure.
    
    Original (under Q with bank account numeraire):
    V(0) = E^Q[e^(-r*T) * (S(T)^2 - K*S(T))^+]
         = E^Q[e^(-r*T) * S(T) * (S(T) - K)^+]
    
    Change to S-measure (where S(t) is the numeraire):
    V(0) = S(0) * E^{Q^S}[(S(T) - K)^+]
    
    Under Q^S measure, by Girsanov theorem:
    dW^{Q^S} = dW^Q - sigma * dt
    
    So S(t) under Q^S follows:
    dS = r*S*dt + sigma*S*dW^Q
       = r*S*dt + sigma*S*(dW^{Q^S} + sigma*dt)
       = (r + sigma^2)*S*dt + sigma*S*dW^{Q^S}
    
    Therefore, under Q^S:
    S(T) = S(0) * exp((r + sigma^2 - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
         = S(0) * exp((r + 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    
    This gives a Black-Scholes call with adjusted drift:
    V(0) = S(0) * BS_call(S(0), K, T, r + 0.5*sigma^2, sigma)
         = S(0) * [S(0)*exp((r + 0.5*sigma^2)*T)*N(d1) - K*N(d2)]
    
    where:
    d1 = [ln(S(0)/K) + (r + 0.5*sigma^2 + 0.5*sigma^2)*T] / (sigma*sqrt(T))
       = [ln(S(0)/K) + (r + sigma^2)*T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
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
    
    V(0) = E^Q[e^(-r*T) * (S(T)^2 - K*S(T))^+]
         = E^Q[e^(-r*T) * S(T) * max(S(T) - K, 0)]
    
    Under Q:
    S(T) = S(0) * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    
    where Z ~ N(0,1)
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

def verify_dynamics():
    """
    Verify the stock dynamics under both measures through simulation.
    This is for educational purposes to show the drift change.
    """
    S_0 = 100.0
    r = 0.05
    sigma = 0.30
    T = 1.0
    n_sims = 1000000
    
    np.random.seed(42)
    Z = np.random.standard_normal(n_sims)
    
    # Under Q (risk-neutral measure)
    S_T_Q = S_0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Under Q^S (stock measure)
    S_T_QS = S_0 * np.exp((r + 0.5 * sigma**2) - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z

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
print("Formula: V(0) = S(0) · E^{Q^S}[(S(T) - K)⁺]")
print(f"\nUnder Q^S measure, S(t) has drift: r + 0.5σ² = {r + 0.5*sigma**2:.6f}")
print("This is equivalent to: V(0) = S(0) · BS_call(S(0), K, T, r+0.5σ², σ)")

start = time.time()
price_analytical = power_option_analytical(S_0, K, T, r, sigma)
time_analytical = time.time() - start

print(f"\nPrice: ${price_analytical:.6f}")
print(f"Time: {time_analytical*1000:.4f} ms")

# Method 2: Monte Carlo with different number of simulations
print("\n" + "=" * 70)
print("METHOD 2: Monte Carlo Simulation (under Q measure)")
print("=" * 70)
print("Formula: V(0) = E^Q[e^(-r·T) · S(T) · max(S(T) - K, 0)]")
print(f"Under Q measure, S(t) has drift: r - 0.5σ² = {r - 0.5*sigma**2:.6f}\n")

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
    print(f"  95% CI: [${price_mc - 1.96*std_error:.6f}, ${price_mc + 1.96*std_error:.6f}]\n")

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Analytical Price (S-measure): ${price_analytical:.6f}")
print(f"\nKey Insight:")
print(f"  • Change of numeraire from M(t)=e^(rt) to S(t)")
print(f"  • Under S-measure: drift changes from r to (r + σ²)")
print(f"  • Payoff S(T)·(S(T)-K)⁺ becomes S(0)·E^{{Q^S}}[(S(T)-K)⁺]")
print(f"  • Result: S(0) times a Black-Scholes call with adjusted drift")
print("=" * 70)

