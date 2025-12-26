import numpy as np
from scipy.stats import norm
import time
import scipy.integrate as integrate

# ============================================================================
# HULL-WHITE MODEL FUNCTIONS
# ============================================================================

def HW_theta(lambd, eta, P0T):
    """Calculate theta function for Hull-White model"""
    dt = 0.0001    
    f0T = lambda t: -(np.log(P0T(t+dt)) - np.log(P0T(t-dt))) / (2*dt)
    theta = lambda t: (1.0/lambd) * (f0T(t+dt) - f0T(t-dt)) / (2.0*dt) + f0T(t) + \
                      (eta*eta / (2.0*lambd*lambd)) * (1.0 - np.exp(-2.0*lambd*t))
    return theta

def GeneratePathsHWEuler(NoOfPaths, NoOfSteps, T, P0T, lambd, eta):    
    """Generate interest rate paths using Hull-White model with Euler discretization"""
    dt_diff = 0.0001    
    f0T = lambda t: -(np.log(P0T(t+dt_diff)) - np.log(P0T(t-dt_diff))) / (2*dt_diff)
    
    # Initial interest rate is forward rate at t->0
    r0 = f0T(0.00001)
    theta = HW_theta(lambd, eta, P0T)
    
    # Generate random numbers
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    R = np.zeros([NoOfPaths, NoOfSteps+1])
    R[:, 0] = r0
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0, NoOfSteps):
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i+1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
        R[:, i+1] = R[:, i] + lambd * (theta(time[i]) - R[:, i]) * dt + eta * (W[:, i+1] - W[:, i])
        time[i+1] = time[i] + dt
        
    paths = {"time": time, "R": R}
    return paths

def HW_B(lambd, eta, T1, T2):
    """Calculate B(T1, T2) factor for Hull-White ZCB formula"""
    return (1.0/lambd) * (np.exp(-lambd*(T2-T1)) - 1.0)

def HW_A(lambd, eta, P0T, T1, T2):
    """Calculate A(T1, T2) factor for Hull-White ZCB formula"""
    tau = T2 - T1
    zGrid = np.linspace(0.0, tau, 250)
    B_r = lambda tau: (1.0/lambd) * (np.exp(-lambd*tau) - 1.0)
    theta = HW_theta(lambd, eta, P0T)    
    temp1 = lambd * integrate.trapezoid(theta(T2-zGrid) * B_r(zGrid), zGrid)
    
    temp2 = (eta*eta / (4.0*np.power(lambd, 3.0))) * \
            (np.exp(-2.0*lambd*tau) * (4*np.exp(lambd*tau) - 1.0) - 3.0) + \
            eta*eta*tau / (2.0*lambd*lambd)
    
    return temp1 + temp2

def HW_ZCB(lambd, eta, P0T, T1, T2, rT1):
    """Calculate zero-coupon bond price P(T1, T2) given r(T1)"""
    B_r = HW_B(lambd, eta, T1, T2)
    A_r = HW_A(lambd, eta, P0T, T1, T2)
    return np.exp(A_r + B_r * rT1)

# ============================================================================
# HULL-WHITE ANALYTICAL ZCB OPTION PRICING
# ============================================================================

def HW_Mu_FrwdMeasure(P0T, lambd, eta, T):
    """Calculate mean of r(T) under T-forward measure"""
    dt = 0.0001    
    f0T = lambda t: -(np.log(P0T(t+dt)) - np.log(P0T(t-dt))) / (2*dt)
    r0 = f0T(0.00001)
    theta = HW_theta(lambd, eta, P0T)
    zGrid = np.linspace(0.0, T, 500)
    
    theta_hat = lambda t, T: theta(t) + eta*eta / lambd * 1.0/lambd * (np.exp(-lambd*(T-t)) - 1.0)
    
    temp = lambda z: theta_hat(z, T) * np.exp(-lambd*(T-z))
    
    r_mean = r0 * np.exp(-lambd*T) + lambd * integrate.trapezoid(temp(zGrid), zGrid)
    
    return r_mean

def HWVar_r(lambd, eta, T):
    """Calculate variance of r(T)"""
    return eta*eta / (2.0*lambd) * (1.0 - np.exp(-2.0*lambd*T))

def HW_ZCB_CallPutOption(CP, K, lambd, eta, P0T, T_option, T_bond):
    """
    Analytical Hull-White price for a call/put option on a zero-coupon bond.
    
    Option payoff at T_option: max(P(T_option, T_bond) - K, 0) for call
                                max(K - P(T_option, T_bond), 0) for put
    
    where P(T_option, T_bond) is the ZCB price at option expiry.
    
    Parameters:
    -----------
    CP : str
        'CALL' or 'PUT'
    K : float
        Strike price for the ZCB
    lambd : float
        Mean reversion speed
    eta : float
        Volatility of short rate
    P0T : function
        Zero-coupon bond curve P(0,T)
    T_option : float
        Option expiry time
    T_bond : float
        Bond maturity time (T_bond > T_option)
    
    Returns:
    --------
    float : Option price at t=0
    """
    # Calculate B and A factors
    B_r = HW_B(lambd, eta, T_option, T_bond)
    A_r = HW_A(lambd, eta, P0T, T_option, T_bond)
    
    # Mean and variance of r(T_option) under T_option-forward measure
    mu_r = HW_Mu_FrwdMeasure(P0T, lambd, eta, T_option)
    v_r = np.sqrt(HWVar_r(lambd, eta, T_option))
    
    # Transform strike
    K_hat = K * np.exp(-A_r)
    
    # Calculate d1 and d2
    a = (np.log(K_hat) - B_r*mu_r) / (B_r*v_r)
    
    d1 = a - B_r*v_r
    d2 = d1 + B_r*v_r
    
    # Option value under T_option-forward measure
    term1 = np.exp(0.5*B_r*B_r*v_r*v_r + B_r*mu_r) * norm.cdf(d1) - K_hat * norm.cdf(d2)    
    value = P0T(T_option) * np.exp(A_r) * term1 
    
    if CP == 'CALL':
        return value
    elif CP == 'PUT':
        return value - P0T(T_bond) + K*P0T(T_option)

# ============================================================================
# MONTE CARLO ZCB OPTION PRICING WITH HULL-WHITE
# ============================================================================

def monte_carlo_zcb_option_hull_white(CP, K, T_option, T_bond, P0T, lambd, eta, n_sims, n_steps=500):
    """
    Price a call/put option on a zero-coupon bond using Monte Carlo with Hull-White model.
    
    Option payoff at T_option:
    - Call: max(P(T_option, T_bond) - K, 0)
    - Put:  max(K - P(T_option, T_bond), 0)
    
    Process:
    1. Simulate short rate r(t) paths using Hull-White up to T_option
    2. Calculate money market account: B(T_option) = exp(∫₀^T_option r(s)ds)
    3. At T_option, calculate ZCB price P(T_option, T_bond) using r(T_option)
    4. Calculate payoff based on option type
    5. Discount: V(0) = E^Q[payoff / B(T_option)]
    
    Parameters:
    -----------
    CP : str
        'CALL' or 'PUT'
    K : float
        Strike price
    T_option : float
        Option expiry time
    T_bond : float
        Bond maturity time (T_bond > T_option)
    P0T : function
        Zero-coupon bond curve
    lambd : float
        Mean reversion speed
    eta : float
        Volatility of short rate
    n_sims : int
        Number of Monte Carlo simulations
    n_steps : int
        Number of time steps
    verbose : bool
        Print diagnostics
    
    Returns:
    --------
    float : Option price at t=0
    """
    
    # Generate Hull-White paths up to T_option
    paths = GeneratePathsHWEuler(n_sims, n_steps, T_option, P0T, lambd, eta)
    r_paths = paths["R"]
    time_grid = paths["time"]
    
    # Calculate money market account B(T_option) = exp(∫₀^T_option r(s)ds)
    integral_r = np.zeros(n_sims)
    for i in range(n_sims):
        integral_r[i] = np.trapezoid(r_paths[i, :], time_grid)
    
    B_T_option = np.exp(integral_r)
    
    # Extract r(T_option) for each path
    r_T_option = r_paths[:, -1]
    
    # Calculate ZCB prices P(T_option, T_bond) for each path
    P_T_option_T_bond = HW_ZCB(lambd, eta, P0T, T_option, T_bond, r_T_option)
    
    # Calculate payoffs based on option type
    if CP == 'CALL':
        payoffs = np.maximum(P_T_option_T_bond - K, 0.0)
    elif CP == 'PUT':
        payoffs = np.maximum(K - P_T_option_T_bond, 0.0)
    else:
        raise ValueError("CP must be 'CALL' or 'PUT'")
    
    # Discount back to t=0 using 1/B(T_option)
    # V(0) = E^Q[payoff / B(T_option)]
    discounted_payoffs = payoffs / B_T_option
    
    # Take expectation
    option_value = np.mean(discounted_payoffs)
    
    return option_value

# ============================================================================
# MAIN PRICING COMPARISON
# ============================================================================

# Parameters
T_option = 1.0              # Option expiry (1 year)
T_bond = 3.0                # Bond maturity (3 years)
K = 0.90                    # Strike price for the ZCB
r = 0.04                    # Risk-free rate

# Hull-White parameters
lambd = 0.1                 # Mean reversion speed
eta = 0.02                  # Volatility of short rate

# Zero-coupon bond curve (market)
P0T = lambda T: np.exp(-r * T)

# Calculate initial ZCB price
P_0_T_option = P0T(T_option)
P_0_T_bond = P0T(T_bond)

# Calculate forward price of the bond
# F(0, T_option, T_bond) = P(0, T_bond) / P(0, T_option)
forward_price = P_0_T_bond / P_0_T_option

# COmpare Call options price
for option_type in ['CALL']:
    print(f"\n{'='*80}")
    print(f"{option_type} OPTION PRICING")
    print(f"{'='*80}")
    
    # Method 1: Hull-White Analytical Formula
    print(f"\n1. HULL-WHITE ANALYTICAL FORMULA")
    print("-" * 80)
    print("   Using closed-form solution with affine ZCB structure")
    start = time.time()
    analytical_price = HW_ZCB_CallPutOption(option_type, K, lambd, eta, P0T, T_option, T_bond)
    analytical_time = time.time() - start
    print(f"   Price: ${analytical_price:.8f}")
    print(f"   Computation time: {analytical_time:.6f} seconds")
    
    # Method 2: Monte Carlo with Hull-White
    print(f"\n2. MONTE CARLO WITH HULL-WHITE MODEL (Q Measure)")
    print("-" * 80)
    print("   Simulating short rate r(t) using Hull-White dynamics")
    print("   Calculating ZCB price from simulated rates at option expiry")
    
    simulation_sizes = [10000, 50000, 100000]
    
    for n_sims in simulation_sizes:
        np.random.seed(42)  # For reproducibility
        start = time.time()
        mc_price = monte_carlo_zcb_option_hull_white(option_type, K, T_option, T_bond, P0T, lambd, eta, n_sims, )
        mc_time = time.time() - start
        
        error = abs(mc_price - analytical_price)
        error_pct = (error / analytical_price) * 100 if analytical_price > 0 else 0
        
        print(f"\n   Simulations: {n_sims:,}")
        print(f"   Price: ${mc_price:.8f}")
        print(f"   Computation time: {mc_time:.4f} seconds")
        print(f"   Absolute error vs Analytical: ${error:.8f}")
        print(f"   Relative error: {error_pct:.4f}%")
    
    print(f"\n{'-'*80}")
    print(f"COMPARISON SUMMARY - {option_type} OPTION")
    print("-" * 80)
    print(f"Hull-White Analytical Price (Benchmark): ${analytical_price:.8f}")
    print(f"\nMonte Carlo (Hull-White) convergence:")
    for n_sims in simulation_sizes:
        np.random.seed(42)
        mc_price = monte_carlo_zcb_option_hull_white(option_type, K, T_option, T_bond, P0T, lambd, eta, n_sims)
        error_pct = abs((mc_price - analytical_price) / analytical_price) * 100 if analytical_price > 0 else 0
        print(f"  {n_sims:>7,} sims: error = {error_pct:>6.3f}%")

print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)
print("Both methods use the Hull-White model with the same parameters.")
print("Monte Carlo simulates the full dynamics under risk-neutral measure Q.")
print("Analytical formula uses the affine structure of Hull-White ZCBs.")
print("=" * 80)
