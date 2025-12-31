
import numpy as np
import enum 
import scipy.stats as st
import scipy.integrate as integrate
import time

# This class defines puts and calls
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T,P0T, lambd, eta):    
    # time-step needed for differentiation
    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    theta = lambda t: 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-np.exp(-2.0*lambd*t))      
    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    R = np.zeros([NoOfPaths, NoOfSteps+1])
    R[:,0]=r0
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        R[:,i+1] = R[:,i] + lambd*(theta(time[i]) - R[:,i]) * dt + eta* (W[:,i+1]-W[:,i])
        time[i+1] = time[i] +dt
        
    # Outputs
    paths = {"time":time,"R":R}
    return paths

def HW_theta(lambd,eta,P0T):
    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    theta = lambda t: 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-np.exp(-2.0*lambd*t))
    return theta
    
def HW_A(lambd,eta,P0T,T1,T2):
    tau = T2-T1
    zGrid = np.linspace(0.0,tau,250)
    B_r = lambda tau: 1.0/lambd * (np.exp(-lambd *tau)-1.0)
    theta = HW_theta(lambd,eta,P0T)    
    temp1 = lambd * integrate.trapezoid(theta(T2-zGrid)*B_r(zGrid),zGrid)
    
    temp2 = eta*eta/(4.0*np.power(lambd,3.0)) * (np.exp(-2.0*lambd*tau)*(4*np.exp(lambd*tau)-1.0) -3.0) + eta*eta*tau/(2.0*lambd*lambd)
    
    return temp1 + temp2

def HW_B(lambd,eta,T1,T2):
    return 1.0/lambd *(np.exp(-lambd*(T2-T1))-1.0)

def HW_ZCB(lambd,eta,P0T,T1,T2,rT1):
    B_r = HW_B(lambd,eta,T1,T2)
    A_r = HW_A(lambd,eta,P0T,T1,T2)
    return np.exp(A_r + B_r *rT1)

def HW_Mu_FrwdMeasure(P0T,lambd,eta,T):
    # time-step needed for differentiation
    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    theta = HW_theta(lambd,eta,P0T)
    zGrid = np.linspace(0.0,T,500)
    
    theta_hat =lambda t,T:  theta(t) + eta*eta / lambd *1.0/lambd * (np.exp(-lambd*(T-t))-1.0)
    
    temp =lambda z: theta_hat(z,T) * np.exp(-lambd*(T-z))
    
    r_mean = r0*np.exp(-lambd*T) + lambd * integrate.trapezoid(temp(zGrid),zGrid)
    
    return r_mean

def HWVar_r(lambd,eta,T):
    return eta*eta/(2.0*lambd) *( 1.0-np.exp(-2.0*lambd *T))

def HW_ZCB_CallPutPrice_Direct(CP, K, lambd, eta, P0T, T1, T2):
    """
    Direct ZCB Call/Put pricing formula using log-normal approach
    Similar to the caplet direct pricing method
    
    From Image 1, the ZCB option pricing is:
    V^ZCB(t_0, T) = P(t_0, T) * E^T [max(α̃ * exp(A_r(τ) + B_r(τ)*r(T)) - K, 0)]
    
    where α̃ = exp(-A_r(τ)) for the correct normalization
    
    This simplifies to pricing a call on:
    S = exp(A_r + B_r * r(T1))
    
    Under the T1 forward measure, r(T1) ~ N(mu_r, v_r^2)
    So S is log-normally distributed.
    """
    
    # Get A_r and B_r for the period [T1, T2]
    A_r = HW_A(lambd, eta, P0T, T1, T2)
    B_r = HW_B(lambd, eta, T1, T2)
    
    # Get the mean and variance of r(T1) under T1-forward measure
    mu_r = HW_Mu_FrwdMeasure(P0T, lambd, eta, T1)
    v_r = np.sqrt(HWVar_r(lambd, eta, T1))
    
    # The ZCB price P(T1, T2) = exp(A_r + B_r * r(T1))
    # This is log-normally distributed
    # E[P(T1,T2)] = exp(A_r + B_r * mu_r + 0.5 * B_r^2 * v_r^2)
    # Volatility of log(P(T1,T2)) = B_r * v_r
    
    # Note: B_r is negative for T2 > T1
    sigma = np.abs(B_r) * v_r
    forward = np.exp(A_r + B_r * mu_r + 0.5 * B_r * B_r * v_r * v_r)
    
    if CP == OptionType.CALL:
        # Call option pricing
        d1 = (np.log(forward / K) + 0.5 * sigma * sigma) / sigma
        d2 = d1 - sigma
        
        # Standard Black-Scholes formula for log-normal variable
        option_value = forward * st.norm.cdf(d1) - K * st.norm.cdf(d2)
        
        # Discount back from T1 to t0
        value = P0T(T1) * option_value
        
        return value
        
    elif CP == OptionType.PUT:
        # Put option pricing using put-call parity
        # Put = Call - Forward + K * DF
        call_value = HW_ZCB_CallPutPrice_Direct(OptionType.CALL, K, lambd, eta, P0T, T1, T2)
        
        # Forward price of ZCB: P(t0, T2)
        # Strike discounted: K * P(t0, T1)
        put_value = call_value - P0T(T2) + K * P0T(T1)
        
        return put_value


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
CP = OptionType.CALL

# Zero-coupon bond curve (market)
P0T = lambda T: np.exp(-r * T)

# Calculate initial ZCB price
P_0_T_option = P0T(T_option)
P_0_T_bond = P0T(T_bond)

# Calculate forward price of the bond
# F(0, T_option, T_bond) = P(0, T_bond) / P(0, T_option)
forward_price = P_0_T_bond / P_0_T_option

# COmpare Call options price


# Method 1: Hull-White Analytical Formula
print(f"\n1. HULL-WHITE ANALYTICAL FORMULA")
print("-" * 80)
print("   Using closed-form solution with affine ZCB structure")
start = time.time()
analytical_price = HW_ZCB_CallPutPrice_Direct(CP, K, lambd, eta, P0T, T_option, T_bond)

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
    mc_price = monte_carlo_zcb_option_hull_white('CALL', K, T_option, T_bond, P0T, lambd, eta, n_sims, )
    mc_time = time.time() - start
    
    error = abs(mc_price - analytical_price)
    error_pct = (error / analytical_price) * 100 if analytical_price > 0 else 0
    
    print(f"\n   Simulations: {n_sims:,}")
    print(f"   Price: ${mc_price:.8f}")
    print(f"   Computation time: {mc_time:.4f} seconds")
    print(f"   Absolute error vs Analytical: ${error:.8f}")
    print(f"   Relative error: {error_pct:.4f}%")
