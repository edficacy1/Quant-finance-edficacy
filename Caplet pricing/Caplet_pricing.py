"""
Caplet pricing
"""
import numpy as np
import enum 
import matplotlib.pyplot as plt
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

def HW_CapletFloorletPrice_Direct(CP,N,K,lambd,eta,P0T,T1,T2):
    """
    Direct caplet/floorlet pricing formula without using ZCB put option relationship
    
    From Image 3, the caplet pricing is:
    V^CPL(t_0) = N * exp(-A_r(tau_k)) * E^{T_k}[max(exp(-B_r(tau_k)*r(T_{k-1})) - K_hat, 0)]
    
    where K_hat = (1 + tau_k * K) * exp(A_r(tau_k))
    
    Under the T_k forward measure, exp(-B_r * r(T_{k-1})) is log-normally distributed.
    Since r(T_{k-1}) ~ N(mu_r, v_r^2), we have:
    exp(-B_r * r(T_{k-1})) ~ LogNormal(-B_r * mu_r, B_r^2 * v_r^2)
    
    This is a standard call option on a log-normal variable.
    """
    
    if CP == OptionType.CALL:
        # Caplet pricing
        tau_k = T2 - T1  # accrual period
        
        # Get A_r and B_r for the period [T1, T2]
        A_r = HW_A(lambd, eta, P0T, T1, T2)
        B_r = HW_B(lambd, eta, T1, T2)
        
        # Get the mean and variance of r(T1) under T2-forward measure
        mu_r = HW_Mu_FrwdMeasure(P0T, lambd, eta, T1)
        v_r = np.sqrt(HWVar_r(lambd, eta, T1))
        
        # Calculate modified strike: K_hat = (1 + tau_k * K) * exp(A_r)
        K_hat = (1.0 + tau_k * K) * np.exp(A_r)
        
        # The underlying variable is S = exp(-B_r * r(T1))
        # E[S] = exp(-B_r * mu_r + 0.5 * B_r^2 * v_r^2)
        # Var(log(S)) = B_r^2 * v_r^2
        
        # For a call option on a log-normal variable:
        # E[max(S - K, 0)] = E[S] * N(d1) - K * N(d2)
        # where d1 = (log(E[S]/K) + 0.5 * sigma^2) / sigma
        #       d2 = d1 - sigma
        
        # Note: B_r is negative for T2 > T1, so -B_r is positive
        sigma = np.abs(B_r) * v_r
        forward = np.exp(-B_r * mu_r + 0.5 * B_r * B_r * v_r * v_r)
        
        d1 = (np.log(forward / K_hat) + 0.5 * sigma * sigma) / sigma
        d2 = d1 - sigma
        
        # Calculate the option value
        option_value = forward * st.norm.cdf(d1) - K_hat * st.norm.cdf(d2)
        
        # Scale by notional, discount factor P(t0, T_k), and exp(-A_r)
        caplet = N * P0T(T2) * np.exp(-A_r) * option_value
        
        return caplet
        
    elif CP == OptionType.PUT:
        # Floorlet pricing (using put-call parity if needed)
        # For now, return 0 as in original code
        return 0.0

def monte_carlo_caplet_hull_white(N, K, tau, T1, T2, P0T, lambd, eta, n_sims, n_steps=1000, verbose=False):
    """
    Price caplet using Monte Carlo with Hull-White model under Q measure.
    
    Caplet payoff: N * tau * max(L(T_1, T_2) - K, 0) paid at T_2
    
    Process:
    1. Simulate short rate r(t) paths using Hull-White model up to T2
    2. Calculate money market account: B(T2) = exp(∫₀^T2 r(s)ds)
    3. Calculate LIBOR at T1: L(T1, T2) = [1/P(T1,T2) - 1] / tau
    4. Calculate payoff: N * tau * max(L(T1, T2) - K, 0)
    5. Discount: V(0) = E^Q[payoff / B(T2)]
    """
    
    # Generate Hull-White paths up to T2
    paths = GeneratePathsHWEuler(n_sims, n_steps, T2, P0T, lambd, eta)
    r_paths = paths["R"]
    time_grid = paths["time"]
    dt = time_grid[1] - time_grid[0]
    
    # Find index closest to T1
    idx_T1 = np.argmin(np.abs(time_grid - T1))
    
    # Calculate money market account B(T2) = exp(∫₀^T2 r(s)ds)
    # Using trapezoidal integration along each path
    integral_r = np.zeros(n_sims)
    for i in range(n_sims):
        integral_r[i] = np.trapezoid(r_paths[i, :], time_grid)
    
    B_T2 = np.exp(integral_r)
    
    # Extract r(T1) for each path
    r_T1 = r_paths[:, idx_T1]
    
    # Calculate P(T1, T1) = 1 (always)
    P_T1_T1 = 1.0
    
    # Calculate P(T1, T2) using Hull-White formula
    P_T1_T2 = HW_ZCB(lambd, eta, P0T, T1, T2, r_T1)
    
    # Calculate LIBOR: L(T1, T2) = [P(T1,T1)/P(T1,T2) - 1] / tau
    L_T1_T2 = (P_T1_T1 / P_T1_T2 - 1.0) / tau
  

    # Calculate payoffs: N * tau * max(L - K, 0)
    payoffs = N * tau * np.maximum(L_T1_T2 - K, 0.0)
    
    # Discount back to t=0 using 1/B(T2) where B(T2) is the money market account
    # V(0) = E^Q[payoff / B(T2)]
    discounted_payoffs = payoffs / B_T2
    
    # Take expectation
    caplet_value = np.mean(discounted_payoffs)
    
    return caplet_value

# ============================================================================
# MAIN PRICING COMPARISON
# ============================================================================

# Parameters
K = 0.055          # Strike rate (5.5%)
tau = 0.25         # Accrual period (3 months)
T1 = 1.0           # LIBOR fixing time (1 year)
T2 = 1.25          # Payment time (T1 + tau)
r = 0.04           # Risk-free rate for discounting
N = 1.0            # Notional
CP = OptionType.CALL

# Hull-White parameters
lambd = 0.1        # Mean reversion speed
eta = 0.02         # Volatility of short rate

# Zero-coupon bond curve (market)
P0T = lambda T: np.exp(-r * T)

# Calculate initial LIBOR from ZCB curve (CONSISTENT with Hull-White)
# L(0, T1, T2) = [P(0,T1)/P(0,T2) - 1] / tau
P_0_T1 = P0T(T1)
P_0_T2 = P0T(T2)
L_0 = (P_0_T1 / P_0_T2 - 1.0) / tau


# Method 1: Hull-White Analytical Formula
print("\n1. HULL-WHITE ANALYTICAL FORMULA (Benchmark)")
print("-" * 80)
print("   Using affine ZCB formula with A_r and B_r under T_k-forward measure")
start = time.time()
analytical_price = HW_CapletFloorletPrice_Direct(CP,N, K, lambd, eta, P0T, T1, T2)

analytical_time = time.time() - start
print(f"   Price: ${analytical_price:.8f}")
print(f"   Computation time: {analytical_time:.6f} seconds")

# Method 2: Monte Carlo with Hull-White
print("\n2. MONTE CARLO WITH HULL-WHITE MODEL (Q Measure)")
print("-" * 80)
print("   Simulating short rate r(t) using Hull-White dynamics")
print("   Calculating LIBOR from simulated zero-coupon bonds")

simulation_sizes = [10000, 50000, 100000]

for n_sims in simulation_sizes:
    np.random.seed(42)
    start = time.time()
    mc_price = monte_carlo_caplet_hull_white(N, K, tau, T1, T2, P0T, lambd, eta, n_sims, verbose=(n_sims==500))
    mc_time = time.time() - start
    
    error = abs(mc_price - analytical_price)
    error_pct = (error / analytical_price) * 100 if analytical_price > 0 else 0
    
    print(f"\n   Simulations: {n_sims:,}")
    print(f"   Price: ${mc_price:.8f}")
    print(f"   Computation time: {mc_time:.4f} seconds")
    print(f"   Absolute error vs Analytical: ${error:.8f}")
    print(f"   Relative error: {error_pct:.4f}%")

