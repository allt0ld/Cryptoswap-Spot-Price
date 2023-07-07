from typing import List
from pandas import DataFrame
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from itertools import product, permutations

MIN_GAMMA = 10**-8
MAX_GAMMA = 10**2 # Normally 10**-2, raised for experimentation

def prod(x: List[float]) -> float:
    """
    Returns the product of a list of numbers.
    """
    
    P: float = 1.0
    
    for _x in x:
        P *= _x
        
    return P

def _geometric_mean(x: List[float]) -> float:
    """
    Returns the geometric mean of a list of numbers.
    """
    x: List[float] = x
    N: int = len(x)
    
    P: float = prod(x)
        
    return P**(1/N)

def _newton_D(ANN: float, gamma: float, xp_unsorted: List[float]) -> float:
    """
    Finding the `D` invariant using Newton's method.

    ANN is A * N**N from the whitepaper multiplied by the
    factor A_MULTIPLIER.
    """
    
    N: int = len(xp_unsorted)
    
    min_A = N**N * 0.1
    max_A = N**N * 100000
    
    if ANN > max_A or ANN < min_A:
        raise Exception("Unsafe value for A")
    if gamma > MAX_GAMMA or gamma < MIN_GAMMA:
        raise Exception("Unsafe value for gamma")
    
    x: List[int] = xp_unsorted.copy() 
    x.sort(reverse=True) # highest to lowest
    
    assert(x[0] >= 10**-9 and x[0] <= 10**15) # dev: unsafe values x[0]
    
    for i in range(1, N):
        assert(x[i] / x[0] >= 10**-4) # dev: unsafe values x[i]

    A = ANN / N**N
    S: float = sum(x)
    P: float = prod(x)

    D0: float = N * _geometric_mean(x) # initial estimate when solving for D

    # callable function representing our invariant to pass to scipy's solver fsolve
    def cryptoswap(D: float) -> List[float]:
        K0: float = P / (D / N)**N
        _g1mk0: float = gamma + 1 - K0
        K: float = A * K0 * gamma**2 / _g1mk0**2
        
        return list(K * D**(N - 1) * S + P - K * D**N - (D / N)**N) # this form = 0

    # ndarray, dict, int, str are the respective types returned
    # maxfev - 255 iterations max
    D, infodict, ier, mesg = fsolve(func=cryptoswap, x0=[D0], full_output=True, maxfev=255)

    # ier == 1 means fsolve found a solution
    if ier == 1:
        for _x in x:
            frac: float = _x / D
            if frac < 10**-2 or frac > 10**2:
                raise Exception("Unsafe value for x[i]")

        return D[0]

    raise Exception(f"D did not converge: {mesg}")

def _fee(fee_params: List[float], xp: List[float]) -> float:
    """
    f = fee_gamma / (fee_gamma + (1 - K))
    where
    K = prod(x) / (sum(x) / N)**N
    """
    N: int = len(xp)
    
    S: float = sum(xp)
    P: float = prod(xp)
    
    fee_gamma = fee_params[0]
    mid_fee = fee_params[1]
    out_fee = fee_params[2]
    
    f = fee_gamma / (fee_gamma + 1 - (P / (S / N)**N))
    
    return f * mid_fee + (1 - f) * out_fee
 
def _newton_y(i: int, ANN: float, gamma: float, D: float, xp_unsorted: List[float]) -> float:
    """
    Calculating x[i] given other balances x[0..n_coins-1] and invariant D.
    ANN = A * N**N
    """
    N: int = len(xp_unsorted)
    
    min_A = N**N * 0.1
    max_A = N**N * 100000
    if ANN > max_A or ANN < min_A:
        raise Exception("Unsafe value for A")
    if gamma > MAX_GAMMA or gamma < MIN_GAMMA:
        raise Exception("Unsafe value for gamma")
    if D > 10**15 or D < 0.01: # normally d < 0.1 is unsafe, lowered for experimentation
        raise Exception("Unsafe value for D")
    
    # adapt to N coins
    x_j: List[float] = xp_unsorted.copy()
    x_j.pop(i) # all j != i
    x_j.sort(reverse=True) # highest to lowest

    A: float = ANN / N**N
    S_x_j: float = sum(x_j)
    P_x_j: float = prod(x_j)
     
    y0: float  = (D / N)**N / P_x_j # initial estimate when solving for y
    # K0_i: float =  P_x_j / (D / N)**(N - 1)

    for x in x_j:
        assert (0.01 <= x / D <= 100)  # dev: unsafe values x[i]

    # callable function representing our invariant to pass to scipy's solver fsolve
    def cryptoswap(y: float) -> List[float]:
        K0: float = P_x_j * y / (D / N)**N
        _g1mk0: float = gamma + 1 - K0
        K: float = A * K0 * gamma**2 / _g1mk0**2

        return list(K * D**(N - 1) * (S_x_j + y) + P_x_j * y - K * D**N - (D / N)**N) #  this form = 0

    # ndarray, dict, int, str are the respective types returned
    # maxfev - 255 iterations max
    y, infodict, ier, mesg = fsolve(func=cryptoswap, x0=[y0], full_output=True, maxfev=255)

    # ier == 1 means fsolve found a solution
    if ier == 1:
        frac: float = y / D
        assert 0.01 <= frac <= 100 # dev: unsafe value for y
        return y[0]

    raise Exception(f"y did not converge: {mesg}")

def get_dy(i: int, j: int, ANN: float, gamma: float, D: float, xp: List[float], dx: float, fee_params: List[float]) -> (float, float):
    """
    Calculate the amount received from swapping `dx`
    amount of the `i`-th coin for the `j`-th coin.

    Parameters
    ----------
    i: int
        Index of 'in' coin
    j: int
        Index of 'out' coin
    ANN: float
        A * N**N
    gamma: float
        Coefficient controlling liquidity concentration
    D: float 
        Total deposits, akin to liquidity depth
    xp: List[float]
        Amounts of each coin
    dx: float
        The input of coin i
    fee_params: List[float]
        List containing fee gamma, mid fee, and out fee for 
        the fee calculation
    
    Returns
    -------
    float
        The output of coin j
    float 
        The output of coin j minus the swap fee
    
    Note
    ----
    This is a "view" function; it doesn't change the state of the pool.
    """
    N = len(xp)
    
    assert i != j  # dev: same input and output coin
    assert i < N  # dev: coin index out of range
    assert j < N  # dev: coin index out of range
    
    xp_copy = xp.copy()
    xp_copy[i] += dx
    
    y: float = _newton_y(j, ANN, gamma, D, xp_copy)
    dy: float = xp_copy[j] - y
    xp_copy[j] = y
    
    dy_fee = dy - _fee(fee_params, xp_copy) * dy
    
    return dy, dy_fee

def get_dydx(i: int, j: int, ANN: float, gamma: float, D: float, xp: List[float], dx: float, fee_params: List[float]) -> (float, float):
    """
    Calculate the effective swap price after trading 
    'dx' amount of the 'i'-th coin for dy of the 'j'-th coin.
    Returns in coin j per coin i in D units.

    Parameters
    ----------
    i: int
        Index of 'in' coin
    j: int
        Index of 'out' coin
    ANN: float
        A * N**N
    gamma: float
        Coefficient controlling liquidity concentration
    D: float 
        Total deposits, akin to liquidity depth
    xp: List[float]
        Amounts of each coin
    fee_params: List[float]
        List containing fee gamma, mid fee, and out fee for 
        the fee calculation
        
    Returns
    -------
    float
        The output of coin j divided by the input of coin i
    float
        The output of coin j divided by the input of coin i minus the swap fee

    Note
    ----
    This is a "view" function; it doesn't change the state of the pool.
    """
    dy, dy_fee = get_dy(i, j, ANN, gamma, D, xp, dx, fee_params)
    
    p: float = dy / dx
    p_fee: float = dy_fee / dx
        
    return p, p_fee

def spot_price(i: int, j: int, ANN: float, gamma: float, D: float, xp: List[float], fee_params: List[float]) -> (float, float):
    """
    Calculate a spot price in coin j per coin i in D units.
    Simply differentiate the Cryptoswap equation F
    to derive the formula below, equal to (dF/dy)/(dF/dx),
    and simplify.
    
    Parameters
    ----------
    i: int
        Index of 'in' coin
    j: int
        Index of 'out' coin
    ANN: float
        A * N**N
    gamma: float
        Coefficient controlling liquidity concentration
    D: float 
        Total deposits, akin to liquidity depth
    xp: List[float]
        Amounts of each coin
    fee_params: List[float]
        List containing fee gamma, mid fee, and out fee for 
        the fee calculation
        
    Returns
    -------
    float
        The spot price of coin j per coin i given current xp
    float
        The spot price of coin j per coin i given current xp minus the swap fee
    """
    N = len(xp)
    
    assert i != j  # dev: same input and output coin
    assert i < N  # dev: coin index out of range
    assert j < N  # dev: coin index out of range
    
    ANNG2 = ANN * gamma**2 # A * N**N * gamma**2  
    
    xp_copy = xp.copy()
    # xp_copy[i] += dx
    
    S = sum(xp_copy)
    P = prod(xp_copy)
    
    K0 = P / (D / N)**N
    _g1pk0 = gamma + 1 + K0
    _g1mk0 = gamma + 1 - K0 
    
    above: float = xp_copy[i] * ((ANNG2 / (D * _g1mk0**2)) * ((_g1pk0 * (S - D)) / _g1mk0 + xp_copy[j]) + 1)
    
    below: float = xp_copy[j] * ((ANNG2 / (D * _g1mk0**2)) * ((_g1pk0 * (S - D)) / _g1mk0 + xp_copy[i]) + 1)
    
    # Spot price in token j per token i
    price_ji: float = above / below
    
    return price_ji, price_ji * (1 - _fee(fee_params, xp_copy))
    
def get_p(i: int, j: int, ANN: float, gamma: float, D: float, xp: List[float], fee_params: List[float]) -> (float, float):
    """
    "Optimized" spot price calculation in coin j per coin i in D units. 
    Implemented in onchain Tricrypto-NG pools and requires a 
    more complex derivation than the formula above.
    
    i AND j MAYBE UNNECESSARY (according to onchain implementation)
    
    Parameters
    ----------
    i: int
        Index of 'in' coin
    j: int
        Index of 'out' coin
    ANN: float
        A * N**N
    gamma: float
        Coefficient controlling liquidity concentration
    D: float 
        Total deposits, akin to liquidity depth
    xp: List[float]
        Amounts of each coin
    fee_params: List[float]
        List containing fee gamma, mid fee, and out fee for 
        the fee calculation
        
    Returns
    -------
    float
        The spot price of coin j per coin i given current xp
    float
        The spot price of coin j per coin i given current xp minus the swap fee
    """
    N = len(xp)
    
    assert i != j  # dev: same input and output coin
    assert i < N  # dev: coin index out of range
    assert j < N  # dev: coin index out of range
    
    ANNG2 = ANN * gamma**2
    
    xp_copy = xp.copy()
    # xp_copy[i] += dx
    
    P = prod(xp_copy)
    
    K0 = P / (D / N)**N
    # gamma**2 <= GK0 <= (gamma + 1)**2
    GK0 = 2 * K0**3 - (2 * gamma + 3) * K0**2 + (gamma + 1)**2
    
    above: float = xp_copy[i] * (GK0 + ANNG2 * K0 * xp_copy[j] / D)
    
    below: float = xp_copy[j] * (GK0 + ANNG2 * K0 * xp_copy[i] / D)
    
    # Spot price in token j per token i
    price_ji: float = above / below
    
    return price_ji, price_ji * (1 - _fee(fee_params, xp_copy))

# Parameters taken from Tricrypto-2 (USDT-BTC-ETH) are indicated below

Ns: List[int] = [2, 3]

A_base: float = 6.3246 # taken from pool named above
A_iter: int = 8 # num. of A to iterate through, including A_base  
A_step: float = 2.5 # increment A by this much

gamma_base: float = 1.1809167828997 # taken from pool named above (at 10**-5 decimals place) and adjusted to 10**1 decimals place

# First list is within gamma's range of values: 10**-8 <= gamma <= 10**-2. Second list is to approximate Stableswap, which Cryptoswap becomes at gamma ~10
gammas: List[float] = [gamma_base * 10**i for i in range(-8, -3 + 1)] + [gamma_base * 10**i for i in range(-2, 1 + 1)]

# taken from pool named above
fee_gamma: float = 0.0005
mid_fee: float = 0.0003 # 3 bps
out_fee: float =  0.003 # 30 bps
fee_params_lists: List[List[float]] = [[fee_gamma, mid_fee, out_fee],]

# xp is normalized to token 0 units (not factoring in precision)
# Slippage is assumed to benefit the pool

# Each List[List[List[float]]] below groups lists of lists of balances for each of the pools' coins by the general degree of imbalance

# In every list of list of balances, scenarios are separated by line, with each line ordered from 
# least to most imbalanced: 

# 2-coin crveth is an example (ETH-CRV):
# perfect balance
# eth dump
# crv dump

two_coin_xps: List[List[List[float]]] = [[[50_000_000, 50_000_000], \
    [55_000_000, 45_500_000], [64_000_000, 37_300_000], \
    [44_500_000, 56_000_000], [40_000_000, 61_000_000], \
    ], \
                                        [[50_000_000, 50_000_000], \
    [100_000_000, 1_500_000], [130_000_000, 1_300_000], \
    [1_900_000, 99_000_000], [124_000_000, 1_240_900], \
    ], \
                                        [[0.1, 0.1], \
    [0.19, 0.011], [0.21, 0.0021], \
    [0.2, 0.008], [0.25, 0.0029], \
    ], \
                                         [[4.99 * 10**14, 4.99 * 10**14], \
    [9.7 * 10**14, 2.5 * 10**13], [9.999 * 10**14, 9.9999 * 10**12], \
    [2.31 * 10**13, 9.75 * 10**14], [9.929 * 10**12, 9.89 * 10**14], \
    ], \
                                        ]

# 3-coin tricrypto-2 is an example (USDT-BTC-ETH):
# perfect balance
# usdt dump: btc and eth equal
# usdt dump: more btc bought than eth
# udst dump: more eth bought than btc
# btc/eth dump: btc and eth equal
# btc/eth dump: more btc sold than eth  
# btc/eth dump: more eth sold than btc

three_coin_xps: List[List[List[float]]] = [[[50_000_000, 50_000_000, 50_000_000], \
    [50_900_000, 49_600_000, 49_600_000], [51_600_000, 49_300_000, 49_300_000], \
    [51_000_000, 49_300_000, 49_775_000], [53_000_000, 48_000_000, 49_250_000], \
    [50_500_000, 49_850_000, 49_675_000], [52_300_000, 49_600_000, 48_200_000], \
    [48_825_000, 50_600_000, 50_600_000], [47_500_000, 51_220_000, 51_220_000], \
    [49_500_000, 50_300_000, 50_210_000], [48_100_000, 51_470_000, 50_400_000], \
    [49_340_000, 50_150_000, 50_500_000], [48_250_000, 50_200_000, 51_500_000], \
    ], \
                                      [[50_000_000, 50_000_000, 50_000_000], \
    [60_000_000, 45_000_000, 45_000_000], [75_000_000, 39_000_000, 39_000_000], \
    [63_000_000, 41_000_000, 48_000_000], [78_000_000, 30_000_000, 46_000_000], \
    [67_000_000, 46_200_000, 39_400_000], [74_000_000, 48_000_000, 30_000_000], \
    [35_500_000, 57_500_000, 57_500_000], [22_000_000, 65_000_000, 65_000_000], \
    [42_000_000, 56_000_000, 51_750_000], [33_000_000, 64_500_000, 51_800_000], \
    [39_000_000, 54_250_000, 56_000_000], [44_000_000, 51_100_000, 55_700_000], \
    ], \
                                        [[50_000_000, 50_000_000, 50_000_000], \
    [152_000_000, 2_000_000, 2_000_000], [151_000_000, 1_560_000, 1_560_000], \
    [150_000_000, 3_000_000, 4_500_000], [155_000_000, 2_000_000, 3_000_000], \
    [150_700_000, 2_000_000, 1_500_000], [155_500_000, 3_000_000, 1_800_000], \
    [2_500_000, 77_100_000, 77_100_000], [2_000_000, 77_250_000, 77_250_000], \
    [1_700_000, 91_500_000, 57_000_000], [1_600_000, 100_000_000, 55_000_000], \
    [1_900_000, 55_000_000, 96_000_000], [1_650_000, 65_000_000, 96_000_000], \
    ], \
                                          [[0.1, 0.1, 0.1], \
    [0.29, 0.005, 0.005], [0.301, 0.0046, 0.0046], \
    [0.3, 0.03, 0.045], [0.31, 0.009, 0.02], \
    [0.2999, 0.035, 0.02], [0.3087, 0.0043, 0.008], \
    [0.04, 0.145, 0.145], [0.007, 0.15, 0.15], \
    [0.04, 0.17, 0.1297], [0.0032, 0.18, 0.132], \
    [0.036, 0.12, 0.177], [0.0036, 0.13, 0.17], \
    ], \
                                          [[3.33 * 10**14, 3.33 * 10**14, 3.33 * 10**14], \
    [8.99 * 10**14, 5 * 10**13, 5 * 10**13], [9.73 * 10**14, 1.3 * 10**13, 1.3 * 10**13], \
    [8.85 * 10**14, 2.34 * 10**13, 3.22 * 10**13], [9.69 * 10**14, 1.04 * 10**13, 2 * 10**13], \
    [8.58 * 10**14, 1.015 * 10 ** 14, 3.16 * 10**13], [9.27 * 10**14, 5.73 * 10**13, 1.31 * 10**13], \
    [3.5 * 10**13, 4.8 * 10**14, 4.8 * 10**14], [1.37 * 10**13, 4.93 * 10**14, 4.93 * 10**14], \
    [7.1 * 10**13, 4.22 * 10**14, 4. * 10**14], [5.16 * 10**13, 4.03 * 10**14, 4.89 * 10**14], \
    [1.632 * 10**14, 3.93 * 10**14, 4.17 * 10**14], [1.21 * 10**13, 4.46 * 10**14, 5.4 * 10**14], \
    ], \
                                          ]

# map N to a list of lists of lists of N-length balances
xps: dict[int: List[List[List[float]]]] = {}
xps[2] = two_coin_xps
xps[3] = three_coin_xps

dx_iter: int = 4 # num. of dx to iterate through  
dx_step: float = 10**-6 # increment dx by this much in token 0 units (not factoring in precision)
dxs: List[float] = [i * dx_step for i in range(1, dx_iter + 1)]

# for csv output
table = []

for N in Ns:

    ij: List[tuple] = list(permutations(range(N), r=2)) # all possible ordered i-j pairs as tuples

    ANNs: List[float] = [A * N**N for A in [A_base + i * A_step for i in range(A_iter)]] 

    for xp in xps[N]:

        # all possible ordered combinations of the parameters defined above
        for param_set in product(ij, ANNs, gammas, xp, dxs, fee_params_lists):
            # indices in param_set: 0 - (i, j); 1 - ANN; 2 - gamma; 3 - xp; 4 - dx; 5 - [fee_gamma, mid_fee, out_fee]
            i, j = param_set[0]
            ANN = param_set[1]
            gamma = param_set[2]
            xp = param_set[3]
            dx = param_set[4]
            fee_params = param_set[5]
            
            notij = list(set(range(N)) - set((i, j)))
            
            D = _newton_D(ANN, gamma, xp)
            
            dydx, dydx_fee = get_dydx(i, j, ANN, gamma, D, xp, dx, fee_params) # discretized derivative
            dy = dydx * dx
            dy_fee = dydx_fee * dx
            
            simple_derivation, simple_derivation_fee = spot_price(i, j, ANN, gamma, D, xp, fee_params) # our formula
            optimized_derivation, optimized_derivation_fee = get_p(i, j, ANN, gamma, D, xp, fee_params) # onchain formula
            
            simple_derivation_delta = dydx - simple_derivation
            optimized_derivation_delta = dydx - optimized_derivation
            delta_diff = simple_derivation_delta - optimized_derivation_delta
        
            simple_derivation_delta_fee = dydx_fee - simple_derivation_fee
            optimized_derivation_delta_fee = dydx_fee - optimized_derivation_fee
            delta_diff_fee = simple_derivation_delta_fee - optimized_derivation_delta_fee
            
            # Each variable matches the corresponding column label below
            table.append([N, (i, j), ANN / N**N, gamma, fee_params, xp[i], xp[j], [xp[k] for k in notij], \
                D, dx, dy, dydx, simple_derivation, optimized_derivation, simple_derivation_delta, \
                          optimized_derivation_delta, delta_diff, \
                          dy_fee, dydx_fee, simple_derivation_fee, optimized_derivation_fee, \
                          simple_derivation_delta_fee, optimized_derivation_delta_fee, delta_diff_fee])

# think of a possible better way than simply overwriting every time
filepath = 'Cryptoswap-Price-Test.csv' 
column_labels = ["N", "(i, j)", "A", "gamma", "[fee_gamma, mid_fee, out_fee]", "xp[i]", "xp[j]", "[xp[not i or j]]", \
                 "D", "dx", "dy", "dy / dx", "Our Formula", "Onchain Formula", "Our Formula Delta", "Onchain Formula Delta", "Difference in Deltas", "dy with Fee", "dy / dx with Fee", "Our Formula with Fee", "Onchain Formula with Fee", "Our Formula with Fee Delta", "Onchain Formula with Fee Delta", "Difference in Deltas with Fees"]
df = DataFrame(data=table, columns=column_labels)
df.to_csv(filepath)

df.hist(column="Our Formula with Fee Delta")
df.hist(column="Onchain Formula with Fee Delta")
df.hist(column="Difference in Deltas with Fees")
plt.show()

'''
have various A, gamma, balances, and n -> write all functions as if n could be > 2
    For simplicity, all these variables should be inputs in each function as necessary -> declare all at bottom of script to pass right into functions
solve for D under Cryptoswap -> _newton_D
choose a very small dx for x_i, solve for x_j -> _newton_y, get the dy for coin j, divide dy by dx to get discretized derivative
    Make sure i and j in all functions have the same meaning (j - output, i - input)
    Specifically, report all spot prices in coin j (output) per coin i (input)
compare the discretized derivative to each price formula *given the same inputs* A, gamma, balances, -> same solved D, and same dx
After each dy / dx calculation apply the fee formula given xp (including dx)
output results in csv form, probably using pandas to_csv
''' 

# Script flow
'''
Constants: MIN_GAMMA, MAX_GAMMA,
    MIN_A, MAX_A - declare within each function given A and N

(NOT NECESSARY FOR NOW)
_xp(x, price_scale, precisions)

_geometric_mean(N, x_unsorted)
_newton_D(ANN, gamma, xp)

_fee(fee_gamma, mid_fee, out_fee, xp)

_newton_y(i, ANN, gamma, D, xp)
get_dy(i, j, ANN, gamma, D, xp, dx, fee_params)
get_dydx(i, j, ANN, gamma, D, xp, dx, fee_params)
    
spot_price(i, j, ANN, gamma, D, xp)
    
get_p(i, j, ANN, gamma, D, xp)
'''

# Questions:
'''

Some thoughts on introducing "noise" to inputs
    
    dx as balance mod (1 bp? * balance)? - with random() or brownian noise, which lies under a normal distribution 
    
    (make things more automatable this way) (can exclude random() or mod part)
            
        why mod? will help further vary the dx if we ever feed in randomly generated balances (variance of taking mod percentage vs. just percentage)
        
            mod floats seems lossy tho
    
Deriving price_scale from the balances assumes the EMA hasn't been applied - is this acceptable for our tests?
    How do you derive price_scale from balances when n > 2? (Probably assume the curve was newly rebalanced)
        
        check _tweak_price behaviour of tricrypto-2 or tricrypto-usdc; p is sent in from _exchange
    
Using copy() a lot - will this consume too much memory in production?
'''

