import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import prod, isclose
from typing import List
from itertools import product, permutations

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
    
    g = fee_gamma / (fee_gamma + 1 - (P / (S / N)**N))
    
    return g * mid_fee + (1 - g) * out_fee

def _newton_D(ANN: float, gamma: float, xp_unsorted: List[float]) -> float:
    """
    Finding the `D` invariant using Newton's method.

    ANN is A * N**N from the whitepaper multiplied by the
    factor A_MULTIPLIER.
    """
    
    N: int = len(xp_unsorted)
    
    x: List[int] = xp_unsorted.copy()

    A = ANN / N**N
    S: float = sum(x)
    P: float = prod(x)

    D0: float = N * sp.stats.gmean(x)  # initial estimate when solving for D

    # represent cryptoswap as a degree 3N polynomial of D for better stability when solving
    # derive from normal form by removing all instances of D in any denominator (i.e. K0)
    degree: int = 3 * N

    # coefficients ranked by degree
    # leave multiplication of small numbers (i.e. gamma) until the end
    c_3N: float = -(gamma + 1)**2
    c_2N: float = N**N * P * ((gamma + 1) * (gamma + 3) - ANN * gamma**2)
    c_2Nm1: float = A * N**(2 * N) * P * S * gamma**2 # deg. 2N - 1
    c_N: float = -(N**(2 * N) * P**2 * (2 * gamma + 3))
    c_0: float  = N**(3 * N) * P**3

    # Array of coefficients ranked by degree
    coefficients: List[float] = [0.0] * (degree + 1) # includes degree 0 

    # ascending degree
    coefficients[-(3 * N + 1)] = c_3N # index 0 
    coefficients[-(2 * N + 1)] = c_2N
    coefficients[-(2 * N - 1 + 1)] = c_2Nm1
    coefficients[-(N + 1)] = c_N
    coefficients[len(coefficients) - 1] = c_0

    roots: List[float] = np.roots(coefficients)
    #print(roots)

    # function representing our invariant
    def cryptoswap(D: float) -> List[float]:
        K0: float = P / (D / N)**N
        _g1mk0: float = gamma + 1 - K0
        K: float = A * K0 * gamma**2 # / _g1mk0**2

        invariant: float = K * D**(N - 1) * S + P * _g1mk0**2 - K * D**N - (D / N)**N * _g1mk0**2 # this form = 0
        
        #return list(invariant) 
        return invariant

    # filter roots: check if each root is real and then positive and then if cryptoswap returns 0
    good_roots = lambda root: np.isreal(root) and root > 0 
    # and isclose(cryptoswap(root), 0.0, abs_tol=10**-2 * min(xp_unsorted))
    
    promising: List[float] = list(map(float, filter(good_roots, roots)))
    if len(promising) > 0:
        D: float = min(promising, key=cryptoswap) # brings cryptoswap closest to 0
        return D
    else: 
        raise Exception(f"No solution for D: {roots} {xp_unsorted}")
    
    # List[float], dict, int, str are returned
    #D, infodict, ier, mesg = sp.optimize.fsolve(func=cryptoswap, x0=[D0], full_output=True, xtol=1e-10)

    # if fsolve found a solution
    #if ier == 1:
        #return D[0]

    raise Exception(f"D did not converge: {mesg} {ANN} {gamma} {x} {D[0]} {D0}")
 
def _newton_y(i: int, ANN: float, gamma: float, D: float, xp_unsorted: List[float]) -> float:
    """
    Calculating x[i] given other balances x[0..n_coins-1] and invariant D.
    ANN = A * N**N
    """
    N: int = len(xp_unsorted)
    
    x_j: List[float] = xp_unsorted.copy()
    x_j.pop(i) # all j != i

    A: float = ANN / N**N
    S_x_j: float = sum(x_j)
    P_x_j: float = prod(x_j)

    # transform Cryptoswap into a polynomial of y (x[i]):
    # ay**3 + by**2 + cy + d = 0
    K0_j = P_x_j / (D / N)**N
    '''
    a: float = K0_j**3
    b: float = -(2 * gamma + 3) * K0_j**2 + ANN * gamma**2 * K0_j / D
    c: float = ANN * gamma**2 * K0_j * (S_x_j - D) / D + (gamma + 1) * (gamma + 3) * K0_j
    d: float = -(gamma + 1)**2
    '''

    a: float = D * K0_j**3
    b: float = -(2 * gamma + 3) * D * K0_j**2 + ANN * K0_j * gamma**2
    c: float = ANN * K0_j * (S_x_j - D) * gamma**2 + (gamma + 1) * (gamma + 3) * D * K0_j
    d: float = -(gamma + 1)**2 * D
    coefficients = [d, c, b, a] # ascending degree

    roots: List[float] = np.polynomial.polynomial.polyroots(coefficients)
    #print(roots)
    
    y0: float  = (D / N)**N / P_x_j # initial estimate when solving for y

    # function representing our invariant
    def cryptoswap(y: float) -> List[float]:
        S: float  = S_x_j + y
        K0: float = P_x_j * y / (D / N)**N
        _g1mk0: float = gamma + 1 - K0
        K: float = A * K0 * gamma**2 # / _g1mk0**2

        invariant: float = K * D**(N - 1) * (S_x_j + y) + P_x_j * y * _g1mk0**2 - K * D**N - (D / N)**N * _g1mk0**2
        #invariant: float = a * y**3 + b * y**2 + c * y + d

        #return list(invariant)
        return invariant

    # filter roots: check if each root is real and then positive and then if cryptoswap returns 0
    good_roots = lambda root: np.isreal(root) and root > 0 
    #and isclose(cryptoswap(root), 0.0, abs_tol=10**-2 * min(xp_unsorted))
    
    promising: List[float] = list(map(float, filter(good_roots, roots)))
    if len(promising) > 0:
        y: float = min(promising, key=cryptoswap) # brings cryptoswap closest to 0
        return y
    else: 
        raise Exception(f"No solution for D: {roots} {xp_unsorted}")

    # List, dict, int, str are returned
    #y, infodict, ier, mesg = sp.optimize.fsolve(func=cryptoswap, x0=[y0], full_output=True, xtol=1e-10)

    # if fsolve found a solution
    #if ier == 1:
        #print(xp_unsorted, i, roots, y[0], cryptoswap2(y[0]))
        #return y[0]

    raise Exception(f"y did not converge: {mesg} {ANN} {gamma} {xp_unsorted} {D} {y[0]} {y0}")

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
    """
    N = len(xp)
    
    assert i != j  # dev: same input and output coin
    assert i < N  # dev: coin index out of range
    assert j < N  # dev: coin index out of range
    
    xp_copy = xp.copy()
    xp_copy[i] += dx
    
    y: float = _newton_y(j, ANN, gamma, D, xp_copy)
    dy: float = xp_copy[j] - y
    assert dy > 0, f"Negative dy: {xp} {D}"
    xp_copy[j] = y
    
    dy_fee = dy * (1 - _fee(fee_params, xp_copy))
    
    return dy, dy_fee

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
    
    S = sum(xp_copy)
    P = prod(xp_copy)
    
    K0 = P / (D / N)**N
    _g1pk0 = gamma + 1 + K0
    _g1mk0 = gamma + 1 - K0 
    
    above: float = xp_copy[j] * ((ANNG2 / (D * _g1mk0**2)) * ((_g1pk0 * (S - D)) / _g1mk0 + xp_copy[i]) + 1)
    
    below: float = xp_copy[i] * ((ANNG2 / (D * _g1mk0**2)) * ((_g1pk0 * (S - D)) / _g1mk0 + xp_copy[j]) + 1)
    
    # Spot price in token j per token i
    price_ji: float = above / below
    price_ji_fee: float = price_ji * (1 - _fee(fee_params, xp_copy))
    
    return price_ji, price_ji_fee
    
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
    
    P = prod(xp_copy)
    
    K0 = P / (D / N)**N
    # gamma**2 <= GK0 <= (gamma + 1)**2
    GK0 = 2 * K0**3 - (2 * gamma + 3) * K0**2 + (gamma + 1)**2
    
    above: float = xp_copy[j] * (GK0 + ANNG2 * K0 * xp_copy[i] / D)
    
    below: float = xp_copy[i] * (GK0 + ANNG2 * K0 * xp_copy[j] / D)
    
    # Spot price in token j per token i
    price_ji: float = above / below
    price_ji_fee: float = price_ji * (1 - _fee(fee_params, xp_copy))
    
    return price_ji, price_ji_fee

# Parameters taken from Tricrypto-2 (USDT-BTC-ETH) are indicated below
# https://etherscan.io/address/0xd51a44d3fae010294c616388b506acda1bfaae46#readContract

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
dx_step: float = 10**-3 # increment in this fraction of the smallest baLance in xp
dxs: List[float] = list((lambda xp: i * dx_step * min(xp) for i in range(1, dx_iter + 1)))

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
            dx = param_set[4](xp) # dx is based on min(xp), so we feed xp to the generator
            fee_params = param_set[5]
            
            notij = list(set(range(N)) - set((i, j)))
            
            D = _newton_D(ANN, gamma, xp)
            
            dy, dy_fee = get_dy(i, j, ANN, gamma, D, xp, dx, fee_params)
            
            dydx = dy / dx # discretized derivative
            dydx_fee = dy_fee / dx

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

filepaths = ['Cryptoswap-Price-Test.csv', 'Cryptoswap-Price-Test-Stats.csv'] 
column_labels = ["N", "(i, j)", "A", "gamma", "[fee_gamma, mid_fee, out_fee]", "xp[i]", "xp[j]", "[xp[not i or j]]", \
                 "D", "dx", "dy", "dy / dx", "Our Formula", "Onchain Formula", "Our Formula Delta", "Onchain Formula Delta", "Difference in Deltas", "dy with Fee", "dy / dx with Fee", "Our Formula with Fee", "Onchain Formula with Fee", "Our Formula with Fee Delta", "Onchain Formula with Fee Delta", "Difference in Deltas with Fees"]
df = pd.DataFrame(data=table, columns=column_labels)
df.to_csv(filepaths[0])
df.describe().to_csv(filepaths[1])

df.hist(column="Our Formula with Fee Delta")
df.hist(column="Onchain Formula with Fee Delta")
df.hist(column="Difference in Deltas with Fees")
plt.show()

'''
To do:

- use different cryptoswap forms when solving 
    newton_y: use the cubic equation, rewrite to be cubic in y

    newton_D: possible forms can be found in tricrypto2 optimization report
        D0 estimate may also be different

    numpy (poly) root solver may be better than fsolve if using polynomial form

- in newton_y and newton_D, check if the solution is actually a solution > 0 (invariant near 0) 
    use math.isclose - set abstol to 1e-10

    could define tol: float = 1e-10

- don't output unnecessary values
    fee_params, dy, dy / dx, ..., difference in deltas without fees

'''
