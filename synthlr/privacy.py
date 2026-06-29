import math
from scipy.optimize import brentq

def gamma_min_DMS(epsilon:float, m:int):
    """
    Compute minimum dummy frequency at each cell
    that DMS satisfies epsilon-DP

    Args:
        epsilon(float): privacy parameter
        m(int)        : synthetic data size

    Return:
        float: minimum dummy frequency
    """
    return m / (math.exp(epsilon) - 1)

def gamma_min_QMS(epsilon:float, m:int)->float:
    """
    Compute minimum dummy frequency at each cell
    that QMS satisfies epsilon-DP

    Args:
        epsilon(float): privacy parameter
        m(int)        : synthetic data size

    Return:
        float: minimum dummy frequency
    """
    if epsilon > 1:
        return 1 /(math.exp(epsilon-1)-1)

    elif epsilon == 1:
        return math.sqrt(m)

    else:
        return (1 / epsilon - 1) * m


def gamma_min_QMS_exact(epsilon: float, m: int) -> float:
    """
    Compute the exact minimum dummy frequency per cell for QMS to satisfy epsilon-DP.

    Finds the minimum gamma satisfying the exact condition from Theorem 1.4 (ineq. 1.10):
      (1 + 1/gamma)(1 + 1/(gamma+m))^(m-1) <= exp(epsilon)

    gamma_min_QMS returns an approximation of this value.

    Args:
        epsilon (float): privacy parameter
        m (int)        : synthetic data size

    Return:
        float: minimum dummy frequency per cell
    """
    # log-scale residual: positive means gamma is too small (DP violated)
    def f(gamma):
        return math.log1p(1.0 / gamma) + (m - 1) * math.log1p(1.0 / (gamma + m)) - epsilon

    # Use the approximation as an initial guess and bracket from there
    gamma_approx = gamma_min_QMS(epsilon, m)

    # Find upper bracket where f < 0 (DP satisfied)
    gamma_hi = gamma_approx
    while f(gamma_hi) > 0:
        gamma_hi *= 2.0

    # Find lower bracket where f > 0 (DP violated)
    gamma_lo = gamma_hi / 2.0
    while f(gamma_lo) < 0:
        gamma_lo /= 2.0
        if gamma_lo < 1e-300:
            break

    return brentq(f, gamma_lo, gamma_hi)
