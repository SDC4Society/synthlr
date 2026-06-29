import numpy as np
from scipy.special import gammaln

def phi_QMS(m, n_plus_gamma):
    """
    Compute phi for QMS

    Args:
        m (int): the size of synthetic data
        n_plus_gamma(float): the size of the mixture data of the raw and dummy data

    Return:
        float: phi
    """
    # the term index takes values from 0 to m-1
    i_vec = np.arange(0, m-2+1)

    # to avoid overflow, evaluate logged values
    log_coe = gammaln(m-1+1) + np.log(n_plus_gamma) - (m-1) * np.log(n_plus_gamma+m)
    log_terms = (i_vec - 1) * np.log(n_plus_gamma+i_vec) + (m - i_vec -1) * np.log(m - i_vec) \
    - gammaln(i_vec+1) - gammaln(m-i_vec-2+1)
    
    # sum up the terms after take exponential 
    return 1 + np.exp(log_coe + log_terms).sum()

