from multiprocessing import Pool, cpu_count
import time

import numpy as np
from scipy.special import gammaln
from scipy.stats import betabinom


def quasi_binomial(p, n, beta):
    """
    Sample a value following a quasi-binomial distribution

    Args:
        p(float)   : succsess probability, 0 <= p <= 1
        n(int)     : number of trials
        beta(float): overdispersion parameter

    Return:
        int: number of success
    """
    assert 0 <= p <= 1

    # guard clause
    if p == 1:
        # all n trials succeed
        return n
    elif p == 0:
        # no trial succeeds
        return 0

    ys = np.arange(0, n+1) # from 0 to n
    
    logprobs = gammaln(n+1) - gammaln(ys+1) - gammaln(n-ys+1) - (n-1) * np.log(1+n*beta) \
    + np.log(p) + (ys-1) * np.log(p + ys * beta) + np.log(1-p) + (n-ys-1) * np.log(1-p + (n-ys)*beta)
    probs = np.exp(logprobs)

    return int(np.random.choice(ys, 1, p=probs)[0])


def _recursive_QMS(m, probs, beta):
    if m == 0:
        return np.zeros(len(probs)).tolist()

    if len(probs) == 1:
        return [m]

    mid = len(probs) // 2
    p1 = sum(probs[:mid])
    p2 = sum(probs[mid:])

    if p1 + p2 == 0:
        return [0] * len(probs)

    m1 = quasi_binomial(p1/(p1+p2), m, beta)
    m2 = m - m1

    left = _recursive_QMS(m1, probs[:mid], beta)
    right = _recursive_QMS(m2, probs[mid:], beta)
    
    return left + right


def _split_QMS(m, probs, beta, layer):
    assert layer > 0
    
    mid = len(probs) // 2
    p1 = sum(probs[:mid])
    p2 = sum(probs[mid:])

    if p1 + p2 == 0:
        return [0] * len(probs)

    m1 = quasi_binomial(p1/(p1+p2), m, beta)
    m2 = m - m1

    new_layer = layer - 1
    if new_layer == 0:
        return ([m1, m2], [probs[:mid], probs[mid:]])
    
    left = _split_QMS(m1, probs[:mid], beta, new_layer)
    right = _split_QMS(m2, probs[mid:], beta, new_layer)

    return (left[0]+right[0], left[1]+right[1])


def _start_recursive_QMS(args):
    """
    wrapper of _recursive_QMS to use multiprocessing
    """
    m, probs, beta = args
    
    return _recursive_QMS(m, probs, beta)


def QMS(m:int, ns, layer:int=3):
    """
    Data synthesis by quasi-multinomial sampling (QMS)

    Args:
        m(int)             : the size of the synthetic data
        ns(list or ndarray): frequency vector of data
        layer(int)         : 2**layer processes run

    Return:
        list: synthesized frequency vector
    """
    assert m > 0

    # wrap list in ndarray
    ns_array = np.array(ns)

    # interplet the data as a probability distribution
    probs = ns_array/ns_array.sum()

    # set overdispersion parameter
    beta = 1 / ns_array.sum()
    
    # single process runs
    if layer < 1:
        out = _recursive_QMS(m, probs, beta)
        
        return out
    
    # split the saple task to 2**layer subtasks
    ms, ps = _split_QMS(m, probs, beta, layer)
    processes = 2 ** layer # too many process can cause an error
    
    # check the split process valid
    assert len(ms) == processes

    betas = [beta] * len(ms)
    tasks = list(zip(ms, ps, betas))
    
    with Pool(processes=processes) as pool:
        results = pool.map(_start_recursive_QMS, tasks)

    # join the results, which is like [[...], [...], ...]
    out = []
    for j in range(len(results)):
        out += results[j]
    
    return out


def _recursive_DMS(m, ns):
    if m == 0:
        return np.zeros(len(ns)).tolist()

    if len(ns) == 1:
        return [m]

    mid = len(ns) // 2
    alpha = sum(ns[:mid])
    beta = sum(ns[mid:])

    if alpha + beta  == 0:
        return [0] * len(ns)

    m1 = betabinom.rvs(m, alpha, beta)
    m2 = m - m1

    left = _recursive_DMS(m1, ns[:mid])
    right = _recursive_DMS(m2, ns[mid:])
    
    return left + right


def _start_recursive_DMS(args):
    m, ns = args
    
    return _recursive_DMS(m, ns)


def _split_DMS(m, ns, layer):
    assert layer > 0
    
    mid = len(ns) // 2
    alpha = sum(ns[:mid])
    beta = sum(ns[mid:])

    if alpha + beta == 0:
        return [0] * len(ns)

    m1 = betabinom.rvs(m, alpha, beta)
    m2 = m - m1

    new_layer = layer - 1
    if new_layer == 0:
        return ([m1, m2], [ns[:mid], ns[mid:]])
    
    left = _split_DMS(m1, ns[:mid], new_layer)
    right = _split_DMS(m2, ns[mid:], new_layer)

    return (left[0]+right[0], left[1]+right[1])


def DMS(m:int, ns, layer:int=3):
    """
    Data synthesis by Dirichlet-multinomial sampling (DMS)

    Args:
        m(int)             : the size of the synthetic data
        ns(list or ndarray): frequency vector of data
        layer(int)         : 2**layer processes run

    Return:
        list: synthesized frequency vector
    """
    assert m > 0

    if layer < 1:
        return _recursive_DMS(m, ns)
    
    ms, ns = _split_DMS(m, ns, layer)
    processes = 2 ** layer
    
    assert len(ms) == processes

    tasks = list(zip(ms, ns))
    
    with Pool(processes=processes) as pool:
        results = pool.map(_start_recursive_DMS, tasks)

    out = []
    for j in range(len(results)):
        out += results[j]
    
    return out


    

if __name__ == '__main__':
    arrow_n = np.arange(1000000)
    dummies = np.ones(len(arrow_n)) * 0.1
    m = 100000

    import sys
    sys.setrecursionlimit(10000)

    start_time = time.time()
    arrow_m = DMS(m, arrow_n+dummies)
    end_time = time.time()

    print("DMS")
    print(len(arrow_n), m)
    print(arrow_m)
    print(f"Execution time: {end_time - start_time} seconds")
    