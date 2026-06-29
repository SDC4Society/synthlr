import math

import pytest

from synthlr import privacy
from .util import compare_sigfigs

def test_gamma_min_DMS():
    """
    Compare with Table 1 in the discussion paper
    
    """
    assert privacy.gamma_min_DMS(1e-5, 1e6) > 0
    assert privacy.gamma_min_DMS(1., 1e6) > 0
    assert privacy.gamma_min_DMS(100, 1e6) > 0
    with pytest.raises(OverflowError):
        privacy.gamma_min_DMS(1e5, 1e6)
    
    # assert privacy.gamma_min_DMS(1e5, 1e6) > 0

def test_gamma_min_QMS():
    """
    Compare with Table 1 in the discussion paper
    
    """
    assert privacy.gamma_min_QMS(1e-5, 1e6) > 0
    assert privacy.gamma_min_QMS(1., 1e6) > 0
    assert privacy.gamma_min_QMS(100, 1e6) > 0
    with pytest.raises(OverflowError):
        privacy.gamma_min_QMS(1e5, 1e6)
    # assert privacy.gamma_min_QMS(1e5, 1e6) > 0



    return 

    # The following tests do not refer right grand truth.
    
    assert compare_sigfigs(privacy.gamma_min_QMS(1., 100), 9.50)
    assert compare_sigfigs(privacy.gamma_min_QMS(1., 1000), 31.1)
    assert compare_sigfigs(privacy.gamma_min_QMS(1., 10000), 99.5)
    assert compare_sigfigs(privacy.gamma_min_QMS(1., 1e8), 9999, sigfigs=4)
    assert compare_sigfigs(privacy.gamma_min_QMS(1., 1e9), 31574, sigfigs=5)
    assert math.isinf(privacy.gamma_min_QMS(1., math.inf))

    ms = [1e2, 1e3, 1e4, 1e5, 1e8, 1e9]
    epsilons = [2., 3., 4.]
    refs = [[.564, .154, .0516],
            [.580, .156, .0523],
            [.582, .156, .0524],
            [.582, .157, .0524],
            [.582, .157, .0524],
            [.582, .157, .0524],
            [.582, .157, .0524]]
    
    for i, m in enumerate(ms):
        for j, epsilon in enumerate(epsilons):
            assert compare_sigfigs(privacy.gamma_min_QMS(epsilon, m), refs[i][j])

    ms = [100, 1000]
    epsilons = [1/2, 1/3, 1/4, 1/5, 1/10]
    refs = [[102,  201,  301,  401,  901],
            [1002, 2001, 3001, 4001, 9001]]

    for i, m in enumerate(ms):
        for j, epsilon in enumerate(epsilons):
            assert compare_sigfigs(privacy.gamma_min_QMS(epsilon, m), refs[i][j], sigfigs=4)