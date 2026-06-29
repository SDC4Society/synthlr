import math

import pytest

from synthlr import phi
from .util import compare_sigfigs

def test_phi_QMS():
    """
    any phi is always in [1, m]
    """
    assert 1 <= phi.phi_QMS(10, 1e6) <= 10
    assert 1 <= phi.phi_QMS(1e6, 10) <= 1e6
    assert 1 <= phi.phi_QMS(1e6, 1e6) <= 1e6

    """
    compare with Table 3 in the discussion paper
    """
    gammas = [math.sqrt(10), 10, math.sqrt(1000), 100, 1000]
    Js = [1e2, 1e3, 1e4, 1e5]
    refs = [[15.7,  .731,   .0642,   .00633],
            [2.98,  .210,   .0201,   .00200],
            [.731,  .0642,  .00633,  .000632],
            [.210,  .0201,  .00200,  .000200],
            [.0201, .00200, .000200, .0000200]]
    
    for i, gamma in enumerate(gammas):
        for j, J in enumerate(Js):
            assert compare_sigfigs(phi.phi_QMS(1000, J*gamma)-1, refs[i][j])
