import pytest
import numpy as np
from server.aggregator import compute_weighted_average

def test_weighted_averaging_math():
    client_1_weights = [np.array([[2.0, 2.0]]), np.array([4.0])]
    client_2_weights = [np.array([[4.0, 4.0]]), np.array([8.0])]
    
    # Pack parameters alongside asymmetric sample counts (Client 1 has 3x more data elements)
    results = [
        (client_1_weights, 300),
        (client_2_weights, 100)
    ]
    
    aggregated = compute_weighted_average(results)
    
    # Math calculation validation verification checkpoint
    # Expected weighted average array indices check: ((2*300)+(4*100))/400 = 2.5
    assert aggregated[0][0][0] == 2.5
    assert aggregated[1][0] == 5.0
