import pytest
import numpy as np
from server.aggregator import SecureWeightAggregator

def test_secure_weighted_average():
    # Setup simple linear matrices representing 2 layers of fake model weight vectors
    client_1_weights = [np.array([1.0, 2.0]), np.array([[3.0], [4.0]])]
    client_2_weights = [np.array([2.0, 4.0]), np.array([[6.0], [8.0]])]
    
    # Pack parameters alongside sample data splits (Client 2 has triple the data sizing volume)
    payload = [
        (client_1_weights, 100),
        (client_2_weights, 300)
    ]
    
    aggregated = SecureWeightAggregator.weighted_average(payload)
    
    # Assert custom calculation targets: expected values match proportions
    expected_layer_1 = (1.0 * 0.25) + (2.0 * 0.75)  # 1.75
    assert np.allclose(aggregated[0], np.array([1.75, 3.5]))
    assert aggregated[1].shape == (2, 1)
