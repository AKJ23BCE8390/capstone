import pytest
import numpy as np
from client.client import HospitalFlowerClient

def test_client_parameter_flow():
    client = HospitalFlowerClient(client_id=99)
    initial_params = client.get_parameters(config={})
    
    assert len(initial_params) == 2
    assert initial_params[0].shape == (3, 3)

    new_params = [np.ones((3, 3)), np.ones((3,))]
    client.set_parameters(new_params)
    updated_params = client.get_parameters(config={})
    
    assert np.array_equal(updated_params[0], np.ones((3, 3)))
