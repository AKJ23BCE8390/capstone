import pytest
import time
from server.straggler import StragglerEnforcer

def test_straggler_guard_filtering():
    enforcer = StragglerEnforcer(timeout_seconds=0.1, min_required_clients=2)
    mock_responses = [("client_1", "res1"), ("client_2", "res2"), ("client_3", "res3")]
    
    # Test valid timely transmission window pass verification
    clean_run = enforcer.intercept_fit_responses(start_time=time.time(), responses=mock_responses)
    assert len(clean_run) == 3

    # Force artificial process latency triggers to evaluate error mitigation loops
    time.sleep(0.15)
    truncated_run = enforcer.intercept_fit_responses(start_time=time.time() - 0.2, responses=mock_responses)
    assert len(truncated_run) == 2  # Truncated down to min_required_clients
