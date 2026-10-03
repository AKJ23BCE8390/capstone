import pytest
from server.straggler import ClientPerformanceTracker

def test_straggler_detection():
    tracker = ClientPerformanceTracker(high_performance_threshold=5.0)
    
    # Simulate normal compute nodes
    assert tracker.log_latency("hospital_alpha", 2.3) is True
    
    # Simulate extreme bottleneck compute failure
    assert tracker.log_latency("hospital_beta", 12.8) is False
    assert "hospital_beta" in tracker.performance_logs
