import time

class ClientPerformanceTracker:
    def __init__(self, high_performance_threshold: float = 30.0):
        """
        Monitors node compute rates to identify struggling infrastructure models.
        """
        self.threshold = high_performance_threshold
        self.performance_logs = {}

    def log_latency(self, client_id: str, round_duration: float) -> bool:
        """
        Logs client durations. Returns False if a node breaches runtime constraints.
        """
        self.performance_logs[client_id] = round_duration
        if round_duration > self.threshold:
            print(f"[Straggler Alert] Node {client_id} breached timeline bounds: {round_duration}s")
            return False
        return True
