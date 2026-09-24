import time
import requests
import logging
from typing import List, Tuple, Union, Dict, Optional
import flwr as fl
from flwr.common import Parameters, Scalar, FitRes, parameters_to_ndarrays, ndarrays_to_parameters

from server.straggler import StragglerEnforcer
from server.aggregator import compute_weighted_average

logger = logging.getLogger("TelemetryFedAvgStrategy")

class TelemetryFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, api_url: str = "http://localhost:5000/api/metrics", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_url = api_url
        self.straggler_guard = StragglerEnforcer(timeout_seconds=30.0, min_required_clients=3)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        start_time = time.time()
        # Enforce defensive timing validations against slow nodes
        filtered_results = self.straggler_guard.intercept_fit_responses(start_time, results)

        if not filtered_results:
            return None, {}

        # Deconstruct network parameters to pure NumPy arrays for processing
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in filtered_results
        ]

        # Calculate combined node weights via localized array equations
        aggregated_ndarrays = compute_weighted_average(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Compute summary metrics metrics to pass across telemetry pipelines
        total_acc, total_loss, count = 0.0, 0.0, 0
        for _, fit_res in filtered_results:
            if fit_res.metrics:
                total_acc += float(fit_res.metrics.get("accuracy", 0.0))
                total_loss += float(fit_res.metrics.get("loss", 0.0))
                count += 1

        avg_acc = (total_acc / count) if count > 0 else 0.0
        avg_loss = (total_loss / count) if count > 0 else 0.0

        # Package data transmission schema for Person 4's dashboard backend API
        telemetry_packet = {
            "round": server_round,
            "accuracy": avg_acc,
            "loss": avg_loss,
            "active_clients": len(filtered_results),
            "failures": len(failures)
        }

        try:
            requests.post(self.api_url, json=telemetry_packet, timeout=2.0)
            logger.info(f"Round {server_round} telemetry forwarded to dashboard server: {telemetry_packet}")
        except Exception as e:
            logger.error(f"Failed to transmit backend data updates: {e}")

        return parameters_aggregated, {"round_loss": avg_loss, "round_accuracy": avg_acc}
