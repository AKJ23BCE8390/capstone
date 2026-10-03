import flwr as fl
import requests
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import Metrics, FitRes, Parameters, Scalar, EvaluateRes

class CustomTelemetryStrategy(fl.server.strategy.FedAvg):
    def __init__(self, dashboard_url: str, *args, **kwargs):
        """
        Custom Federated Aggregation Strategy tracking performance
        and streaming metrics to a centralized dashboard.
        """
        super().__init__(*args, **kwargs)
        self.dashboard_url = dashboard_url

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        # Call the base FedAvg aggregation implementation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if not results:
            return aggregated_parameters, aggregated_metrics

        total_examples = 0
        running_accuracy = 0.0
        running_loss = 0.0
        max_epsilon = 0.0
        successful_clients = len(results)
        failed_clients = len(failures)

        # Safely compute weighted averages from reporting nodes
        for _, fit_res in results:
            num_examples = fit_res.num_examples
            total_examples += num_examples
            
            # Extract client metrics (handles cases where Person 1/3 keys are missing)
            running_accuracy += fit_res.metrics.get("accuracy", 0.0) * num_examples
            running_loss += fit_res.metrics.get("loss", 0.0) * num_examples
            max_epsilon = max(max_epsilon, fit_res.metrics.get("epsilon", 0.0))

        global_accuracy = running_accuracy / total_examples if total_examples > 0 else 0.0
        global_loss = running_loss / total_examples if total_examples > 0 else 0.0

        # Package payload for Person 4's Node/Express Telemetry API
        telemetry_data = {
            "round": int(server_round),
            "accuracy": float(round(global_accuracy, 4)),
            "loss": float(round(global_loss, 4)),
            "epsilon": float(round(max_epsilon, 2)),
            "active_nodes": int(successful_clients),
            "dropped_nodes": int(failed_clients)
        }

        # Non-blocking network API dispatch step
        try:
            print(f"[Network Engine] Sending Round {server_round} statistics to Dashboard...")
            response = requests.post(self.dashboard_url, json=telemetry_data, timeout=3)
            if response.status_code == 200:
                print("[Network Engine] Telemetry data committed successfully.")
            else:
                print(f"[Network Engine] Dashboard API warning: Status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[Network Engine] Telemetry Pipeline Offline (API Unreachable). Error: {e}")

        return aggregated_parameters, aggregated_metrics
