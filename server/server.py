import flwr as fl
import logging
from server.strategy import TelemetryFedAvg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CentralServerOrchestrator")

def run_server():
    logger.info("Configuring centralized global model network parameters...")
    
    strategy = TelemetryFedAvg(
        api_url="http://localhost:5000/api/metrics",
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3
    )

    logger.info("Opening network listener interface on port 8080. Awaiting nodes...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )

if __name__ == "__main__":
    run_server()
