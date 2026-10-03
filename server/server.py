import os
import argparse
import flwr as fl
from strategy import CustomTelemetryStrategy

def main():
    parser = argparse.ArgumentParser(description="Hardened Federated Aggregator Orchestrator Node")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding address")
    parser.add_argument("--port", type=int, default=8080, help="Orchestration port")
    parser.add_argument("--rounds", type=int, default=5, help="Total execution epochs")
    parser.add_argument("--dashboard", type=str, default="http://127.0.0", help="Target UI endpoint")
    args = parser.parse_args()

    print("=================================================================")
    print("      LAUNCHING PRIVACY-PRESERVING FL AGGREGATION HUB            ")
    print("=================================================================")
    print(f" Binding Address    : {args.host}:{args.port}")
    print(f" Evaluation Rounds  : {args.rounds}")
    print(f" Telemetry Destination: {args.dashboard}")
    print("=================================================================")

    # Instantiate strategy passing temporal rules and UI parameters
    strategy = CustomTelemetryStrategy(
        dashboard_url=args.dashboard,
        fraction_fit=1.0,           # Train on all available checked clients
        min_fit_clients=2,          # Minimum concurrent active compute instances required
        min_available_clients=2,    # Threshold block to wait for before booting rounds
    )

    # Launch framework listening loops
    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
