import sys
import os
import argparse
import flwr as fl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from client.client import HospitalFlowerClient

def main():
    parser = argparse.ArgumentParser(description="Registers a local hospital node onto target central aggregators.")
    parser.add_argument("--id", type=int, required=True, help="Unique numeric parameter identifying the specific node.")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Address parameters pointing to the server context.")
    args = parser.parse_args()

    print(f"[LAUNCH] Staging Flower framework lifecycle client context for Hospital Node ID: {args.id}")
    client = HospitalFlowerClient(client_id=args.id)
    
    fl.client.start_numpy_client(server_address=args.server, client=client)

if __name__ == "__main__":
    main()
