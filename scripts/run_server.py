import sys
import os
import argparse

# Appends repository absolute directories onto system runtime profiles
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from server.server import run_server

def main():
    parser = argparse.ArgumentParser(description="Launches the Central Federated Aggregator node pipeline.")
    parser.add_argument("--port", type=int, default=8080, help="Target server execution listener network port.")
    args = parser.parse_args()
    
    print(f"[BOOT] Initializing network environment orchestration on port: {args.port}")
    run_server()

if __name__ == "__main__":
    main()
