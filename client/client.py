import sys
import flwr as fl
import numpy as np

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, client_id: str):
        self.client_id = client_id
        # Define a basic 2-layer linear shape array to satisfy initial shape handshake checks
        self.weights = [np.zeros((10, 2), dtype=np.float32), np.zeros((2,), dtype=np.float32)]
        print(f"[Client Initialize] Node {self.client_id} successfully loaded into memory.")
        
    def get_parameters(self, config):
        return self.weights

    def set_parameters(self, parameters):
        self.weights = parameters

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print(f"\n[Client {self.client_id}] Successfully pulled global model configuration payload...")
        print(f"[Client {self.client_id}] Processing local training round configurations...")
        
        # Output mock metrics back up to your custom strategy pipeline
        return self.get_parameters(config={}), 100, {"accuracy": 0.88, "loss": 0.24, "epsilon": 1.5}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.24, 100, {"accuracy": 0.88}

if __name__ == "__main__":
    # Fallback to default if no argument is passed explicitly
    cid = sys.argv[1] if len(sys.argv) > 1 else "hospital_generic"
    
    print(f"Connecting to Orchestrator Hub as client node ID: {cid}...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", 
        client=HospitalClient(cid)
    )
