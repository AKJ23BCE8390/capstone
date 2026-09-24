import logging
from collections import OrderedDict
import flwr as fl
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HospitalClientNode")

class HospitalFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: int):
        self.client_id = client_id
        # Creates mock local array structures to isolate framework setup until Person 1 delivers code
        self.mock_weights = [np.random.randn(3, 3), np.zeros((3,))]
        logger.info(f"Hospital Client Node {self.client_id} successfully initialized.")

    def get_parameters(self, config) -> list:
        logger.info("Extracting framework array parameters snapshot.")
        return self.mock_weights

    def set_parameters(self, parameters: list):
        logger.info("Updating local configuration state parameters.")
        self.mock_weights = parameters

    def fit(self, parameters: list, config: dict) -> tuple:
        """
        Executes local processing scripts behind corporate security boundary firewalls.
        """
        logger.info(f"Client {self.client_id}: Fit request received. Running local calculations...")
        self.set_parameters(parameters)
        
        # Simulated metric logs matching training signatures
        simulated_loss = float(np.random.uniform(0.1, 0.5))
        simulated_acc = float(np.random.uniform(0.75, 0.95))
        
        # Returns updated layers, local training pool sample counts, and performance logs
        return self.get_parameters(config={}), 100, {"loss": simulated_loss, "accuracy": simulated_acc}

    def evaluate(self, parameters: list, config: dict) -> tuple:
        """
        Validates model configuration state criteria against unique demographic subsets.
        """
        logger.info(f"Client {self.client_id}: Validation evaluation triggered.")
        self.set_parameters(parameters)
        
        simulated_val_loss = float(np.random.uniform(0.15, 0.6))
        simulated_val_acc = float(np.random.uniform(0.7, 0.9))
        
        return simulated_val_loss, 20, {"accuracy": simulated_val_acc}
