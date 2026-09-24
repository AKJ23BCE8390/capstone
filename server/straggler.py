import time
import logging
from typing import List, Tuple, Any, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StragglerGuard")

class StragglerEnforcer:
    def __init__(self, timeout_seconds: float = 30.0, min_required_clients: int = 3):
        """
        Manages straggler nodes by monitoring time limits and operational node thresholds.
        """
        self.timeout_seconds = timeout_seconds
        self.min_required_clients = min_required_clients

    def intercept_fit_responses(
        self, 
        start_time: float, 
        responses: List[Tuple[Any, Any]]
    ) -> List[Tuple[Any, Any]]:
        """
        Filters client packages according to elapsed time conditions.
        """
        duration = time.time() - start_time
        if duration > self.timeout_seconds:
            logger.warning(f"Round execution time limit breached! Duration: {duration:.2f}s")
            if len(responses) >= self.min_required_clients:
                logger.info(f"Sufficient clients available ({len(responses)}). Dropping late stragglers.")
                return responses[:self.min_required_clients]
            else:
                logger.error("Critical: Insufficient healthy nodes available to enforce drops safely.")
        return responses
