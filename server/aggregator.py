import logging
from typing import Dict, Any

logger = logging.getLogger("ServerAuth")

class ConnectionAuthenticator:
    def __init__(self, expected_token: str = "HOSPITAL_SECURE_FL_2026"):
        """
        Authenticates incoming client registration handshakes based on token parameters.
        """
        self.expected_token = expected_token

    def validate_client(self, metadata: Dict[str, Any]) -> bool:
        """
        Verifies client authentication credentials.
        """
        client_token = metadata.get("auth_token")
        if client_token == self.expected_token:
            logger.info("Client handshake authentication verified successfully.")
            return True
        logger.warning("Unauthorized client connection attempt blocked.")
        return False
