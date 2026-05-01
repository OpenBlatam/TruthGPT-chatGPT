"""
Blockchain Provider Layer - System 5.9
=======================================

Handles RPC connectivity, network management, and low-level Web3 orchestration.
"""

import os
import logging
from typing import Optional, Dict, Any

try:
    from web3 import Web3
    try:
        from web3.middleware import ExtraDataToPOAMiddleware as poa_middleware
    except ImportError:
        try:
            from web3.middleware import geth_poa_middleware as poa_middleware
        except ImportError:
            poa_middleware = None
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    Web3 = None
    poa_middleware = None

logger = logging.getLogger("BlockchainProvider")

class BlockchainProvider:
    """
    Manages connections to various blockchain networks.
    Supports Ethereum Mainnet, Sepolia, and other EVM-compatible chains.
    """
    
    def __init__(self, rpc_url: Optional[str] = None):
        self.rpc_url = rpc_url or os.getenv("ETH_RPC_URL", "https://eth.llamarpc.com")
        self.w3 = None
        self.connected = False
        self._initialize_connection()

    def _initialize_connection(self):
        """Initialize Web3 connection if library is available."""
        if not WEB3_AVAILABLE:
            logger.warning("Web3 library not found. Running in MOCK mode.")
            return

        try:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            # Support for Geth/PoA chains (like Sepolia or Polygon)
            if poa_middleware:
                self.w3.middleware_onion.inject(poa_middleware, layer=0)
            
            if self.w3.is_connected():
                self.connected = True
                logger.info(f"Connected to blockchain via {self.rpc_url}")
            else:
                logger.error(f"Failed to connect to {self.rpc_url}")
        except Exception as e:
            logger.error(f"Connection error: {e}")

    def get_web3(self):
        """Get the Web3 instance."""
        if not self.connected and WEB3_AVAILABLE:
            self._initialize_connection()
        return self.w3

    def is_healthy(self) -> bool:
        """Check if the connection is active and healthy."""
        if not WEB3_AVAILABLE:
            return False
        try:
            return self.w3.is_connected()
        except:
            return False

    def get_network_info(self) -> Dict[str, Any]:
        """Retrieve basic network information."""
        if not self.connected or not WEB3_AVAILABLE:
            return {"status": "Disconnected", "mode": "Mock" if not WEB3_AVAILABLE else "Real"}
        
        try:
            chain_id = self.w3.eth.chain_id
            block_number = self.w3.eth.block_number
            gas_price = self.w3.eth.gas_price
            
            return {
                "status": "Connected",
                "chain_id": chain_id,
                "block_number": block_number,
                "gas_price_gwei": Web3.from_wei(gas_price, 'gwei'),
                "mode": "Real"
            }
        except Exception as e:
            return {"status": "Error", "error": str(e)}

# Global Provider Instance
provider = BlockchainProvider()
