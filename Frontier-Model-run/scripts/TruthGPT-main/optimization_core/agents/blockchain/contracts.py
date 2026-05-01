"""
Smart Contract Registry - System 5.9
====================================

Manages ABIs and addresses for common smart contracts.
"""

from typing import Dict, Any

# Common ERC20 ABI (Minimal)
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    }
]

# Popular Contract Addresses (Mainnet)
CONTRACT_REGISTRY = {
    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "LINK": "0x514910771AF9Ca656af840dff83E8264EcF986CA"
}

def get_erc20_abi():
    """Returns the standard ERC20 ABI."""
    return ERC20_ABI

def get_contract_address(symbol: str) -> str:
    """Returns the address for a common symbol."""
    return CONTRACT_REGISTRY.get(symbol.upper(), "")

def get_contract_info(symbol: str) -> Dict[str, Any]:
    """Returns a dictionary with contract details."""
    address = get_contract_address(symbol)
    if not address:
        return {}
    return {
        "symbol": symbol.upper(),
        "address": address,
        "abi": ERC20_ABI
    }
