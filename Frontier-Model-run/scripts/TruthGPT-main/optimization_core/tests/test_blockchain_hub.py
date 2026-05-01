"""
Blockchain Hub Verification Suite
=================================

Validates the connectivity and balance checking logic of the Blockchain Hub.
"""

import sys
from pathlib import Path

# Add current dir to path
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from agents.blockchain.hub import hub
from agents.blockchain.provider import WEB3_AVAILABLE

def test_hub_functionality():
    print("Starting Blockchain Hub Verification...")
    print(f"Web3 Library Available: {WEB3_AVAILABLE}")
    
    # 1. Test Gas Status
    print("\n[1] Checking Gas Status...")
    gas = hub.get_gas_status()
    print(f"Status: {gas['status']}")
    if gas['status'] == "Connected":
        print(f"Gas Price: {gas['gas_price_gwei']} Gwei")
    
    # 2. Test ETH Balance (Vitalik's address as sample)
    vitalik = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    print(f"\n[2] Checking ETH Balance for {vitalik}...")
    balance = hub.check_eth_balance(vitalik)
    print(f"Balance: {balance['balance']} {balance['symbol']}")
    
    # 3. Test Token Balance (USDT on Vitalik's)
    print(f"\n[3] Checking USDT Balance for {vitalik}...")
    token_balance = hub.check_token_balance(vitalik, "USDT")
    print(f"USDT Balance: {token_balance.get('balance', 'N/A')}")
    
    # 4. Test Audit Simulation
    print(f"\n[4] Simulating Contract Audit...")
    audit = hub.audit_smart_contract("0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D") # Uniswap V2
    print(f"Safety Score: {audit['safety_score']}")
    print(f"Findings: {len(audit['findings'])}")

if __name__ == "__main__":
    test_hub_functionality()
