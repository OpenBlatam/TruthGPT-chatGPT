// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract PolicyCollateral {

    address payable public policyOwner;
    uint256 public collateralAmount;
    uint256 public policyPayoutAmount;
    uint256 public policyExpirationDate;
    bool public policyActive;
    address public policyIssuer;

    event PolicyActivated(uint256 expirationDate);
    event CollateralDeposited(uint256 amount);
    event CollateralWithdrawn(uint256 amount);
    event PolicyPaidOut(uint256 amount);

    constructor(
        uint256 _policyPayoutAmount,
        uint256 _policyExpirationDate,
        address _policyIssuer
    ) {
        policyOwner = payable(msg.sender);
        collateralAmount = 0;
        policyPayoutAmount = _policyPayoutAmount;
        policyExpirationDate = _policyExpirationDate;
        policyIssuer = _policyIssuer;
        policyActive = false;
    }

    function activatePolicy() public {
        require(!policyActive, "Policy is already active");
        require(msg.sender == policyIssuer, "Only policy issuer can activate the policy");
        policyActive = true;
        emit PolicyActivated(policyExpirationDate);
    }

    function depositCollateral() public payable {
        require(policyActive, "Policy is not active");
        require(msg.value > 0, "Collateral amount must be greater than 0");
        collateralAmount += msg.value;
        emit CollateralDeposited(msg.value);
    }

    function withdrawCollateral(uint256 amount) public {
        require(policyActive, "Policy is not active");
        require(msg.sender == policyOwner, "Only policy owner can withdraw collateral");
        require(amount <= collateralAmount, "Insufficient collateral balance");
        collateralAmount -= amount;
        payable(policyOwner).transfer(amount);
        emit CollateralWithdrawn(amount);
    }

    function payoutPolicy() public {
        require(policyActive, "Policy is not active");
        require(block.timestamp < policyExpirationDate, "Policy has expired");
        require(msg.sender == policyIssuer, "Only policy issuer can payout the policy");
        payable(policyOwner).transfer(policyPayoutAmount);
        emit PolicyPaidOut(policyPayoutAmount);
    }

    function getPolicyDetails() public view returns (
        address payable,
        uint256,
        uint256,
        uint256,
        bool,
        address
    ) {
        return (policyOwner, collateralAmount, policyPayoutAmount, policyExpirationDate, policyActive, policyIssuer);
    }
}
