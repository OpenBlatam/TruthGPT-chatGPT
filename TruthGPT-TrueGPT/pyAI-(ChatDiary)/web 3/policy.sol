// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TwitterAPI {
    uint256 constant private COST_PER_REQUEST = 100; // cost per request in wei
    address private owner; // owner of the contract
    uint256 private totalRequests; // total number of requests made
    uint256 private totalCost; // total cost of all requests made

    mapping(address => uint256) private requestCounts; // number of requests made by each user

    constructor() {
        owner = msg.sender;
    }

    // This function can be used to make a request to the Twitter API
    function makeRequest() public payable {
        // Calculate the cost of the request
        uint256 cost = COST_PER_REQUEST * 1 wei;

        // Check that the caller has paid the required cost
        require(msg.value >= cost, "Insufficient payment for request");

        // Update the contract state
        totalRequests += 1;
        totalCost += cost;
        requestCounts[msg.sender] += 1;

        // Transfer the payment to the owner of the contract
        (bool success, ) = owner.call{value: cost}("");
        require(success, "Payment transfer failed");
    }

    // This function can be used to get the total number of requests made
    function getTotalRequests() public view returns (uint256) {
        return totalRequests;
    }

    // This function can be used to get the total cost of all requests made
    function getTotalCost() public view returns (uint256) {
        return totalCost;
    }

    // This function can be used to get the number of requests made by a specific user
    function getRequestCount(address user) public view returns (uint256) {
        return requestCounts[user];
    }

    // This function can be used by the owner of the contract to withdraw the accumulated funds
    function withdraw() public {
        require(msg.sender == owner, "Only the owner can withdraw funds");
        (bool success, ) = msg.sender.call{value: address(this).balance}("");
        require(success, "Withdrawal failed");
    }
}
