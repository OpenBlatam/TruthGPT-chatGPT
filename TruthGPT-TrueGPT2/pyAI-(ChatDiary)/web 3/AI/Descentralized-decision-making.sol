// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "github.com/oraclize/ethereum-api/provableAPI.sol";

interface OpenAI {
    function predict(string memory data) external view returns (uint256);
}

contract DecentralizedDecisionMaking is usingProvable {
    OpenAI openAI;
    uint256 public prediction;

    constructor(address openAIAddress) {
        openAI = OpenAI(openAIAddress);
    }

    function makeDecision(string memory inputData) public payable {
        require(msg.value > 0, "Please provide some funds to make a decision.");
        provable_query("URL", "json(https://api.random.org/json-rpc/1/invoke).result.random.data.0");
        prediction = openAI.predict(inputData);
    }

    function __callback(bytes32, string memory result, bytes memory) public {
        require(msg.sender == provable_cbAddress(), "Callback can only be called by the provable callback address.");
        prediction = uint256(keccak256(abi.encodePacked(result))) % 100;
    }

    function withdraw() public {
        payable(msg.sender).transfer(address(this).balance);
    }
}
