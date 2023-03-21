// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openai/contracts/IEthPredictor.sol";

contract OpenAIDecision {
    IEthPredictor private predictor;

    constructor(address _predictorAddress) {
        predictor = IEthPredictor(_predictorAddress);
    }

    function makeDecision(uint256 _data) public view returns (bool) {
        // Call the OpenAI predictor to get the prediction
        uint256 prediction = predictor.predict(_data);

        // Make a decision based on the prediction
        if (prediction >= 0.5) {
            return true;
        } else {
            return false;
        }
    }
}
