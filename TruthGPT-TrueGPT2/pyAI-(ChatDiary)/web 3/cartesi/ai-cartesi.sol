pragma solidity ^0.8.0;

import "@cartesi/descartes-sdk/contracts/DescartesInterface.sol";

contract MyAIApplication {
    DescartesInterface private descartes;
    bytes private offchainResult;

    constructor(DescartesInterface _descartes) {
        descartes = _descartes;
    }

    function predict(string calldata input) public {
        bytes memory inputBytes = bytes(input);
        bytes memory taskData = abi.encodeWithSignature("predict(bytes)", inputBytes);
        descartes.createTask(taskData);
    }

    function getPredictionResult() public returns (string memory) {
        bytes memory result = descartes.getTaskResult();
        offchainResult = result;
        return abi.decode(result, (string));
    }
}
