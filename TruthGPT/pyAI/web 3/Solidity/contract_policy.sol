pragma solidity ^0.8.0;

contract Policy {
    function policy_with_temperature(uint[] memory q_values, uint temperature) public view returns (uint) {
        // Compute the softmax of the Q-values with temperature scaling
        uint[] memory scaled_q_values = new uint[](q_values.length);
        uint total = 0;
        for (uint i = 0; i < q_values.length; i++) {
            scaled_q_values[i] = exp(q_values[i] / temperature);
            total += scaled_q_values[i];
        }

        // Normalize the scaled Q-values
        uint[] memory prob = new uint[](q_values.length);
        for (uint i = 0; i < q_values.length; i++) {
            prob[i] = scaled_q_values[i] / total;
        }

        // Generate a random number between 0 and 1
        uint rand_val = uint(keccak256(abi.encodePacked(block.timestamp, block.difficulty))) % 1000 / 1000;

        // Compute the cumulative distribution function (CDF) of the probabilities
        uint[] memory cdf = new uint[](q_values.length);
        uint sum = 0;
        for (uint i = 0; i < q_values.length; i++) {
            sum += prob[i];
            cdf[i] = sum;
        }

        // Find the first index where the CDF is greater than or equal to the random value
        uint action = 0;
        for (uint i = 0; i < q_values.length; i++) {
            if (cdf[i] >= rand_val) {
                action = i;
                break;
            }
        }

        return action;
    }
}
