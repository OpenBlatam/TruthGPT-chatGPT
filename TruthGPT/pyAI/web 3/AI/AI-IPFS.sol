pragma solidity ^0.8.0;

import "ipfs-cid/contracts/IPFSStorage.sol";

contract PolicyFunction {
    // Define the policy and reward criteria
    uint256 constant REWARD_AMOUNT = 1 ether;
    string constant SPAM_KEYWORD = "spam";

    // Define the function that implements the policy and returns the result, reward, and IPFS hash
    function detectSpamAndReward(string calldata message) public returns (bool, uint256, bytes32) {
        // Check if the message contains the spam keyword
        bool isSpam = containsKeyword(message, SPAM_KEYWORD);

        // Store the message on IPFS and get the hash
        bytes32 ipfsHash = IPFSStorage.store(message);

        // Assign the reward if the message is spam
        uint256 reward = isSpam ? REWARD_AMOUNT : 0;

        // Return the result, reward, and IPFS hash
        return (isSpam, reward, ipfsHash);
    }

    // Define a helper function to check if a string contains a keyword
    function containsKeyword(string calldata message, string calldata keyword) internal pure returns (bool) {
        bytes memory msgBytes = bytes(message);
        bytes memory keywordBytes = bytes(keyword);

        uint256 keywordLength = keywordBytes.length;

        if (msgBytes.length < keywordLength) {
            return false;
        }

        for (uint256 i = 0; i <= msgBytes.length - keywordLength; i++) {
            bool found = true;
            for (uint256 j = 0; j < keywordLength; j++) {
                if (msgBytes[i + j] != keywordBytes[j]) {
                    found = false;
                    break;
                }
            }

            if (found) {
                return true;
            }
        }

        return false;
    }
}
