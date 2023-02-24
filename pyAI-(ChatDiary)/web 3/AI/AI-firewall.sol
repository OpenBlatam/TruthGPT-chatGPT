pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";

contract AIDAO is Ownable {
    uint256 private sensitiveData;

    function setSensitiveData(uint256 _data) external onlyOwner {
        // Only the contract owner can set sensitive data
        sensitiveData = _data;
    }

    function getSensitiveData() external view returns (uint256) {
        // Only allow authorized users to access sensitive data
        require(msg.sender == owner() || msg.sender == address(this), "Unauthorized access");
        return sensitiveData;
    }
}
