// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { CartesiMath } from "@cartesi/util/contracts/CartesiMath.sol";
import { RISC-V } from "@cartesi/machine/contracts/RISC-V.sol";
import { ProofOfStake } from "@metis.io/mrc20/contracts/token/Staking/ProofOfStake.sol";
import { ERC20 } from "@metis.io/mrc20/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "./IVoting.sol";

contract AIDAO is ProofOfStake, Ownable, RISC-V {
    IVoting private dao;
    ERC20 private token;

    uint256 public voteThreshold;

    constructor(address _daoAddress, address _tokenAddress, uint256 _voteThreshold) ProofOfStake(_tokenAddress) RISC-V() {
        dao = IVoting(_daoAddress);
        token = ERC20(_tokenAddress);
        voteThreshold = _voteThreshold;
    }

    function voteOnProposal(uint256 _proposalId, bytes calldata _inputData) external onlyOwner {
        require(dao.isActive(address(this)), "Contract is not a member of the DAO");
        require(balanceOf(address(this)) >= voteThreshold, "Contract does not have enough staked tokens");

        // Compute the input hash using CartesiMath
        bytes32 inputHash = keccak256(_inputData);

        // Load the input hash into the RISC-V machine
        machine.initialize(0x2000);
        machine.storeData(inputHash, 0);
        machine.run();

        // Get the result from the RISC-V machine
        uint256 result = machine.loadWord(0);

        // Normalize the result to a vote value between 0 and 100
        uint256 vote = result.mod(101);

        dao.vote(_proposalId, vote);
    }
}
