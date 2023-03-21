pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";
import "@openzeppelin/contracts/utils/math/Math.sol";
import "./OpenAI.sol";
import "./IVoting.sol";

contract AIDAO is Ownable {
    using SafeMath for uint256;

    OpenAI private ai;
    IVoting private dao;
    IERC20 private token;

    uint256 public voteThreshold;

    constructor(address _aiAddress, address _daoAddress, address _tokenAddress, uint256 _voteThreshold) {
        ai = OpenAI(_aiAddress);
        dao = IVoting(_daoAddress);
        token = IERC20(_tokenAddress);
        voteThreshold = _voteThreshold;
    }

    function voteOnProposal(uint256 _proposalId, bytes memory _inputData) external onlyOwner {
        require(dao.isActive(address(this)), "Contract is not a member of the DAO");
        require(dao.getVoteTokenBalance(address(this)) >= voteThreshold, "Contract does not have enough voting power");
        uint256 vote = ai.predictVote(_inputData);
        dao.vote(_proposalId, vote);
    }
}
