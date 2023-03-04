// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";

interface IAI {
    function makeDecision(bytes memory data) external returns (bool);
}

interface IAPI {
    function submitProposal(string memory proposal) external;
    function castVote(uint256 proposalId, uint256 votes) external;
}

contract DAO is Ownable {
    IAI public ai;
    IAPI public api;

    uint256 public constant VOTING_POWER = 1;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;

    struct Proposal {
        string name;
        address creator;
        uint256 votes;
        bool executed;
    }

    Proposal[] public proposals;

    event ProposalAdded(uint256 proposalId, string name, address creator);
    event ProposalExecuted(uint256 proposalId);

    constructor(address _aiAddress, address _apiAddress) {
        ai = IAI(_aiAddress);
        api = IAPI(_apiAddress);
    }

    function submitProposal(string memory _name) public {
        require(ai.makeDecision(bytes(_name)), "AI rejected proposal");
        proposals.push(Proposal(_name, msg.sender, 0, false));
        emit ProposalAdded(proposals.length - 1, _name, msg.sender);
    }

    function castVote(uint256 _proposalId, uint256 _votes) public {
        Proposal storage proposal = proposals[_proposalId];
        require(!proposal.executed, "Proposal already executed");
        balanceOf[msg.sender] -= _votes;
        proposal.votes += _votes;
        balanceOf[address(this)] += _votes;
        if (proposal.votes >= totalSupply / 2) {
            proposal.executed = true;
            emit ProposalExecuted(_proposalId);
        }
        api.castVote(_proposalId,uint256(_votes));
        }
        function transfer(address _recipient, uint256 _amount) public {
            require(balanceOf[msg.sender] >= _amount, "Insufficient balance");
            balanceOf[msg.sender] -= _amount;
            balanceOf[_recipient] += _amount;
        }

        function addSupply(address _recipient, uint256 _amount) public onlyOwner {
            balanceOf[_recipient] += _amount;
            totalSupply += _amount;
        }

        function removeSupply(address _recipient, uint256 _amount) public onlyOwner {
            require(balanceOf[_recipient] >= _amount, "Insufficient balance");
            balanceOf[_recipient] -= _amount;
            totalSupply -= _amount;
        }
}
