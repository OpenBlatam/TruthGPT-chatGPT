// SPDX-License-Identifier: AGPL-3.0-only
pragma solidity ^0.8.16;

import 'protocol/packages/core/contracts/oracle/interfaces/OptimisticOracleV2Interface.sol';
import './DAOCensus.sol';

contract umaInteract is DAOCensus {
    mapping(string => Deployment) propDetails;
    mapping(string => Proposal) lastDeploymentProposal;
    mapping(address => DAOdashboard) dashboards;

    // Create an Optimistic oracle
    OptimisticOracleV2Interface oo = OptimisticOracleV2Interface(0xA5B9d8a0B0Fa04Ba71BDD68069661ED5C0848884);
    //mumbai 0x60E6140330F8FE31e785190F39C1B5e5e833c2a9
    // goerli 0xA5B9d8a0B0Fa04Ba71BDD68069661ED5C0848884
    //optimism 0x255483434aba5a75dc60c1391bB162BCd9DE2882

    // Use the yes no idetifier to ask arbitary questions, such as the weather on a particular day.
    bytes32 identifier = bytes32('YES_OR_NO_QUERY');

    uint256 requestTime = 0; // Store the request time so we can re-use it later.

    function registerDAO(
        address DAOaddress,
        uint16 limit,
        uint256 payment
    ) public {
        dashboards[DAOaddress] = DAOdashboard(true, limit, payment);
        emit DaoRegistered(DAOaddress, limit, payment);
    }

    function submitDeployment(string memory hash, address DAOaddress) public {
        propDetails[hash] = Deployment(DAOaddress, hash, State.SUBMITED);
        emit DeploymentSubmitted(DAOaddress, hash);
    }

    function getProposalQuestion(string memory IPFSHash) internal pure returns (bytes memory) {
        return bytes(string.concat(IPFSHash, ' : 0 to accept this front end - else 1 '));
    }

    // Submit a data request to the Optimistic oracle.
    function requestData(string memory IPFSHash) public {
        bytes memory ancillaryData = getProposalQuestion(IPFSHash);
        requestTime = block.timestamp; // Set the request time to the current block time.
        IERC20 bondCurrency = IERC20(0xB4FBF271143F4FBf7B91A5ded31805e42b2208d6); // Use Görli WETH as the bond currency.

        //mumbai 0xA6FA4fB5f76172d178d61B04b0ecd319C5d1C0aa
        //goerli 0xB4FBF271143F4FBf7B91A5ded31805e42b2208d6
        //polygon 0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619
        //opt 0x4200000000000000000000000000000000000006

        uint256 reward = 0; // Set the reward to 0 (so we dont have to fund it from this contract).*// Now, make the price request to the Optimistic oracle and set the liveness to 30 so it will settle quickly.
        oo.requestPrice(identifier, requestTime, ancillaryData, bondCurrency, reward);
        oo.setCustomLiveness(identifier, requestTime, ancillaryData, 30);
        propDetails[IPFSHash].state = State.VOTE_ONGOING;
        lastDeploymentProposal[IPFSHash] = Proposal(requestTime, ancillaryData);
        emit ProposalUpdated(propDetails[IPFSHash].DAOaddress, IPFSHash, State.VOTE_ONGOING, ancillaryData);
        notification(
            'new request for hash', // this is notificaiton title
            string.concat(IPFSHash, ' you can vote in favour or again this front end deployment') // notification body
        );
    }

    // Settle the request once it's gone through the liveness period of 30 seconds. This acts the finalize the voted on price.
    // In a real world use of the Optimistic Oracle this should be longer to give time to disputers to catch bat price proposals.
    function settleRequest(string memory IPFSHash) public {
        Proposal storage prop = lastDeploymentProposal[IPFSHash];
        oo.settle(address(this), identifier, prop.time, prop.data);
        if (lastDeploymentProposal[IPFSHash].time == 0) return;
        int256 result = getSettledData(IPFSHash);
        State newState = State.REFUSED;
        if (result == 0) newState = State.ACCEPTED;
        string memory response =string.concat(IPFSHash,' has been', (newState == State.ACCEPTED ? 'accepted ' : 'refused'));
        notification('request settled', response);
        updatePropState(IPFSHash);
    }

    // Fetch the resolved price from the Optimistic Oracle that was settled.
       function getSettledData(string memory IPFSHash) public view returns (int256) {
        Proposal memory prop = lastDeploymentProposal[IPFSHash];
        return oo.getRequest(address(this), identifier, prop.time, prop.data).resolvedPrice;
    }

    function updatePropState(string memory IPFSHash) public {
        if (lastDeploymentProposal[IPFSHash].time == 0) return;
        int256 result = getSettledData(IPFSHash);
        State newState = State.REFUSED;
        if (result == 0) newState = State.ACCEPTED;
        propDetails[IPFSHash].state = newState;
        emit ProposalUpdated(
            propDetails[IPFSHash].DAOaddress,
            IPFSHash,
            newState,
            lastDeploymentProposal[IPFSHash].data
        );

        updatePropState(IPFSHash);
    }

    function submitHosting(string calldata name) public {}
}
