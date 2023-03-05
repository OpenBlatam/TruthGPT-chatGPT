// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import { ByteHasher } from './helpers/ByteHasher.sol';
import { IWorldID } from './interfaces/IWorldID.sol';

import "usingtellor/contracts/UsingTellor.sol";

interface IMidpoint {
    function callMidpoint(uint64 midpointId, bytes calldata _data) external returns(uint256 requestId);
}

contract Contract is UsingTellor {
    using ByteHasher for bytes;

    uint256 public balance;

    uint256 public count;

    mapping(address => _Nprofit) public nps;

    struct _Nprofit {
        string name;
        string description;
        uint256 goal;
    }

    ///////////////////////////////////////////////////////////////////////////////
    ///                                  ERRORS                                ///
    //////////////////////////////////////////////////////////////////////////////

    /// @notice Thrown when attempting to reuse a nullifier
    error InvalidNullifier();

    /// @dev The WorldID instance that will be used for verifying proofs
    IWorldID internal immutable worldId;

    /// @dev The application's action ID
    uint256 internal immutable actionId;

    /// @dev The WorldID group ID (1)
    uint256 internal immutable groupId = 1;

    /// @dev Whether a nullifier hash has been used already. Used to prevent double-signaling
    mapping(uint256 => bool) internal nullifierHashes;

    event RequestMade(uint256 requestId, string tweet);

    address constant startpointAddress = 0x47a4905D4C2Eabd58abBDFEcBaeB07F1A29b660c;


    // Midpoint ID
    uint64 constant midpointID = 458;

    /// @param _worldId The WorldID instance that will verify the proofs
    /// @param _actionId The action ID for your application
    constructor(IWorldID _worldId, string memory _actionId, address payable _tellorAddress) UsingTellor(_tellorAddress) {
        worldId = _worldId;
        actionId = abi.encodePacked(_actionId).hashToField();
    }

    /// @param root The of the Merkle tree, returned by the SDK.
    /// @param nullifierHash The nullifier for this proof, preventing double signaling, returned by the SDK.
    /// @param proof The zero knowledge proof that demostrates the claimer is registered with World ID, returned by the SDK.
    /// @dev Feel free to rename this method however you want! We've used `claim`, `verify` or `execute` in the past.
    function verifyAndExecute(
        address input,
        uint256 root,
        uint256 nullifierHash,
        uint256[8] calldata proof
    ) public {
        count ++;

        // first, we make sure this person hasn't done this before
        if (nullifierHashes[nullifierHash]) revert InvalidNullifier();

        // then, we verify they're registered with WorldID, and the input they've provided is correct
        worldId.verifyProof(
            root,
            groupId,
            abi.encodePacked(input).hashToField(),
            nullifierHash,
            actionId,
            proof
        );

        // finally, we record they've done this, so they can't do it again (proof of uniqueness)
        nullifierHashes[nullifierHash] = true;

        // your logic here, make sure to emit some kind of event afterwards!

        count ++;

    }

    /*
        Bet possible outcomes
    */
    enum BetResult {
        NONE,
        WIN,
        TIE,
        LOSE
    }

    /*
        Very basic Bet identifying info and results
    */
    struct Bet {
        address owner;
        uint256 resultAvailabiityTimestamp;
        BetResult result;

        mapping(BetResult => address[]) betAddresses;
        mapping(BetResult => uint256[]) betAmounts;
        mapping(BetResult => uint256) betCount;
    }

    event BetPlaced(address _address, uint256 _amount, BetResult _result);
    event Query(bytes queryData, bytes32 queryId);

    event BetSettled(address _address, uint256 _amount, BetResult _result);

    event LogInt(string str, uint256 value);

    /*
        Internal state of Bets
    */
    mapping(uint256 => Bet) public bets;

    function createBet(uint256 betId, uint256 resultAvailabiityTimestamp) internal {
        bets[betId].owner = msg.sender;
        bets[betId].resultAvailabiityTimestamp = resultAvailabiityTimestamp;

        // bytes memory _b = abi.encode(
        //     "TellorKpr",
        //     abi.encode(
        //         address(this),
        //     )
        // );
        // bytes32 _betQueryId = keccak256(_b);
    }

    function getBetResult(uint256 betId) public returns (BetResult) {
        // Be efficient af
        if (bets[betId].result != BetResult.NONE) {
            return bets[betId].result;
        }

        return bets[betId].result = BetResult.WIN;

        // find queryData and queryId
        bytes memory _betQueryData = abi.encode("BetResult", abi.encode(betId));
        bytes32 _betQueryId = keccak256(_betQueryData);

        // run the query
        uint256 _timestamp;
        bytes memory _value;
        emit Query(_betQueryData, _betQueryId);
        (_value, _timestamp) = getDataBefore(
            _betQueryId,
            bets[betId].resultAvailabiityTimestamp - 1 hours
        );

        // emit result
        bets[betId].result = abi.decode(_value, (BetResult));
        return bets[betId].result;
    }

    function callMidpoint(string memory tweet) public {

        // Argument String
        bytes memory args = abi.encodePacked(tweet, bytes1(0x00));

        // Call Your Midpoint
        uint256 requestId = IMidpoint(startpointAddress).callMidpoint(midpointID, args);

        // For Demonstration Purposes Only
        emit RequestMade(requestId, tweet);
    }

    function bet(uint256 betId, BetResult betResult) public payable {
        // can't bet for a tie, can't bet on settled bets
        require(betResult != BetResult.TIE && betResult != BetResult.NONE);
        require(bets[betId].result == BetResult.NONE);

        // store bet
        uint256 _betCount = bets[betId].betCount[betResult];
        emit LogInt("_betCount", _betCount);

        bets[betId].betAddresses[betResult].push(msg.sender);
        bets[betId].betAmounts[betResult].push(msg.value);
        bets[betId].betCount[betResult] += 1;

        callMidpoint("New Donation for the street dogs!");

        emit LogInt("betAmounts", msg.value);
        emit LogInt("betResults", bets[betId].betCount[betResult]);

        emit BetPlaced(msg.sender, msg.value, betResult);
    }

    function settleBet(uint256 betId) public {
        // can't settle no-results bets
        BetResult _winResult = getBetResult(betId);
        BetResult _loseResult = _winResult == BetResult.WIN ? BetResult.LOSE : BetResult.WIN;
        require(_winResult != BetResult.NONE);
        require(_winResult != _loseResult);

        // emit win earnings
        uint256 i = 0;
        uint256 _paid = 0;
        for (i = 0; i < bets[betId].betCount[_winResult]; i++) {
            address payable _address = payable(bets[betId].betAddresses[_winResult][i]);
            uint256 _amount = bets[betId].betAmounts[_winResult][i];

            _address.transfer(_amount);
            _paid += _amount;

            emit BetSettled(_address, _amount, _winResult);
        }

        // todo: emit loser donations
    }

    function addNP(
        string memory _name,
        string memory _description,
        uint256 _goal
    ) public {
        require(nps[msg.sender].goal  == 0);

        nps[msg.sender] = _Nprofit(
            _name,
            _description,
            _goal
        );

        createBet(1, block.timestamp + 1 hours);
    }

    uint addressRegistryCount;



}
