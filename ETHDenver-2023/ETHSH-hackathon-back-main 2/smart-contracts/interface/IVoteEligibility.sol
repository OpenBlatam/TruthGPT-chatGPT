pragma solidity ^0.8.14;

interface IVoteEligibility {
    function requestData(string memory link) external;

    function settleRequest(string memory link) external;

    function getSettledData(string memory link) external view returns (int256);

    function updatePropState(string memory link) external;

    function getPropInfos(string memory link) external view;
}
