pragma solidity ^0.8.0;

import "@aragon/os/contracts/kernel/Kernel.sol";
import "./DecisionMaker.sol";

contract MyDAO {
    Kernel public kernel;

    constructor(Kernel _kernel) {
        kernel = _kernel;
    }

    function createDecisionMakerApp(address aiAddress) external returns (address) {
        // Create the new DecisionMaker app instance
        bytes memory initializeData = abi.encodeWithSelector(
            bytes4(keccak256("initialize(address)")),
            aiAddress
        );
        address decisionMakerAppProxy = kernel.createProxy(
            address(new DecisionMaker(aiAddress)),
            initializeData,
            true // Set upgradeable flag to true
        );

        // Grant permissions for the app to perform actions in the DAO
        bytes32 appManagerRole = keccak256("APP_MANAGER_ROLE");
        kernel.grantPermission(decisionMakerAppProxy, appManagerRole);

        // Return the address of the new app instance
        return decisionMakerAppProxy;
    }
}
