const DecisionMaker = artifacts.require("DecisionMaker");

module.exports = function (deployer) {
    const aiAddress = "0x123456789..."; // Replace with your AI module address
    deployer.deploy(DecisionMaker, aiAddress);
};
