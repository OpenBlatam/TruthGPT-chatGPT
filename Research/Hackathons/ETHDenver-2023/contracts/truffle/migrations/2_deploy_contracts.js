const DecisionMaker = artifacts.require("DecisionMaker");

module.exports = function (deployer) {
    const aiAddress = "0x123456789..."; // Replace with your AI module address
    deployer.deploy(DecisionMaker, aiAddress);
};

module.exports = {
    networks: {
        development: {
            host: "127.0.0.1:8545",
            port: 7545,
            network_id: "*"
        }
    }
};
