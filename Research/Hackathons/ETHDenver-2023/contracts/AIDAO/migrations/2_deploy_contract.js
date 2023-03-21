const AIDAO = artifacts.require("AIDAO");

module.exports = function (deployer) {
    deployer.deploy(AIDAO, <daoAddress>, <tokenAddress>, <priceFeedAddress>, <tellorAddress>, <voteThreshold>, <defenderAddress>, <ceramicUrl>);
        };
