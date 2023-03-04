require("@nomiclabs/hardhat-waffle");
require("@nomiclabs/hardhat-ethers");
require("@nomiclabs/hardhat-etherscan");

module.exports = {
  solidity: "0.8.0",
  networks: {
    hardhat: {},
  },
  etherscan: {
    apiKey: "YOUR_ETHERSCAN_API_KEY",
  },
};

