require('dotenv').config();
const HDWalletProvider = require('@truffle/hdwallet-provider');
const privateKey = process.env.PRIVATE_KEY;
const infuraProjectId = process.env.INFURA_PROJECT_ID;

module.exports = {
  networks: {
    goerli: {
      provider: () => new HDWalletProvider(privateKey, `https://goerli.infura.io/v3/${infuraProjectId}`),
      network_id: 5,
      gas: 8000000,
      gasPrice: 20000000000 // 20 gwei
    }
  },
};
