const axios = require('axios');
const { OpenAI } = require('@openai/api');

const web3 = new Web3('https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID');
const openai = new OpenAI(process.env.OPENAI_API_KEY);

web3.eth.getBlockNumber()
  .then(async blockNumber => {
    const block = await web3.eth.getBlock(blockNumber, true);
    const { hash, transactions } = block;

    const transactionHashes = transactions.map(transaction => transaction.hash);
    const input = `The latest block on Ethereum is ${hash} and it contains the following transactions: ${transactionHashes.join(', ')}`;

    const response = await openai.completions.create({
      engine: 'davinci',
      prompt: input,
      maxTokens: 1024,
