const { ethers } = require("hardhat");
const hre = require("hardhat");
const fs = require("fs");

async function main() {
    // Get the accounts from the hardhat network
    const accounts = await ethers.getSigners();

    // Get the DAO, token, price feed, and Tellor addresses
    const daoAddress = "0x..."; // replace with the DAO address
    const tokenAddress = "0x..."; // replace with the token address
    const priceFeedAddress = "0x..."; // replace with the price feed address
    const tellorAddress = "0x..."; // replace with the Tellor address
    const voteThreshold = 1000; // set the vote threshold

    // Get the Defender API key and create a Defender client
    const apiKey = "YOUR_DEFENDER_API_KEY";
    const defender = await ethers.getContract("DefenderRelay", apiKey);

    // Create the AIDAO contract instance
    const AIDAO = await ethers.getContractFactory("AIDAO");
    const aidao = await AIDAO.deploy(
        daoAddress,
        tokenAddress,
        priceFeedAddress,
        tellorAddress,
        voteThreshold,
        defender.address
    );

    // Print the address of the AIDAO contract
    console.log("AIDAO deployed to:", aidao.address);
}

main()
    .then(() => process.exit(0))
    .catch(error => {
        console.error(error);
        process.exit(1);
    });
