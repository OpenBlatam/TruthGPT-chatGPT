async function main() {
    const [deployer] = await ethers.getSigners();

    console.log("Deploying contract with the account:", deployer.address);

    const DecisionMaker = await ethers.getContractFactory("DecisionMaker");
    const decisionMaker = await DecisionMaker.deploy("<your AI address>");

    console.log("DecisionMaker contract deployed to address:", decisionMaker.address);
}

main()
    .then(() => process.exit(0))
    .catch(error => {
        console.error(error);
        process.exit(1);
    });
