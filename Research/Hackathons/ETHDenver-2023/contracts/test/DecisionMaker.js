const { expect } = require("chai");

describe("DecisionMaker", function() {
    let DecisionMaker;
    let decisionMaker;

    beforeEach(async function () {
        DecisionMaker = await ethers.getContractFactory("DecisionMaker");
        decisionMaker = await DecisionMaker.deploy("0x0000000000000000000000000000000000000000");
        await decisionMaker.deployed();
    });

    it("should set the correct owner and AI address", async function() {
        expect(await decisionMaker.decision.owner()).to.equal(await ethers.provider.getSigner(0).getAddress());
        expect(await decisionMaker.decision.aiAddress()).to.equal("0x0000000000000000000000000000000000000000");
    });

    it("should update the AI address", async function() {
        await decisionMaker.updateAIAddress("0x1111111111111111111111111111111111111111");
        expect(await decisionMaker.decision.aiAddress()).to.equal("0x1111111111111111111111111111111111111111");
    });

    it("should make a decision", async function() {
        await decisionMaker.makeDecision("0x1234567890");
        expect(await decisionMaker.decision.result()).to.equal(true);
        expect(await decisionMaker.decision.executed()).to.equal(true);
        expect(await decisionMaker.decision.timestamp()).to.be.above(0);
    });

    it("should revert if the decision has already been executed", async function() {
        await decisionMaker.makeDecision("0x1234567890");
        await expect(decisionMaker.makeDecision("0x1234567890")).to.be.revertedWith("Decision has already been executed.");
    });

    it("should revert if the caller is not the owner", async function() {
        const [_, signer1] = await ethers.getSigners();
        await expect(decisionMaker.connect(signer1).makeDecision("0x1234567890")).to.be.revertedWith("Only the owner can execute a decision.");
        await expect(decisionMaker.connect(signer1).updateAIAddress("0x1111111111111111111111111111111111111111")).to.be.revertedWith("Only the owner can update the AI address.");
    });
});
