import React, { useState } from "react";
import { ethers } from "ethers";
import AIDecisionMaker from "../contracts/AIDecisionMaker.json";

const provider = new ethers.providers.Web3Provider(window.ethereum);
const signer = provider.getSigner();

const aiDecisionMakerAddress = "<insert contracts address here>";

const aiDecisionMakerContract = new ethers.Contract(
    aiDecisionMakerAddress,
    AIDecisionMaker.abi,
    signer
);

const Chatbot = () => {
    const [inputText, setInputText] = useState("");
    const [decision, setDecision] = useState(false);

    const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setInputText(event.target.value);
    };

    const handleDecision = async () => {
        // Convert input text to bytes
        const inputBytes = ethers.utils.toUtf8Bytes(inputText);

        // Call makeDecision function of AIDecisionMaker contracts
        const decision = await aiDecisionMakerContract.makeDecision(inputBytes);

        // Update state with decision output
        setDecision(decision);
    };

    return (
        <div>
            <input type="text" value={inputText} onChange={handleInputChange} />
    <button onClick={handleDecision}>Get Decision</button>
    {decision ? (
        <div>Decision: {decision.toString()}</div>
    ) : (
        <div>No decision yet</div>
    )}
    </div>
);
};

export default Chatbot;
