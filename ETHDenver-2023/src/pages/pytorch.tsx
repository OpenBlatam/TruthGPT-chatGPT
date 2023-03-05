import React, { useState } from "react";
import { ethers } from "ethers";
//import AIDecisionMaker from "./contracts/AIDecisionMaker.json";
//import * as torch from "pytorchjs/dist/torch";

//const provider = new ethers.providers.Web3Provider(window.ethereum);
const signer = provider.getSigner();

const aiDecisionMakerAddress = "<insert contracts address here>";

const aiDecisionMakerContract = new ethers.Contract(
    aiDecisionMakerAddress,
    AIDecisionMaker.abi,
    signer
);

const modelPath = "<insert path to saved model here>";

const Chatbotpytorch = () => {
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

        if (decision) {
            // Load the ML model
            const model = await torch.load(modelPath);

            // Preprocess the input data
            const inputData = preprocessInputData(inputText);

            // Make a decision based on the model's output
            const mlDecision = makeDecision(model, inputData);

            // Execute the decision based on the ML model's output
            executeDecision(mlDecision);
        }
    };

    const preprocessInputData = (inputData) => {
        // Preprocess the input data as needed by the ML model
        // ...
        return preprocessedData;
    };

    const makeDecision = (model, inputData) => {
        const inputTensor = torch.tensor(inputData);
        const outputTensor = model.forward(inputTensor);
        const decision = outputTensor.dataSync()[0] > 0.5;
        return decision;
    };

    const executeDecision = (decision) => {
        // Execute the decision based on the ML model's output
        // ...
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

export default Chatbotpytorch;
