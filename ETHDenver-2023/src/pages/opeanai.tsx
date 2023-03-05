import React, { useState } from "react";
import * as tf from "@tensorflow/tfjs";
import tensorflowLogo from '../../src/assets/Frameworks/tensorflow-logo.png';
import openai from "@openai/api";

// Initialize OpenAI API client
const openaiApiKey = process.env.REACT_APP_OPENAI_API_KEY; // Your OpenAI API key
const openaiApi = new openai(openaiApiKey);

const Openai = () => {
    // Define the machine learning model
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    model.compile({ optimizer: "sgd", loss: "meanSquaredError" });

    const [inputText, setInputText] = useState("");
    const [decision, setDecision] = useState("");

    const handleInputChange = (event) => {
        setInputText(event.target.value);
    };

    const handleDecision = async () => {
        // Convert input text to a numerical value
        const inputData = parseFloat(inputText);

        // Make a decision based on the machine learning model's output
        const mlDecision = makeDecision(model, inputData);

        // Get response from OpenAI's GPT-3 model
        const openaiResponse = await openaiApi.completions.create({
            engine: "davinci",
            prompt: `Should I make a decision based on the input ${inputText}?`,
            maxTokens: 50,
            n: 1,
            stop: "\n",
        });

        // Update state with decision output
        setDecision(`${mlDecision ? "Yes" : "No"}. OpenAI says: ${openaiResponse.data.choices[0].text}`);
    };

    const makeDecision = (model, inputData) => {
        const inputTensor = tf.tensor2d([[inputData]], [1, 1]);
        const outputTensor = model.predict(inputTensor);
        const outputData = outputTensor.dataSync();
        const mlDecision = outputData[0] > 0.5;
        return mlDecision;
    };

    return (
        <div>
            <input type="text" value={inputText} onChange={handleInputChange} />
            <button onClick={handleDecision}>
                <img src="https://www.tensorflow.org/images/tf_logo_social.png" alt="TensorFlow logo" width="150" height="150" />
                Get Decision
            </button>
            {decision && <div>Decision: {decision}</div>}
        </div>
    );
};

export default Openai;
