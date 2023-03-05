import React, { useState } from 'react';
import Web3 from 'web3';
import DecisionMaker from '../../contracts/truffle/build/contracts/DecisionMaker.json';

const web3 = new Web3(Web3.givenProvider || 'http://localhost:8545');

function Smartcontract() {
    const [inputData, setInputData] = useState([]);
    const [decisionResult, setDecisionResult] = useState(null);
    const [loading, setLoading] = useState(false);

    async function makeDecision() {
        try {
            setLoading(true);

            const accounts = await web3.eth.getAccounts();
            const networkId = await web3.eth.net.getId();
            const networkData = DecisionMaker.networks[networkId];
            const decisionMakerInstance = new web3.eth.Contract(
                DecisionMaker.abi,
                networkData.address
            );

            await decisionMakerInstance.methods.makeDecision(inputData).send({ from: accounts[0] });

            const decisionResult = await decisionMakerInstance.methods.decision().call();
            setDecisionResult(decisionResult.result);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    }

    return (
        <div>
            <h1>Decision Maker in smart contract </h1>
            <div>
                <label>Input Data:</label>
                <input
                    type="text"
                    value={inputData}
                    onChange={(event) => setInputData(event.target.value)}
                />
            </div>
            <button onClick={makeDecision} disabled={loading}>
                Make Decision
            </button>
            {decisionResult !== null && (
                <div>
                    <p>Decision Result: {decisionResult ? 'Yes' : 'No'}</p>
                </div>
            )}
        </div>
    );
}

export default Smartcontract;
