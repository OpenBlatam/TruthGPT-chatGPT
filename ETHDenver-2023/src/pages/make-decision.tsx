import React, { useState, useEffect } from 'react';
import Web3 from 'web3';

const AI_DECISION_MAKER_ADDRESS = '0x...';

const AI_DECISION_MAKER_ABI = [
    {
        inputs: [
            {
                internalType: 'address',
                name: '_aiAddress',
                type: 'address',
            },
        ],
        stateMutability: 'nonpayable',
        type: 'constructor',
    },
    {
        inputs: [
            {
                internalType: 'bytes',
                name: '_data',
                type: 'bytes',
            },
        ],
        name: 'makeDecision',
        outputs: [
            {
                internalType: 'bool',
                name: '',
                type: 'bool',
            },
        ],
        stateMutability: 'view',
        type: 'function',
    },
];

function App() {
    const [web3, setWeb3] = useState(null);
    const [contract, setContract] = useState(null);
    const [decisions, setDecisions] = useState([]);

    useEffect(() => {
        const init = async () => {
            const provider = await Web3.givenProvider;
            if (provider) {
                const web3Instance = new Web3(provider);
                setWeb3(web3Instance);

                const contractInstance = new web3Instance.eth.Contract(
                    AI_DECISION_MAKER_ABI,
                    AI_DECISION_MAKER_ADDRESS,
                );
                setContract(contractInstance);

                contractInstance.events.DecisionMade().on('data', (event) => {
                    const decision = event.returnValues[0];
                    setDecisions(prevState => [...prevState, decision]);
                });
            }
        };

        init();
    }, []);

    const makeDecision = async () => {
        if (!contract) return;
        const decision = await contract.methods.makeDecision().call();
        setDecisions(prevState => [...prevState, decision]);
    };

    return (
        <div>
            <h1>AI Decision Maker</h1>
    <button onClick={makeDecision} disabled={!contract}>
    Make Decision
    </button>
    <ul>
    {decisions.map((decision, index) => (
            <li key={index}>{decision ? 'Accepted' : 'Rejected'}</li>
        ))}
    </ul>
    </div>
);
}

export default App;
