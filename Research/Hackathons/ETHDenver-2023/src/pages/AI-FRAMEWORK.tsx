import React, { useState } from 'react';
import { ethers } from 'ethers';
//import ChatbotContract from './contracts/ChatbotContract.json';
//import './App.css';

function OpenAi() {
    const [provider, setProvider] = useState<ethers.providers.Web3Provider | null>(null);
    const [contract, setContract] = useState<ethers.Contract | null>(null);
    const [message, setMessage] = useState('');
    const [response, setResponse] = useState('');

    const connectWallet = async () => {
        try {
            const ethereum = window.ethereum;
            if (!ethereum) {
                alert('No Ethereum wallet detected. Please install Metamask or another Ethereum wallet extension and try again.');
                return;
            }

            const accounts = await ethereum.request({ method: 'eth_requestAccounts' });
            const provider = new ethers.providers.Web3Provider(ethereum);
            const signer = provider.getSigner();
            const contractAddress = '<your contracts address here>';
            const contract = new ethers.Contract(contractAddress, ChatbotContract.abi, signer);

            setProvider(provider);
            setContract(contract);
            alert('Connected to Ethereum wallet.');
        } catch (error) {
            alert(`Error connecting to Ethereum wallet: ${error.message}`);
        }
    };

    const handleInteract = async () => {
        try {
            if (!contract) {
                alert('No contracts instance found.');
                return;
            }

            const tx = await contract.interact(message);
            await tx.wait();
            const response = await contract.getResponse();
            setResponse(response);
        } catch (error) {
            alert(`Error interacting with chatbot: ${error.message}`);
        }
    };

    return (
        <div className="App">
            <header className="App-header">
                <h1>Adaptive AI Chatbot</h1>

                {!provider ? (
                    <button onClick={connectWallet}>Connect to Ethereum wallet</button>
                ) : (
                    <>
                        <label htmlFor="message">Enter message:</label>
                        <input id="message" type="text" value={message} onChange={(e) => setMessage(e.target.value)} />
                        <button onClick={handleInteract}>Interact</button>

                        {response && (
                            <>
                                <h2>Response:</h2>
                                <p>{response}</p>
                            </>
                        )}
                    </>
                )}
            </header>
        </div>
    );
}

export default OpenAi;
