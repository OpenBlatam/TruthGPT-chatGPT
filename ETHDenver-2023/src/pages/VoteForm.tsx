import { useState } from 'react';
import { ethers } from 'ethers';
//import abi from './AIDAO.json';

const contractAddress = '0x123...'; // replace with the deployed contract address
// const provider = new ethers.providers.Web3Provider(window.ethereum);
// const signer = provider.getSigner();
// const contract = new ethers.Contract(contractAddress, abi, signer) as AIDAO;

interface AIDAO {
    voteOnProposal: (proposalId: number, inputData: string) => Promise<ethers.providers.TransactionResponse>;
}

function VoteForm(): JSX.Element {
    const [proposalId, setProposalId] = useState<number>();
    const [inputData, setInputData] = useState<string>('');
    const [message, setMessage] = useState<string>('');

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        try {
            if (!proposalId) {
                setMessage('Proposal ID is required');
                return;
            }
            if (inputData === '') {
                setMessage('Input data is required');
                return;
            }
            const tx = await contract.voteOnProposal(proposalId, inputData);
            const receipt = await tx.wait();
            setMessage(`Transaction confirmed: ${receipt.transactionHash}`);
        } catch (error) {
            console.error(error);
            setMessage(`Error: ${error.message}`);
        }
    };

    return (
        <form onSubmit={handleSubmit}>
        <label htmlFor="proposal-id">Proposal ID:</label>
    <input type="number" id="proposal-id" value={proposalId} onChange={(event) => setProposalId(Number(event.target.value))} required /><br />
    <label htmlFor="input-data">Input Data:</label>
    <textarea id="input-data" rows={4} cols={50} value={inputData} onChange={(event) => setInputData(event.target.value)} required></textarea><br />
    <button type="submit">Submit</button>
        <div>{message}</div>
        </form>
);
}
