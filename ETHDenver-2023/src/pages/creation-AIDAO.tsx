import { useState } from 'react';
import { ethers } from 'ethers';
import MyDAO from '../contracts/MyDAO.json';

type Props = {
    provider: ethers.providers.Web3Provider;
};

export default function MyDAOForm({ provider }: Props) {
    const [aiAddress, setAIAddress] = useState<string>('');
    const [txHash, setTxHash] = useState<string>('');

    async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
        event.preventDefault();

        const signer = provider.getSigner();
        const myDAOAddress = '0x...'; // Replace with the deployed address of your MyDAO contracts
        const myDAO = new ethers.Contract(myDAOAddress, MyDAO.abi, signer);

        try {
            const tx = await myDAO.createDecisionMakerApp(aiAddress);
            setTxHash(tx.hash);
        } catch (error) {
            console.error(error);
        }
    }

    return (
        <form onSubmit={handleSubmit}>
            <label>
                AI Address:
        <input type="text" value={aiAddress} onChange={(event) => setAIAddress(event.target.value)} />
    </label>
    <button type="submit">Create Decision Maker App</button>
    {txHash && <p>Transaction hash: {txHash}</p>}
    </form>
    );
    }
