import React, { useState } from 'react';
import { ethers } from 'ethers';
import styled from 'styled-components';

const Container = styled.div`
  display: flex;
  flex-direction: column;
  max-width: 800px;
  margin: 0 auto;
`;

const Title = styled.h1`
  font-size: 3xl;
  font-weight: bold;
  margin-bottom: 4px;
`;

const Input = styled.input`
  border-width: 1px;
  border-color: gray-300;
  border-radius: 4px;
  padding: 8px;
  font-size: 16px;
  margin-bottom: 16px;
`;

const Textarea = styled.textarea`
  border-width: 1px;
  border-color: gray-300;
  border-radius: 4px;
  padding: 8px;
  font-size: 16px;
  margin-bottom: 16px;
`;

const Button = styled.button`
  background-color: blue-500;
  color: white;
  padding: 12px;
  font-size: 16px;
  border-radius: 4px;
  align-self: flex-start;
  transition: background-color 0.2s ease;
  &:hover {
    background-color: blue-600;
  }
`;

const ResultContainer = styled.div`
  margin-top: 32px;
  padding: 16px;
  border-width: 1px;
  border-color: gray-300;
  border-radius: 4px;
  background-color: gray-50;
`;

const ResultTitle = styled.h2`
  font-size: xl;
  font-weight: bold;
  margin-bottom: 8px;
`;

const ResultAddress = styled.p`
  word-break: break-all;
  font-size: 16px;
  font-family: monospace;
`;

const CreateAIDAO = () => {
    const [aiAddress, setAIAddress] = useState('');
    const [inputData, setInputData] = useState('');
    const [contractAddress, setContractAddress] = useState('');

    const createAIDAO = async () => {
        const provider = new ethers.providers.Web3Provider(window.ethereum);
        const signer = provider.getSigner();
        const decisionMakerFactory = new ethers.ContractFactory(
            DecisionMaker.abi,
            DecisionMaker.bytecode,
            signer
        );
        const decisionMaker = await decisionMakerFactory.deploy(aiAddress);
        await decisionMaker.makeDecision(inputData);
        setContractAddress(decisionMaker.address);
    };

    return (
        <Container>
            <Title>Create an AI DAO</Title>
            <div>
                <label htmlFor="ai-address">AI Address:</label>
                <Input
                    id="ai-address"
                    type="text"
                    value={aiAddress}
                    onChange={(e) => setAIAddress(e.target.value)}
                />
            </div>
            <div>
                <label htmlFor="input-data">Input Data:</label>
                <Textarea
                    id="input-data"
                    value={inputData}
                    onChange={(e) => setInputData(e.target.value)}
                ></Textarea>
            </div>
            <Button onClick={createAIDAO}>Create AI DAO</Button>
            {contractAddress && (
                <ResultContainer>
                    <ResultTitle>AI DAO contract address:</ResultTitle>
                    <ResultAddress>{contractAddress}</ResultAddress>
                </ResultContainer>
            )}
        </Container>
    );
};

export default CreateAIDAO;
