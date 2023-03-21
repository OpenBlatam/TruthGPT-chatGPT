import { useState } from 'react';
import styled from 'styled-components';

interface Props {
    contract: ContractType;
}

const base64Encode = (str: string): string => {
    if (typeof btoa === 'function') {
        return btoa(str);
    } else {
        // Polyfill for btoa() in Node.js
        return Buffer.from(str).toString('base64');
    }
};

const FormContainer = styled.form`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  padding: 32px;
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0px 0px 16px rgba(0, 0, 0, 0.1);
`;

const InputContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
`;

const InputLabel = styled.label`
  font-size: 18px;
  font-weight: 500;
  color: #333;
`;

const Input = styled.input`
  padding: 8px;
  border-radius: 4px;
  border: 1px solid #ccc;
  font-size: 16px;
`;

const TextArea = styled.textarea`
  padding: 8px;
  border-radius: 4px;
  border: 1px solid #ccc;
  font-size: 16px;
  resize: none;
`;

const ErrorText = styled.span`
  font-size: 14px;
  color: red;
`;

const Button = styled.button`
  padding: 8px 16px;
  border-radius: 4px;
  border: none;
  background-color: #007bff;
  color: #fff;
  font-size: 16px;
  font-weight: 500;
  cursor: pointer;

  &:hover {
    background-color: #0069d9;
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const Spinner = styled.div`
  border: 4px solid #ccc;
  border-top-color: #007bff;
  border-radius: 50%;
  width: 20px;
  height: 20px;
  animation: spin 1s linear infinite;

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
`;

const SuccessText = styled.div`
  font-size: 16px;
  color: #4caf50;
  font-weight: 500;
  text-align: center;
`;

interface Props {
    contract: ContractType;
}

const base64Encode = (str: string): string => {
    if (typeof btoa === 'function') {
        return btoa(str);
    } else {
// Polyfill for btoa() in Node.js
        return Buffer.from(str).toString('base64');
    }
};

const FormContainer = styled.form display: flex; flex-direction: column; align-items: center; gap: 16px; padding: 32px; background-color: #fff; border-radius: 8px; box-shadow: 0px 0px 16px rgba(0, 0, 0, 0.1);

const InputContainer = styled.div display: flex; flex-direction: column; gap: 8px; width: 100%;;

const InputLabel = styled.label font-size: 18px; font-weight: 500; color: #333;;

const Input = styled.input padding: 8px; border-radius: 4px; border: 1px solid #ccc; font-size: 16px;;

const TextArea = styled.textarea padding: 8px; border-radius: 4px; border: 1px solid #ccc; font-size: 16px; resize: none;;

const ErrorText = styled.span font-size: 14px; color: red;;

const Button = styled.button`
padding: 8px 16px;
border-radius: 4px;
border: none;
background-color: #007bff;
color: #fff;
font-size: 16px;
font-weight: 500;
cursor: pointer;

&:hover {
background-color: #0069d9;
}

&:disabled {
opacity: 0.5;
cursor: not-allowed;
}
`;

const Spinner = styled.div`
border: 4px solid #ccc;
border-top-color: #007bff;
border-radius: 50%;
width: 20px;
height: 20px;
animation: spin 1s linear infinite;

@keyframes spin {
to {
transform: rotate(360deg);
}
}
`;

const SuccessText = styled.div font-size: 16px; color: #4caf50; font-weight: 500; text-align: center;;

const ProposalForm: React.FC<Props> = ({ contract }) => {
    const [proposalId, setProposalId] = useState<number | undefined>();
    const [inputData, setInputData] = useState<string>('');
    const [proposalIdError, setProposalIdError] = useState<string>('');
    const [inputDataError, setInputDataError] = useState<string>('');
    const [message, setMessage] = useState<string>('');
    const [submitting, setSubmitting] = useState<boolean>(false);

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        scss
        Copy code
// Clear previous error messages
        setProposalIdError('');
        setInputDataError('');

// Validate inputs
        if (!proposalId || proposalId < 0 || isNaN(proposalId)) {
            setProposalIdError('Proposal ID must be a positive integer');
            return;
        }

        if (inputData.trim() === '') {
            setInputDataError('Input data cannot be empty');
            return;
        }

// Disable the submit

        button and display a spinner while the transaction is being processed
        setSubmitting(true);
        try {
            const tx = await contract.voteOnProposal(proposalId, inputData);
            const receipt = await tx.wait();
            setMessage(Transaction confirmed: ${receipt.transactionHash});
        } catch (error) {
            console.error(error);
            setMessage(Error: ${error.message});
        }

// Reset form fields and state
        setProposalId(undefined);
        setInputData('');
        setSubmitting(false);
    };

    return (
        <FormContainer onSubmit={handleSubmit}>
            <InputContainer>
                <InputLabel htmlFor="proposal-id">Proposal ID:</InputLabel>
                <Input
                    type="number"
                    id="proposal-id"
                    value={proposalId ?? ''}
                    onChange={(event) => setProposalId(Number(event.target.value))}
                    required
                />
                {proposalIdError && <ErrorText>{proposalIdError}</ErrorText>}
            </InputContainer>

            <InputContainer>
                <InputLabel htmlFor="input-data">Input Data:</InputLabel>
                <TextArea
                    id="input-data"
                    rows={4}
                    cols={50}
                    value={inputData}
                    onChange={(event) => setInputData(event.target.value)}
                    maxLength={200}
                    required
                />
                {inputDataError && <ErrorText>{inputDataError}</ErrorText>}
            </InputContainer>
            <Button type="submit" disabled={submitting}>
                {submitting ? <Spinner /> : 'Submit'}
            </Button>
            {message && <SuccessText>{message}</SuccessText>}
        </FormContainer>
    );
};

export default ProposalForm
