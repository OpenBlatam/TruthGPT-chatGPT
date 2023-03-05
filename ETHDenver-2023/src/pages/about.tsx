import type { NextPage } from 'next'

import { Heading, Flex, Text, Stack } from '@chakra-ui/layout'
import {Input, Button, ButtonGroup, Box} from '@chakra-ui/react'
import { Image } from '@chakra-ui/react'
import {WorldIDWidget} from "@worldcoin/id";
import {VerificationResponse} from "@worldcoin/id/dist/types";
import {useAccount, useContract, usePrepareContractWrite, useContractWrite, useProvider, useSigner} from "wagmi";

import { useState } from "react";
import { ProposalForm } from './VoteForm';
import { defaultAbiCoder as abi } from "@ethersproject/abi";

const About: NextPage = () => {

    const [isVerified, setIsVerified] = useState(false);
    const [proof, setProof] = useState("");
    const [txHash, setTxHash] = useState("");

    const [name, setName] = useState("");
    const [description, setDescription] = useState("");
    const [goal, setGoal] = useState(0);

    const handleChangeName = (event) => setName(event.target.value)
    const handleChangeDescription = (event) => setDescription(event.target.value)
    const handleChangeGoal = (event) => setGoal(event.target.value)

    const [np, setNp] = useState({
        name:"",
        description:"",
        goal:0,
    });

    const { address, isConnecting, isDisconnected, isConnected } = useAccount()

    const { data: signer, isError, isLoading } = useSigner()



     // @ts-ignore
    const contract = useContract({
            address: '0x10E08Eb5275b269A7c125Af54047Bb89F13AeeDe',
            abi: [
                {
                    "inputs": [
                        {
                            "internalType": "contracts IWorldID",
                            "name": "_worldId",
                            "type": "address"
                        },
                        {
                            "internalType": "string",
                            "name": "_actionId",
                            "type": "string"
                        },
                        {
                            "internalType": "address payable",
                            "name": "_tellorAddress",
                            "type": "address"
                        }
                    ],
                    "stateMutability": "nonpayable",
                    "type": "constructor"
                },
                {
                    "inputs": [],
                    "name": "InvalidNullifier",
                    "type": "error"
                },
                {
                    "anonymous": false,
                    "inputs": [
                        {
                            "indexed": false,
                            "internalType": "address",
                            "name": "_address",
                            "type": "address"
                        },
                        {
                            "indexed": false,
                            "internalType": "uint256",
                            "name": "_amount",
                            "type": "uint256"
                        },
                        {
                            "indexed": false,
                            "internalType": "enum Contract.BetResult",
                            "name": "_result",
                            "type": "uint8"
                        }
                    ],
                    "name": "BetPlaced",
                    "type": "event"
                },
                {
                    "anonymous": false,
                    "inputs": [
                        {
                            "indexed": false,
                            "internalType": "address",
                            "name": "_address",
                            "type": "address"
                        },
                        {
                            "indexed": false,
                            "internalType": "uint256",
                            "name": "_amount",
                            "type": "uint256"
                        },
                        {
                            "indexed": false,
                            "internalType": "enum Contract.BetResult",
                            "name": "_result",
                            "type": "uint8"
                        }
                    ],
                    "name": "BetSettled",
                    "type": "event"
                },
                {
                    "anonymous": false,
                    "inputs": [
                        {
                            "indexed": false,
                            "internalType": "string",
                            "name": "str",
                            "type": "string"
                        },
                        {
                            "indexed": false,
                            "internalType": "uint256",
                            "name": "value",
                            "type": "uint256"
                        }
                    ],
                    "name": "LogInt",
                    "type": "event"
                },
                {
                    "anonymous": false,
                    "inputs": [
                        {
                            "indexed": false,
                            "internalType": "bytes",
                            "name": "queryData",
                            "type": "bytes"
                        },
                        {
                            "indexed": false,
                            "internalType": "bytes32",
                            "name": "queryId",
                            "type": "bytes32"
                        }
                    ],
                    "name": "Query",
                    "type": "event"
                },
                {
                    "anonymous": false,
                    "inputs": [
                        {
                            "indexed": false,
                            "internalType": "uint256",
                            "name": "requestId",
                            "type": "uint256"
                        },
                        {
                            "indexed": false,
                            "internalType": "string",
                            "name": "tweet",
                            "type": "string"
                        }
                    ],
                    "name": "RequestMade",
                    "type": "event"
                },
                {
                    "inputs": [
                        {
                            "internalType": "string",
                            "name": "_name",
                            "type": "string"
                        },
                        {
                            "internalType": "string",
                            "name": "_description",
                            "type": "string"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_goal",
                            "type": "uint256"
                        }
                    ],
                    "name": "addNP",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [],
                    "name": "balance",
                    "outputs": [
                        {
                            "internalType": "uint256",
                            "name": "",
                            "type": "uint256"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "uint256",
                            "name": "betId",
                            "type": "uint256"
                        },
                        {
                            "internalType": "enum Contract.BetResult",
                            "name": "betResult",
                            "type": "uint8"
                        }
                    ],
                    "name": "bet",
                    "outputs": [],
                    "stateMutability": "payable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "uint256",
                            "name": "",
                            "type": "uint256"
                        }
                    ],
                    "name": "bets",
                    "outputs": [
                        {
                            "internalType": "address",
                            "name": "owner",
                            "type": "address"
                        },
                        {
                            "internalType": "uint256",
                            "name": "resultAvailabiityTimestamp",
                            "type": "uint256"
                        },
                        {
                            "internalType": "enum Contract.BetResult",
                            "name": "result",
                            "type": "uint8"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "string",
                            "name": "tweet",
                            "type": "string"
                        }
                    ],
                    "name": "callMidpoint",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [],
                    "name": "count",
                    "outputs": [
                        {
                            "internalType": "uint256",
                            "name": "",
                            "type": "uint256"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "uint256",
                            "name": "betId",
                            "type": "uint256"
                        }
                    ],
                    "name": "getBetResult",
                    "outputs": [
                        {
                            "internalType": "enum Contract.BetResult",
                            "name": "",
                            "type": "uint8"
                        }
                    ],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "bytes32",
                            "name": "_queryId",
                            "type": "bytes32"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_timestamp",
                            "type": "uint256"
                        }
                    ],
                    "name": "getDataAfter",
                    "outputs": [
                        {
                            "internalType": "bytes",
                            "name": "_value",
                            "type": "bytes"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_timestampRetrieved",
                            "type": "uint256"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "bytes32",
                            "name": "_queryId",
                            "type": "bytes32"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_timestamp",
                            "type": "uint256"
                        }
                    ],
                    "name": "getDataBefore",
                    "outputs": [
                        {
                            "internalType": "bytes",
                            "name": "_value",
                            "type": "bytes"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_timestampRetrieved",
                            "type": "uint256"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "bytes32",
                            "name": "_queryId",
                            "type": "bytes32"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_timestamp",
                            "type": "uint256"
                        }
                    ],
                    "name": "getIndexForDataAfter",
                    "outputs": [
                        {
                            "internalType": "bool",
                            "name": "_found",
                            "type": "bool"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_index",
                            "type": "uint256"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "bytes32",
                            "name": "_queryId",
                            "type": "bytes32"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_timestamp",
                            "type": "uint256"
                        }
                    ],
                    "name": "getIndexForDataBefore",
                    "outputs": [
                        {
                            "internalType": "bool",
                            "name": "_found",
                            "type": "bool"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_index",
                            "type": "uint256"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "bytes32",
                            "name": "_queryId",
                            "type": "bytes32"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_timestamp",
                            "type": "uint256"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_maxAge",
                            "type": "uint256"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_maxCount",
                            "type": "uint256"
                        }
                    ],
                    "name": "getMultipleValuesBefore",
                    "outputs": [
                        {
                            "internalType": "bytes[]",
                            "name": "_values",
                            "type": "bytes[]"
                        },
                        {
                            "internalType": "uint256[]",
                            "name": "_timestamps",
                            "type": "uint256[]"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "bytes32",
                            "name": "_queryId",
                            "type": "bytes32"
                        }
                    ],
                    "name": "getNewValueCountbyQueryId",
                    "outputs": [
                        {
                            "internalType": "uint256",
                            "name": "",
                            "type": "uint256"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "bytes32",
                            "name": "_queryId",
                            "type": "bytes32"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_timestamp",
                            "type": "uint256"
                        }
                    ],
                    "name": "getReporterByTimestamp",
                    "outputs": [
                        {
                            "internalType": "address",
                            "name": "",
                            "type": "address"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "bytes32",
                            "name": "_queryId",
                            "type": "bytes32"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_index",
                            "type": "uint256"
                        }
                    ],
                    "name": "getTimestampbyQueryIdandIndex",
                    "outputs": [
                        {
                            "internalType": "uint256",
                            "name": "",
                            "type": "uint256"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [],
                    "name": "idMappingContract",
                    "outputs": [
                        {
                            "internalType": "contracts IMappingContract",
                            "name": "",
                            "type": "address"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "bytes32",
                            "name": "_queryId",
                            "type": "bytes32"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_timestamp",
                            "type": "uint256"
                        }
                    ],
                    "name": "isInDispute",
                    "outputs": [
                        {
                            "internalType": "bool",
                            "name": "",
                            "type": "bool"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "address",
                            "name": "",
                            "type": "address"
                        }
                    ],
                    "name": "nps",
                    "outputs": [
                        {
                            "internalType": "string",
                            "name": "name",
                            "type": "string"
                        },
                        {
                            "internalType": "string",
                            "name": "description",
                            "type": "string"
                        },
                        {
                            "internalType": "uint256",
                            "name": "goal",
                            "type": "uint256"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "bytes32",
                            "name": "_queryId",
                            "type": "bytes32"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_timestamp",
                            "type": "uint256"
                        }
                    ],
                    "name": "retrieveData",
                    "outputs": [
                        {
                            "internalType": "bytes",
                            "name": "",
                            "type": "bytes"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "address",
                            "name": "_addy",
                            "type": "address"
                        }
                    ],
                    "name": "setIdMappingContract",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "uint256",
                            "name": "betId",
                            "type": "uint256"
                        }
                    ],
                    "name": "settleBet",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [],
                    "name": "tellor",
                    "outputs": [
                        {
                            "internalType": "contracts ITellor",
                            "name": "",
                            "type": "address"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "bytes32",
                            "name": "_id",
                            "type": "bytes32"
                        }
                    ],
                    "name": "valueFor",
                    "outputs": [
                        {
                            "internalType": "int256",
                            "name": "_value",
                            "type": "int256"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_timestamp",
                            "type": "uint256"
                        },
                        {
                            "internalType": "uint256",
                            "name": "_statusCode",
                            "type": "uint256"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "address",
                            "name": "input",
                            "type": "address"
                        },
                        {
                            "internalType": "uint256",
                            "name": "root",
                            "type": "uint256"
                        },
                        {
                            "internalType": "uint256",
                            "name": "nullifierHash",
                            "type": "uint256"
                        },
                        {
                            "internalType": "uint256[8]",
                            "name": "proof",
                            "type": "uint256[8]"
                        }
                    ],
                    "name": "verifyAndExecute",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                }
            ],
            signerOrProvider: signer,
        })

    const setWorldIDProof = async (proof: VerificationResponse) => {
        setIsVerified(true)

       /* console.log("proof")
        console.log(proof)
        console.log(address)
        console.log(proof.merkle_root)
        console.log(proof.nullifier_hash)
        console.log(abi.decode(["uint256[8]"], proof.proof)[0])
        console.log(proof.proof)

        const claimResult = await contracts.verifyAndExecute(
            address,
            proof.merkle_root,
            proof.nullifier_hash,
            abi.decode(["uint256[8]"], proof.proof)[0],
            { gasLimit: 10000000 },
        );
        setTxHash((claimResult as Record<string, string>).hash);
        console.log("Airdrop claimed successfully!", claimResult);*/

        const res = await contract.nps(
            address,
            { gasLimit: 10000000 },
        );
        console.log("Airdrop claimed successfully!", res);
        setNp(res)
    }

    const handleCreate = async () => {
        const res = await contract.addNP(
            name,
            description,
            goal,
            { gasLimit: 10000000 },
        );
        console.log("Airdrop claimed successfully!", res);
        setNp({
            "name":name,
            "description": description,
            "goal":goal
        })
    }
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
    <Flex
      direction="row"
      width="100%"
      height="90%"
      alignItems="center"
      justifyContent="space-between"
      padding="2rem"
    >
      <Flex
        width="50%"
        height="100%"
        direction="column"
      >


          <Heading>
              Profile
          </Heading>

          <div id="world-id-container"></div>

          {(isConnected && !isVerified) &&
              <div>
                  <Text>
                      Verify yourself to create a profile
                  </Text>

                  <WorldIDWidget
                      signal={address}
                      actionId={"wid_staging_71a98c5a470f53875783af0504c302c9"}
                      onSuccess={(proof) => setWorldIDProof(proof)}
                      debug
                  />
              </div>
          }

          {(isVerified && np.goal == 0) &&
              <Stack spacing={3}>
                  <Text>Create Profile</Text>
                  <Input
                      placeholder='Nombre'
                      onChange={handleChangeName}
                  />
                  <Input
                      placeholder='Descripcion'
                      onChange={handleChangeDescription}
                  />
                  <Input
                      placeholder='Meta'
                      onChange={handleChangeGoal}
                  />
                  <Button onClick={handleCreate} colorScheme='blue'>Button</Button>
              </Stack>
          }

          {(np.goal > 0) &&
              <Stack spacing={3}>

                  <Box boxSize='sm'>
                      <Text>{np.name}</Text>
                      <Image src='https://images.squarespace-cdn.com/content/v1/566f06865a5668e19853f7d6/1565278933742-V78Q6YLWQ9P347OBCM9E/Streetdog+logo+with+white_round.jpg' alt='Dan Abramov' />
                      <Text>{np.description}</Text>
                  </Box>

              </Stack>
          }
      </Flex>

    </Flex>
  )
}

export default About
