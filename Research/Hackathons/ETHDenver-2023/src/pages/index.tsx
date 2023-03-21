import { Button, Link } from '@chakra-ui/react'
import type { NextPage } from 'next'
import Head from 'next/head'
import styles from '../styles/Home.module.css'
import { useSendTransaction } from 'wagmi'
import { BigNumber } from '@ethersproject/bignumber'
import AnimatedText from 'react-animated-text-content';
import { AwesomeButtonProgress } from "react-awesome-button";
import Image from 'next/image'
import profilePic from '../assets/Frameworks/truthgpt.png'
import {ButtonBase} from "@mui/material";
import { AwesomeButton } from "react-awesome-button";
import VoteForm from './VoteForm';
import App from "./make-decision";
import OpenAi from "./AI-FRAMEWORK";
import CreateAIDAO from "./hack";
import pytorch from "./pytorch";
import tensorflow from "./tensorflow";
import Pytorch from "./pytorch";
import Tensorflow from "./tensorflow";
import Chatbottensor from "./tensorflow";
import Chatbot from "./chatbot";
import Smartcontract from "./existing-sm";

const Home: NextPage = () => {

  // TODO: Adapt to the Backend is Next Project the send tx is = ? the string
  const { data, isIdle, isError, isLoading, isSuccess, sendTransaction } =
      useSendTransaction({
        request: {
          to: 'yanniksood.eth',
          value: BigNumber.from('10000000000000000'), // .1 ETH
        },
      })

  return (
      <div className={styles.container} style={{ backgroundColor: '#000000' }}>
        <Head>
            <a href="https://github.com/Blockchain-Mexico/TruthGPT--TrueGPT-" title="TruthGPT Github Repository">Open source</a>
          <meta name="description" content="ETH + Next.js DApp Boilerplate" />
          <link rel="icon" href="/favicon.ico" />
        </Head>

        <main className={styles.main}>
            <div className="grid-element">
                <Image
                    src={profilePic}
                    alt="Picture of the author"
                    width={200} automatically provided
                    height={200} automatically provided
                    blurDataURL="data:..." automatically provided
                    placeholder="blur" // Optional blur-up while loading
                />
            </div>
            <h1 className={styles.title} style={{ fontSize: '0.1 rem' }}>
                TruthGPT
                <AnimatedText
                    type="words"
                    animation={{
                        x: '200px',
                        y: '-20px',
                        scale: 1.1,
                        ease: 'ease-in-out',
                    }}
                    animationType="lights"
                    interval={0.00006}
                    duration={5.85}
                    tag="p"
                    className="animated-paragraph"
                    includeWhiteSpaces
                    threshold={0.1}
                    rootMargin="20%"
                />
                <AnimatedText
                    type="words"
                    animation={{
                        x: '200px',
                        y: '-20px',
                        scale: 1.1,
                        ease: 'ease-in-out',
                    }}
                    animationType="rifle"
                    interval={0.0006}
                    duration={1.25}
                    tag="p"
                    className="animated-paragraph"
                    includeWhiteSpaces
                    threshold={0.1}
                    rootMargin="20%"
                >
                    Make your Web3 journey a AI-powered smart contracts
                </AnimatedText>
            </h1>
            <CreateAIDAO></CreateAIDAO>
            <Chatbot></Chatbot>
            <Smartcontract></Smartcontract>
            <div className={styles.grid}>
            <Link href='/Research/Hackathons/ETHDenver-2023/src/pages/confirmation' >
              <Button
                  backgroundColor="#9c44dc"
                  borderRadius="25px"
                  margin={2.5}
                  _hover={{
                    bg: '#E4007C'
                  }}
                  _active={{
                    bg: '#E4007C'
                  }}
              >
                <p> Create AI Smart contract </p>
              </Button>
            </Link>
          </div>
          <div>
            <Button
                backgroundColor="#F3BA2F"
                borderRadius="30px"
                margin={2.5}
                _hover={{
                  bg: '#E4007C'
                }}
                _active={{
                  bg: '#E4007C'
                }}
                onClick={() => sendTransaction()}
            >
              <p> Creat AI DAO </p>
            </Button>
          </div>
          <div>
            <Button
                backgroundColor="#32CD32"
                borderRadius="25px"
                margin={2.5}
                _hover={{
                  bg: '#00'
                }}
                _active={{
                  bg: '#0000'
                }}
                onClick={() => sendTransaction()}
            >
              <p> Go !</p>
            </Button>
          </div>
        </main>
      </div>
  )
}

export default Home
