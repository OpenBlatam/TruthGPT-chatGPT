import { Button, Link } from '@chakra-ui/react'
import type { NextPage } from 'next'
import Head from 'next/head'
import styles from '../styles/Home.module.css'
import { useSendTransaction } from 'wagmi'
import { BigNumber } from '@ethersproject/bignumber'
import AnimatedText from 'react-animated-text-content';
import { AwesomeButtonProgress } from "react-awesome-button";
import Image from 'next/image'
import profilePic from '../assets/logos/BFG1.5@2x.png'
import {ButtonBase} from "@mui/material";
import { AwesomeButton } from "react-awesome-button";
import VoteForm from './VoteForm';

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
      <div className={styles.container}>
        <Head>
          <title> TruthGPT </title>
          <meta name="description" content="ETH + Next.js DApp Boilerplate" />
          <link rel="icon" href="/favicon.ico" />
        </Head>

        <main className={styles.main}>
            <div className="grid-element">
                <Image
                    src={profilePic}
                    alt="Picture of the author"
                    width={315} automatically provided
                    height={81} automatically provided
                    blurDataURL="data:..." automatically provided
                    placeholder="blur" // Optional blur-up while loading
                />
            </div>
            <h2 className={styles.title}>
            TruthGPT
                <AnimatedText
                type="words" // animate words or chars
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
            >

            </AnimatedText>
            <AnimatedText
                type="words" // animate words or chars
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
              With Safety
            </AnimatedText>

          </h2>

          <div className={styles.grid}>
            <Link href='/confirmation' >
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
                <p>Bet </p>
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
              <p> Donate </p>
            </Button>
          </div>
          <div>
            <Button
                backgroundColor="#32CD32"
                borderRadius="25px"
                margin={2.5}
                _hover={{
                  bg: '#E4007C'
                }}
                _active={{
                  bg: '#E4007C'
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
