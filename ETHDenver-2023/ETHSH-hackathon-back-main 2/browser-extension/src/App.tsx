import React, { useEffect, useState } from 'react';
// import logo from './images/ethglobalLogo.png';
import './App.css';
import { useEnsResolver, useContractRead } from 'wagmi'
import {
  WagmiConfig,
  createClient,
  configureChains,
  chain,
  defaultChains,
} from 'wagmi'
import { publicProvider } from 'wagmi/providers/public'

const { chains, provider, webSocketProvider } = configureChains(
    [chain.mainnet, chain.polygon],
    [publicProvider()],
)

const client = createClient({
  autoConnect: true,
  provider,
  webSocketProvider,
})

export const App = () => {
  const [url, setUrl] = useState<string>('');
  let trigger = false

  // Get current URL
  useEffect(() => {
    const queryInfo = {active: true, lastFocusedWindow: true};

    chrome.tabs && chrome.tabs.query(queryInfo, tabs => {
      const url = tabs[0].url || "";
      if (!url.includes("https://ethglobal-dao-tool.eth") && !url.includes("https://ethglobal-dao-tool.eth.limo")) {
        // alert("BE CAREFUL\nYou're on an unapproved dApp by the DAO!")
        chrome.tabs.executeScript({
          code:`alert("BE CAREFUL\\nYou're on an unapproved dApp by the DAO!")`
        });
      }
      setUrl(url);
    });
  }, [trigger]);

  chrome.tabs.onUpdated.addListener( function (tabId, changeInfo, tab) {
    if (changeInfo.status == 'complete') {
      trigger = !trigger
    }
  })

  return (
    <WagmiConfig client={client}>
      <div className="App">
      <header className="App-header">
        <img src={"./ethglobalLogo.png"} className="App-logo" alt="logo" />
        <EnsResolver/>
        <p>
          {url}
        </p>
      </header>
    </div>
    </WagmiConfig>
  );
}

const EnsResolver = () => {
  const { data, isError, isLoading } = useEnsResolver({
    name: 'ethglobal-dao-tool.eth',
  })
  if (isLoading) return <div>Fetching resolver…</div>
  if (isError) return <div>Error fetching resolver</div>
  return <div>ENS Resolved</div>
}

const ReadContract = () => {
  const { data, isError, isLoading } = useContractRead({
    address: '0xf9191de4976Bcd558E0013Ce0Dc34846301D4Fec',
    // abi: wagmigotchiABI,
    functionName: 'getPropInfo',
  })
}