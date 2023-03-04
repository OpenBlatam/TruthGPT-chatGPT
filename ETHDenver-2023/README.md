# 👽 Welcome!

Creation for a framework of AI layer in web 3 (Decision making dex)

To create a smart contract that uses TensorFlow, you will need to use a programming language that is compatible with both Ethereum smart contracts and TensorFlow. Currently, Solidity is the most widely-used language for Ethereum smart contracts, but it does not support TensorFlow natively.

One possible approach is to use a combination of Solidity and Python. You can write a Solidity smart contract that provides an interface for interacting with TensorFlow, and use Python to implement the actual TensorFlow functionality. Here is a rough outline of how this could work:

Write the Solidity smart contract interface that defines the functions for interacting with TensorFlow. For example:
php
Copy code
contract IDecisionMaker {
function makeDecision(bytes data) public view returns (bool);
}
Write a Python script that implements the makeDecision function using TensorFlow. For example:
python
Copy code
import tensorflow as tf

model = tf.keras.models.load_model('my_model.h5')

def make_decision(data):
# preprocess data
x = preprocess_data(data)

    # use TensorFlow model to make decision
    y_pred = model.predict(x)

    return y_pred > 0.5
Use a library like eth-abi to encode and decode function calls between Solidity and Python. For example:
sql
Copy code
from eth_abi import encode_abi, decode_abi

# encode function call
encoded_data = encode_abi(['bytes'], [data])

# call Solidity contract function
result = contract.functions.makeDecision(encoded_data).call()

# decode result
decision = decode_abi(['bool'], result)[0]
Deploy the Solidity contract and call its functions from other Ethereum contracts or applications.
Note that this is just a high-level overview and there are many details to consider when implementing this approach, such as gas costs, data serialization, and error handling. It may also be possible to use other programming languages or frameworks that have better support for both Ethereum and TensorFlow, such as Vyper or PySyft.


It was a copy past ultra sound money.

Run:

yarn 

yarn run dev 


Why is this boilerplate the best? 

- Only open source libraries used!
- Easy Setup [3 lines!]
- Uses the latest stack
- Not dependent on any paid product or service
- It works really well

<img width="1435" alt="Screen Shot 2022-07-06 at 3 07 26 PM" src="https://user-images.githubusercontent.com/100057052/177624635-bf983ec8-38fb-422a-8e23-10d08c229e76.png">

## 💻 Latest Updates [July 6th, 2022]

- Integrated Rainbowkit
- Updated Wagmi
- Cleaned up the Nav code

## ⭐️ Star this repo!
If this boilerplate helps you build Ethereum dapps faster - please star this project, every star helps!

## 🤝 Need help?
If you need help with setting up the boilerplate or have other questions - reach out on [twitter](https://twitter.com/ilyxium)

## 🚀 Quick Start

```bash
git clone https://github.com/ilyxium/eth-nextjs-boilerplate.git
# then
npm install
# then
npm run dev
```

Please note: Some users have mentioned issues with using yarn, so I recommend using NPM

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## 🛠 What's under the hood

This boilerplate uses the following open source libraries to bring you a fully functional, mobile optimized dApp:

- [next.js](https://nextjs.org/docs)
- [react.js](https://reactjs.org/)
- [wagmi](https://wagmi-xyz.vercel.app/)
- [ethers.js](https://docs.ethers.io/v5/)
- [chakra](https://chakra-ui.com/)
- [rainbowkit](https://www.rainbowkit.com/docs/introduction)

## 💎 What you get

- Connect Wallet + One wagmi hook [send a transaction] done for you! 💲
- Scalable header + mobile menu ↔️
- Mobile friendly 📱

<img width="489" alt="Screen Shot 2022-07-06 at 3 13 53 PM" src="https://user-images.githubusercontent.com/100057052/177625571-5ee7a1fe-bbb1-4031-b3f5-cfb535171763.png">

- 3 pages, including a 404 page, and routing done ✅
- Examples of buttons and links 👽

## 🔌 Connectors

The boilerplate now uses [rainbowkit](https://www.rainbowkit.com/docs/introduction). They make it super easy to configure the connector, and allows you to choose chains, styling, and more. 

Change your chains in app.tsx:

```
  const { chains, provider } = configureChains(
    [chain.mainnet, chain.polygon, chain.optimism, chain.arbitrum],
    [
      alchemyProvider(),
      publicProvider()
    ]
  );
```


## 😃 Connect with me
Feel free to reach out on twitter [@yanniksood](https://twitter.com/yanniksood) and drop your questions, comments or suggestions on how I can improve this!

# Enjoy 
