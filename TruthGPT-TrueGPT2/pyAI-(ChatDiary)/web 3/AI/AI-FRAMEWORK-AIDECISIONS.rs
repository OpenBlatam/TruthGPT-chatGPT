use std::error::Error;
use std::sync::Arc;

use web3::transports::WebSocket;
use web3::types::{FilterBuilder, Log};
use web3::{api::Web3, transports::Batch, Transport};

use ethcontract::{batch::CallBatch, contract::Options, prelude::*, Http};

// Address of the AIDecisionMaker contract
const AI_DECISION_MAKER_ADDRESS: &str = "0x...";

// ABI of the AIDecisionMaker contract
const AI_DECISION_MAKER_ABI: &str = r#"[{"inputs":[{"internalType":"address","name":"_aiAddress","type":"address"}],"stateMutability":"nonpayable","type":"constructor"},{"inputs":[{"internalType":"bytes","name":"_data","type":"bytes"}],"name":"makeDecision","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"}]"#;

async fn listen_for_decisions<T: Transport + Send + Sync + 'static>(
    web3: Arc<Web3<T>>,
    ai_decision_maker: Arc<Contract<T>>,
) -> Result<(), Box<dyn Error>> {
    // Set up the stream filter
    let filter = FilterBuilder::default()
        .address(vec![Address::from_str(AI_DECISION_MAKER_ADDRESS)?])
        .topics(Some(vec![ai_decision_maker.event_topic("DecisionMade")]), None, None, None)
        .build();

    // Start the stream
    let mut stream = web3.eth_subscribe().subscribe_logs(filter).await?;
    while let Some(log) = stream.next().await {
        let decision_data = log.data.0;
        let decision = ai_decision_maker
            .event::<(bool,)>("DecisionMade")
            .decode(&decision_data, log.topics.clone())?
            .0;

        if decision {
            println!("Decision accepted!");
        } else {
            println!("Decision rejected!");
        }
    }

    Ok(())
}

#[async_std::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the Ethereum provider and Web3 instance
    let transport = WebSocket::new("ws://localhost:8546").await?;
    let web3 = Arc::new(Web3::new(transport));

    // Initialize the Ethereum wallet
    let accounts = web3.eth().accounts().await?;
    let wallet = accounts[0];

    // Initialize the AIDecisionMaker contract
    let ai_decision_maker = Arc::new(
        Contract::new(
            web3.clone(),
            Address::from_str(AI_DECISION_MAKER_ADDRESS)?,
            AI_DECISION_MAKER_ABI.as_bytes(),
        )
        .options(Options::default().set_from(wallet)),
    );

    listen_for_decisions(web3, ai_decision_maker).await?;

    Ok(())
}
