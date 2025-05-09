// use kafka for better perfomance ?

use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::message::ToBytes;
use serde::{Serialize, Deserialize};
use foundry::prelude::*;

const KAFKA_TOPIC: &str = "policy_votes";

#[derive(Serialize, Deserialize)]
struct VoteMessage {
    proposal_id: u64,
    vote: u64,
}

contract! {
    struct AIDAO {
        ai: address,
        dao: address,
        token: address,
        vote_threshold: u256,
        kafka_bootstrap_servers: String,
        kafka_producer: Option<FutureProducer>,
        kafka_consumer: Option<StreamConsumer>,
    }

    impl AIDAO {
        pub fn vote_on_proposal(&mut self, proposal_id: u256, input_data: Vec<u8>) {
            assert_eq!(self.is_owner(), true, "Only contract owner can call this function");
            let dao = ERC20::at(&self.dao);
            let balance = dao.balance_of(&self.address());
            assert!(balance >= self.vote_threshold, "Contract does not have enough voting power");
            let ai = OpenAI::at(&self.ai);
            let vote = ai.predict_vote(input_data);
            dao.vote(proposal_id, vote);
            self.send_vote_message(proposal_id, vote);
        }

        fn send_vote_message(&mut self, proposal_id: u256, vote: u256) {
            let message = VoteMessage {
                proposal_id: proposal_id.into(),
                vote: vote.into(),
            };
            let producer = self.kafka_producer.get_or_insert_with(|| {
                let mut config = ClientConfig::new();
                config.set("bootstrap.servers", &self.kafka_bootstrap_servers);
                FutureProducer::from_config(&config).unwrap()
            });
            let key = "".to_owned();
            let payload = serde_json::to_vec(&message).unwrap();
            let record = FutureRecord::to(KAFKA_TOPIC)
                .payload(&payload)
                .key(&key);
            let _ = producer.send(record, 0).map_err(|e| eprintln!("Error sending message: {:?}", e));
        }

        fn receive_vote_message(&mut self) {
            let consumer = self.kafka_consumer.get_or_insert_with(|| {
                let mut config = ClientConfig::new();
                config.set("bootstrap.servers", &self.kafka_bootstrap_servers);
                config.set("group.id", "policy-votes");
                config.set("auto.offset.reset", "earliest");
                StreamConsumer::from_config(&config).unwrap()
            });
            consumer.subscribe(&[KAFKA_TOPIC]).unwrap();
            for message in consumer.start().wait() {
                match message {
                    Err(e) => eprintln!("Error receiving message: {:?}", e),
                    Ok(message) => {
                        let key = message.key().unwrap_or_default();
                        let value = message.payload().unwrap();
                        let message = serde_json::from_slice::<VoteMessage>(&value).unwrap();
                        self.handle_vote_message(message.proposal_id, message.vote);
                        consumer.commit_message(&message, rdkafka::consumer::CommitMode::Async).unwrap();
                    }
                }
            }
        }

        fn handle_vote_message(&mut self, proposal_id: u64, vote: u64) {
            // Do something with the received vote
        }
    }
}
