use foundry::prelude::*;

contract! {
    struct AIDAO {
        ai: address,
        dao: address,
        token: address,
        vote_threshold: u256,
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
        }
    }
}
