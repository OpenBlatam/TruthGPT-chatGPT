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

#[cfg(test)]
mod tests {
    use super::*;
    use foundry::testkit::*;

    #[test]
    fn vote_on_proposal_test() {
        let mut context = TestContext::new();

        let ai = context.create_contract(
            AIDAO::builder()
                .ai(context.get_account(0))
                .dao(context.get_account(1))
                .token(context.get_account(2))
                .vote_threshold(100.into())
                .build()
        );
        let dao = context.create_contract(ERC20::builder().build());
        let input_data = b"some input data".to_vec();

        // Set up initial balances
        dao.set_balance(ai.address(), 200.into()).unwrap();

        // Call the function to vote on a proposal
        ai.call_fn("vote_on_proposal", (0.into(), input_data)).unwrap();

        // Verify that the vote was cast
        assert_eq!(dao.get_votes(0.into()), 1.into());
    }
}
