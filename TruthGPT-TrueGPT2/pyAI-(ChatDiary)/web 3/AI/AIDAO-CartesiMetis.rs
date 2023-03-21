// SPDX-License-Identifier: MIT
use cartesi_math::CartesiMath;
use riscv::RISCVMachine;
use metis_staking::Staking;
use metis_token::ERC20;
use meta_voting::IVoting;
use openzeppelin_access_control::Ownable;
use foundry::prelude::*;

struct AIDAO {
    dao: Box<dyn IVoting>,
    token: Box<dyn ERC20>,
    foundry: Foundry<web3::transports::WebSocket>,
    vote_threshold: u256,
}

impl AIDAO {
    pub async fn new(
        dao_address: Address,
        token_address: Address,
        foundry_url: &str,
        vote_threshold: u256,
    ) -> Self {
        let dao = IVoting::at(dao_address);
        let token = ERC20::at(token_address);
        let foundry = Foundry::new(foundry_url).await.unwrap();
        Self {
            dao,
            token,
            foundry,
            vote_threshold,
        }
    }

    pub async fn vote_on_proposal(&mut self, proposal_id: u256, input_data: Vec<u8>) {
        let address = self.address();
        let staked_balance = Staking::balance_of(&address).call(&mut self.foundry).await.unwrap();

        assert!(self.dao.is_active(&address).call(&mut self.foundry).await.unwrap(), "Contract is not a member of the DAO");
        assert!(staked_balance >= self.vote_threshold, "Contract does not have enough staked tokens");

        // Compute the input hash using CartesiMath
        let input_hash = input_data.keccak256();

        // Load the input hash into the RISC-V machine
        let mut machine = RISCVMachine::new();
        machine.initialize(0x2000);
        machine.store_data(input_hash, 0);
        machine.run();

        // Get the result from the RISC-V machine
        let result = machine.load_word(0);

        // Normalize the result to a vote value between 0 and 100
        let vote = result.modulo(101);

        self.dao
            .vote(proposal_id, vote)
            .send(&mut self.foundry)
            .await
            .unwrap();
    }
}

impl Contract for AIDAO {}
impl Staking for AIDAO {}
impl Ownable for AIDAO {}
