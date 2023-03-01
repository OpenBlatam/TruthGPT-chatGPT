import { BigInt, Bytes, log } from '@graphprotocol/graph-ts';
import { log } from '@graphprotocol/graph-ts';
import { ProposalUpdated, DaoRegistered, DeploymentSubmitted } from '../generated/DAOCensus/DAOCensus';
import { Dao, Deployment, Proposal } from '../generated/schema';

export function handleDaoRegistered(event: DaoRegistered): void {
  log.info("test",[])
  let dao = Dao.load(event.params.DAOAddress.toHexString());
  if (!dao) {
    dao = new Dao(event.params.DAOAddress.toHexString());
  }
  dao.isValidated = true;
  dao.save();
}
const states = ['NOT_INITIALISATED', 'SUBMITED', 'VOTE_ONGOING', 'REFUSED', 'ACCEPTED'];
export function handleDeploymentSubmitted(event:DeploymentSubmitted):void{
  
    const deployment = new Deployment(event.params.IPSFHash);
    deployment.dao = event.params.DAOAddress.toHexString();
    deployment.timeCreated = event.block.timestamp;
    deployment.state = "SUBMITED"


  
  deployment.save()
}
export function handleProposalUpdated(event: ProposalUpdated): void {
  log.info("test",[])
  const proposal = new Proposal(event.transaction.hash.toHexString());
  proposal.time = event.block.timestamp;
  proposal.data = 'to change for the graph';
  proposal.result = states[event.params.State];

  let deployment = Deployment.load(event.params.IPFSHash);
  if (!deployment) {
    deployment = new Deployment(event.params.IPFSHash);
    deployment.dao = event.params.DAOaddress.toHexString();
    deployment.timeCreated = event.block.timestamp;
  }
  deployment.state = states[event.params.State];

  proposal.deployment = deployment.id;
  proposal.save();
  deployment.save();
}
