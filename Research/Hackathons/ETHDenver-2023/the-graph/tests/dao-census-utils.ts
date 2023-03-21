import { newMockEvent } from "matchstick-as"
import { ethereum } from "@graphprotocol/graph-ts"
import { ProposalUpdated } from "../generated/DAOCensus/DAOCensus"

export function createProposalUpdatedEvent(
  IPFSHash: string,
  State: i32
): ProposalUpdated {
  let proposalUpdatedEvent = changetype<ProposalUpdated>(newMockEvent())

  proposalUpdatedEvent.parameters = new Array()

  proposalUpdatedEvent.parameters.push(
    new ethereum.EventParam("IPFSHash", ethereum.Value.fromString(IPFSHash))
  )
  proposalUpdatedEvent.parameters.push(
    new ethereum.EventParam(
      "State",
      ethereum.Value.fromUnsignedBigInt(BigInt.fromI32(State))
    )
  )

  return proposalUpdatedEvent
}
