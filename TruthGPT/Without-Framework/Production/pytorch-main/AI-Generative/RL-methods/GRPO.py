import torch
import torch.nn.functional as F

def get_token_log_probs(model, input_ids, attention_mask):
    """
    Compute log-probabilities of tokens under the given model.
    Returns: Tensor of shape (batch_size, seq_len)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    # Gather log probs of actual tokens
    token_logp = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    return token_logp


def compute_probability_ratio(curr_logp, old_logp):
    """
    Compute the probability ratio between current and old policies: r = exp(curr - old).
    """
    return torch.exp(curr_logp - old_logp)


def compute_clipped_ratio(ratio, epsilon):
    """
    Clip the ratio to [1-epsilon, 1+epsilon]
    """
    return torch.clamp(ratio, 1 - epsilon, 1 + epsilon)


def compute_kl_penalty(curr_logp, ref_logp):
    """
    Compute per-token KL divergence penalty term: exp(diff) - diff - 1, where diff = ref - curr
    """
    diff = ref_logp - curr_logp
    return torch.exp(diff) - diff - 1


def compute_surrogate_advantage(ratio, clipped_ratio, advantages):
    """
    Compute the surrogate advantage loss per token using PPO-style clipping.
    """
    loss1 = ratio * advantages
    loss2 = clipped_ratio * advantages
    return torch.min(loss1, loss2)


def compute_per_token_loss(ratio, clipped_ratio, advantages, kl_penalty, beta):
    """
    Combine surrogate advantage and KL penalty per token: -(adv_loss - beta * kl_penalty)
    """
    adv_loss = compute_surrogate_advantage(ratio, clipped_ratio, advantages)
    return - (adv_loss - beta * kl_penalty)


def compute_grpo_loss(
    current_model,
    old_model,
    ref_model,
    input_ids,
    attention_mask,
    advantages,
    beta=1.0,
    epsilon=0.2,
):
    """
    Compute the GRPO loss for a batch of sequences.
    """
    # Compute log-probs
    curr_logp = get_token_log_probs(current_model, input_ids, attention_mask)
    with torch.no_grad():
        old_logp = get_token_log_probs(old_model, input_ids, attention_mask)
        ref_logp = get_token_log_probs(ref_model, input_ids, attention_mask)

    # Ratios and penalties
    ratio = compute_probability_ratio(curr_logp, old_logp)
    clipped = compute_clipped_ratio(ratio, epsilon)
    kl_penalty = compute_kl_penalty(curr_logp, ref_logp)

    # Per-token loss
    per_token_loss = compute_per_token_loss(ratio, clipped, advantages, kl_penalty, beta)

    # Mask and average
    mask = attention_mask.float()
    # Avoid division by zero
    lengths = mask.sum(dim=1).clamp(min=1)
    loss_per_seq = (per_token_loss * mask).sum(dim=1) / lengths
    return loss_per_seq.mean()

# Example usage:
# loss = compute_grpo_loss(model, old_model, ref_model,
#                         input_ids, attention_mask, advantages,
#                         beta=0.05, epsilon=0.1)