import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define Policy Network (LLM-like)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

# Define Value Network (for VAPO)
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.network(state)

# Placeholder Environment (e.g., math reasoning task)
class ReasoningEnv:
    def __init__(self, state_dim=100, action_dim=10):
        self.state_dim = state_dim
        self.action_dim = action_dim # Information about action space size

    def reset(self):
        # Returns a torch tensor
        return torch.randn(self.state_dim)

    def step(self, action): # action is an integer
        # Simulate reward (1 for correct, -1 for incorrect)
        reward = 1.0 if np.random.rand() > 0.5 else -1.0
        next_state = torch.randn(self.state_dim) # Next state is independent of current state/action
        done = np.random.rand() > 0.95 # Random termination
        return next_state, reward, done # next_state is tensor, reward float, done bool

# Hybrid RL Algorithm (DAPO + VAPO + ORZ)
class HybridRL:
    def __init__(self, state_dim, action_dim, hidden_dim=128, device_str='cpu'):
        self.device = torch.device(device_str)
        
        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.value = ValueNetwork(state_dim, hidden_dim).to(self.device)
        
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=3e-4)
        # Value function learning rate can sometimes be larger or use a different optimizer
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=1e-3) 
        
        self.env = ReasoningEnv(state_dim=state_dim, action_dim=action_dim)
        
        self.epsilon_low = 0.1   # PPO clip factor for lower bound (e.g., 1 - 0.1 = 0.9)
        self.epsilon_high = 0.3  # PPO clip factor for upper bound (e.g., 1 + 0.3 = 1.3)
                                 # Original comment "Clip-Higher" implies asymmetric clipping is intentional.
        
        self.gamma = 0.99        # Discount factor for future rewards
        self.lam = 0.95          # Lambda for Generalized Advantage Estimation (GAE)
        
        self.value_criterion = nn.MSELoss() # Mean Squared Error for value loss

    def compute_gae(self, rewards_tensor, values_tensor, dones_tensor, last_value_tensor):
        """
        Computes Generalized Advantage Estimation (GAE).
        Args:
            rewards_tensor (Tensor): Tensor of rewards [r_1, ..., r_T] collected from the environment.
            values_tensor (Tensor): Tensor of state value estimates [V(s_0), ..., V(s_{T-1})].
                                    These should be detached from the computation graph.
            dones_tensor (Tensor): Tensor of done flags [d_1, ..., d_T], where d_i indicates if state s_i is terminal.
            last_value_tensor (Tensor): Value estimate of the final state s_T (scalar tensor).
                                        This should be 0 if s_T is terminal, otherwise bootstrapped. Detached.
        Returns:
            Tensor: Computed advantages [A_0, ..., A_{T-1}].
        """
        advantages = torch.zeros_like(rewards_tensor, device=self.device)
        gae = 0.0 # Stores the GAE value at step t+1
        
        # Iterate backwards from T-1 down to 0
        for t in reversed(range(len(rewards_tensor))):
            # V_s_prime is the value of the state s_{t+1}
            if t == len(rewards_tensor) - 1: # This is the last transition (s_{T-1} -> s_T)
                v_s_prime = last_value_tensor  # V(s_T)
            else:
                v_s_prime = values_tensor[t+1] # V(s_{t+1})
            
            # dones_tensor[t] is d_{t+1} (is s_{t+1} terminal?)
            # rewards_tensor[t] is r_{t+1} (reward from (s_t, a_t) leading to s_{t+1})
            # values_tensor[t] is V(s_t)
            # Bellman residual (TD error)
            delta = rewards_tensor[t] + self.gamma * v_s_prime * (1.0 - dones_tensor[t].float()) - values_tensor[t]
            # GAE recursion: delta_t + gamma * lambda * (1-done_{t+1}) * GAE_{t+1}
            gae = delta + self.gamma * self.lam * (1.0 - dones_tensor[t].float()) * gae
            advantages[t] = gae
        return advantages

    def is_episode_valid_for_dapo(self, episode_rewards_list):
        """
        DAPO Dynamic Sampling: Checks if the episode's "accuracy" is strictly between 0 and 1.
        Accuracy is defined as the fraction of positive rewards in the episode.
        """
        if not episode_rewards_list: # Handle cases with no rewards
            return False
        
        positive_rewards_count = sum(1 for r in episode_rewards_list if r > 0)
        total_rewards_count = len(episode_rewards_list)
        
        if total_rewards_count == 0: # Should be caught by the first check, but for robustness
            return False

        accuracy = positive_rewards_count / total_rewards_count
        
        # DAPO rule: filter to ensure 0 < accuracy < 1
        return 0 < accuracy < 1

    def reward_zoning(self, state_tensor, action_int):
        """ 
        ORZ Placeholder: Model-based reward zoning. 
        This function would typically use a learned model to predict reward potential
        or adjust rewards based on state-action zones.
        """
        zone_reward = 0.0 
        return zone_reward

    def train(self, num_episodes=1000, max_steps_per_episode=50):
        for episode in range(num_episodes):
            # Lists to store trajectory data for the current episode
            states_list, actions_list, rewards_list, dones_list, old_log_probs_list = [], [], [], [], []
            current_episode_raw_rewards = [] # For DAPO check

            current_state_tensor = self.env.reset().to(self.device) # Initial state s_0
            episode_terminated_naturally = False # Flag to check if 'done' was from env or max_steps

            # --- Trajectory Collection ---
            for step_num in range(max_steps_per_episode):
                # Prepare state tensor for policy (add batch dim if needed)
                state_input_for_policy = current_state_tensor.unsqueeze(0) if current_state_tensor.dim() == 1 else current_state_tensor
                
                # Sample action from policy (and get its log probability) using current policy
                # Gradients are not tracked here as these are experiences for PPO update later.
                with torch.no_grad(): 
                    action_probs = self.policy(state_input_for_policy)
                    dist = torch.distributions.Categorical(probs=action_probs)
                    action_tensor = dist.sample() # Sampled action, e.g., tensor([3])
                    old_log_prob_tensor = dist.log_prob(action_tensor) # Log prob of sampled action

                # Store s_t, a_t, log_prob(a_t|s_t)
                states_list.append(current_state_tensor)
                actions_list.append(action_tensor) 
                old_log_probs_list.append(old_log_prob_tensor)

                action_int = action_tensor.item() # Convert to Python int for env.step
                next_state_tensor, reward_float, done_bool = self.env.step(action_int)
                next_state_tensor = next_state_tensor.to(self.device)

                # Apply ORZ: Adjust reward with model-based zoning
                zone_reward_float = self.reward_zoning(current_state_tensor, action_int)
                final_reward_float = reward_float + zone_reward_float
                
                # Store r_{t+1}, d_{t+1}
                rewards_list.append(torch.tensor([final_reward_float], dtype=torch.float32, device=self.device))
                # dones_list stores if s_{t+1} is terminal
                dones_list.append(torch.tensor([done_bool], dtype=torch.bool, device=self.device)) 
                current_episode_raw_rewards.append(final_reward_float)

                current_state_tensor = next_state_tensor
                episode_terminated_naturally = done_bool

                if episode_terminated_naturally:
                    break # End trajectory if environment signals done
            
            # --- DAPO Dynamic Sampling Check ---
            if not self.is_episode_valid_for_dapo(current_episode_raw_rewards):
                if (episode + 1) % 100 == 0 or episode == num_episodes - 1 or num_episodes < 100 : # Log occasional skips
                     print(f"Episode {episode + 1}: Skipped by DAPO (accuracy rule failed).")
                continue # Skip learning update for this episode

            if not states_list: # Should not happen if DAPO passed and episode had steps
                continue
                
            # --- Prepare Tensors for Learning ---
            # s_tensor: [s_0, ..., s_{T-1}], shape [T, state_dim]
            s_tensor = torch.stack(states_list)
            # a_tensor: [a_0, ..., a_{T-1}], shape [T]
            a_tensor = torch.stack(actions_list).squeeze() 
            if a_tensor.dim() == 0: a_tensor = a_tensor.unsqueeze(0) # Handle T=1 case
            # old_lp_tensor: [logp(a_0|s_0), ..., logp(a_{T-1}|s_{T-1})], shape [T]
            old_lp_tensor = torch.stack(old_log_probs_list).squeeze()
            if old_lp_tensor.dim() == 0: old_lp_tensor = old_lp_tensor.unsqueeze(0)

            # r_tensor: [r_1, ..., r_T], shape [T]
            r_tensor = torch.cat(rewards_list).squeeze()
            if r_tensor.dim() == 0: r_tensor = r_tensor.unsqueeze(0)
            # d_tensor: [d_1, ..., d_T], shape [T] (d_i means s_i is terminal)
            d_tensor = torch.cat(dones_list).squeeze()
            if d_tensor.dim() == 0: d_tensor = d_tensor.unsqueeze(0)

            # --- Compute Value Estimates, GAE, and Returns ---
            with torch.no_grad(): # Value estimates for GAE/Returns should be detached targets
                # V(s_0), ..., V(s_{T-1})
                values_pred_s_t = self.value(s_tensor).squeeze()
                if values_pred_s_t.dim() == 0: values_pred_s_t = values_pred_s_t.unsqueeze(0)

                # V(s_T): Value of the state after the last action (current_state_tensor is s_T)
                last_state_input_for_value = current_state_tensor.unsqueeze(0) if current_state_tensor.dim() == 1 else current_state_tensor
                
                # If episode ended naturally, V(s_T) = 0. Else, bootstrap from value network.
                last_value_s_T = (torch.tensor(0.0, device=self.device) 
                                  if episode_terminated_naturally 
                                  else self.value(last_state_input_for_value).squeeze().detach())
                if last_value_s_T.dim() > 0 : last_value_s_T = last_value_s_T.squeeze() # Ensure scalar tensor


            # Compute advantages A_t = GAE(s_t, a_t)
            # values_pred_s_t and last_value_s_T are already detached.
            adv_tensor = self.compute_gae(r_tensor, values_pred_s_t, d_tensor, last_value_s_T)
            
            # Compute returns R_t = A_t + V(s_t) as targets for value function update
            # values_pred_s_t is detached, so returns_tensor will also be detached.
            returns_tensor = adv_tensor + values_pred_s_t 

            # --- Policy Update (PPO-like) ---
            # Get log_probs for actions a_tensor under the *current* policy pi_theta_new
            new_action_probs = self.policy(s_tensor) # s_tensor is [s_0, ..., s_{T-1}]
            new_dist = torch.distributions.Categorical(probs=new_action_probs)
            new_log_probs = new_dist.log_prob(a_tensor) # a_tensor is [a_0, ..., a_{T-1}]

            # Ratio r_t(theta) = pi_theta_new(a_t|s_t) / pi_theta_old(a_t|s_t)
            # old_lp_tensor is log_prob from pi_theta_old, already detached.
            ratio = torch.exp(new_log_probs - old_lp_tensor) 
            
            # PPO's Clipped Surrogate Objective
            # adv_tensor is already effectively detached. Explicitly detaching is fine for clarity.
            detached_adv = adv_tensor.detach()
            surr1 = ratio * detached_adv
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon_low, 1.0 + self.epsilon_high) * detached_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # --- Value Function Update ---
            # Get current value predictions V_phi(s_t) for loss calculation
            current_values_for_loss = self.value(s_tensor).squeeze()
            if current_values_for_loss.dim() == 0: current_values_for_loss = current_values_for_loss.unsqueeze(0)
            
            # Value loss: MSE( V_phi(s_t) - R_t )
            # returns_tensor is already detached.
            value_loss = self.value_criterion(current_values_for_loss, returns_tensor) 

            # --- Optimization Step ---
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            # Optional: Gradient clipping for policy network
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer_policy.step()

            self.optimizer_value.zero_grad()
            value_loss.backward()
            # Optional: Gradient clipping for value network
            # torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0) 
            self.optimizer_value.step()

            # --- Logging ---
            if (episode + 1) % 10 == 0 or episode == num_episodes - 1 or num_episodes < 10:
                avg_reward_this_episode = r_tensor.mean().item() if r_tensor.numel() > 0 else float('nan')
                print(f"Episode {episode + 1}/{num_episodes}, Steps: {len(states_list)}, "
                      f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, "
                      f"Avg Reward: {avg_reward_this_episode:.2f}")
        
        print("Training finished.")

# Main execution block
if __name__ == "__main__":
    STATE_DIM = 100
    ACTION_DIM = 10
    NUM_EPISODES = 500 # Adjust as needed for training duration
    MAX_STEPS_PER_EPISODE = 50

    # Determine device
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # For Apple Silicon
        DEVICE = "mps"
        print("Using MPS (Apple Silicon GPU)")
    else:
        DEVICE = "cpu"
        print("Using CPU")
        
    hybrid_rl_agent = HybridRL(STATE_DIM, ACTION_DIM, device_str=DEVICE)
    hybrid_rl_agent.train(num_episodes=NUM_EPISODES, max_steps_per_episode=MAX_STEPS_PER_EPISODE)