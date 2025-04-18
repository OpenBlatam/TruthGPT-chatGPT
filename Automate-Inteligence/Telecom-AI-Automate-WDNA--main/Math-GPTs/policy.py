def optimize_policy(policy, states, reward_func, num_iterations):
  optimal_theta = None
  max_reward = -np.inf

  for _ in range(num_iterations):
    theta = random_initialization()
    expected_rewards = []

    for state in states:
      action = policy(state, theta)
      reward = reward_func(state, action)
      expected_rewards.append(reward)

    avg_reward = np.mean(expected_rewards)

    if avg_reward > max_reward:
      max_reward = avg_reward
      optimal_theta = theta

  return optimal_theta