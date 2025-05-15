Component	Description
Reward Function	Can be binary (correct/wrong), scalar from reward models, or heuristic scores
Group Sampling	Multiple completions generated per prompt
Advantage Computation	Relative to group mean and standardized by group std deviation
Policy Update Objective	PPO-style clipped surrogate with KL penalty, using group-relative advantages
Stability Mechanism	KL penalty anchors policy to a frozen reference, preventing catastrophic forgetting
Effectiveness	Amplifies probability of success, especially with verifiable rewards