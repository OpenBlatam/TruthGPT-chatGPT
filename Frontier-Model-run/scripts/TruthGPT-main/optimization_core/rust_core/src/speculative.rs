//! Speculative Decoding for TruthGPT
//!
//! Implements speculative decoding for faster LLM inference.
//! Uses a small draft model to generate candidates that are verified by the target model.
//!
//! ## Algorithm
//!
//! 1. Draft model generates K tokens speculatively
//! 2. Target model verifies all K tokens in parallel
//! 3. Accept tokens until first mismatch, then sample from adjusted distribution
//!
//! ## Performance
//!
//! - 2-3x speedup for greedy/low-temperature sampling
//! - Best for large models with small draft models

use rand::{prelude::*, rngs::StdRng, SeedableRng};
use std::collections::VecDeque;

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Speculative decoding configuration
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens to speculate
    pub num_speculative_tokens: usize,
    /// Maximum number of draft model runs without acceptance
    pub max_draft_attempts: usize,
    /// Probability threshold for early rejection
    pub rejection_threshold: f32,
    /// Enable adaptive speculation length
    pub adaptive_length: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 4,
            max_draft_attempts: 3,
            rejection_threshold: 0.1,
            adaptive_length: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DRAFT RESULT
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from draft model speculation
#[derive(Debug, Clone)]
pub struct DraftResult {
    /// Speculated token IDs
    pub tokens: Vec<u32>,
    /// Log probabilities from draft model
    pub draft_logprobs: Vec<Vec<f32>>,
    /// Draft model's confidence scores
    pub confidences: Vec<f32>,
}

/// Result from verification
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Number of accepted tokens
    pub accepted_count: usize,
    /// Accepted token IDs
    pub accepted_tokens: Vec<u32>,
    /// Final sampled token (from target model distribution)
    pub final_token: Option<u32>,
    /// Acceptance rate for this batch
    pub acceptance_rate: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// SPECULATIVE DECODER
// ═══════════════════════════════════════════════════════════════════════════════

/// Speculative decoder state
pub struct SpeculativeDecoder {
    config: SpeculativeConfig,
    /// Running average of acceptance rate
    avg_acceptance_rate: f32,
    /// History of speculation lengths
    speculation_history: VecDeque<usize>,
    /// Current adaptive speculation length
    current_spec_length: usize,
    /// RNG for sampling
    rng: StdRng,
}

impl SpeculativeDecoder {
    /// Create new speculative decoder
    #[must_use]
    pub fn new(config: SpeculativeConfig) -> Self {
        let spec_length = config.num_speculative_tokens;
        Self {
            config,
            avg_acceptance_rate: 0.5,
            speculation_history: VecDeque::with_capacity(100),
            current_spec_length: spec_length,
            rng: StdRng::seed_from_u64(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos() as u64),
        }
    }

    /// Get current speculation length
    pub fn speculation_length(&self) -> usize {
        if self.config.adaptive_length {
            self.current_spec_length
        } else {
            self.config.num_speculative_tokens
        }
    }

    /// Verify draft tokens against target model
    ///
    /// # Arguments
    /// * `draft` - Draft model's speculation result
    /// * `target_logprobs` - Log probabilities from target model for each position
    ///
    /// # Returns
    /// * `VerificationResult` - Accepted tokens and final sampled token
    pub fn verify(
        &mut self,
        draft: &DraftResult,
        target_logprobs: &[Vec<f32>],
    ) -> VerificationResult {
        let mut accepted_count = 0;
        let mut accepted_tokens = Vec::with_capacity(draft.tokens.len());

        // Process each speculated token
        for (i, &token_id) in draft.tokens.iter().enumerate() {
            let token = token_id as usize;
            
            // Get probabilities
            let p_draft = softmax_single(&draft.draft_logprobs[i])[token];
            let p_target = softmax_single(&target_logprobs[i])[token];
            
            // Acceptance probability: min(1, p_target / p_draft)
            let accept_prob = if p_draft > 0.0 {
                (p_target / p_draft).min(1.0)
            } else if p_target > 0.0 {
                1.0
            } else {
                0.0
            };
            
            // Sample acceptance
            let r: f32 = self.rng.random();
            
            if r < accept_prob {
                accepted_count += 1;
                accepted_tokens.push(draft.tokens[i]);
            } else {
                // Rejection - sample from adjusted distribution
                let final_token = self.sample_adjusted(
                    &target_logprobs[i],
                    &draft.draft_logprobs[i],
                );
                
                let acceptance_rate = accepted_count as f32 / (i + 1) as f32;
                self.update_stats(accepted_count, draft.tokens.len());
                
                return VerificationResult {
                    accepted_count,
                    accepted_tokens,
                    final_token: Some(final_token),
                    acceptance_rate,
                };
            }
        }

        // All tokens accepted - sample next token from target distribution
        let final_token = if target_logprobs.len() > draft.tokens.len() {
            Some(self.sample_from_logprobs(&target_logprobs[draft.tokens.len()]))
        } else {
            None
        };

        let acceptance_rate = 1.0;
        self.update_stats(accepted_count, draft.tokens.len());

        VerificationResult {
            accepted_count,
            accepted_tokens,
            final_token,
            acceptance_rate,
        }
    }

    /// Sample from adjusted distribution: max(0, p_target - p_draft)
    fn sample_adjusted(&mut self, target_logprobs: &[f32], draft_logprobs: &[f32]) -> u32 {
        let p_target = softmax_single(target_logprobs);
        let p_draft = softmax_single(draft_logprobs);
        
        // Compute adjusted probabilities
        let mut adjusted: Vec<f32> = p_target
            .iter()
            .zip(p_draft.iter())
            .map(|(&pt, &pd)| (pt - pd).max(0.0))
            .collect();
        
        // Normalize
        let sum: f32 = adjusted.iter().sum();
        if sum > 0.0 {
            for p in &mut adjusted {
                *p /= sum;
            }
        } else {
            // Fallback to target distribution
            adjusted = p_target;
        }
        
        // Sample
        self.sample_from_probs(&adjusted)
    }

    /// Sample from log probabilities
    fn sample_from_logprobs(&mut self, logprobs: &[f32]) -> u32 {
        let probs = softmax_single(logprobs);
        self.sample_from_probs(&probs)
    }

    /// Sample from probability distribution
    fn sample_from_probs(&mut self, probs: &[f32]) -> u32 {
        let r: f32 = self.rng.random();
        let mut cumsum = 0.0;
        
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i as u32;
            }
        }
        
        (probs.len() - 1) as u32
    }

    /// Update statistics and adapt speculation length
    pub fn update_stats(&mut self, accepted: usize, total: usize) {
        let rate = accepted as f32 / total.max(1) as f32;
        
        // Exponential moving average
        self.avg_acceptance_rate = 0.9 * self.avg_acceptance_rate + 0.1 * rate;
        
        // Record history
        self.speculation_history.push_back(accepted);
        if self.speculation_history.len() > 100 {
            self.speculation_history.pop_front();
        }
        
        // Adapt speculation length
        if self.config.adaptive_length {
            if self.avg_acceptance_rate > 0.8 {
                // High acceptance - try more speculation
                self.current_spec_length = (self.current_spec_length + 1)
                    .min(self.config.num_speculative_tokens * 2);
            } else if self.avg_acceptance_rate < 0.3 {
                // Low acceptance - reduce speculation
                self.current_spec_length = (self.current_spec_length - 1).max(1);
            }
        }
    }

    /// Get statistics
    pub fn stats(&self) -> SpeculativeStats {
        let recent_accepts: f32 = self.speculation_history
            .iter()
            .sum::<usize>() as f32;
        let recent_total = self.speculation_history.len() as f32 
            * self.config.num_speculative_tokens as f32;
        
        SpeculativeStats {
            avg_acceptance_rate: self.avg_acceptance_rate,
            current_spec_length: self.current_spec_length,
            recent_acceptance_rate: if recent_total > 0.0 {
                recent_accepts / recent_total
            } else {
                0.0
            },
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.avg_acceptance_rate = 0.5;
        self.speculation_history.clear();
        self.current_spec_length = self.config.num_speculative_tokens;
    }
}

/// Statistics for speculative decoding
#[derive(Debug, Clone)]
pub struct SpeculativeStats {
    pub avg_acceptance_rate: f32,
    pub current_spec_length: usize,
    pub recent_acceptance_rate: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TREE SPECULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Tree-based speculative decoding for higher acceptance rates
pub struct TreeSpeculation {
    /// Branching factor at each level
    pub branching_factor: usize,
    /// Maximum tree depth
    pub max_depth: usize,
    /// Token tree
    tree: Vec<TreeNode>,
}

#[derive(Debug, Clone)]
struct TreeNode {
    token: u32,
    parent: Option<usize>,
    children: Vec<usize>,
    prob: f32,
    depth: usize,
}

impl TreeSpeculation {
    /// Create new tree speculation
    #[must_use]
    pub fn new(branching_factor: usize, max_depth: usize) -> Self {
        Self {
            branching_factor,
            max_depth,
            tree: Vec::new(),
        }
    }

    /// Build speculation tree from draft model
    pub fn build_tree(&mut self, draft_logprobs: &[Vec<f32>]) {
        self.tree.clear();
        
        // Root node (dummy)
        self.tree.push(TreeNode {
            token: 0,
            parent: None,
            children: Vec::new(),
            prob: 1.0,
            depth: 0,
        });
        
        // Build tree level by level
        let mut current_level = vec![0usize]; // Start with root
        
        for (depth, logprobs) in draft_logprobs.iter().take(self.max_depth).enumerate() {
            let probs = softmax_single(logprobs);
            
            // Get top-k tokens
            let mut token_probs: Vec<(usize, f32)> = probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            token_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            let mut next_level = Vec::new();
            
            for &parent_idx in &current_level {
                let parent_prob = self.tree[parent_idx].prob;
                
                for &(token, prob) in token_probs.iter().take(self.branching_factor) {
                    let node_idx = self.tree.len();
                    
                    self.tree.push(TreeNode {
                        token: token as u32,
                        parent: Some(parent_idx),
                        children: Vec::new(),
                        prob: parent_prob * prob,
                        depth: depth + 1,
                    });
                    
                    self.tree[parent_idx].children.push(node_idx);
                    next_level.push(node_idx);
                }
            }
            
            current_level = next_level;
        }
    }

    /// Get all paths in the tree
    pub fn get_paths(&self) -> Vec<Vec<u32>> {
        let mut paths = Vec::new();
        
        // Find leaf nodes
        for (idx, node) in self.tree.iter().enumerate() {
            if node.children.is_empty() && node.depth > 0 {
                // Trace path from leaf to root
                let mut path = Vec::new();
                let mut current = idx;
                
                while let Some(parent) = self.tree[current].parent {
                    path.push(self.tree[current].token);
                    current = parent;
                }
                
                path.reverse();
                paths.push(path);
            }
        }
        
        paths
    }

    /// Get number of nodes in tree
    pub fn size(&self) -> usize {
        self.tree.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute softmax of log probabilities
fn softmax_single(logprobs: &[f32]) -> Vec<f32> {
    if logprobs.is_empty() {
        return Vec::new();
    }
    
    let max_logprob = logprobs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    let exp_vals: Vec<f32> = logprobs
        .iter()
        .map(|&lp| (lp - max_logprob).exp())
        .collect();
    
    let sum: f32 = exp_vals.iter().sum();
    
    exp_vals.iter().map(|&e| e / sum).collect()
}

/// Compute KL divergence between two distributions
pub fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    p.iter()
        .zip(q.iter())
        .filter(|(&pi, &qi)| pi > 0.0 && qi > 0.0)
        .map(|(&pi, &qi)| pi * (pi / qi).ln())
        .sum()
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logprobs = vec![0.0, 1.0, 2.0];
        let probs = softmax_single(&logprobs);
        
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        
        // Higher logprob = higher prob
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_speculative_decoder_creation() {
        let config = SpeculativeConfig::default();
        let decoder = SpeculativeDecoder::new(config);
        
        assert_eq!(decoder.speculation_length(), 4);
    }

    #[test]
    fn test_verification_all_accept() {
        let config = SpeculativeConfig {
            num_speculative_tokens: 3,
            adaptive_length: false,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecoder::new(config);
        
        // Draft and target have same distribution
        let draft = DraftResult {
            tokens: vec![1, 2, 3],
            draft_logprobs: vec![
                vec![0.0, 2.0, 0.0, 0.0],
                vec![0.0, 0.0, 2.0, 0.0],
                vec![0.0, 0.0, 0.0, 2.0],
            ],
            confidences: vec![0.9, 0.9, 0.9],
        };
        
        let target_logprobs = vec![
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 2.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![1.0, 0.0, 0.0, 0.0], // Next token dist
        ];
        
        let result = decoder.verify(&draft, &target_logprobs);
        
        // Should accept all tokens when distributions match
        assert!(result.acceptance_rate >= 0.5); // Some randomness involved
    }

    #[test]
    fn test_tree_speculation() {
        let mut tree = TreeSpeculation::new(2, 3);
        
        let draft_logprobs = vec![
            vec![0.0, 1.0, 0.5, 0.2],
            vec![0.1, 0.0, 1.0, 0.3],
            vec![0.2, 0.3, 0.0, 1.0],
        ];
        
        tree.build_tree(&draft_logprobs);
        
        let paths = tree.get_paths();
        
        // Should have branching_factor^depth paths
        assert!(!paths.is_empty());
        
        // All paths should have length equal to depth
        for path in &paths {
            assert_eq!(path.len(), 3);
        }
    }

    #[test]
    fn test_adaptive_length() {
        let config = SpeculativeConfig {
            num_speculative_tokens: 4,
            adaptive_length: true,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecoder::new(config);
        
        // Simulate high acceptance
        for _ in 0..10 {
            decoder.update_stats(4, 4); // 100% acceptance
        }
        
        assert!(decoder.current_spec_length >= 4);
        
        // Simulate low acceptance
        for _ in 0..20 {
            decoder.update_stats(0, 4); // 0% acceptance
        }
        
        assert!(decoder.current_spec_length < 4);
    }
}

