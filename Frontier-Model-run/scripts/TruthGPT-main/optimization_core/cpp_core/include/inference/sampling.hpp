#pragma once

/**
 * @file sampling.hpp
 * @brief High-performance token sampling strategies
 * 
 * Implements various sampling methods:
 * - Greedy (argmax)
 * - Top-K sampling
 * - Top-P (nucleus) sampling
 * - Temperature scaling
 * - Repetition penalty
 * - Min-P sampling
 * - Typical sampling
 * - Mirostat sampling
 */

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
#include <queue>

namespace truthgpt {
namespace inference {

// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLING CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Sampling configuration
 */
struct SamplingConfig {
    float temperature = 1.0f;
    int top_k = 50;
    float top_p = 0.9f;
    float min_p = 0.0f;
    float typical_p = 1.0f;
    float repetition_penalty = 1.0f;
    int repetition_window = 64;
    bool do_sample = true;
    
    // Mirostat
    bool use_mirostat = false;
    float mirostat_tau = 5.0f;
    float mirostat_eta = 0.1f;
    float mirostat_mu = 2.0f * 5.0f;  // 2 * tau
    
    unsigned int seed = 42;
};

// ═══════════════════════════════════════════════════════════════════════════════
// TOKEN SAMPLER
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief High-performance token sampler
 */
class TokenSampler {
public:
    explicit TokenSampler(const SamplingConfig& config = SamplingConfig())
        : config_(config), rng_(config.seed), mirostat_mu_(config.mirostat_mu) {}

    /**
     * @brief Sample next token from logits
     * @param logits Raw logits from model [vocab_size]
     * @param past_tokens Previously generated tokens for repetition penalty
     * @return Sampled token ID
     */
    int sample(
        std::vector<float>& logits,
        const std::vector<int>& past_tokens = {}
    ) {
        // Apply repetition penalty
        if (config_.repetition_penalty != 1.0f && !past_tokens.empty()) {
            apply_repetition_penalty(logits, past_tokens);
        }
        
        // Greedy decoding
        if (!config_.do_sample) {
            return std::max_element(logits.begin(), logits.end()) - logits.begin();
        }
        
        // Mirostat sampling
        if (config_.use_mirostat) {
            return sample_mirostat(logits);
        }
        
        // Apply temperature
        if (config_.temperature != 1.0f && config_.temperature > 0) {
            apply_temperature(logits, config_.temperature);
        }
        
        // Convert to probabilities
        softmax_inplace(logits);
        
        // Apply Min-P filtering
        if (config_.min_p > 0.0f) {
            apply_min_p(logits, config_.min_p);
        }
        
        // Apply typical sampling
        if (config_.typical_p < 1.0f) {
            apply_typical(logits, config_.typical_p);
        }
        
        // Apply Top-K filtering
        if (config_.top_k > 0 && config_.top_k < static_cast<int>(logits.size())) {
            apply_top_k(logits, config_.top_k);
        }
        
        // Apply Top-P filtering
        if (config_.top_p < 1.0f) {
            apply_top_p(logits, config_.top_p);
        }
        
        // Renormalize and sample
        renormalize(logits);
        return multinomial_sample(logits);
    }

    /**
     * @brief Set random seed
     */
    void set_seed(unsigned int seed) {
        rng_.seed(seed);
    }

    /**
     * @brief Reset mirostat state
     */
    void reset_mirostat() {
        mirostat_mu_ = 2.0f * config_.mirostat_tau;
    }

    /**
     * @brief Get current configuration
     */
    const SamplingConfig& config() const { return config_; }
    
    /**
     * @brief Update configuration
     */
    void set_config(const SamplingConfig& config) {
        config_ = config;
        rng_.seed(config.seed);
    }

private:
    SamplingConfig config_;
    std::mt19937 rng_;
    float mirostat_mu_;

    /**
     * @brief Apply temperature scaling
     */
    void apply_temperature(std::vector<float>& logits, float temperature) {
        float inv_temp = 1.0f / temperature;
        for (float& l : logits) {
            l *= inv_temp;
        }
    }

    /**
     * @brief Apply softmax in-place
     */
    void softmax_inplace(std::vector<float>& logits) {
        float max_logit = *std::max_element(logits.begin(), logits.end());
        
        float sum = 0.0f;
        for (float& l : logits) {
            l = std::exp(l - max_logit);
            sum += l;
        }
        
        float inv_sum = 1.0f / sum;
        for (float& l : logits) {
            l *= inv_sum;
        }
    }

    /**
     * @brief Apply repetition penalty
     */
    void apply_repetition_penalty(
        std::vector<float>& logits,
        const std::vector<int>& past_tokens
    ) {
        size_t window_start = past_tokens.size() > static_cast<size_t>(config_.repetition_window)
            ? past_tokens.size() - config_.repetition_window
            : 0;
        
        for (size_t i = window_start; i < past_tokens.size(); ++i) {
            int token = past_tokens[i];
            if (token >= 0 && token < static_cast<int>(logits.size())) {
                if (logits[token] > 0) {
                    logits[token] /= config_.repetition_penalty;
                } else {
                    logits[token] *= config_.repetition_penalty;
                }
            }
        }
    }

    /**
     * @brief Apply Top-K filtering
     */
    void apply_top_k(std::vector<float>& probs, int k) {
        // Find k-th largest element
        std::vector<float> sorted = probs;
        std::nth_element(sorted.begin(), sorted.begin() + k, sorted.end(),
                         std::greater<float>());
        float threshold = sorted[k];
        
        // Zero out everything below threshold
        for (float& p : probs) {
            if (p < threshold) {
                p = 0.0f;
            }
        }
    }

    /**
     * @brief Apply Top-P (nucleus) filtering
     */
    void apply_top_p(std::vector<float>& probs, float p) {
        // Sort indices by probability
        std::vector<size_t> indices(probs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return probs[a] > probs[b];
        });
        
        // Find cutoff
        float cumsum = 0.0f;
        size_t cutoff_idx = indices.size();
        
        for (size_t i = 0; i < indices.size(); ++i) {
            cumsum += probs[indices[i]];
            if (cumsum > p) {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        // Zero out tokens beyond cutoff
        for (size_t i = cutoff_idx; i < indices.size(); ++i) {
            probs[indices[i]] = 0.0f;
        }
    }

    /**
     * @brief Apply Min-P filtering
     */
    void apply_min_p(std::vector<float>& probs, float min_p) {
        float max_prob = *std::max_element(probs.begin(), probs.end());
        float threshold = max_prob * min_p;
        
        for (float& p : probs) {
            if (p < threshold) {
                p = 0.0f;
            }
        }
    }

    /**
     * @brief Apply typical sampling
     */
    void apply_typical(std::vector<float>& probs, float typical_p) {
        // Calculate entropy
        float entropy = 0.0f;
        for (float p : probs) {
            if (p > 0) {
                entropy -= p * std::log(p);
            }
        }
        
        // Calculate "typicality" for each token
        std::vector<std::pair<float, size_t>> typical_scores;
        typical_scores.reserve(probs.size());
        
        for (size_t i = 0; i < probs.size(); ++i) {
            if (probs[i] > 0) {
                float neg_log_prob = -std::log(probs[i]);
                float score = std::abs(neg_log_prob - entropy);
                typical_scores.emplace_back(score, i);
            }
        }
        
        // Sort by typicality (lower = more typical)
        std::sort(typical_scores.begin(), typical_scores.end());
        
        // Keep tokens until cumsum > typical_p
        float cumsum = 0.0f;
        std::vector<bool> keep(probs.size(), false);
        
        for (const auto& [score, idx] : typical_scores) {
            if (cumsum >= typical_p) break;
            keep[idx] = true;
            cumsum += probs[idx];
        }
        
        // Zero out non-typical tokens
        for (size_t i = 0; i < probs.size(); ++i) {
            if (!keep[i]) {
                probs[i] = 0.0f;
            }
        }
    }

    /**
     * @brief Mirostat sampling (maintains target perplexity)
     */
    int sample_mirostat(std::vector<float>& logits) {
        // Convert to probabilities
        softmax_inplace(logits);
        
        // Sort tokens by probability
        std::vector<std::pair<float, int>> sorted_probs;
        sorted_probs.reserve(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            sorted_probs.emplace_back(logits[i], static_cast<int>(i));
        }
        std::sort(sorted_probs.begin(), sorted_probs.end(), std::greater<>());
        
        // Calculate truncation point based on mu
        float cumsum = 0.0f;
        int truncation_idx = 0;
        
        for (size_t i = 0; i < sorted_probs.size(); ++i) {
            cumsum += sorted_probs[i].first;
            if (cumsum > std::exp(-mirostat_mu_)) {
                truncation_idx = static_cast<int>(i);
                break;
            }
        }
        
        // Renormalize truncated distribution
        float norm_sum = 0.0f;
        for (int i = 0; i <= truncation_idx; ++i) {
            norm_sum += sorted_probs[i].first;
        }
        
        // Sample from truncated distribution
        std::uniform_real_distribution<float> dist(0.0f, norm_sum);
        float r = dist(rng_);
        
        cumsum = 0.0f;
        int sampled_token = sorted_probs[0].second;
        float sampled_prob = sorted_probs[0].first;
        
        for (int i = 0; i <= truncation_idx; ++i) {
            cumsum += sorted_probs[i].first;
            if (r < cumsum) {
                sampled_token = sorted_probs[i].second;
                sampled_prob = sorted_probs[i].first;
                break;
            }
        }
        
        // Update mu
        float surprise = -std::log2(sampled_prob);
        float error = surprise - config_.mirostat_tau;
        mirostat_mu_ -= config_.mirostat_eta * error;
        
        return sampled_token;
    }

    /**
     * @brief Renormalize probabilities
     */
    void renormalize(std::vector<float>& probs) {
        float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        if (sum > 0) {
            float inv_sum = 1.0f / sum;
            for (float& p : probs) {
                p *= inv_sum;
            }
        }
    }

    /**
     * @brief Sample from multinomial distribution
     */
    int multinomial_sample(const std::vector<float>& probs) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_);
        
        float cumsum = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i) {
            cumsum += probs[i];
            if (r < cumsum) {
                return static_cast<int>(i);
            }
        }
        
        return static_cast<int>(probs.size() - 1);
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// BEAM SEARCH
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Beam hypothesis
 */
struct BeamHypothesis {
    std::vector<int> tokens;
    float score;
    bool is_finished;
    
    bool operator<(const BeamHypothesis& other) const {
        return score < other.score;  // Higher score = better
    }
};

/**
 * @brief Beam search decoder
 */
class BeamSearchDecoder {
public:
    BeamSearchDecoder(int num_beams, float length_penalty = 1.0f)
        : num_beams_(num_beams), length_penalty_(length_penalty) {}

    /**
     * @brief Initialize beams with input
     */
    void init(const std::vector<int>& input_ids) {
        beams_.clear();
        finished_.clear();
        
        BeamHypothesis initial;
        initial.tokens = input_ids;
        initial.score = 0.0f;
        initial.is_finished = false;
        
        for (int i = 0; i < num_beams_; ++i) {
            beams_.push_back(initial);
        }
    }

    /**
     * @brief Update beams with next token probabilities
     * @param all_logprobs Log probabilities for each beam [num_beams, vocab_size]
     * @param eos_token_id End of sequence token
     */
    void step(
        const std::vector<std::vector<float>>& all_logprobs,
        int eos_token_id
    ) {
        // Collect all candidates
        std::priority_queue<std::pair<float, std::pair<int, int>>> candidates;
        
        for (int beam_idx = 0; beam_idx < static_cast<int>(beams_.size()); ++beam_idx) {
            if (beams_[beam_idx].is_finished) continue;
            
            const auto& logprobs = all_logprobs[beam_idx];
            
            for (int token_idx = 0; token_idx < static_cast<int>(logprobs.size()); ++token_idx) {
                float new_score = beams_[beam_idx].score + logprobs[token_idx];
                candidates.push({new_score, {beam_idx, token_idx}});
            }
        }

        // Select top num_beams candidates
        std::vector<BeamHypothesis> new_beams;
        
        while (new_beams.size() < static_cast<size_t>(num_beams_) && !candidates.empty()) {
            auto [score, indices] = candidates.top();
            candidates.pop();
            
            auto [beam_idx, token_idx] = indices;
            
            BeamHypothesis new_hyp;
            new_hyp.tokens = beams_[beam_idx].tokens;
            new_hyp.tokens.push_back(token_idx);
            new_hyp.score = score;
            new_hyp.is_finished = (token_idx == eos_token_id);
            
            if (new_hyp.is_finished) {
                // Apply length penalty
                float length_factor = std::pow(
                    static_cast<float>(new_hyp.tokens.size()), 
                    length_penalty_
                );
                new_hyp.score /= length_factor;
                finished_.push_back(new_hyp);
            } else {
                new_beams.push_back(new_hyp);
            }
        }
        
        beams_ = std::move(new_beams);
    }

    /**
     * @brief Get best finished hypothesis
     */
    const BeamHypothesis& best() const {
        if (finished_.empty()) {
            return *std::max_element(beams_.begin(), beams_.end());
        }
        return *std::max_element(finished_.begin(), finished_.end());
    }

    /**
     * @brief Get all beams
     */
    const std::vector<BeamHypothesis>& beams() const { return beams_; }

    /**
     * @brief Check if all beams are finished
     */
    bool all_finished() const {
        return beams_.empty() || 
               std::all_of(beams_.begin(), beams_.end(),
                          [](const auto& b) { return b.is_finished; });
    }

private:
    int num_beams_;
    float length_penalty_;
    std::vector<BeamHypothesis> beams_;
    std::vector<BeamHypothesis> finished_;
};

} // namespace inference
} // namespace truthgpt




