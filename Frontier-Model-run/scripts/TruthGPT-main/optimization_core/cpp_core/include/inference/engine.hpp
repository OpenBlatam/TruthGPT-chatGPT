#pragma once

/**
 * @file engine.hpp
 * @brief Refactored inference engine with clean architecture
 * 
 * Features:
 * - Strategy pattern for sampling methods
 * - Builder pattern for configuration
 * - Clean separation of concerns
 */

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include "../common/types.hpp"

namespace optimization_core {
namespace inference {

// ============================================================================
// Configuration
// ============================================================================

struct GenerationConfig {
    i32 max_new_tokens = 128;
    f32 temperature = 1.0f;
    f32 top_p = 0.9f;
    i32 top_k = 50;
    f32 repetition_penalty = 1.0f;
    bool do_sample = true;
    i32 num_beams = 1;
    f32 length_penalty = 1.0f;
    i32 eos_token_id = -1;
    i32 pad_token_id = -1;
    
    // Builder pattern
    GenerationConfig& with_max_tokens(i32 n) { max_new_tokens = n; return *this; }
    GenerationConfig& with_temperature(f32 t) { temperature = t; return *this; }
    GenerationConfig& with_top_p(f32 p) { top_p = p; return *this; }
    GenerationConfig& with_top_k(i32 k) { top_k = k; return *this; }
    GenerationConfig& with_repetition_penalty(f32 p) { repetition_penalty = p; return *this; }
    GenerationConfig& with_sampling(bool s) { do_sample = s; return *this; }
    GenerationConfig& with_beam_search(i32 beams, f32 len_pen = 1.0f) {
        num_beams = beams;
        length_penalty = len_pen;
        return *this;
    }
    GenerationConfig& with_eos(i32 id) { eos_token_id = id; return *this; }
    
    // Preset configurations
    static GenerationConfig greedy() {
        return GenerationConfig().with_sampling(false).with_temperature(1.0f);
    }
    
    static GenerationConfig sampling(f32 temp = 0.8f, f32 top_p = 0.9f) {
        return GenerationConfig()
            .with_sampling(true)
            .with_temperature(temp)
            .with_top_p(top_p);
    }
    
    static GenerationConfig beam(i32 num_beams = 4) {
        return GenerationConfig()
            .with_sampling(false)
            .with_beam_search(num_beams);
    }
};

// ============================================================================
// Generation Result
// ============================================================================

struct GenerationResult {
    std::vector<i32> token_ids;
    std::vector<f32> logprobs;
    f32 total_logprob = 0.0f;
    f64 generation_time_ms = 0.0;
    i32 tokens_generated = 0;
    
    f64 tokens_per_second() const {
        return generation_time_ms > 0 
            ? tokens_generated / (generation_time_ms / 1000.0) : 0.0;
    }
};

// ============================================================================
// Sampling Strategies
// ============================================================================

namespace sampling {

/**
 * @brief Greedy sampling - select highest probability token
 */
inline i32 greedy(const std::vector<f32>& logits) {
    return std::max_element(logits.begin(), logits.end()) - logits.begin();
}

/**
 * @brief Apply temperature scaling
 */
inline void apply_temperature(std::vector<f32>& logits, f32 temperature) {
    if (temperature <= 0 || temperature == 1.0f) return;
    for (f32& l : logits) {
        l /= temperature;
    }
}

/**
 * @brief Apply top-k filtering
 */
inline void apply_top_k(std::vector<f32>& logits, i32 k) {
    if (k <= 0 || k >= static_cast<i32>(logits.size())) return;
    
    // Find k-th largest
    std::vector<f32> sorted = logits;
    std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end(),
                     std::greater<f32>());
    f32 threshold = sorted[k - 1];
    
    // Zero out values below threshold
    for (f32& l : logits) {
        if (l < threshold) l = -std::numeric_limits<f32>::infinity();
    }
}

/**
 * @brief Apply top-p (nucleus) filtering
 */
inline void apply_top_p(std::vector<f32>& probs, f32 p) {
    if (p >= 1.0f) return;
    
    // Sort indices by probability
    std::vector<usize> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&probs](usize a, usize b) { return probs[a] > probs[b]; });
    
    // Find cutoff
    f32 cumsum = 0.0f;
    usize cutoff = indices.size();
    for (usize i = 0; i < indices.size(); ++i) {
        cumsum += probs[indices[i]];
        if (cumsum > p) {
            cutoff = i + 1;
            break;
        }
    }
    
    // Zero out values after cutoff
    for (usize i = cutoff; i < indices.size(); ++i) {
        probs[indices[i]] = 0.0f;
    }
}

/**
 * @brief Apply repetition penalty
 */
inline void apply_repetition_penalty(std::vector<f32>& logits,
                                      const std::vector<i32>& prev_tokens,
                                      f32 penalty) {
    if (penalty == 1.0f) return;
    
    for (i32 token : prev_tokens) {
        if (token >= 0 && token < static_cast<i32>(logits.size())) {
            if (logits[token] > 0) {
                logits[token] /= penalty;
            } else {
                logits[token] *= penalty;
            }
        }
    }
}

/**
 * @brief Softmax transformation
 */
inline std::vector<f32> softmax(const std::vector<f32>& logits) {
    std::vector<f32> probs = logits;
    f32 max_val = *std::max_element(probs.begin(), probs.end());
    
    f32 sum = 0.0f;
    for (f32& p : probs) {
        p = std::exp(p - max_val);
        sum += p;
    }
    
    if (sum > 0) {
        for (f32& p : probs) {
            p /= sum;
        }
    }
    
    return probs;
}

} // namespace sampling

// ============================================================================
// Token Sampler
// ============================================================================

class TokenSampler {
public:
    explicit TokenSampler(u32 seed = 42) : rng_(seed) {}
    
    /**
     * @brief Sample next token from logits
     */
    i32 sample(std::vector<f32> logits, const GenerationConfig& config) {
        // Apply temperature
        sampling::apply_temperature(logits, config.temperature);
        
        // Convert to probabilities
        auto probs = sampling::softmax(logits);
        
        if (!config.do_sample) {
            return sampling::greedy(probs);
        }
        
        // Apply top-k
        if (config.top_k > 0) {
            sampling::apply_top_k(probs, config.top_k);
        }
        
        // Apply top-p
        if (config.top_p < 1.0f) {
            sampling::apply_top_p(probs, config.top_p);
        }
        
        // Renormalize
        f32 sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        if (sum > 0) {
            for (f32& p : probs) p /= sum;
        }
        
        // Sample
        return sample_from_distribution(probs);
    }
    
    void set_seed(u32 seed) { rng_.seed(seed); }

private:
    std::mt19937 rng_;
    
    i32 sample_from_distribution(const std::vector<f32>& probs) {
        std::uniform_real_distribution<f32> dist(0.0f, 1.0f);
        f32 r = dist(rng_);
        
        f32 cumsum = 0.0f;
        for (usize i = 0; i < probs.size(); ++i) {
            cumsum += probs[i];
            if (r < cumsum) {
                return static_cast<i32>(i);
            }
        }
        
        return static_cast<i32>(probs.size() - 1);
    }
};

// ============================================================================
// Beam Search
// ============================================================================

struct BeamCandidate {
    std::vector<i32> tokens;
    f32 score = 0.0f;
    bool finished = false;
    
    bool operator<(const BeamCandidate& other) const {
        return score < other.score;
    }
};

class BeamSearch {
public:
    BeamSearch(i32 num_beams, f32 length_penalty = 1.0f)
        : num_beams_(num_beams), length_penalty_(length_penalty) {}
    
    void initialize(const std::vector<i32>& input_ids) {
        beams_.clear();
        beams_.push_back({input_ids, 0.0f, false});
    }
    
    void step(const std::vector<std::vector<f32>>& beam_logits, i32 eos_id) {
        std::vector<BeamCandidate> candidates;
        
        for (usize beam_idx = 0; beam_idx < beams_.size(); ++beam_idx) {
            if (beams_[beam_idx].finished) {
                candidates.push_back(beams_[beam_idx]);
                continue;
            }
            
            const auto& logits = beam_logits[beam_idx];
            auto probs = sampling::softmax(logits);
            
            // Get top candidates
            std::vector<std::pair<f32, i32>> top;
            for (i32 i = 0; i < static_cast<i32>(probs.size()); ++i) {
                top.emplace_back(probs[i], i);
            }
            std::partial_sort(top.begin(), top.begin() + num_beams_ * 2, top.end(),
                            std::greater<std::pair<f32, i32>>());
            
            for (i32 k = 0; k < std::min(num_beams_ * 2, static_cast<i32>(top.size())); ++k) {
                BeamCandidate cand;
                cand.tokens = beams_[beam_idx].tokens;
                cand.tokens.push_back(top[k].second);
                cand.score = beams_[beam_idx].score + std::log(top[k].first + 1e-10f);
                cand.finished = (top[k].second == eos_id);
                
                // Length penalty
                f32 penalty = std::pow(static_cast<f32>(cand.tokens.size()), length_penalty_);
                cand.score /= penalty;
                
                candidates.push_back(cand);
            }
        }
        
        // Keep top beams
        std::partial_sort(candidates.begin(),
                         candidates.begin() + std::min(num_beams_, static_cast<i32>(candidates.size())),
                         candidates.end(),
                         [](const BeamCandidate& a, const BeamCandidate& b) {
                             return a.score > b.score;
                         });
        
        beams_.assign(candidates.begin(),
                     candidates.begin() + std::min(num_beams_, static_cast<i32>(candidates.size())));
    }
    
    bool is_finished() const {
        return std::all_of(beams_.begin(), beams_.end(),
                          [](const BeamCandidate& b) { return b.finished; });
    }
    
    std::vector<i32> best_sequence() const {
        return beams_.empty() ? std::vector<i32>{} : beams_[0].tokens;
    }
    
    std::vector<std::vector<i32>> all_sequences() const {
        std::vector<std::vector<i32>> seqs;
        for (const auto& beam : beams_) {
            seqs.push_back(beam.tokens);
        }
        return seqs;
    }

private:
    i32 num_beams_;
    f32 length_penalty_;
    std::vector<BeamCandidate> beams_;
};

// ============================================================================
// Inference Engine
// ============================================================================

class InferenceEngine {
public:
    using ForwardFn = std::function<std::vector<f32>(const std::vector<i32>&)>;
    using BatchForwardFn = std::function<std::vector<std::vector<f32>>(
        const std::vector<std::vector<i32>>&)>;
    
    explicit InferenceEngine(u32 seed = 42) : sampler_(seed) {}
    
    /**
     * @brief Generate tokens autoregressively
     */
    GenerationResult generate(
        const std::vector<i32>& input_ids,
        ForwardFn forward_fn,
        const GenerationConfig& config = {}
    ) {
        Timer timer;
        GenerationResult result;
        result.token_ids = input_ids;
        
        std::vector<i32> prev_tokens = input_ids;
        
        for (i32 i = 0; i < config.max_new_tokens; ++i) {
            // Get logits
            std::vector<f32> logits = forward_fn(result.token_ids);
            
            // Apply repetition penalty
            if (config.repetition_penalty != 1.0f) {
                sampling::apply_repetition_penalty(logits, prev_tokens, 
                                                   config.repetition_penalty);
            }
            
            // Sample
            i32 next_token = sampler_.sample(logits, config);
            
            // Check EOS
            if (next_token == config.eos_token_id) {
                break;
            }
            
            result.token_ids.push_back(next_token);
            prev_tokens.push_back(next_token);
            
            // Log probability
            auto probs = sampling::softmax(logits);
            f32 logprob = std::log(probs[next_token] + 1e-10f);
            result.logprobs.push_back(logprob);
            result.total_logprob += logprob;
        }
        
        result.generation_time_ms = timer.elapsed_ms();
        result.tokens_generated = result.token_ids.size() - input_ids.size();
        
        return result;
    }
    
    /**
     * @brief Generate with beam search
     */
    std::vector<i32> generate_beam(
        const std::vector<i32>& input_ids,
        BatchForwardFn forward_fn,
        const GenerationConfig& config = {}
    ) {
        BeamSearch beam_search(config.num_beams, config.length_penalty);
        beam_search.initialize(input_ids);
        
        for (i32 i = 0; i < config.max_new_tokens; ++i) {
            auto sequences = beam_search.all_sequences();
            auto beam_logits = forward_fn(sequences);
            
            beam_search.step(beam_logits, config.eos_token_id);
            
            if (beam_search.is_finished()) {
                break;
            }
        }
        
        return beam_search.best_sequence();
    }
    
    void set_seed(u32 seed) { sampler_.set_seed(seed); }

private:
    TokenSampler sampler_;
};

// ============================================================================
// Factory
// ============================================================================

inline std::unique_ptr<InferenceEngine> create_engine(u32 seed = 42) {
    return std::make_unique<InferenceEngine>(seed);
}

} // namespace inference
} // namespace optimization_core












