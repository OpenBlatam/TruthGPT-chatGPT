/**
 * @file example_inference.cpp
 * @brief Practical example of using the inference engine
 * 
 * This example demonstrates:
 * - Token sampling strategies (greedy, top-k, top-p)
 * - Temperature scaling
 * - Beam search
 * - Full text generation pipeline
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <map>

#include "../include/optimization_core.hpp"

using namespace optimization_core;
using namespace optimization_core::inference;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n\n";
}

// Simple vocabulary for demonstration
std::map<i32, std::string> vocab = {
    {0, "<eos>"},
    {1, "The"},
    {2, "quick"},
    {3, "brown"},
    {4, "fox"},
    {5, "jumps"},
    {6, "over"},
    {7, "the"},
    {8, "lazy"},
    {9, "dog"},
    {10, "."},
    {11, "A"},
    {12, "cat"},
    {13, "runs"},
    {14, "fast"},
    {15, "and"},
    {16, "plays"},
    {17, "with"},
    {18, "a"},
    {19, "ball"},
};

std::string tokens_to_text(const std::vector<i32>& tokens) {
    std::string text;
    for (auto t : tokens) {
        if (vocab.count(t)) {
            if (!text.empty() && vocab[t] != "." && vocab[t] != "<eos>") {
                text += " ";
            }
            text += vocab[t];
        }
    }
    return text;
}

// ============================================================================
// Example 1: Greedy Decoding
// ============================================================================

void example_greedy_decoding() {
    print_separator("Example 1: Greedy Decoding");
    
    std::cout << "Greedy decoding always picks the most probable token.\n\n";
    
    TokenSampler sampler(42);
    
    // Simulate logits where token 4 ("fox") is most likely
    std::vector<f32> logits(20, 0.0f);
    logits[4] = 10.0f;  // fox
    logits[3] = 8.0f;   // brown
    logits[2] = 6.0f;   // quick
    
    auto config = GenerationConfig::greedy();
    
    std::cout << "Logits (top 3):\n";
    std::cout << "  Token 4 ('fox'):   " << logits[4] << "\n";
    std::cout << "  Token 3 ('brown'): " << logits[3] << "\n";
    std::cout << "  Token 2 ('quick'): " << logits[2] << "\n\n";
    
    i32 selected = sampler.sample(logits, config);
    std::cout << "Selected token: " << selected << " ('" << vocab[selected] << "')\n";
    std::cout << "\n→ Greedy always picks the highest logit!\n";
}

// ============================================================================
// Example 2: Temperature Sampling
// ============================================================================

void example_temperature() {
    print_separator("Example 2: Temperature Sampling");
    
    std::vector<f32> logits(5, 0.0f);
    logits[0] = 2.0f;
    logits[1] = 1.0f;
    logits[2] = 0.5f;
    logits[3] = 0.3f;
    logits[4] = 0.1f;
    
    std::cout << "Base logits: [2.0, 1.0, 0.5, 0.3, 0.1]\n\n";
    
    auto show_distribution = [&](f32 temp, const std::string& desc) {
        auto probs = sampling::softmax(logits);
        
        // Apply temperature manually for display
        std::vector<f32> scaled_logits = logits;
        for (auto& l : scaled_logits) l /= temp;
        probs = sampling::softmax(scaled_logits);
        
        std::cout << "Temperature " << std::fixed << std::setprecision(1) 
                  << temp << " (" << desc << "):\n  [";
        for (usize i = 0; i < probs.size(); ++i) {
            std::cout << std::setprecision(3) << probs[i];
            if (i < probs.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    };
    
    show_distribution(0.1f, "very focused");
    show_distribution(0.5f, "focused");
    show_distribution(1.0f, "neutral");
    show_distribution(1.5f, "diverse");
    show_distribution(2.0f, "very diverse");
    
    std::cout << "\n→ Lower temperature = more deterministic\n";
    std::cout << "→ Higher temperature = more random/creative\n";
}

// ============================================================================
// Example 3: Top-K and Top-P Sampling
// ============================================================================

void example_top_k_top_p() {
    print_separator("Example 3: Top-K and Top-P Sampling");
    
    std::vector<f32> logits(10, 0.0f);
    logits[0] = 5.0f;  // 0.60
    logits[1] = 3.0f;  // 0.20
    logits[2] = 2.0f;  // 0.10
    logits[3] = 1.0f;  // 0.05
    logits[4] = 0.5f;  // 0.03
    // Rest are 0.0f (~0.02 total)
    
    auto probs = sampling::softmax(logits);
    
    std::cout << "Token probabilities:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "  Token " << i << ": " << std::fixed << std::setprecision(2) 
                  << (probs[i] * 100) << "%\n";
    }
    std::cout << "  Others: ~2%\n\n";
    
    // Top-K
    std::cout << "Top-K (k=3):\n";
    std::cout << "  Only considers tokens 0, 1, 2\n";
    std::cout << "  Effectively samples from ~90% of probability mass\n\n";
    
    // Top-P
    std::cout << "Top-P (p=0.9):\n";
    std::cout << "  Includes tokens until cumulative prob >= 90%\n";
    std::cout << "  May include more or fewer tokens than top-k\n\n";
    
    // Demonstrate with actual sampling
    TokenSampler sampler(42);
    
    auto config_k = GenerationConfig::sampling(1.0f, 1.0f);
    config_k.top_k = 3;
    
    auto config_p = GenerationConfig::sampling(1.0f, 0.9f);
    
    std::map<i32, i32> counts_k, counts_p;
    for (int i = 0; i < 1000; ++i) {
        counts_k[sampler.sample(logits, config_k)]++;
    }
    
    TokenSampler sampler2(123);
    for (int i = 0; i < 1000; ++i) {
        counts_p[sampler2.sample(logits, config_p)]++;
    }
    
    std::cout << "Sample distribution (1000 samples):\n";
    std::cout << "  Top-K: {";
    for (const auto& [k, v] : counts_k) {
        std::cout << k << ":" << v << " ";
    }
    std::cout << "}\n";
    
    std::cout << "  Top-P: {";
    for (const auto& [k, v] : counts_p) {
        std::cout << k << ":" << v << " ";
    }
    std::cout << "}\n";
}

// ============================================================================
// Example 4: Repetition Penalty
// ============================================================================

void example_repetition_penalty() {
    print_separator("Example 4: Repetition Penalty");
    
    std::vector<f32> logits(10, 1.0f);
    logits[4] = 3.0f;  // "fox" - highest
    
    std::cout << "Scenario: Model wants to generate 'fox' again\n\n";
    std::cout << "Initial logits for 'fox': " << logits[4] << "\n";
    
    // Previous tokens include "fox" (token 4)
    std::vector<i32> prev_tokens = {1, 2, 3, 4};  // "The quick brown fox"
    
    // Apply penalty
    std::vector<f32> penalized = logits;
    sampling::apply_repetition_penalty(penalized, prev_tokens, 1.5f);
    
    std::cout << "After penalty (1.5x): " << penalized[4] << "\n";
    std::cout << "Other tokens unchanged: " << penalized[5] << "\n\n";
    
    std::cout << "Effect:\n";
    std::cout << "  Without penalty: Would likely choose 'fox' again\n";
    std::cout << "  With penalty: Other tokens become more likely\n";
    std::cout << "\n→ Helps avoid repetitive text generation!\n";
}

// ============================================================================
// Example 5: Full Generation Pipeline
// ============================================================================

void example_generation_pipeline() {
    print_separator("Example 5: Full Generation Pipeline");
    
    InferenceEngine engine(42);
    
    // Mock model forward function
    int step = 0;
    auto model_forward = [&step](const std::vector<i32>& tokens) -> std::vector<f32> {
        std::vector<f32> logits(20, 0.0f);
        
        // Simulate a simple pattern: "The quick brown fox jumps over the lazy dog."
        std::vector<i32> pattern = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0};
        
        usize pos = tokens.size() - 1;
        if (pos < pattern.size()) {
            i32 next_token = pattern[pos];
            logits[next_token] = 10.0f;  // Make it very likely
            
            // Add some noise
            for (int i = 0; i < 20; ++i) {
                if (i != next_token) {
                    logits[i] = -2.0f + (i % 3) * 0.5f;
                }
            }
        }
        
        step++;
        return logits;
    };
    
    std::cout << "Generating text with different strategies:\n\n";
    
    // Strategy 1: Greedy
    {
        auto config = GenerationConfig::greedy()
            .with_max_tokens(15)
            .with_eos(0);
        
        std::vector<i32> prompt = {1};  // "The"
        auto result = engine.generate(prompt, model_forward, config);
        
        std::cout << "1. Greedy decoding:\n";
        std::cout << "   Output: " << tokens_to_text(result.token_ids) << "\n";
        std::cout << "   Tokens generated: " << result.tokens_generated << "\n";
        std::cout << "   Time: " << std::fixed << std::setprecision(2) 
                  << result.generation_time_ms << " ms\n\n";
    }
    
    // Strategy 2: Sampling with temperature
    {
        auto config = GenerationConfig::sampling(0.7f, 0.9f)
            .with_max_tokens(15)
            .with_eos(0);
        
        std::vector<i32> prompt = {1};
        auto result = engine.generate(prompt, model_forward, config);
        
        std::cout << "2. Sampling (temp=0.7, top_p=0.9):\n";
        std::cout << "   Output: " << tokens_to_text(result.token_ids) << "\n";
        std::cout << "   Tokens/sec: " << std::setprecision(0) 
                  << result.tokens_per_second() << "\n\n";
    }
    
    // Strategy 3: With repetition penalty
    {
        auto config = GenerationConfig::sampling(0.8f, 0.95f)
            .with_max_tokens(15)
            .with_eos(0)
            .with_repetition_penalty(1.2f);
        
        std::vector<i32> prompt = {1};
        auto result = engine.generate(prompt, model_forward, config);
        
        std::cout << "3. With repetition penalty (1.2x):\n";
        std::cout << "   Output: " << tokens_to_text(result.token_ids) << "\n\n";
    }
}

// ============================================================================
// Example 6: Beam Search
// ============================================================================

void example_beam_search() {
    print_separator("Example 6: Beam Search");
    
    std::cout << "Beam search maintains multiple hypotheses:\n\n";
    
    BeamSearch beam(3, 0.6f);  // 3 beams, length penalty 0.6
    beam.initialize({1});  // Start with "The"
    
    std::cout << "Initial beam: [1] = \"The\"\n\n";
    
    // Simulate a few steps
    std::vector<std::vector<std::vector<f32>>> all_logits = {
        // Step 1: After "The"
        {{-1, -1, 5.0f, 3.0f, -1, -1, -1, -1, -1, -1}},  // "quick" or "brown"
        // Step 2: After expansion
        {{-1, -1, -1, 5.0f, 3.0f, -1, -1, -1, -1, -1},
         {-1, -1, -1, 4.0f, 4.0f, -1, -1, -1, -1, -1},
         {-1, -1, -1, 3.0f, 5.0f, -1, -1, -1, -1, -1}},
    };
    
    std::cout << "Step 1: Expanding beams\n";
    beam.step(all_logits[0], 0);
    
    auto seqs = beam.all_sequences();
    std::cout << "  Active beams: " << seqs.size() << "\n";
    for (usize i = 0; i < seqs.size(); ++i) {
        std::cout << "    Beam " << i << ": " << tokens_to_text(seqs[i]) << "\n";
    }
    
    std::cout << "\n→ Beam search explores multiple paths simultaneously!\n";
    std::cout << "→ Useful for finding globally better sequences.\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════╗
║                    Inference Engine Examples                   ║
║                     Optimization Core C++                      ║
╚═══════════════════════════════════════════════════════════════╝
)" << std::endl;

    example_greedy_decoding();
    example_temperature();
    example_top_k_top_p();
    example_repetition_penalty();
    example_generation_pipeline();
    example_beam_search();
    
    print_separator("Summary");
    std::cout << "Sampling strategies:\n";
    std::cout << "  1. Greedy - deterministic, picks max logit\n";
    std::cout << "  2. Temperature - controls randomness\n";
    std::cout << "  3. Top-K - limits vocabulary to top K tokens\n";
    std::cout << "  4. Top-P (nucleus) - limits by cumulative probability\n";
    std::cout << "  5. Repetition penalty - discourages repetition\n";
    std::cout << "  6. Beam search - explores multiple hypotheses\n";
    std::cout << "\nRecommended combinations:\n";
    std::cout << "  • Factual: greedy or temp=0.3, top_p=0.9\n";
    std::cout << "  • Creative: temp=0.8-1.0, top_p=0.95, rep_pen=1.1\n";
    std::cout << "  • Diverse: temp=1.2, top_k=50, top_p=0.95\n";
    std::cout << "\n";
    
    return 0;
}












