/**
 * ggml Inference Engine Header
 * 
 * High-performance CPU inference using ggml library.
 * 6-10x faster than Python for LLM inference.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>

namespace truthgpt {
namespace inference {

/**
 * GGML Inference Engine
 * 
 * Features:
 * - Efficient CPU inference
 * - INT4/INT8 quantization
 * - Memory efficient
 * - Backend of llama.cpp
 */
class GGMLEngine {
public:
    GGMLEngine();
    ~GGMLEngine();
    
    /**
     * Load model from file.
     */
    void load_model(const std::string& model_path);
    
    /**
     * Generate tokens from input.
     * 
     * @param input_ids Input token IDs
     * @param max_new_tokens Maximum tokens to generate
     * @param temperature Sampling temperature
     * @return Generated token logits
     */
    std::vector<float> generate(
        const std::vector<int>& input_ids,
        int max_new_tokens = 64,
        float temperature = 0.7f
    );
    
private:
    class GGMLEngineImpl;
    std::unique_ptr<GGMLEngineImpl> impl_;
};

} // namespace inference
} // namespace truthgpt












