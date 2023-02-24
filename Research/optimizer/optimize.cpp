#include <cmath>
#include <vector>

class AdagradOptimizer {
public:
    AdagradOptimizer(float learning_rate = 0.01, float epsilon = 1e-8)
        : learning_rate_(learning_rate), epsilon_(epsilon) {}

    void update(std::vector<float>& weights, const std::vector<float>& gradients) {
        if (cache_.empty()) {
            cache_.resize(gradients.size(), 0.0f);
        }

        for (int i = 0; i < weights.size(); i++) {
            cache_[i] += gradients[i] * gradients[i];
            float adaptive_learning_rate = learning_rate_ / (std::sqrt(cache_[i]) + epsilon_);
            weights[i] -= adaptive_learning_rate * gradients[i];
        }
    }

private:
    float learning_rate_;
    float epsilon_;
    std::vector<float> cache_;
};
