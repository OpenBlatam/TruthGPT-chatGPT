#ifndef AUTOREG_TRANSFORMER_H
#define AUTOREG_TRANSFORMER_H

#include <torch/torch.h>
#include <vector>

class AutoRegTransformerImpl : public torch::nn::Module {
public:
    AutoRegTransformerImpl(int num_layers, int input_size, int hidden_size, int output_size);
    std::vector<std::vector<float>> forward(const std::vector<int>& input_seq);
private:
    int num_layers;
    int hidden_size;
    torch::nn::Embedding embedding;
    torch::nn::TransformerEncoder encoder;
    torch::nn::Linear linear;
};

TORCH_MODULE(AutoRegTransformer);

float train(AutoRegTransformer& model, const std::vector<std::vector<int>>& input_seqs,
            const std::vector<std::vector<int>>& output_seqs, float learning_rate, int batch_size);

#endif  // AUTOREG_TRANSFORMER_H
