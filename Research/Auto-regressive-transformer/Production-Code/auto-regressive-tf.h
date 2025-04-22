#ifndef AUTOREG_TRANSFORMER_H
#define AUTOREG_TRANSFORMER_H

#include <vector>

class AutoRegTransformer {
public:
    AutoRegTransformer(int num_layers, int input_size, int hidden_size, int output_size);
    std::vector<std::vector<float>> operator()(const std::vector<int>& input_seq);
private:
    int num_layers;
    int hidden_size;
    EmbeddingLayer embedding;
    std::vector<TransformerEncoderLayer> encoder_layers;
    TransformerEncoder encoder;
    LinearLayer linear;
};

float train(AutoRegTransformer& model, const std::vector<std::vector<int>>& input_seqs,
            const std::vector<std::vector<int>>& output_seqs, float learning_rate, int batch_size);

#endif  // AUTOREG_TRANSFORMER_H
