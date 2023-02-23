#include "transformer_layer.h"

class Encoder {
public:
    Encoder(int num_layers, int num_heads, int hidden_size, float dropout_rate = 0.1)
        : num_layers_(num_layers)
    {
        for (int i = 0; i < num_layers_; ++i) {
            layers_.emplace_back(num_heads, hidden_size, dropout_rate);
        }
    }

    Eigen::MatrixXf operator()(const Eigen::MatrixXf& inputs, bool training = true) {
        Eigen::MatrixXf output = inputs;
        for (int i = 0; i < num_layers_; ++i) {
            output = layers_[i](output, training);
        }
        return output;
    }

private:
    int num_layers_;
    std::vector<TransformerLayer> layers_;
};

class Decoder {
public:
    Decoder(int num_layers, int num_heads, int hidden_size, float dropout_rate = 0.1)
        : num_layers_(num_layers)
    {
        for (int i = 0; i < num_layers_; ++i) {
            self_attention_layers_.emplace_back(num_heads, hidden_size, dropout_rate);
            encoder_attention_layers_.emplace_back(num_heads, hidden_size, dropout_rate);
        }
    }

    Eigen::MatrixXf operator()(const Eigen::MatrixXf& inputs, const Eigen::MatrixXf& encoder_output, bool training = true) {
        Eigen::MatrixXf output = inputs;
        for (int i = 0; i < num_layers_; ++i) {
            // Self-attention layer
            output = self_attention_layers_[i](output, training);

            // Encoder attention layer
            Eigen::MatrixXf attention_query = output;
            Eigen::MatrixXf attention_key = encoder_output;
            Eigen::MatrixXf attention_value = encoder_output;
            output = encoder_attention_layers_[i](attention_query, attention_key, attention_value, training);
        }
        return output;
    }

private:
    int num_layers_;
    std::vector<TransformerLayer> self_attention_layers_;
    std::vector<TransformerLayer> encoder_attention_layers_;
};

class Transformer {
public:
    Transformer(int num_encoder_layers, int num_decoder_layers, int num_heads, int hidden_size, float dropout_rate = 0.1)
        : encoder_(num_encoder_layers, num_heads, hidden_size, dropout_rate),
        decoder_(num_decoder_layers, num_heads, hidden_size, dropout_rate)
    {}

    Eigen::MatrixXf operator()(const Eigen::MatrixXf& encoder_inputs, const Eigen::MatrixXf& decoder_inputs, bool training = true) {
        Eigen::MatrixXf encoder_output = encoder_(encoder_inputs, training);
        Eigen::MatrixXf decoder_output = decoder_(decoder_inputs, encoder_output, training);
        return decoder_output;
    }

private:
    Encoder encoder_;
    Decoder decoder_;
};
