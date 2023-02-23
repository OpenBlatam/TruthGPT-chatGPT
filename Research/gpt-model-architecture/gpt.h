#ifndef TRANSFORMER_LAYER_H
#define TRANSFORMER_LAYER_H

#include <Eigen/Dense>

class MatrixOps {
public:
    MatrixOps(int hidden_size);

    Eigen::MatrixXf matmul(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B);
    Eigen::MatrixXf transpose(const Eigen::MatrixXf& A);
    Eigen::MatrixXf softmax(const Eigen::MatrixXf& x);
    Eigen::MatrixXf layer_norm(const Eigen::MatrixXf& x, const Eigen::VectorXf& weights);
    Eigen::MatrixXf relu(const Eigen::MatrixXf& x);

private:
    int hidden_size_;
};

class TransformerLayer {
public:
    TransformerLayer(int num_heads, int hidden_size, float dropout_rate = 0.1);

    Eigen::MatrixXf operator()(const Eigen::MatrixXf& inputs, bool training = true);

private:
    int num_heads_;
    int hidden_size_;
    float dropout_rate_;
    Eigen::MatrixXf q_weights_;
    Eigen::MatrixXf k_weights_;
    Eigen::MatrixXf v_weights_;
    Eigen::MatrixXf attention_output_weights_;
    Eigen::MatrixXf feedforward_weights1_;
    Eigen::MatrixXf feedforward_weights2_;
    Eigen::VectorXf layer_norm_weights1_;
    Eigen::VectorXf layer_norm_weights2_;

    MatrixOps ops_;
};

#endif  // TRANSFORMER_LAYER_H
