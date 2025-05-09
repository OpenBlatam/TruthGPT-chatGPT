#include "autoregressive-transformer.h"
#include <Eigen/Dense>
#include <unordered_map>

using ActivationFunction = std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)>;

namespace {
    Eigen::MatrixXd softmax(const Eigen::MatrixXd& X) {
        Eigen::MatrixXd shifted = X.rowwise() - X.colwise().maxCoeff();
        Eigen::MatrixXd exp_shifted = shifted.array().exp();
        return exp_shifted.array().rowwise() / exp_shifted.array().rowwise().sum();
    }
}

Eigen::MatrixXd relu(const Eigen::MatrixXd& x) {
    return x.array().max(0.0);
}

Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x) {
    return 1.0 / (1.0 + (-x).array().exp());
}

Eigen::MatrixXd elu(const Eigen::MatrixXd& x) {
    return x.array().max(0.0) + 1e-6 * (x.array().exp() - 1.0).min(0.0);
}

Eigen::MatrixXd leaky_relu(const Eigen::MatrixXd& x) {
    return x.array().max(0.01 * x.array());
}

AutoregressiveTransformer::AutoregressiveTransformer(int input_size, int hidden_size, int output_size, std::string activation_func, double learning_rate, int batch_size)
    : W1(hidden_size, input_size),
      b1(hidden_size),
      W2(output_size, hidden_size),
      b2(output_size),
      dW1(hidden_size, input_size),
      db1(hidden_size),
      dW2(output_size, hidden_size),
      db2(output_size),
      h1(hidden_size),
      activation_func(activation_func),
      learning_rate(learning_rate),
      batch_size(batch_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    double std_dev1 = sqrt(2.0 / (input_size + hidden_size));
    double std_dev2 = sqrt(2.0 / (hidden_size + output_size));
    W1.setRandom();
    W1 *= std_dev1;
    W2.setRandom();
    W2 *= std_dev2;
    b1.setZero();
    b2.setZero();

    std::unordered_map<std::string, ActivationFunction> activation_map = {
        {"relu", relu},
        {"sigmoid", sigmoid},
        {"elu", elu},
        {"leaky_relu", leaky_relu}
    };
    auto it = activation_map.find(activation_func);
    if (it == activation_map.end()) {
        throw std::invalid_argument("Invalid activation function");
    }
    activation = it->second;
}

Eigen::MatrixXd AutoregressiveTransformer::forward(const Eigen::MatrixXd& X) {
    h1 = W1 * X.colwise().replicate(batch_size) + b1.replicate(1, batch_size);
    Eigen::MatrixXd a1 = activation(h1);
    A2 = softmax(W2 * a1 + b2.replicate(1, batch_size));
    return A2;
}

void AutoregressiveTransformer::backward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    int n_samples = X.cols();
    Eigen::MatrixXd dH2 = (A2 - Y.colwise().replicate(batch_size)) / n_samples;
    dW2 = dH2 * activation(h1).transpose();
    db2 = dH2.rowwise().sum();
    Eigen::MatrixXd dH1 = (W2.transpose() * dH2).array() * activation_deriv(h1).array();
    dW1 = dH1 * X.transpose();
    db1 = dH1.rowwise().sum();
}
