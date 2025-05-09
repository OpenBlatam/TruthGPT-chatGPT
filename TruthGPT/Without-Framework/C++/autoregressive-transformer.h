#ifndef AUTOREGRESSIVE_TRANSFORMER_H
#define AUTOREGRESSIVE_TRANSFORMER_H

#include <Eigen/Dense>
#include <functional>
#include <string>
#include <unordered_map>

class AutoregressiveTransformer {
public:
    AutoregressiveTransformer(int input_size, int hidden_size, int output_size, std::string activation_func, double learning_rate, int batch_size);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& X);
    void backward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
    void update();
private:
    Eigen::MatrixXd W1, b1, W2, b2, dW1, db1, dW2, db2, h1, A2;
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> activation;
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> activation_deriv;
    double learning_rate;
    int batch_size;
    std::string activation_func;
};

Eigen::MatrixXd relu(const Eigen::MatrixXd& x);
Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x);
Eigen::MatrixXd elu(const Eigen::MatrixXd& x);
Eigen::MatrixXd leaky_relu(const Eigen::MatrixXd& x);

#endif // AUTOREGRESSIVE_TRANSFORMER_H
