#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

int main()
{
    // Define the function to be optimized
    auto func = [](const VectorXd& x) {
        return x(0)*x(0) + x(1)*x(1);
    };

    // Define the gradient of the function
    auto grad = [](const VectorXd& x) {
        VectorXd g(2);
        g(0) = 2*x(0);
        g(1) = 2*x(1);
        return g;
    };

    // Initialize the parameters
    VectorXd x(2);
    x << 1, 1;

    // Set the learning rate and number of iterations
    double alpha = 0.1;
    int num_iterations = 100;

    // Run the gradient descent algorithm
    for (int i = 0; i < num_iterations; i++) {
        VectorXd gradient = grad(x);
        x -= alpha * gradient;
        double cost = func(x);
        std::cout << "Iteration " << i << ", cost = " << cost << std::endl;
    }

    // Print the final parameters
    std::cout << "Final parameters: " << x.transpose() << std::endl;

    return 0;
}
