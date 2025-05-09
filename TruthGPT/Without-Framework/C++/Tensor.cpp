#include <iostream>
#include <vector>

class Tensor {
public:
    Tensor(const std::vector<int>& shape, double* data) : shape_(shape), data_(data) {}

    double operator()(const std::vector<int>& indices) const {
        int index = compute_index(indices);
        return data_[index];
    }

    double& operator()(const std::vector<int>& indices) {
        int index = compute_index(indices);
        return data_[index];
    }

    int num_dimensions() const {
        return shape_.size();
    }

    const std::vector<int>& shape() const {
        return shape_;
    }

    int size() const {
        int size = 1;
        for (int dim : shape_) {
            size *= dim;
        }
        return size;
    }

private:
    std::vector<int> shape_;
    double* data_;

    int compute_index(const std::vector<int>& indices) const {
        int index = 0;
        int stride = 1;
        for (int i = num_dimensions() - 1; i >= 0; --i) {
            index += indices[i] * stride;
            stride *= shape_[i];
        }
        return index;
    }
};

int main() {
    double data[4] = {1.0, 2.0, 3.0, 4.0};
    std::vector<int> shape = {2, 2};
    Tensor t(shape, data);
    std::cout << t({0, 0}) << "\n";  // prints 1.0
    std::cout << t({0, 1}) << "\n";  // prints 2.0
    std::cout << t({1, 0}) << "\n";  // prints 3.0
    std::cout << t({1, 1}) << "\n";  // prints 4.0
    t({0, 0}) = 5.0;
    std::cout << t({0, 0}) << "\n";  // prints 5.0
}
