#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

template <typename T>
using Vector = std::vector<T>;

template <typename T>
class Tensor {
public:
    Tensor() {}
    Tensor(T value) : m_value(value) {}

    static Tensor<T> scalar(T value) {
        return Tensor<T>(value);
    }

    static Tensor<T> variable(size_t index) {
        Tensor<T> tensor;
        tensor.m_indices = {index};
        return tensor;
    }

    Tensor<T> operator+(const Tensor<T>& other) const {
        Tensor<T> result;
        result.m_operation = "+";
        result.m_lhs = *this;
        result.m_rhs = other;
        return result;
    }

    Tensor<T> operator*(const Tensor<T>& other) const {
        Tensor<T> result;
        result.m_operation = "*";
        result.m_lhs = *this;
        result.m_rhs = other;
        return result;
    }

    Tensor<T> transpose() const {
        Tensor<T> result;
        result.m_operation = "T";
        result.m_lhs = *this;
        return result;
    }

    Tensor<T> inverse() const {
        Tensor<T> result;
        result.m_operation = "^-1";
        result.m_lhs = *this;
        return result;
    }

    Matrix<T> evaluate() const {
        if (!m_indices.empty()) {
            throw std::logic_error("Cannot evaluate a non-constant tensor");
        }

        if (m_operation.empty()) {
            return {{m_value}};
        }

        Matrix<T> lhs = m_lhs.evaluate();
        Matrix<T> rhs = m_rhs.evaluate();

        if (m_operation == "+") {
            if (lhs.size() != rhs.size() || lhs[0].size() != rhs[0].size()) {
                throw std::logic_error("Incompatible dimensions for addition");
            }
            Matrix<T> result(lhs.size(), Vector<T>(lhs[0].size()));
            for (size_t i = 0; i < lhs.size(); ++i) {
                for (size_t j = 0; j < lhs[0].size(); ++j) {
                    result[i][j] = lhs[i][j] + rhs[i][j];
                }
            }
            return result;
        }

        if (m_operation == "*") {
            if (lhs[0].size() != rhs.size()) {
                throw std::logic_error("Incompatible dimensions for multiplication");
            }
            Matrix<T> result(lhs.size(), Vector<T>(rhs[0].size()));
            for (size_t i = 0; i < lhs.size(); ++i) {
                for (size_t j = 0; j < rhs[0].size(); ++j) {
                    T sum = 0;
                    for (size_t k = 0; k < lhs[0].size(); ++k) {
                        sum += lhs[i][k] * rhs[k][j];
                    }
                    result[i][j] = sum;
                }
            }
            return result;
        }

        if (m_operation == "T") {
            Matrix<T> result(lhs[0].size(), Vector<T>(lhs.size()));
            for (size_t i = 0; i < lhs.size(); ++i) {
                for (size_t j = 0; j < lhs[0].size(); ++j) {
                    result[j][i] = lhs[i][
