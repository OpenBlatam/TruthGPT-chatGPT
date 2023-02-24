#include <cmath>
#include <vector>

class AdamOptimizer {
public:
    AdamOptimizer(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : alpha(alpha), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

    std::vector<double> update(std::vector<double> params, const std::vector<double>& gradients) {
        if (m.empty()) m.resize(params.size());
        if (v.empty()) v.resize(params.size());

        t++;
        for (int i = 0; i < params.size(); i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];
            v[i] = beta2 * v[i] + (1 - beta2) * std::pow(gradients[i], 2);
            double m_hat = m[i] / (1 - std::pow(beta1, t));
            double v_hat = v[i] / (1 - std::pow(beta2, t));
            params[i] -= alpha * m_hat / (std::sqrt(v_hat) + epsilon);
        }

        return params;
    }

private:
    double alpha;
    double beta1;
    double beta2;
    double epsilon;
    int t;
    std::vector<double> m;
    std::vector<double> v;
};
