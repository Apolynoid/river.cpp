# ifndef LIN_REG_H
# define LIN_REG_H

# include <array>

# include "Regressor.h"

namespace rivercpp {
// assume that l1 = l2 = 0.0, and clip gradient 1e12 thus almost unreachable
// use Squared as loss, SGD(0.01) as optimizer
template <int num_features, double learning_rate=0.01>
class LinearRegression : public Regressor {
private:
    std::array<double, num_features> _weights{};
    double intercept = 0.0;
    double _raw_dot_one(const std::vector<double>& x) const {
        double res = intercept;
        for (size_t i=0;i<num_features;i++) {
            res += x[i] * _weights[i];
        }
        return res;
    }
    void _fit(const std::vector<double>& x, double y) {
        double loss_gradient = (_raw_dot_one(x) - y) * 2;
        intercept -= learning_rate * loss_gradient;
        for (size_t i=0;i<num_features;i++) {
            _weights[i] -= learning_rate * loss_gradient * x[i];
        }
    }
public:
    void learn_one(const std::vector<double>& x, double y) override {
        _fit(x, y);
    }
    // mean_func of RegressionLoss just returns y
    double predict_one(const std::vector<double>& x) override {
        return _raw_dot_one(x);
    }
};

}

# endif