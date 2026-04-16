# ifndef PA_H
# define PA_H

# include <cmath>
# include <vector>
# include <numeric>
# include <algorithm>

# include "Loss.h"
# include "utils.h"

namespace rivercpp {
template<double C, int mode=1, bool learn_intercept=true, int num_features>
class BasePA {
private:
    static inline double _calc_tau_0(const std::vector<double>& x, const double loss) {
        double norm = square_sum(x);
        if (norm > 0.0) return loss / norm;
        return 0.0;
    }
    static inline double _calc_tau_1(const std::vector<double>& x, const double loss) {
        double norm = square_sum(x);
        if (norm > 0.0) return std::fmin(C, loss / norm);
        return 0.0;
    }
    static inline double _calc_tau_2(const std::vector<double>& x, const double loss) {
        return loss / (square_sum(x) + 0.5 / C);
    }
protected:
    double intercept = 0.0;
    std::vector<double> weights;
public:
    BasePA() : weights(num_features, 0.0) {};
    double calc_tau(const std::vector<double>& x, const double loss) {
        if constexpr (mode == 0) return _calc_tau_0(x, loss);
        if constexpr (mode == 1) return _calc_tau_1(x, loss);
        if constexpr (mode == 2) return _calc_tau_2(x, loss);
        return 0.0;
    }
};

template<double eps=0.1, double C=1.0, int mode=1, bool learn_intercept=true, int num_features>
class PARegressor : public BasePA<C, mode, learn_intercept, num_features>, public Regressor {
public:
    virtual void learn_one(const std::vector<double>& x, double y) override {
        double y_pred = predict_one(x);
        double tau = this->calc_tau(x, EpsilonInsensitiveHinge<eps>(y, y_pred));
        double step = std::copysign(tau, y - y_pred);
        for (size_t i=0;i<x.size();i++) {
            this->weights[i] += step * x[i];
        }
        if constexpr(learn_intercept) this->intercept += step;
    }
    virtual double predict_one(const std::vector<double>& x) override {
        return std::inner_product(x.begin(), x.end(), this->weights.begin(), this->intercept);
    }
};

}

# endif