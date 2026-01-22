# ifndef STANDARD_SCALER_H
# define STANDARD_SCALER_H

# include "Transformer.h"

# include <cmath>
# include <vector>
# include <array>

namespace rivercpp {
// always with_std
template <int num_features>
class StandardScaler : public Transformer {
private:
    std::array<int, num_features> counts;
    std::array<double, num_features> means;
    std::array<double, num_features> vars;
public:
    void learn_one(const std::vector<double>& x, int y) override {
        for (int i=0;i<x.size();i++) {
            counts[i] += 1;
            double old_mean = means[i];
            means[i] += (x[i] - old_mean) / counts[i];
            vars[i] += ((x[i] - old_mean) * (x[i] - means[i]) - vars[i]) / counts[i];
        }
    }
    std::vector<double> transform_one(const std::vector<double>& x) override {
        std::vector<double> res(x.size(), 0.0);
        for (int i=0;i<x.size();i++) {
            res[i] = vars[i] > 0.0 ? (x[i] - means[i]) / std::sqrt(vars[i]) : 0.0;
        }
        return res;
    }
};
}

# endif