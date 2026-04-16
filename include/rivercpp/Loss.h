# ifndef LOSS_H
# define LOSS_H

# include <cmath>
# include <algorithm>

namespace rivercpp {
template <double eps=0.1>
class EpsilonInsensitiveHinge {
public:
    static inline double operator()(const double y_true, const double y_pred) {
        double y = y_true * 2 - 1.0;
        return std::fmax(std::fabs(y - y_pred) - eps, 0.0);
    }
    static inline double gradient(const double y_true, const double y_pred) {
        double diff = y_pred - y_true * 2 - 1.0;
        return (diff > eps) - (diff < -eps);
    }
    static inline double mean_func(double y_pred) { return y_pred; }
};
}

# endif