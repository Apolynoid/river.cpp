# ifndef REGRESSOR_H
# define REGRESSOR_H

# include <vector>

namespace rivercpp {
class Regressor {
public:
    virtual void learn_one(const std::vector<double>& x, double y) = 0;
    virtual double predict_one(const std::vector<double>& x) = 0;
    virtual ~Regressor() = default;
};
}

# endif