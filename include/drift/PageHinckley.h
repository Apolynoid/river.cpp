# ifndef PAGE_HINCKLEY_H
# define PAGE_HINCKLEY_H

# include <limits>

# include "stats.h"

// We are only interested about error_val increase
class PageHinckley {
private:
    Mean _x_mean;
    double _sum_increase = 0.0;
    double _min_increase = std::numeric_limits<double>::max();
    double threshold;
    double delta;
    double alpha;
    int min_instances;
    void _reset() {
        drift_detected = false;
        _x_mean = Mean();
        _sum_increase = 0.0;
        _min_increase = std::numeric_limits<double>::max();
    }
    bool _test_increase(double test_increase) {
        return test_increase > threshold;
    }
public:
    bool drift_detected = false;
    PageHinckley(double threshold=50.0, double delta=0.005, double alpha=0.9999, int min_instances=30) : 
        threshold(threshold), delta(delta), alpha(alpha), min_instances(min_instances) { _reset(); }
    PageHinckley(PageHinckley&& other) = default;
    PageHinckley& operator=(PageHinckley&& other) = default;
    void update(double x) {
        if (drift_detected) {
            _reset();
        }
        _x_mean.update(x);
        double dev = x - _x_mean.get();

        _sum_increase = alpha * _sum_increase + dev - delta;
        
        if (_sum_increase < _min_increase) {
            _min_increase = _sum_increase;
        }

        if (_x_mean.n >= min_instances) {
            double test_increase = _sum_increase - _min_increase;
            drift_detected = _test_increase(test_increase);
        }
    }
};

# endif