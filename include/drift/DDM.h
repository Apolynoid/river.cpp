# ifndef DDM_H
# define DDM_H

# include <cmath>
# include <limits>

# include "stats.h"

class DDM {
private:
    Mean _p;
    int warm_start;
    double drift_threshold;
    double _ps_min = std::numeric_limits<double>::max();
    double _p_min = 0.0;
    double _s_min = 0.0;
    void _reset() {
        drift_detected = false;
        _p = Mean();
        _ps_min = std::numeric_limits<double>::max();
        _p_min = 0.0;
        _s_min = 0.0;
    }
public:
    bool drift_detected = false;
    DDM(double drift_threshold=3.0, int warm_start=30) 
        : warm_start(warm_start), drift_threshold(drift_threshold) {
        _reset();
    }
    DDM(DDM&& other) = default;
    DDM& operator=(DDM&& other) = default;
    void update(double x) {
        if (drift_detected) {
            _reset();
        }
        _p.update(x);
        
        double p_i = _p.get();
        double n = _p.n;
        double s_i = std::sqrt(p_i * (1.0 - p_i) / n);

        if (n > warm_start) {
            if (p_i + s_i < _ps_min) {
                _p_min = p_i;
                _s_min = s_i;
                _ps_min = _p_min + _s_min;
            }
            if (p_i + s_i > _p_min + _s_min * drift_threshold) {
                drift_detected = true;
            }
        }
    }
};

# endif