# ifndef HDDM_W_H
# define HDDM_W_H

# include <cmath>
# include <limits>

class EWMean {
private:
    double fading_factor;
    double mean = 0.0;
public:
    EWMean(double fading_factor=0.5) : fading_factor(fading_factor) {}
    void update(double x) {
        if (mean == 0.0) {
            mean = x;
        } else {
            mean = fading_factor * x + (1 - fading_factor) * mean;
        }
    }
    double get() const { return mean; }
};

class SampleInfo {
private:
    EWMean _ewma;
    double _lambd_sq;
    double _c_lambd_sq;
public:
    bool is_init = false;
    double ibc = 1.0;
    SampleInfo(double lambd) : _ewma(lambd), _lambd_sq(lambd * lambd), _c_lambd_sq((1-lambd) * (1-lambd)) {}
    double ewma() const { return _ewma.get(); }
    void update(double x) {
        _ewma.update(x);
        is_init = true;
        ibc = _lambd_sq + _c_lambd_sq * ibc;
    }
};

// We only care about model drift towards the worse part
// so no need for two side test -- two_sided_test always false
class HDDM_W {
private:
    double drift_confidence;
    double lambda_val;
    SampleInfo _total;
    SampleInfo _s1_incr;
    SampleInfo _s2_incr;  
    double _incr_cutpoint = std::numeric_limits<double>::max();
    void _reset() {
        drift_detected = false;
        _total = SampleInfo(lambda_val);
        _s1_incr = SampleInfo(lambda_val);
        _s2_incr = SampleInfo(lambda_val);
        _incr_cutpoint = std::numeric_limits<double>::max();
    }
    double _mcdiarmid_bound(double ibc, double confidence) {
        return std::sqrt(ibc * std::log(1.0 / confidence) / 2.0);
    }

    bool _has_mean_changed(const SampleInfo& sample1, const SampleInfo& sample2, double confidence) {
        if (!(sample1.is_init && sample2.is_init)) return false;
        double ibc_sum = sample1.ibc + sample2.ibc;
        double bound = _mcdiarmid_bound(ibc_sum, confidence);

        return sample2.ewma() - sample1.ewma() > bound;
    }

    bool _detect_mean_incr(double confidence) {
        return _has_mean_changed(_s1_incr, _s2_incr, confidence);
    }

    void _update_incr_stats(double x, double confidence) {
        double eps = _mcdiarmid_bound(_total.ibc, confidence);

        if (_total.ewma() + eps < _incr_cutpoint) {
            _incr_cutpoint = _total.ewma() + eps;
            _s1_incr = _total;
            _s2_incr = SampleInfo(lambda_val);
        } else {
            _s2_incr.update(x);
        }
    }
public:
    bool drift_detected = false;
    HDDM_W(double drift_confidence=0.001, double lambda_val=0.05) : 
        drift_confidence(drift_confidence), lambda_val(lambda_val),
        _total(lambda_val), _s1_incr(lambda_val), _s2_incr(lambda_val)
        { _reset(); }
    HDDM_W(HDDM_W&&) noexcept = default;
    HDDM_W& operator=(HDDM_W&&) noexcept = default;
    void update(double x) {
        if (drift_detected) {
            _reset();
        }
        _total.update(x);
        
        _update_incr_stats(x, drift_confidence);
        drift_detected = _detect_mean_incr(drift_confidence);
    }
};

# endif