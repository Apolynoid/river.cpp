# ifndef AM_RULES_H
# define AM_RULES_H

# include <algorithm>
# include <array>
# include <cassert>
# include <cmath>
# include <concepts>
# include <optional>
# include <vector>

# include "drift/DetectorConcept.h"
# include "drift/ADWIN.h"
# include "LinReg.h"
# include "QOSplitter.h"
# include "Regressor.h"
# include "drift/stats.h"

namespace rivercpp {
class NumericLiteral {
public:
    bool neg;
    int on;
    double at;
    NumericLiteral(int on, double at, bool neg=false) : neg(neg), on(on), at(at) {}
    bool operator()(const std::vector<double>& x) const {
        assert((x.size() > static_cast<size_t>(on)) && "NumericLiteral(): Feature index out of bounds!");
        if (!neg) return x[on] <= at;
        else return x[on] > at;
    }
};

class MeanRegressor : public Regressor {
private:
    Mean mean;
public:
    virtual void learn_one(const std::vector<double>& x, double y) override {
        mean.update(y);
    }
    virtual double predict_one(const std::vector<double>& x) override {
        return mean.get();
    }
};

template <std::derived_from<Regressor> PredModel>
class AdaptiveRegressor : public Regressor {
private:
    PredModel model_predictor;
    MeanRegressor mean_predictor;
    double fading_factor;
    double _mae_mean = 0.0;
    double _mae_model = 0.0;
public:
    AdaptiveRegressor(double fading_factor=0.99) : fading_factor(fading_factor) {}
    virtual void learn_one(const std::vector<double>& x, double y) override {
        double abs_error_mean = std::fabs(y - mean_predictor.predict_one(x));
        double abs_error_model = std::fabs(y - model_predictor.predict_one(x));
        _mae_mean = fading_factor * _mae_mean + abs_error_mean;
        _mae_model = fading_factor * _mae_model + abs_error_model;
        mean_predictor.learn_one(x, y);
        model_predictor.learn_one(x, y);
    }
    virtual double predict_one(const std::vector<double>& x) override {
        if (_mae_mean <= _mae_model) return mean_predictor.predict_one(x);
        else return model_predictor.predict_one(x);
    }
};

// just use QOSplitter
template <int num_features>
// We'll just get it all numeric
class HoeffdingRule {  
protected:
    std::array<QOSplitter<num_features>, num_features> splitters;
    virtual void _update_target_stats(double y, double w) = 0;
    virtual void _update_feature_stats(int feat_idx, double feat_val, double w) = 0;
    double _hoeffding_bound(double r_heur, double delta) {
        return r_heur * std::sqrt(-std::log(delta) / (2.0 * total_weight));
    }
public:
    std::vector<NumericLiteral> literals;
    double total_weight = 0.0;
    bool covers(const std::vector<double>& x) {
        return std::all_of(literals.begin(), literals.end(), 
            [&x](const NumericLiteral& lit) { return lit(x); });
    }
    void update(const std::vector<double>& x, double y, double w) {
        total_weight += w;
        _update_target_stats(y, w);
        for (int i=0;i<num_features;i++) {
            splitters[i].update(x[i], y, w);
            _update_feature_stats(i, x[i], w);
        }
    }
};

template <int num_features, int m_min=30, double anormaly_threshold=-0.75, 
    std::derived_from<Regressor> PredModel=AdaptiveRegressor<LinearRegression<num_features>>, 
    IsDetectorFactory DriftDetectorFactory=DetectorFactory<ADWIN<5>, 0.002>>
class RegRule : public HoeffdingRule<num_features> {
private:
    std::array<Var, num_features> _feat_stats;
    DriftDetectorFactory::DetectorType drift_detector;
    Var _target_stats;
protected:
    void _update_target_stats(double y, double w) override {
        _target_stats.update(y, w);
    }
    void _update_feature_stats(int feat_idx, double feat_val, double w) override {
        _feat_stats[feat_idx].update(feat_val, w);
    }
public:
    double last_expansion_attempt_at = 0.0;
    PredModel pred_model;
    std::optional<RegRule<num_features, m_min, anormaly_threshold, PredModel, DriftDetectorFactory>> 
        expand(double delta, double tau) {
        std::vector<BranchFactoryRegression> suggestions;
        for (int i=0;i<num_features;i++) {
            suggestions.push_back(this->splitters[i].best_evaluated_split_suggestion(_target_stats, i));
        }
        std::sort(suggestions.begin(), suggestions.end());

        bool should_expand = false;
        if (suggestions.size() < 2) should_expand = true;
        else {
            BranchFactoryRegression& b_split = suggestions[suggestions.size() - 1];
            BranchFactoryRegression& sb_split = suggestions[suggestions.size() - 2];
            double hb = this->_hoeffding_bound(VarianceRatioSplitCriterion<>::range_of_merit(_target_stats), delta);

            if (b_split.merit > 0 && (b_split.merit - sb_split.merit > hb || hb < tau)) should_expand = true;
        }

        if (should_expand) {
            BranchFactoryRegression& b_split = suggestions[suggestions.size() - 1];
            int branch_no = VarianceRatioSplitCriterion<>::select_best_branch(b_split.children_stats);
            // always numerical
            NumericLiteral lit(b_split.feature, b_split.threshold, branch_no!=0);

            bool literal_updated = false;
            for (NumericLiteral& l : this->literals) {
                if (lit.on == l.on && lit.neg == l.neg) {
                    if (!l.neg && lit.at < l.at) {
                        l.at = lit.at;
                        literal_updated = true;
                        break;
                    } else if (l.neg && lit.at > l.at) {
                        l.at = lit.at;
                        literal_updated = true;
                        break;
                    }
                }
            }
            if (!literal_updated) this->literals.push_back(lit);

            RegRule<num_features, m_min, anormaly_threshold, PredModel, DriftDetectorFactory> updated_rule;
            updated_rule.literals = this->literals;

            return updated_rule;
        }
        last_expansion_attempt_at = this->total_weight;
        return std::nullopt;
    }
    bool drift_test(double y, double y_pred) {
        double abs_error = std::fabs(y - y_pred);
        drift_detector.update(abs_error);
        return drift_detector.drift_detected;
    }
    double score_one(const std::vector<double>& x) {
        double score = 0.0;
        int hits = 0;
        for (int i=0;i<num_features;i++) {
            double mean = _feat_stats[i].mean.get();
            double var = _feat_stats[i].get();
            if (var > 0.0) {
                double proba = (2 * var) / (var + (x[i] - mean) * (x[i] - mean));
                if (proba > 0.0 && proba < 1.0) {
                    score += std::log(proba) - std::log(1.0 - proba);
                    hits++;
                }
            }
        }
        return hits > 0 ? score / hits : 0.0;
    }
    void learn_one(const std::vector<double>& x, double y) {
        this->update(x, y, 1.0);
        pred_model.learn_one(x, y);
    }
    double predict_one(const std::vector<double>& x) {
        return pred_model.predict_one(x);
    }
};

// assume ordered_rule_set is always true
// and we always use adaptive regressor
template <int num_features, int m_min=30, double anormaly_threshold=-0.75, double delta=1e-7, double tau=0.05>
class AMRules : public Regressor {
private:
    using Rule = RegRule<num_features, m_min, anormaly_threshold>;
    std::vector<Rule> _rules;
    Rule _default_rule;
    Rule _new_rule() {
        return Rule();
    }
public:
    AMRules() {}
    virtual void learn_one(const std::vector<double>& x, double y) {
        bool any_covered = false;
        std::vector<int> to_del;
        for (size_t i=0;i<_rules.size();i++) {
            Rule& rule = _rules[i];
            if (!rule.covers(x)) continue;
            if (rule.total_weight > m_min && rule.score_one(x) < anormaly_threshold) continue;

            double y_pred = rule.predict_one(x);
            
            if (rule.drift_test(y, y_pred)) {
                to_del.push_back(i);
                continue;
            }

            any_covered = true;
            rule.learn_one(x, y);

            if (rule.total_weight - rule.last_expansion_attempt_at >= m_min) {
                auto opt_rule = rule.expand(delta, tau);

                if (opt_rule.has_value()) {
                    opt_rule->pred_model = std::move(rule.pred_model);
                    _rules[i] = std::move(opt_rule.value());
                }

                break;
            }
        }

        if (!any_covered) {
            _default_rule.learn_one(x, y);
            if (_default_rule.total_weight - _default_rule.last_expansion_attempt_at >= m_min) {
                auto opt_rule = _default_rule.expand(delta, tau);
                if (opt_rule.has_value()) {
                    opt_rule->pred_model = std::move(_default_rule.pred_model);
                    _rules.push_back(std::move(opt_rule.value()));
                    _default_rule = Rule();
                }
            }
        }

        for (auto it = to_del.rbegin(); it != to_del.rend(); ++it) {
            _rules.erase(_rules.begin() + *it);
        }
    }
    virtual double predict_one(const std::vector<double>& x) {
        for (auto& rule : _rules) {
            if (rule.covers(x)) return rule.predict_one(x);
        }
        return _default_rule.predict_one(x);
    }
};
}

# endif