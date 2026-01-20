# ifndef GAUSSIAN_SPLITTER_H
# define GAUSSIAN_SPLITTER_H

# include <cmath>
# include <limits>
# include <vector>
# include <unordered_map>
# include "TreeBase.h"

class Gaussian {
private:
    double _mean = 0.0;
    double _S = 0.0;
    double ddof = 1;
public:
    double n = 0.0;
    void update(double x, double w=1.0) {
        double mean_old = _mean;
        n += w;
        _mean += (w / n) * (x - _mean);
        _S += w * (x - mean_old) * (x - _mean);
    }
    double get_var() {
        if (n > ddof) {
            return _S / (n - ddof);
        }
        return 0.0;
    }
    double cdf(double x) {
        double var = get_var();
        if (var == 0.0) return 0.0;
        return 0.5 * (1.0 + std::erf((x - _mean) / std::sqrt(var * 2.0)));
    }
    double operator()(double x) {
        double var = get_var();
        if (var == 0.0) return 0.0;
        return std::exp(-0.5 * (x - _mean) * (x - _mean) / var) / std::sqrt(2 * M_PI * var);
    }
};

template <int num_features, int num_labels>
class GaussianSplitter {
private:
    std::vector<Gaussian> _att_dist_per_class{std::vector<Gaussian>(num_labels)};
    std::vector<double> _min_per_class{std::vector<double>(num_labels, std::numeric_limits<double>::max())};
    std::vector<double> _max_per_class{std::vector<double>(num_labels, std::numeric_limits<double>::lowest())};
    int n_split;
    std::vector<double> _split_point_suggestions() {
        std::vector<double> res;
        res.reserve(n_split);
        double min_value = std::numeric_limits<double>::max();
        double max_value = std::numeric_limits<double>::lowest();
        for (int i=0;i<num_labels;i++) {
            if (_min_per_class[i] < min_value) min_value = _min_per_class[i];
            if (_max_per_class[i] > max_value) max_value = _max_per_class[i];
        }
        if (min_value < std::numeric_limits<double>::max()) {
            double bin_size = max_value - min_value;
            bin_size /= n_split + 1;
            for (int i=0;i<n_split;i++) {
                double split_value = min_value + bin_size * (i + 1);
                if (split_value > min_value && split_value < max_value) res.push_back(split_value);
            }
        }
        return res;
    }
    std::pair<std::vector<double>, std::vector<double> > _class_dists_from_binary_split(double split_value) {
        std::vector<double> lhs_dist(num_labels, -1.0);
        std::vector<double> rhs_dist(num_labels, -1.0);
        for (int i=0;i<num_labels;i++) {
            if (split_value < _min_per_class[i]) {
                rhs_dist[i] = _att_dist_per_class[i].n;
            } else if (split_value >= _max_per_class[i]) {
                lhs_dist[i] = _att_dist_per_class[i].n;
            } else {
                lhs_dist[i] = _att_dist_per_class[i].cdf(split_value) * _att_dist_per_class[i].n;
                rhs_dist[i] = _att_dist_per_class[i].n - lhs_dist[i];
            }
        }
        return std::make_pair(lhs_dist, rhs_dist);
    }
public:
    GaussianSplitter(int n_split=10) : n_split(n_split) {}
    void update(double att_val, int target_val, double w=1.0) {
        if (att_val < _min_per_class[target_val]) _min_per_class[target_val] = att_val;
        if (att_val > _max_per_class[target_val]) _max_per_class[target_val] = att_val;
        _att_dist_per_class[target_val].update(att_val, w);
    }
    double cond_proba(double att_val, int target_val) {
        return _att_dist_per_class[target_val](att_val);
    }
    BranchFactory<num_features, num_labels> best_evaluated_split_suggestion(const std::unordered_map<int, double>& pre_split_dist, 
        int att_idx, double min_branch_fraction) {
        BranchFactory<num_features, num_labels> best_suggestion;
        std::vector<double> suggested_split_values = _split_point_suggestions();
        for (const double split_value : suggested_split_values) {
            std::pair<std::vector<double>, std::vector<double> > post_split_dist = _class_dists_from_binary_split(split_value);
            double merit = InfoGainSplitCriterion::merit_of_split(pre_split_dist, post_split_dist, min_branch_fraction);
            if (merit > best_suggestion.merit) {
                best_suggestion = BranchFactory<num_features, num_labels>(merit, att_idx, split_value);
            }
        }
        return best_suggestion;
    }
};

# endif