# ifndef UTILS_H
# define UTILS_H

# include <cmath>
# include <limits>
# include <vector>
# include <unordered_map>
# include <random>
# include <algorithm>
# include "GaussianSplitter.h"

namespace rivercpp {
double sum(const std::unordered_map<int, double>& x) {
    double res = 0.0;
    for (const auto& kv : x) {
        res += kv.second;
    }
    return res;
}

double max_value(const std::unordered_map<int, double>& x) {
    return std::max_element(x.begin(), x.end(), [](const auto& a, const auto& b) { return a.second < b.second;})->second;
}

double max_index(const std::vector<double>& x) {
    return std::distance(x.begin(), std::max_element(x.begin(), x.end()));
}

int poisson(int lambda, std::default_random_engine* gen) {
    static std::poisson_distribution<> dis(lambda);
    return dis(*gen, std::poisson_distribution<>::param_type(lambda));
}

template <int num_features, int num_labels>
void do_naive_bayes_prediction(std::vector<double>& votes, const std::vector<double>& x, 
    const std::unordered_map<int, double>& observed_class_distribution, 
    const std::vector<GaussianSplitter<num_features, num_labels>*>& splitters){
    double total_weight = sum(observed_class_distribution);
    if (total_weight == 0.0) {
        return;
    }
    for (const auto& kv : observed_class_distribution) {
        if (kv.second > 0) {
            votes[kv.first] = std::log(kv.second / total_weight);
        } else {
            votes[kv.first] = 0.0;
            continue;
        }

        for (size_t i=0;i<splitters.size();i++) {
            if (splitters[i] == nullptr) {
                continue;
            }
            double tmp = splitters[i]->cond_proba(x[i], kv.first);
            votes[kv.first] += tmp > 0 ? std::log(tmp) : std::numeric_limits<double>::lowest();
        }
    }

    double max_ll = *std::max_element(votes.begin(), votes.end());

    double lse = 0.0;
    for (double d : votes) {
        lse += std::exp(d - max_ll);
    }
    lse = max_ll + std::log(lse);

    for (size_t i=0;i<votes.size();i++) {
        votes[i] = std::exp(votes[i] - lse);
    }
}

void normalize_values_in_dict(std::vector<double>& res_dict, const std::unordered_map<int, double>& dictionary, 
    double factor=0.0, bool raise_error=false){
    if (factor == 0.0) {
        factor = sum(dictionary);
    }
    if (factor == 0.0) {
        if (raise_error)
            throw std::runtime_error("Can not normalize, normalization factor is 0");
        for (auto& kv : dictionary) {
            res_dict[kv.first] = kv.second;
        }
    } else {
        for (auto& kv : dictionary) {
            res_dict[kv.first] = kv.second / factor;
        }
    }
}
}

# endif