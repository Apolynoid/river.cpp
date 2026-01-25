# ifndef TREE_BASE_H
# define TREE_BASE_H

# include <cmath>
# include <limits>
# include <numeric>
# include <stdexcept>
# include <vector>
# include <array>
# include <unordered_map>
# include <unordered_set>

namespace rivercpp {
template <int num_features> 
constexpr std::array<int, num_features> feature_array() {
    std::array<int, num_features> res;
    for (int i = 0; i < num_features; i++) res[i] = i;
    return res;
}

template <int num_features, int num_labels>
class LeafNaiveBayesAdaptive;

template <int num_features, int num_labels>
class BranchOrLeaf {
protected:
    
public:
    bool is_leaf;
    std::unordered_map<int, double> stats;
    
    BranchOrLeaf(bool is_leaf, std::unordered_map<int, double> stats={}) : is_leaf(is_leaf), stats(stats) {}
    virtual BranchOrLeaf* next(const std::vector<double>& x) { throw std::runtime_error("Next Not Implied!"); }
    virtual BranchOrLeaf* traverse(const std::vector<double>& x, bool until_leaf=true) = 0;
    virtual std::vector<LeafNaiveBayesAdaptive<num_features, num_labels>*> iter_leaves() = 0;
    virtual void prediction(std::vector<double>& proba, const std::vector<double>& x) { throw std::runtime_error("Prediction Not Implied!"); }
    virtual void learn_one(const std::vector<double>& x, int y, double w=1.0) { throw std::runtime_error("Learning Not Implied!"); }
    virtual double total_weight() = 0;
};

// under gaussian splitter we always use this class
template <int num_features, int num_labels>
class NumericBinaryBranch : public BranchOrLeaf<num_features, num_labels> {
private:
    double threshold;
    int feature;
public:
    BranchOrLeaf<num_features, num_labels>* children[2];
    NumericBinaryBranch(int feature, double threshold, BranchOrLeaf<num_features, num_labels>* left, 
        BranchOrLeaf<num_features, num_labels>* right, std::unordered_map<int, double> stats={}) : 
        BranchOrLeaf<num_features, num_labels>(false, stats), threshold(threshold), feature(feature), children{left, right} {}
    inline int branch_no(const std::vector<double>& x) const { return x[feature] <= threshold ? 0 : 1; }

    double total_weight() override { return children[0]->total_weight() + children[1]->total_weight(); }
    BranchOrLeaf<num_features, num_labels>* most_common_path() const {
        BranchOrLeaf<num_features, num_labels>* res;
        return (children[0]->total_weight() < children[1]->total_weight()) ? children[1] : children[0];
    }
    BranchOrLeaf<num_features, num_labels>* next(const std::vector<double>& x) override { return children[branch_no(x)]; }
    // currently no need for until_leaf
    BranchOrLeaf<num_features, num_labels>* traverse(const std::vector<double>& x, const bool until_leaf=true) override {
        BranchOrLeaf<num_features, num_labels>* cur = this;
        while (!cur->is_leaf) {
            cur = cur->next(x);
        }
        return cur;
    }
    std::vector<LeafNaiveBayesAdaptive<num_features, num_labels>*> iter_leaves() override {
        std::vector<LeafNaiveBayesAdaptive<num_features, num_labels>*> res;
        if (children[0]) {
            std::vector<LeafNaiveBayesAdaptive<num_features, num_labels>*> left_leaves = children[0]->iter_leaves();
            res.insert(res.end(), left_leaves.begin(), left_leaves.end());
        }
        if (children[1]) {
            std::vector<LeafNaiveBayesAdaptive<num_features, num_labels>*> right_leaves = children[1]->iter_leaves();
            res.insert(res.end(), right_leaves.begin(), right_leaves.end());
        }
        return res;
    }
};

template <int num_features, int num_labels>
class BranchFactory {
    double threshold = -1.0;
public:
    double merit = std::numeric_limits<double>::lowest();
    int feature = -1;
    BranchFactory(const double merit=std::numeric_limits<double>::lowest(), 
        const int feature=-1, const double threshold=-1.0) 
        : threshold(threshold), merit(merit), feature(feature) {}
    bool operator<(const BranchFactory& rhs) const { return merit < rhs.merit; }
    bool operator==(const BranchFactory& rhs) const { return merit == rhs.merit; }
    NumericBinaryBranch<num_features, num_labels>* assemble(const std::unordered_map<int, double>& stats, 
        int depth, BranchOrLeaf<num_features, num_labels>* children[2]) const {
        return new NumericBinaryBranch(feature, threshold, children[0], children[1], stats);
    }
};

template <int num_features, int num_labels>
class HoeffdingTree;

template <int num_features, int num_labels>
class GaussianSplitter;

double sum(const std::unordered_map<int, double>& x);

// use default NBA Leaf
template <int num_features, int num_labels>
class LeafNaiveBayesAdaptive : public BranchOrLeaf<num_features, num_labels> {
protected:
    std::vector<GaussianSplitter<num_features, num_labels>*> splitters = std::vector<GaussianSplitter<num_features, num_labels>*>(num_features, nullptr);
    double _mc_correct_weight = 0.0;
    double _nb_correct_weight = 0.0;
public:
    double last_split_attempt_at = 0.0;
    int depth;
    bool is_active = true;
    LeafNaiveBayesAdaptive(int depth) : BranchOrLeaf<num_features, num_labels>(true), depth(depth) {}
    double total_weight() override {
        return sum(this->stats);
    }
    BranchOrLeaf<num_features, num_labels>* traverse(const std::vector<double>& x, const bool until_leaf=true) override {
        return this;
    }
    std::vector<LeafNaiveBayesAdaptive*> iter_leaves() override {
        std::vector<LeafNaiveBayesAdaptive*> res;
        res.push_back(this);
        return res;
    }
    std::vector<BranchFactory<num_features, num_labels>> best_split_suggestions(HoeffdingTree<num_features, num_labels>* tree, 
        double max_share_to_split, double min_branch_fraction); 
    double calculate_promise();
    bool observed_class_distribution_is_pure() const {
        int count = 0;
        for (const auto& kv : this->stats) {
            if (kv.second > 0) count += 1;
            if (count > 1) return false;
        }
        return true;
    }
    void deactivate();
    virtual void update_splitters(const std::vector<double>& x, int y, double w); 
    void prediction(std::vector<double>& proba, const std::vector<double>& x) override;
    void learn_one(const std::vector<double>& x, int y, double w=1.0) override; 
};

template <int num_features, int num_labels>
class RandomLeafNaiveBayesAdaptive : public LeafNaiveBayesAdaptive<num_features, num_labels> {
protected:
    int max_features;
    std::vector<int> feature_indices;
    std::default_random_engine* rng;
    void _sample_features(std::vector<int> &feature_indices, int max_features) {
        feature_indices.clear();
        feature_indices.resize(max_features);
        std::array<int, num_features> features = feature_array<num_features>();
        if (num_features > max_features) {
            std::sample(features.begin(), features.end(), feature_indices.begin(), max_features, *rng);
            return;
        }
        feature_indices.insert(feature_indices.end(), features.begin(), features.end());
    }
public:
    RandomLeafNaiveBayesAdaptive(int depth, int max_features, std::default_random_engine* rng) 
        : LeafNaiveBayesAdaptive<num_features, num_labels>(depth), max_features(max_features), rng(rng) {}
    virtual void update_splitters(const std::vector<double>& x, int y, double w); 
};

class InfoGainSplitCriterion {
private:
    static double _compute_entropy_dict(const std::unordered_map<int, double>& dist) {
        double entropy = 0.0;
        double dis_sums = 0.0;
        for (const auto& [_, d] : dist) {
            if (d > 0.0) {
                entropy -= d * std::log2(d);
                dis_sums += d;
            }
        }
        return dis_sums > 0.0 ? ((entropy + dis_sums * std::log2(dis_sums)) / dis_sums) : 0.0;
    }
    static double _compute_entropy_vector(const std::vector<double>& dist) {
        double entropy = 0.0;
        double dis_sums = 0.0;
        for (const auto& d : dist) {
            if (d > 0.0) {
                entropy -= d * std::log2(d);
                dis_sums += d;
            }
        }
        return dis_sums > 0.0 ? ((entropy + dis_sums * std::log2(dis_sums)) / dis_sums) : 0.0;
    }
    static double _compute_entropy_pair(const std::pair<std::vector<double>, std::vector<double> >& dist) {
        double dist_weights[2] = {std::accumulate(dist.first.begin(), dist.first.end(), 0.0), 
            std::accumulate(dist.second.begin(), dist.second.end(), 0.0)};
        double total_weight = dist_weights[0] + dist_weights[1];
        double entropy = dist_weights[0] * _compute_entropy_vector(dist.first) + dist_weights[1] * _compute_entropy_vector(dist.second);
        return entropy / total_weight;
    }
public:
    static double merit_of_split(const std::unordered_map<int, double>& pre_split_dist, 
        const std::pair<std::vector<double>, std::vector<double> >& post_split_dist, double min_branch_fraction) {
        if (num_subsets_greater_than_frac(post_split_dist, min_branch_fraction) < 2) 
            return std::numeric_limits<double>::lowest();
        return compute_entropy(pre_split_dist) - compute_entropy(post_split_dist);
    }
    static double range_of_merit(const std::unordered_map<int, double>& pre_split_dist) {
        int num_classes = pre_split_dist.size();
        num_classes = num_classes > 2 ? num_classes : 2;
        return std::log2(num_classes);
    }

    static double compute_entropy(const std::unordered_map<int, double>& dist) { return _compute_entropy_dict(dist); }
    static double compute_entropy(const std::pair<std::vector<double>, std::vector<double> >& dist) { return _compute_entropy_pair(dist); }

    static int num_subsets_greater_than_frac(const std::pair<std::vector<double>, std::vector<double> >& distribution, 
        double min_frac) {
        int num_greater = 0;
        double dist_nums[2] = {std::accumulate(distribution.first.begin(), distribution.first.end(), 0.0), 
            std::accumulate(distribution.second.begin(), distribution.second.end(), 0.0)};
        double total_weight = dist_nums[0] + dist_nums[1];
        if (total_weight > 0)
            for (int i=0;i<2;i++) {
                if (dist_nums[i] / total_weight > min_frac) num_greater += 1;
            }
        return num_greater;
    } 
};

template <int num_features, int num_labels>
constexpr int estimate_branch_memory_bytes();
template <int num_features, int num_labels>
constexpr int estimate_leaf_memory_bytes();
template <int num_features, int num_labels>
int estimate_tree_memory_bytes(HoeffdingTree<num_features, num_labels>* tree);
}

# endif