# ifndef TREE_BASE_TPP
# define TREE_BASE_TPP

# include "TreeBase.h"

# include <vector>
# include <queue>
# include <algorithm>

# include "HoeffdingTree.h"
# include "HoeffdingTreeClassifier.h"
# include "utils.h"
# include "GaussianSplitter.h"

namespace rivercpp {
template <int num_features, int num_labels>
std::vector<BranchFactory<num_features, num_labels> > LeafNaiveBayesAdaptive<num_features, num_labels>::best_split_suggestions(
    HoeffdingTree<num_features, num_labels>* tree, 
    double max_share_to_split, double min_branch_fraction) {
    std::vector<BranchFactory<num_features, num_labels> > best_suggestions;
    double maj_class = max_value(this->stats);
    if (maj_class > 0.0 && maj_class / total_weight() > max_share_to_split) {
        best_suggestions.push_back(BranchFactory<num_features, num_labels>());
    } else {
        // super
        if (tree->merit_preprune) {
            BranchFactory<num_features, num_labels> null_split;
            best_suggestions.push_back(null_split);
        }
        for (int i=0;i<num_features;i++) {
            if (splitters[i] != nullptr) {
                best_suggestions.push_back(splitters[i]->best_evaluated_split_suggestion(this->stats, i, min_branch_fraction));
            }
        }
    }
    return best_suggestions;
}

template <int num_features, int num_labels>
double LeafNaiveBayesAdaptive<num_features, num_labels>::calculate_promise() {
    double total_seen = sum(this->stats);
    if (total_seen > 0.0) {
        return total_seen - max_value(this->stats);
    }
    return 0.0;
}

template <int num_features, int num_labels>
void LeafNaiveBayesAdaptive<num_features, num_labels>::deactivate() {
    is_active = false;
    for (int i=0;i<num_features;i++) {
        if (splitters[i] != nullptr) {
            delete splitters[i];
            splitters[i] = nullptr;
        }
    }
}

template <int num_features, int num_labels>
void LeafNaiveBayesAdaptive<num_features, num_labels>::update_splitters(const std::vector<double>& x, int y, double w) {
    for (size_t i=0;i<x.size();i++) {
        if (splitters[i] == nullptr) {
            // should copy from saved splitter but we'll just let go
            splitters[i] = new GaussianSplitter<num_features, num_labels>(i);
        }
        splitters[i]->update(x[i], y, w);
    }
}

template <int num_features, int num_labels>
void LeafNaiveBayesAdaptive<num_features, num_labels>::prediction(std::vector<double>& proba, const std::vector<double>& x) {
    if (is_active && _nb_correct_weight >= _mc_correct_weight) {
       do_naive_bayes_prediction<num_features, num_labels>(proba, x, this->stats, splitters);
    } else {
        normalize_values_in_dict(proba, this->stats);
    }
}

template <int num_features, int num_labels>
void LeafNaiveBayesAdaptive<num_features, num_labels>::learn_one(const std::vector<double>& x, int y, double w) {
    if(is_active) {
        std::vector<double> mc_pred(num_labels, 0.0);
        normalize_values_in_dict(mc_pred, this->stats);
        if (this->stats.size() == 0 || max_index(mc_pred) == y) {
            _mc_correct_weight += w;
        }
        std::vector<double> nb_pred(num_labels, -1.0);
        do_naive_bayes_prediction<num_features, num_labels>(nb_pred, x, this->stats, splitters);
        if (max_index(nb_pred) == y) {
            _nb_correct_weight += w;
        }
    }
    // learn
    this->stats[y] += w;
    if(is_active) {
        update_splitters(x, y, w);
    }
}

template <int num_features, int num_labels>
int estimate_tree_memory_bytes(HoeffdingTree<num_features, num_labels>* tree) {
    int total = 0;
    // this
    total += sizeof(HoeffdingTreeClassifier<num_features, num_labels>);
    // classifier-classes
    total += num_features * sizeof(int);
    // root
    std::queue<BranchOrLeaf<num_features, num_labels>*> q;
    q.push(tree->_root);
    while (!q.empty()) {
        BranchOrLeaf<num_features, num_labels>* b = q.front();
        q.pop();
        if (b->is_leaf) {
            total += estimate_leaf_memory_bytes<num_features, num_labels>();
        } else {
            total += estimate_branch_memory_bytes<num_features, num_labels>();
            NumericBinaryBranch<num_features, num_labels>* branch = static_cast<NumericBinaryBranch<num_features, num_labels>*>(b);
            q.push(branch->children[0]);
            q.push(branch->children[1]);
        }
    }
    return total;
}

template <int num_features, int num_labels>
constexpr int estimate_branch_memory_bytes() {
    int total = 0;
    // this
    total += sizeof(NumericBinaryBranch<num_features, num_labels>);
    // stats
    total += num_features * (sizeof(int) + sizeof(double));

    return total;
}

template <int num_features, int num_labels>
constexpr int estimate_leaf_memory_bytes() {
    int total = 0;

    // this
    total += sizeof(LeafNaiveBayesAdaptive<num_features, num_labels>);

    // splitters
    total += num_features * sizeof(GaussianSplitter<num_features, num_labels>*);

    // splitters
    const int szGaussian = sizeof(Gaussian);
    int s = 0;
    s += sizeof(GaussianSplitter<num_features, num_labels>);
    s += num_labels * szGaussian;
    s += num_labels * sizeof(double);
    s += num_labels * sizeof(double);
    total += s * num_features;

    // stats
    total += num_features * (sizeof(int) + sizeof(double));

    return total;
}

template <int num_features, int num_labels>
void RandomLeafNaiveBayesAdaptive<num_features, num_labels>::update_splitters(const std::vector<double>& x, int y, double w) {
    if (feature_indices.size() == 0) {
        _sample_features(feature_indices, max_features);
    }
    for (size_t i=0;i<feature_indices.size();i++) {
        if (this->splitters[i] == nullptr) {
            // should copy from saved splitter but we'll just let go
            this->splitters[i] = new GaussianSplitter<num_features, num_labels>(i);
        }
        this->splitters[i]->update(x[i], y, w);
    }
}
}

# endif