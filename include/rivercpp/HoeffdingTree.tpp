# ifndef HOEFFDING_TREE_TPP
# define HOEFFDING_TREE_TPP

# include "HoeffdingTree.h"

# include <vector>
# include <algorithm>

# include "TreeBase.h"

namespace rivercpp {
template <int num_features, int num_labels>
void HoeffdingTree<num_features, num_labels>::estimate_leaves() {
    _active_leaf_size_estimate = estimate_leaf_memory_bytes<num_features, num_labels>();
    _inactive_leaf_size_estimate = estimate_leaf_memory_bytes<num_features, num_labels>();
}

template <int num_features, int num_labels>
void HoeffdingTree<num_features, num_labels>::_enforce_size_limit() {
    double tree_size = _size_estimate_overhead_fraction + (
        _active_leaf_size_estimate
        + _n_inactive_leaves * _inactive_leaf_size_estimate
    );
    if (_n_inactive_leaves > 0 || tree_size > _max_byte_size) {
        if (stop_mem_management) {
            _growth_allowed = false;
            return;
        }
    }
    std::vector<LeafNaiveBayesAdaptive<num_features, num_labels>*> leaves = _root->iter_leaves();
    std::sort(leaves.begin(), leaves.end(), [](LeafNaiveBayesAdaptive<num_features, num_labels>* a, LeafNaiveBayesAdaptive<num_features, num_labels>* b) { 
        return a->calculate_promise() < b->calculate_promise();});
    size_t max_active = 0;
    while (max_active < leaves.size()) {
        max_active++;
        if ((max_active * _active_leaf_size_estimate + 
            (leaves.size() - max_active) * _inactive_leaf_size_estimate)
            * _size_estimate_overhead_fraction > _max_byte_size) {
            max_active--;
            break;
        }
    }
    int cutoff = leaves.size() - max_active;
    for (int i=0;i<cutoff;i++) {
        if (leaves[i]->is_active) {
            leaves[i]->deactivate();
            _n_inactive_leaves++;
            _n_active_leaves--;
        }
    }
    for (size_t i=cutoff;i<leaves.size();i++) {
        if (!leaves[i]->is_active && leaves[i]->depth < max_depth) {
            leaves[i]->is_active = true;
            _n_active_leaves++;
            _n_inactive_leaves--;
        }
    }
}

template <int num_features, int num_labels>
void HoeffdingTree<num_features, num_labels>::_estimate_model_size() {
    int model_size = estimate_tree_memory_bytes<num_features, num_labels>(this);
    if (model_size > _max_byte_size) {
        _enforce_size_limit();
    }
}
}

# endif