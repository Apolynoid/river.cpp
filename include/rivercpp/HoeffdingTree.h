# ifndef HOEFFDING_TREE_H
# define HOEFFDING_TREE_H

# include <cmath>

namespace rivercpp {
template <int num_features, int num_labels>
class BranchOrLeaf;

template <int num_features, int num_labels>
class HoeffdingTree {
protected:
    int max_depth;
    bool binary_split;
    double max_size;
    int memory_estimate_period;
    bool stop_mem_management;

    double _max_byte_size;
    int _n_active_leaves = 0;
    int _n_inactive_leaves = 0;
    double _inactive_leaf_size_estimate = 0.0;
    double _active_leaf_size_estimate = 0.0;
    double _size_estimate_overhead_fraction = 1.0;
    bool _growth_allowed = true;
    int _train_weight_seen_by_model = 0;
public:
    bool merit_preprune;
    BranchOrLeaf<num_features, num_labels>* _root = nullptr;
    HoeffdingTree(int max_depth = 980,
        bool binary_split = false,
        double max_size = 100.0,
        int memory_estimate_period = 1000000,
        bool stop_mem_management = false,
        bool remove_poor_attrs = false,
        bool merit_preprune = false) : 
        max_depth(max_depth),
        binary_split(binary_split),
        max_size(max_size), 
        memory_estimate_period(memory_estimate_period),
        stop_mem_management(stop_mem_management),
        _max_byte_size(max_size * (1 << 20)),
        merit_preprune(merit_preprune) {
        estimate_leaves();
    }

    void estimate_leaves();

    double _hoeffding_bound(double range_val, double confidence, double n) {
        return range_val * std::sqrt(-std::log(confidence) / (2.0 * n));
    }

    void _enforce_size_limit();
    void _estimate_model_size();
};
}

# endif