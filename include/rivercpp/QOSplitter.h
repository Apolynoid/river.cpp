# ifndef QO_SPLITTER_H
# define QO_SPLITTER_H

# include <cassert>
# include <map>

# include "drift/stats.h"

namespace rivercpp {
class BranchFactoryRegression {
public:
    double threshold = -1.0;
    double merit = std::numeric_limits<double>::lowest();
    int feature = -1;
    std::vector<Var> children_stats;
    BranchFactoryRegression(const double merit=std::numeric_limits<double>::lowest(), 
        const int feature=-1, const double threshold=-1.0, const std::vector<Var>& children_stats={}) 
        : threshold(threshold), merit(merit), feature(feature), children_stats(children_stats) {}
    bool operator<(const BranchFactoryRegression& rhs) const { return merit < rhs.merit; }
    bool operator==(const BranchFactoryRegression& rhs) const { return merit == rhs.merit; }
};


template <int min_samples_split=5>
class VarianceRatioSplitCriterion {
public:
    static double range_of_merit(const Var& pre_split_dist) { return 1.0; } 
    static double merit_of_split(const Var& pre_split_dist, const std::vector<Var>& post_split_dist) {
        double vr = 0.0;
        double n = pre_split_dist.mean.n;

        int count = 0;
        for (const Var& v : post_split_dist) {
            int n_i = v.mean.n;
            if (n_i >= min_samples_split) count += 1;
        }
        if (static_cast<size_t>(count) == post_split_dist.size()) {
            vr = 1.0;
            double var = pre_split_dist.get();
            for (const Var& v : post_split_dist) {
                int n_i = v.mean.n;
                vr -= (n_i / n) * v.get() / var;
            }
        }
        return vr;
    }
    static int select_best_branch(const std::vector<Var>& children_stats) {  
        double n0 = children_stats[0].mean.n;
        double n1 = children_stats[1].mean.n;

        double vr0 = n0 * children_stats[0].get();
        double vr1 = n1 * children_stats[1].get();

        return vr0 <= vr1 ? 0 : 1;
    }
};

struct Slot {
public:
    Mean x_stats;
    Var y_stats;
    Slot(double x, double y, double w=1.0) {
        x_stats.update(x, w);
        y_stats.update(y, w);
    }
    Slot& operator+=(const Slot& o) {
        x_stats += o.x_stats;
        y_stats += o.y_stats;
        return *this;
    }
    void update(double x, double y, double w=1.0) {
        x_stats.update(x, w);
        y_stats.update(y, w);
    }
};

class FeatureQuantizer {
private:
    double radius;
    std::map<int, Slot> hash;
public:
    FeatureQuantizer(double radius) : radius(radius) {}
    void update(double x, double y, double w=1.0) {
        int index = static_cast<int>(std::floor(x / radius));
        auto [it, inserted] = hash.try_emplace(index, x, y, w);
        if (!inserted) {
            it->second.update(x, y, w);
        }
    }
    size_t size() const {
        return hash.size();
    }

    struct StateYield {
        double x;
        Var left_stats;
    };
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = StateYield;
        using difference_type   = std::ptrdiff_t;
        using pointer           = const StateYield*;
        using reference         = const StateYield&;

    private:
        std::map<int, Slot>::const_iterator current_it;
        std::map<int, Slot>::const_iterator end_it;
        
        Var aux_stats;
        StateYield current_yield;

        void update_state() {
            if (current_it != end_it) {
                aux_stats += current_it->second.y_stats;
                
                current_yield.x = current_it->second.x_stats.get();
                current_yield.left_stats = aux_stats;
            }
        }

    public:
        Iterator(std::map<int, Slot>::const_iterator start, 
                 std::map<int, Slot>::const_iterator end) 
            : current_it(start), end_it(end) {
            update_state(); 
        }
        reference operator*() const { return current_yield; }
        pointer operator->() const { return &current_yield; }
        Iterator& operator++() {
            ++current_it;
            update_state();
            return *this;
        }
        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }
        friend bool operator==(const Iterator& a, const Iterator& b) { 
            return a.current_it == b.current_it; 
        }
        friend bool operator!=(const Iterator& a, const Iterator& b) { 
            return a.current_it != b.current_it; 
        }
    };

    Iterator begin() const { 
        return Iterator(hash.begin(), hash.end()); 
    }
    
    Iterator end() const { 
        return Iterator(hash.end(), hash.end()); 
    }
};

// always ban multiway
template <int num_features>
class QOSplitter {
private:
    double radius;
    FeatureQuantizer _quantizer;
public:
    QOSplitter(double radius=0.25) : radius(radius), _quantizer(radius) {
        assert((radius > 0.0) && "radius must be positive");
    }
    void update(double att_val, double target_val, double w=1.0) {
        _quantizer.update(att_val, target_val, w);
    }
    BranchFactoryRegression best_evaluated_split_suggestion(const Var& pre_split_dist, int att_idx) {
        BranchFactoryRegression candidate;
        if (_quantizer.size() == 1) return candidate;
        auto it = _quantizer.begin();
        auto end = _quantizer.end();
        double prev_x = it->x;
        ++it;
        for (;it!=end;++it) {
            Var right_stats = pre_split_dist - it->left_stats;
            std::vector<Var> post_split_dists = {it->left_stats, right_stats}; 
            double merit = VarianceRatioSplitCriterion<>::merit_of_split(pre_split_dist, post_split_dists);
            if (merit > candidate.merit) {
                candidate = BranchFactoryRegression(merit, att_idx, (prev_x + it->x) / 2.0, post_split_dists);
            }
            prev_x = it->x;
        }
        return candidate;
    }
};
}

# endif