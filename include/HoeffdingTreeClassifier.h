# ifndef HOEFFDING_TREE_CLASSIFIER_H
# define HOEFFDING_TREE_CLASSIFIER_H

# include <unordered_set>
# include "Classifier.h"
# include "TreeBase.h"
# include "HoeffdingTree.h"

template<int num_features, int num_labels>
class HoeffdingTreeClassifier : public HoeffdingTree<num_features, num_labels>, public Classifier {
protected:
    std::unordered_set<int> classes;
    double delta;
    double tau;
    double min_branch_fraction;
    double max_share_to_split;
    int grace_period;
    virtual BranchOrLeaf<num_features, num_labels>* _new_leaf(LeafNaiveBayesAdaptive<num_features, num_labels>* parent=nullptr) {
        int depth;
        if (parent == nullptr) {
            depth = 0;
        } else {
            depth = parent->depth + 1;
        }
        return new LeafNaiveBayesAdaptive<num_features, num_labels>(depth);
    }
    void _attempt_to_split(LeafNaiveBayesAdaptive<num_features, num_labels>* leaf, NumericBinaryBranch<num_features, num_labels>* parent, int parent_branch) {
        if (!leaf->observed_class_distribution_is_pure()) {
            std::vector<BranchFactory<num_features, num_labels>> best_split_suggestions = leaf->best_split_suggestions(this, max_share_to_split, min_branch_fraction);
            std::sort(best_split_suggestions.begin(), best_split_suggestions.end());
            bool should_split = false;
            if (best_split_suggestions.size() < 2) {
                should_split = best_split_suggestions.size() > 0;
            } else {
                double hoeffding_bound = this->_hoeffding_bound(InfoGainSplitCriterion::range_of_merit(leaf->stats), 
                    delta, leaf->total_weight());
                const BranchFactory<num_features, num_labels>& best_suggestion = best_split_suggestions[best_split_suggestions.size() - 1];
                const BranchFactory<num_features, num_labels>& second_best_suggestion = best_split_suggestions[best_split_suggestions.size() - 2];
                if ((best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound) 
                    || (hoeffding_bound < tau)) {
                    should_split = true;
                }
                // no remove poor attr since NaiveBayesAdaptive does not support it
            }
            if (should_split) {
                const BranchFactory<num_features, num_labels>& split_decision = best_split_suggestions[best_split_suggestions.size() - 1];
                if (split_decision.feature < 0) {
                    leaf->deactivate();
                    this->_n_active_leaves--;
                    this->_n_inactive_leaves++;
                } else {
                    BranchOrLeaf<num_features, num_labels>* leaves[2] = 
                        { _new_leaf(leaf), _new_leaf(leaf) };
                    NumericBinaryBranch<num_features, num_labels>* new_split = split_decision.assemble(leaf->stats, leaf->depth, leaves);
                    this->_n_active_leaves++;
                    if (parent == nullptr) {
                        this->_root = new_split;
                    } else {
                        parent->children[parent_branch] = new_split;
                    }
                }
                this->_enforce_size_limit();
            }
        }
    }
public:
    HoeffdingTreeClassifier(int grace_period = 200, double delta = 1e-7, double tau = 0.05,
        double max_share_to_split = 0.99, 
        double min_branch_fraction = 0.01) : 
        HoeffdingTree<num_features, num_labels>(), grace_period(grace_period), delta(delta), tau(tau),
        max_share_to_split(max_share_to_split), min_branch_fraction(min_branch_fraction) {}
    void learn_one(const std::vector<double>& x, int y, double w=1.0) override {
        classes.insert(y);
        this->_train_weight_seen_by_model += w;
        if (!this->_root) {
            this->_root = _new_leaf();
            this->_n_active_leaves = 1;
        }
        BranchOrLeaf<num_features, num_labels>* pbnode = nullptr;
        BranchOrLeaf<num_features, num_labels>* bnode = nullptr;
        // find last two nodes
        bnode = this->_root;
        while (!bnode->is_leaf) {
            pbnode = bnode;
            bnode = bnode->next(x);
        }
        LeafNaiveBayesAdaptive<num_features, num_labels>* node = static_cast<LeafNaiveBayesAdaptive<num_features, num_labels>*>(bnode);
        NumericBinaryBranch<num_features, num_labels>* pnode = static_cast<NumericBinaryBranch<num_features, num_labels>*>(pbnode);
        // we assume node is always a leaf, thus no more test for multiway
        node->learn_one(x, y, w);
        if (this->_growth_allowed && node->is_active) {
            if (node->depth >= this->max_depth) {
                node->deactivate();
                this->_n_active_leaves--;
                this->_n_inactive_leaves++;
            } else {
                double weight_seen = node->total_weight();
                double weight_diff = weight_seen - node->last_split_attempt_at;
                if (weight_diff >= grace_period) {
                    int p_branch = 0;
                    if (pnode) {
                        p_branch = pnode->branch_no(x);
                    }
                    _attempt_to_split(node, pnode, p_branch);
                    node->last_split_attempt_at = weight_seen;
                }
            }
        }
        if (this->_train_weight_seen_by_model % this->memory_estimate_period == 0) {
            this->_estimate_model_size();
        }
    }

    virtual std::vector<double> predict_proba_one(const std::vector<double>& x) override {
        std::vector<double> proba(num_labels, 0.0);
        if (this->_root) {
            BranchOrLeaf<num_features, num_labels>* leaf = this->_root->traverse(x);
            leaf->prediction(proba, x);
        }
        return proba;
    }
};

# endif