# ifndef ARFCLASSIFIER_H
# define ARFCLASSIFIER_H

# include <iostream>
# include <vector>
# include <cmath>
# include <random>

# include "drift/DetectorConcept.h"
# include "Classifier.h"
# include "HoeffdingTreeClassifier.tpp"
# include "Metrics.h"
# include "utils.h"
# include "drift/ADWIN.h"

namespace rivercpp {
template <int num_features, int num_labels>
class BaseTreeClassifier : public HoeffdingTreeClassifier<num_features, num_labels> {
protected:
    std::default_random_engine* rng;
    int max_features;
public:
    BaseTreeClassifier(std::default_random_engine* rng=nullptr, int max_features=2, int grace_period = 200, 
        double delta = 1e-7, double tau = 0.05,
        double max_share_to_split = 0.99, 
        double min_branch_fraction = 0.01) : 
        HoeffdingTreeClassifier<num_features, num_labels>(grace_period, delta, tau, max_share_to_split, min_branch_fraction),
        rng(rng), max_features(max_features) {}   
    virtual BranchOrLeaf<num_features, num_labels>* _new_leaf(LeafNaiveBayesAdaptive<num_features, num_labels>* parent=nullptr) {
        int depth;
        if (parent == nullptr) {
            depth = 0;
        } else {
            depth = parent->depth + 1;
        }
        return new RandomLeafNaiveBayesAdaptive<num_features, num_labels>(depth, max_features, rng);
    } 
    virtual ~BaseTreeClassifier() = default;
};

// max features is always sqrt
// warning_detection & drift_detection always on
template <int num_features, int num_labels, 
    IsDetectorFactory WarningDetectorFactory=DetectorFactory<ADWIN<5>, 0.01>, 
    IsDetectorFactory DriftDetectorFactory=DetectorFactory<ADWIN<5>, 0.001> >
class ARFClassifier : public Classifier {
protected:
    std::vector<Classifier*> models;
    std::vector<BaseTreeClassifier<num_features, num_labels>*> _background;
    std::vector<typename DriftDetectorFactory::DetectorType> _drift_detectors;
    std::vector<typename WarningDetectorFactory::DetectorType> _warning_detectors;
    std::vector<Accuracy<num_labels> > _metrics;
    std::vector<int> _drift_tracker;
    std::vector<int> _warning_tracker;
    std::default_random_engine* _rng;
    int n_models;
    int max_features;
    int seed;
    int grace_period;
    int lambda_value;
    double delta;
    double tau;
    double max_share_to_split;
    double min_branch_fraction;
    
    void _init_ensemble() {
        for (Classifier* model : models) {
            if (model != nullptr)
                delete model;
        }
        models.clear();
        models.reserve(n_models);
        for (int i = 0; i < n_models; i++) {
            models.push_back(new BaseTreeClassifier<num_features, num_labels>
                (_rng, max_features, grace_period, delta, tau, max_share_to_split, min_branch_fraction));
        }
    }
    inline int _drift_detector_input(int y_true, int y_pred) {
        return (y_true == y_pred) ? 0 : 1;
    }
public:
    ARFClassifier(int n_models=10, int max_features=(int)(std::sqrt(num_features)),  
        int seed=-1, int grace_period=50, int lambda_value=6, 
        double delta=0.01, double tau = 0.05, 
        double max_share_to_split = 0.99, double min_branch_fraction = 0.01) 
        : n_models(n_models), max_features(max_features), seed(seed), 
        grace_period(grace_period), lambda_value(lambda_value), delta(delta), tau(tau), 
        max_share_to_split(max_share_to_split), min_branch_fraction(min_branch_fraction) {
        if (seed != -1) {
            _rng = new std::default_random_engine(seed);
        } else {
            _rng = new std::default_random_engine(std::random_device()());
        }
        _metrics = std::vector<Accuracy<num_labels> >(n_models);
        _background = std::vector<BaseTreeClassifier<num_features, num_labels>*>(n_models, nullptr);
        _drift_detectors = std::vector<typename DriftDetectorFactory::DetectorType>(n_models);
        _warning_detectors = std::vector<typename WarningDetectorFactory::DetectorType>(n_models);
        for (int i=0;i<n_models;i++) {
            _drift_detectors[i] = DriftDetectorFactory::create();
            _warning_detectors[i] = WarningDetectorFactory::create();
        }
        _drift_tracker = std::vector<int>(n_models, 0);
        _warning_tracker = std::vector<int>(n_models, 0);
    }
    ~ARFClassifier() {
        for (Classifier* model : models) {
            if (model != nullptr)
                delete model;
        }
        for (BaseTreeClassifier<num_features, num_labels>* model : _background) {
            if (model != nullptr)
                delete model;
        }
        if (_rng != nullptr) {
            delete _rng;
        }
    }
    void learn_one(const std::vector<double>& x, int y, double w=1.0) {
        if (models.size() == 0) {
            _init_ensemble();
        }
        for (int i=0;i<n_models;i++) {
            Classifier* model = models[i];
            int y_pred = model->predict_one(x);
            _metrics[i].update(y, y_pred);
            int k = poisson(lambda_value, _rng);
            if (k > 0) {
                if (_background[i] != nullptr) {
                    _background[i]->learn_one(x, y, k);
                }
                model->learn_one(x, y, k);

                int drift_input = _drift_detector_input(y, y_pred);
                _warning_detectors[i].update(drift_input);
                if (_warning_detectors[i].drift_detected) {
                    if (_background[i]) {
                        delete _background[i];
                    }
                    _background[i] = new BaseTreeClassifier<num_features, num_labels>
                        (_rng, max_features, grace_period, delta, tau, max_share_to_split, min_branch_fraction);
                    _warning_detectors[i] = WarningDetectorFactory::create();
                    _warning_tracker[i]++;
                }

                _drift_detectors[i].update(drift_input);
                if (_drift_detectors[i].drift_detected) {
                    if (_background[i] != nullptr) {
                        if (models[i]) {
                            delete models[i];
                        }
                        models[i] = _background[i];
                        _background[i] = nullptr;
                        _warning_detectors[i] = WarningDetectorFactory::create();
                        _drift_detectors[i] = DriftDetectorFactory::create();
                        _metrics[i] = Accuracy<num_labels>();
                    } else {
                        if (models[i]) {
                            delete models[i];
                        }
                        models[i] = new BaseTreeClassifier<num_features, num_labels>
                            (_rng, max_features, grace_period, delta, tau, max_share_to_split, min_branch_fraction);
                        _drift_detectors[i] = DriftDetectorFactory::create();
                        _metrics[i] = Accuracy<num_labels>();
                    }
                    _drift_tracker[i]++;
                }
            }
        }
    }
    std::vector<double> predict_proba_one(const std::vector<double>& x) {
        std::vector<double> proba(num_labels, 0.0);
        if (models.size() == 0) {
            _init_ensemble();
        } else {
            for (int i=0;i<n_models;i++) {
                Classifier* model = models[i];
                std::vector<double> y_proba_temp = model->predict_proba_one(x);
                double metric_value = _metrics[i].get();
                for (int j=0;j<num_labels;j++) {
                    proba[j] += (metric_value > 0.0) ? y_proba_temp[j] * metric_value : y_proba_temp[j];
                }
            }
            double total = std::accumulate(proba.begin(), proba.end(), 0.0);
            for (int i=0;i<num_labels;i++) {
                if (total > 0.0) {
                    proba[i] /= total;
                } else {
                    proba[i] = 0.0;
                }
            }
        }
        return proba;
    }
};
}

# endif