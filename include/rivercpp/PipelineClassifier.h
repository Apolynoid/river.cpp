# ifndef PIPELINE_CLASSIFIER_H
# define PIPELINE_CLASSIFIER_H

# include <stdexcept>

# include "Classifier.h"
# include "Transformer.h"

namespace rivercpp {
class PipelineClassifier : public Classifier {
private:
    Transformer* transformer;
    Classifier* classifier;
public:
    PipelineClassifier(const PipelineClassifier& other) = delete;
    PipelineClassifier& operator=(const PipelineClassifier& other) = delete;
    PipelineClassifier(Transformer* transformer, Classifier* classifier) : 
        transformer(transformer), classifier(classifier) {}
    ~PipelineClassifier() {
        delete transformer;
        delete classifier;
    }
    void learn_one(const std::vector<double>& x, int y, double w=1.0) override {
        transformer->learn_one(x, y);
        classifier->learn_one(transformer->transform_one(x), y, w);
    }
    int predict_one(const std::vector<double>& x) override {
        return classifier->predict_one(transformer->transform_one(x));
    }
    std::vector<double> predict_proba_one(const std::vector<double>& x) override {
        throw std::runtime_error("Prediction Proba Not Implied!");
    }
};
}

# endif