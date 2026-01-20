# ifndef CLASSIFIER_H
# define CLASSIFIER_H

# include <vector>
# include <algorithm>

class Classifier {
public:
    virtual void learn_one(const std::vector<double>& x, int y, double w=1.0) = 0;
    virtual std::vector<double> predict_proba_one(const std::vector<double>& x) = 0;
    virtual int predict_one(const std::vector<double>& x) {
        std::vector<double> proba = predict_proba_one(x);
        return std::distance(proba.begin(), std::max_element(proba.begin(), proba.end()));
    }
};

# endif