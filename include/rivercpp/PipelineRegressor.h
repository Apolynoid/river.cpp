# ifndef PIPELINE_REGRESSOR_H
# define PIPELINE_REGRESSOR_H

# include <stdexcept>

# include "Regressor.h"
# include "Transformer.h"

namespace rivercpp {
class PipelineRegressor : public Regressor {
private:
    Transformer* transformer;
    Regressor* regressor;
public:
    PipelineRegressor(const PipelineRegressor& other) = delete;
    PipelineRegressor& operator=(const PipelineRegressor& other) = delete;
    PipelineRegressor(Transformer* transformer, Regressor* regressor) : 
        transformer(transformer), regressor(regressor) {}
    ~PipelineRegressor() {
        delete transformer;
        delete regressor;
    }
    void learn_one(const std::vector<double>& x, double y) override {
        transformer->learn_one(x, y);
        regressor->learn_one(transformer->transform_one(x), y);
    }
    double predict_one(const std::vector<double>& x) override {
        return regressor->predict_one(transformer->transform_one(x));
    }
};
}

# endif