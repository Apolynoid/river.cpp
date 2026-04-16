#include <cstdio>
#include <chrono>
#include <string>

#include "rivercpp/io/CSVReader.h"
#include "rivercpp/AMRules.h"
#include "rivercpp/Metrics.h"
#include "rivercpp/PipelineRegressor.h"
#include "rivercpp/StandardScaler.h"
#include "rivercpp/drift/DetectorConcept.h"
#include "rivercpp/drift/DDM.h"

constexpr int NUM_FEATURES = 6; 

int main() {
    std::string data_path = "../data/trump_approval.csv";
    rivercpp::CSVReader<double> reader(data_path);

    rivercpp::Regressor* model = new rivercpp::PipelineRegressor(
        new rivercpp::StandardScaler<NUM_FEATURES>(), 
        new rivercpp::AMRules<NUM_FEATURES>()
    );

    rivercpp::MSE metric;

    auto begin = std::chrono::high_resolution_clock::now();

    while(reader.next()) {
        double pred = model->predict_one(reader.features);
        metric.update(reader.label, pred);
        model->learn_one(reader.features, reader.label);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    printf("MSE: %lf\n", metric.get());
    printf("Time measured: %lld ms.\n", static_cast<long long>(elapsed.count()));

    delete model;
    return 0;
}