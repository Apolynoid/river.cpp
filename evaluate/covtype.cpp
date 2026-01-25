#include <cstdio>
#include <chrono>
#include <string>

#include "rivercpp/io/CSVReader.h"
#include "rivercpp/ARFClassifier.h"
#include "rivercpp/PipelineClassifier.h"
#include "rivercpp/StandardScaler.h"
#include "rivercpp/drift/DetectorConcept.h"
#include "rivercpp/drift/DDM.h"

constexpr int NUM_FEATURES = 54; 
constexpr int NUM_CLASSES = 7;

int main() {
    std::string data_path = "data/covtype.data";
    rivercpp::CSVReader reader(data_path, false);

    // DetectorFactory<ADWIN<5>, 0.01>, DetectorFactory<ADWIN<5>, 0.001>
    // DetectorFactory<DDM, 2.0>, DetectorFactory<DDM, 3.0>
    // DetectorFactory<HDDM_W, 0.005>, DetectorFactory<HDDM_W, 0.001>
    rivercpp::Classifier* model = new rivercpp::PipelineClassifier(
        new rivercpp::StandardScaler<NUM_FEATURES>(), 
        new rivercpp::ARFClassifier<NUM_FEATURES, NUM_CLASSES, 
            rivercpp::DetectorFactory<rivercpp::DDM, 2.0>, 
            rivercpp::DetectorFactory<rivercpp::DDM, 3.0> >(10, 5, 42) // (n_models, max_features, seed)
    );

    rivercpp::Accuracy<NUM_CLASSES> metric;

    auto begin = std::chrono::high_resolution_clock::now();

    while(reader.next()) {
        int pred = model->predict_one(reader.features);
        metric.update(reader.label-1, pred);
        model->learn_one(reader.features, reader.label-1);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    printf("Accuracy: %lf\n", metric.get());
    printf("Time measured: %lld ms.\n", static_cast<long long>(elapsed.count()));

    delete model;
    return 0;
}