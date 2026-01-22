#include <cstdio>
#include <deque>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <functional> // for std::greater
#include <cstdint>   // for uint64_t

#include "rivercpp/ARFClassifier.h"
#include "rivercpp/PipelineClassifier.h"
#include "rivercpp/StandardScaler.h"
#include "rivercpp/drift/DetectorConcept.h"
#include "rivercpp/drift/DDM.h"
#include "rivercpp/drift/HDDM_W.h"
#include "rivercpp/drift/PageHinckley.h"

constexpr int WINDOW_SIZE = 5000;
constexpr double HOT_PERCENT = 0.2;
constexpr int NUM_FEATURES = 7; 

std::unordered_map<int, int> last_access;
std::unordered_map<int, int> block_freq;

struct WindowItem {
    std::vector<double> x;
    int ts;
    int block;
    int y_pred;
    bool is_hot_ground_truth;
    int block_freq_at_event;
};

std::vector<double> extract_features(int ts, int block, int tid, const std::string& op_str, int size, int count, uint64_t pc) {
    std::vector<double> features(NUM_FEATURES);

    if (last_access.count(block)) {
        features[0] = std::log1p(ts - last_access[block]);
    } else {
        features[0] = 100.0;
    }

    features[1] = static_cast<double>(block_freq.count(block) ? block_freq[block] : 0);

    features[2] = static_cast<double>(tid);

    features[3] = (op_str == "write") ? 1.0 : 0.0;

    features[4] = std::log1p(static_cast<double>(size));

    features[5] = static_cast<double>(count);

    features[6] = static_cast<double>(std::hash<uint64_t>{}(pc));

    return features;
}

int main() {
    int total = 0;
    std::deque<WindowItem> window;
    
    // 一些常见组合
    // DetectorFactory<ADWIN<5>, 0.01>, DetectorFactory<ADWIN<5>, 0.001>
    // DetectorFactory<DDM, 2.0>, DetectorFactory<DDM, 3.0>
    // DetectorFactory<HDDM_W, 0.005>, DetectorFactory<HDDM_W, 0.001>
    rivercpp::Classifier* model = new rivercpp::PipelineClassifier(
        new rivercpp::StandardScaler<NUM_FEATURES>(), 
        new rivercpp::ARFClassifier<NUM_FEATURES, 2, 
            rivercpp::DetectorFactory<rivercpp::DDM, 2.0>, 
            rivercpp::DetectorFactory<rivercpp::DDM, 3.0> >(10, 5, 42) // (n_models, max_features, seed)
    );

    rivercpp::Accuracy<2> metric;
    
    FILE* f = fopen("DNN_dynamic.trace", "r");
    if (!f) {
        perror("Error opening trace file");
        return 1;
    }

    int n_instr, size, page, tid, count;
    uint64_t pc;
    char op_str[10];
    char p_str[10]; // "p" column
    char hot_str[5];
    
    auto begin = std::chrono::high_resolution_clock::now();

    while (fscanf(f, "%d %9s %d %d %d %lx %9s %4s\n", &tid, op_str, &size, &page, &count, &pc, p_str, hot_str) != EOF) {
        int ts = total++;
        int block = page;
        bool is_hot = (hot_str[0] == 'h');

        std::vector<double> x = extract_features(ts, block, tid, op_str, size, count, pc);
        
        int y_pred = model->predict_one(x);

        int current_freq = block_freq.count(block) ? block_freq[block] : 0;
        window.push_back({x, ts, block, y_pred, is_hot, current_freq});

        block_freq[block]++;
        last_access[block] = ts;

        if (window.size() > WINDOW_SIZE) {
            WindowItem old_item = window.front();
            window.pop_front();

            block_freq[old_item.block]--;
            if (block_freq[old_item.block] == 0) {
                block_freq.erase(old_item.block);
            }

            if (!block_freq.empty()) {
                std::vector<int> freqs;
                freqs.reserve(block_freq.size());
                for (const auto& kv : block_freq) {
                    freqs.push_back(kv.second);
                }

                int hot_n = static_cast<int>(std::ceil(freqs.size() * HOT_PERCENT));
                if (hot_n > 0 && hot_n <= freqs.size()) {
                    std::nth_element(freqs.begin(), freqs.begin() + hot_n - 1, freqs.end(), std::greater<int>());
                    int thres = freqs[hot_n - 1];

                    bool y_true = (old_item.block_freq_at_event >= thres);

                    model->learn_one(old_item.x, y_true ? 1 : 0);
                }
            }

            metric.update(old_item.is_hot_ground_truth ? 1 : 0, old_item.y_pred);
        }

        if (total % 100000 == 0) {
            printf("%d records processed.%lf\n", total, metric.get());
        }
    }
    
    fclose(f);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    printf("Total records: %d\n", total);
    printf("Accuracy: %lf\n", metric.get());
    printf("Time measured: %lld ns.\n", elapsed.count());
    
    delete model;
    return 0;
}