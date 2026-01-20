# ifndef HEAT_PREDICTOR_H
# define HEAT_PREDICTOR_H

# include <cmath>
# include <vector>
# include <unordered_map>
# include <deque>

# include "HotList.h"
# include "ARFClassifier.h"
# include "StandardScaler.h"
# include "PipelineClassifier.h"

struct EvaluationItem {
    std::vector<double> x;
    double heat;
    int last_access;
    int pred_label;
};

template <class T>
class OrderedDict {
private:
    std::unordered_map<int, T> map;
    std::deque<int> order;
public:
    void insert(int key, T value) {
        map[key] = value;
        order.push_back(key);
    }
    T pop_front() {
        T value = map[order.front()];
        map.erase(order.front());
        order.pop_front();
        return value;
    }

    size_t size() const { return order.size(); }

    bool is_in(int key) const { return map.find(key) != map.end(); }

    T& operator[](int key) { return map[key]; }
};

template <int key_index, int ts_index>
class EvaluationQueue {
private:
    int max_size;
    double alpha;
    double heating;
    bool training;
    int ts = 0;
    int hot_list_cap;
    double hot_thred = 0.0;
    HotList hot_list;
    OrderedDict<EvaluationItem> item_dict;
    inline double _decay(double heat, double last_ts, double cur_ts) {
        return std::exp((cur_ts - last_ts) * alpha) * heat;
    }
    void _update_heat(EvaluationItem& item_data, bool access = true) {
        double decay = _decay(item_data.heat, item_data.last_access, ts);
        item_data.heat = decay + (access ? heating : 0.0);
        item_data.last_access = ts;
    }
public:
    EvaluationQueue(int max_size=100, double alpha=-0.03, double heating=200.0, bool training=true)
        : max_size(max_size), alpha(alpha), heating(heating), training(training), 
        hot_list_cap(50 * max_size), hot_list(HotList(hot_list_cap)) {}
    double p80_threshold() {
        if (hot_list.length == 0) return hot_thred;
        return hot_list.get_p80_heat();
    }
    void dequeue(std::vector<double>& item_data, int& pred_label, bool& is_hot) {
        EvaluationItem item = item_dict.pop_front();
        _update_heat(item, false);

        if (training) {
            hot_list.insert(item.heat);
        }

        item_data = item.x;
        pred_label = item.pred_label;
        is_hot = item.heat > p80_threshold();
    }
    // returns if a value is returned
    bool enqueue(const std::vector<double>& item, int label, std::vector<double>& item_data, int& pred_label, bool& is_hot) {
        int item_key = (int)(item[key_index]);
        ts = (int)(item[ts_index]);

        if (item_dict.is_in(item_key)) {
            _update_heat(item_dict[item_key]);
            return false;
        } else {
            bool return_val = false;
            if (item_dict.size() >= max_size) {
                dequeue(item_data, pred_label, is_hot);
            }
            EvaluationItem item_data = {item, heating, ts, label};
            item_dict.insert(item_key, item_data);
            return return_val;
        }
    }
};

template <int num_labels>
class HeatPredictor {
private:
    EvaluationQueue<0, 1> queue;
    Accuracy<num_labels> accuracy;
    Classifier* model;
    std::vector<double> _build_features(const std::vector<double>& item) {
        std::vector<double> feature = {std::log2(item[3]+1), item[2], 
            (int)(item[0]) % 256, (int)(item[4]) % 256, item[5]};
        return feature;
    }
public:
    HeatPredictor(int max_size=200, double alpha=-0.03, double heating=200.0, bool training=true) 
        : queue(max_size, alpha, heating, training) {
        model = new PipelineClassifier(new StandardScaler<5>(), new ARFClassifier<5, num_labels>(5, 2, 42));
    }
    int predict(int n_instr, int operation, int size, int page, int pc, int tid, int hotness, double& accu) {
        std::vector<double> item = {page, n_instr, operation, size, pc, tid};
        std::vector<double> feature = _build_features(item);
        int y_pred = model->predict_one(feature);

        std::vector<double> past_item; 
        int past_pred;
        bool is_hot;
        if (queue.enqueue(item, y_pred, past_item, past_pred, is_hot)) {
            std::vector<double> train_feature = _build_features(past_item);
            model->learn_one(train_feature, is_hot?1:0);

            accuracy.update(hotness, past_pred);
        }
        accu = accuracy.get();
        return y_pred;
    }
};

# endif