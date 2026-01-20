# ifndef METRICS_H
# define METRICS_H

template <int num_labels>
class ConfusionMatrix {
private:
    double data[num_labels][num_labels] = {0.0};
    double sum_row[num_labels] = {0.0};
    double sum_col[num_labels] = {0.0};
    int n_samples = 0;
public:
    double total_weight = 0.0;
    void update(int y_true, int y_pred, double w=1.0) {
        n_samples++;
        data[y_true][y_pred] += w;
        total_weight += w;
        sum_row[y_true] += w;
        sum_col[y_pred] += w;
    }
    double total_true_positives() {
        double total = 0.0;
        for (int i=0;i<num_labels;i++) {
            total += data[i][i];
        }
        return total;
    }
};

template <int num_labels>
class Accuracy {
private:
    ConfusionMatrix<num_labels> cm;
public:
    void update(int y_true, int y_pred, double w=1.0) {
        cm.update(y_true, y_pred, w);
    }
    double get() {
        if (cm.total_weight > 0.0) {
            return cm.total_true_positives() / cm.total_weight;
        } else {
            return 0.0;
        }
    }
};

# endif