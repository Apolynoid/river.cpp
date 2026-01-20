# ifndef MEAN_H
# define MEAN_H

class Mean {
private:
    double _mean = 0.0;
public:
    double n = 0.0;
    void update(double x, double w=1.0) {
        n += w;
        _mean += (w / n) * (x - _mean);
    }
    double get() { return _mean; }
};

# endif