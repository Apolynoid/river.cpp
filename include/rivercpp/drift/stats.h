# ifndef STATS_H
# define STATS_H

namespace rivercpp {
class Mean {
private:
    double _mean = 0.0;
public:
    double n = 0.0;
    void update(double x, double w=1.0) {
        n += w;
        _mean += (w / n) * (x - _mean);
    }
    double get() const { return _mean; }
    Mean& operator+=(const Mean& other) {
        double old_n = n;
        n += other.n;
        _mean = (old_n * _mean + other.n * other.get()) / n;
        return *this;
    }
    Mean& operator-=(const Mean& other) {
        double old_n = n;
        n -= other.n;
        if (n > 0.0) _mean = (old_n * _mean - other.n * other.get()) / n;
        else { n = 0.0; _mean = 0.0; }
        return *this;
    }
};

class Var {
private:
    int ddof;
    double _S = 0.0;
public:
    Mean mean;
    Var(int ddof=1) : ddof(ddof) {}
    void update(double x, double w=1.0) {
        double mean_old = mean.get();
        mean.update(x, w);
        _S += w * (x - mean_old) * (x - mean.get());
    }
    double get() const { 
        if (mean.n > ddof) {
            return _S / (mean.n - ddof);
        }
        return 0.0;
    }
    Var& operator+=(const Var& other) {
        double S = _S + other._S + (mean.get() - other.mean.get()) * (mean.get() - other.mean.get()) 
            * (mean.n * other.mean.n) / (mean.n + other.mean.n);
        mean += other.mean;
        _S = S;
        return *this;
    }
    Var& operator-=(const Var& other) {
        mean -= other.mean;
        double S = _S - other._S - (mean.get() - other.mean.get()) * (mean.get() - other.mean.get()) 
            * (mean.n * other.mean.n) / (mean.n + other.mean.n);
        _S = S;
        return *this;
    }
    Var operator-(const Var& other) const {
        Var res = *this;
        res -= other;
        return res;
    }
};
}

# endif