# ifndef HOT_LIST_H
# define HOT_LIST_H

# include <cmath>
# include <vector>
# include <functional>
# include <algorithm>

class HotList {
    static constexpr double rate = 0.8;
    std::vector<double> p80_max_heap;
    std::vector<double> p20_min_heap;
    int max_length;
public:
    int length = 0;
    HotList(int max_length) : max_length(max_length) {}
    double get_p80_heat() {
        if (p80_max_heap.empty()) return 0.0;
        return p80_max_heap[0];
    }
    void insert(double heat) {
        if (length >= max_length) {
            // should not add new heat unless it's greater than max_heap
            if (heat <= p80_max_heap[0]) return;
            p80_max_heap.back() = heat;
            std::push_heap(p80_max_heap.begin(), p80_max_heap.end(), std::less<double>());
        } else {
            length++;
            int max_heap_length = std::ceil(length * rate);
            if (max_heap_length > p80_max_heap.size()) {
                // insert into p80_max_heap
                if (p20_min_heap.empty() || heat <= p20_min_heap[0]) {
                    p80_max_heap.push_back(heat);
                    std::push_heap(p80_max_heap.begin(), p80_max_heap.end(), std::less<double>());
                } else {
                    // we must push it into p20 & move the least in p20 to p80
                    p80_max_heap.push_back(p20_min_heap[0]);
                    std::push_heap(p80_max_heap.begin(), p80_max_heap.end(), std::less<double>());
                    std::pop_heap(p20_min_heap.begin(), p20_min_heap.end(), std::greater<double>());
                    p20_min_heap.back() = heat;
                    std::push_heap(p20_min_heap.begin(), p20_min_heap.end(), std::greater<double>());
                }
            } else {
                // insert into p20_min_heap
                // we assume now p80 shoule never be empty
                if (heat >= p80_max_heap[0]) {
                    p20_min_heap.push_back(heat);
                    std::push_heap(p20_min_heap.begin(), p20_min_heap.end(), std::greater<double>());
                } else {
                    p20_min_heap.push_back(p80_max_heap[0]);
                    std::push_heap(p20_min_heap.begin(), p20_min_heap.end(), std::greater<double>());
                    std::pop_heap(p80_max_heap.begin(), p80_max_heap.end(), std::less<double>());
                    p80_max_heap.back() = heat;
                    std::push_heap(p80_max_heap.begin(), p80_max_heap.end(), std::less<double>());
                }
            }
        }
    }
};

# endif