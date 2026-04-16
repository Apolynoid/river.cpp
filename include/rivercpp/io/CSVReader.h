#ifndef IO_CSVREADER_H
#define IO_CSVREADER_H

#include <cstdio>
#include <vector>
#include <utility>
#include <string>
#include <charconv>
#include <string_view>

namespace rivercpp {

template <typename T=int>
class CSVReader {
private:
    FILE* file;
    int label_idx;
    char delimiter;
    char line_buffer[1024];
public:
    std::vector<double> features;
    T label;

    explicit CSVReader(const std::string& filename, bool skip_header = true, int label_idx = -1, char delim = ',') 
        : label_idx(label_idx), delimiter(delim) {
        file = std::fopen(filename.c_str(), "r");
        features.reserve(64); 
        // skip header
        if (skip_header) std::fgets(line_buffer, sizeof(line_buffer), file);
    }

    ~CSVReader() { if (file) std::fclose(file); }

    bool next() {
        if (!std::fgets(line_buffer, sizeof(line_buffer), file)) {
            return false;
        }

        features.clear();
        char* start = line_buffer;
        char* p = line_buffer;

        int current_col = 0;

        while (true) {
            if (*p == delimiter || *p == '\n' || *p == '\r' || *p == '\0') {
                if (p > start) {
                    double val = 0;
                    std::from_chars(start, p, val);
                    if (current_col == label_idx) label = static_cast<T>(val);
                    else features.push_back(val);
                }
                current_col++;
                if (*p == '\0' || *p == '\n' || *p == '\r') {
                    if (label_idx == -1 && !features.empty()) {
                        label = static_cast<T>(features.back());
                        features.pop_back();
                    }
                    break;
                }
                start = ++p;
            } else {
                ++p;
            }
        }

        return true;
    }
};

} // namespace river

#endif