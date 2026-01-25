#ifndef IO_CSVREADER_H
#define IO_CSVREADER_H

#include <cstdio>
#include <vector>
#include <utility>
#include <string>
#include <charconv>
#include <string_view>

namespace rivercpp {

class CSVReader {
private:
    FILE* file;
    char delimiter;
    char line_buffer[1024];

public:
    std::vector<double> features;
    int label = 0;

    explicit CSVReader(const std::string& filename, bool skip_header = true, char delim = ',') 
        : delimiter(delim) {
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

        while (true) {
            if (*p == delimiter || *p == '\n' || *p == '\r' || *p == '\0') {
                if (p > start) {
                    double val = 0;
                    std::from_chars(start, p, val);
                    features.push_back(val);
                }
                if (*p == '\0' || *p == '\n' || *p == '\r') break;
                start = ++p;
            } else {
                ++p;
            }
        }

        if (!features.empty()) {
            label = static_cast<int>(features.back());
            features.pop_back();
        }

        return true;
    }
};

} // namespace river

#endif