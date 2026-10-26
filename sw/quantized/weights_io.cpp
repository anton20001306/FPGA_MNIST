#include "weights_io.h"
#include <fstream>
#include <iostream>

bool load_weights(const std::string& path, float* buf, int count) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[weights_io] Cannot open: " << path << "\n";
        return false;
    }
    f.read(reinterpret_cast<char*>(buf), count * sizeof(float));
    if (f.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
        std::cerr << "[weights_io] File too short: " << path
                  << " (expected " << count << " floats)\n";
        return false;
    }
    return true;
}

bool save_weights(const std::string& path, const float* buf, int count) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[weights_io] Cannot write: " << path << "\n";
        return false;
    }
    f.write(reinterpret_cast<const char*>(buf), count * sizeof(float));
    return true;
}
