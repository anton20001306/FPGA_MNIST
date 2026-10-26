#pragma once
#include <string>
#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// mnist_loader.h
// Parses the MNIST IDX binary file format.
//
// IDX format for images:
//   offset 0:  magic number  0x00000803  (big-endian int32)
//   offset 4:  num_images               (big-endian int32)
//   offset 8:  num_rows   = 28          (big-endian int32)
//   offset 12: num_cols   = 28          (big-endian int32)
//   offset 16: pixel data (uint8, row-major, 28*28 per image)
//
// IDX format for labels:
//   offset 0:  magic number  0x00000801  (big-endian int32)
//   offset 4:  num_labels               (big-endian int32)
//   offset 8:  label data (uint8, one per image, value 0-9)
// ---------------------------------------------------------------------------

struct MNISTData {
    int num_images;
    int rows;       // always 28
    int cols;       // always 28
    // images[i] points to 28*28 floats normalized to [0,1], CHW = [1][28][28]
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
};

// Load images and labels from the given IDX binary files.
// Pixel values are normalized to [0.0, 1.0] by dividing by 255.
// Returns false on any error (file not found, wrong magic number, etc.)
bool load_mnist(const std::string& images_path,
                const std::string& labels_path,
                MNISTData& out);
