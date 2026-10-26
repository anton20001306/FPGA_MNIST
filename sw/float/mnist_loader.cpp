#include "mnist_loader.h"
#include <fstream>
#include <iostream>
#include <cstdint>

// ---------------------------------------------------------------------------
// Read a big-endian 32-bit integer from the file stream.
// MNIST IDX files store all integers in big-endian format.
// ---------------------------------------------------------------------------
static uint32_t read_be32(std::ifstream& f) {
    uint8_t buf[4];
    f.read(reinterpret_cast<char*>(buf), 4);
    return (uint32_t(buf[0]) << 24) |
           (uint32_t(buf[1]) << 16) |
           (uint32_t(buf[2]) <<  8) |
           (uint32_t(buf[3]));
}

bool load_mnist(const std::string& images_path,
                const std::string& labels_path,
                MNISTData& out)
{
    // -----------------------------------------------------------------------
    // Open images file
    // -----------------------------------------------------------------------
    std::ifstream img_f(images_path, std::ios::binary);
    if (!img_f.is_open()) {
        std::cerr << "[mnist_loader] Cannot open images file: " << images_path << "\n";
        return false;
    }

    uint32_t img_magic = read_be32(img_f);
    if (img_magic != 0x00000803) {
        std::cerr << "[mnist_loader] Bad magic number in images file: "
                  << std::hex << img_magic << "\n";
        return false;
    }

    int num_images = (int)read_be32(img_f);
    int rows       = (int)read_be32(img_f);
    int cols       = (int)read_be32(img_f);
    int pixels     = rows * cols;  // 784

    // -----------------------------------------------------------------------
    // Open labels file
    // -----------------------------------------------------------------------
    std::ifstream lbl_f(labels_path, std::ios::binary);
    if (!lbl_f.is_open()) {
        std::cerr << "[mnist_loader] Cannot open labels file: " << labels_path << "\n";
        return false;
    }

    uint32_t lbl_magic = read_be32(lbl_f);
    if (lbl_magic != 0x00000801) {
        std::cerr << "[mnist_loader] Bad magic number in labels file: "
                  << std::hex << lbl_magic << "\n";
        return false;
    }

    int num_labels = (int)read_be32(lbl_f);
    if (num_labels != num_images) {
        std::cerr << "[mnist_loader] Image/label count mismatch: "
                  << num_images << " vs " << num_labels << "\n";
        return false;
    }

    // -----------------------------------------------------------------------
    // Read pixel data and normalize to [0, 1]
    // Each image stored as [1][28][28] — one channel, 28 rows, 28 cols.
    // -----------------------------------------------------------------------
    out.num_images = num_images;
    out.rows       = rows;
    out.cols       = cols;
    out.images.resize(num_images, std::vector<float>(pixels));
    out.labels.resize(num_images);

    std::vector<uint8_t> raw(pixels);
    for (int n = 0; n < num_images; n++) {
        img_f.read(reinterpret_cast<char*>(raw.data()), pixels);
        for (int p = 0; p < pixels; p++)
            out.images[n][p] = raw[p] / 255.0f;
    }

    // -----------------------------------------------------------------------
    // Read labels
    // -----------------------------------------------------------------------
    for (int n = 0; n < num_images; n++) {
        uint8_t lbl;
        lbl_f.read(reinterpret_cast<char*>(&lbl), 1);
        out.labels[n] = (int)lbl;
    }

    std::cout << "[mnist_loader] Loaded " << num_images
              << " images (" << rows << "x" << cols << ")\n";
    return true;
}
