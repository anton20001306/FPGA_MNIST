#include <iostream>
#include <string>
#include "mnist_loader.h"
#include "cnn.h"

// ---------------------------------------------------------------------------
// main.cpp — Entry point.
//
// Usage:
//   ./mnist_cnn <images.idx> <labels.idx> [weights_dir]
//
//   images.idx  : MNIST test images file  (e.g. t10k-images-idx3-ubyte)
//   labels.idx  : MNIST test labels file  (e.g. t10k-labels-idx1-ubyte)
//   weights_dir : directory with trained .bin weight files (optional).
//                 If omitted, random He-initialised weights are used
//                 (useful for smoke-testing shapes before training).
//
// The program runs inference on all loaded images and prints accuracy.
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <images.idx> <labels.idx> [weights_dir]\n";
        return 1;
    }

    std::string images_path  = argv[1];
    std::string labels_path  = argv[2];
    std::string weights_dir  = (argc >= 4) ? argv[3] : "";

    // -----------------------------------------------------------------------
    // 1. Load MNIST data
    // -----------------------------------------------------------------------
    MNISTData data;
    if (!load_mnist(images_path, labels_path, data)) {
        std::cerr << "Failed to load MNIST data.\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    // 2. Build model and load weights
    // -----------------------------------------------------------------------
    TinyCNN cnn;

    if (weights_dir.empty()) {
        std::cout << "[main] No weights directory given — using random weights.\n"
                  << "       Accuracy will be ~10% (chance). This is a shape test only.\n";
        cnn.init_random_weights();
    } else {
        if (!cnn.load_weights(weights_dir)) {
            std::cerr << "[main] Failed to load weights from: " << weights_dir << "\n";
            return 1;
        }
    }

    // -----------------------------------------------------------------------
    // 3. Run inference on every image and count correct predictions
    // -----------------------------------------------------------------------
    int correct = 0;
    int total   = data.num_images;

    std::cout << "[main] Running inference on " << total << " images...\n";

    for (int n = 0; n < total; n++) {
        int predicted = cnn.forward(data.images[n].data());
        if (predicted == data.labels[n]) correct++;

        // Progress print every 1000 images
        if ((n + 1) % 1000 == 0) {
            float acc = 100.0f * correct / (n + 1);
            std::cout << "  [" << (n + 1) << "/" << total << "]  accuracy so far: "
                      << acc << "%\n";
        }
    }

    // -----------------------------------------------------------------------
    // 4. Final report
    // -----------------------------------------------------------------------
    float accuracy = 100.0f * correct / total;
    std::cout << "\n========================================\n";
    std::cout << "  Total images : " << total    << "\n";
    std::cout << "  Correct      : " << correct  << "\n";
    std::cout << "  Accuracy     : " << accuracy << "%\n";
    std::cout << "========================================\n";

    // -----------------------------------------------------------------------
    // 5. Verbose demo — print first 10 predictions with logits
    // -----------------------------------------------------------------------
    std::cout << "\nFirst 10 predictions (logits shown for debugging):\n";
    float logits[10];
    for (int n = 0; n < 10 && n < total; n++) {
        int pred = cnn.forward_verbose(data.images[n].data(), logits);
        std::cout << "  Image " << n
                  << "  label=" << data.labels[n]
                  << "  pred="  << pred
                  << (pred == data.labels[n] ? "  OK" : "  WRONG")
                  << "\n    logits: ";
        for (int k = 0; k < 10; k++)
            std::cout << logits[k] << " ";
        std::cout << "\n";
    }

    return 0;
}
