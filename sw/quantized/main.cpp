#include <iostream>
#include <string>
#include "mnist_loader.h"
#include "cnn.h"

// ---------------------------------------------------------------------------
// main.cpp — Quantized MNIST inference
//
// Usage:
//   ./mnist_cnn_q <images.idx> <labels.idx> <weights_dir>
//
// Steps:
//   1. Load MNIST test data (as float, same as float model)
//   2. Load float weights from common/weights/
//   3. Calibrate activation scales on first 100 images (float forward pass)
//   4. Quantize model (float32 → int8 weights, int32 biases)
//   5. Run pure-integer inference on all 10,000 images
//   6. Report accuracy (compare with float model's 98.75%)
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <images.idx> <labels.idx> <weights_dir>\n"
                  << "\n  Weights directory required for quantized model.\n"
                  << "  (No random-weight mode — quantization needs real weights.)\n";
        return 1;
    }

    std::string images_path = argv[1];
    std::string labels_path = argv[2];
    std::string weights_dir = argv[3];

    // -----------------------------------------------------------------------
    // 1. Load MNIST data
    // -----------------------------------------------------------------------
    MNISTData data;
    if (!load_mnist(images_path, labels_path, data)) {
        std::cerr << "Failed to load MNIST data.\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    // 2. Build quantized model
    // -----------------------------------------------------------------------
    QuantizedCNN cnn;

    if (!cnn.load_weights(weights_dir)) {
        std::cerr << "[main] Failed to load weights from: " << weights_dir << "\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    // 3. Calibrate (float forward on 100 images → activation scales)
    // -----------------------------------------------------------------------
    std::cout << "[main] Calibrating quantization on 100 images...\n";
    cnn.calibrate(data, 100);

    // -----------------------------------------------------------------------
    // 4. Quantize (float → int8/int32)
    // -----------------------------------------------------------------------
    cnn.quantize_model();
    cnn.print_quant_params();

    // -----------------------------------------------------------------------
    // 5. Run quantized inference
    // -----------------------------------------------------------------------
    int correct = 0;
    int total   = data.num_images;

    std::cout << "[main] Running quantized inference on " << total << " images...\n";

    for (int n = 0; n < total; n++) {
        int predicted = cnn.forward(data.images[n].data());
        if (predicted == data.labels[n]) correct++;

        if ((n + 1) % 1000 == 0) {
            float acc = 100.0f * correct / (n + 1);
            std::cout << "  [" << (n + 1) << "/" << total
                      << "]  accuracy so far: " << acc << "%\n";
        }
    }

    // -----------------------------------------------------------------------
    // 6. Final report
    // -----------------------------------------------------------------------
    float accuracy = 100.0f * correct / total;
    std::cout << "\n========================================\n";
    std::cout << "  Quantized model (int8)\n";
    std::cout << "  Total images : " << total    << "\n";
    std::cout << "  Correct      : " << correct  << "\n";
    std::cout << "  Accuracy     : " << accuracy << "%\n";
    std::cout << "  (Float ref   : 98.75%)\n";
    std::cout << "========================================\n";

    return 0;
}
