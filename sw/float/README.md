# sw/ — Software Directory

This directory contains two completely separate programs — a **C++ inference engine** and a **Python training pipeline**. They do not call each other. The only connection between them is the `weights/` directory.

---

## Directory Structure

```
sw/
├── main.cpp              ← C++ program entry point
├── cnn.h / cnn.cpp       ← The full CNN model
├── conv_layer.h/.cpp     ← Convolution layer
├── pool_layer.h/.cpp     ← Max pooling layer
├── fc_layer.h/.cpp       ← Fully connected layer
├── activations.h         ← relu, softmax, argmax
├── mnist_loader.h/.cpp   ← Reads MNIST binary files
├── weights_io.h/.cpp     ← Reads/writes weight files
├── Makefile              ← Build instructions
│
├── data/                 ← MNIST image + label files + data scripts
│   ├── download_mnist.py         ← Downloads real MNIST data (run from data/)
│   ├── gen_test_data.py          ← Generates fake MNIST data (for shape testing)
│   ├── train-images-idx3-ubyte   (60,000 training images)
│   ├── train-labels-idx1-ubyte   (60,000 training labels)
│   ├── t10k-images-idx3-ubyte    (10,000 test images)
│   └── t10k-labels-idx1-ubyte    (10,000 test labels)
│
├── weights/              ← Trained weight values
│   ├── conv1_weights.bin
│   ├── conv1_bias.bin
│   ├── conv2_weights.bin
│   ├── conv2_bias.bin
│   ├── fc1_weights.bin
│   ├── fc1_bias.bin
│   ├── fc2_weights.bin
│   ├── fc2_bias.bin
│   └── best_model.pt     (PyTorch checkpoint)
│
└── python/               ← Training scripts
    ├── train_tiny_cnn.py
    └── verify_output.py
```

---

## Two Separate Programs

```
Python side                          C++ side
──────────────────────────────       ──────────────────────────────
Trains the CNN using PyTorch         Runs inference (prediction only)
Learns the best weight values        Uses the weights Python found
Writes weights to weights/*.bin      Reads weights from weights/*.bin
                    │                         │
                    └──── weights/*.bin ──────┘
                              (the bridge)
```

---

## What Each File Does

### C++ Files — Inference Engine

| File | Role |
|------|------|
| `main.cpp` | Entry point. Loads data, builds the model, runs all 10,000 images, prints accuracy |
| `cnn.h/.cpp` | The brain — owns all weight buffers and calls each layer in order: CONV1→ReLU→Pool→CONV2→ReLU→Pool→FC1→ReLU→FC2→argmax |
| `conv_layer.h/.cpp` | Does the convolution math — 6 nested loops |
| `pool_layer.h/.cpp` | Does the max pooling math |
| `fc_layer.h/.cpp` | Does the fully connected math |
| `activations.h` | `relu()`, `softmax()`, `argmax()` — inline helper functions |
| `mnist_loader.h/.cpp` | Opens the `data/` files and reads raw images into float arrays |
| `weights_io.h/.cpp` | Opens the `weights/` files and reads float values into arrays |

> `.h` files are **headers** — they declare what functions exist (like a table of contents).
> `.cpp` files are the **implementations** — the actual code that runs.

### Python Files

| File | Role |
|------|------|
| `data/download_mnist.py` | One-time download — fetches the 4 MNIST files into `data/` (run from `data/`) |
| `data/gen_test_data.py` | Makes fake random MNIST data — only for testing shapes without real data (run from `data/`) |
| `python/train_tiny_cnn.py` | Trains the CNN using PyTorch on GPU, saves weights to `weights/*.bin` |
| `python/verify_output.py` | Sanity check — runs both PyTorch and C++ on the same images, confirms outputs match |

### Makefile

Not a program — a **recipe for building**. Tells the compiler which `.cpp` files to compile and link into one binary called `mnist_cnn`.

```bash
make          # compile all .cpp files → produces ./mnist_cnn
make clean    # delete compiled files
make run      # run ./mnist_cnn with test data + trained weights
make smoke    # run ./mnist_cnn with random weights (shape test only, ~10% accuracy)
```

---

## Full Flow — From Nothing to 98.7%

```
Step 1: cd data/ && python3 download_mnist.py
        → fetches raw image/label files directly into data/

Step 2: cd python/ && python3 train_tiny_cnn.py
        → reads data/, trains on GPU, learns weights
        → writes weights/*.bin

Step 3: make
        → compiles all .cpp files → builds ./mnist_cnn binary

Step 4: ./mnist_cnn data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte weights/
        → loads images from data/
        → loads weights from weights/
        → runs each image through CONV→Pool→CONV→Pool→FC→FC
        → prints accuracy: ~98.7%
```

---

## Why C++ and Not Just Python?

The **FPGA hardware will mirror this C++ code exactly**. Each `.cpp` layer becomes a `.sv` (SystemVerilog) module doing the same arithmetic — explicit loops, fixed arrays, integer math. Python is only used for training because FPGAs cannot run PyTorch.
