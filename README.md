# MNIST CNN on FPGA

Handwritten digit recognition (0–9) using a CNN — software model first, then hardware.

---

## Part 1 — Software (Complete)

### Requirements

- `g++` with C++17
- `python3` with `torch` and `torchvision`

### Run

**1. Build the C++ binary**
```bash
cd sw/
make
```

**2. Download MNIST data**
```bash
cd sw/data/
python3 download_mnist.py
```

**3. Train the model and export weights**
```bash
cd sw/python/
python3 train_tiny_cnn.py
```

**4. Run inference**
```bash
cd sw/
./mnist_cnn data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte weights/
```

Expected output: **~98.7% accuracy** on 10,000 test images.

### Other useful commands

All run from `sw/`:

```bash
# Remove built objects and binary
make clean

# Rebuild from scratch
make clean && make

# Run inference with trained weights (shortcut for step 4)
make run

# Smoke test — runs with random weights to verify shapes only (~10% accuracy expected)
make smoke
```

### Verify C++ matches PyTorch

```bash
cd sw/python/
python3 verify_output.py
```

---

## Part 2 — Hardware (In Progress)

> SystemVerilog implementation of each CNN layer on FPGA.
> To be added as `hw/` is built out.

