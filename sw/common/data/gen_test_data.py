#!/usr/bin/env python3
"""
gen_test_data.py — Generate synthetic MNIST-format IDX binary files.

Creates tiny fake data (100 images, 28x28, random pixels, random labels)
so you can test the C++ binary shape pipeline without real MNIST data.

Usage:
    python3 gen_test_data.py [output_dir]
    output_dir defaults to . (current directory — sw/common/data/)
"""
import struct
import numpy as np
import os
import sys

def write_images_idx(path, images):
    """Write images in MNIST IDX3 format (big-endian)."""
    n, rows, cols = images.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('>IIII', 0x00000803, n, rows, cols))
        f.write(images.astype(np.uint8).tobytes())
    print(f"  Written {n} images → {path}")

def write_labels_idx(path, labels):
    """Write labels in MNIST IDX1 format (big-endian)."""
    n = len(labels)
    with open(path, 'wb') as f:
        f.write(struct.pack('>II', 0x00000801, n))
        f.write(labels.astype(np.uint8).tobytes())
    print(f"  Written {n} labels → {path}")

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(0)
    n = 100  # small enough to run fast

    # Random pixel data (uint8 0-255)
    images = np.random.randint(0, 256, size=(n, 28, 28), dtype=np.uint8)
    # Random labels 0-9
    labels = np.random.randint(0, 10, size=n, dtype=np.uint8)

    write_images_idx(os.path.join(out_dir, "test-images-idx3-ubyte"), images)
    write_labels_idx(os.path.join(out_dir, "test-labels-idx1-ubyte"), labels)

    print(f"\nSynthetic test data written to {out_dir}/")
    print("Run from sw/float/ or sw/quantized/:")
    print("  ./mnist_cnn ../common/data/test-images-idx3-ubyte ../common/data/test-labels-idx1-ubyte")
    print("Expected accuracy ~10% (random weights + random labels).")

if __name__ == "__main__":
    main()
