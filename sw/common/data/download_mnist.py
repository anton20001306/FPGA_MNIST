#!/usr/bin/env python3
"""
download_mnist.py — Download the real MNIST dataset.

Downloads the 4 standard IDX files from a public mirror and saves them
uncompressed in the current directory (sw/common/data/) ready for the C++ inference binary.

Usage:
    python3 download_mnist.py [output_dir]
    output_dir defaults to . (current directory — sw/common/data/)
"""
import urllib.request
import gzip
import os
import sys

MNIST_URL_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

def download_and_decompress(url, out_path):
    gz_path = out_path + ".gz"
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, gz_path)
    print(f"  Decompressing → {out_path}")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            f_out.write(f_in.read())
    os.remove(gz_path)

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    for fname in FILES:
        out_name = fname.replace(".gz", "")
        out_path = os.path.join(out_dir, out_name)
        if os.path.exists(out_path):
            print(f"  Already exists, skipping: {out_path}")
            continue
        url = MNIST_URL_BASE + fname
        download_and_decompress(url, out_path)

    print("\nDone! MNIST data downloaded to", out_dir)
    print("Test set inference (run from sw/float/ or sw/quantized/):")
    print("  ./mnist_cnn ../common/data/t10k-images-idx3-ubyte ../common/data/t10k-labels-idx1-ubyte ../common/weights/")
    print("\nTraining set inference:")
    print("  ./mnist_cnn ../common/data/train-images-idx3-ubyte ../common/data/train-labels-idx1-ubyte ../common/weights/")

if __name__ == "__main__":
    main()
