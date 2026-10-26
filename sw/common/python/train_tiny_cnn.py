#!/usr/bin/env python3
"""
train_tiny_cnn.py — Train the Tiny CNN on MNIST using PyTorch.

Architecture (must match cnn.cpp exactly):
    Input    1×28×28
    CONV1    4 filters, 3×3, no pad  → 4×26×26  + ReLU
    MaxPool1 2×2                     → 4×13×13
    CONV2    8 filters, 3×3, no pad  → 8×11×11  + ReLU
    MaxPool2 2×2                     →  8×5×5
    Flatten                          → 200
    FC1      200→64                  + ReLU
    FC2      64→10                   (logits)

After training, weights are exported to ../weights/ as raw float32
binary files — one file per weight/bias tensor — ready for the C++ model.

Usage:
    python3 train_tiny_cnn.py [--epochs N] [--data_dir DIR] [--weights_dir DIR]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import struct
import os
import argparse

# ---------------------------------------------------------------------------
# Model — must be identical in structure to cnn.cpp
# ---------------------------------------------------------------------------
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # padding=0 (no padding), stride=1 — matches conv2d() in C++
        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=4, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels=4,  out_channels=8, kernel_size=3, padding=0)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(200, 64)
        self.fc2   = nn.Linear(64,  10)
        self.relu  = nn.ReLU()

    def forward(self, x):
        # x: [batch, 1, 28, 28]
        x = self.relu(self.conv1(x))   # → [batch, 4, 26, 26]
        x = self.pool(x)               # → [batch, 4, 13, 13]
        x = self.relu(self.conv2(x))   # → [batch, 8, 11, 11]
        x = self.pool(x)               # → [batch, 8,  5,  5]
        x = x.view(x.size(0), -1)      # → [batch, 200]
        x = self.relu(self.fc1(x))     # → [batch, 64]
        x = self.fc2(x)                # → [batch, 10]  (raw logits)
        return x

# ---------------------------------------------------------------------------
# MNIST Dataset — reads the same IDX binary files the C++ loader reads
# ---------------------------------------------------------------------------
class MNISTDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.images = self._load_images(images_path)
        self.labels = self._load_labels(labels_path)

    @staticmethod
    def _read_be32(f):
        return struct.unpack('>I', f.read(4))[0]

    def _load_images(self, path):
        with open(path, 'rb') as f:
            magic = self._read_be32(f)
            assert magic == 0x00000803, f"Bad images magic: {magic:#010x}"
            n     = self._read_be32(f)
            rows  = self._read_be32(f)
            cols  = self._read_be32(f)
            data  = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape to [N, 1, H, W] and normalize to [0, 1] — same as C++ loader
        imgs = data.reshape(n, 1, rows, cols).astype(np.float32) / 255.0
        return torch.from_numpy(imgs)

    def _load_labels(self, path):
        with open(path, 'rb') as f:
            magic = self._read_be32(f)
            assert magic == 0x00000801, f"Bad labels magic: {magic:#010x}"
            n    = self._read_be32(f)
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return torch.from_numpy(data.copy()).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# ---------------------------------------------------------------------------
# Weight export — dumps every tensor as raw float32 binary (C order)
# Naming matches weights_io.h expectations.
# ---------------------------------------------------------------------------
def export_weights(model, weights_dir):
    os.makedirs(weights_dir, exist_ok=True)

    def save(name, tensor):
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        path = os.path.join(weights_dir, name)
        arr.tofile(path)
        print(f"  Saved {name:30s}  shape={list(tensor.shape)}  ({arr.nbytes} bytes)")

    print(f"\nExporting weights to {weights_dir}/")
    # conv1: weight [C_out=4, C_in=1, K=3, K=3], bias [4]
    save("conv1_weights.bin", model.conv1.weight)
    save("conv1_bias.bin",    model.conv1.bias)
    # conv2: weight [8, 4, 3, 3], bias [8]
    save("conv2_weights.bin", model.conv2.weight)
    save("conv2_bias.bin",    model.conv2.bias)
    # fc1: weight [64, 200], bias [64]
    save("fc1_weights.bin",   model.fc1.weight)
    save("fc1_bias.bin",      model.fc1.bias)
    # fc2: weight [10, 64], bias [10]
    save("fc2_weights.bin",   model.fc2.weight)
    save("fc2_bias.bin",      model.fc2.bias)
    print("Export complete.\n")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # Datasets
    train_ds = MNISTDataset(
        os.path.join(args.data_dir, "train-images-idx3-ubyte"),
        os.path.join(args.data_dir, "train-labels-idx1-ubyte"))
    test_ds  = MNISTDataset(
        os.path.join(args.data_dir, "t10k-images-idx3-ubyte"),
        os.path.join(args.data_dir, "t10k-labels-idx1-ubyte"))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)

    print(f"Train  : {len(train_ds)} images")
    print(f"Test   : {len(test_ds)} images\n")

    model = TinyCNN().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Reduce LR by 0.5 if test accuracy plateaus for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # -- Training --
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)

        avg_loss = total_loss / len(train_ds)

        # -- Evaluation --
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds   = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
        acc = 100.0 * correct / len(test_ds)

        scheduler.step(acc)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  test_acc={acc:.2f}%"
              + ("  ← best" if acc > best_acc else ""))

        if acc > best_acc:
            best_acc = acc
            # Save best model checkpoint
            torch.save(model.state_dict(), os.path.join(args.weights_dir, "best_model.pt"))

    print(f"\nBest test accuracy: {best_acc:.2f}%")

    # Load best checkpoint before exporting weights
    model.load_state_dict(torch.load(os.path.join(args.weights_dir, "best_model.pt"),
                                     weights_only=True))
    export_weights(model, args.weights_dir)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--data_dir",    default="../data",
                        help="Directory containing MNIST IDX files")
    parser.add_argument("--weights_dir", default="../weights",
                        help="Directory to save weights and checkpoints")
    args = parser.parse_args()

    os.makedirs(args.weights_dir, exist_ok=True)
    train(args)
