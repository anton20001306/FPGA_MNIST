#pragma once
#include <string>

// ---------------------------------------------------------------------------
// weights_io.h
// Load / save weight tensors as raw float32 binary files.
//
// File format: just a flat sequence of IEEE 754 float32 values in the
// canonical layer order (same order as the C arrays).
//
// This simple format makes it easy to:
//   - Export from PyTorch (numpy .tofile())
//   - Load into the C++ model here
//   - Eventually feed into HW testbenches
//
// Expected weight files (in a weights/ directory):
//   conv1_weights.bin  — [4][1][3][3]   = 36  floats
//   conv1_bias.bin     — [4]             =  4  floats
//   conv2_weights.bin  — [8][4][3][3]   = 288 floats
//   conv2_bias.bin     — [8]             =  8  floats
//   fc1_weights.bin    — [64][200]       = 12800 floats
//   fc1_bias.bin       — [64]            =  64 floats
//   fc2_weights.bin    — [10][64]        =  640 floats
//   fc2_bias.bin       — [10]            =  10 floats
// ---------------------------------------------------------------------------

// Load `count` floats from a binary file into `buf`.
// Returns false if the file cannot be opened or is too short.
bool load_weights(const std::string& path, float* buf, int count);

// Save `count` floats from `buf` to a binary file.
// Returns false on error.
bool save_weights(const std::string& path, const float* buf, int count);
