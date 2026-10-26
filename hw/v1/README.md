# CNN Accelerator v1 — BRAM-based MNIST Inference

Int8 quantized CNN accelerator that classifies 28x28 MNIST handwritten digits.
Layer-by-layer compute: CONV1 -> ReLU -> Pool1 -> CONV2 -> ReLU -> Pool2 -> FC1 -> ReLU -> FC2 -> Argmax.

## Prerequisites

- **MNIST test data** in `sw/common/data/` (t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte)
- **Trained weights** in `sw/common/weights/` (conv1_weights.bin, conv1_bias.bin, etc.)
- **Python 3** with `numpy` (for hex generation)
- **Simulator**: Questa/ModelSim or Vivado xsim (or Verilator for lint-only)

If data or weights are missing, run from the project root:
```bash
cd sw/common/data && python3 download_mnist.py
cd sw/common/python && python3 train_tiny_cnn.py
```

## Quick Start

From the `hw/v1/` directory:

```bash
# Full simulation (hex generation + compile + run) — 20 images, Questa
make sim

# Full simulation with Vivado xsim
make sim-xsim
```

This will:
1. Run `export_hex.py` to quantize weights and generate golden test vectors
2. Compile all RTL + testbench
3. Simulate: load each image, run inference, compare against golden predictions
4. Print per-image PASS/FAIL and overall results

## Makefile Targets

| Target             | Description                                          |
|--------------------|------------------------------------------------------|
| `make sim`         | Full flow: generate hex + compile + simulate (Questa)|
| `make sim-xsim`   | Full flow using Vivado xsim                          |
| `make hex`         | Only generate .hex files (no simulation)             |
| `make check`       | Syntax check only (auto-detects Questa/xvlog/Verilator)|
| `make check-questa`| Syntax check with Questa                             |
| `make check-xvlog` | Syntax check with Vivado xvlog                       |
| `make check-verilator`| Lint check with Verilator                         |
| `make clean`       | Remove all generated simulation artifacts            |

## Configurable Options

### Number of test images

Default is 20. Override with `NUM_TEST`:

```bash
make NUM_TEST=100 sim       # Test on 100 images
make NUM_TEST=1 sim         # Quick single-image test
make NUM_TEST=1000 sim      # Large test (takes longer)
```

More images = longer simulation time. Each image takes ~111K clock cycles (~1.1ms at 100 MHz).
Rough time estimates:

| NUM_TEST | Simulation time (approx) |
|----------|--------------------------|
| 1        | ~2 sec                   |
| 20       | ~35 sec                  |
| 100      | ~3 min                   |
| 1000     | ~30 min                  |

### Calibration images

The export script uses 100 calibration images by default to compute activation ranges.
To change this:

```bash
cd tb/scripts
python3 export_hex.py --num_test 20 --num_calib 200
```

### Custom data/weight paths

If your MNIST data or weights are in a different location:

```bash
cd tb/scripts
python3 export_hex.py \
    --data_dir /path/to/mnist/data \
    --weights_dir /path/to/weights \
    --out_dir ../hex \
    --num_test 50
```

### VCD waveform dump

The testbench supports VCD dump for waveform viewing (with GTKWave, etc.).
To enable, add the `+VCD` plusarg when running manually:

```bash
cd tb/hex
vsim -batch -do "run -all; quit" cnn_top_tb +VCD
gtkwave cnn_top_tb.vcd   # view waveforms
```

## Directory Structure

```
v1/
├── rtl/                     SystemVerilog source (10 modules)
│   ├── cnn_pkg.sv           Parameter package (all layer dimensions)
│   ├── mac_int8.sv          Int8 multiply-accumulate (combinational)
│   ├── requantize.sv        Int32 -> Int8 requantization (2-cycle pipeline)
│   ├── conv_engine.sv       Convolution engine (6-deep loop FSM)
│   ├── pool_engine.sv       2x2 max pooling engine
│   ├── fc_engine.sv         Fully connected engine (requant or raw output)
│   ├── argmax.sv            10-class argmax (sequential comparator)
│   ├── bram_sp.sv           Single-port BRAM (activation buffers)
│   ├── rom_sp.sv            Single-port ROM (weights/biases, $readmemh)
│   └── cnn_top.sv           Top-level: FSM + 7 BRAMs + 8 ROMs + 7 engines
├── tb/                      Testbench and verification
│   ├── cnn_top_tb.sv        Top-level testbench
│   ├── scripts/
│   │   └── export_hex.py    Quantize weights + generate hex + golden vectors
│   └── hex/                 Generated files (by export_hex.py)
│       ├── *_weights.hex    Quantized int8 weights (per layer)
│       ├── *_bias.hex       Quantized int32 biases (per layer)
│       ├── requant_params.svh   SV include: requantization multiplier/shift
│       ├── num_test.svh     SV include: NUM_TEST_IMAGES localparam
│       ├── test_images.hex  N x 784 int8 pixel values
│       └── test_expected.hex    N golden expected class labels
├── docs/
│   ├── mapping.md           C++ to SystemVerilog mapping reference
│   └── scaling.md           How to scale to larger models
├── Makefile
└── README.md                This file
```

## How It Works

### Verification flow

```
sw/common/weights/*.bin ──┐
                          ├──> export_hex.py ──> tb/hex/*.hex
sw/common/data/t10k-*   ──┘       |
                                  ├──> requant_params.svh (multiplier/shift constants)
                                  ├──> test_images.hex    (quantized input pixels)
                                  └──> test_expected.hex  (golden predictions)

                                            │
                                            v

                                     cnn_top_tb.sv
                                     ┌──────────────────────────────┐
                                     │  For each test image:        │
                                     │  1. Write 784 pixels to BRAM │
                                     │  2. Pulse start              │
                                     │  3. Wait for done            │
                                     │  4. Compare pred_class vs    │
                                     │     golden expected          │
                                     │  5. Report PASS / FAIL       │
                                     └──────────────────────────────┘
```

### Inference pipeline (inside cnn_top)

```
Input BRAM (784 x int8)
    │
    v
CONV1 engine (1->4 ch, 3x3) + ReLU ──> BRAM (2704 x int8)
    │
    v
POOL1 engine (2x2 max) ──> BRAM (676 x int8)
    │
    v
CONV2 engine (4->8 ch, 3x3) + ReLU ──> BRAM (968 x int8)
    │
    v
POOL2 engine (2x2 max) ──> BRAM (200 x int8)
    │
    v
FC1 engine (200->64) + ReLU ──> BRAM (64 x int8)
    │
    v
FC2 engine (64->10, raw int32) ──> BRAM (10 x int32)
    │
    v
Argmax ──> pred_class (4-bit digit 0-9)
```

### Performance

- ~111,122 clock cycles per inference (1 MAC/cycle)
- ~1.11 ms at 100 MHz
- ~900 inferences/sec
