# Parameterized CNN Accelerator — SystemVerilog

Parameterized SystemVerilog conversion of the thedatabus.in single-layer CNN
accelerator. All modules import their defaults from `cnn_pkg.sv` — no magic
numbers anywhere. The original Verilog sources live in `../thedatabus_in/`.

---

## How to Run the Makefile

### Prerequisites

One of the following SystemVerilog compilers must be on your `PATH`:

| Tool | Command | Source |
|------|---------|--------|
| Questa / ModelSim | `vlog` | Intel FPGA / Siemens EDA |
| Vivado | `xvlog` | AMD / Xilinx |
| Verilator | `verilator` | Open-source |

### Available Targets

```
make check              # auto-detects the first available tool and runs it
make check-questa       # force Questa (vlog -sv)
make check-xvlog        # force Vivado (xvlog --sv)
make check-verilator    # force Verilator (--lint-only)
make clean              # remove all generated artifacts (work/, xsim.dir/, logs)
```

### What `make check` Does

1. Compiles `cnn_pkg.sv` (package) first — every module depends on it.
2. Compiles all 13 design modules in dependency order.
3. Reports PASS/FAIL. No simulation is run — this is a syntax and elaboration check only.

### Example

```
$ make check
=== Using Questa (vlog) ===
Compiling package...
Compiling all modules...
=== Questa syntax check PASSED ===
```

---

## Module Reference

### Pipeline Overview

```
                 weight1
                   |
activation --> [CONVOLVER] --> [ACTIVATION] --> [POOLER] --> data_out
                               (ReLU/Tanh)     (Max/Avg)
```

All modules below import `cnn_pkg` and use its values as defaults.
Parameters can still be overridden at instantiation.

---

### `cnn_pkg.sv` — Parameter Package

Central package that every module imports. Edit values here to reconfigure
the entire pipeline.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_WIDTH` | 16 | N — total bit width of activations and weights |
| `FRAC_BITS` | 12 | Q — fractional bits in fixed-point (format: 1 sign + 3 integer + 12 frac) |
| `INPUT_SIZE` | 10 | n — input activation map dimension (n x n) |
| `KERNEL_SIZE` | 3 | k — convolution kernel dimension (k x k) |
| `STRIDE` | 1 | s — convolution stride |
| `ACT_TYPE` | 0 | Activation function: 0 = ReLU, 1 = Tanh |
| `TANH_ADDR_WIDTH` | 10 | AW — tanh ROM address width (2^AW entries) |
| `TANH_DATA_WIDTH` | 16 | DW — tanh ROM data width (= DATA_WIDTH) |
| `POOL_SIZE` | 2 | p — pooling window dimension (p x p) |
| `POOL_TYPE` | 1 | 0 = average pooling, 1 = max pooling |
| `POOL_SQR_INV` | 16'b0000_0100_0000_0000 | 1/p^2 in (N,Q) format — needed for average pooling |

Derived constants (auto-calculated):

| Parameter | Formula | Description |
|-----------|---------|-------------|
| `CONV_OUT_SIZE` | (n - k + 1) / s | Convolution output dimension |
| `POOL_IN_SIZE` | = CONV_OUT_SIZE | Pooler input dimension |
| `POOL_OUT_SIZE` | POOL_IN_SIZE / p | Final output dimension |
| `CONV_PIPE_FILL` | (k-1)*n + (k-1) | Clock cycles before first valid conv output |
| `KERNEL_ELEMENTS` | k * k | Number of weights in kernel |
| `WEIGHT_BUS_WIDTH` | k*k * N | Packed weight bus width in bits |

---

### `accelerator.sv` — Top-Level Module

Wires together convolver, activation, and pooler into a single streaming pipeline.

| Parameter | Default Source | Description |
|-----------|---------------|-------------|
| `N_IMG` | `INPUT_SIZE` | Input image dimension |
| `K` | `KERNEL_SIZE` | Kernel size |
| `P` | `POOL_SIZE` | Pooling window size |
| `S` | `STRIDE` | Convolution stride |
| `PTYPE` | `POOL_TYPE` | Pooling type (0=avg, 1=max) |
| `ACT` | `ACT_TYPE` | Activation type (0=ReLU, 1=Tanh) |
| `N` | `DATA_WIDTH` | Bit width |
| `Q` | `FRAC_BITS` | Fractional bits |
| `AW` | `TANH_ADDR_WIDTH` | Tanh LUT address width |
| `DW` | `TANH_DATA_WIDTH` | Tanh LUT data width |
| `P_SQR_INV` | `POOL_SQR_INV` | 1/p^2 for average pooling |

**Ports**: `clk`, `global_rst`, `ce`, `activation[N-1:0]`, `weight1[K*K*N-1:0]`
-> `data_out[N-1:0]`, `valid_op`, `end_op`, plus debug ports `conv_out`, `conv_valid`, `conv_end`.

---

### `convolver.sv` — Streaming Convolution Engine

K*K MAC units arranged in K rows. Each row ends with a shift register of
depth (n-k) for row buffering. The input activation is broadcast to all MACs;
weights are fixed per position. Pipeline fill time: (k-1)*n + (k-1) cycles.

| Parameter | Default Source | Description |
|-----------|---------------|-------------|
| `N_IMG` | `INPUT_SIZE` | Activation map dimension |
| `K` | `KERNEL_SIZE` | Kernel dimension |
| `S` | `STRIDE` | Stride |
| `N` | `DATA_WIDTH` | Bit width |
| `Q` | `FRAC_BITS` | Fractional bits |

**Ports**: `clk`, `ce`, `global_rst`, `activation[N-1:0]`, `weight1[K*K*N-1:0]`
-> `conv_op[N-1:0]`, `valid_conv`, `end_conv`.

---

### `mac_manual.sv` — Multiply-Accumulate Unit

Computes `p = a * b + c` using fixed-point arithmetic (qmult + qadd).
Registered output for timing closure.

| Parameter | Default Source | Description |
|-----------|---------------|-------------|
| `N` | `DATA_WIDTH` | Bit width |
| `Q` | `FRAC_BITS` | Fractional bits |

**Ports**: `clk`, `sclr` (sync clear), `ce`, `a[N-1:0]` (activation),
`b[N-1:0]` (weight), `c[N-1:0]` (previous sum) -> `p[N-1:0]` (result).

---

### `qmult.sv` — Fixed-Point Multiplier

Multiplies two (N,Q) fixed-point numbers. Converts inputs to positive
magnitude, multiplies, re-quantizes to N bits, and converts back to 2's
complement if the result is negative.

| Parameter | Default Source | Description |
|-----------|---------------|-------------|
| `N` | `DATA_WIDTH` | Bit width |
| `Q` | `FRAC_BITS` | Fractional bits |

**Ports**: `clk`, `rst`, `a[N-1:0]`, `b[N-1:0]`
-> `q_result[N-1:0]`, `overflow`.

---

### `qadd.sv` — Fixed-Point Adder

Simple addition — 2's complement handles sign automatically.

| Parameter | Default Source | Description |
|-----------|---------------|-------------|
| `N` | `DATA_WIDTH` | Bit width |
| `Q` | `FRAC_BITS` | Fractional bits |

**Ports**: `a[N-1:0]`, `b[N-1:0]` -> `c[N-1:0]`.

---

### `relu.sv` — ReLU Activation

`f(x) = max(0, x)`. Checks MSB (sign bit): if negative, output zero.

| Parameter | Default Source | Description |
|-----------|---------------|-------------|
| `N` | `DATA_WIDTH` | Bit width |

**Ports**: `din_relu[N-1:0]` -> `dout_relu[N-1:0]`.

---

### `tanh_lut.sv` — Tanh Activation (LUT + Interpolation)

ROM-based tanh with linear interpolation between adjacent entries for improved
precision. Exploits symmetry (store positive half only) and saturation
(|x| > threshold outputs +/-1).

| Parameter | Default Source | Description |
|-----------|---------------|-------------|
| `AW` | `TANH_ADDR_WIDTH` | ROM address width (2^AW entries) |
| `DW` | `TANH_DATA_WIDTH` | ROM data width |
| `N` | `DATA_WIDTH` | Bit width |
| `Q` | `FRAC_BITS` | Fractional bits |

**Ports**: `clk`, `rst`, `phase[N-1:0]` -> `tanh[DW-1:0]`.

**Requires**: `tanh_data.mem` file (symlinked from `../thedatabus_in/`).

---

### `pooler.sv` — Streaming Pooling Unit

Performs p x p pooling on an m x m input. For max pooling (PTYPE=1), retains
the maximum value in each window. For average pooling (PTYPE=0), sums values
and divides by p^2 using `P_SQR_INV`.

| Parameter | Default Source | Description |
|-----------|---------------|-------------|
| `M` | `POOL_IN_SIZE` | Input dimension (post-convolution) |
| `P` | `POOL_SIZE` | Pooling window dimension |
| `N` | `DATA_WIDTH` | Bit width |
| `Q` | `FRAC_BITS` | Fractional bits |
| `PTYPE` | `POOL_TYPE` | 0 = average, 1 = max |
| `P_SQR_INV` | `POOL_SQR_INV` | 1/p^2 in (N,Q) format |

**Ports**: `clk`, `ce`, `master_rst`, `data_in[N-1:0]`
-> `data_out[N-1:0]`, `valid_op`, `end_op`.

---

### `control_logic2.sv` — Pooler Control FSM

Generates all control signals (`sel`, `rst_m`, `load_sr`, `global_rst`,
`op_en`/valid_op, `end_op`) as the pooling window moves across the input.
Handles 5 distinct cases: normal, end-of-neighborhood, end-of-row,
end-of-neighborhood on last row, and end-of-last-neighborhood.

| Parameter | Default Source | Description |
|-----------|---------------|-------------|
| `M` | `POOL_IN_SIZE` | Input matrix dimension |
| `P` | `POOL_SIZE` | Pooling window dimension |

**Ports**: `clk`, `master_rst`, `ce`
-> `sel[1:0]`, `rst_m`, `op_en`, `load_sr`, `global_rst`, `end_op`.

---

### `comparator2.sv` — Signed Comparator / Adder

For max pooling (PTYPE=1): outputs the larger of two signed fixed-point values
using full 2's complement sign-aware comparison. For average pooling (PTYPE=0):
outputs the sum of the two inputs.

| Parameter | Default Source | Description |
|-----------|---------------|-------------|
| `N` | `DATA_WIDTH` | Bit width |
| `Q` | `FRAC_BITS` | Fractional bits |
| `PTYPE` | `POOL_TYPE` | 0 = sum (for average), 1 = max |

**Ports**: `ce`, `ip1[N-1:0]`, `ip2[N-1:0]` -> `comp_op[N-1:0]`.

---

### `max_reg.sv` — Max / Accumulator Register

Holds the running maximum (or running sum for average pooling) within each
pooling neighborhood. Resets on `master_rst` or `rst_m`.

| Parameter | Default Source | Description |
|-----------|---------------|-------------|
| `N` | `DATA_WIDTH` | Bit width |

**Ports**: `clk`, `ce`, `din[N-1:0]`, `rst_m`, `master_rst` -> `reg_op[N-1:0]`.

---

### `input_mux.sv` — Pooler Input Multiplexer

Selects between shift register output (`sel=01`), max register output
(`sel=00`), or zero (`sel=10`) to feed into the comparator.

| Parameter | Default Source | Description |
|-----------|---------------|-------------|
| `N` | `DATA_WIDTH` | Bit width |

**Ports**: `ip1[N-1:0]` (SR out), `ip2[N-1:0]` (max_reg out), `sel[1:0]`
-> `op[N-1:0]`.

---

### `variable_shift_reg.sv` — Parameterized Shift Register

Delay line used for row buffering in both the convolver (depth = n-k) and
the pooler (depth = m/p).

| Parameter | Default Source | Description |
|-----------|---------------|-------------|
| `WIDTH` | `DATA_WIDTH` | Data width per stage |
| `SIZE` | 3 | Number of shift stages (set at instantiation) |

**Ports**: `clk`, `ce`, `rst`, `d[WIDTH-1:0]` -> `out[WIDTH-1:0]`.

---

## File List

```
cnn_pkg.sv              Parameter package (compile first)
qadd.sv                 Fixed-point adder
qmult.sv                Fixed-point multiplier
relu.sv                 ReLU activation
variable_shift_reg.sv   Parameterized shift register
mac_manual.sv           Multiply-accumulate unit
max_reg.sv              Max / accumulator register
input_mux.sv            Pooler input mux
comparator2.sv          Signed comparator / adder
control_logic2.sv       Pooler control FSM
tanh_lut.sv             Tanh LUT with interpolation
convolver.sv            Streaming convolution engine
pooler.sv               Streaming pooling unit
accelerator.sv          Top-level CNN accelerator
tanh_data.mem           Tanh ROM data (symlink)
Makefile                Syntax check targets
```
