# Makefile — Syntax check for parameterized CNN accelerator modules
# Supports: Questa (vlog), Vivado (xvlog), or Verilator
#
# Usage:
#   make check          — syntax check all SV files (auto-detects tool)
#   make check-questa   — force Questa (vlog)
#   make check-xvlog    — force Vivado (xvlog)
#   make check-verilator — force Verilator
#   make clean          — remove generated artifacts

# ── File lists ──
PKG    := cnn_pkg.sv
# Package must be compiled first; remaining order follows dependency chain
SRCS   := qadd.sv \
          qmult.sv \
          relu.sv \
          variable_shift_reg.sv \
          mac_manual.sv \
          max_reg.sv \
          input_mux.sv \
          comparator2.sv \
          control_logic2.sv \
          tanh_lut.sv \
          convolver.sv \
          pooler.sv \
          accelerator.sv

ALL_SV := $(PKG) $(SRCS)

# ── Tool detection ──
QUESTA    := $(shell command -v vlog 2>/dev/null)
XVLOG     := $(shell command -v xvlog 2>/dev/null)
VERILATOR := $(shell command -v verilator 2>/dev/null)

# ── Default target: auto-detect ──
.PHONY: check check-questa check-xvlog check-verilator clean

check:
ifdef QUESTA
	@echo "=== Using Questa (vlog) ==="
	@$(MAKE) --no-print-directory check-questa
else ifdef XVLOG
	@echo "=== Using Vivado (xvlog) ==="
	@$(MAKE) --no-print-directory check-xvlog
else ifdef VERILATOR
	@echo "=== Using Verilator ==="
	@$(MAKE) --no-print-directory check-verilator
else
	@echo "ERROR: No supported SV compiler found (vlog, xvlog, or verilator)"
	@echo "Install one of: Questa/ModelSim, Vivado, or Verilator"
	@exit 1
endif

# ── Questa / ModelSim ──
check-questa:
	@echo "Compiling package..."
	vlog -sv $(PKG)
	@echo "Compiling all modules..."
	vlog -sv $(SRCS)
	@echo "=== Questa syntax check PASSED ==="

# ── Vivado xvlog ──
check-xvlog:
	@echo "Compiling package..."
	xvlog --sv $(PKG)
	@echo "Compiling all modules..."
	xvlog --sv $(SRCS)
	@echo "=== xvlog syntax check PASSED ==="

# ── Verilator (lint only) ──
check-verilator:
	@echo "Linting all modules..."
	verilator --lint-only -sv --top-module accelerator $(ALL_SV)
	@echo "=== Verilator lint PASSED ==="

# ── Clean ──
clean:
	rm -rf work xsim.dir .Xil
	rm -f *.log *.jou *.pb transcript modelsim.ini vsim.wlf
	rm -f xvlog.log xvlog.pb
