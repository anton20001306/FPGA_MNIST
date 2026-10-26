# Makefile — Syntax check and simulation for CNN v1
# Usage:
#   make check          — auto-detect tool and compile RTL
#   make check-questa   — use Questa (vlog)
#   make check-xvlog    — use Vivado (xvlog)
#   make hex            — run export_hex.py to generate .hex files
#   make sim            — compile + simulate with Questa (vsim)
#   make sim-xsim       — compile + simulate with Vivado (xsim)
#   make clean

RTL_DIR := rtl
TB_DIR  := tb
HEX_DIR := tb/hex
SCR_DIR := tb/scripts

PKG := $(RTL_DIR)/cnn_pkg.sv

SRCS := $(RTL_DIR)/bram_sp.sv \
        $(RTL_DIR)/rom_sp.sv \
        $(RTL_DIR)/mac_int8.sv \
        $(RTL_DIR)/requantize.sv \
        $(RTL_DIR)/conv_engine.sv \
        $(RTL_DIR)/pool_engine.sv \
        $(RTL_DIR)/fc_engine.sv \
        $(RTL_DIR)/argmax.sv \
        $(RTL_DIR)/cnn_top.sv

TB_SRCS := $(TB_DIR)/cnn_top_tb.sv

ALL_SV := $(PKG) $(SRCS)

QUESTA    := $(shell command -v vlog 2>/dev/null)
XVLOG     := $(shell command -v xvlog 2>/dev/null)
VERILATOR := $(shell command -v verilator 2>/dev/null)

NUM_TEST ?= 20

.PHONY: check check-questa check-xvlog check-verilator hex sim sim-xsim clean

# ─── Syntax check ───
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
	@echo "ERROR: No supported SV compiler found"
	@exit 1
endif

check-questa:
	@echo "Compiling package..."
	vlog -sv $(PKG)
	@echo "Compiling all modules..."
	vlog -sv $(SRCS)
	@echo "=== Questa syntax check PASSED ==="

check-xvlog:
	xvlog --sv $(PKG)
	xvlog --sv $(SRCS)
	@echo "=== xvlog syntax check PASSED ==="

check-verilator:
	verilator --lint-only -sv --top-module cnn_top $(ALL_SV)
	@echo "=== Verilator lint PASSED ==="

# ─── Generate hex files from trained weights + MNIST data ───
hex:
	@echo "=== Generating hex files ==="
	cd $(SCR_DIR) && python3 export_hex.py --num_test $(NUM_TEST)
	@echo "=== Hex generation complete ==="

# ─── Simulation (Questa/ModelSim) ───
# Run from HEX_DIR so $readmemh finds weight and test hex files
sim: hex
	@echo "=== Compiling for simulation (Questa) ==="
	cd $(HEX_DIR) && vlog -sv ../../$(PKG)
	cd $(HEX_DIR) && vlog -sv $(addprefix ../../, $(SRCS))
	cd $(HEX_DIR) && vlog -sv +incdir+. ../../$(TB_SRCS)
	@echo "=== Running simulation ==="
	cd $(HEX_DIR) && vsim -batch -do "run -all; quit" cnn_top_tb
	@echo "=== Simulation complete ==="

# ─── Simulation (Vivado xsim) ───
sim-xsim: hex
	@echo "=== Compiling for simulation (xsim) ==="
	cd $(HEX_DIR) && xvlog --sv ../../$(PKG)
	cd $(HEX_DIR) && xvlog --sv $(addprefix ../../, $(SRCS))
	cd $(HEX_DIR) && xvlog --sv --include . ../../$(TB_SRCS)
	cd $(HEX_DIR) && xelab cnn_top_tb -s sim_snap
	@echo "=== Running simulation ==="
	cd $(HEX_DIR) && xsim sim_snap -runall
	@echo "=== Simulation complete ==="

# ─── Clean ───
clean:
	rm -rf work xsim.dir .Xil
	rm -f *.log *.jou *.pb transcript modelsim.ini vsim.wlf
	rm -rf $(HEX_DIR)/work $(HEX_DIR)/xsim.dir $(HEX_DIR)/.Xil
	rm -f $(HEX_DIR)/*.log $(HEX_DIR)/*.jou $(HEX_DIR)/*.pb
	rm -f $(HEX_DIR)/transcript $(HEX_DIR)/modelsim.ini $(HEX_DIR)/vsim.wlf
	rm -f $(HEX_DIR)/*.vcd $(HEX_DIR)/*.wdb
