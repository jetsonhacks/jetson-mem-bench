# Makefile — CPU/GPU STREAM-like bandwidth microbench
# - Auto-detects CUDA SM via nvidia-smi; fallback SM=110 (Jetson Thor / Blackwell)
# - Override manually:  make SM=87   (AGX Orin / Ampere)
# - Builds: build/cpu_stream and build/gpu_stream

# -------------------- Compilers & Flags --------------------
CC      ?= gcc
CFLAGS  ?= -std=c11 -Ofast -funroll-loops -ffast-math -fopenmp -march=native -Wall -Wextra -Wshadow -Wconversion
LDFLAGS ?= -fopenmp

NVCC    ?= nvcc
NVFLAGS ?= -O3 -Xcompiler "-Ofast,-funroll-loops,-ffast-math,-march=native" -Xptxas -v

# On aarch64 toolchains, disable NT store builtin by default (portable)
ARCH := $(shell uname -m)
ifeq ($(ARCH),aarch64)
  CFLAGS += -DNT_STORE=0
endif

# -------------------- SM Auto-Detection --------------------
# Try to read compute capability like "8.7" or "11.0" from the first GPU
SM_DEFAULT := 110
SM_RAW := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1)
# Strip non-digits: "8.7" -> "87", "11.0" -> "110"
SM_FROM_NVSMI := $(shell echo "$(SM_RAW)" | sed 's/[^0-9]//g')

# Allow user override: e.g., `make SM=87`
SM ?= $(if $(SM_FROM_NVSMI),$(SM_FROM_NVSMI),$(SM_DEFAULT))

# Generate both native SASS and PTX fallback for forward-compat
GENCODE := -gencode arch=compute_$(SM),code=sm_$(SM) \
           -gencode arch=compute_$(SM),code=compute_$(SM)

NVFLAGS += $(GENCODE)

# -------------------- Layout --------------------
BUILD_DIR := build
SRC_DIR   := src

CPU_BIN   := $(BUILD_DIR)/cpu_stream
GPU_BIN   := $(BUILD_DIR)/gpu_stream

.PHONY: all cpu gpu clean info gpu-ptx

all: info cpu gpu

info:
	@echo "==> CUDA SM target: $(SM)  (raw='$(SM_RAW)')"
	@echo "==> CC:  $(CC)"
	@echo "==> NVCC: $(NVCC)"

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

cpu: $(CPU_BIN)
gpu: $(GPU_BIN)

$(CPU_BIN): $(SRC_DIR)/cpu_stream.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(GPU_BIN): $(SRC_DIR)/gpu_stream.cu | $(BUILD_DIR)
	$(NVCC) $(NVFLAGS) $< -o $@

# Optional: build PTX-only binary (lets driver JIT on very new GPUs if toolchain lacks SM)
gpu-ptx: | $(BUILD_DIR)
	$(NVCC) -O3 -Xptxas -v -gencode arch=compute_$(SM),code=compute_$(SM) $(SRC_DIR)/gpu_stream.cu -o $(GPU_BIN)

clean:
	rm -rf $(BUILD_DIR)
