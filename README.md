# Jetson Memory Bandwidth Benchmark

**STREAM-style CPU/GPU memory bandwidth benchmark for Jetson and CUDA devices.**

Measure sustained DRAM bandwidth using STREAM-like **Triad** and **Copy** kernels for both CPU (OpenMP) and GPU (CUDA).  
Reports solo vs contended performance, GiB/s vs GB/s, theoretical vs sustained, and exports CSV/JSON/Markdown results.

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/jetsonhacks/jetson-mem-bench.git
cd jetson-mem-bench
make -j
```

On Jetson before running tests:

```bash
sudo nvpmodel -m 0     # performance mode
# Turn off Dynamic Voltage and Frequency Scaling; Clocks full speed
# to restore and turn DVFS back on : sudo jetson_clocks --restore
sudo jetson_clocks     
export OMP_NUM_THREADS=$(nproc)
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```
Run tests:
```bash
# Run default Triad benchmark
python3 tools/bench.py

# Run best-case Copy benchmark
python3 tools/bench.py --op copy
```

## Platforms

- **Jetson AGX Thor** (SM=110, theoretical 273 GB/s)

- **Jetson AGX Orin** (SM=87, theoretical 204.8 GB/s)

- **Jetson Orin Nano** (SM=87, theoretical 68 GB/s)

Also runs on x86 + CUDA-capable GPUs

## More Information

See the [User Guide](docs/GUIDE.md) for details on:

- How the benchmarks work

- Units (GiB/s vs GB/s)

- Solo vs contended interpretation

- CLI parameters and tuning knobs

- Example results and common pitfalls
