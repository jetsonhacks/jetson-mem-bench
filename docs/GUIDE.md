# Memory Bandwidth Benchmark — User Guide

Measure **sustained main-memory bandwidth** on Jetson (and PCs) for:
- **CPU** (OpenMP) — STREAM-like **Triad** and **Copy**
- **GPU** (CUDA) — STREAM-like **Triad** and **Copy** with vectorized (16-B) loads/stores and unrolling
- **Solo vs Contended** runs — CPU alone, GPU alone, then **both together** on the **same DRAM bus**

Outputs: **GiB/s** (2^30) and **GB/s** (10^9), **% drop under contention**, **% of theoretical**, **overlap %**, CSV/JSON/Markdown artifacts.

---

## 1) Install & Build

Requirements
- GCC/Clang with OpenMP
- CUDA Toolkit (for GPU runs)
- (Jetson) `nvpmodel`, `jetson_clocks` available

Build
```bash
make -j
```

The Makefile auto-detects GPU SM with nvidia-smi and targets it (fallback SM=110 for Jetson AGX Thor). Override manually, e.g.:
```bash
make SM=87   # AGX Orin
```

## 2) Quick Start
# Triad (default op), recommended
```bash
python3 tools/bench.py
```

# Best-case Copy (pure load+store)
```bash
python3 tools/bench.py --op copy
```

# Save Markdown in results/
```bash
python3 tools/bench.py --markdown-out triad_summary.md
```

Artifacts are written to results/ (created automatically):

results/results_<op>_<timestamp>.csv

results/results_<op>_<timestamp>.json

(optional) your Markdown file

## 3) What It Measures

- **Triad:** `a[i] = b[i] + α * c[i]`
- **Copy:** `a[i] = b[i]`

On CPU we use **double**; on GPU we use **float** by default. GPU uses 16-byte vectorized access and unrolling.

**Solo** = one engine at a time  
**Contended** = CPU and GPU launched **simultaneously**


## 4) Units & “Theoretical” Bandwidth

- **GiB/s** = 2^30 bytes/s  
- **GB/s** = 10^9 bytes/s  
  1 GiB/s ≈ 1.0737 GB/s

“**Theoretical**” = DRAM data-rate × bus-width.  
Built-in lookups: Thor=273, Orin=204.8, Nano=68 GB/s.

Override anytime:

python3 tools/bench.py --theoretical-gbs 300

## 5) Key CLI Flags (runner)

- `--op {triad,copy}` – Kernel selection (default: triad)  
- `--iters N` – Repetitions per phase (default: 25)  
- `--cpu-ws-gib X` – CPU working set size in GiB (auto if omitted)  
- `--gpu-ws-gib X` – GPU working set size in GiB (auto if omitted)  
- `--cpu-only / --gpu-only` – Run a subset of phases  
- `--both-only` – Skip solos; run contention only  
- `--model-name "…"` – Override model label (detects Jetson via device tree)  
- `--theoretical-gbs X` – Override datasheet bandwidth for % comparisons  
- `--markdown-out FILE.md` – Save the table + short explainer under results/  


Artifacts

CSV: “Solo/Contended GiB/s & GB/s, Drop %”

JSON: config snapshot, results, contention_overlap_pct

Markdown: pretty table + 4-bullet explainer

## 6) Recommended Tuning Knobs

On Jetson before running

```bash
sudo nvpmodel -m 0     # performance mode
sudo jetson_clocks     # lock clocks
export OMP_NUM_THREADS=$(nproc)
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```


- **Working set sizing**

- Use big arrays (≫ last-level cache). The runner auto-sizes from MemAvailable; override with --cpu-ws-gib / --gpu-ws-gib if needed.

- **GPU occupancy**

Kernel defaults are strong (16-B vectorization, unroll=4, block=512). If you want to experiment, you can add a simple sweep in the script; 256/512/1024 are good tries.

- **CPU non-temporal stores**

Portable fallback is enabled by default. On some aarch64 toolchains we compile with -DNT_STORE=0 to avoid non-portable builtins. This is fine for STREAM-like loads/stores.


## 7) Interpreting Results

- Solo vs Theoretical: expect 60–80%.  
- Copy vs Triad: CPU Copy > Triad; GPU Triad ≈ Copy or higher.  
- Contended: total ~70–80% of theoretical.  
- Overlap %: if <80, increase `--iters` or WS. ; otherwise contention might be under-represented.

## 8) Reproducibility

- GPU vector-tail fixed: we snap N_eff.  
- Bytes moved:
  - Copy: `N*2*size*iters`
  - Triad: `N*3*size*iters`
- Results in `results/` with timestamp.

CSV/JSON include: Solo/Contended (GiB/s & GB/s), drop %, theoretical %, and contention_overlap_pct.

Results live in results/ with timestamps.

## 9) Common Pitfalls

- Unpinned clocks → lower numbers  
- Small arrays → cache artifacts  
- Low overlap % → underestimates contention  
- Confuse GB vs GiB → always convert (1 GiB/s ≈ 1.0737 GB/s)

10) Examples

AGX Thor (theoretical 273 GB/s)
```bash
make SM=110 -j
python3 tools/bench.py --op triad
python3 tools/bench.py --op copy
```


AGX Orin (theoretical 204.8 GB/s)
```bash
make SM=87 -j
python3 tools/bench.py --op triad --model-name "Jetson AGX Orin"
```


Orin Nano 8GB (theoretical 68 GB/s - Super 120 GB/s)
```bash
make SM=87 -j
python3 tools/bench.py --op copy --model-name "Jetson Orin Nano"
```

11) FAQ (short)

**Why don’t I hit theoretical?**
DRAM refresh, command/address overheads, controller arbitration, ECC/protocol, and kernel/runtime overheads keep sustained bandwidth below the ideal line.

**Why are contended sums < theoretical?**
Same overheads apply, plus arbitration inefficiency. Hitting ~70–80% of theoretical as a sum is a healthy system.

**CPU vs GPU types?**
CPU uses double; GPU uses float by default for bandwidth. You can rebuild the GPU for doubles with -DUSE_DOUBLE=1 in src/gpu_stream.cu (expect a small drop).

Happy measuring! If you extend the suite (e.g., add STREAM Add/Scale, or a GPU block-size sweep), keep the reporting format stable so CSV/JSON remain machine-friendly.
