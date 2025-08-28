#!/usr/bin/env python3
"""
Professional runner: builds once (via Makefile), runs CPU/GPU solo and contended,
prints GiB/s & GB/s, theoretical %, and writes CSV/JSON/Markdown.

Usage:
  make -j && python3 tools/bench.py
  python3 tools/bench.py --cpu-ws-gib 24 --gpu-ws-gib 12 --iters 50
  python3 tools/bench.py --model-name "Jetson AGX Thor" --theoretical-gbs 273
  python3 tools/bench.py --op copy    # or triad (default)
  python3 tools/bench.py --markdown-out results.md
"""

import argparse, csv, json, math, os, platform, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

THEORETICAL_GBPS_MAP = {
    "jetson agx thor": 273.0,
    "jetson agx orin": 204.8,
    "jetson orin nano": 68.0,
}

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"

def run(cmd, cwd=None):
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out or "", err or ""

def parse_val(s: str, key: str) -> float:
    import re
    m = re.search(rf"{key}\s*:\s*([0-9.]+)", s)
    return float(m.group(1)) if m else float("nan")

def meminfo_kb(k: str) -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0].rstrip(':') == k and parts[1].isdigit():
                    return int(parts[1])
    except Exception:
        pass
    return 0

def gib_to_gb(x: float) -> float:
    return x * (2**30 / 1e9) if not math.isnan(x) else float("nan")

def pct_drop(a: float, b: float) -> float:
    return 100.0*(a-b)/a if (a>0 and not math.isnan(a) and not math.isnan(b)) else float("nan")

def normalize_model(raw: str|None):
    if not raw: return None
    s = raw.lower()
    if "thor" in s: return "jetson agx thor"
    if "agx orin" in s: return "jetson agx orin"
    if "orin nano" in s: return "jetson orin nano"
    return raw.strip()

def detect_model():
    for p in ("/sys/firmware/devicetree/base/model","/proc/device-tree/model"):
        try:
            b = Path(p).read_bytes().replace(b"\x00", b"")
            s = b.decode(errors="ignore").strip()
            if s: return s
        except Exception: pass
    return None

def choose_sizes(total_gib: float, avail_gib: float, cpu_ws: float|None, gpu_ws: float|None):
    cpu = min(32.0, max(2.0, total_gib * 0.20))
    gpu = min(16.0, max(1.0, total_gib * 0.12))
    if cpu_ws is not None: cpu = max(0.25, float(cpu_ws))
    if gpu_ws is not None: gpu = max(0.25, float(gpu_ws))
    head = 1.5 if avail_gib >= 4 else 0.8
    cpu = max(0.5, min(cpu, (avail_gib - head) * 0.6))
    gpu = max(0.5, min(gpu, (avail_gib - head) * 0.4))
    # Convert to element counts for doubles/floats handled in C/CUDA
    N_cpu = int((cpu * (1024**3)) / (3*8))    # Triad WS formula; Copy uses 2 streams internally
    N_gpu = int((gpu * (1024**3)) / (3*4))
    N_cpu = max(N_cpu, 16*1024*1024)
    N_gpu = max(N_gpu, 16*1024*1024)
    return N_cpu, N_gpu, round(cpu,3), round(gpu,3)

def emit_csv(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric","Solo GiB/s","Solo GB/s","Contended GiB/s","Contended GB/s","Drop %"])
        for r in rows: w.writerow(r)

def make_markdown(theo_gbs, cpu_solo, cpu_cont, gpu_solo, gpu_cont):
    d_cpu = pct_drop(cpu_solo, cpu_cont); d_gpu = pct_drop(gpu_solo, gpu_cont)
    lines = []
    if theo_gbs: lines += [f"**Theoretical memory bandwidth:** {theo_gbs:.1f} GB/s", ""]
    lines += [
        "| Metric    | Solo (GiB/s / GB/s)     | Contended (GiB/s / GB/s) | Drop % |",
        "|-----------|-------------------------|--------------------------|-------:|",
    ]
    if not math.isnan(cpu_solo): lines.append(f"| CPU | {cpu_solo:6.2f} / {gib_to_gb(cpu_solo):6.2f} | {cpu_cont:6.2f} / {gib_to_gb(cpu_cont):6.2f} | {d_cpu:5.1f} |")
    if not math.isnan(gpu_solo): lines.append(f"| GPU | {gpu_solo:6.2f} / {gib_to_gb(gpu_solo):6.2f} | {gpu_cont:6.2f} / {gib_to_gb(gpu_cont):6.2f} | {d_gpu:5.1f} |")
    lines += [
        "",
        "### What these numbers mean",
        "- **Theoretical bandwidth** is a best-case ceiling from memory specs; real runs lose throughput to refresh, arbitration, and protocol overheads.",
        "- **Solo** measures each engine alone; **Contended** launches CPU and GPU simultaneously on the same LPDDR, showing real sharing behavior.",
        "- **CPU stores & write-allocate** can increase traffic (common in Triad); we use Copy to probe best-case sustained load/store.",
        "- Units: **GiB/s** = 2^30 bytes/s; **GB/s** = 10^9 bytes/s. 1 GiB/s ≈ 1.074 GB/s.",
    ]
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-ws-gib", type=float)
    ap.add_argument("--gpu-ws-gib", type=float)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--both-only", action="store_true")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--gpu-only", action="store_true")
    ap.add_argument("--model-name", type=str)
    ap.add_argument("--theoretical-gbs", type=float)
    ap.add_argument("--op", choices=["triad","copy"], default="triad", help="Kernel op for both CPU and GPU")
    ap.add_argument("--markdown-out", type=str)
    args = ap.parse_args()

    # Ensure binaries exist
    if not (BUILD/"cpu_stream").exists() or (not (BUILD/"gpu_stream").exists() and not args.cpu_only):
        print("[info] building…")
        rc, out, err = run(["make", "-j"], cwd=ROOT)
        if rc != 0:
            print(err, file=sys.stderr); sys.exit(1)

    # Memory sizes
    total = meminfo_kb("MemTotal")/(1024.0*1024.0)
    avail = meminfo_kb("MemAvailable")/(1024.0*1024.0)
    N_cpu, N_gpu, cpu_ws, gpu_ws = choose_sizes(total, avail, args.cpu_ws_gib, args.gpu_ws_gib)

    print(f"[info] RAM total ≈ {total:.1f} GiB | MemAvailable ≈ {avail:.1f} GiB")
    print(f"[info] CPU WS: {cpu_ws:.2f} GiB → N_cpu={N_cpu:,} doubles   | GPU WS: {gpu_ws:.2f} GiB → N_gpu={N_gpu:,} floats")
    print(f"[info] Iterations: {args.iters} | op: {args.op}")

    raw_model = args.model_name or detect_model()
    norm = normalize_model(raw_model) if raw_model else None
    theo_gbs = args.theoretical_gbs if args.theoretical_gbs is not None else (THEORETICAL_GBPS_MAP.get(norm, None) if norm else None)

    cpu_bin = str(BUILD/"cpu_stream")
    gpu_bin = str(BUILD/"gpu_stream") if (BUILD/"gpu_stream").exists() else None

    def run_cpu(N): return run([cpu_bin, "-n", str(N), "-t", str(args.iters), "-op", args.op, "-nt", "1"])
    def run_gpu(N): return run([gpu_bin, "-n", str(N), "-t", str(args.iters), "-op", args.op, "-bs", "512"])

    cpu_solo = gpu_solo = cpu_cont = gpu_cont = float("nan")

    if not args.both_only:
        if not args.gpu_only:
            print("\n=== Phase 1: CPU solo ===")
            rc, out, err = run_cpu(N_cpu); print(err.strip()); print(out.strip())
            key = "CPU_Copy_GiBs" if args.op=="copy" else "CPU_Triad_GiBs"
            cpu_solo = parse_val(out, key)
        if not args.cpu_only and gpu_bin:
            print("\n=== Phase 2: GPU solo ===")
            rc, out, err = run_gpu(N_gpu); print(err.strip()); print(out.strip())
            key = "GPU_Copy_GiBs" if args.op=="copy" else "GPU_Triad_GiBs"
            gpu_solo = parse_val(out, key)

    # Contended (both together)
    if (not args.cpu_only) and (not args.gpu_only) and gpu_bin:
        print("\n=== Phase 3: CPU + GPU together (contention) ===")
        t0 = time.monotonic()
        cpu_p = subprocess.Popen([cpu_bin,"-n",str(N_cpu),"-t",str(args.iters),"-op",args.op,"-nt","1"],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(0.2)
        t1 = time.monotonic()
        gpu_p = subprocess.Popen([gpu_bin,"-n",str(N_gpu),"-t",str(args.iters),"-op",args.op,"-bs","512"],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        gout, gerr = gpu_p.communicate(); t2 = time.monotonic()
        cout, cerr = cpu_p.communicate(); t3 = time.monotonic()

        print("[CPU stderr]"); print(cerr.strip())
        print("[CPU stdout]"); print(cout.strip())
        print("[GPU stderr]"); print(gerr.strip())
        print("[GPU stdout]"); print(gout.strip())

        cpu_key = "CPU_Copy_GiBs" if args.op=="copy" else "CPU_Triad_GiBs"
        gpu_key = "GPU_Copy_GiBs" if args.op=="copy" else "GPU_Triad_GiBs"
        cpu_cont = parse_val(cout, cpu_key)
        gpu_cont = parse_val(gout, gpu_key)

        # Overlap warning
        cpu_start, cpu_end = t0, t3
        gpu_start, gpu_end = t1, t2
        ov = max(0.0, min(cpu_end, gpu_end) - max(cpu_start, gpu_start))
        longdur = max(cpu_end-cpu_start, gpu_end-gpu_start)
        if longdur > 0 and ov/longdur < 0.80:
            print(f"[warn] contention overlap only ~{(ov/longdur)*100:.1f}% — increase --iters or WS for truer contention.",
                  file=sys.stderr)
    else:
        print("\n=== Phase 3 skipped (need both CPU and GPU) ===")
        if not math.isnan(cpu_solo): cpu_cont = cpu_solo
        if not math.isnan(gpu_solo): gpu_cont = gpu_solo

    RESULTS_DIR = ROOT / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"results_{args.op}_{ts}.csv"
    json_path = RESULTS_DIR / f"results_{args.op}_{ts}.json"

    # CSV rows
    rows = [
        ["CPU", f"{cpu_solo:.3f}" if not math.isnan(cpu_solo) else "", f"{gib_to_gb(cpu_solo):.3f}" if not math.isnan(cpu_solo) else "",
                f"{cpu_cont:.3f}" if not math.isnan(cpu_cont) else "", f"{gib_to_gb(cpu_cont):.3f}" if not math.isnan(cpu_cont) else "",
                f"{pct_drop(cpu_solo,cpu_cont):.1f}" if not math.isnan(cpu_solo) and not math.isnan(cpu_cont) else ""],
        ["GPU", f"{gpu_solo:.3f}" if not math.isnan(gpu_solo) else "", f"{gib_to_gb(gpu_solo):.3f}" if not math.isnan(gpu_solo) else "",
                f"{gpu_cont:.3f}" if not math.isnan(gpu_cont) else "", f"{gib_to_gb(gpu_cont):.3f}" if not math.isnan(gpu_cont) else "",
                f"{pct_drop(gpu_solo,gpu_cont):.1f}" if not math.isnan(gpu_solo) and not math.isnan(gpu_cont) else ""],
    ]
    emit_csv(csv_path, rows)

    meta = {
        "timestamp": ts,
        "host": {"uname": " ".join(platform.uname()), "python": sys.version.split()[0], "cpus": os.cpu_count()},
        "memory": {"ram_total_gib": round(total,2), "ram_available_gib": round(avail,2)},
        "model": {"raw": raw_model, "normalized": norm, "theoretical_gbs": theo_gbs},
        "config": {"iters": args.iters, "cpu_ws_gib": None if args.cpu_ws_gib is None else float(args.cpu_ws_gib),
                   "gpu_ws_gib": None if args.gpu_ws_gib is None else float(args.gpu_ws_gib),
                   "N_cpu_doubles": int(N_cpu), "N_gpu_floats": int(N_gpu), "op": args.op},
        "results": {
            "cpu_solo_gib_s": None if math.isnan(cpu_solo) else round(cpu_solo,3),
            "cpu_contended_gib_s": None if math.isnan(cpu_cont) else round(cpu_cont,3),
            "cpu_drop_pct": None if math.isnan(pct_drop(cpu_solo,cpu_cont)) else round(pct_drop(cpu_solo,cpu_cont),1),
            "gpu_solo_gib_s": None if math.isnan(gpu_solo) else round(gpu_solo,3),
            "gpu_contended_gib_s": None if math.isnan(gpu_cont) else round(gpu_cont,3),
            "gpu_drop_pct": None if math.isnan(pct_drop(gpu_solo,gpu_cont)) else round(pct_drop(gpu_solo,gpu_cont),1),
            "cpu_solo_gb_s": None if math.isnan(cpu_solo) else round(gib_to_gb(cpu_solo),3),
            "gpu_solo_gb_s": None if math.isnan(gpu_solo) else round(gib_to_gb(gpu_solo),3),
        },
        "notes": "Units: GiB/s (2^30) and GB/s (10^9). CPU=OpenMP; GPU=CUDA; Copy=best-case; Triad=AXPY-like."
    }
    Path(json_path).write_text(json.dumps(meta, indent=2))

    # Console summary
    def fmt_pair(v): return "n/a" if math.isnan(v) else f"{v:7.2f} GiB/s | {gib_to_gb(v):7.2f} GB/s"
    print("\n================ SUMMARY =================")
    if norm: print(f"Model: {norm}")
    if theo_gbs: print(f"Theoretical: {theo_gbs:.1f} GB/s")
    print(f"CPU solo: {fmt_pair(cpu_solo)}   | CPU contended: {fmt_pair(cpu_cont)}   | drop: {pct_drop(cpu_solo,cpu_cont):5.1f}%")
    print(f"GPU solo: {fmt_pair(gpu_solo)}   | GPU contended: {fmt_pair(gpu_cont)}   | drop: {pct_drop(gpu_solo,gpu_cont):5.1f}%")
    if theo_gbs and not math.isnan(cpu_solo):
        print(f"CPU solo vs theoretical: {gib_to_gb(cpu_solo)/theo_gbs*100.0:5.1f}%")
    if theo_gbs and not math.isnan(gpu_solo):
        print(f"GPU solo vs theoretical: {gib_to_gb(gpu_solo)/theo_gbs*100.0:5.1f}%")
    print("Note: Contention runs CPU and GPU simultaneously on the same DRAM bus.")

    md = make_markdown(theo_gbs, cpu_solo, cpu_cont, gpu_solo, gpu_cont)
    print("\n---------------- Markdown ----------------")
    print(md)
    print("------------------------------------------------------------\n")
    if args.markdown_out:
        Path(args.markdown_out).write_text(md)
        print(f"[info] Markdown saved: {args.markdown_out}")
        print(f"[info] CSV: {csv_path}")
        print(f"[info] JSON: {json_path}")

if __name__ == "__main__":
    sys.exit(main())
