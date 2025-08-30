#!/usr/bin/env python3
"""
bench.py — STREAM-like memory bandwidth orchestrator for Jetson & CUDA systems.

Features
- Runs CPU (OpenMP) and GPU (CUDA) Triad/Copy benchmarks: solo & contended.
- Reports GiB/s and GB/s, % drop under contention, % of theoretical, overlap %.
- Auto-detects Jetson model and sets theoretical GB/s (Thor/Orin/Orin Nano).
- Detects Jetson Orin Nano "SUPER" mode (102 GB/s) via nvpmodel/EMC heuristic.
- Warns when Orin Nano is not in MAXN / MAXN SUPER and suggests the right ID.
- Auto-retries the contended phase until overlap ≥ target (default 80%).
- Saves CSV and JSON (and optional Markdown) to results/ with timestamps.

Requirements
- Built binaries in build/: cpu_stream, gpu_stream (run `make -j` if missing).
"""

import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ------------------------- Constants / Defaults -------------------------

THEORETICAL_GBPS_MAP = {
    "jetson agx thor": 273.0,     # LPDDR5X 320-bit (datasheet)
    "jetson agx orin": 204.8,     # LPDDR5 256-bit (datasheet)
    "jetson orin nano": 68.0,     # LPDDR5 (standard mode)
}

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
RESULTS_DIR_DEFAULT = ROOT / "results"

# Contention overlap control
OVERLAP_TARGET = 80.0
OVERLAP_MAX_RETRIES = 2  # double iters up to this many times

# ------------------------------ Utilities ------------------------------

def run(cmd, cwd=None):
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out or "", err or ""

def try_read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return ""

def meminfo_kb(key: str) -> int:
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split()
                if parts and parts[0].rstrip(":") == key and parts[1].isdigit():
                    return int(parts[1])
    except Exception:
        pass
    return 0

def gib_to_gb(x: float) -> float:
    return x * (2**30 / 1e9) if not math.isnan(x) else float("nan")

def pct_drop(a: float, b: float) -> float:
    return 100.0*(a-b)/a if (a>0 and not math.isnan(a) and not math.isnan(b)) else float("nan")

def parse_val(s: str, key: str) -> float:
    m = re.search(rf"{re.escape(key)}\s*:\s*([0-9.]+)", s)
    return float(m.group(1)) if m else float("nan")

def detect_model():
    # Read device-tree model if available (Jetson)
    for p in ("/sys/firmware/devicetree/base/model", "/proc/device-tree/model"):
        try:
            b = Path(p).read_bytes().replace(b"\x00", b"")
            s = b.decode(errors="ignore").strip()
            if s:
                return s
        except Exception:
            pass
    return None

def normalize_model(raw: str | None):
    if not raw:
        return None
    s = raw.lower()
    if "thor" in s:
        return "jetson agx thor"
    if "agx orin" in s:
        return "jetson agx orin"
    if "orin nano" in s:
        return "jetson orin nano"
    return raw.strip()

# ------------------------ Orin Super Mode Detection ------------------------

def detect_orin_super_mode() -> tuple[bool, str]:
    """
    Best-effort detection for Jetson Orin Nano 'SUPER' mode.
    Returns (is_super, source_note).
    """
    # 1) nvpmodel -q (current mode text can include 'SUPER')
    rc, out, err = run(["nvpmodel", "-q"])
    if rc == 0:
        txt = (out + "\n" + err).lower()
        if "super" in txt:
            return True, "nvpmodel -q"
    # 2) Heuristic: EMC max_rate very high implies SUPER clocks (value in Hz)
    emc_max = try_read("/sys/kernel/debug/bpmp/debug/clk/emc/max_rate")
    try:
        if emc_max:
            hz = int(emc_max)
            if hz >= 3_000_000_000:  # >= 3.0 GHz EMC
                return True, "emc max_rate heuristic"
    except Exception:
        pass
    return False, "default"

def list_power_profiles_from_conf(conf_path="/etc/nvpmodel.conf"):
    """
    Parse POWER_MODEL lines from /etc/nvpmodel.conf if present.
    Returns list of dicts: [{"id": 2, "name": "MAXN SUPER"}, ...]
    """
    profiles = []
    text = try_read(conf_path)
    if not text:
        return profiles
    for line in text.splitlines():
        if "POWER_MODEL" in line:
            # Example: POWER_MODEL ID=2 NAME=MAXN SUPER
            m_id = re.search(r'ID\s*=\s*([0-9]+)', line)
            m_nm = re.search(r'NAME\s*=\s*(.+)$', line)
            if m_id and m_nm:
                pid = int(m_id.group(1))
                name = m_nm.group(1).strip()
                profiles.append({"id": pid, "name": name})
    return profiles

def current_power_profile_text():
    rc, out, err = run(["nvpmodel", "-q"])
    if rc == 0:
        return (out + "\n" + err).strip()
    return ""

def find_super_profile_id(profiles):
    """
    Return (id, name) for a profile whose name contains 'SUPER' (case-insensitive), else None.
    """
    for p in profiles:
        if "super" in p["name"].lower():
            return p["id"], p["name"]
    return None, None

def find_maxn_profile_id(profiles):
    """
    Return (id, name) for a profile whose name contains 'MAXN' (case-insensitive), else None.
    """
    for p in profiles:
        if "maxn" in p["name"].lower():
            return p["id"], p["name"]
    return None, None

# ------------------------------ Sizing / Output ------------------------------

def choose_sizes(total_gib: float, avail_gib: float, cpu_ws: float | None, gpu_ws: float | None):
    # Heuristic working-set sizing; clamp by available memory with a headroom.
    cpu = min(32.0, max(2.0, total_gib * 0.20))
    gpu = min(16.0, max(1.0, total_gib * 0.12))
    if cpu_ws is not None:
        cpu = max(0.25, float(cpu_ws))
    if gpu_ws is not None:
        gpu = max(0.25, float(gpu_ws))
    head = 1.5 if avail_gib >= 4 else 0.8
    cpu = max(0.5, min(cpu, (avail_gib - head) * 0.6))
    gpu = max(0.5, min(gpu, (avail_gib - head) * 0.4))
    # Convert to element counts for doubles (CPU) / floats (GPU) based on Triad WS
    N_cpu = int((cpu * (1024**3)) / (3 * 8))
    N_gpu = int((gpu * (1024**3)) / (3 * 4))
    N_cpu = max(N_cpu, 16 * 1024 * 1024)
    N_gpu = max(N_gpu, 16 * 1024 * 1024)
    return N_cpu, N_gpu, round(cpu, 3), round(gpu, 3)

def emit_csv(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Solo GiB/s", "Solo GB/s", "Contended GiB/s", "Contended GB/s", "Drop %"])
        for r in rows:
            w.writerow(r)

def make_markdown(theo_gbs, cpu_solo, cpu_cont, gpu_solo, gpu_cont):
    d_cpu = pct_drop(cpu_solo, cpu_cont)
    d_gpu = pct_drop(gpu_solo, gpu_cont)
    lines = []
    if theo_gbs:
        lines += [f"**Theoretical memory bandwidth:** {theo_gbs:.1f} GB/s", ""]
    lines += [
        "| Metric    | Solo (GiB/s / GB/s)     | Contended (GiB/s / GB/s) | Drop % |",
        "|-----------|-------------------------|--------------------------|-------:|",
    ]
    if not math.isnan(cpu_solo):
        lines.append(f"| CPU | {cpu_solo:6.2f} / {gib_to_gb(cpu_solo):6.2f} | {cpu_cont:6.2f} / {gib_to_gb(cpu_cont):6.2f} | {d_cpu:5.1f} |")
    if not math.isnan(gpu_solo):
        lines.append(f"| GPU | {gpu_solo:6.2f} / {gib_to_gb(gpu_solo):6.2f} | {gpu_cont:6.2f} / {gib_to_gb(gpu_cont):6.2f} | {d_gpu:5.1f} |")
    lines += [
        "",
        "### What these numbers mean",
        "- **Theoretical bandwidth** is a best-case ceiling from memory specs; real runs lose throughput to refresh, arbitration, and protocol overheads.",
        "- **Solo** measures each engine alone; **Contended** launches CPU and GPU simultaneously on the same LPDDR, showing real sharing behavior.",
        "- **CPU stores & write-allocate** can increase traffic (common in Triad); Copy probes best-case load/store.",
        "- Units: **GiB/s** = 2^30 bytes/s; **GB/s** = 10^9 bytes/s. 1 GiB/s ≈ 1.074 GB/s.",
    ]
    return "\n".join(lines)

# ------------------------------- Main Runner -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-ws-gib", type=float, help="CPU working set size in GiB (auto if omitted)")
    ap.add_argument("--gpu-ws-gib", type=float, help="GPU working set size in GiB (auto if omitted)")
    ap.add_argument("--iters", type=int, default=25, help="Iterations per phase (default: 25)")
    ap.add_argument("--both-only", action="store_true", help="Run contention only (skip solos)")
    ap.add_argument("--cpu-only", action="store_true", help="Run CPU phases only")
    ap.add_argument("--gpu-only", action="store_true", help="Run GPU phases only")
    ap.add_argument("--model-name", type=str, help="Override model label")
    ap.add_argument("--theoretical-gbs", type=float, help="Override theoretical GB/s")
    ap.add_argument("--orin-super", choices=["auto", "yes", "no"], default="auto",
                    help="Treat Jetson Orin Nano as running in SUPER mode (102 GB/s). Default: auto-detect.")
    ap.add_argument("--op", choices=["triad", "copy"], default="triad", help="Kernel op for both CPU and GPU")
    ap.add_argument("--markdown-out", type=str, help="Save Markdown table to results/<name>")
    ap.add_argument("--outdir", type=str, help="Directory for CSV/JSON/Markdown outputs (default: results/)")
    args = ap.parse_args()

    # Ensure binaries exist
    if not (BUILD / "cpu_stream").exists() or ((not (BUILD / "gpu_stream").exists()) and not args.cpu_only):
        print("[info] building…")
        rc, out, err = run(["make", "-j"], cwd=ROOT)
        if rc != 0:
            print(err, file=sys.stderr)
            sys.exit(1)

    # Output dir
    outdir = Path(args.outdir) if args.outdir else RESULTS_DIR_DEFAULT
    outdir.mkdir(parents=True, exist_ok=True)

    # Memory sizing
    total = meminfo_kb("MemTotal") / (1024.0 * 1024.0)
    avail = meminfo_kb("MemAvailable") / (1024.0 * 1024.0)
    N_cpu, N_gpu, cpu_ws, gpu_ws = choose_sizes(total, avail, args.cpu_ws_gib, args.gpu_ws_gib)

    print(f"[info] RAM total ≈ {total:.1f} GiB | MemAvailable ≈ {avail:.1f} GiB")
    print(f"[info] CPU WS: {cpu_ws:.2f} GiB → N_cpu={N_cpu:,} doubles   | GPU WS: {gpu_ws:.2f} GiB → N_gpu={N_gpu:,} floats")
    print(f"[info] Iterations: {args.iters} | op: {args.op}")

    # Model & theoretical bandwidth
    raw_model = args.model_name or detect_model()
    norm = normalize_model(raw_model) if raw_model else None

    # Orin SUPER handling
    theo_gbs = None
    detected_super = False
    super_source = None
    if args.theoretical_gbs is not None:
        theo_gbs = args.theoretical_gbs
    else:
        if norm == "jetson orin nano":
            if args.orin_super == "yes":
                detected_super, super_source = True, "override"
            elif args.orin_super == "no":
                detected_super, super_source = False, "override"
            else:
                detected_super, super_source = detect_orin_super_mode()
            theo_gbs = 102.0 if detected_super else 68.0
        else:
            theo_gbs = THEORETICAL_GBPS_MAP.get(norm, None) if norm else None

    if norm == "jetson orin nano":
        print(f"[info] Orin Nano SUPER mode: {'yes' if detected_super else 'no'} (source: {super_source})")

    # If Orin Nano and not SUPER, try to guide user to the correct nvpmodel ID
    profiles = []
    suggested_super = None
    suggested_maxn = None
    current_profile = current_power_profile_text()
    if norm == "jetson orin nano":
        profiles = list_power_profiles_from_conf()
        pid_super, name_super = find_super_profile_id(profiles)
        pid_maxn, name_maxn = find_maxn_profile_id(profiles)
        if pid_super is not None:
            suggested_super = {"id": pid_super, "name": name_super}
        if pid_maxn is not None:
            suggested_maxn = {"id": pid_maxn, "name": name_maxn}
        if not detected_super:
            if suggested_super:
                print(f"[hint] To enable SUPER mode: sudo nvpmodel -m {suggested_super['id']}    # {suggested_super['name']}")
            elif suggested_maxn:
                print(f"[hint] SUPER profile not found; try MAXN: sudo nvpmodel -m {suggested_maxn['id']}    # {suggested_maxn['name']}")
            else:
                print("[hint] Could not parse /etc/nvpmodel.conf. Try: grep POWER_MODEL /etc/nvpmodel.conf")

    # Binaries
    cpu_bin = str(BUILD / "cpu_stream")
    gpu_bin = str(BUILD / "gpu_stream") if (BUILD / "gpu_stream").exists() else None

    def run_cpu(N): return run([cpu_bin, "-n", str(N), "-t", str(args.iters), "-op", args.op, "-nt", "1"])
    def run_gpu(N): return run([gpu_bin, "-n", str(N), "-t", str(args.iters), "-op", args.op, "-bs", "512"])

    cpu_solo = gpu_solo = cpu_cont = gpu_cont = float("nan")
    overlap_pct = float("nan")
    contended_iters_used = args.iters
    overlap_retries = 0

    # Solo phases
    if not args.both_only:
        if not args.gpu_only:
            print("\n=== Phase 1: CPU solo ===")
            rc, out, err = run_cpu(N_cpu)
            if err.strip():
                print(err.strip())
            print(out.strip())
            key = "CPU_Copy_GiBs" if args.op == "copy" else "CPU_Triad_GiBs"
            cpu_solo = parse_val(out, key)

        if not args.cpu_only and gpu_bin:
            print("\n=== Phase 2: GPU solo ===")
            rc, out, err = run_gpu(N_gpu)
            if err.strip():
                print(err.strip())
            print(out.strip())
            key = "GPU_Copy_GiBs" if args.op == "copy" else "GPU_Triad_GiBs"
            gpu_solo = parse_val(out, key)

    # Helper: one contended attempt
    def run_contended_attempt(iters_for_contend: int):
        print(f"\n=== Phase 3: CPU + GPU together (contention) — iters={iters_for_contend} ===")
        t0 = time.monotonic()
        cpu_p = subprocess.Popen(
            [cpu_bin, "-n", str(N_cpu), "-t", str(iters_for_contend), "-op", args.op, "-nt", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        time.sleep(0.2)  # small stagger
        t1 = time.monotonic()
        gpu_p = subprocess.Popen(
            [gpu_bin, "-n", str(N_gpu), "-t", str(iters_for_contend), "-op", args.op, "-bs", "512"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        gout, gerr = gpu_p.communicate(); t2 = time.monotonic()
        cout, cerr = cpu_p.communicate(); t3 = time.monotonic()

        print("[CPU stderr]"); print(cerr.strip())
        print("[CPU stdout]"); print(cout.strip())
        print("[GPU stderr]"); print(gerr.strip())
        print("[GPU stdout]"); print(gout.strip())

        cpu_key = "CPU_Copy_GiBs" if args.op == "copy" else "CPU_Triad_GiBs"
        gpu_key = "GPU_Copy_GiBs" if args.op == "copy" else "GPU_Triad_GiBs"
        cpu_val = parse_val(cout, cpu_key)
        gpu_val = parse_val(gout, gpu_key)

        # Overlap estimation
        cpu_start, cpu_end = t0, t3
        gpu_start, gpu_end = t1, t2
        overlap = max(0.0, min(cpu_end, gpu_end) - max(cpu_start, gpu_start))
        cpu_dur = cpu_end - cpu_start
        gpu_dur = gpu_end - gpu_start
        longdur = max(cpu_dur, gpu_dur)
        ov_pct = (overlap / longdur * 100.0) if longdur > 0 else float("nan")
        if not math.isnan(ov_pct):
            print(f"[info] contention overlap: {ov_pct:.1f}% (cpu_dur={cpu_dur:.2f}s, gpu_dur={gpu_dur:.2f}s)")
        return cpu_val, gpu_val, ov_pct

    # Contended with auto-retry
    if (not args.cpu_only) and (not args.gpu_only) and gpu_bin:
        attempt_iters = args.iters
        for attempt in range(OVERLAP_MAX_RETRIES + 1):
            c_val, g_val, ov = run_contended_attempt(attempt_iters)
            cpu_cont, gpu_cont, overlap_pct = c_val, g_val, ov
            contended_iters_used = attempt_iters
            overlap_retries = attempt
            if not math.isnan(ov) and ov >= OVERLAP_TARGET:
                break
            if attempt < OVERLAP_MAX_RETRIES:
                print(f"[info] overlap {ov:.1f}% < target {OVERLAP_TARGET:.0f}% — increasing iters and retrying…")
                attempt_iters *= 2
    else:
        print("\n=== Phase 3 skipped (need both CPU and GPU) ===")
        if not math.isnan(cpu_solo):
            cpu_cont = cpu_solo
        if not math.isnan(gpu_solo):
            gpu_cont = gpu_solo

    # Output artifacts
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / f"results_{args.op}_{ts}.csv"
    json_path = outdir / f"results_{args.op}_{ts}.json"

    rows = [
        ["CPU",
         f"{cpu_solo:.3f}" if not math.isnan(cpu_solo) else "",
         f"{gib_to_gb(cpu_solo):.3f}" if not math.isnan(cpu_solo) else "",
         f"{cpu_cont:.3f}" if not math.isnan(cpu_cont) else "",
         f"{gib_to_gb(cpu_cont):.3f}" if not math.isnan(cpu_cont) else "",
         f"{pct_drop(cpu_solo, cpu_cont):.1f}" if not math.isnan(cpu_solo) and not math.isnan(cpu_cont) else ""],
        ["GPU",
         f"{gpu_solo:.3f}" if not math.isnan(gpu_solo) else "",
         f"{gib_to_gb(gpu_solo):.3f}" if not math.isnan(gpu_solo) else "",
         f"{gpu_cont:.3f}" if not math.isnan(gpu_cont) else "",
         f"{gib_to_gb(gpu_cont):.3f}" if not math.isnan(gpu_cont) else "",
         f"{pct_drop(gpu_solo, gpu_cont):.1f}" if not math.isnan(gpu_solo) and not math.isnan(gpu_cont) else ""],
    ]
    emit_csv(csv_path, rows)

    meta = {
        "timestamp": ts,
        "host": {"uname": " ".join(platform.uname()), "python": sys.version.split()[0], "cpus": os.cpu_count()},
        "memory": {"ram_total_gib": round(total, 2), "ram_available_gib": round(avail, 2)},
        "model": {
            "raw": raw_model,
            "normalized": norm,
            "theoretical_gbs": theo_gbs,
            "orin_super_mode": (detected_super if norm == "jetson orin nano" else None),
            "orin_super_source": (super_source if norm == "jetson orin nano" else None),
        },
        "power_profiles": {
            "current": current_profile,
            "from_conf": profiles,
            "suggest_super": suggested_super,
            "suggest_maxn": suggested_maxn,
        } if norm == "jetson orin nano" else None,
        "config": {
            "iters": args.iters,
            "cpu_ws_gib": None if args.cpu_ws_gib is None else float(args.cpu_ws_gib),
            "gpu_ws_gib": None if args.gpu_ws_gib is None else float(args.gpu_ws_gib),
            "N_cpu_doubles": int(N_cpu),
            "N_gpu_floats": int(N_gpu),
            "op": args.op,
            "outdir": str(outdir),
        },
        "results": {
            "cpu_solo_gib_s": None if math.isnan(cpu_solo) else round(cpu_solo, 3),
            "cpu_contended_gib_s": None if math.isnan(cpu_cont) else round(cpu_cont, 3),
            "cpu_drop_pct": None if math.isnan(pct_drop(cpu_solo, cpu_cont)) else round(pct_drop(cpu_solo, cpu_cont), 1),
            "gpu_solo_gib_s": None if math.isnan(gpu_solo) else round(gpu_solo, 3),
            "gpu_contended_gib_s": None if math.isnan(gpu_cont) else round(gpu_cont, 3),
            "gpu_drop_pct": None if math.isnan(pct_drop(gpu_solo, gpu_cont)) else round(pct_drop(gpu_solo, gpu_cont), 1),
            "cpu_solo_gb_s": None if math.isnan(cpu_solo) else round(gib_to_gb(cpu_solo), 3),
            "gpu_solo_gb_s": None if math.isnan(gpu_solo) else round(gib_to_gb(gpu_solo), 3),
            "contention_overlap_pct": None if math.isnan(overlap_pct) else round(overlap_pct, 1),
            "overlap_target_pct": OVERLAP_TARGET,
            "overlap_retries": overlap_retries,
            "contended_iters_used": contended_iters_used,
        },
        "notes": "Units: GiB/s (2^30) and GB/s (10^9). CPU=OpenMP; GPU=CUDA; Copy=best-case; Triad=AXPY-like. Overlap is fraction of longer phase.",
    }
    json_path.write_text(json.dumps(meta, indent=2))

    # Console summary
    def fmt_pair(v): return "n/a" if math.isnan(v) else f"{v:7.2f} GiB/s | {gib_to_gb(v):7.2f} GB/s"
    total_contended = (cpu_cont + gpu_cont) if (not math.isnan(cpu_cont) and not math.isnan(gpu_cont)) else float("nan")

    print("\n================ SUMMARY =================")
    if norm:
        print(f"Model: {norm}")
    if theo_gbs:
        print(f"Theoretical: {theo_gbs:.1f} GB/s")
    if norm == "jetson orin nano":
        print(f"(SUPER mode: {'yes' if detected_super else 'no'}; source: {super_source})")
    print(f"CPU solo: {fmt_pair(cpu_solo)}   | CPU contended: {fmt_pair(cpu_cont)}   | drop: {pct_drop(cpu_solo, cpu_cont):5.1f}%")
    print(f"GPU solo: {fmt_pair(gpu_solo)}   | GPU contended: {fmt_pair(gpu_cont)}   | drop: {pct_drop(gpu_solo, gpu_cont):5.1f}%")
    if not math.isnan(total_contended):
        print(f"Total contended (CPU+GPU): {total_contended:7.2f} GiB/s | {gib_to_gb(total_contended):7.2f} GB/s")
    if not math.isnan(overlap_pct):
        print(f"Contention overlap: {overlap_pct:.1f}% (target {OVERLAP_TARGET:.0f}%, retries {overlap_retries}, iters used {contended_iters_used})")
    if theo_gbs and not math.isnan(cpu_solo):
        print(f"CPU solo vs theoretical: {gib_to_gb(cpu_solo)/theo_gbs*100.0:5.1f}%")
    if theo_gbs and not math.isnan(gpu_solo):
        print(f"GPU solo vs theoretical: {gib_to_gb(gpu_solo)/theo_gbs*100.0:5.1f}%")
    print("Note: Contention runs CPU and GPU simultaneously on the same DRAM bus.")

    md = make_markdown(theo_gbs, cpu_solo, cpu_cont, gpu_solo, gpu_cont)
    print("\n---------------- Markdown ----------------")
    print(md)
    print("------------------------------------------------------------\n")

    print(f"[info] CSV:  {csv_path}")
    print(f"[info] JSON: {json_path}")
    if args.markdown_out:
        md_path = outdir / args.markdown_out
        md_path.write_text(md)
        print(f"[info] Markdown saved: {md_path}")

if __name__ == "__main__":
    sys.exit(main())
