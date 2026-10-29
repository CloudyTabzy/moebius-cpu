"""A/B harness: allocator, rusty_alloc options and rayon pool size.

Runs `examples/profile.rs` (one Exact inpaint, twice per process) under each
configuration, interleaved round-robin so every configuration sees the same
power states. Records, per run: total time, encode / decode / median
UNet step, and the process's peak working set (Windows `GetProcessMemoryInfo`).

    python tools/bench_alloc.py [--rounds 3] [--only sys,ra,...] [--dry-run]

Needs two builds of the example (the script makes both):
    target/release/examples/profile.exe       default features (rusty_alloc)
    target-sys/release/examples/profile.exe   --no-default-features --features cli
rusty_alloc options are passed as RUSTY_ALLOC_<NAME> environment variables,
so no rebuild is needed per option. Results: weights/bench_alloc.json + a table.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RA_EXE = os.path.join(ROOT, "target", "release", "examples", "profile.exe")
SYS_EXE = os.path.join(ROOT, "target-sys", "release", "examples", "profile.exe")
WEIGHTS = "weights/ft_places2"

# name -> (exe, env overrides, rayon threads; 0 = physical cores)
CONFIGS = {
    "sys":         (SYS_EXE, {}, 0),
    "ra":          (RA_EXE, {}, 0),
    "ra_purge0":   (RA_EXE, {"RUSTY_ALLOC_PURGE_DELAY": "0"}, 0),
    "ra_purge100": (RA_EXE, {"RUSTY_ALLOC_PURGE_DELAY": "100"}, 0),
    "ra_purge1s":  (RA_EXE, {"RUSTY_ALLOC_PURGE_DELAY": "1000"}, 0),
    "ra_reserve":  (RA_EXE, {"RUSTY_ALLOC_RESERVE_OS_MEMORY": str(2 * 1024 * 1024)}, 0),  # KiB
    "ra_t6":       (RA_EXE, {}, 6),
    "ra_t16":      (RA_EXE, {}, 16),
}


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_ws(proc: subprocess.Popen) -> tuple[int, int]:
    """Peak working set and page faults of a finished child (its handle stays open)."""
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(wt.HANDLE(int(proc._handle)), ctypes.byref(c), c.cb)
    return (c.PeakWorkingSetSize, c.PageFaultCount) if ok else (0, 0)


RUN = re.compile(r"run (\d+): total ([\d.]+) s \| denoise ([\d.]+) s over (\d+) steps \(first ([\d.]+), median ([\d.]+)")
STAGE = re.compile(r"^\s+(encode|decode)\s+([\d.]+) s")


def run_once(name: str) -> dict:
    exe, env, threads = CONFIGS[name]
    proc = subprocess.Popen([exe, WEIGHTS, str(threads)], cwd=ROOT, env={**os.environ, **env},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = proc.communicate()
    peak, faults = peak_ws(proc)
    runs, stages = [], []
    for line in out.splitlines():
        if m := STAGE.match(line):
            stages.append((m[1], float(m[2])))
        if m := RUN.search(line):
            runs.append({"total": float(m[2]), "step_median": float(m[6])})
    enc = [v for k, v in stages if k == "encode"]
    dec = [v for k, v in stages if k == "decode"]
    for i, r in enumerate(runs):
        r["encode"], r["decode"] = enc[i], dec[i]
    if proc.returncode != 0 or len(runs) != 2:
        print(out, file=sys.stderr)
        raise SystemExit(f"{name}: profile failed (rc {proc.returncode})")
    return {"runs": runs, "peak_ws_mib": peak / 2**20, "page_faults": faults}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--only", default=",".join(CONFIGS))
    ap.add_argument("--idle", type=float, default=20.0, help="seconds of idle before each process")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    names = args.only.split(",")
    if args.dry_run:
        for n in names:
            exe, env, t = CONFIGS[n]
            print(n, exe, env, "threads", t or "physical")
        return
    for cmd in (["cargo", "build", "--release", "--features", "cli", "--example", "profile"],
                ["cargo", "build", "--release", "--no-default-features", "--features", "cli",
                 "--example", "profile", "--target-dir", "target-sys"]):
        subprocess.run(cmd, cwd=ROOT, check=True)
    results = {n: [] for n in names}
    for r in range(args.rounds):
        for n in names:
            time.sleep(args.idle)
            res = run_once(n)
            results[n].append(res)
            print(f"round {r} {n:12s} totals {[x['total'] for x in res['runs']]} peak {res['peak_ws_mib']:.0f} MiB", flush=True)
    json.dump(results, open(os.path.join(ROOT, "weights", "bench_alloc.json"), "w"), indent=1)
    print(f"\n{'config':12s} {'run0 total':>10s} {'run1 total':>10s} {'step':>6s} {'encode':>7s} {'decode':>7s} {'peak MiB':>9s} {'faults':>9s}")
    med = lambda xs: statistics.median(xs)
    for n in names:
        rs = results[n]
        print(f"{n:12s} {med([x['runs'][0]['total'] for x in rs]):10.2f} {med([x['runs'][1]['total'] for x in rs]):10.2f} "
              f"{med([x['runs'][1]['step_median'] for x in rs]):6.2f} {med([x['runs'][1]['encode'] for x in rs]):7.2f} "
              f"{med([x['runs'][1]['decode'] for x in rs]):7.2f} {med([x['peak_ws_mib'] for x in rs]):9.0f} {med([x['page_faults'] for x in rs]):9.0f}")


if __name__ == "__main__":
    main()
