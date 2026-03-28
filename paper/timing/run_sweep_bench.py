import csv
import os
import subprocess
import sys

PY = sys.executable
BENCH = "bench_one_tmat.py"
#BENCH = "bench_one_fob.py"
#BENCH = "bench_one_rcd.py"
def run_case(num, lmax, lmax_glob, runs=10, warmup=2, jit_forward=0, jit_vg=0):
    cmd = [
        PY,
        BENCH,
        "--num",
        str(num),
        "--lmax",
        str(lmax),
        "--lmax_glob",
        str(lmax_glob),
        "--runs",
        str(runs),
        "--warmup",
        str(warmup),
        "--jit_forward",
        str(jit_forward),
        "--jit_vg",
        str(jit_vg),
    ]
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    last_csv = None
    for line in p.stdout:
        sys.stdout.write(line) 
        if line.count(",") == 5:
            last_csv = line.strip()

    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"child failed with code {rc}")
    if last_csv is None:
        raise RuntimeError("no CSV line produced")
    return last_csv.split(",")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num", "lmax", "num_params", "fwd", "fb", "peak_exec_rss_gb"])
        w.writerows(rows)


def main():
    os.makedirs("benchmarks", exist_ok=True)
    rows_N = []
    nums = [2, 5, 10, 20, 30, 40, 50]
    lmaxs = [1, 2, 3, 4, 5, 6, 7, 8, 10]
    title = f"last_jit_num_{nums[0]}_{nums[-1]}_lmax_{lmaxs[0]}_{lmaxs[-1]}_fob"
    for num in nums:
        rows_N.append(run_case(num=num, lmax=3, lmax_glob=6, jit_forward=1, jit_vg=1))
    write_csv(f"benchmarks/scaling_scatterers_{title}.csv", rows_N)

    rows_L = []
    for lmax in lmaxs:
        rows_L.append(run_case(num=5, lmax=lmax, lmax_glob=lmax, jit_forward=1, jit_vg=1))
    write_csv(f"benchmarks/scaling_lmax_{title}.csv", rows_L)
    
if __name__ == "__main__":
    main()