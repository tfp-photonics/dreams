import argparse
import os
import threading
import time

import jax
import jax.numpy as jnp
import numpy as onp
import psutil
from func_helper import dreams_rcd 
from jax import jit, value_and_grad
from jax.tree_util import tree_map

jax.config.update("jax_enable_x64", True)

_proc = psutil.Process(os.getpid())


def current_rss_gb():
    return _proc.memory_info().rss / 1e9


def monitor_peak(stop_event, state, interval=0.002):
    while not stop_event.is_set():
        rss = current_rss_gb()
        if rss > state["peak"]:
            state["peak"] = rss
        time.sleep(interval)


def block(x):
    return tree_map(
        lambda y: y.block_until_ready() if hasattr(y, "block_until_ready") else y,
        x,
    )


def build_problem(num, lmax, lmax_glob):

    eps_medium = 1.5**2
    eps_object = (3.5 + 0.02j) ** 2  ##

    pitch = 600.0
    R = 170.0
    zmax = 0.0
    r_init = 10.0

    ni = onp.linspace(0.0, 360.0, num + 1)[:-1]
    x = R * onp.cos(ni * onp.pi / 180.0)
    y = R * onp.sin(ni * onp.pi / 180.0)
    z = onp.linspace(-zmax, zmax, x.shape[0])

    positions = onp.array([x, y, z]).T  # (num,3)
    radii = onp.ones((num,), dtype=onp.float64) * r_init

    wl = 1050.0
    k0 = 2.0 * onp.pi / wl

    cfg = {
        "lmax": int(lmax),
        "lmax_glob": int(lmax_glob),
        "pitch": float(pitch),
        "eps_medium": eps_medium,
        "eps_object": onp.asarray(eps_object),
        "rmax_coef": 1,
        "helicity": False,
        "kx": 0.0,
        "ky": 0.0,
        "k0": float(k0),
    }

    param = onp.concatenate([positions.reshape(-1), radii], axis=0)  # (4*num,)
    return cfg, eps_object, k0, jnp.array(param)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, required=True)
    ap.add_argument("--lmax", type=int, required=True)
    ap.add_argument("--lmax_glob", type=int, required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--jit_forward", type=int, default=0)
    ap.add_argument("--jit_vg", type=int, default=0)
    args = ap.parse_args()
    print("start main")
    cfg, eps_object, k0, param = build_problem(args.num, args.lmax, args.lmax_glob)

    # forward
    f = lambda p: dreams_rcd(p, eps_object=eps_object, k0=k0, cfg=cfg)

    # forward+backward (reverse-mode)
    vg = lambda p: value_and_grad(
        lambda pp: dreams_rcd(pp, eps_object=eps_object, k0=k0, cfg=cfg)
    )(p)

    if args.jit_forward:
        f = jit(f)
    if args.jit_vg:
        vg = jit(vg)

    block(f(param))
    block(vg(param))
    for _ in range(args.warmup):
        block(f(param))
        block(vg(param))

    baseline_rss = current_rss_gb()
    state = {"peak": baseline_rss}
    stop_event = threading.Event()
    mon = threading.Thread(target=monitor_peak, args=(stop_event, state), daemon=True)
    mon.start()

    # Forward timing
    t0 = time.perf_counter()
    for _ in range(args.runs):
        block(f(param))
    t1 = time.perf_counter()
    forward_time = (t1 - t0) / args.runs

    # Forward+backward timing
    t0 = time.perf_counter()
    for _ in range(args.runs):
        block(vg(param))
    t1 = time.perf_counter()
    forward_backward_time = (t1 - t0) / args.runs

    stop_event.set()
    mon.join()

    peak_exec_rss_gb = state["peak"] - baseline_rss

    print(
        f"{args.num},{args.lmax},{param.size},"
        f"{forward_time:.6g},{forward_backward_time:.6g},{peak_exec_rss_gb:.6g}"
    )

if __name__ == "__main__":
    main()
