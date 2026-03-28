import argparse
import resource
import time

import jax
import jax.numpy as jnp
import numpy as onp
from func_helper import tmat_dreams
from jax import jit, value_and_grad
from jax.tree_util import tree_map

jax.config.update("jax_enable_x64", True)


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def block(x):
    return tree_map(
        lambda y: y.block_until_ready() if hasattr(y, "block_until_ready") else y, x
    )


def build_problem(num, lmax, lmax_glob):
    eps_emb = 1.0
    eps_obj = 2.5**2
    epsilons = onp.tile(onp.array([eps_obj, eps_emb]), (num, 1))
    radius = 80.0
    R = 400.0
    ni = onp.linspace(0.0, 360.0, num + 1)[:-1]
    x = R * onp.cos(ni * onp.pi / 180.0)
    y = R * onp.sin(ni * onp.pi / 180.0)
    z = onp.zeros_like(x)
    positions = onp.array([x, y, z]).T
    radii = onp.ones(num) * radius
    wl1 = 800.0
    k0 = 2 * onp.pi / wl1

    cfg = {
        "num": num,
        "lmax": lmax,
        "k0": k0,
        "lmax_glob": lmax_glob,
        "eps_emb": eps_emb,
        "eps_obj": eps_obj,
        "epsilons": epsilons,
        "poltype": "parity",
    }

    param = onp.append(positions.flatten(), radii)
    return cfg, jnp.array(param)


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

    cfg, param = build_problem(args.num, args.lmax, args.lmax_glob)

    f = lambda p: tmat_dreams(p, cfg)
    vg = lambda p: value_and_grad(tmat_dreams)(p, cfg)

    if args.jit_forward:
        f = jit(f)
    if args.jit_vg:
        vg = jit(vg)

    block(f(param))
    block(vg(param))

    for _ in range(args.warmup):
        block(f(param))
        block(vg(param))

    t0 = time.perf_counter()
    for _ in range(args.runs):
        block(f(param))
    t1 = time.perf_counter()
    forward_time = (t1 - t0) / args.runs

    t0 = time.perf_counter()
    for _ in range(args.runs):
        block(vg(param))
    t1 = time.perf_counter()
    forward_backward_time = (t1 - t0) / args.runs

    print(
        f"{args.num},{args.lmax},{param.size},{forward_time:.6g},{forward_backward_time:.6g},{peak_rss_gb():.6g}"
    )


if __name__ == "__main__":
    main()
