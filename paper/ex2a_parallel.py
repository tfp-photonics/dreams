import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import jax
import jax.numpy as jnp
import nlopt
import numpy as np
from func_helper import dreams_rcd, treams_rcd, treams_rcd_parallel
from refractiveindex import RefractiveIndexMaterial

jax.config.update("jax_enable_x64", True)

threads_per_worker = 1  # if  many workers
os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_worker)
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false "
    f"intra_op_parallelism_threads={threads_per_worker}"
)

# ---------------------------
# constraints 
# ---------------------------


def _split_params(p, n_spheres: int):
    """p = [pos_flat (3n), radii (n)]"""
    pos = jnp.reshape(p[: 3 * n_spheres], (n_spheres, 3))
    rad = p[3 * n_spheres :]
    return pos, rad


def overlap_jax(pos, radii, safespace: float):  # , eps: float = 1e-18):
    # def overlap_jax(pos, radii, safespace: float, eps: float = 1e-18):
    """Overlap margin, <= 0 means no overlaps (with safespace)."""
    pos = jnp.reshape(pos, (-1, 3))
    d = pos[:, None, :] - pos[None, :, :]
    rd = radii[:, None] + radii[None, :]
    I, J = jnp.triu_indices(pos.shape[0], k=1)
    dmat = jnp.linalg.norm(d[I, J, :], axis=-1)
    rd_ij = rd[I, J]
    return jnp.max(rd_ij - dmat) + safespace


def limit_sphere_jax(pos, radii, pitch: float):
    """No sphere crosses circumscribing sphere of radius pitch/2 at origin."""
    pos = jnp.reshape(pos, (-1, 3))
    r2 = jnp.sum(pos**2, axis=1)
    mask0 = jnp.isclose(r2, 0.0)
    rs = jnp.sqrt(jnp.where(mask0, 1.0, r2))
    rs = jnp.where(mask0, 0.0, rs)
    dist = rs + radii
    return jnp.max(dist - pitch / 2.0)

# ---------------------------
# NLopt optimizer
# ---------------------------


def run_optimizer_nlopt(
    p0: np.ndarray,
    n_steps: int,
    *,
    dreams_obj,  
    n_spheres: int,
    pitch: float,
    safespace: float,
    rl: float,
):
    """
    Returns:
      p_opt (np.ndarray), va (list[float]), pas (list[np.ndarray])
    """
    p0 = np.asarray(p0, dtype=np.float64)

    va = []
    pas = []

    # Build jitted functions once
    obj_vg = jax.jit(jax.value_and_grad(dreams_obj))

    def c_overlap(p):
        pos, rad = _split_params(p, n_spheres)
        return overlap_jax(pos, rad, safespace)

    def c_limit(p):
        pos, rad = _split_params(p, n_spheres)
        return limit_sphere_jax(pos, rad, pitch)

    c1_vg = jax.value_and_grad(c_overlap)
    c2_vg = jax.value_and_grad(c_limit)

    # Warmup compile 
    _v, _g = obj_vg(p0)
    jax.block_until_ready(_v)
    _v, _g = c1_vg(p0)
    jax.block_until_ready(_v)
    _v, _g = c2_vg(p0)
    jax.block_until_ready(_v)

    opt = nlopt.opt(nlopt.LD_MMA, p0.size)

    index = 0

    def nlopt_objective(params, grad_out):
        nonlocal index
        if index % 10 == 0:
            print(f"evaluating objective function: {index}")
        index += 1

        v, g = obj_vg(np.asarray(params, dtype=np.float64))
        if grad_out.size > 0:
            grad_out[:] = np.asarray(g, dtype=np.float64).ravel()

        v_f = float(np.asarray(v))
        va.append(v_f)
        pas.append(np.array(params, copy=True))
        return v_f

    opt.set_max_objective(nlopt_objective)

    def nlopt_c1(params, grad_out):
        v, g = c1_vg(np.asarray(params, dtype=np.float64))
        if grad_out.size > 0:
            grad_out[:] = np.asarray(g, dtype=np.float64).ravel()
        return float(np.asarray(v))

    def nlopt_c2(params, grad_out):
        v, g = c2_vg(np.asarray(params, dtype=np.float64))
        if grad_out.size > 0:
            grad_out[:] = np.asarray(g, dtype=np.float64).ravel()
        return float(np.asarray(v))

    opt.add_inequality_constraint(nlopt_c1, 1e-8)
    opt.add_inequality_constraint(nlopt_c2, 1e-8)

    # Lower bounds: last n entries >= rl
    lb = [-float("inf")] * p0.size
    lb[3 * n_spheres :] = [float(rl)] * n_spheres
    opt.set_lower_bounds(lb)

    opt.set_maxeval(int(n_steps))
 
    p_opt = opt.optimize(np.array(p0, copy=True))
    pas.append(np.array(p_opt, copy=True))
    return np.asarray(p_opt, dtype=np.float64), va, pas


def run_one_wl(
    wl: float, base_positions: np.ndarray, base_radii: np.ndarray, *, cfg_common: dict
):
    """
    Runs everything for one wl and writes an HDF5 file.
    Returns out_path.
    """

    k0 = 2 * np.pi / wl

    si = RefractiveIndexMaterial("main", "Si", "Schinke")
    eps_object = si.get_epsilon(wl)

    cfg = dict(cfg_common)
    cfg["k0"] = k0
    cfg["eps_object"] = eps_object

    # ---- objective as a pure function of params ----
    def dreams_obj(p):
        return dreams_rcd(p, eps_object=eps_object, k0=k0, cfg=cfg)
    
    # ---- optimize ----
    n_spheres = base_radii.shape[0]
    p0 = np.concatenate([base_positions.reshape(-1), base_radii]).astype(np.float64)
    r_treams, _, _, _, _ = treams_rcd(p0, cfg)
    r_dreams = dreams_obj(p0)
    print("COMPARE vs treams", np.isclose(r_treams, r_dreams))
    p_opt, va, pas = run_optimizer_nlopt(
        p0,
        cfg_common["n_steps"],
        dreams_obj=dreams_obj,
        n_spheres=n_spheres,
        pitch=cfg_common["pitch"],
        safespace=cfg_common["safespace"],
        rl=cfg_common["rl"],
    )

    # ---- postprocess ----
    posf = p_opt[: 3 * n_spheres].reshape((-1, 3))
    radf = p_opt[3 * n_spheres :]

    r_treams2, _, _, _, _ = treams_rcd(p_opt, cfg)
    r_dreams2 = dreams_obj(p_opt)

    # ---- sweep ----
    wls_range = np.arange(900.0, 1200.0, 2.0)
    eps_objects_range = si.get_epsilon(wls_range)

    cfg_sweep = dict(cfg)
    cfg_sweep["k0s"] = 2 * np.pi / wls_range
    cfg_sweep["eps_objects"] = eps_objects_range
    cfg_sweep["lmax"] = 15

    rcdfinal_15, r1_final, r2_final, phasor_x, phasor_y = treams_rcd_parallel(
        p_opt, cfg_sweep
    )

    # ---- output name ----
    name = (
        f"parallel-si-safe-{cfg_common['safespace']}-rcd-circle-R-{cfg_common['R']}-zshift-{cfg_common['zmax']}-"
        f"nosub-num-{n_spheres}-randr-{cfg_common['shift2']}-randpos-{cfg_common['shift']}-rinit-{cfg_common['r_init']}-"
        f"lmax-{cfg_common['lmax']}-{cfg_common['lmax_glob']}-nsteps-{cfg_common['n_steps']}-wl-{wl}-pitch-{cfg_common['pitch']}-"
        f"rmax-{cfg_common['rmax_coef']}-rl-{cfg_common['rl']}-with-limit.h5"
    )

    os.makedirs("paper_results", exist_ok=True)
    out_path = os.path.join("paper_results", name)

    try:
        with open(__file__) as f:
            script = f.read()
    except Exception:
        script = "# __file__ not available"

    with h5py.File(out_path, "w") as f:
        f["script"] = script
        f["values"] = np.asarray(va)
        f["wl"] = float(wl)
        f["k0"] = float(k0)

        f["R"] = float(cfg_common["R"])
        f["safety"] = float(cfg_common["safespace"])
        f["pitch"] = float(cfg_common["pitch"])
        f["lmax"] = int(cfg_common["lmax"])
        f["lmax_glob"] = int(cfg_common["lmax_glob"])
        f["rmax_coef"] = float(cfg_common["rmax_coef"])
        f["helicity"] = bool(cfg_common["helicity"])
        f["kx"] = float(cfg_common["kx"])
        f["ky"] = float(cfg_common["ky"])

        f["pos_init"] = np.asarray(base_positions)
        f["radii_init"] = np.asarray(base_radii)
        f["pos_final"] = np.asarray(posf)
        f["radii_final"] = np.asarray(radf)

        f["eps_obj"] = np.asarray(eps_object)
        f["eps_emb"] = float(cfg_common["eps_medium"])

        f["rcd_final_wls_150_lm15"] = np.asarray(rcdfinal_15)
        f["r1_final_wls_150_lm15"] = np.asarray(r1_final)
        f["r2_final_wls_150_lm15"] = np.asarray(r2_final)
        f["wls_150_range"] = np.asarray(wls_range)
        f["phasor_x"] = np.asarray(phasor_x)
        f["phasor_y"] = np.asarray(phasor_y)


        f["r_treams_initial"] = np.asarray(r_treams)
        f["r_dreams_initial"] = np.asarray(r_dreams)
        f["r_treams_final"] = np.asarray(r_treams2)
        f["r_dreams_final"] = np.asarray(r_dreams2)

    return out_path


def main():
    # ----  parameters & setup ----
    lmax = 7
    lmax_glob = 7
    rmax_coef = 1
    wls = np.array([950.0, 1050.0])
    eps_medium = 1.5**2
    pitch = 600.0
    r_init = 10.0
    num = 5
    R = 170.0
    zmax = 0.0
    shift = 0.0
    shift2 = 0.0
    safespace = 5.0

    ni = np.linspace(0, 360.0, num + 1)[:-1]
    x = R * np.cos(ni * np.pi / 180.0)
    y = R * np.sin(ni * np.pi / 180.0)
    z = np.linspace(-zmax, zmax, x.shape[0])

    positions = np.array([x, y, z]).T
    positions = positions + np.random.uniform(-shift, shift, size=positions.shape)

    radii = np.ones_like(positions[:, 0]) * r_init
    radii = radii + np.random.uniform(-shift2, shift2, size=radii.shape[0])

    helicity = False
    kx = 0.0
    ky = 0.0
    rl = 5.0
    n_steps = 200

    # initial params 
    param0 = np.concatenate([positions.reshape(-1), radii]).astype(np.float64)

    cfg_common = dict(
        lmax=lmax,
        lmax_glob=lmax_glob,
        pitch=pitch,
        eps_medium=eps_medium,
        rmax_coef=rmax_coef,
        helicity=helicity,
        kx=kx,
        ky=ky,
        rl=rl,
        n_steps=n_steps,
        R=R,
        zmax=zmax,
        shift=shift,
        shift2=shift2,
        safespace=safespace,
        r_init=r_init,
        param0=param0, 
    )

    # ---- multiprocessing over wavelengths ----
    ctx = mp.get_context("spawn")
    max_workers = len(wls)
    print(f"max workers {max_workers}")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = [
            ex.submit(run_one_wl, float(wl), positions, radii, cfg_common=cfg_common)
            for wl in wls
        ]
        for fut in as_completed(futures):
            out_path = fut.result()
            print("saved:", out_path)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
