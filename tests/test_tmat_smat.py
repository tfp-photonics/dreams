import jax
import jax.numpy as np
import numpy as onp
import math
from dreams.jax_tr import smat_spheres, smat_spheres_full, tr
from dreams.jax_tmat import global_tmat, sphere, core_shell_sphere
from test_cfg import config, prepare, check_dist, limit_box
from dreams.jax_waves import plane_wave, default_plane_wave
from dreams.jax_la_bounded import lsumsw2d_shift_vmap
import treams.lattice as la
import treams
import treams.lattice as la
import time
import timeit
import os
import pytest
from dreams.jax_coord import car2pol as car2pol_jax
from treams.special import car2pol as car2pol_ref 
from dreams.jax_recursive import incgamma, _redincgamma
from dreams.jax_la_bounded import _recsw2d
import treams.special as cs
out_dir = os.path.dirname(__file__) + "/out/"
max_lmax = 3

def t_matrix_jax_sphere(cfg):
    _, radii, epsilons = prepare(cfg)
    t_mat = sphere(cfg.lmax, cfg.k0, radii[0], epsilons[0], helicity=cfg.helicity)
    return t_mat

def t_matrix_jax_coreshell(cfg):
    _, radii, epsilons = prepare(cfg)
    rad = np.array([cfg.r_core, cfg.r_shell])
    epsilon = np.array([cfg.eps_fg, cfg.eps_sh, cfg.eps_bg])
    t_mat = core_shell_sphere(cfg.lmax, cfg.k0, rad, epsilon, poltype=cfg.polarization)
    return t_mat

def t_matrix_treams_coreshell(cfg):
    rad = np.array([cfg.r_core, cfg.r_shell])
    epsilon = np.array([cfg.eps_fg, cfg.eps_sh, cfg.eps_bg])
    sphere = treams.TMatrix.sphere(
        cfg.lmax, cfg.k0, rad, epsilon, cfg.polarization
    )
    return sphere


def t_matrix_treams_sphere(cfg):
    materials = [treams.Material(cfg.eps_fg), treams.Material(cfg.eps_bg)]
    _, radii, _ = prepare(cfg)
    sphere = treams.TMatrix.sphere(
        cfg.lmax, cfg.k0, radii[0], materials, cfg.polarization
    )
    return sphere

def t_matrix_cluster_jax(cfg):
    pos, radii, epsilons = prepare(cfg)
    t_matrix, _, _ = global_tmat(pos, radii, epsilons, cfg.lmax, cfg.k0, cfg.helicity)
    return t_matrix

def s_matrix_local_jax(cfg):
    pos, radii, epsilons = prepare(cfg) 
    s_matrix, modes_s = smat_spheres(
        radii,
        epsilons,
        cfg.eps_bg,
        cfg.lmax,
        cfg.k0,
        pos,
        cfg.helicity,
        cfg.kx,
        cfg.ky,
        cfg.lat,
        # eps_below=cfg.eps_below,
        # eps_above=cfg.eps_above,
        rmax_coef=cfg.rmax_coef,
        local=True
    )
    
    return s_matrix

def s_matrix_global_jax(cfg):
    pos, radii, epsilons = prepare(cfg) 
    ans = limit_box(pos, radii, pitch=cfg.lat, min_gap=0.0)  
    if ans > 0:
        raise ValueError(f"Sphere crosses unit-cell boundary; max violation {ans:.3e}")
    s_matrix, modes_s = smat_spheres(
        radii,
        epsilons,
        cfg.eps_bg,
        cfg.lmax,
        cfg.k0,
        pos,
        cfg.helicity,
        cfg.kx,
        cfg.ky,
        cfg.lat,
        origin=None,
        rmax_coef=cfg.rmax_coef,
        local=False
    )

    return s_matrix

def s_matrix_global_stack_jax(cfg):
    pos, radii, epsilons = prepare(cfg) 
    s_matrix, modes_s = smat_spheres_full(
        radii,
        epsilons,
        cfg.eps_bg,
        cfg.lmax,
        cfg.k0,
        pos,
        cfg.helicity,
        cfg.kx,
        cfg.ky,
        cfg.lat,
        eps_below=cfg.eps_below,
        eps_above=cfg.eps_above,
        origin=None,
        rmax_coef=cfg.rmax_coef,
        local=False
    )

    return s_matrix


def t_matrix_cluster_treams(cfg) -> treams.TMatrix:
    materials = [treams.Material(cfg.eps_fg), treams.Material(cfg.eps_bg)]
    pos, radii, _ = prepare(cfg)
    spheres = [
        treams.TMatrix.sphere(cfg.lmax, cfg.k0, r, materials, cfg.polarization)
        for r in radii
    ]
    cluster = treams.TMatrix.cluster(spheres, pos)
    tm = cluster.interaction.solve()
    tm = tm.expand(treams.SphericalWaveBasis.default(cfg.lmax))
    return tm


def s_matrix_local_treams(cfg):
    lattice = treams.Lattice.square(cfg.lat)
    materials = [treams.Material(cfg.eps_fg), treams.Material(cfg.eps_bg)]
    pos, radii, _ = prepare(cfg)
    spheres = [
        treams.TMatrix.sphere(cfg.lmax, cfg.k0, r, materials, cfg.polarization)
        for r in radii
    ]
    tm = treams.TMatrix.cluster(spheres, pos)
    metasurf_t = tm.latticeinteraction.solve(lattice, [cfg.kx, cfg.ky])
    b = lattice.reciprocal
    # diff_orders = la.diffr_orders_circle(b, rmax=cfg.rmax_coef * cfg.k0)
    # kpar = diff_orders @ b
    # pwb = treams.PlaneWaveBasisByComp.default(kpar)
    kpar = [0, 0]
    pwb = treams.PlaneWaveBasisByComp.diffr_orders(kpar, lattice, cfg.rmax_coef * cfg.k0 )
    metasurf_s = treams.SMatrices.from_array(metasurf_t, pwb)
    return metasurf_s

    print("pwb LEN", len(pwb), pwb.kvecs(cfg.k0))
    print("t meta", metasurf_t.shape)

    illu = treams.plane_wave(kpar, 0, k0=cfg.k0, basis= pwb, material=cfg.eps_bg) 
    print("illu in fun", illu)
    return metasurf_s.tr(illu)[0]

def s_matrix_global_treams(cfg):
    lattice = treams.Lattice.square(cfg.lat)
    global_t = t_matrix_cluster_treams(cfg)
    kpar = [cfg.kx, cfg.ky]
    metasurf_t = global_t.latticeinteraction.solve(lattice, kpar)
    b = lattice.reciprocal
    pwb = treams.PlaneWaveBasisByComp.diffr_orders(kpar, lattice, cfg.rmax_coef * cfg.k0 )

    # diff_orders = la.diffr_orders_circle(b, rmax=cfg.rmax_coef * cfg.k0)
    # kpar = diff_orders @ b
    #pwb = treams.PlaneWaveBasisByComp.default(kpar)
    metasurf_s = treams.SMatrices.from_array(metasurf_t, pwb)
    return metasurf_s
    

def s_matrix_global_stack_treams(cfg):
    lattice = treams.Lattice.square(cfg.lat)
    global_t = t_matrix_cluster_treams(cfg)
    kpar = [cfg.kx, cfg.ky]
    metasurf_t = global_t.latticeinteraction.solve(lattice, kpar)
    b = lattice.reciprocal
    pwb = treams.PlaneWaveBasisByComp.diffr_orders(kpar, lattice, cfg.rmax_coef * cfg.k0 )
    metasurf_s = treams.SMatrices.from_array(metasurf_t, pwb)
    
    dist = np.array([0, 0, cfg.lat / 2])
    mats = {
        "bg": treams.Material(cfg.eps_bg),
        "above": treams.Material(cfg.eps_above),
        "below": treams.Material(cfg.eps_below),
    }
    epsint1 = np.array([cfg.eps_above, cfg.eps_bg])
    epsint2 = np.array([cfg.eps_bg, cfg.eps_below])
    inter1 = treams.SMatrices.interface(pwb, cfg.k0, epsint1, poltype=cfg.polarization)
    prop = treams.SMatrices.propagation(dist, pwb, cfg.k0, mats["bg"], poltype=cfg.polarization)
    inter2 = treams.SMatrices.interface(pwb, cfg.k0, epsint2, poltype=cfg.polarization)
    stck = [inter1, prop, metasurf_s, prop, inter2]
    s_matrix = treams.SMatrices.stack(stck)
    return s_matrix

def compare(cfg, f1, f2, rtol=1e-12, atol=1e-14):
    start = time.time()
    v1 = f1(cfg)
    time1 = time.time() - start
    v2 = f2(cfg)
    np.save("v1.npy", v1)
    np.save("v2.npy", onp.array(v2))

    time2 = time.time() - (start + time1)
    print("timing", time2)
    v1 = onp.array(v1)
    v2 = onp.asarray(v2, dtype=onp.complex128)
    rel_err = onp.linalg.norm(v1 - v2) / max(onp.linalg.norm(v2), 1.0)
    max_abs = onp.max(onp.abs(v1 - v2))
    ok = onp.allclose(v1, v2, rtol=rtol, atol=atol)
    if not ok:
        raise AssertionError(
            f"{f1.__name__} vs {f2.__name__} failed: "
            f"rel_err={rel_err:.3e}, max||={max_abs:.3e}, rtol={rtol}, atol={atol}"
        )
    return time1, time2, rel_err

def test_t_mat_sphere_parity():
    cfg = config()
    cfg.lmax = 3
    cfg.helicity = False
    compare(cfg, t_matrix_jax_sphere, t_matrix_treams_sphere, rtol=1e-12, atol=1e-14 )


def test_t_mat_sphere_helicity():
    cfg = config()
    cfg.lmax = 3
    cfg.helicity = True
    compare(cfg, t_matrix_jax_sphere, t_matrix_treams_sphere,  rtol=1e-12, atol=1e-14 )

def test_t_mat_coreshell_parity():
    cfg = config()
    cfg.lmax = 3
    cfg.helicity = False
    compare(cfg, t_matrix_jax_coreshell, t_matrix_treams_coreshell,  rtol=1e-12, atol=1e-14 )

def test_t_mat():
    cfg = config()
    times = []
    errors = []
    lmaxs = np.arange(1, max_lmax + 1)
    for lmax in lmaxs:
        cfg.lmax = lmax
        t1, t2, error = compare(
            cfg, t_matrix_cluster_jax, t_matrix_cluster_treams,  rtol=1e-12, atol=1e-14 
        )
        times.append([t1, t2])
        errors.append(error)
    times = onp.array(times)
    errors = onp.array(errors)

#@pytest.mark.skip(reason="Skipping this test temporarily")
def test_s_mat_local():
    cfg = config()
    times = []
    errors = []
    lmaxs = onp.arange(max_lmax, max_lmax + 1)
    for lmax in lmaxs:
        cfg.lmax = lmax
        t1, t2, error = compare(
            cfg, s_matrix_local_jax, s_matrix_local_treams,  rtol=1e-12, atol=1e-13 
        )   
        times.append([t1, t2])
        errors.append(error)
    times = onp.array(times)
    errors = onp.array(errors)
    print('times', times, 'errors', errors)

def test_s_mat_global():
    cfg = config()
    times = []
    errors = []
    lmaxs = onp.arange(1, max_lmax + 1)
    cfg.helicity = False
    for lmax in lmaxs:
        cfg.lmax = lmax
        t1, t2, error = compare(
            cfg, s_matrix_global_jax, s_matrix_global_treams,  rtol=1e-12, atol=1e-14 
        )   
        times.append([t1, t2])
        errors.append(error)
    times = onp.array(times)
    errors = onp.array(errors)
    print('times', times, 'errors', errors)


if __name__ == "__main__":
    test_s_mat_local()