import jax.numpy as np
import numpy as anp
import h5py
import treams
from refractiveindex import RefractiveIndexMaterial
from functools import partial
import nlopt
import jax
from jax import value_and_grad
from dreams.jax_tmat import defaultmodes
from func_helper import dreams_rcd, treams_rcd, treams_rcd_parallel

jax.config.update("jax_enable_x64", True)


def overlap(pos, radii):
    pos = np.reshape(pos, (-1, 3))
    d = pos[:, None, :] - pos
    rd = radii[:, None] + radii
    nonzero = np.triu_indices(len(pos), k=1)
    dmat = np.linalg.norm(d[nonzero], axis=-1)
    rd = rd[np.triu_indices(len(radii), k=1)]
    return np.max(rd - dmat) + safety


def limit(pos, radii):
    pos = np.reshape(pos, (-1, 3))
    r2 = np.sum(pos**2, axis=1)
    mask0 = np.isclose(r2, 0.0)
    rs = np.sqrt(np.where(mask0, 1.0, r2))
    rs = np.where(mask0, 0.0, rs)
    dist = rs + radii
    return np.max(dist - pitch / 2)


def limit_unit(pos, radii):
    pos = np.reshape(pos, (-1, 3))
    ans1 = np.max(np.abs(pos[:, 0]) - pitch / 2)
    ans2 = np.max(np.abs(pos[:, 1]) - pitch / 2)
    ans3 = np.max(np.abs(pos[:, 2]) - pitch / 2)
    return np.max(np.array([ans1, ans2, ans3]))


def nlopt_constraint(params, gd):
    v, g = value_and_grad(
        lambda p: overlap(p[: 3 * len(radii)], p[3 * len(radii) :])
    )(params)
    if gd.size > 0:
        gd[:] = anp.asarray(g).ravel()
    return v.item()


def nlopt_constraint2(params, gd):
    v, g = value_and_grad(
        lambda p: limit(p[: 3 * len(radii)], p[3 * len(radii) :])
    )(params)
    if gd.size > 0:
        gd[:] = anp.asarray(g).ravel()
    return v.item()


def nlopt_constraint2l(params, gd):
    v, g = value_and_grad(
        lambda p: limit_unit(p[: 3 * len(radii)], p[3 * len(radii) :])
    )(params)
    if gd.size > 0:
        gd[:] = anp.asarray(g).ravel()
    return v.item()


def optimizer(p, n_steps):
    opt = nlopt.opt(nlopt.LD_MMA, p.flatten().shape[0])

    v_g = jax.jit(value_and_grad(opt_func_global))
    _ = v_g(np.asarray(p, dtype=np.float64))

    index = 0

    def nlopt_objective(params, gd):
        nonlocal index
        if index % 10 == 0:
            print(f"evaluating objective function: {index}")
        index += 1
        v, g = v_g(np.asarray(params, dtype=np.float64))

        if gd.size > 0:
            gd[:] = anp.asarray(g).ravel()

        v_np = anp.array(v)
        va.append(v_np)
        pas.append(anp.array(params, copy=True))
        return v_np.item()

    opt.set_max_objective(nlopt_objective)
    opt.add_inequality_constraint(nlopt_constraint, 1e-8)
    opt.add_inequality_constraint(nlopt_constraint2, 1e-8)

    ln = p.shape[0] // 4
    bound = [-float("inf")] * p.shape[0]
    bound[3 * p.shape[0] // 4 :] = [rl] * ln
    opt.set_lower_bounds(bound)
    opt.set_maxeval(n_steps)

    p_opt = opt.optimize(anp.asarray(p, dtype=anp.float64))
    pas.append(anp.array(p_opt, copy=True))
    return v_g

def opt_func_global(params):
    drcds = []
    for i, wl in enumerate(wls):
        k0 = 2 * anp.pi / wl
        eps_object = eps_objects[i]
        drcds.append(dreams_rcd_func(params, eps_object, k0))
    drcds = np.array(drcds)
    x, y = drcds[0], drcds[1]
    eps = 1e-8
    return (2 * x * y + eps * np.min(drcds)) / (x + y + eps)


lmax = 7
lmax_glob = 8
rmax_coef = 1
wls = anp.array([950.0, 1050.0])
k0s = 2 * np.pi / wls
si = RefractiveIndexMaterial("main", "Si", "Schinke")
eps_objects = si.get_epsilon(wls)
eps_medium = 1.5**2
pitch = 600.0
r_init = 10.0
num = 5
R = 170.0
zmax = 0.0

ni = np.linspace(0, 360.0, num + 1)[:-1]
x = R * np.cos(ni * np.pi / 180)
y = R * np.sin(ni * np.pi / 180)
z = np.linspace(-zmax, zmax, x.shape[0])

shift = 0.0
shift2 = 0.0
loops = 1 if shift == 0 else 2

for _ in range(loops):
    seed = anp.random.randint(0, 2**32 - 1)
    anp.random.seed(seed)

    positions = np.array([x, y, z]).T
    positions = positions + anp.random.uniform(-shift, shift, size=positions.shape)
    radii = np.ones_like(positions[:, 0]) * r_init
    radii = radii + anp.random.uniform(-shift2, shift2, size=positions.shape[0])

    modes = defaultmodes(lmax)
    helicity = False
    kx = 0.0
    ky = 0.0
    rl = 5.0
    safety = 5.0
    n_steps = 250
    kpars = anp.array([kx, ky])

    va = []
    pas = []

    pos = positions.flatten()
    cfg = {
        "lmax": lmax,
        "lmax_glob": lmax_glob,
        "pitch": pitch,
        "eps_medium": eps_medium,
        "eps_objects": eps_objects,
        "rmax_coef": rmax_coef,
        "helicity": helicity,
        "kx": kx,
        "ky": ky,
        "k0s": 2 * anp.pi / wls,
        "wls": wls,
    }

    params_init = np.append(pos, radii)

    _ = overlap(pos, radii)
    _ = limit(pos, radii)

    dreams_rcd_func = partial(dreams_rcd, cfg=cfg)

    v_g = optimizer(params_init, n_steps)

    posf = pas[-1][: int(3.0 / 4.0 * len(pas[-1]))].reshape((-1, 3))
    radf = pas[-1][int(3.0 / 4.0 * len(pas[-1])) :]
    v_final, _ = v_g(np.asarray(pas[-1], dtype=np.float64))
    va.append(anp.asarray(v_final))
    v_final_treams, tr_x, tr_y, rxs, rys = treams_rcd_parallel(pas[-1], cfg)
    x, y = v_final_treams[0], v_final_treams[1]
    eps = 1e-8
    v_final_treams = (2 * x * y + eps * anp.min(v_final_treams)) / (x + y + eps)

    print("CLOSE: dreams", v_final, "treams", v_final_treams)
    params_final = anp.concatenate((anp.array(posf).flatten(), anp.array(radf)))

    wls_range = np.arange(900.0, 1200.0, 2)
    eps_objects_range = si.get_epsilon(wls_range)
    cfg["k0s"] = 2 * anp.pi / wls_range
    cfg["eps_objects"] = eps_objects_range
    cfg["lmax"] = 15

    rcdfinal_15, rx_final, ry_final, phasors_x, phasors_y = treams_rcd_parallel(
        params_final, cfg
    )
    rcdinit_15, rx_init, ry_init, _, _ = treams_rcd_parallel(params_init, cfg)

    name = (
        f"1r-balanced-epsminxy-zshift-{zmax}-numwls-{len(wls)}-safecor-{safety}-"
        f"notnorm-si-rcd-circle-adapt-R-{R}-nosub-num-{len(radii)}-randr-{shift2}-"
        f"randpos-{shift}-rinit-{r_init}-lmax-{lmax}-{lmax_glob}-nsteps-{n_steps}-"
        f"wls-{wls[0]}-{wls[-1]}-pitch-{pitch}-rmax-{rmax_coef}-rl-{rl}-"
        f"with-limit-seed-{seed}-jit.h5"
    )

    try:
        with open(__file__, "r") as file:
            script = file.read()
    except NameError:
        script = ""
    print("name", name)
    with h5py.File("paper_results/" + name, "w") as f:
        f["script"] = script
        f["values"] = anp.asarray(va)
        f["R"] = R
        f["radii_final"] = anp.asarray(radf)
        f["pos_final"] = anp.asarray(posf)
        f["radii_init"] = anp.asarray(radii)
        f["pos_init"] = anp.asarray(positions)
        f["safety"] = safety
        f["pitch"] = pitch
        f["lmax"] = lmax
        f["lmax_glob"] = lmax_glob
        f["rmax_coef"] = rmax_coef
        f["kx"] = kx
        f["ky"] = ky
        f["wls"] = anp.asarray(wls)
        f["eps_objs"] = anp.asarray(eps_objects)
        f["eps_emb"] = eps_medium
        f["rcd_final_wls_150_lm15"] = anp.asarray(rcdfinal_15)
        f["r1_final_wls_150_lm15"] = anp.asarray(rx_final)
        f["r2_final_wls_150_lm15"] = anp.asarray(ry_final)
        f["rcd_init_wls_150_lm15"] = anp.asarray(rcdinit_15)
        f["r1_init_wls_150_lm15"] = anp.asarray(rx_init)
        f["r2_init_wls_150_lm15"] = anp.asarray(ry_init)
        f["wls_150_range"] = anp.asarray(wls_range)
        f["phasor_x"] = anp.asarray(phasors_x)
        f["phasor_y"] = anp.asarray(phasors_y)