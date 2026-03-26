import jax
import jax.numpy as jnp
import numpy as anp
import treams
from joblib import Parallel, delayed
from treams._core import SphericalWaveBasis as SWB

from dreams.jax_op import _sw_pw_expand
from dreams.jax_tmat import (
    defaultmodes,
    tmats_interact,
    tmats_no_int,
)
from dreams.jax_tr import smat_spheres, tr
from dreams.jax_waves import default_plane_wave, efield, plane_wave


def unpack_params(params, cfg):
    params = jnp.asarray(params)
    n_pos = 3 * cfg["num"]
    pos_flat = params[:n_pos]
    radii = params[n_pos:]
    positions = pos_flat.reshape(-1, 3)
    return positions, radii


def tmat_dreams(params, cfg):
    k0 = cfg["k0"]
    positions, radii = unpack_params(params, cfg)
    modes = defaultmodes(cfg["lmax"], cfg["num"])
    tm = tmats_no_int(
        radii, cfg["epsilons"], cfg["lmax"], k0, cfg["poltype"] == "helicity"
    )
    tm = tmats_interact(
        tm, positions, modes, k0, cfg["poltype"] == "helicity", cfg["eps_emb"]
    )
    return jnp.sum(jnp.abs(tm))


def tmat_treams(params, cfg):
    positions, radii = unpack_params(params, cfg)
    eps_obj = cfg["eps_obj"]
    eps_emb = cfg["eps_emb"]
    lmax = cfg["lmax"]
    k0 = cfg["k0"]
    pol = cfg["pol"]
    poltype = cfg["poltype"]
    treams.config.POLTYPE = poltype
    materials = [treams.Material(eps_obj), treams.Material(eps_emb)]
    spheres = [treams.TMatrix.sphere(lmax, k0, radius, materials) for radius in radii]
    tm = treams.TMatrix.cluster(spheres, positions)
    sph_cluster = tm.interaction.solve()
    return anp.sum(anp.abs(sph_cluster))


def power_dreams(
    grid, weights, sca_dr, pidx, l, m, pols, positions, k0, eps_emb, poltype, B=128
):
    P = grid.shape[0]
    n_chunks = (P + B - 1) // B
    pad = n_chunks * B - P

    grid_p = jnp.pad(grid, ((0, pad), (0, 0)))
    w_p = jnp.pad(weights, (0, pad))

    grid_chunks = grid_p.reshape(n_chunks, B, 3)
    w_chunks = w_p.reshape(n_chunks, B)

    def step(acc, x):

        pts, ww = x  # (B,3), (B,)
        op = efield(
            pts,
            pidx,
            l,
            m,
            pols,
            positions,
            k0,
            eps_emb,
            modetype="singular",
            poltype=poltype,
        )
        fld = jnp.asarray(op @ sca_dr).reshape(B, 3)
        I = jnp.sum(jnp.abs(fld) ** 2, axis=-1)  # (B,)
        return acc + jnp.sum(I * ww), None  # scalar carry

    total, _ = jax.lax.scan(step, 0.0, (grid_chunks, w_chunks))
    return total


def fob_norm(params, cfg):
    k0 = cfg["k0"]
    d_solid_surf = cfg["d_solid_surf"].flatten()
    positions, radii = unpack_params(params, cfg)
    modes = defaultmodes(cfg["lmax"], cfg["num"])
    poltype = cfg["poltype"]

    ts = tmats_no_int(radii, cfg["epsilons"], cfg["lmax"], k0, poltype == "helicity")
    tm = tmats_interact(ts, positions, modes, k0, poltype == "helicity", cfg["eps_emb"])
    inc = treams.plane_wave(
        [0, 0, cfg["Sgn"] * k0], cfg["pol"], k0=k0, material=cfg["eps_emb"]
    )
    plane_basis = inc.basis
    pidx, l, m, pols = defaultmodes(cfg["lmax"], nmax=cfg["num"])
    inc_expand = _sw_pw_expand(
        plane_basis,
        pidx,
        l,
        m,
        pols,
        positions,
        k0,
        cfg["eps_emb"],
        poltype=poltype,
        modetype="regular",
        treams=True,
    ) @ jnp.array(inc)
    sca_dr = tm @ inc_expand

    forward = power_dreams(
        cfg["grid_f"],
        d_solid_surf,
        sca_dr,
        pidx,
        l,
        m,
        pols,
        positions,
        k0,
        cfg["eps_emb"],
        poltype,
        B=20,
    )
    backward = power_dreams(
        cfg["grid_b"],
        d_solid_surf,
        sca_dr,
        pidx,
        l,
        m,
        pols,
        positions,
        k0,
        cfg["eps_emb"],
        poltype,
        B=20,
    )
    return forward / (forward + backward)


def fob(params, cfg):
    k0 = cfg["k0"]
    d_solid_surf = cfg["d_solid_surf"].flatten()
    positions, radii = unpack_params(params, cfg)
    modes = defaultmodes(cfg["lmax"], cfg["num"])
    poltype = cfg["poltype"]
    ts = tmats_no_int(radii, cfg["epsilons"], cfg["lmax"], k0, poltype == "helicity")
    tm = tmats_interact(ts, positions, modes, k0, poltype == "helicity", cfg["eps_emb"])
    inc = treams.plane_wave(
        [0, 0, cfg["Sgn"] * k0], cfg["pol"], k0=k0, material=cfg["eps_emb"]
    )
    plane_basis = inc.basis
    pidx, l, m, pols = defaultmodes(cfg["lmax"], nmax=cfg["num"])
    inc_expand = _sw_pw_expand(
        plane_basis,
        pidx,
        l,
        m,
        pols,
        positions,
        k0,
        cfg["eps_emb"],
        poltype=poltype,
        modetype="regular",
        treams=True,
    ) @ jnp.array(inc)
    sca_dr = tm @ inc_expand
    forward = power_dreams(
        cfg["grid_f"],
        d_solid_surf,
        sca_dr,
        pidx,
        l,
        m,
        pols,
        positions,
        k0,
        cfg["eps_emb"],
        poltype,
        B=20,
    )
    # return forward
    backward = power_dreams(
        cfg["grid_b"],
        d_solid_surf,
        sca_dr,
        pidx,
        l,
        m,
        pols,
        positions,
        k0,
        cfg["eps_emb"],
        poltype,
        B=20,
    )
    return forward / backward


def fob_treams(params, cfg):
    positions, radii = unpack_params(params, cfg)
    eps_obj = cfg["eps_obj"]
    eps_emb = cfg["eps_emb"]
    lmax = cfg["lmax"]
    wavelengths = cfg["wavelengths"]
    Sgn = cfg["Sgn"]
    pol = cfg["pol"]
    poltype   = cfg["poltype"]
    grid_f = cfg["grid_f"]
    grid_b = cfg["grid_b"]
    domg = cfg["d_solid_surf"].flatten()
    treams.config.POLTYPE = poltype

    materials = [treams.Material(eps_obj), treams.Material(eps_emb)]
    k0s = 2 * anp.pi / wavelengths

    spheres = [
        [treams.TMatrix.sphere(lmax, k0, radius, materials) for radius in radii]
        for k0 in k0s
    ]
    sph_cluster = [treams.TMatrix.cluster(tm_list, positions) for tm_list in spheres]
    sph_cluster = [tm.interaction.solve() for tm in sph_cluster]

    fwd_trms = anp.zeros_like(k0s)
    bck_trms = anp.zeros_like(k0s)
    tmats = []

    for i, k0 in enumerate(k0s):
        tm = sph_cluster[i]
        inc = treams.plane_wave(
            [0, 0, Sgn * k0],
            pol,
            k0=k0,
            material=tm.material,
            poltype=poltype,
        )
        sca_tr = tm @ inc.expand(tm.basis)

        Ef = sca_tr.efield(grid_f)
        Eb = sca_tr.efield(grid_b)

        forward = anp.sum(anp.sum(anp.abs(Ef) ** 2, axis=-1) * domg)
        backward = anp.sum(anp.sum(anp.abs(Eb) ** 2, axis=-1) * domg)

        fwd_trms[i] = forward
        bck_trms[i] = backward
        tmats.append(tm)

    return fwd_trms, bck_trms, tmats


def xs_treams(tm, illu, flux=0.5):
    if not tm.material.isreal:
        raise NotImplementedError
    illu = treams.PhysicsArray(illu)
    illu_basis = illu.basis
    illu_basis = illu_basis[-2] if isinstance(illu_basis, tuple) else illu_basis
    if not isinstance(illu_basis, SWB):
        illu = illu.expand(tm.basis)

    p = tm @ illu
    p_invksq = p * anp.power(tm.ks[tm.basis.pol], -2)
    tot = 0.5 * anp.real(p.conjugate().T * p_invksq.expand(p.basis)) / flux
    del illu.modetype
    return (
        0.5 * anp.real(p.conjugate().T @ p_invksq.expand(p.basis)) / flux,
        -0.5 * anp.real(illu.conjugate().T @ p_invksq) / flux,
        tot,
    )


def multipole_xs_treams(
    tmats,
    cfg,
    desired_terms=None,
):
    Sgn = cfg["Sgn"]
    pol = cfg["pol"]
    eps_emb = cfg["eps_emb"]
    wls = cfg["wavelengths"]
    lmax_glob = cfg["lmax_glob"]
    wls = anp.asarray(wls)
    k0s = 2 * anp.pi / wls
    if desired_terms is None:
        desired_terms = [
            "E$_1$",
            "M$_1$",
            "E$_2$",
            "M$_2$",
            "E$_3$",
            "M$_3$",
            "E$_4$",
            "M$_4$",
            "E$_5$",
            "M$_5$",
            "E$_6$",
            "M$_6$",
        ]

    basis = treams.SphericalWaveBasis.default(lmax_glob)

    basis_map = {
        f"E$_{l}$": (basis.pol == 1) & (basis.l == l)
        for l in range(1, len(desired_terms) // 2 + 1)
    }
    basis_map.update(
        {
            f"M$_{l}$": (basis.pol == 0) & (basis.l == l)
            for l in range(1, len(desired_terms) // 2 + 1)
        }
    )

    xss = {term: [] for term in desired_terms if term in basis_map}

    for i, k0 in enumerate(k0s):
        tm = tmats[i]
        tm = tm.expand(basis)
        inc = treams.plane_wave([0, 0, Sgn * k0], list(pol), k0=k0, material=eps_emb)
        xsa, _, tot = xs_treams(tm, inc)

        for term, mask in basis_map.items():
            value = anp.sum(tot[mask])
            xss[term].append(value)

    for term in xss:
        xss[term] = anp.asarray(xss[term])

    return xss


def field_wl_treams(params, cfg):
    Sgn = cfg["Sgn"]
    pol = cfg["pol"]
    eps_emb = cfg["eps_emb"]
    eps_obj = cfg["eps_obj"]
    materials = [treams.Material(eps_obj), treams.Material(eps_emb)]
    lmax = cfg["lmax"]
    k0 = cfg["k0"]
    grid_f = cfg["grid_f"]
    grid_b = cfg["grid_b"]
    d_solid_surf = cfg["d_solid_surf"]
    radii = params[int(len(params) * 3 / 4) :]
    positions = params[: int(len(params) * 3 / 4)].reshape((-1, 3))
    tms = [treams.TMatrix.sphere(lmax, k0, radius, materials) for radius in radii]
    tm = treams.TMatrix.cluster(tms, positions).interaction.solve()
    inc = treams.plane_wave([0, 0, Sgn * k0], pol, k0=tm.k0, material=tm.material)
    sca_tr = tm @ inc.expand(tm.basis)
    fbck = sca_tr.efield(grid_b)
    ffwd = sca_tr.efield(grid_f)
    intb = anp.array(anp.sum(anp.abs(fbck.__array__()) ** 2, axis=-1)).reshape(
        d_solid_surf.size
    )
    intf = anp.array(anp.sum(anp.abs(ffwd.__array__()) ** 2, axis=-1)).reshape(
        d_solid_surf.size
    )
    return intf, intb


def from_array(tm, basis):
    """S-matrix from an array of (cylindrical) T-matrices.

    Create a S-matrix for a two-dimensional array of objects described by the
    T-Matrix or an one-dimensional array of objects described by a cylindrical
    T-matrix.

    Args:
        tm (TMatrix or TMatrixC): (Cylindrical) T-matrix to place in the array.
        basis (PlaneWaveBasisByComp): Basis definition.
        eta (float or complex, optional): Splitting parameter in the lattice sum.

    Returns:
        SMatrix
    """
    if isinstance(tm, treams.TMatrixC):
        basis = basis.permute(-1)
    pu, pd = (tm.expandlattice(basis=basis, modetype=i) for i in ("up", "down"))
    au, ad = (tm.expand.eval_inv(basis, i) for i in ("up", "down"))
    eye = anp.eye(len(basis))
    res = treams.SMatrices([[eye + pu @ au, pu @ ad], [pd @ au, eye + pd @ ad]])

    res_mult = []
    for l in range(1, tm.basis.l.max() + 1):
        for em in anp.unique(tm.basis.pol):
            m_sca = tm.basis.m
            l_sca = tm.basis.l
            pol_sca = tm.basis.pol
            arg1 = l == l_sca
            arg2 = pol_sca == em
            mask = arg1 & arg2
            pui = pu[:, mask]
            pdi = pd[:, mask]
            aui = au[mask, :]
            adi = ad[mask, :]
            resi = treams.SMatrices([[pui @ aui, pui @ adi], [pdi @ aui, pdi @ adi]])

            res_mult.append(resi)

    if isinstance(tm, treams.TMatrixC):
        res = res.permute()
    return res, res_mult


def dreams_rcd(params, eps_object, k0, cfg):
    pitch = cfg["pitch"]
    eps_medium = cfg["eps_medium"]
    lmax = cfg["lmax"]
    lmax_glob = cfg["lmax_glob"]
    # k0          = cfg["k0"]
    rmax_coef = cfg["rmax_coef"]
    helicity = cfg["helicity"]
    kx, ky = cfg["kx"], cfg["ky"]
    poltype = "helicity" if helicity else "parity"
    treams.config.POLTYPE = poltype

    kpars = anp.array([kx, ky], dtype=jnp.float64)
    nobj = params.size // 4  # positions are 3N and radii N
    radii = jnp.asarray(params[3 * nobj :], dtype=jnp.float64)
    pos = jnp.asarray(params[: 3 * nobj], dtype=jnp.float64).reshape((-1, 3))
    epsilons = jnp.stack(
        [jnp.full((nobj,), eps_object), jnp.full((nobj,), eps_medium)], axis=1
    )
    S, modes = smat_spheres(
        radii,
        epsilons,
        eps_medium,
        lmax,
        k0,
        pos,
        helicity,
        kx,
        ky,
        pitch,
        rmax_coef=rmax_coef,
        lmax_glob=lmax_glob,
    )
    a = anp.array([[pitch, 0.0], [0.0, pitch]], dtype=jnp.float64)
    b = treams.lattice.reciprocal(anp.array(a))
    kvec = (
        anp.array(kpars)
        + treams.lattice.diffr_orders_circle(b, rmax=float(rmax_coef * k0)) @ b
    )
    pwb = default_plane_wave(jnp.asarray(kvec, jnp.float64))

    # x-illum
    inc_x, _ = plane_wave(
        kpars, [1, 0, 0], k0=k0, basis=pwb, epsilon=eps_medium, poltype=poltype
    )
    out_x = tr(S, k0, pitch, helicity, inc_x, pwb, epsilon=eps_medium, direction=-1)

    # y-illum
    inc_y, _ = plane_wave(
        kpars, [0, 1, 0], k0=k0, basis=pwb, epsilon=eps_medium, poltype=poltype
    )
    out_y = tr(S, k0, pitch, helicity, inc_y, pwb, epsilon=eps_medium, direction=-1)

    r_x = out_x[1]
    r_y = out_y[1]
    return jnp.abs(r_x - r_y)


def treams_rcd(params, cfg):
    pitch = cfg["pitch"]
    eps_medium = cfg["eps_medium"]
    eps_object = cfg["eps_object"]
    lmax_glob = cfg["lmax_glob"]
    try:
        lmax = cfg["lmax"]
    except:
        lmax = lmax_glob
        
    k0 = cfg["k0"]
    rmax_coef = cfg["rmax_coef"]
    helicity = cfg["helicity"]
    kx, ky = cfg["kx"], cfg["ky"]

    nobj = params.size // 4
    radii = anp.asarray(params[3 * nobj :], float)
    pos = anp.asarray(params[: 3 * nobj], float).reshape((-1, 3))

    lattice = treams.Lattice.square(pitch)
    treams.config.POLTYPE = "helicity" if helicity else "parity"

    mats = [
        treams.Material(anp.array([eps_object])),
        treams.Material(anp.array([eps_medium])),
    ]
    spheres = [treams.TMatrix.sphere(lmax, k0, r, mats) for r in radii]
    tm_cluster = treams.TMatrix.cluster(spheres, pos).interaction.solve()
    tm_global = tm_cluster.expand(treams.SphericalWaveBasis.default(lmax_glob))
    tm_lat = tm_global.latticeinteraction.solve(lattice, [kx, ky])
    pwb = treams.PlaneWaveBasisByComp.diffr_orders([kx, ky], lattice, rmax_coef * k0)
    array, list_array = from_array(tm_lat, pwb)
    # Rx (x-illum)
    pol = 1  # [1,0,0] different phase factor
    plw_x = treams.plane_wave([kx, ky], pol, k0=k0, basis=pwb, material=mats[1])
    tr_x = array.tr(plw_x)[1]
    rxs = []
    for ar in list_array:
        rix = ar[1, 0] @ plw_x
        rxs.append(rix)
    # Ry (y-illum)
    pol = 0  # [0,1,0]
    plw_y = treams.plane_wave([kx, ky], pol, k0=k0, basis=pwb, material=mats[1])
    tr_y = array.tr(plw_y)[1]
    rys = []
    for ar in list_array:
        riy = ar[1, 0] @ plw_y
        rys.append(riy)
    ans = anp.abs(tr_y - tr_x)
    return ans, tr_x, tr_y, rxs, rys


def rcd_k0(i, cfg, params):
    k0s = cfg["k0s"]
    eps_objects = cfg["eps_objects"]
    cfg_i = dict(cfg)
    cfg_i["k0"] = float(k0s[i])
    cfg_i["eps_object"] = eps_objects[i]
    return treams_rcd(params, cfg_i)


def treams_rcd_parallel(params, cfg):
    k0s = cfg["k0s"]
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(rcd_k0)(i, cfg=cfg, params=params) for i in range(len(k0s))
    )
    ans_list, rx_list, ry_list, ph_x_list, ph_y_list = zip(*results)
    ans = anp.stack(ans_list, axis=0)
    rx = anp.stack(rx_list, axis=0)
    ry = anp.stack(ry_list, axis=0)
    ph_x = anp.stack(ph_x_list, axis=0)
    ph_y = anp.stack(ph_y_list, axis=0)
    return ans, rx, ry, ph_x, ph_y
