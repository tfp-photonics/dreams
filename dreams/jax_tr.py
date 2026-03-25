from functools import partial
import itertools
import numpy as anp
import jax.numpy as np
from jax import jit
import treams.lattice as la
from dreams.jax_smat import (
    arrayt,
    interface,
    propagation,
    sdefaultmodes,
    stack,
)
from dreams.jax_misc import wave_vec_z, defaultmodes, refractive_index, basischange
from dreams.jax_tmat import global_tmat, tmats_no_int
from dreams.jax_misc import diffr_orders_circle


def illuminate(sm, illu, illu2=None, /, *, smat=None):
    """Field coefficients above and below the S-matrix.

    Given an illumination defined by the coefficients of each incoming mode
    calculate the coefficients for the outgoing field above and below the S-matrix.
    If a second SMatrix is given, the field expansions in between are also
    calculated.

    Args:
        sm (array-like): 2×2 block S-matrix. Each block is an (N, N) array
            acting on incoming/outgoing mode coefficients.
        illu (array-like): Illumination, if `modetype` is specified, the direction
            will be chosen accordingly.
        illu2 (array-like, optional): Second illumination. If used, the first
            argument is taken to be coming from below and this one to be coming from
            above.
        smat (SMatrix, optional): Second S-matrix for the calculation of the
            field expansion between two S-matrices.

    Returns:
        tuple: If `smat is None`, returns (field_up, field_down).
        Otherwise returns (field_up, field_down, field_in_up, field_in_down).
    """
    modetype = getattr(illu, "modetype", "up")
    if isinstance(modetype, tuple):
        modetype = modetype[max(-2, -len(modetype))]

    illu2 = np.zeros(np.shape(illu)[-2:], dtype=illu.dtype) if illu2 is None else illu2

    if modetype == "down":
        illu, illu2 = illu2, illu

    if smat is None:
        field_up = sm[0, 0] @ illu + sm[0, 1] @ illu2
        field_down = sm[1, 0] @ illu + sm[1, 1] @ illu2
        return field_up, field_down

    n = sm[0, 0].shape[0]
    stmp = np.eye(n, dtype=sm[0, 0].dtype) - sm[0, 1] @ smat[1, 0]
    field_in_up = np.linalg.solve(stmp, sm[0, 0] @ illu + sm[0, 1] @ smat[1, 1] @ illu2)
    field_in_down = smat[1, 0] @ field_in_up + smat[1, 1] @ illu2
    field_up = smat[0, 1] @ illu2 + smat[0, 0] @ field_in_up
    field_down = sm[1, 0] @ illu + sm[1, 1] @ field_in_down
    return field_up, field_down, field_in_up, field_in_down


def poynting_avg_z(modes, k0, epsilon, mu, poltype, modetype):
    r"""Time-averaged z-component of the Poynting vector.

    Calculate the time-averaged Poynting vector's z-component

    .. math::

        \langle S_z \rangle
        = \frac{1}{2}
        \Re (\boldsymbol E \times \boldsymbol H^\ast) \boldsymbol{\hat{z}}

    on one side of the S-matrix with the given coefficients.

    Args:
        modes (array_like, shape (N, 3)):
            Per-mode data ``[kx, ky, pol]``, where ``pol`` is 0 or 1.
        k0 (float):
            Vacuum wave number.
        epsilon (float or array_like):
            Relative permittivity of the medium.
        mu (float or array_like):
            Relative permeability of the medium.
        poltype ({"parity", "helicity"}):
            Polarization basis of the modes.
        modetype ({"up", "down"}):
            Direction of propagation.

    Returns:
        (a, b):
            Two arrays of shape ``(N, N)`` to be used as quadratic forms
            in the Poynting-flux expressions.
    """
    kx = modes[:, 0]
    ky = modes[:, 1]
    pol = modes[:, 2].astype(int)
    ks = np.array(k0 * refractive_index(epsilon, mu))

    kzs = wave_vec_z(kx, ky, ks[pol], nondifsum=False) * (2 * (modetype == "up") - 1)

    impedance = np.sqrt(mu / epsilon)
    gamma = kzs / (ks[pol] * impedance)
    selection = (kx[:, None] == kx) & (ky[:, None] == ky)
    if poltype == "parity":
        selection = selection & (pol[:, None] == pol)
        a = selection * ((1 - pol) * gamma.conjugate() + pol * gamma) * 0.25
        b = selection * ((1 - pol) * gamma.conjugate() - pol * gamma) * 0.25
        return a, b
    if poltype == "helicity":
        hpol = 2 * pol - 1  # length N, values -1,+1
        hh = hpol[:, None] * hpol[None, :]  # (N,N)
        gamma_conj_col = gamma.conj()[:, None]  # (N, 1), uses row index i
        gamma_row = gamma[None, :]  # (1, N), uses column index j
        a = selection * (hh * gamma_conj_col + gamma_row) * 0.25
        b = selection * (hh * gamma_conj_col - gamma_row) * 0.25
        return a, b
    raise ValueError(f"invalid poltype: '{poltype}'")


# @partial(jit, static_argnums=(1, 2, 3, 6, 7, 8))
def tr(
    q, k0, pitch, helicity, illu, basis, epsilon=1, mu=1, direction=1, modetype="up"
):
    """Transmittance and reflectance for one S-matrix.

    Args:
        q: S-matrix of shape (2, 2, N, N).
        k0: Vacuum wavenumber.
        pitch: Lattice constant (square array).
        helicity: True for helicity basis, False for parity.
        illu: Incoming plane-wave coefficients.
        basis: Mode list (kx, ky, pol).
        epsilon: Medium permittivity (scalar or pair).
        mu: Relative permeability.
        direction: +1 or -1 for incidence direction.
        modetype: "up" or "down".

    Returns:
        Array [T, R] with transmittance and reflectance.
    """
    if direction not in (-1, 1):
        raise ValueError(f"direction must be '-1' or '1', but is '{direction}''")
    trans, refl = illuminate(q, illu)
    if not isinstance(epsilon, tuple):
        epsilon = epsilon, epsilon
    if helicity:
        poltype = "helicity"
    else:
        poltype = "parity"
    paz = [poynting_avg_z(basis, k0, m, mu, poltype, modetype) for m in epsilon]
    # paz = [poynting_avg_z(anp.asarray(basis), k0, m, mu, poltype, modetype) for m in epsilon]
    s_t = np.real(trans.conjugate().T @ paz[0][0] @ trans)
    s_r = np.real(refl.conjugate().T @ paz[1][0] @ refl)
    s_i = np.real(np.conjugate(illu).T @ paz[1][0] @ illu)
    s_ir = np.real(
        refl.conjugate().T @ paz[1][1] @ illu - np.conjugate(illu).T @ paz[1][1] @ refl
    )
    return np.array([s_t / (s_i + s_ir), s_r / (s_i + s_ir)])


def field_outside(q, modes, illu):
    """
    Field coefficients above and below the Q-matrix

    Given an illumination defined by the coefficients of each incoming mode
    calculate the coefficients for the outgoing field above and below the Q-matrix.

    Args:
        illu (tuple): A 2-tuple of arrays, with the entries corresponding to
            upwards and downwards incoming modes.

    Returns:
        tuple
    """
    kx, ky, pol = modes
    illu = [np.zeros_like(kx) if i is None else i for i in illu]
    field_above = q[0, 0, :, :] @ illu[0] + q[0, 1] @ illu[1]
    field_below = q[1, 0, :, :] @ illu[0] + q[1, 1] @ illu[1]
    return field_above, field_below


def poynting_avg(coeffs, modes, ks, helicity, epsilon=1, mu=None, above=True):
    r"""
    Time-averaged z-component of the Poynting vector

    Calculate the time-averaged Poynting vector's z-component

    .. math::

        \langle S_z \rangle = \frac{1}{2} \Re (\boldsymbol E \times \boldsymbol H^\ast) \boldsymbol{\hat{z}}

    on one side of the Q-matrix with the given coefficients.

    Args:
        coeffs (2-tuple): The first entry are the upwards propagating modes the
            second one the downwards propagating modes
        above (bool, optional): Calculate the Poynting vector above or below the
            Q-matrix

    Returns:
        float
    """
    if (not isinstance(epsilon, anp.ndarray)) & (not isinstance(epsilon, list)):
        epsilon = anp.array([epsilon, epsilon], complex)
    if mu is None:
        mu = anp.ones_like(epsilon)
    choice = int(bool(above))
    kx, ky, pol = modes
    kz = wave_vec_z(kx, ky, ks[:, pol])
    selections = pol == 0, pol == 1
    pref = (
        kz[choice, selections[0]] / ks[choice, 0],
        kz[choice, selections[1]] / ks[choice, 1],
    )

    coeffs = [np.zeros_like(kx) if c is None else np.array(c) for c in coeffs]
    allcoeffs = [
        (1, -1, coeffs[0][selections[0]]),
        (1, 1, coeffs[0][selections[1]]),
        (-1, -1, coeffs[1][selections[0]]),
        (-1, 1, coeffs[1][selections[1]]),
    ]
    res = 0
    if helicity:
        for (dira, pola, a), (dirb, polb, b) in itertools.product(allcoeffs, repeat=2):
            res = res + a @ (
                np.conjugate(b)
                * (
                    pola * polb * np.conjugate(pref[(polb + 1) // 2]) * dirb
                    + pref[(pola + 1) // 2] * dira
                )
            )
        res = res * 0.25
    else:
        for (dira, _, a), (dirb, _, b) in itertools.product(allcoeffs[::2], repeat=2):
            res = res + a @ (np.conjugate(b) * np.conjugate(pref[0]) * dirb)
        for (dira, _, a), (dirb, _, b) in itertools.product(allcoeffs[1::2], repeat=2):
            res = res + a @ (np.conjugate(b) * pref[1] * dira)
        res = res * 0.5
    return np.real(res / anp.conjugate(anp.sqrt(mu[choice] / epsilon[choice])))


# @partial(jit, static_argnums=(2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17))
def smat_spheres(
    radii,
    epsilons,
    eps_medium,
    lmax,
    k0,
    positions,
    helicity,
    kx,
    ky,
    pitch,
    origin=None,
    eta=0.0,
    mu=1,
    kappa=0,
    rmax_coef=1,
    local=False,
    lmax_glob=None,
):
    """S-matrix for a periodic array of spheres.

    Args:
        radii (array_like, shape (num,)): Radii of the spheres.
        epsilons (array_like): Permittivities of the scatterers;
            shape and interpretation must match your T-matrix code.
        eps_medium (float or complex): Embedding permittivity.
        lmax (int): Maximum multipole order for the local T-matrix.
        k0 (float): Vacuum wavenumber.
        positions (array_like, shape (num, 3)): Sphere positions in the unit cell.
        helicity (bool): True for helicity basis, False for parity.
        kx (float or array_like): In-plane Bloch wavevector x-component.
        ky (float or array_like): In-plane Bloch wavevector y-component.
        pitch (float): Lattice constant (square lattice).
        origin (array_like, shape (3,), optional): Origin of the lattice coordinate
            system. If None, defaults to (0, 0, 0).
        eta (float or complex, optional): Ewald splitting parameter.
        mu (float or array_like, optional): Relative permeability.
        kappa (float or array_like, optional): Chirality parameter.
        rmax_coef (float, optional): Cutoff factor for diffraction-order radius.
        local (bool, optional): If True, use interacting local
            T-matrices.
        lmax_glob (int, optional): Global multipole cutoff for the array.
            If None, defaults to lmax.
    """
    if lmax_glob is None:
        lmax_glob = lmax
    if not local:
        tmat, _, _ = global_tmat(
            positions, radii, epsilons, lmax, k0, lmax_glob=lmax_glob, helicity=helicity
        )
        positions = anp.zeros((1, 3))
    else:
        tmat = tmats_no_int(radii, epsilons, lmax, k0, helicity)
        modes = defaultmodes(lmax, len(radii))
    smat, modes = smat_array(
        tmat,
        lmax_glob,
        helicity,
        kx,
        ky,
        k0,
        pitch,
        eps_medium,
        origin=origin,
        eta=eta,
        mu=mu,
        kappa=kappa,
        rmax_coef=rmax_coef,
        positions=positions,
        nondifsum=True,  # for now
    )
    return smat, modes


# @partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14))
def smat_array(
    tmat,
    lmax,
    helicity,
    kx,
    ky,
    k0,
    pitch,
    eps_medium,
    origin=None,
    eta=0.0,
    mu=1,
    kappa=0,
    rmax_coef=1,
    positions=None,
    nondifsum=True,
):
    a = anp.array([[pitch, 0], [0, pitch]])
    b = la.reciprocal(a)

    if nondifsum:
        kpar = anp.array([kx, ky])
        kpars = kpar + la.diffr_orders_circle(b, rmax=rmax_coef * k0) @ b
    else:
        kpar = np.array([kx, ky])
        kpars = kpar + diffr_orders_circle(b, rmax=rmax_coef * k0) @ b
    answer = arrayt(
        tmat,
        lmax,
        helicity,
        kpars,
        k0,
        a,
        epsilon_medium=eps_medium,
        origin=origin,
        eta=eta,
        mu=mu,
        kappa=kappa,
        positions=positions,
        nondifsum=nondifsum,
    )[0]
    modes = sdefaultmodes(kpars, nondifsum)
    return answer, modes


# @partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14))
def smat_spheres_full(
    radii,
    epsilons,
    eps_medium,
    lmax,
    k0,
    positions,
    helicity,
    kx,
    ky,
    pitch,
    eps_below=None,
    eps_above=None,
    origin=None,
    eta=0.0,
    mu=1,
    kappa=0,
    rmax_coef=1,
    local=False,
    lmax_glob=None,
):
    if lmax_glob == None:
        lmax_glob = lmax
    if eps_below is None:
        eps_below = eps_medium
    if eps_above is None:
        eps_above = eps_medium
    smat, modes = smat_spheres(
        radii,
        epsilons,
        eps_medium,
        lmax,
        k0,
        positions,
        helicity,
        kx,
        ky,
        pitch,
        origin=origin,
        eta=eta,
        mu=mu,
        kappa=kappa,
        rmax_coef=rmax_coef,
        local=local,
        lmax_glob=lmax_glob,
    )
    # return smat, modes
    sfull, modes = stacking(
        smat,
        modes,
        helicity,
        kx,
        ky,
        k0,
        pitch,
        eps_medium,
        eps_below=eps_below,
        eps_above=eps_above,
        mu=mu,
        kappa=kappa,
    )
    return sfull, modes


# @partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14))
def smat_full(
    tmat,
    lmax,
    helicity,
    kx,
    ky,
    k0,
    pitch,
    eps_medium,
    eps_below=1,
    eps_above=1,
    origin=None,
    eta=0.0,
    mu=1,
    kappa=0,
    rmax_coef=1,
    positions=None,
):
    smat, modes = smat_array(
        tmat,
        lmax,
        helicity,
        kx,
        ky,
        k0,
        pitch,
        eps_medium,
        origin=origin,
        eta=eta,
        mu=mu,
        kappa=kappa,
        rmax_coef=rmax_coef,
        positions=positions,
        nondifsum=True,
    )
    sfull, modes = stacking(
        smat,
        modes,
        helicity,
        kx,
        ky,
        k0,
        pitch,
        eps_medium,
        eps_below=eps_below,
        eps_above=eps_above,
        mu=mu,
        kappa=kappa,
    )
    return sfull, modes


def stacking(
    smat,
    modes,
    helicity,
    kx,
    ky,
    k0,
    pitch,
    eps_medium,
    eps_below=1,
    eps_above=1,
    mu=1,
    kappa=0,
):
    try:
        eps_medium = eps_medium.item()
    except:
        pass
    if not helicity:
        smat = changebasis(smat, modes)
    r = np.array([0, 0, pitch / 2])
    inter1 = interface(eps_above, eps_medium, k0, modes, mu=mu, kappa=kappa)
    prop = propagation(eps_medium, r, k0, modes, mu=mu, kappa=kappa)
    inter2 = interface(eps_medium, eps_below, k0, modes, mu=mu, kappa=kappa)
    stck = [inter1, prop, smat, prop, inter2]
    st = stack(stck)
    if not helicity:
        st = changebasis(st, modes)
    return st, modes


def changebasis(q, modes=None):
    """
    Swap between helicity and parity basis

    Args:
        modes (array, optional): Change the number of modes while changing the basis

    Returns:
        SMatrix
    """

    mat = basischange(modes)
    q00 = mat.T @ q[0, 0, :, :] @ mat
    q01 = mat.T @ q[0, 1, :, :] @ mat
    q10 = mat.T @ q[1, 0, :, :] @ mat
    q11 = mat.T @ q[1, 1, :, :] @ mat
    qc1 = np.stack((q00, q01), axis=0)
    qc2 = np.stack((q10, q11), axis=0)
    q = np.stack((qc1, qc2), axis=0)

    return q
