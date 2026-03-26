import numpy as anp
import jax
from jax import lax
import jax.numpy as np
from functools import partial
from dreams.jax_coeffs import mie, mie_one_l
from dreams.jax_coord import car2sph
from dreams.jax_op import sw_translate, sw_expand
from dreams.jax_misc import defaultmodes, refractive_index, defaultlmax, basischange
from jax import config, jit
from jax.scipy.linalg import block_diag

config.update("jax_enable_x64", True)
import treams.special as sp


def _offset(l):
    return 2 * (l - 1) * (l + 1)


def _place_Tl(T, Tl, l, lmax):
    """
    Place 2x2 block Tl along the diagonal, repeated (2l+1) times,
    starting at offset(l). Works under jit/scan (no dynamic arange).
    """
    start = _offset(l)
    reps = 2 * l + 1
    max_reps = 2 * lmax + 1

    def body(T, m):
        row = start + 2 * m

        def do_update(T):
            return lax.dynamic_update_slice(T, Tl, (row, row))

        # Only update for the first 'reps' tiles
        T = lax.cond(m < reps, do_update, lambda T: T, T)
        return T, None

    T, _ = lax.scan(body, T, np.arange(max_reps))  # max_reps is concrete
    return T


def core_shell_sphere(
    lmax: int, k0: float, radii, epsilon, mu=None, kappa=None, poltype="helicity"
):
    """T-matrix of a multilayered sphere.

    Construct the T-matrix of a homogeneous sphere or a concentric multilayered
    sphere of the given multipole order. The sphere may consist of multiple
    spherical shells with arbitrary material parameters.

    The radii are specified from the inside to the outside. The material
    parameters are likewise ordered from the innermost region to the embedding
    medium, so that the last entry corresponds to the surrounding medium.

    Args:
        lmax (int): Maximum multipole order of the T-matrix.
        k0 (float): Vacuum wavenumber.
        radii (float or array-like): Radii from inside to outside. For a simple
            sphere, this can be a single radius. For a multilayered sphere, this
            should contain the radii of all concentric shells in increasing order.
        epsilon (array-like): Relative permittivities of all layers and the
            embedding medium. Its length must be ``len(radii) + 1``.
        mu (array-like, optional): Relative permeabilities of all layers and the
            embedding medium. If None, all values are set to 1.
        kappa (array-like, optional): Chirality parameters of all layers and the
            embedding medium. If None, all values are set to 0.
        poltype (str, optional): Polarization basis of the returned T-matrix.
            Supported values are ``"helicity"`` and ``"parity"``. Defaults to
            ``"helicity"``.

    Returns:
        ndarray: T-matrix of shape ``(N, N)``, where
        ``N = 2 * lmax * (lmax + 2)``.
    """
    if mu is None:
        mu = np.ones_like(epsilon)
    if kappa is None:
        kappa = np.zeros_like(epsilon)

    radii = np.atleast_1d(np.asarray(radii))
    if radii.shape[0] != len(epsilon) - 1:
        raise ValueError("incompatible lengths of radii and materials")

    dim = 2 * lmax * (lmax + 2)
    T = np.zeros((dim, dim), dtype=np.complex128)

    def outer(T, l):
        Tl = mie_one_l(l, k0, radii, epsilon, mu, kappa)  # (2,2)
        Tl = Tl[::-1, ::-1]
        T = _place_Tl(T, Tl, l, lmax)
        return T, None

    T, _ = lax.scan(outer, T, np.arange(1, lmax + 1))
    if poltype == "helicity":
        return T
    modes = defaultmodes(lmax, 1)
    mat = basischange(modes)
    T = mat.T @ T @ mat
    return T


@partial(jit, static_argnums=(0, 4))
def sphere(lmax, k0, rad, epsilon, helicity):
    """T-Matrix of a sphere.

    Construct the T-matrix of the given order and material for a sphere, in a given polarization basis.

    Args:
        lmax (int): Positive integer for the maximum degree of the T-matrix.
        k0 (float): Wave number in vacuum.
        rad (float or array): Radius  of the sphere.
        epsilon (list[Material]): The permittivities: the first material  in the list specifies the sphere, the last material specifies the embedding medium.
        helicity (bool): Flag for helicity basis.

    Returns:
        TMatrix
    """
    tmat = sphere_parity(lmax, k0, rad, epsilon)
    if helicity:
        modes = defaultmodes(lmax, 1)
        mat = basischange(modes)
        tmat = mat.T @ tmat @ mat
    return tmat


# @partial(jit, static_argnums = (3, 5))
def elchi(positions, radii, epsilon, lmax=3, k=2 * np.pi):
    """
    Compute a chirality metric from the global T-matrix.

    The global T-matrix is built in the helicity basis (helicity=True) and
    split into + and - helicity sub-blocks. The function compares the singular
    values of these blocks and returns a normalized difference.

    Args:
        positions: Array of shape (num, 3), particle positions.
        radii: Array of shape (num,), particle radii.
        epsilon: Array of shape (num, 2), [eps_sphere_i, eps_env].
        lmax: Maximum multipole order used for the local T-matrices.
        k: Vacuum wavenumber.

    Returns:
        float: Chirality metric (0 ~ achiral).
    """
    tglobal, modes2, pos = global_tmat(
        positions, radii, epsilon, lmax, k, helicity=True
    )
    modes = defaultmodes(lmax, len(radii))
    pidx, l, m, pol = modes
    zeroth = pidx == 0
    modes2 = l[zeroth], m[zeroth], pol[zeroth]
    selections = modes2[2] == 0, modes2[2] == 1
    spp = np.linalg.svd(
        tglobal[anp.ix_(selections[1], selections[1])], compute_uv=False
    )

    spm = np.linalg.svd(
        tglobal[anp.ix_(selections[1], selections[0])], compute_uv=False
    )
    smp = np.linalg.svd(
        tglobal[anp.ix_(selections[0], selections[1])], compute_uv=False
    )
    smm = np.linalg.svd(
        tglobal[anp.ix_(selections[0], selections[0])], compute_uv=False
    )
    plus = np.concatenate((spp, spm))
    minus = np.concatenate((smm, smp))
    ans = np.linalg.norm(plus - minus) / np.sqrt(np.sum(np.power(np.abs(tglobal), 2)))
    return ans


def xs_1(
    lind, tmat, illu, k0, epsilon, flux=0.5, num=1, positions=np.zeros(3), helicity=True
):
    r"""
    Partial scattering cross section for selected multipole orders and polarizations.

    This is a variant of `xs` that resolves the scattering cross section into
    electric and magnetic contributions for a chosen set of multipole orders
    `lind`. The T-matrix is assumed to be given in a single combined basis
    (possibly global) for `num` particles.

    The calculation is based on the scattered coefficients

        p = T @ a

    where `a` are the illumination coefficients. The scattered power is
    expressed mode-wise and then summed only over the selected multipole
    orders and polarizations.

    Args:
        lind (array-like): Array of multipole orders `l` to include in the sum.
        tmat (ndarray): T-matrix, shape (N, N).
        illu (array-like): Illumination coefficients a_{slm}, shape (N,).
        k0 (float): Vacuum wavenumber.
        epsilon (array-like): Permittivity information used to compute
            refractive indices via `refractive_index(epsilon)`.
        flux (float, optional): Incident flux used for normalization.
            A plane wave in these units has flux = 0.5 (default).
        num (int, optional): Number of particles represented in `tmat`.
        positions (array-like, optional): Particle positions, shape (num, 3),
            used to construct the singular spherical-wave expansion operator.
            Default is a single particle at the origin.
        helicity (bool, optional): Helicity basis flag used in `sw_expand`.

    Returns:
        tuple[float, float]:
            xs_e1: Scattering cross section summed over electric-type modes
                   (pol == 1) with l in `lind`.
            xs_m1: Scattering cross section summed over magnetic-type modes
                   (pol == 0) with l in `lind`.
    """
    r"""Scattering and extinction cross section.
    Returns sum of electric and sum of magnetic multipole components of the scattering cross section for a given lmax

    """
    lmax = defaultlmax(len(tmat), num)
    modes = defaultmodes(lmax, num)
    l, m, pol = modes[1:]
    p = tmat @ np.array(illu)
    ks = k0 * refractive_index(epsilon)
    p_invksq = p * np.power(ks[pol], -2)
    swe = sw_expand(positions, modes, k0, helicity, epsilon, modetype="singular")
    p_invksq = swe @ p_invksq
    e1 = (pol == 1) & np.isin(l, lind)
    m1 = (pol == 0) & np.isin(l, lind)
    tot = 0.5 * np.real(p.conjugate().T * p_invksq) / flux
    xs_e1 = np.sum(tot[e1])
    xs_m1 = np.sum(tot[m1])
    return xs_e1, xs_m1


def xs(tmat, illu, k0, epsilon, flux=0.5, num=1, positions=np.zeros(3), helicity=True):
    r"""Scattering and extinction cross section.

    Possible for all T-matrices (global and local) in non-absorbing embedding. The
    values are calculated by

    .. math::

        \sigma_\mathrm{sca}
        = \frac{1}{2 I}
        a_{slm}^\ast T_{s'l'm',slm}^\ast k_{s'}^{-2} C_{s'l'm',s''l''m''}^{(1)}
        T_{s''l''m'',s'''l'''m'''} a_{s'''l'''m'''} \\
        \sigma_\mathrm{ext}
        = \frac{1}{2 I}
        a_{slm}^\ast k_s^{-2} T_{slm,s'l'm'} a_{s'l'm'}

    where :math:`a_{slm}` are the expansion coefficients of the illumination,
    :math:`T` is the T-matrix, :math:`C^{(1)}` is the (regular) translation
    matrix and :math:`k_s` are the wave numbers in the medium. All repeated indices
    are summed over. The incoming flux is :math:`I`.

    Args:
        tmat (ndarray): T-matrix, shape (N, N).
        illu (array-like): Illumination coefficients a_{slm}, shape (N,).
        k0 (float): Vacuum wavenumber.
        epsilon (array-like): Permittivity information used to compute
            refractive indices via `refractive_index(epsilon)`.
        flux (float, optional): Incident flux used for normalization.
            A plane wave in these units has flux = 0.5 (default).
        num (int, optional): Number of particles .
        positions (array-like, optional): Particle positions, shape (num, 3),
            used to build the singular spherical-wave expansion operator for the
            scattered field. Default is a single particle at the origin.
        helicity (bool, optional): Helicity basis flag used in `sw_expand`.
        
    tuple[float, float]:
        sigma_sca: Total scattering cross section.
        sigma_ext: Total extinction cross section.
    """
    lmax = defaultlmax(len(tmat), num)
    modes = defaultmodes(lmax, num)
    l, m, pol = modes[1:]
    p = tmat @ np.array(illu)
    ks = k0 * refractive_index(epsilon)
    p_invksq = p * np.power(ks[pol], -2)
    swe = sw_expand(positions, modes, k0, helicity, epsilon, modetype="singular")
    p_invksq = swe @ p_invksq
    return (
        0.5 * np.real(p.conjugate().T @ p_invksq) / flux,
        -0.5 * np.real(illu.conjugate().T @ p_invksq) / flux,
    )


def xs_ext_avg(tm, ks):
    r"""Rotation and polarization averaged extinction cross section.

    The average is calculated as

    .. math::

        \langle \sigma_\mathrm{ext} \rangle
        = -2 \pi \sum_{slm} \frac{\Re(T_{slm,slm})}{k_s^2}

    where :math:`k_s` is the wave number in the embedding medium for the
    polarization :math:`s`. It is only implemented for global T-matrices.

    Returns:
        float or complex
    """
    res = -2 * np.pi * tm.trace().real / (ks * ks)

    return res


def xs_sca_avg(tm, ks):
    r"""Rotation and polarization averaged scattering cross section.

    The average is calculated as

    .. math::

        \langle \sigma_\mathrm{sca} \rangle
        = 2 \pi \sum_{slm} \sum_{s'l'm'}
        \frac{|T_{slm,s'l'm'}|^2}{k_s^2}

    where :math:`k_s` is the wave number in the embedding medium for the
    polarization :math:`s`. It is only implemented for global T-matrices.

    Returns:
        float or complex
    """
    re, im = tm.real, tm.imag
    res = 2 * np.pi * np.sum((re * re + im * im) / (ks * ks))
    return res.real


def sphere_parity(lmax, k, rad, epsilon, mu=None):
    """
    T-matrix (parity basis) of a single homogeneous sphere.

    Args:
        lmax (int): Maximum multipole order.
        k (float): Vacuum wavenumber k0.
        rad (float): Radius of the sphere.
        epsilon (array-like): Length-2 array/list of permittivities
            [eps_sphere, eps_env].
        mu (array-like, optional): Permeabilities corresponding to epsilon.
            If None, mu = 1 for all media.

    Returns:
        ndarray: Diagonal T-matrix of shape (N, N) in parity basis, where
            N = 2 * lmax * (lmax + 2).
    """
    epsilon = np.array(epsilon)
    if mu is None:
        mu = np.ones_like(epsilon)
    l = anp.arange(1, lmax + 1)
    n = 2 * l + 1
    lm = anp.repeat(l, n)
    radii = np.ones_like(lm) * rad
    cnn = mie(lm, mu, epsilon, radii, k)
    return np.diag(cnn.flatten()) * (-1)


# @partial(jit, static_argnums=(2, 4))
def tmats_no_int(radii, epsilon, lmax, k0, helicity):
    """
    Compute local T-matrices for multiple non-interacting spheres.

    Each sphere i is described by its radius radii[i] and by a pair of
    permittivities epsilon[i] = [eps_sphere_i, eps_env]. The embedding
    permittivity is assumed to be the same for all spheres and taken from
    the second entry (index 1).

    Args:
        radii (array-like): Radii of the spheres, shape (num,).
        epsilon (array-like): Permittivities, shape (num, 2). For each i:
            epsilon[i, 0] = permittivity of sphere i,
            epsilon[i, 1] = permittivity of the embedding medium.
        lmax (int): Maximum multipole order.
        k0 (float): Vacuum wavenumber.
        helicity (bool): If True, transform the result to helicity basis.

    Returns:
        ndarray: Block-diagonal local T-matrix of shape (num*N, num*N),
            where N = 2 * lmax * (lmax + 2) is the size of the single-sphere
            T-matrix in parity basis.
    """
    num = radii.shape[0]
    sphere_ = sphere_parity(lmax, k0, radii[0], epsilon[0])
    shape = sphere_.shape[0]

    def create_sphere(rad, eps):
        return sphere_parity(lmax, k0, rad, eps)

    matrices = jax.vmap(create_sphere, in_axes=(0, 0))(radii, epsilon)
    tlocal = block_diag(*matrices)
    if helicity:
        modes = defaultmodes(lmax, num)
        mat = basischange(modes)
        tlocal = mat.T @ tlocal @ mat
    return tlocal


#   @partial(jit, static_argnums=(2, 4))
def tmats_interact(ts, positions, modes, k0, helicity, epsilon, mu=1, kappa=0):
    """
    Dress local T-matrices with multiple-scattering interactions.

    Args:
        ts (ndarray): Local T-matrix of all particles combined (block diagonal),
            shape (M, M).
        positions (array): Particle positions in Cartesian coordinates,
            shape (num, 3).
        modes (tuple): Mode indices as returned by defaultmodes(lmax, num).
        k0 (float): Vacuum wavenumber.
        helicity (bool): Helicity basis flag.
        epsilon (array-like): Medium permittivity information.
        mu, kappa: Magnetic and chiral parameters for the medium.

    Returns:
        ndarray: Full T-matrix including multiple scattering, shape (M, M).
    """
    translation = sw_expand(
        positions,
        modes,
        k0,
        helicity,
        epsilon,
        mu,
        kappa,
        modetype="regular",
        to_modetype="singular",
    )
    finalt = np.linalg.solve(
        np.eye(ts.shape[0]) - ts @ np.reshape(translation, ts.shape), ts
    )
    return finalt


#   @partial(jit, static_argnums=(2, 4, 5))
def globfromloc(
    tlocal, positions, lmax, k0, num, helicity, epsilon, lmax_glob=None, mu=1, kappa=0
):
    """
    Convert local, interacting T-matrix to a global T-matrix about the origin.

    This function:
      1. Includes interactions between particles via tmats_interact.
      2. Translates the T-matrix from particle-centered coordinates to a
         common origin using spherical-wave translation operators.

    Args:
        tlocal (ndarray): Local block-diagonal T-matrix WITHOUT interactions.
        positions (array): Particle positions, shape (num, 3).
        lmax (int): Local maximum multipole order.
        k0 (float): Vacuum wavenumber.
        num (int): Number of particles.
        helicity (bool): Helicity basis flag.
        epsilon (array-like): Medium permittivity container. Only
            epsilon[0, 1] (a scalar) is used as the embedding permittivity.
        lmax_glob (int, optional): Maximum multipole order for the global
            T-matrix. If None, lmax_glob = lmax.
        mu, kappa: Medium parameters.

    Returns:
        tuple:
            global_t (ndarray): Global T-matrix about the origin.
            modes2 (tuple): Mode indices for the global basis (l, m, pol).
            positions (array): The origin, shape (1, 3).
    """
    if lmax_glob is None:
        lmax_glob = lmax
    modes = defaultmodes(lmax, num)
    pidx, l, m, pol = modes
    positions = positions.reshape((-1, 3))
    finalt = tmats_interact(tlocal, positions, modes, k0, helicity, epsilon, mu, kappa)
    ks = k0 * refractive_index(epsilon, mu, kappa)
    origin = np.zeros((3,))
    origin = np.reshape(origin, (1, 3))
    modes2 = defaultmodes(lmax_glob, 1)[1:]
    rs = car2sph(positions - origin)
    ain = sw_translate(
        *(m[:, None] for m in modes[1:]),
        *modes2,
        ks[pol[:, None]] * rs[pidx, :1],
        rs[pidx, 1:2],
        rs[pidx, 2:],
        helicity=helicity,
        singular=False,
    )
    pout = sw_translate(
        *(m[:, None] for m in modes2),
        *modes[1:],
        ks[pol] * rs[pidx, 0],
        np.pi - rs[pidx, 1],
        rs[pidx, 2] + np.pi,
        helicity=helicity,
        singular=False,
    )
    global_t = pout @ finalt @ ain
    positions = origin
    return global_t, modes2, positions


def global_tmat(positions, radii, epsilon, lmax, k0, helicity, lmax_glob=None):
    """
    Build the global T-matrix for multiple spheres.

    This is a convenience wrapper that:
      1. Constructs local single-particle T-matrices (no interaction).
      2. Adds multiple-scattering interactions.
      3. Translates everything to a common origin.

    Args:
        positions (array): Particle positions, shape (num, 3).
        radii (array-like): Radii of the spheres, shape (num,).
        epsilon (array-like): Permittivities, shape (num, 2). For each i:
            epsilon[i, 0] = permittivity of sphere i,
            epsilon[i, 1] = permittivity of the embedding medium (same for all).
        lmax (int): Local maximum multipole order.
        k0 (float): Vacuum wavenumber.
        helicity (bool): If True, result is in helicity basis.
        lmax_glob (int, optional): Global maximum multipole order. If None,
            lmax_glob = lmax.

    Returns:
        tuple:
            global_t (ndarray): Global T-matrix about the origin.
            modes2 (tuple): Global modes (l, m, pol) for a single center.
            positions (array): The origin, shape (1, 3).
    """
    if lmax_glob is None:
        lmax_glob = lmax
    num = radii.shape[0]
    tlocal = tmats_no_int(radii, epsilon, lmax, k0, helicity)
    global_t, modes2, positions = globfromloc(
        tlocal, positions, lmax, k0, num, helicity, epsilon[0, 1], lmax_glob=lmax_glob
    )
    return global_t, modes2, positions
