import numpy as anp

import jax.numpy as np
import treams._core as core
from dreams.jax_primitive import wignerd
from dreams.jax_coord import car2sph
from dreams.jax_misc import defaultmodes, refractive_index, defaultlmax
from dreams.jax_waves import tl_vsw_A, tl_vsw_B, tl_vsw_rA, tl_vsw_rB, tau_fun, pi_fun
from jax import config, debug

config.update("jax_enable_x64", True)
from jax.lax import lgamma as loggamma


def sw_translate(
    lambda_, mu, pol, l, m, qol, kr, theta, phi, helicity=False, singular=True
):
    """
    Translation coefficients between spherical vector waves.

    This function computes the translation matrix between spherical
    vector waves at two centers. It dispatches to TE/TM (parity) or
    helicity implementations and to regular or singular waves.

    Args:
        lambda_ (array_like): Degrees l' of the destination modes.
        mu (array_like): Orders m' of the destination modes.
        pol (array_like): Polarizations of the destination modes
            (0/1 for TE/TM or -/+ helicity).
        l (array_like): Degrees l of the source modes.
        m (array_like): Orders m of the source modes.
        qol (array_like): Polarizations of the source modes
            (0/1 for TE/TM or -/+ helicity).
        kr (array_like): Radial arguments k r for the translation.
        theta (array_like): Polar angles between source and destination
            centers (in radians).
        phi (array_like): Azimuthal angles between source and
            destination centers (in radians).
        helicity (bool, optional): If True, treat pol/qol as helicity
            indices. If False, treat them as parity (TE/TM). Defaults
            to False.
        singular (bool, optional): If True, use singular (outgoing)
            spherical waves at the destination. If False, use regular
            spherical waves. Defaults to True.

    Returns:
        jax.Array: Translation matrix of shape determined by the
        broadcast of the input mode indices. Each entry maps a source
        mode (l, m, qol) to a destination mode (lambda_, mu, pol).
    """
    kr = kr + 0j
    if helicity:
        if singular:
            return translate_sh(lambda_, mu, pol, l, m, qol, kr, theta, phi)
        return translate_rh(lambda_, mu, pol, l, m, qol, kr, theta, phi)
    if singular:
        return translate_sp(lambda_, mu, pol, l, m, qol, kr, theta, phi)
    return translate_rp(lambda_, mu, pol, l, m, qol, kr, theta, phi)


def translate_rp(lambda_, mu, pol, l, m, qol, kr, theta, phi):
    mask = (pol == qol) & ((pol == 0) | (pol == 1))
    answer = tl_vsw_rA(lambda_, mu, l, m, kr, theta, phi) * mask + tl_vsw_rB(
        lambda_, mu, l, m, kr, theta, phi
    ) * (mask == False)
    return answer


def translate_sp(lambda_, mu, pol, l, m, qol, kr, theta, phi):
    mask = (pol == qol) & ((pol == 0) | (pol == 1))
    answer = tl_vsw_A(lambda_, mu, l, m, kr, theta, phi) * mask + tl_vsw_B(
        lambda_, mu, l, m, kr, theta, phi
    ) * (mask == False)
    return answer


def translate_sh(lambda_, mu, pol, l, m, qol, kr, theta, phi):
    mask = (pol == qol) & ((pol == 0) | (pol == 1))
    answer = (
        tl_vsw_A(lambda_, mu, l, m, kr, theta, phi)
        + (2 * pol - 1) * tl_vsw_B(lambda_, mu, l, m, kr, theta, phi)
    ) * mask
    return answer


def translate_rh(lambda_, mu, pol, l, m, qol, kr, theta, phi):
    kr = kr + np.zeros_like(l + lambda_)
    theta = theta + np.zeros_like(l + lambda_)
    phi = phi + np.zeros_like(l + lambda_)
    mask = (pol == qol) & ((pol == 0) | (pol == 1))
    answer = (
        tl_vsw_rA(lambda_, mu, l, m, kr, theta, phi)
        + (2 * pol - 1) * tl_vsw_rB(lambda_, mu, l, m, kr, theta, phi)
    ) * mask

    return answer


def sw_expand(
    positions,
    modes,
    k0,
    helicity,
    epsilon,
    mu=1,
    kappa=0,
    modetype="singular",
    to_modetype=None,
):
    """
    Spherical-wave translation matrix for a set of scatterers.

    Builds the pairwise translation matrix between spherical vector
    waves centered at the given particle positions.

    Args:
        positions (array_like): Particle positions of shape (num, 3)
            in Cartesian coordinates.
        modes (tuple): Tuple (pidx, l, m, pol) as returned by
            defaultmodes(lmax, num). pidx is the particle index for
            each mode.
        k0 (float): Vacuum wavenumber.
        helicity (bool): If True, interpret pol as helicity indices.
            If False, use parity (TE/TM).
        epsilon (array_like or float): Relative permittivity of the
            embedding medium. Passed to refractive_index.
        mu (array_like or float, optional): Relative permeability of
            the embedding medium. Defaults to 1.
        kappa (array_like or float, optional): Chirality parameter of
            the embedding medium. Defaults to 0.
        modetype (str, optional): Mode type of the source waves
            ("regular" or "singular"). Defaults to "singular".
        to_modetype (str, optional): Mode type of the destination waves
            ("regular" or "singular"). If None, defaults to modetype.

    Returns:
        jax.Array: Translation matrix of shape (N, N), where N is the
        number of spherical modes in `modes`. The matrix maps
        coefficients at one set of centers to coefficients at another
        set of centers, using the chosen mode types.
    """
    if to_modetype is None:
        to_modetype = modetype
    positions = np.reshape(positions, (-1, 3))
    ind = positions[:, None, :] - positions
    rs = car2sph(ind)

    rs = np.array(rs)

    ks = k0 * refractive_index(epsilon, mu, kappa)
    pidx, l, m, pol = modes
    translation = sw_translate(
        *(m[:, None] for m in modes[1:]),
        *modes[1:],
        ks[pol] * rs[pidx[:, None], pidx, 0],
        rs[pidx[:, None], pidx, 1],
        rs[pidx[:, None], pidx, 2],
        helicity=helicity,
        singular=modetype != to_modetype,
    )
    return translation


def pw_translate(kx, ky, kz, x, y, z, where=True):
    r"""translate(kx, ky, kz, x, y, z)

    Translation coefficient for plane wave modes

    The translation coefficient is the phase factor
    :math:`\mathrm e^{\mathrm i \boldsymbol k \boldsymbol r}`.

    Args:
        kx, ky, kz (float or complex, array_like): Wave vector components
        x, y, z (float, array_like): Translation vector components

    Returns:
        complex
    """
    zterm = kz * z
    ans = np.where(where, np.exp(+1j * (kx * x + ky * y + zterm)), 0)
    return ans


def _sw_pw_expand(
    basis,
    pidx,
    l,
    m,
    pol,
    positions,
    k0,
    material,
    poltype,
    modetype=None,
    where=True,
    treams=True,
):
    """Plane-wave to spherical-wave expansion matrix.

    Args:
        basis: Plane-wave basis (treams object if treams=True,
            otherwise array with directions + polarization).
        pidx (array_like): Particle index for each spherical mode.
        l, m, pol (array_like): Spherical mode indices.
        positions (array_like): Particle positions, shape (num, 3).
        k0 (float): Vacuum wavenumber.
        material: Embedding material / permittivity info.
        poltype (str): "helicity" or "parity".
        modetype (str, optional): Plane-wave mode type ("up"/"down").
        where (bool array, optional): Mask; entries outside are zeroed.
        treams (bool, optional): If True, interpret `basis` as treams
            basis; otherwise as plain array.

    Returns:
        jax.Array: Matrix of shape (N_sph, N_pw), mapping plane-wave
        modes (columns) to spherical modes (rows).
    """
    if isinstance(basis, core.PlaneWaveBasisByComp):
        modetype = "up" if modetype is None else modetype

    if treams:
        kvecs = basis.kvecs(k0, material, modetype)
        pw2sw = to_sw(
            *(m[:, None] for m in (l, m, pol)),
            *kvecs,
            basis.pol,
            poltype=poltype,
        )
    else:
        basis = np.array(basis)
        plane_pol = basis[:, -1].astype(int)
        ks = k0 * refractive_index(material)
        ks = ks[plane_pol]
        kvecs = ks * basis[:, 0], ks * basis[:, 1], ks * basis[:, 2]
        pw2sw = to_sw(
            *(m[:, None] for m in (l, m, pol)),
            ks * basis[:, 0],
            ks * basis[:, 1],
            ks * basis[:, 2],
            plane_pol,
            poltype=poltype,
        )
    res = pw2sw * pw_translate(
        *kvecs,
        positions[pidx, None, 0],
        positions[pidx, None, 1],
        positions[pidx, None, 2],
    )

    # res[..., np.logical_not(where)] = 0

    return res


def to_sw(l, m, polsw, kx, ky, kz, polpw, poltype=None):
    """
    Coefficient for the expansion of a plane wave in spherical waves.
     Returns the coefficient for the basis change from a plane wave to a spherical wave.
     For multiple positions only diagonal values (with respect to the position) are
     returned.
     Args:
         l (array_like): Degree(s) of the spherical waves.
         m (array_like): Order(s) of the spherical waves.
         polsw (array_like): Polarizations of the spherical waves
             (0/1).
         kx (array_like): x components of the plane-wave wave vectors.
         ky (array_like): y components of the plane-wave wave vectors.
         kz (array_like): z components of the plane-wave wave vectors.
         polpw (array_like): Polarizations of the plane waves (0/1).
         poltype (str, optional): Polarization type, "helicity" or
             "parity". If None, uses config.POLTYPE.
     Returns:
         jax.Array: Expansion coefficients of shape (N_sph, N_pw),
         where N_sph is the number of spherical modes and N_pw is the
         number of plane-wave modes.
    """
    poltype = config.POLTYPE if poltype is None else poltype
    if poltype == "helicity":
        return _to_sw_h(l, m, polsw, kx, ky, kz, polpw)
    elif poltype == "parity":
        return _to_sw_p(l, m, polsw, kx, ky, kz, polpw)
    raise ValueError(f"invalid poltype '{poltype}'")


def _to_sw_h(l, m, polvsw, kx, ky, kz, polpw):
    kxy = np.sqrt(kx * kx + ky * ky)
    pref = np.where(kxy == 0, 1, np.power((kx - 1j * ky) / kxy, m))
    k = np.sqrt(kx * kx + ky * ky + kz * kz)
    costheta = kz / k
    ans = (
        2
        * np.sqrt(np.pi * (2 * l + 1) / (l * (l + 1)))
        * np.exp(
            0.5
            * (
                loggamma((l - m + 1).astype(float))
                - loggamma((l + m + 1).astype(float))
            )
        )
        * np.power(1j, l)
        * pref
    ) * (tau_fun(l, m, costheta) + (2 * polpw - 1) * pi_fun(l, m, costheta))
    ans = np.where(polvsw != polpw, 0.0j, ans)
    return ans


def _to_sw_p(l, m, polvsw, kx, ky, kz, polpw):
    kxy = np.sqrt(kx * kx + ky * ky)
    pref = np.where(kxy == 0, 1, np.power((kx - 1j * ky) / kxy, m))
    k = np.sqrt(kx * kx + ky * ky + kz * kz)
    costheta = kz / k
    pref = pref * (
        2
        * np.sqrt(np.pi * (2 * l + 1) / (l * (l + 1)))
        * np.exp(
            0.5
            * (
                loggamma((l - m + 1).astype(float))
                - loggamma((l + m + 1).astype(float))
            )
        )
        * np.pow(1j, l)
    )
    ans = np.where(
        polvsw == polpw, pref * tau_fun(l, m, costheta), pref * pi_fun(l, m, costheta)
    )
    return ans


def rotate(t, phi, theta, psi, rad=True, modes=None):
    """
    Rotate a T-matrix in a spherical-wave basis.

    If `modes` is not given, the T-matrix is assumed to be a single-center
    global T-matrix, and the modes are inferred from its size using
    defaultlmax(len(t)) and defaultmodes(lmax, num=1).

    If `modes` is given (as returned by defaultmodes(lmax, num)), the
    particle indices `pidx` are used to restrict the rotation to be
    block-diagonal in the particle index: modes belonging to different
    particles do not mix, which corresponds to rotating a local basis with
    several scatterers.

    Args:
        t (ndarray): T-matrix in a spherical-wave basis, shape (N, N).
        phi (float): First Euler angle (rotation about z).
        theta (float): Second Euler angle (rotation about y).
        psi (float): Third Euler angle (rotation about z).
        rad (bool, optional): If False, angles are given in degrees.
            If True (default), angles are in radians.
        modes (tuple, optional): Tuple (pidx, l, m, pol) as returned by
            defaultmodes(lmax, num). If None, a single-center basis is
            assumed and modes are constructed automatically.

    Returns:
        ndarray: Rotated T-matrix with the same shape as `t`.
    """
    if not rad:
        phi = phi * np.pi / 180.0
        theta = theta * np.pi / 180.0
        psi = psi * np.pi / 180.0

    # Infer modes if not provided: assume single center (num=1)
    if modes is None:
        lmax = defaultlmax(len(t))
        modes = defaultmodes(lmax, 1)

    pidx, l, m, pol = modes
    n = len(l)

    l1 = l[:, None]
    l2 = l[None, :]
    m1 = m[:, None]
    m2 = m[None, :]
    pol1 = pol[:, None]
    pol2 = pol[None, :]
    pidx1 = pidx[:, None]
    pidx2 = pidx[None, :]
    cond_pol = ((pol1 == 0) & (pol2 == 0)) | ((pol1 == 1) & (pol2 == 1))
    cond_l = l1 == l2

    cond_pidx = pidx1 == pidx2

    cond = cond_pol & cond_l & cond_pidx

    # Wigner D-matrix on the allowed entries
    mat = np.where(cond, wignerd(l2, m1, m2, phi, theta, psi), 0.0)
    return mat @ t @ np.conj(mat).T
