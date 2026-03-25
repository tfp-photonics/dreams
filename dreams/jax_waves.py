import numpy as anp
import jax
import jax.numpy as np
from dreams.jax_primitive import spherical_hankel1, spherical_jn, lpmv
from jax import config
from jax.lax import lgamma as loggamma
from dreams.jax_coord import car2sph, vsph2car
from dreams.jax_misc import refractive_index, wave_vec_z
from dreams.jax_misc import minusonepow_dif as minusonepow
from jax import lax

config.update("jax_enable_x64", True)
import treams.special as sp

SQPI = np.sqrt(np.pi)
M_SQRT1_2 = np.sqrt(1 / 2)
M_SQRT2 = np.sqrt(2)


def tau_fun(l, m, x):
    r"""
    Angular function tau

    .. math::

        \tau^l_m(x)
        = \left.\frac{\mathrm d}{\mathrm d \theta}P_l^m(\cos\theta)\right|_{x = \cos\theta}

    where :math:`P^l_m` is the associated Legendre polynomial.

    Args:
        l (integer, array_like): degree :math:`l \geq 0`
        m (integer, array_like): order :math:`|m| \leq l`
        x (float or complex, array_like): argument

    Returns:
        float or complex
    """
    ans = np.where(
        l == m,
        -l * lpmv(m - 1, l, np.arccos(x)),
        np.where(
            l == -m,
            0.5 * lpmv(m + 1, l, np.arccos(x)),
            (
                lpmv(m + 1, l, np.arccos(x))
                - (l + m) * (l - m + 1) * lpmv(m - 1, l, np.arccos(x))
            )
            * 0.5,
        ),
    )
    return ans


def pi_fun(l, m, x):
    r"""
    Angular function

    .. math::

        \pi_l^m(x) = \frac{m P_l^m(x)}{\sqrt{1 - x^2}}

    where :math:`P^l_m` is the associated Legendre polynomial.

    Args:
        l (int): degree :math:`l \geq 0`
        m (int): order :math:`|m| \leq l`
        x (float or complex): argument

    Returns:
        float or complex
    """
    st = np.sqrt(1 - x * x)
    str = np.real(st)
    sti = np.imag(st)
    ans = np.where(
        str * str + sti * sti < 1e-40,
        np.where(
            m == 1,
            -np.power(x, l + 1) * l * (l + 1) * 0.5,
            np.where(m == -1, -np.power(x, l + 1) * 0.5, 0.0),
        ),
        m * lpmv(m, l, np.arccos(x)) / st,
    )
    return ans


def plane_wave(
    kvec, pol, *, k0=None, basis=None, epsilon=None, mu=1, modetype=None, poltype=None
):
    """Array describing a plane wave.

    Args:
        kvec (array_like): Wave vector (2- or 3-component).
        pol (int or array_like): Polarization index (0/1 or -1/1) or
            3-element Cartesian electric field.
        basis (array_like, optional): Mode list; shape (N, 4) for 3D
            [qx, qy, qz, pol] or (N, 3) for 2D [kx, ky, pol].
        k0 (float, optional): Vacuum wave number.
        epsilon (float or array_like, optional): Relative permittivity.
        mu (float or array_like, optional): Relative permeability.
        modetype ({"up", "down"}, optional): For partial (kx, ky) usage.
        poltype ({"parity", "helicity"}, optional): Polarization basis.
    """
    kvec = np.array(kvec)
    if len(kvec) == 2:
        return _plane_wave_partial(
            kvec,
            pol,
            k0=k0,
            basis=basis,
            epsilon=epsilon,
            mu=mu,
            modetype=modetype,
            poltype=poltype,
        )
    if len(kvec) == 3:
        return _plane_wave(
            kvec,
            pol,
            k0=k0,
            basis=basis,
            epsilon=epsilon,
            mu=mu,
            modetype=modetype,
            poltype=poltype,
        )
    raise ValueError(f"invalid length of 'kvec': {len(kvec)}")


def _plane_wave(
    kvec, pol, *, k0, basis=None, epsilon=None, mu=1, modetype=None, poltype=None
):
    if basis is None:
        basis = default_plane_wave_unit(kvec)
    ks = k0 * refractive_index(epsilon, mu)
    norm = np.sqrt(np.sum(np.power(kvec, 2)))
    qvec = kvec / norm
    if pol in (0, -1):
        pol = np.array([1, 0])
    elif pol == 1:
        pol = np.array([0, 1])
    elif len(pol) == 3:
        pol = np.array(pol)
        if None not in (k0, epsilon):
            kvec = ks * qvec[:, None]
        else:
            kvec = qvec
        if poltype == "parity":
            pol = np.array(
                [
                    -vpw_M(kvec[:, 0][0], kvec[:, 0][1], kvec[:, 0][2], 0, 0, 0) @ pol,
                    vpw_N(kvec[:, 1][0], kvec[:, 1][1], kvec[:, 1][2], 0, 0, 0) @ pol,
                ]
            )
        elif poltype == "helicity":
            pol = vpw_A(kvec[0], kvec[1], kvec[2], 0, 0, 0, [1, 0]) @ pol
        else:
            raise ValueError(f"invalid 'poltype': {poltype}")
    res = np.array(
        [pol[(x[3]).astype(int)] * (np.abs(qvec - x[:3]) < 1e-14).all() for x in basis]
    )
    return res, basis


def default_plane_wave_unit(kvecs):
    """Default basis from the given wave vectors.

    For each wave vector the two polarizations 1 and 0 are taken.

    Example:
        >>> PlaneWaveBasisByUnitVector.default([[0, 0, 5], [0, 3, 4]])
        PlaneWaveBasisByUnitVector(
            qx=[0. 0. 0. 0.],
            qy=[0.  0.  0.6 0.6],
            qz=[1.  1.  0.8 0.8],
            pol=[1 0 1 0],
        )

    Args:
        kvecs (array-like): Wave vectors in Cartesian coordinates.
    """
    kvecs = anp.atleast_2d(kvecs)
    shape1 = 2 * kvecs.shape[0]
    modes = anp.empty((shape1, 4), kvecs.dtype)
    col4 = anp.array([1, 0] * kvecs.shape[0])[:, None]
    cols3 = anp.repeat(kvecs, 2, axis=-2)
    modes = anp.concatenate((cols3, col4), axis=-1)
    qx, qy, qz, pol = modes.T
    norm = anp.sqrt((qx * qx + qy * qy + qz * qz).astype(complex))
    norm = anp.where(np.abs(norm - 1) < 1e-14, 1, norm)
    qx, qy, qz = (anp.true_divide(i, norm) for i in (qx, qy, qz))
    return anp.array([qx, qy, qz, pol]).T


def _plane_wave_partial(
    kpar, pol, *, k0=None, basis=None, epsilon=None, mu=1, modetype=None, poltype=None
):
    if basis is None:
        basis = default_plane_wave(kpar)
    if pol in (0, -1):
        pol = np.array([1, 0])
    elif pol == 1:
        pol = np.array([0, 1])
    elif len(pol) == 3:
        modetype = "up" if modetype is None else modetype
        if modetype not in ("up", "down"):
            raise ValueError(f"invalid 'modetype': {modetype}")
        # kvecs = np.array(basis.kvecs(k0, material, modetype))
        kx = basis[:, 0].real
        ky = basis[:, 1].real
        pol_basis = basis[:, 2].real.astype(int)
        ks = k0 * refractive_index(epsilon, mu)
        ks = np.array(ks)[pol_basis]
        kzs = wave_vec_z(kx, ky, ks, nondifsum=False) * (2 * (modetype == "up") - 1)
        kvecs = np.array([kx, ky, kzs])

        if poltype == "parity":
            kvec = kvecs[:, 1]
            # breakpoint()
            pol = np.array(pol)
            pol = [
                -vpw_M(kvec[0], kvec[1], kvec[2], 0, 0, 0) @ pol,
                vpw_N(kvec[0], kvec[1], kvec[2], 0, 0, 0) @ pol,
            ]
        elif poltype == "helicity":
            kvec = kvecs[:, [1, 0]]
            pol = vpw_A(*kvec, 0, 0, 0, [1, 0]) @ pol
        else:
            raise ValueError(f"invalid 'poltype': {poltype}")
    # res = np.array([pol[x[2].astype(int)] * (np.abs(np.array(kpar) - x[:2]) < 1e-14).all() for x in basis])#.squeeze()
    pol_idx = basis[:, 2].astype(np.int32)  # (N,)
    pol_factor = np.array(pol)[pol_idx]  # (N,)
    diff = np.abs(basis[:, :2] - kpar) < 1e-14  # (N, 2) bool
    mask = np.all(diff, axis=-1)
    mask = mask.astype(pol_factor.dtype)
    res = pol_factor * mask
    return res, basis


def default_plane_wave(kpars, alignment="xy"):
    """Default basis for partial plane waves in the (kx, ky) plane.

    This supports alignment='xy'.
    """
    kpars = np.atleast_2d(kpars)

    if alignment != "xy":
        raise NotImplementedError("alignment != 'xy' is not implemented")

    kpars = np.atleast_2d(kpars)
    shape1 = 2 * kpars.shape[0]
    modes = np.empty((shape1, 3), kpars.dtype)
    col3 = np.array([1, 0] * kpars.shape[0])[:, None]
    cols2 = np.repeat(kpars, 2, axis=-2)
    modes = np.concatenate((cols2, col3), axis=-1)
    return np.array(modes)


def sph_harm(m, l, phi, theta):
    """Spherical harmonics for complex argument"""
    ans = (
        0.25
        * np.sqrt(2 * l + 1)
        * 2
        / np.sqrt(np.pi)
        * np.exp(0.5 * (loggamma(l - m + 1.0) - loggamma(l + m + 1.0)))
        * lpmv(m, l, theta)
        * np.exp(1j * m * phi)
    )
    return ans


def _spherical_jn_div_x(l, x):
    """Spherical Bessel function of the first kind devided by its argument"""
    return (spherical_jn(l - 1, x) + spherical_jn(l + 1, x)) / (2 * l + 1)


def _spherical_hankel1_div_x(l, x):
    """Spherical Hankel function of the first kind devided by its argument"""
    return (spherical_hankel1(l - 1, x) + spherical_hankel1(l + 1, x)) / (2 * l + 1)


def vsh_Z(l, m, theta, phi):
    r"""
    Vector spherical harmonic :math:`\mathbf Z_{lm}` in spherical coordinates.

    Defined from the scalar spherical harmonic :math:`Y_{lm}` as

    .. math::

        \mathbf Z_{lm}(\theta,\phi)
        = i\,Y_{lm}(\theta,\phi)\,\hat{\mathbf r},

    where :math:`\hat{\mathbf r}` is the radial unit vector.

    Parameters
    ----------
    l : int or array_like
        Degree :math:`l \ge 0`.
    m : int or array_like
        Order :math:`|m| \le l`.
    theta : array_like
        Polar angle :math:`\theta`.
    phi : array_like
        Azimuthal angle :math:`\phi`.

    Returns
    -------
    ndarray
        Complex array with shape ``(..., 3)`` containing the
        :math:`(r,\theta,\phi)` components.
    """
    Z0 = 1j * sph_harm(m, l, phi, theta)
    ans = np.stack([Z0, np.zeros_like(Z0), np.zeros_like(Z0)], axis=-1)  # .T
    return ans


def vsh_Y(l, m, theta, phi):
    r"""
    Vector spherical harmonic :math:`\mathbf Y_{lm}` in spherical coordinates.

    It is defined by

    .. math::

        \mathbf Y_{lm}(\theta,\phi)
        = i \sqrt{\frac{2l+1}{4\pi\,l(l+1)}\frac{(l-m)!}{(l+m)!}}
          \left[ \tau_l^m(\cos\theta)\,\hat{\boldsymbol\theta}
               + i\,\pi_l^m(\cos\theta)\,\hat{\boldsymbol\phi} \right],

    where :math:`\pi_l^m` and :math:`\tau_l^m` are the angular functions
    implemented as :func:`pi_fun` and :func:`tau_fun`.

    Parameters
    ----------
    l : int or array_like
        Degree, :math:`l \ge 0`.
    m : int or array_like
        Order, :math:`|m| \le l`.
    theta : float or array_like
        Polar angle.
    phi : float or array_like
        Azimuthal angle.

    Returns
    -------
    ndarray, shape (..., 3)
        Complex vector with components in the
        :math:`(\hat{\boldsymbol r},\hat{\boldsymbol\theta},\hat{\boldsymbol\phi})`
        basis.
    """
    pref = (
        1j
        * np.sqrt((2 * l + 1) / (4 * np.pi * l * (l + 1)))
        * np.exp(
            (loggamma((l - m + 1).astype(float)) - loggamma((l + m + 1).astype(float)))
            * 0.5
        )
    ) * np.exp(1j * m * phi)
    Y1 = pref * tau_fun(l, m, np.cos(theta))
    Y2 = pref * 1j * pi_fun(l, m, np.cos(theta))
    out = np.stack([np.zeros_like(Y1), Y1, Y2], axis=-1)
    mask = (l == 0)[None, :, None]
    ans = np.where(mask, np.zeros_like(out), out)
    return ans


def vsh_X(l, m, theta, phi):
    r"""
    Vector spherical harmonic :math:`\mathbf X_{lm}` in spherical coordinates.

    Defined from the scalar spherical harmonics by

    .. math::

        \mathbf X_{lm}(\theta,\phi)
        = \frac{\nabla \times
           \bigl(r\,Y_{lm}(\theta,\phi)\hat{\mathbf r}\bigr)}
           {\sqrt{l(l+1)}},

    which can be written as

    .. math::

        \mathbf X_{lm}(\theta,\phi)
        = i \sqrt{\frac{2l+1}{4\pi\,l(l+1)}}
          \sqrt{\frac{(l-m)!}{(l+m)!}}
          \bigl(
              i\,\pi_l^m(\cos\theta)\,\hat{\boldsymbol\theta}
            - \tau_l^m(\cos\theta)\,\hat{\boldsymbol\phi}
          \bigr).

    Parameters / Returns
    --------------------
    Same as :func:`vsh_Z`.
    """
    pref = (
        1j
        * np.sqrt((2 * l + 1) / (4 * np.pi * l * (l + 1)))
        * np.exp(
            (loggamma((l - m + 1).astype(float)) - loggamma((l + m + 1).astype(float)))
            * 0.5
        )
    ) * np.exp(1j * m * phi)
    X1 = 1j * pref * pi_fun(l, m, np.cos(theta))
    X2 = -pref * tau_fun(l, m, np.cos(theta))
    out = np.array([np.zeros_like(X1), X1, X2])
    out = np.stack([np.zeros_like(X1), X1, X2], axis=-1)
    mask = (l == 0)[None, :, None]
    ans = np.where(mask, np.zeros_like(out), out)
    return ans


def vsw_N(l, m, kr, theta, phi):
    """Regular vector spherical wave N"""
    Z = vsh_Z(l, m, theta, phi)
    Y = vsh_Y(l, m, theta, phi)
    kr_safe = np.where(kr == 0, 1.0, kr)
    hl = spherical_hankel1(l, kr_safe)
    hlm1 = np.where(
        l == 0, np.exp(1j * kr_safe) / kr_safe, spherical_hankel1(l - 1, kr_safe)
    )
    pref = hlm1 - (l / kr_safe) * hl
    out = np.stack(
        [
            Z[..., 0] * _spherical_hankel1_div_x(l, kr) * np.sqrt(l * (l + 1)),
            Y[..., 1] * pref,
            Y[..., 2] * pref,
        ],
        axis=-1,
    )

    return out


def vsw_M(l, m, kr, theta, phi):
    """Regular vector spherical wave M"""
    vsh = vsh_X(l, m, theta, phi)
    sh = spherical_hankel1(l, kr)
    out = np.stack([vsh[..., 0], vsh[..., 1] * sh, vsh[..., 2] * sh], axis=-1)
    return out


def vsw_rN(l, m, kr, theta, phi):
    """Regular vector spherical wave N"""
    Z = vsh_Z(l, m, theta, phi)
    Y = vsh_Y(l, m, theta, phi)
    pref = _spherical_jn_div_x(l, kr) + jax.vmap(
        jax.jacrev(spherical_jn, argnums=1, holomorphic=True), in_axes=(0, 0)
    )(l, kr + 0.0j)
    cf = _spherical_jn_div_x(l, kr) * np.sqrt(l * (l + 1))
    out = np.array([Z[:, 0] * cf, Y[:, 1] * pref, Y[:, 2] * pref]).T
    return out


def vsw_rM(l, m, kr, theta, phi):
    """Regular vector spherical wave M"""
    vsh = vsh_X(l, m, theta, phi)
    sh = spherical_jn(l, kr)
    out = np.array([vsh[:, 0], vsh[:, 1] * sh, vsh[:, 2] * sh]).T
    return out


def vsw_A(l, m, kr, theta, phi, pol):
    """Regular vector spherical wave of well-defined helicity"""
    sign = np.where(pol > 0, 1, -1)
    N = vsw_N(l, m, kr, theta, phi)
    M = vsw_M(l, m, kr, theta, phi)
    out = np.array(
        [
            M_SQRT1_2 * (N[:, 0] + sign * M[:, 0]),
            M_SQRT1_2 * (N[:, 1] + sign * M[:, 1]),
            M_SQRT1_2 * (N[:, 2] + sign * M[:, 2]),
        ]
    ).T
    return out


def vsw_rA(l, m, kr, theta, phi, pol):
    """Regular vector spherical wave of well-defined helicity"""
    sign = np.where(pol > 0, 1, -1)
    rN = vsw_rN(l, m, kr, theta, phi)
    rM = vsw_rM(l, m, kr, theta, phi)
    out = np.array(
        [
            M_SQRT1_2 * (rN[:, 0] + sign * rM[:, 0]),
            M_SQRT1_2 * (rN[:, 1] + sign * rM[:, 1]),
            M_SQRT1_2 * (rN[:, 2] + sign * rM[:, 2]),
        ]
    ).T
    return out


def vpw_M(
    kx,
    ky,
    kz,
    x,
    y,
    z,
):
    """Vector plane wave M"""
    k = np.sqrt(kx * kx + ky * ky + kz * kz)
    kpar = np.sqrt(kx * kx + ky * ky)
    phase = np.exp(1j * (kx * x + ky * y + kz * z))
    out1 = np.array([np.nan, np.nan, np.nan]).T
    out2 = np.array([np.zeros_like(phase), -1j * phase, np.zeros_like(phase)]).T
    out3 = np.array(
        [1j * ky * phase / kpar, -1j * kx * phase / kpar, np.zeros_like(phase)]
    ).T
    cond_k0 = k == 0
    cond_kpar0 = (kpar == 0) & ~cond_k0
    ans = np.select([cond_k0, cond_kpar0], [out1, out2], out3)
    return ans


def vpw_N(
    kx,
    ky,
    kz,
    x,
    y,
    z,
):
    """Vector plane wave N"""
    k = np.sqrt(kx * kx + ky * ky + kz * kz)
    kpar = np.sqrt(kx * kx + ky * ky)
    phase = np.exp(1j * (kx * x + ky * y + kz * z))

    def out2():
        sign = np.where(
            np.imag(kz) == 0,
            np.where(np.real(kz) >= 0, 1, -1),
            np.where(np.imag(kz) > 0, 1, -1),
        )
        ans = np.array([-phase * sign, np.zeros_like(phase), np.zeros_like(phase)]).T
        return ans

    out3 = np.array(
        [-kx * kz * phase / (k * kpar), -ky * kz * phase / (k * kpar), kpar * phase / k]
    ).T
    out1 = np.full_like(out3, np.nan)
    cond_k0 = k == 0
    cond_kpar0 = (kpar == 0) & ~cond_k0
    ans = np.select([cond_k0, cond_kpar0], [out1, out2()], out3)
    return ans


def vpw_A(kx, ky, kz, x, y, z, pol):
    """Regular vector plane wave of well-defined helicity"""
    sign = np.where(np.array(pol) > 0, 1, -1)
    N = vpw_N(kx, ky, kz, x, y, z)
    M = vpw_M(kx, ky, kz, x, y, z)
    out = np.array(
        [
            M_SQRT1_2 * (N[:, 0] + sign * M[:, 0]),
            M_SQRT1_2 * (N[:, 1] + sign * M[:, 1]),
            M_SQRT1_2 * (N[:, 2] + sign * M[:, 2]),
        ]
    ).T
    return out


def efield(r, pidx, l, m, pol, positions, k0, epsilon, modetype, poltype):
    r"""Electric field of vector spherical waves.

    Evaluate the electric field in Cartesian coordinates for a set of
    vector spherical waves centred at one particle position.

    The function builds regular or singular waves (Bessel or Hankel)
    in either helicity or parity basis and returns the corresponding
    electric field components at the observation points.

    Args:
        r (array-like): Evaluation points in Cartesian coordinates,
            shape ``(..., 3)``.
        pidx (int): Index of the particle/center in ``positions``.
        l (array-like): Multipole degrees :math:`l \ge 0`.
        m (array-like): Multipole orders :math:`|m| \le l`.
        pol (array-like): Polarization indices of the modes. Interpreted
            as helicity (±1 or 0/1) if ``poltype="helicity"`` or as
            TE/TM index (0/1) if ``poltype="parity"``.
        positions (array-like): Particle centres in Cartesian
            coordinates, shape ``(npart, 3)``.
        k0 (float): Vacuum wave number.
        epsilon (float or array-like): Relative permittivity of the
            embedding medium.
        modetype ({"regular", "singular"}): Type of vector spherical
            wave; regular uses spherical Bessel functions, singular
            uses spherical Hankel functions.
        poltype ({"helicity", "parity"}): Polarization basis of the
            vector spherical waves.

    Returns:
        ndarray: Complex electric field at the evaluation points with
        shape ``r.shape[:-1] + (3, N)`` where the last two axes are the
        Cartesian components (Ex, Ey, Ez) and the mode index.
    """

    r = r[..., None, :]  # (P,1,3)
    pos = positions[pidx]  # (N,3)
    rsph = car2sph(r - pos[None, :, :])  # (P,N,3)

    ks = k0 * refractive_index(epsilon)  # scalar or (2,)
    kr = rsph[..., 0] * ks[pol][None, :]  # (P,N)
    th = rsph[..., 1]
    ph = rsph[..., 2]

    if poltype == "helicity":
        if modetype == "regular":
            res = vsw_rA(l, m, kr, th, ph, pol)
        else:
            res = vsw_A(l, m, kr, th, ph, pol)
    else:

        polw = pol[None, :, None]  # (1,N,1)
        # polw = pol[:, None, None]
        if modetype == "regular":
            res = (1 - polw) * vsw_rM(l, m, kr, th, ph) + polw * vsw_rN(
                l, m, kr, th, ph
            )
        else:
            # res = np.ones((2, 180, 3))
            vm = vsw_M(l, m, kr, th, ph)
            vn = vsw_N(l, m, kr, th, ph)
            # print("vm", vm.shape, "vn", vn.shape)
            res = (1 - polw) * vm + polw * vn
    res = vsph2car(res, rsph)
    ans = res.swapaxes(-1, -2)
    return ans


_HELPER_CACHE_A = {}


def precompute_helper_stack_A(lambda_, mu, l, m):
    l_np = anp.asarray(l, dtype=anp.int64)
    m_np = anp.asarray(m, dtype=anp.int64)
    lam_np = anp.asarray(lambda_, dtype=anp.int64)
    mu_np = anp.asarray(mu, dtype=anp.int64)

    key = (
        l_np.shape,
        lam_np.shape,
        int(l_np.max()),
        int(lam_np.max()),
        int(m_np.min()),
        int(m_np.max()),
        int(mu_np.min()),
        int(mu_np.max()),
    )
    if key in _HELPER_CACHE_A:
        return _HELPER_CACHE_A[key]

    p0 = l_np + lam_np
    dif0 = p0 - (anp.maximum(anp.abs(lam_np - l_np), anp.abs(m_np - mu_np)) - 1)
    upper_value = int(anp.max(dif0))
    n_steps = max(0, (upper_value + 1) // 2)

    helpers = []
    for i in range(n_steps):
        p = p0 - 2 * i
        helpers.append(sp._tl_vsw_helper(l_np, m_np, lam_np, -mu_np, p, p))
    helper_stack = anp.stack(helpers, axis=0)  # shape (n_steps, ...), numpy complex

    _HELPER_CACHE_A[key] = helper_stack
    return helper_stack


_HELPER_CACHE_B = {}


def precompute_helper_stack_B(lambda_, mu, l, m):
    l_np = anp.asarray(l, dtype=anp.int64)  # (n,)
    m_np = anp.asarray(m, dtype=anp.int64)  # (n,)
    lam_np = anp.asarray(lambda_, dtype=anp.int64)  # (n,1) or (n,)
    mu_np = anp.asarray(mu, dtype=anp.int64)  # (n,1) or (n,)

    if lam_np.ndim == 1:
        lam_np = lam_np[:, None]
    if mu_np.ndim == 1:
        mu_np = mu_np[:, None]

    key = (
        l_np.shape,
        lam_np.shape,
        int(l_np.max()),
        int(lam_np.max()),
        int(m_np.min()),
        int(m_np.max()),
        int(mu_np.min()),
        int(mu_np.max()),
    )
    if key in _HELPER_CACHE_B:
        return _HELPER_CACHE_B[key]

    l_row = l_np[None, :]  # (1,n)
    m_row = m_np[None, :]  # (1,n)

    p0 = l_row + lam_np - 1
    dif0 = p0 - (anp.maximum(anp.abs(lam_np - l_row) + 1, anp.abs(m_row - mu_np)) - 1)
    upper_value = int(anp.max(dif0))
    n_steps = max(0, (upper_value + 1) // 2)

    helpers = []
    for i in range(n_steps):
        p = p0 - 2 * i
        helpers.append(sp._tl_vsw_helper(l_row, m_row, lam_np, -mu_np, p, p - 1))

    helper_stack = anp.stack(helpers, axis=0)  # (n_steps, n, n)
    _HELPER_CACHE_B[key] = helper_stack
    return helper_stack



#@jax.jit
def tl_vsw_rA_scan(lambda_, mu, l, m, kr, theta, phi, helper_stack):
    helper_stack = np.asarray(helper_stack)
    pref = (
        0.5
        * minusonepow(m)
        * np.sqrt(
            (2 * l + 1) * (2 * lambda_ + 1) / (l * (l + 1) * lambda_ * (lambda_ + 1))
        )
        * np.exp(1j * (m - mu) * phi)
    )
    p0 = l + lambda_
    dif0 = p0 - (np.maximum(np.abs(lambda_ - l), np.abs(m - mu)) - 1)
    res0 = np.zeros_like(p0, dtype=np.complex128)

    def body(res, i):
        p = p0 - 2 * i
        dif = dif0 - 2 * i
        term = (
            helper_stack[i]
            * (l * (l + 1) + lambda_ * (lambda_ + 1) - p * (p + 1))
            * spherical_jn(p, kr)
            * lpmv(m - mu, p, theta)
        )
        return res + term * (dif > 0), None

    idx = np.arange(helper_stack.shape[0], dtype=np.int32)
    res, _ = lax.scan(body, res0, idx)
    ans =  res * pref
    jax.block_until_ready(ans)
    return ans


def tl_vsw_rA(lambda_, mu, l, m, kr, theta, phi):
    helperA = np.asarray(precompute_helper_stack_A(lambda_, mu, l, m))
    return tl_vsw_rA_scan(lambda_, mu, l, m, kr, theta, phi, helperA)  # jitted


#@jax.jit
def tl_vsw_A_scan(lambda_, mu, l, m, kr, theta, phi, helper_stack):
    helper_stack = np.asarray(helper_stack)

    pref = (
        0.5
        * minusonepow(m)
        * np.sqrt(
            (2 * l + 1) * (2 * lambda_ + 1) / (l * (l + 1) * lambda_ * (lambda_ + 1))
        )
        * np.exp(1j * (m - mu) * phi)
    )

    p0 = l + lambda_
    dif0 = p0 - (np.maximum(np.abs(lambda_ - l), np.abs(m - mu)) - 1)

    res0 = np.zeros_like(p0, dtype=np.complex128)

    def body(res, i):
        #jax.debug.print("in bodz {}", i, ordered=True)
        p = p0 - 2 * i
        dif = dif0 - 2 * i
        term = (
            helper_stack[i]
            * (l * (l + 1) + lambda_ * (lambda_ + 1) - p * (p + 1))
            * spherical_hankel1(p, kr)
            * lpmv(m - mu, p, theta)
        )
        #jax.debug.print(" end in bodz {}", i, ordered=True)
        return res + term * (dif > 0), None

    idx = np.arange(helper_stack.shape[0], dtype=np.int32)
    res, _ = lax.scan(body, res0, idx)
    #jax.debug.print("END SCAN", ordered=True)
    ans =  res * pref
    #jax.debug.print("END ANS", ordered=True)
    jax.block_until_ready(ans)
    return ans


def tl_vsw_A(lambda_, mu, l, m, kr, theta, phi):
    #print("call helper")
    helper_a = np.asarray(precompute_helper_stack_A(lambda_, mu, l, m))
    #print("after helper")
    return tl_vsw_A_scan(lambda_, mu, l, m, kr, theta, phi, helper_a)  # jitted scan


#@jax.jit
def tl_vsw_B_scan(lambda_, mu, l, m, kr, theta, phi, helper_stack):
    pref = (
        0.5
        * minusonepow(m)
        * np.sqrt(
            (2 * l + 1) * (2 * lambda_ + 1) / (l * (l + 1) * lambda_ * (lambda_ + 1))
        )
        * np.exp(1j * (m - mu) * phi)
    )
    p0 = l + lambda_ - 1
    dif0 = p0 - (np.maximum(np.abs(lambda_ - l) + 1, np.abs(m - mu)) - 1)
    res0 = np.zeros_like(p0, dtype=np.complex128)

    def body(res, i):
        p = p0 - 2 * i
        dif = dif0 - 2 * i
        sqrt_fac = np.sqrt(
            np.abs(
                (l + lambda_ + 1 + p)
                * (l + lambda_ + 1 - p)
                * (p + lambda_ - l)
                * (p - lambda_ + l)
            )
        )
        term = (
            helper_stack[i]
            * sqrt_fac
            * spherical_hankel1(p, kr)
            * lpmv(m - mu, p, theta)
        )

        return res + term * (dif > 0), None

    idx = np.arange(helper_stack.shape[0], dtype=np.int32)
    res, _ = lax.scan(body, res0, idx)
    ans =  res * pref
    jax.block_until_ready(ans)
    return ans


def tl_vsw_B(lambda_, mu, l, m, kr, theta, phi):
    helper_b = np.asarray(precompute_helper_stack_B(lambda_, mu, l, m))
    return tl_vsw_B_scan(lambda_, mu, l, m, kr, theta, phi, helper_b)


#@jax.jit
def tl_vsw_rB_scan(lambda_, mu, l, m, kr, theta, phi, helper_stack):
    helper_stack = np.asarray(helper_stack)
    l = l[None, :]
    m = m[None, :]
    pref = (
        0.5
        * minusonepow(m)
        * np.sqrt(
            (2 * l + 1) * (2 * lambda_ + 1) / (l * (l + 1) * lambda_ * (lambda_ + 1))
        )
        * np.exp(1j * (m - mu) * phi)
    )
    p0 = l + lambda_ - 1
    dif0 = p0 - (np.maximum(np.abs(lambda_ - l) + 1, np.abs(m - mu)) - 1)
    res0 = np.zeros_like(p0, dtype=np.complex128)

    def body(res, i):
        p = p0 - 2 * i
        dif = dif0 - 2 * i
        mask = dif > 0
        p_safe = np.where(mask, p, 1)
        sqrt_fac = np.sqrt(
            np.abs(
                (l + lambda_ + 1 + p_safe)
                * (l + lambda_ + 1 - p_safe)
                * (p_safe + lambda_ - l)
                * (p_safe - lambda_ + l)
            )
        )
        term = (
            helper_stack[i]
            * sqrt_fac
            * spherical_jn(p_safe, kr)
            * lpmv(m - mu, p_safe, theta)
        )
        term = np.where(mask, term, 0)
        return res + term, None

    idx = np.arange(helper_stack.shape[0], dtype=np.int32)
    res, _ = lax.scan(body, res0, idx)
    ans =  res * pref
    jax.block_until_ready(ans)
    return ans


def tl_vsw_rB(lambda_, mu, l, m, kr, theta, phi):
    helper_b = np.asarray(precompute_helper_stack_B(lambda_, mu, l, m))
    return tl_vsw_rB_scan(lambda_, mu, l, m, kr, theta, phi, helper_b)
