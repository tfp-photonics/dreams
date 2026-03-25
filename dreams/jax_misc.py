import numpy as anp
import jax.numpy as np
from jax import config
import treams.special.cython_special as cs

config.update("jax_enable_x64", True)


def minusonepow(l):
    r"""(-1) raised to an integer power.

    Computes :math:`(-1)^l` for integer-like input.

    Args:
        l (array_like of int): Exponent(s).

    Returns:
        ndarray: Values in ``{-1, +1}`` with the same shape as ``l``.
    """
    l = anp.asarray(l)
    return anp.where(l % 2 == 0, 1, -1)


def minusonepow_dif(l):
    r"""(-1) raised to an integer power.

    Computes :math:`(-1)^l` for integer-like input.

    Args:
        l (array_like of int): Exponent(s).

    Returns:
        ndarray: Values in ``{-1, +1}`` with the same shape as ``l``.
    """
    l = np.asarray(l)
    return np.where(l % 2 == 0, 1, -1)


def defaultmodes(lmax, nmax=1):
    """
    Default sortation of modes

    Default sortation of the T-Matrix entries, including degree `l`, order `m` and
    polarization `p`.

    Args:
        lmax (int): Maximal value of `l`
        nmax (int, optional): Number of particles, defaults to `1`

    Returns:
        tuple
    """
    return (
        *anp.array(
            [
                [n, l, m, p]
                for n in range(0, nmax)
                for l in range(1, lmax + 1)
                for m in range(-l, l + 1)
                for p in range(1, -1, -1)
            ]
        ).T,
    )


#  @partial(jit, static_argnums = (0,))
def sdefaultmodes(kpars, nondifsum=True):
    """
    Default ordering of plane-wave modes.

    Given tangential wave vectors ``kpars = (kx, ky)`` this returns
    a tuple ``(kx, ky, pol)`` where each pair is duplicated for the
    two polarizations 1 and 0.

    Args:
        kpars (array_like, shape (N, 2)): Tangential components of the wave vector.
        nondifsum (bool, optional): If True, use NumPy-on-JAX (``anp``),
            otherwise plain NumPy.

    Returns:
        tuple of arrays: ``(kx, ky, pol)`` each of length ``2N``.
    """
    if nondifsum:
        kpars = anp.array(kpars).reshape(-1, 2)
        res = anp.repeat(kpars, axis=0, repeats=2)
        pols = anp.empty(2 * kpars.shape[0], int)
        pols = anp.where(anp.arange(len(pols)) % 2, 0, 1)

    else:
        kpars = kpars.reshape(-1, 2)
        res = np.repeat(kpars, axis=0, repeats=2)
        pols = np.empty(2 * kpars.shape[0], int)
        pols = anp.where(anp.arange(len(pols)) % 2, 0, 1)

    return (*res.T, pols)


def defaultlmax(dim, nmax=1):
    """
    Default maximal degree

    Given the dimension of the T-matrix return the estimated maximal value of `l`. This
    is the inverse of defaultdim. A value of zero is allowed for empty T-matrices.

    Args:
        dim (int): Dimension of the T-matrix, respectively number of modes
        nmax (int, optional): Number of particles, defaults to `1`

    Returns:
        int
    """
    res = np.sqrt(1 + dim * 0.5 / nmax) - 1
    res_int = int(np.rint(res))
    if np.abs(res - res_int) > 1e-8 * np.maximum(np.abs(res), np.abs(res_int)):
        raise ValueError("cannot estimate the default lmax")
    return res_int


def defaultdim(lmax, nmax=1):
    """
    Default dimension

    Given the maximal value of `l` return the size of the corresponding T-matrix. This
    is the inverse of defaultlmax. A value of zero is allowed.

    Args:
        lmax (int): Maximal value of `l`
        nmax (int, optional): Number of particles, defaults to `1`

    Returns:
        int
    """
    if lmax < 0 or nmax < 0:
        raise ValueError("maximal order must be positive")
    return 2 * lmax * (lmax + 2) * nmax


def refractive_index(epsilon=1, mu=1, kappa=0):
    r"""
    Refractive index of a (chiral) medium

    The refractive indices in a chiral medium :math:`\sqrt{\epsilon\mu} \mp \kappa` are
    returned with the negative helicity result first.

    Args:
        epsilon (float or complex, array_like, optional): Relative permittivity,
            defaults to 1.
        mu (float or complex, array_like, optional): Relative permeability, defaults to 1.
        kappa (float or complex, array_like, optional): Chirality parameter, defaults to 0.

    Returns:
        float or complex, (2,)-array

    """
    try:
        epsilon = anp.array(epsilon)
        n = anp.sqrt(epsilon * mu)
        res = anp.stack((n - kappa, n + kappa), axis=-1)
        res = anp.where(anp.imag(res) < 0, -res, res)
    except:
        epsilon = np.array(epsilon)
        n = np.sqrt(epsilon * mu)
        res = np.stack((n - kappa, n + kappa), axis=-1)
        res = np.where(anp.imag(res) < 0, -res, res)
    return res


def basischange(out, in_=None):
    """
    Coefficients for the basis change between helicity and parity modes

    Args:
        out (3- or 4-tuple of (M,)-arrays): Output modes, the last array is taken as
            polarization.
        in_ (3- or 4-tuple of (N,)-arrays, optional): Input modes, if none are given,
            equal to the output modes

    Returns:
        ndarray: Change-of-basis matrix of shape ``(M, N)``.
    """
    if in_ is None:
        in_ = out
    out = anp.array([*zip(*out)])
    in_ = anp.array([*zip(*in_)])
    res = anp.zeros((out.shape[0], in_.shape[0]))
    out = out[:, None, :]
    sqhalf = anp.sqrt(0.5)
    equal = (out[:, :, :-1] == in_[:, :-1]).all(axis=-1)
    minus = anp.logical_and(out[:, :, -1] == in_[:, -1], in_[:, -1] == 0)
    res[equal] = sqhalf
    res[anp.logical_and(equal, minus)] = -sqhalf
    return res


def wave_vec_z(kx, ky, k, nondifsum=True):
    r"""
    Z component of the wave vector with positive imaginary part

    The result is :math:`k_z = \sqrt{k^2 - k_x^2 - k_y^2}` with
    :math:`\arg k_z \in \[ 0, \pi )`.

    Args:
        kx (float, array_like): X component of the wave vector
        ky (float, array_like): Y component of the wave vector
        k (float or complex, array_like): Wave number
        nondifsum (bool, optional): If True, use plain numpy (``anp``),
            otherwise jax numpy.
    Returns:
        ndarray: Complex z-components with the same broadcasted shape
        as ``kx``, ``ky``, and ``k``.
    """
    if nondifsum:
        ksq = (k * k - kx * kx - ky * ky).astype(complex)
        res = anp.sqrt(ksq)
        if res.ndim == 0 and anp.imag(res) < 0:
            res = -res
        elif res.ndim > 0:
            res = -res * (anp.imag(res) < 0) + res * (anp.imag(res) >= 0)
        return res

    else:
        ksq = (k * k - kx * kx - ky * ky).astype(complex)
        res = np.sqrt(ksq)
        if res.ndim == 0 and anp.imag(res) < 0:
            res = -res
        elif res.ndim > 0:
            res = -res * (anp.imag(res) < 0) + res * (anp.imag(res) >= 0)
        return res


# wigner3j = anp.vectorize(cs.wigner3j)
def wigner3j(l1, l2, l3, m1, m2, m3):
    r"""Vectorized Wigner 3-j symbol.

    Thin wrapper around :func:`treams.special.cython_special.wigner3j`
    that evaluates the Wigner 3-j symbol on array inputs.

    All arguments are broadcast to the shape of ``l3``.

    Args:
        l1, l2, l3 (array_like of int):  Degrees.
        m1, m2, m3 (array_like of int): Orders.

    Returns:
        ndarray: Complex array with the same shape as ``l3`` containing
        :math:`\left(\begin{smallmatrix}
        l_1 & l_2 & l_3 \\ m_1 & m_2 & m_3
        \end{smallmatrix}\right)`.
    """
    array = anp.zeros(l3.shape)
    l1 = l1 + anp.zeros_like(l3)
    l2 = l2 + anp.zeros_like(l3)
    m1 = m1 + anp.zeros_like(l3)
    m2 = m2 + anp.zeros_like(l3)
    m3 = m3 + anp.zeros_like(l3)
    arg = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
    arg = np.nonzero((l1 >= 0) & (l2 >= 0) & (l3 >= 0))

    if len(arg) != 0:
        array[arg] = anp.vectorize(cs.wigner3j, otypes=[anp.complex128])(
            l1[arg], l2[arg], l3[arg], m1[arg], m2[arg], m3[arg]
        )
    return array


def diffr_orders_circle(b, rmax):
    """
    Diffraction orders inside a circular cutoff.

    Given a 2D reciprocal lattice with row vectors :math:`\mathbf{b}_0` and
    :math:`\mathbf{b}_1` stored in ``b``, this function returns all pairs of
    integer indices :math:`(m, n)` such that the reciprocal lattice vector

    .. math::

        \mathbf{G}_{mn} = m\,\mathbf{b}_0 + n\,\mathbf{b}_1

    satisfies :math:`|\mathbf{G}_{mn}| \le r_\text{max}`.

    Args:
        b: (2, 2) array, reciprocal lattice vectors as rows.
        rmax: float, maximal radius in |G|.

    Returns:
        int64 array of shape (K, 2) with (m, n) indices.
    """
    b = np.asarray(b, dtype=float)
    if b.shape != (2, 2):
        raise ValueError("Wrong shape")
    rmax = float(rmax)

    if rmax < 0:
        return np.zeros((0, 2), dtype=np.int64)

    b0 = b[0]
    b1 = b[1]
    norm_b0 = np.sqrt(np.sum(b0**2))
    norm_b1 = np.sqrt(np.sum(b1**2))

    # avoid division by zero
    eps = 1e-15
    norm_b0 = np.maximum(norm_b0, eps)
    norm_b1 = np.maximum(norm_b1, eps)

    #  safe bounds on m, n (slightly overestimated ok)
    m_max = int(np.ceil(rmax / norm_b0)) + 1
    n_max = int(np.ceil(rmax / norm_b1)) + 1

    ms = np.arange(-m_max, m_max + 1, dtype=np.int64)
    ns = np.arange(-n_max, n_max + 1, dtype=np.int64)
    M, N = np.meshgrid(ms, ns, indexing="ij")  # shape (Mm, Nn)
    mn = np.stack([M.ravel(), N.ravel()], axis=1)  # (K, 2)

    # mask G by radius
    G = mn[:, 0:1] * b0[None, :] + mn[:, 1:2] * b1[None, :]
    r2 = np.sum(G**2, axis=1)
    mask = r2 <= rmax**2

    phi = np.arctan2(G[:, 1], G[:, 0])

    # sort first by radius then by angle on each circle
    order = np.lexsort((phi[mask], r2[mask]))
    res = mn[mask]
    res = res[order]
    return res
