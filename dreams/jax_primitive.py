import scipy
import jax
import jax.numpy as np
from jax import config
import treams.special as sp

config.update("jax_enable_x64", True)


def mod_jn(v, z):
    bessel = np.where(v < 0, np.zeros_like(z), sp.spherical_jn(v, z).astype(z.dtype))
    return np.where((np.abs(z) < 1e-8) & (v != 0), np.zeros_like(z), bessel)


def spherical_jn(v, z):
    """
    Spherical Bessel function  of the first kind j_v(z) with custom JVP.

    Args:
        v (array_like): Integer order(s).
        z (array_like): Real or complex argument.

    Returns:
        jax.Array: j_v(z) with broadcasting over v and z.
    """
    v, z = np.asarray(v), np.asarray(z)

    #   Require the order v to be integer type: simplifies
    #   the JVP rule below.
    assert np.issubdtype(v.dtype, np.integer)

    #   Promote the input to inexact (float/complex).
    #   Note that np.result_type() accounts for the enable_x64 flag.
    z = z.astype(np.result_type(complex, z.dtype))

    #   Wrap scipy function to return the expected dtype.
    _scipy_jv = lambda v, z: mod_jn(v, z).astype(z.dtype)

    #   Define the expected shape & dtype of output.
    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=np.broadcast_shapes(v.shape, z.shape), dtype=z.dtype
    )
    #   use vectorize=True because scipy.special.jv handles broadcasted inputs.
    return jax.pure_callback(
        _scipy_jv, result_shape_dtype, v, z, vmap_method="legacy_vectorized"
    )


spherical_jn = jax.custom_jvp(spherical_jn)


@spherical_jn.defjvp
def _jv_jvp(primals, tangents):
    """
    The Jacobian-vector product (JVP) for the spherical Bessel function of the first kind j_v(z).

    Uses the recurrence relation:
    .. math::
        \frac{d}{dz} j_v(z) = \frac{v}{z} j_v(z) - j_{v+1}(z).
    At the limit z = 0, the derivative is set to 0 for numerical stability.

    ### Special Case for v = 1:
    For small z, the spherical Bessel function j_1(z) has the asymptotic expansion:

    .. math::
        j_1(z) \approx \frac{z}{3} \quad \text{as } z \to 0.

    Differentiating this gives:

    .. math::
        \frac{d}{dz} j_1(z) \approx \frac{1}{3}, \quad \text{so} \quad \lim_{z \to 0} \frac{d}{dz} j_1(z) = \frac{1}{3}.

    This result follows from the standard small-argument expansion of spherical Bessel functions.

    ### References:
    - NIST DLMF: https://dlmf.nist.gov/10.49#E3
    - Abramowitz & Stegun, Handbook of Mathematical Functions, Chapter 10.

    """
    v, z = primals
    _, z_dot = tangents  # Note: v_dot is always 0 because v is integer.
    jv_v_z = spherical_jn(v, z)
    jv_plus_1 = spherical_jn(v + 1, z)
    djv_dz = np.where(
        (np.abs(z) < 1e-8) & (v != 1),
        0.0j,
        np.where((np.abs(z) < 1e-8) & (v == 1), 1 / 3, v * jv_v_z / z - jv_plus_1),
    )
    return jv_v_z, z_dot * djv_dz


def mod_h1(v, z):
    hankel = np.where(
        np.abs(z) < 1e-8, np.zeros_like(z), sp.hankel1(v + 0.5, z).astype(z.dtype)
    )
    return np.where(
        np.abs(z) < 1e-8, np.zeros_like(z), hankel * np.sqrt(np.pi / (2 * z))
    )


def spherical_hankel1(v, z):
    """
    Spherical Hankel function  of the first kind h_v^{(1)}(z) with custom JVP.

    Args:
        v (array_like): Integer order(s).
        z (array_like): Real or complex argument.

    Returns:
        jax.Array: h_v^{(1)}(z) with broadcasting over v and z.
    """
    v, z = np.asarray(v), np.asarray(z)
    z = z.astype(np.result_type(complex, z.dtype))
    assert np.issubdtype(v.dtype, np.integer)
    _scipy_h1 = lambda v, z: mod_h1(v, z)
    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=np.broadcast_shapes(v.shape, z.shape), dtype=z.dtype
    )

    return jax.pure_callback(
        _scipy_h1, result_shape_dtype, v, z, vmap_method="legacy_vectorized"
    )


spherical_hankel1 = jax.custom_jvp(spherical_hankel1)


@spherical_hankel1.defjvp
def _hv_jvp(primals, tangents):
    """
    The Jacobian-vector product (JVP) for the spherical Hankel function h_v^{(1)}(z).
    Uses the same recurrence relation as for the spherical Bessel function of the first kind.
    At the limit z = 0, the derivative is set to 0 for numerical stability.
    """
    v, z = primals
    _, z_dot = tangents

    v, z = np.asarray(v), np.asarray(z)
    z = z.astype(np.result_type(complex, z.dtype))
    dhv_dz = np.where(
        np.abs(z) < 1e-8,
        np.zeros_like(z),
        v * spherical_hankel1(v, z) / z - spherical_hankel1(v + 1, z),
    )

    return spherical_hankel1(v, z), z_dot * dhv_dz


def _lpmv_impl(m, v, z):
    r"""
    Associated legendre polynomials of real and complex argument
    This function computes the associated Legendre polynomials of cosine of the argument z!
    This is to avoid problems during differentiation of the function,
    so that arguments that will not appear in the physical problems are not considered.

    Args:
        m (float, array_like): Order
        v (float, array_like): Degree
        z (float or complex, array_like): Argument of the cosine

    Returns:
        float or complex

    References:
        - `Wikipedia: Associated Legendre polynomials <https://en.wikipedia.org/wiki/Associated_Legendre_polynomials>`_
    """
    m, v, z = np.asarray(m), np.asarray(v), np.asarray(z)
    assert np.issubdtype(v.dtype, np.integer)
    assert np.issubdtype(m.dtype, np.integer)
    z = z.astype(np.result_type(float, z.dtype))

    def _lpmv(m, v, x):
        return np.where(
            (x > np.pi) | (x < 0.0),
            (-1) ** (m + v) * sp.lpmv(m, v, np.cos(x)),
            sp.lpmv(m, v, np.cos(x)),
        )

    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=np.broadcast_shapes(m.shape, v.shape, z.shape), dtype=z.dtype
    )
    return jax.pure_callback(
        _lpmv, result_shape_dtype, m, v, z, vmap_method="legacy_vectorized"
    )


lpmv = jax.custom_jvp(_lpmv_impl)


@lpmv.defjvp
def _lpmv_jvp(primals, tangents):
    """
    The Jacobian-vector product (JVP) for the associated Legendre polynomials of real and complex argument.
    Note that the argument of the associated Legendre polynomial function is implicitly a cosine of x!

    Recurrence formulas from Wikipedia:
    https://en.wikipedia.org/wiki/Associated_Legendre_polynomials
    (Recurrence formula paragraph, Line 10 from above)
    .. math::
        \sqrt{1 - x^2} \frac{d}{dx} P_{\ell}^{m}(x) =
        \frac{1}{2} \left[ (\ell + m)(\ell - m + 1) P_{\ell}^{m-1}(x) - P_{\ell}^{m+1}(x) \right]

    This expression is modified, because the desired expression is \frac{d}{d theta} P_{\ell}^{m}(cos(theta))
    The final formula is achieved by modifying the left hand side:
    \sqrt{1 - cos(theta)^2} \frac{d}{dx} P_{\ell}^{m}(cos(theta)) = - sin(theta) \frac{d}{d theta} P_{\ell}^{m}(cos(theta)) =
    \frac{cos(theta)}{theta} \frac{d}{d theta} P_{\ell}^{m}(cos(theta)) = \frac{d}{d theta} P_{\ell}^{m}(cos(theta)) ,

    The function  lpmv(m, v, x) calls P_{\ell}^{m}(cos(theta)) under the hood.
    """
    m, v, x = primals
    _, _, x_dot = tangents
    dlpmv_dx = 0.5 * (-(m + v) * (v - m + 1) * lpmv(m - 1, v, x) + lpmv(m + 1, v, x))

    return lpmv(m, v, x), x_dot * dlpmv_dx


def _wignerd_impl(l, m, k, phi, theta, psi):
    r"""
    Wigner-D symbol

    .. math::

        D^l_{mk}(\varphi, \theta, \psi) = \mathrm e^{-\mathrm i m \varphi} d^l_{mk}(\theta) \mathrm e^{-\mathrm i k \psi}

    Note:
        Mathematica use a different sign convention, which means taking the negative
        angles.

    Args:
        l (integer): Degree :math:`l \geq 0`
        m (integer): Order :math:`|m| \leq l`
        k (integer): Order :math:`|k| \leq l`
        phi, theta, psi (float or complex): Angles

    Returns:
        complex

    References:
        - `Wikipedia: Wigner D-matrix <https://en.wikipedia.org/wiki/Wigner_D-matrix>`_
    """
    return np.exp(1j * (phi * m + psi * k)) * sp.wignersmalld(l, m, k, theta)


wignerd = jax.custom_jvp(_wignerd_impl)


@wignerd.defjvp
def _w_jvp(primals, tangents):
    """
    The Jacobian-vector product (JVP) for Wigner D-function.
    Only derivative wrt phi is needed
    """
    l, m, k, phi, theta, psi = primals
    _, _, _, phi_dot, _, _ = tangents
    dv_dphi = 1j * m * wignerd(l, m, k, phi, theta, psi)
    return wignerd(l, m, k, phi, theta, psi), phi_dot * dv_dphi


def _erfc_impl(z):
    z = np.asarray(z)
    #   Require the order  v to be integer type: this simplifies
    #   the JVP rule below.
    #   Promote the input to inexact (float/complex).
    #   Note that np.result_type() accounts for the enable_x64 flag.
    z = z.astype(np.result_type(complex, z.dtype))
    #   Wrap scipy function to return the expected dtype.
    _erfc = lambda z: scipy.special.erfc(z).astype(z.dtype)

    #   Define the expected shape & dtype of output.
    result_shape_dtype = jax.ShapeDtypeStruct(shape=z.shape, dtype=z.dtype)
    return jax.pure_callback(
        _erfc, result_shape_dtype, z, vmap_method="legacy_vectorized"
    )


erfc = jax.custom_jvp(_erfc_impl)


@erfc.defjvp
def erf_jvp(primals, tangents):
    (z,) = primals
    (z_dot,) = tangents
    de_dz = -2 * np.exp(-(z**2)) / np.sqrt(np.pi)
    return erfc(z), z_dot * de_dz


def exp1(z):
    """
    Exponential integral function.
    Arguments:
    z : array_like
        Real or complex argument.
    """
    z = np.asarray(z)
    #   Require the order v to be integer type: this simplifies
    #   the JVP rule below.
    #   Promote the input to inexact (float/complex).
    #   Note that np.result_type() accounts for the enable_x64 flag.
    z = z.astype(np.result_type(complex, z.dtype))
    #   Wrap scipy function to return the expected dtype.
    _exp1 = lambda z: scipy.special.exp1(z).astype(z.dtype)

    #   Define the expected shape & dtype of output.
    result_shape_dtype = jax.ShapeDtypeStruct(shape=z.shape, dtype=z.dtype)

    return jax.pure_callback(
        _exp1, result_shape_dtype, z, vmap_method="legacy_vectorized"
    )


exp1 = jax.custom_jvp(exp1)


@exp1.defjvp
def exp1_jvp(primals, tangents):
    """
    The Jacobian-vector product (JVP) for the exponential integral function.
    To avoid numerical instability, this implementation shifts `x = 0` slightly
    to `1e-8` to ensure a finite derivative
    """
    (z,) = primals
    (z_dot,) = tangents  # Note: v_dot is always 0 because v is integer.
    safe_z = np.where(z == 0, 1e-8, z)  # at z=0 function diverges
    de_dz = -np.exp(-safe_z) / safe_z
    return exp1(z), z_dot * de_dz
