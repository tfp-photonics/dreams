from functools import partial
from jax import config, lax
import jax
from jax.lax import lgamma as loggamma
import jax.numpy as np
import numpy as anp
from numpy import pi, sqrt, real, imag
from dreams.jax_primitive import exp1
from dreams.jax_primitive import erfc

config.update("jax_enable_x64", True)
SQPI = sqrt(pi)
M_SQRT1_2 = sqrt(1 / 2)
M_SQRT2 = sqrt(2)


def _redincgamma(n, z, size_loop):
    r"""
    Reduced incomplete gamma function

    Returns

    .. math::

        \frac{\Gamma(n, z)}{(-z)^n}

    for integer and half-integer n. Branch for negative values is taken below
    the real axis.
    """
    singularity = 1e-7  # Value of the singularity: smaller is stronger

    def branch_0(n, z):
        ans = np.piecewise(
            n + 0.0j,
            [
                2 * n == 0,
                (2 * n > 0) & ((2 * n + 2) % 4 != 0),
            ],
            [
                lambda n: -anp.exp(0) - anp.log(singularity) + 0.5j * pi,
                lambda n: jax.scipy.special.gamma(n.real)
                * (1 + np.exp(-1j * pi * n))
                / (2 * singularity),
                lambda n: np.array(0.0j),
            ],
        )
        return ans - np.power(-1.0j, 2 * n) / n

    def branch_1(n, z):
        ans = incgamma(n, z, size_loop)
        power = np.power(-1/z - 0.j , n)
        power = choose_branch(-1/z, power)
        return ans * power

    # Use magnitude of z to choose branch
    ans = lax.cond(
        (real(z) * real(z) + imag(z) * imag(z)) < 1e-12,
        lambda: branch_0(n, z),
        lambda: branch_1(n, z),
    )
    return ans


def choose_branch(z, f):
    re = z.real
    im = z.imag
    on_neg_real = (re <= 0) & np.isclose(im, 0.0)
    sqr = np.where(on_neg_real, np.conj(f), f)
    return sqr


# @partial(jax.jit, static_argnums=(2,))
def incgamma(n, z, size_loop):
    r"""
    incgamma(n, z)

    Upper incomplete Gamma function of integer and half-integer degree and real and complex
    argument

    This function is defined as

    .. math'

        \Gamma(n, z) = \int_z^\infty t^{n - 1} \mathrm e^{-t} \mathrm dt

    The negative real axis is the branch cut of the implemented function.

    References:
        - `DLMF: 8.2 <https://dlmf.nist.gov/8.2>`
    Args:
        n (float or array_like): Order (integer or half-integer).
        z (complex or array_like): Argument.
        size_loop (int): Maximum number of recursion steps for the scan.

    Returns:
        complex: Γ(n, z) evaluated at the given order and argument.
    """
    z = z - 0.0j

    def down(n):
        sq = np.sqrt(z - 0.0j)
        sqr = choose_branch(z, sq)
        init = np.where(
            (np.array(2 * n).astype(float) % 2) == 0,
            np.array([0, choose_branch(z, exp1(z))]),
            np.array([0.5, sqrt(pi) * erfc(sqr)]),
        )

        @jax.jit
        def body_fun(value, i):
            ind = init[0] - i
            power = choose_branch(z, np.power(z, ind - 1))
            value2 = (value - power * np.exp(-z)) / (ind - 1)
            ans = jax.lax.cond(i == -100, lambda: value, lambda: value2)
            return ans, i

        end_scan = np.abs(n - init[0]).astype(int)
        fixed_array = np.arange(size_loop)
        current = np.where((fixed_array < end_scan), fixed_array, -100)
        ans, _ = lax.scan(body_fun, init[1], current)
        return ans

    def up(n):
        sq = np.sqrt(z - 0.0j)
        sqr = choose_branch(z, sq)
        init = np.where(
            np.array((2 * n).astype(float) % 2 == 0.0),
            np.array([1, np.exp(-z)]),
            np.array([0.5, sqrt(pi) * erfc(sqr)]),
        )

        @jax.jit
        def body_fun(value, i):
            ind = init[0] + i
            power = choose_branch(z, np.power(z, ind))
            value2 = ind * value + power * np.exp(-z)
            ans = np.where(i == -1, value, value2)
            return ans, i

        end_scan = np.abs(n - init[0]).astype(int)
        fixed_array = np.arange(size_loop)
        current = np.where((fixed_array < end_scan), fixed_array, -1)
        ans, _ = lax.scan(body_fun, init[1], current)
        return ans

    twicen = 2 * n

    def branch_0(z):
        ans = lax.cond(
            twicen <= 0,
            lambda: complex(np.inf),
            lambda: (np.exp(loggamma(n + 0.0))).astype(complex),
        )
        return ans

    def branch_1(z):
        def opt1():
            ans = exp1(np.array(z))
            ans = choose_branch(z, ans)
            return ans

        def opt2():
            ans = np.exp(-z)
            return ans

        def opt3():
            sq = np.sqrt(z - 0.0j)
            sqr = choose_branch(z, sq)
            return sqrt(pi) * erfc(sqr)

        conds = [(2 * n) == 0, (2 * n) == 2, (2 * n) == 1, (2 * n) > 2]
        ans = np.piecewise(
            n + 0.0j,
            conds,
            [
                lambda n: opt1(),
                lambda n: opt2(),
                lambda n: opt3(),
                lambda n: up(n),
                lambda n: down(n),
            ],
        ).astype(complex)
        return ans

    ans = lax.cond(z == 0.0, lambda: branch_0(z), lambda: branch_1(z))
    return ans


@partial(jax.jit, static_argnums=(3,))
def _intkambe(n, z, eta, size_loop):
    """
    Kambe integral :math:`I_n(z, \eta)` for integer order ``n``.

    This implements the recurrence relations for the integrals introduced by
    K. Kambe, using special-case closed forms for orders
    :math:`n = -3, -2, -1, 0, 1, 2, 3` and upward / downward recursion
    outside this range.


    Args:
        n (int): Order of the integral.
        z (float or complex): Argument :math:`z`.
        eta (float or complex): Cutoff parameter :math:`\eta`.
        size_loop (int): Maximum number of recursion iterations in the
            upward / downward schemes.

    Returns:
        complex: Value of :math:`I_n(z, \eta)`.
    References:
        .. [ ] `K. Kambe, Zeitschrift Fuer Naturforschung A 23, 9 (1968). <https://doi.org/10.1515/zna-1968-0908>`
    """

    def z_0(n, z, eta):
        conds = [
            np.array(n).astype(int) == -4,
            np.array(n).astype(int) == -3,
            np.array(n).astype(int) > -2,
            np.array(n).astype(int) == -2,
        ]
        funcs = [
            lambda n: -_intkambe_m2(z, eta) + np.exp(0.5 / (eta * eta)) / eta,
            lambda n: np.exp(1 / (2 * eta * eta)) - 1,
            lambda n: np.array(np.inf),  # np.array(10+0.j), # np.array(np.inf), # #
            lambda n: _intkambe_m2(z, eta),
            lambda n: down(n),
        ]
        res = np.piecewise(np.array(n).astype(complex), conds, funcs)
        return res.astype(complex)

    # @jax.jit
    def down(n):
        init = np.where(
            (n.astype(float) % 2) == 0,
            np.array([-2, nm2(), 0, n0()]),
            np.array([-3, nm3(), -1, nm1()]),
        )

        @jax.jit
        def body_fun(vals, i):
            ind2 = init[2] - 2 * i  # was init[0]
            ind = ind2 - 2
            value, value2 = vals
            value_new = (
                (ind + 1) * value
                - z * z * value2
                + np.power(eta, ind + 1)
                * np.exp((-z * z * eta * eta + 1 / (eta * eta)) * 0.5)
            )
            next_val = np.where(i == -1, value, value_new)
            value = np.where(i == -1, value2, value)
            return (next_val, value), i

        end_scan = ((init[0] - n).astype(int) // 2).astype(int)
        fixed_array = np.arange(size_loop)
        current = np.where((fixed_array < end_scan), fixed_array, -1)
        init_carry = (init[1], init[3])
        (res0, res1), _ = lax.scan(body_fun, init_carry, current)
        return res0

    def n0():
        ans = _intkambe_0(z, eta)
        return ans

    def nm1():
        ans = _intkambe_m1(z, eta, size_loop)
        return ans

    def nm2():
        ans = _intkambe_m2(z, eta)
        return ans

    def nm3():

        ans = _intkambe_m3(z, eta)
        return ans

    def np1():
        ans = (
            np.exp((-z * z * eta * eta + 1 / (eta * eta)) * 0.5) - _intkambe_m3(z, eta)
        ) / (z * z)
        return ans

    def np2():
        ans = (
            n0()
            - _intkambe_m2(z, eta)
            + eta * np.exp((-z * z * eta * eta + 1 / (eta * eta)) * 0.5)
        ) / (z * z)
        return ans

    def np3():
        ans = (
            2 * np1()
            - nm1()
            + eta * eta * np.exp((-z * z * eta * eta + 1 / (eta * eta)) * 0.5)
        ) / (z * z)
        return ans

    def up(n):
        init = np.where(
            (np.array(n).astype(float) % 2) == 0,
            np.array([0, n0(), 2, np2()]),
            np.array([1, np1(), 3, np3()]),
        )

        @jax.jit
        def body_fun(vals, i):
            ind = init[0] + 2 * i
            value, value2 = vals
            value_new = (
                (ind + 3) * value2
                - value
                + np.power(eta, ind + 3)
                * np.exp((-z * z * eta * eta + 1 / (eta * eta)) * 0.5)
            ) / (z * z)
            next_val = np.where(i == -1, value2, value_new)
            value2 = np.where(i == -1, value, value2)
            return (value2, next_val), i

        end_scan = ((n - init[2])).astype(int) // 2
        fixed_array = anp.arange(size_loop)
        current = np.where((fixed_array < end_scan), fixed_array, -1)
        init_carry = (init[1], init[3])
        (res0, res1), _ = lax.scan(body_fun, init_carry, current)
        return res1

    @jax.jit
    def z_diff(n, z, eta):
        conds = [
            n == -3,
            n == -2,
            n == -1,
            n == 0,
            n == 1,
            n == 2,
            n == 3,
            n < -3,
            n > 3,
        ]
        funcs = [
            lambda n: nm3(),
            lambda n: nm2(),
            lambda n: nm1(),
            lambda n: n0(),
            lambda n: np1(),
            lambda n: np2(),
            lambda n: np3(),
            lambda n: down(n),
            lambda n: up(n),
        ]
        res = np.piecewise(n.astype(complex), conds, funcs)
        return res.astype(complex)

    ans = lax.cond(z == 0.0, lambda: z_0(n, z, eta), lambda: z_diff(n, z, eta))
    ans = np.where(eta == 0, (np.inf + 0.0j), ans)  # in case eta is 1j/ kzeta
    return ans


@jax.jit
def _intkambe_m3(z, eta):
    r"""Kambe integral of order -3

    Implements]

    .. math::

       I_{-3}(z, \eta) = \sum_{n = 0}^\infty \frac{1}{n!}
       \left(\frac{z^2}{4}\right)^{n + 1}
       \Gamma(-n - 1, \frac{z^2 \eta^2}{2})
    """
    acc = 0
    mult = z * z * 0.25
    macc = mult

    @jax.jit
    def body_fun(mu, init):
        macc, acc = init
        acc = acc + macc * incgamma(-1 - mu, z * z * eta * eta * 0.5, 21)
        macc = macc * mult / (mu + 1)
        return (macc, acc)

    acc = lax.fori_loop(0, 21, body_fun, (macc, acc))[1]
    return acc


@partial(jax.jit, static_argnums=(2,))
def _intkambe_m1(z, eta, size_loop):
    r"""Kambe integral of order -1

    Implements

    .. math::

       I_{-1}(z, \eta) = \frac{1}{2} \sum_{n = 0}^\infty \frac{1}{n!}
       \left(\frac{z^2}{4}\right)^{n}
       \Gamma(-n, \frac{z^2 \eta^2}{2})
    """
    acc = 0.0j
    mult = z * z * 0.25
    macc = 0.5

    @jax.jit
    def body_fun(mu, init):
        macc, acc = init
        acc = acc + macc * incgamma(-mu, z * z * eta * eta * 0.5, size_loop)
        macc = macc * mult / (mu + 1)
        return (macc, acc)

    acc = lax.fori_loop(0, 21, body_fun, (macc, acc))[1]
    return acc


@jax.jit
def _cintkambe_m2(z, eta):
    """Kambe integral of order -2 for complex double"""
    # if np.real(z) < 0:
    #     z = -z

    z = np.where(np.real(z) < 0, -z, z)
    faddp = erfc((z * eta - 1j / eta) * M_SQRT1_2) * np.exp(-1j * z)
    faddm = erfc((z * eta + 1j / eta) * M_SQRT1_2) * np.exp(1j * z)
    return -0.5j * (faddp - faddm) * SQPI * M_SQRT1_2


@jax.jit
def _dintkambe_m2(z, eta):
    """Kambe integral of order -2 for double"""
    z = np.where(z < 0, -z, z)
    faddp = np.imag(erfc((z * eta - 1j / eta) * M_SQRT1_2) * np.exp(-1j * z))
    faddm = np.imag(erfc((z * eta + 1j / eta) * M_SQRT1_2) * np.exp(1j * z))
    return 0.5 * (faddp - faddm) * SQPI * M_SQRT1_2


@jax.jit
def _intkambe_m2(z, eta):
    """Kambe integral of order 0 for fused type"""
    return lax.cond(
        type(z) == float,
        lambda: _dintkambe_m2(z, eta).astype(complex),
        lambda: _cintkambe_m2(z, eta),
    )


@jax.jit
def _intkambe_0(z, eta):
    """Kambe integral of order 0 for fused type"""
    return lax.cond(
        type(z) == float,
        lambda: _dintkambe_0(z, eta).astype(complex),
        lambda: _cintkambe_0(z, eta),
    )


@jax.jit
def _dintkambe_0(z, eta):
    z = np.where(real(z) < 0, -z, z)
    faddp = np.real(erfc((z * eta - 1j / eta) * M_SQRT1_2) * np.exp(-1j * z))
    faddm = np.real(erfc((z * eta + 1j / eta) * M_SQRT1_2) * np.exp(1j * z))
    ans = (faddp + faddm) * SQPI * 0.5 * M_SQRT1_2 / z
    return ans


@jax.jit
def _cintkambe_0(z, eta):
    z = np.where(real(z) < 0, -z, z)
    faddp = erfc((z * eta - 1j / eta) * M_SQRT1_2) * np.exp(-1j * z)
    faddm = erfc((z * eta + 1j / eta) * M_SQRT1_2) * np.exp(1j * z)
    ans = (faddp + faddm) * SQPI * 0.5 * M_SQRT1_2 / z
    return ans
