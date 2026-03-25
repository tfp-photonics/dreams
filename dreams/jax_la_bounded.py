import jax
import jax.numpy as np
import numpy as anp
from equinox.internal._loop.bounded import bounded_while_loop
from jax import config, lax
from jax.lax import lgamma as loggamma
from numpy import pi, sqrt
from dreams.jax_recursive import incgamma, _intkambe, _redincgamma, choose_branch
from dreams.jax_coord import car2pol

config.update("jax_enable_x64", True)

M_SQRT2 = sqrt(2.0)


# ---------------------------------------------------------------------------
# Basic geometry helpers
# ---------------------------------------------------------------------------


def area(a, b):
    """(Signed) area between two-vectors a and b."""
    return a[0] * b[1] - a[1] * b[0]


def volume(a, b, c):
    """(Signed) volume between three-vectors a, b, and c."""
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        + a[1] * (b[2] * c[0] - b[0] * c[2])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


# ---------------------------------------------------------------------------
# Incomplete-gamma based helpers
# ---------------------------------------------------------------------------


def zero3d(eta, size_loop):
    r"""
    Value to add for the zero point subtraction in two-dimensional lattices:

    .. math::

        \frac 1 {4\pi} \Gamma\left(-\frac 1 2, -\frac 1 {2\eta^2}\right)

    where `eta` is the cut between real and reciprocal space.
    """
    val = -0.5 / (eta * eta) + 0.0j
    # conjugated because cython took conj value for square root of negative values
    return incgamma(-0.5, val, size_loop) / (4 * pi)


def _check_eta(eta, k, a, ds, dl):
    """
    Choose default eta if eta == 0, depending on dimension dl (1, 2, or 3).
    ds is unused here but kept for signature compatibility.
    """
    if dl == 1:
        res = 1 / (k * anp.abs(a[0, 0]))
    elif dl == 2:
        res = 1 / (k * np.sqrt(np.abs(area(a[0], a[1]))))
    elif dl == 3:
        res = 1 / (k * anp.power(abs(volume(a[0], a[1], a[2])), 2.0 / 3))
    else:
        return anp.nan

    # Limit |res| from below
    res = np.where(abs(res) < 0.125, res * 0.125 / abs(res), res)
    ans = res * sqrt(2 * pi)
    ans = np.where(eta != 0, eta, ans)
    return ans


# ---------------------------------------------------------------------------
# 2D real-space summand
# ---------------------------------------------------------------------------


def _realsw2d(l, m, kr, phi, eta, size_loop):
    r"""
    Summand of the real contribution in a 2D lattice using spherical solutions

    Computes

    .. math::

       (k r)^l \mathrm e^{\mathrm i m \varphi} I_{2l}(k r, \eta)

    with the Kambe integral :math:`I_{2l}`.

    """
    kr = kr.flatten()[0]
    return np.power(kr, l) * _intkambe(2 * l, kr, eta, size_loop) * np.exp(1j * m * phi)


# ---------------------------------------------------------------------------
# Reciprocal lattice helpers
# ---------------------------------------------------------------------------


def recvec2(a):
    """Reciprocal vectors in a two-dimensional lattice."""
    ar = area(a[0], a[1])
    # if ar == 0:
    #     raise ValueError("vectors are linearly dependent")
    ar = 2 * pi / ar
    b00 = a[1][1] * ar
    b01 = -a[1][0] * ar
    b10 = -a[0][1] * ar
    b11 = a[0][0] * ar
    b = np.array([[b00, b01], [b10, b11]])
    return b


# ---------------------------------------------------------------------------
# 2D lattice sums: public entry points
# ---------------------------------------------------------------------------


def lsumsw2d(l, m, k, kpar, a, r, eta, size_loop):
    """2D lattice sum (real + reciprocal)."""
    eta = _check_eta(eta, k, a, 3, 2)
    recsum = recsumsw2d(l, m, k, kpar, a, r, eta, size_loop)
    realsum = realsumsw2d(l, m, k, kpar, a, r, eta, size_loop)
    return recsum + realsum


# @partial(jax.jit)
def lsumsw2d_shift_vmap(ls, ms, ks, kpar, a, rs, eta):
    """
    Vectorized version of lsumsw2d over:
        - positions rs (2 axes),
        - k-values ks,
        - multipole indices ls, ms.

    Shape of result: (len(rs), rs.shape[1], len(ks), len(ls))
    """
    kpar = kpar.flatten()
    size_loop = 25  # anp.max(ls - anp.abs(ms) + 1)

    def func(ii, ji, ki, li):
        return lsumsw2d(
            np.array(ls)[li],
            np.array(ms)[li],
            np.array(ks).flatten()[ki],
            kpar,
            a,
            np.array(rs)[ii, ji].flatten(),
            eta,
            size_loop,
        )

    f1 = jax.vmap(func, in_axes=(None, None, None, 0))
    f2 = jax.vmap(f1, in_axes=(None, None, 0, None))
    f3 = jax.vmap(f2, in_axes=(None, 0, None, None))
    f4 = jax.vmap(f3, in_axes=(0, None, None, None))

    ans = f4(
        np.arange(len(rs)),
        np.arange(rs.shape[1]),
        np.arange(len(ks)),
        np.arange(len(ls)),
    )
    ans = ans.reshape(-1, ans.shape[-1])
    return ans


# ---------------------------------------------------------------------------
# 2D real-space lattice sum
# ---------------------------------------------------------------------------


def realsumsw2d(l, m, k, kpar, a, ri, eta, size_loop):
    """Real-space contribution (2D). Arguments are passed elementwise."""

    kpar = kpar.flatten()

    def main_branch(l, m, k, kpar, a, ri, eta):

        realsum = lax.cond(
            (ri[:2] == 0.0).all(),
            lambda: loop(l, m, k, kpar, a, ri, eta, 1),
            lambda: loop(l, m, k, kpar, a, ri, eta, 0),
        )
        return realsum

    def loop(l, m, k, kpar, a, ri, eta, start):
        realsum = 0.0j
        pprev = np.inf + 0.0j
        prev = pprev
        dim = 2

        def cond_fun(carry):
            point, realsum, prev, pprev, ind_while, ind_for, ind = carry
            cond1 = ind_while[ind] == 0
            cond2 = np.abs(realsum - pprev) < 1e-10
            return (cond1 & cond2) == False

        def body_fun(carry):
            point, realsum, prev, pprev, ind_while, ind_for, ind = carry
            i = ind_for[ind]
            j = ind_while[ind]

            pprev = np.where(j == 0, prev, pprev)
            prev = np.where(j == 0, realsum, prev)

            point = np.where(j == 0, np.ones(2) * (-i), point)
            vec = -1 * a @ point
            coord = np.append(vec, np.array([0.0]))
            coord = coord - ri
            coord = car2pol(coord.flatten())

            realsum = realsum + _realsw2d(
                l, m, k * coord[0], coord[1], eta_loc, size_loop
            ) * np.exp(-1j * np.dot(kpar, vec))

            cond, point = cubeedge_next(point, dim, i)
            return (point, realsum, prev, pprev, ind_while, ind_for, ind + 1)

        loop_tot = 20
        eta_loc = _check_eta(eta, k, a, 3, 2)

        ind_while = anp.array(
            [
                j
                for i in anp.arange(start, loop_tot)
                for j in anp.arange(8 * i + (i == 0))
            ]
        )
        ind_for = anp.array(
            [
                i
                for i in anp.arange(start, loop_tot)
                for _ in anp.arange(8 * i + (i == 0))
            ]
        )
        point = -np.ones(2) * ind_for[0]
        ind = 0

        point, realsum, prev, pprev, ind_while, ind_for, ind = bounded_while_loop(
            cond_fun,
            body_fun,
            (point, realsum, prev, pprev, ind_while, ind_for, ind),
            max_steps=len(ind_for),
        )

        return (
            -1j
            * (-1) ** ((l + m) // 2)
            * np.sqrt((2 * l + 1) * 0.5)
            * realsum
            * np.exp(
                (loggamma(l + m + 1.0) + loggamma(l - m + 1.0)) * 0.5
                - loggamma((l + m) // 2 + 1.0)
                - loggamma((l - m) // 2 + 1.0)
            )
            / (pi * np.power(2, l))
        )

    ans = lax.cond(
        (kpar[:2] == 0.0).all() & (ri[:2] == 0.0).all() & ((m % 2) == 1),
        lambda: 0.0j,
        lambda: lax.cond(
            (l + m) % 2 == 1,
            lambda: np.array(0.0j),
            lambda: lax.cond(
                l < np.abs(m),
                lambda: np.nan * 1j,
                lambda: main_branch(l, m, k, kpar, a, ri, eta),
            ),
        ),
    )
    return ans


# ---------------------------------------------------------------------------
# 2D reciprocal-space lattice sum
# ---------------------------------------------------------------------------


def recsumsw2d(l, m, k, kpar, a, ri, eta, size_loop):

    def main_branch(l, m, k, kpar, a, ri, eta):
        eta_loc = _check_eta(eta, k, a, 3, 2)
        recsum = 0.0j
        prev = np.inf + 0.0j
        pprev = prev
        dim = 2
        b = recvec2(a)

        def cond_fun(carry):
            point, recsum, prev, pprev, ind_while, ind_for, ind = carry
            cond1 = ind_while[ind] == 0
            cond2 = np.abs(recsum - pprev) < 1e-10
            return (cond1 & cond2) == False

        @jax.jit
        def body_fun(carry):
            point, recsum, prev, pprev, ind_while, ind_for, ind = carry
            i = ind_for[ind]
            j = ind_while[ind]

            pprev = np.where(j == 0, prev, pprev)
            prev = np.where(j == 0, recsum, prev)

            point = np.where(j == 0, np.ones(2) * (-i), point)
            pointf = point.copy()

            vec = kpar
            vec = 1 * b @ pointf + 1 * vec
            coord = car2pol(vec.flatten())

            add = _recsw2d(l, m, coord[0] / k, coord[1], eta_loc, size_loop) * np.exp(
                -1j * np.dot(vec, ri[:-1])
            )
            cond, point = cubeedge_next(point, dim, i)
            recsum = recsum + add
            return (point, recsum, prev, pprev, ind_while, ind_for, ind + 1)

        loop_tot = 20
        ind_while = anp.array(
            [j for i in anp.arange(0, loop_tot) for j in anp.arange(8 * i + (i == 0))]
        )
        ind_for = anp.array(
            [i for i in anp.arange(0, loop_tot) for _ in anp.arange(8 * i + (i == 0))]
        )
        point = -np.ones(2) * ind_for[0]
        ind = 0

        point, recsum, prev, pprev, ind_while, ind_for, ind = bounded_while_loop(
            cond_fun,
            body_fun,
            (point, recsum, prev, pprev, ind_while, ind_for, ind),
            max_steps=len(ind_for),
        )
        recsum = recsum * (
            np.sqrt((2 * l + 1) * 0.5)
            * np.power(1j, m)
            * np.exp((loggamma(l + m + 1.) + loggamma(l - m + 1.)) * 0.5)
            / (np.abs(area(a[0], a[1])) * k * k * (2**l))
        )
        recsum  = np.where((l == 0) & (ri == 0.0).all(), recsum + zero3d(eta, size_loop), recsum)        
        return recsum

    ans = lax.cond(
        (kpar[:2] == 0.0).all() & (ri[:2] == 0.0).all() & ((m % 2) == 1),
        lambda: 0.0j,
        lambda: lax.cond(
            (l + m) % 2 == 1,
            lambda: np.array(0.0j),
            lambda: lax.cond(
                l < np.abs(m),
                lambda: np.nan * 1j,
                lambda: main_branch(l, m, k, kpar, a, ri, eta),
            ),
        ),
    )
    return ans


def _recsw2d(l, m, beta, phi, eta, size_loop):
    r"""
    Summand of the reciprocal contribution in a 2D lattice using spherical solutions.
    """

    val = (beta * beta - 1) / (2 * eta * eta)

    def branch_0(l, m, beta, phi, eta):
        n = np.asarray(0.5 - l // 2, dtype=np.float64)
        l2 = np.asarray(l // 2 + 1, dtype=np.float64)
        ans = lax.cond(
            m == 0,
            lambda: incgamma(0.5 - l // 2, val, size_loop)
            * M_SQRT2
            / np.exp(loggamma(l2)),
            lambda: 0.0j,
        )
        return ans

    def branch_1(l, m, beta, phi, eta):
        res = 0.0j
        power = choose_branch(beta, np.power(beta, l))
        macc = power / (
            eta
            * np.exp(loggamma(((l + m) // 2 + 1.0)) + loggamma(((l - m) // 2 + 1.0)))
        )
        mult = 2 * eta * eta / (beta * beta)
        fixed_array = anp.arange(size_loop)
        current = np.where(
            fixed_array < ((l - np.abs(m)) // 2 + 1),
            fixed_array,
            -100,
        )

        def body_fun(carry, n):
            macc, res = carry
            res_new = res + _redincgamma(0.5 - n, val, size_loop) * macc
            res = jax.lax.cond(n == -100, lambda: res, lambda: res_new)
            macc_new = (
                macc
                * mult
                * ((l + m) // 2 - n)
                * ((l - m) // 2 - n)
                / np.array((n + 1), dtype=float)
            )
            macc = np.where(n == -100, macc, macc_new)
            return (macc, res), n

        init_carry = (macc, res)
        (macc, res), _ = lax.scan(body_fun, init_carry, current)
        return res * np.exp(1j * m * phi)

    ans = lax.cond(
        beta == 0.0,
        lambda: branch_0(l, m, beta, phi, eta),
        lambda: branch_1(l, m, beta, phi, eta),
    )
    return ans


# ---------------------------------------------------------------------------
# Cube iteration utilities
# ---------------------------------------------------------------------------


def cubeedge_next(r, d, n):
    """
    Next point on the surface of a d-dimensional cube of side length 2n.

    The array `r` is expected to be `d` long, the initial point to iterate over the whole
    cube is `(-n, -n, ..., -n)`. Returns 0 once `(n, n, ..., n)` is reached.
    """
    if d < 1:
        return 0, r

    def f1(r, d, n):
        cond, r2 = cube_next(r[1:], d - 1, n)
        if r2.shape != (0,):
            r = np.concatenate((np.array([r[0]]), r2))
        ca = lax.cond(cond, lambda: 1, lambda: 0)
        aa = lax.cond(
            cond,
            lambda: r,
            lambda: np.concatenate((np.array([-n]), r[1:])),
        )
        return np.concatenate((np.array([ca]), aa))

    def f2(r, d, n):
        r = np.concatenate((np.array([n]), r[1:]))
        return np.concatenate((np.array([1]), r))

    def f3(r, d, n):
        cond, r2 = cube_next(r[1:], d - 1, n)
        if r2.shape != (0,):
            r = np.concatenate((np.array([r[0]]), r2))
        r = lax.cond(
            cond,
            lambda: r,
            lambda: np.concatenate((np.array([r[0] + 1]), r[1:])),
        )
        return np.concatenate((np.array([1]), r))

    def f4(r, d, n):
        if r.shape[0] > 1:
            cond, r2 = cubeedge_next(r[1:], d - 1, n)
            if r2.shape != (0,):
                r = np.concatenate((np.array([r[0]]), r2))
            if r.shape != (0,):
                r = lax.cond(
                    cond,
                    lambda: r,
                    lambda: np.concatenate((np.array([r[0] + 1]), r[1:])),
                )
        return np.concatenate((np.array([1]), r))

    conds = [(r[0] == n), (d == 1) & (r[0] == -n), (r[0] == -n)]
    res = np.select(
        conds,
        [f1(r, d, n), f2(r, d, n), f3(r, d, n)],
        f4(r, d, n),
    )
    return np.array(res[0], int), np.array(res[1:], float)


def cube_next(r, d, n):
    """
    Next point in a d-dimensional cube of side length 2n.

    The array `r` is expected to be `d` long, the initial point to iterate over the whole
    cube is `(-n, -n, ..., -n)`. Returns 0 once `(n, n, ..., n)` is reached.
    """
    if d < 1:
        return 0, r

    end = r[d - 1]
    cond = end == n
    preend = r[d - 2]
    cond2 = preend == n

    ans_end = np.where(cond, -n, end + 1)
    ind_cond0 = np.where(cond, 0, 1)

    if len(r) == 2:
        ans_preend = np.where(cond & cond2, -n, preend + 1)
        ind_cond1 = np.where(cond & cond2, 0, 1)
        r = np.array([ans_preend, ans_end])
        ans = ind_cond0 * ind_cond1, r
    elif len(r) == 1:
        r = np.array([ans_end])
        ans = ind_cond0, r

    return ans
