#!/usr/bin/env python
import jax
import jax.numpy as np
from jax import vmap, grad, jacrev
from dreams.jax_primitive import spherical_jn, spherical_hankel1


def innermost_interface(l: int, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    # x: (2,2), z: (2,), complex allowed
    sb = spherical_jn(l, x)
    sh = spherical_hankel1(l, x)
    d_jn = vmap(jacrev(spherical_jn, argnums=1, holomorphic=True), in_axes=(None, 0))(
        l, x.flatten()
    ).reshape(x.shape)
    d_h1 = vmap(
        jacrev(spherical_hankel1, argnums=1, holomorphic=True), in_axes=(None, 0)
    )(l, x.flatten()).reshape(x.shape)
    psi = d_jn + sb / x
    chi = d_h1 + sh / x

    z0, z1 = z[0], z[1]
    zs = (z1 + z0) / (2j * z0)
    zd = (z1 - z0) / (2j * z0)

    x10, x11 = x[1, 0], x[1, 1]
    x10_2, x11_2 = x10 * x10, x11 * x11
    c0 = np.array(
        [
            (chi[1, 0] * sb[0, 0] - sh[1, 0] * psi[0, 0]) * zs * x10_2,
            (chi[1, 0] * sb[0, 1] + sh[1, 0] * psi[0, 1]) * zd * x10_2,
        ],
        dtype=np.complex128,
    )

    c1 = np.array(
        [
            (chi[1, 1] * sb[0, 0] + sh[1, 1] * psi[0, 0]) * zd * x11_2,
            (chi[1, 1] * sb[0, 1] - sh[1, 1] * psi[0, 1]) * zs * x11_2,
        ],
        dtype=np.complex128,
    )

    c2 = np.array(
        [
            (-psi[1, 0] * sb[0, 0] + sb[1, 0] * psi[0, 0]) * zs * x10_2,
            (-psi[1, 0] * sb[0, 1] - sb[1, 0] * psi[0, 1]) * zd * x10_2,
        ],
        dtype=np.complex128,
    )

    c3 = np.array(
        [
            (-psi[1, 1] * sb[0, 0] - sb[1, 1] * psi[0, 0]) * zd * x11_2,
            (-psi[1, 1] * sb[0, 1] + sb[1, 1] * psi[0, 1]) * zs * x11_2,
        ],
        dtype=np.complex128,
    )
    return np.stack([c0, c1, c2, c3], axis=1)


def interface_matrix(l: int, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    # x: (2,2), z: (2,), complex allowed
    sb = spherical_jn(l, x)
    sh = spherical_hankel1(l, x)
    d_jn = vmap(jacrev(spherical_jn, argnums=1, holomorphic=True), in_axes=(None, 0))(
        l, x.flatten()
    ).reshape(x.shape)
    d_h1 = vmap(
        jacrev(spherical_hankel1, argnums=1, holomorphic=True), in_axes=(None, 0)
    )(l, x.flatten()).reshape(x.shape)
    psi = d_jn + sb / x
    chi = d_h1 + sh / x

    z0, z1 = z[0], z[1]
    zs = (z1 + z0) / (2j * z0)
    zd = (z1 - z0) / (2j * z0)

    x10, x11 = x[1, 0], x[1, 1]
    x10_2, x11_2 = x10 * x10, x11 * x11

    c0 = np.array(
        [
            (chi[1, 0] * sb[0, 0] - sh[1, 0] * psi[0, 0]) * zs * x10_2,
            (chi[1, 0] * sb[0, 1] + sh[1, 0] * psi[0, 1]) * zd * x10_2,
            (chi[1, 0] * sh[0, 0] - sh[1, 0] * chi[0, 0]) * zs * x10_2,
            (chi[1, 0] * sh[0, 1] + sh[1, 0] * chi[0, 1]) * zd * x10_2,
        ]
    )
    c1 = np.array(
        [
            (chi[1, 1] * sb[0, 0] + sh[1, 1] * psi[0, 0]) * zd * x11_2,
            (chi[1, 1] * sb[0, 1] - sh[1, 1] * psi[0, 1]) * zs * x11_2,
            (chi[1, 1] * sh[0, 0] + sh[1, 1] * chi[0, 0]) * zd * x11_2,
            (chi[1, 1] * sh[0, 1] - sh[1, 1] * chi[0, 1]) * zs * x11_2,
        ]
    )
    c2 = np.array(
        [
            (-psi[1, 0] * sb[0, 0] + sb[1, 0] * psi[0, 0]) * zs * x10_2,
            (-psi[1, 0] * sb[0, 1] - sb[1, 0] * psi[0, 1]) * zd * x10_2,
            (-psi[1, 0] * sh[0, 0] + sb[1, 0] * chi[0, 0]) * zs * x10_2,
            (-psi[1, 0] * sh[0, 1] - sb[1, 0] * chi[0, 1]) * zd * x10_2,
        ]
    )
    c3 = np.array(
        [
            (-psi[1, 1] * sb[0, 0] - sb[1, 1] * psi[0, 0]) * zd * x11_2,
            (-psi[1, 1] * sb[0, 1] + sb[1, 1] * psi[0, 1]) * zs * x11_2,
            (-psi[1, 1] * sh[0, 0] - sb[1, 1] * chi[0, 0]) * zd * x11_2,
            (-psi[1, 1] * sh[0, 1] + sb[1, 1] * chi[0, 1]) * zs * x11_2,
        ]
    )
    return np.stack([c0, c1, c2, c3], axis=1)


def layer_params(i, x, eps, mu, kappa):
    nr_in, nr_out = np.sqrt(eps[i] * mu[i]), np.sqrt(eps[i + 1] * mu[i + 1])
    z_in, z_out = np.sqrt(mu[i] / eps[i]), np.sqrt(mu[i + 1] / eps[i + 1])
    kap_in, kap_out = kappa[i], kappa[i + 1]
    xi = x[i]
    x_2x2 = np.array(
        [
            [xi * (nr_in - kap_in), xi * (nr_in + kap_in)],
            [xi * (nr_out - kap_out), xi * (nr_out + kap_out)],
        ],
        dtype=np.result_type(x, eps, mu, kappa),
    )
    z_pair = np.array([z_in, z_out], dtype=np.result_type(eps, mu))
    return x_2x2, z_pair


def mie_one_l(l: int, k0: float, radii, eps, mu, kappa):
    """
    radii: (n,)       # radii from inner to outer shell boundaries
    eps, mu, kappa: (n+1,)  # materials from core .. embedding
    returns: (2,2) T_l in helicity basis
    """
    x = (k0 * np.asarray(radii)).astype(np.complex128)  # (n,)
    # eps = anp.asarray(eps); mu = anp.asarray(mu); kappa = anp.asarray(kappa)
    eps = np.asarray(eps)
    mu = np.asarray(mu)
    kappa = np.asarray(kappa)
    n = x.shape[0]
    # i = 0  → innermost interface → initial 4x2 'state'
    x0, z0 = layer_params(0, x, eps, mu, kappa)
    state = innermost_interface(l, x0, z0).T  # (4,2)

    # i = 1..n-1 → general 4x4 interfaces
    def body(state, i):
        xi, zi = layer_params(i, x, eps, mu, kappa)
        M = interface_matrix(l, xi, zi).T  # (4,4)
        return M @ state, None

    state, _ = jax.lax.scan(body, state, np.arange(1, n))
    # split and solve: A^{-1} B
    A = state[:2, :].T  # (2,2)  (top rows)
    B = state[2:, :].T  # (2,2)  (bottom rows)
    T = np.linalg.solve(A, B)  # (2,2)
    return T


def mie(l, mu_a, epsilon, r, k0):
    """
    Compute Mie coefficients a_l, b_l in the parity basis for a homogeneous sphere.

    Args:
        l (int or array_like of int):
            Multipole orders.
        mu_a (array_like, shape (2,)):
            Relative permeabilities (mu_medium, mu_sphere).
        epsilon (array_like, shape (2,)):
            Relative permittivities (eps_sphere, eps_medium).
        r (float or array_like):
            Sphere radius. Broadcasts against `l`.
        k0 (float):
            Vacuum wavenumber.

    Returns:
        jax.Array:
            Flattened coefficients of shape (2 * N,), where N = len(l).
            For each entry l[i], the output is ordered as
            coeffs[2*i]   = a_l[i]
            coeffs[2*i+1] = b_l[i].
    """
    l = np.asarray([l]).flatten()
    r = np.asarray([r]).flatten().astype("float64")

    epsilon_m = epsilon[1]  # medium
    epsilon_p = epsilon[0]  # particle

    rho = (k0 * r * epsilon_m**0.5 + 0.0j).astype(
        complex
    )  # medium       # COMPLEX TO AVOID NANS for bessel & hankel
    rho1 = (k0 * r * epsilon_p**0.5 + 0.0j).astype(complex)
    mu = mu_a[0]
    mu1 = mu_a[1]
    an_n = mu * epsilon_p * (
        rho
        * vmap(jacrev(spherical_jn, argnums=1, holomorphic=True), in_axes=(0, 0))(
            l, rho
        )
        + spherical_jn(l, rho)
    ) * spherical_jn(l, rho1) - mu1 * epsilon_m * (
        rho1
        * vmap(jacrev(spherical_jn, argnums=1, holomorphic=True), in_axes=(0, 0))(
            l, rho1
        )
        + spherical_jn(l, rho1)
    ) * spherical_jn(
        l, rho
    )
    an_d = mu * epsilon_p * (
        rho
        * vmap(grad(spherical_hankel1, argnums=1, holomorphic=True), in_axes=(0, 0))(
            l, rho
        )
        + spherical_hankel1(l, rho)
    ) * vmap(spherical_jn, in_axes=(0, 0))(l, rho1) - mu1 * epsilon_m * (
        rho1
        * vmap(grad(spherical_jn, argnums=1, holomorphic=True), in_axes=(0, 0))(l, rho1)
        + spherical_jn(l, rho1)
    ) * spherical_hankel1(
        l, rho
    )

    an = an_n / an_d
    bn_n = mu1 * (
        rho
        * vmap(grad(spherical_jn, argnums=1, holomorphic=True), in_axes=(0, 0))(l, rho)
        + spherical_jn(l, rho)
    ) * spherical_jn(l, rho1) - mu * (
        rho1
        * vmap(grad(spherical_jn, argnums=1, holomorphic=True), in_axes=(0, 0))(l, rho1)
        + spherical_jn(l, rho1)
    ) * spherical_jn(
        l, rho
    )
    bn_d = mu1 * (
        rho
        * vmap(grad(spherical_hankel1, argnums=1, holomorphic=True), in_axes=(0, 0))(
            l, rho
        )
        + spherical_hankel1(l, rho)
    ) * spherical_jn(l, rho1) - mu * (
        rho1
        * vmap(grad(spherical_jn, argnums=1, holomorphic=True), in_axes=(0, 0))(l, rho1)
        + spherical_jn(l, rho1)
    ) * spherical_hankel1(
        l, rho
    )
    bn = bn_n / bn_d
    return np.vstack((an, bn)).T.flatten()


def fresnel(ks, kzss, zs):
    r"""fresnel(ks, kzs, zs)

    Fresnel coefficient for a planar interface.

    The first dimension contains the numbers for the two media, the second dimenison
    indexes the polarizations.

    The result is an array relating incoming light of negative (index `0`) and
    positive (index `1`) helicity with the outgoing modes, which are indexed in the same
    way. The first dimension of the array are the outgoing and the second dimension the
    incoming modes.

    Args:
        ks (float or complex): Wave numbers
        kzss (float): Z component of the waves
        zs (float or complex): Impedances

    Returns:
        complex (2, 2, 2, 2)-array
    """
    final = []
    for _i, kzs in enumerate(kzss):
        ap = [
            ks[0][0] * kzs[0][1] + ks[0][1] * kzs[0][0],
            ks[1][0] * kzs[1][1] + ks[1][1] * kzs[1][0],
        ]
        bp = ks[0][1] * kzs[1][1] + ks[1][1] * kzs[0][1]
        cp = ks[0][0] * kzs[1][0] + ks[1][0] * kzs[0][0]
        am = [
            ks[0][0] * kzs[0][1] - ks[0][1] * kzs[0][0],
            ks[1][0] * kzs[1][1] - ks[1][1] * kzs[1][0],
        ]

        zs_diff = zs[0] - zs[1]
        zs_prod = 4 * zs[0] * zs[1]
        pref = 1 / (zs_diff * zs_diff * ap[0] * ap[1] + zs_prod * bp * cp)
        a000 = [
            (zs[0] + zs[1]) * ks[1][0] * kzs[0][0] * bp * zs[1] * 4 * pref,
            -(
                (zs[0] - zs[1])
                * ks[1][0]
                * kzs[0][1]
                * (ks[1][1] * kzs[0][0] - ks[0][0] * kzs[1][1])
            )
            * zs[1]
            * 4
            * pref,
        ]
        a001 = [
            -(
                (zs[0] - zs[1])
                * ks[1][1]
                * kzs[0][0]
                * (ks[1][0] * kzs[0][1] - ks[0][1] * kzs[1][0])
            )
            * zs[1]
            * 4
            * pref,
            (zs[0] + zs[1]) * ks[1][1] * kzs[0][1] * cp * zs[1] * 4 * pref,
        ]
        a010 = [
            (
                -zs_diff * zs_diff * am[1] * ap[0]
                - zs_prod * bp * (ks[1][0] * kzs[0][0] - ks[0][0] * kzs[1][0])
            )
            * pref,
            -(2 * (zs[0] * zs[0] - zs[1] * zs[1]) * ks[1][0] * kzs[1][1] * ap[0])
            * pref,
        ]
        a011 = [
            -(2 * (zs[0] * zs[0] - zs[1] * zs[1]) * ks[1][1] * kzs[1][0] * ap[0])
            * pref,
            (
                zs_diff * zs_diff * am[1] * ap[0]
                - zs_prod * cp * (ks[1][1] * kzs[0][1] - ks[0][1] * kzs[1][1])
            )
            * pref,
        ]
        a100 = [
            (
                -zs_diff * zs_diff * am[0] * ap[1]
                - zs_prod * bp * (ks[0][0] * kzs[1][0] - ks[1][0] * kzs[0][0])
            )
            * pref,
            -(2 * (zs[1] * zs[1] - zs[0] * zs[0]) * ks[0][0] * kzs[0][1] * ap[1])
            * pref,
        ]
        a101 = [
            -(2 * (zs[1] * zs[1] - zs[0] * zs[0]) * ks[0][1] * kzs[0][0] * ap[1])
            * pref,
            (
                zs_diff * zs_diff * am[0] * ap[1]
                - zs_prod * cp * (ks[0][1] * kzs[1][1] - ks[1][1] * kzs[0][1])
            )
            * pref,
        ]
        a110 = [
            (zs[0] + zs[1]) * ks[0][0] * kzs[1][0] * bp * zs[0] * 4 * pref,
            -(
                (zs[1] - zs[0])
                * ks[0][0]
                * kzs[1][1]
                * (ks[0][1] * kzs[1][0] - ks[1][0] * kzs[0][1])
            )
            * zs[0]
            * 4
            * pref,
        ]
        a111 = [
            -(
                (zs[1] - zs[0])
                * ks[0][1]
                * kzs[1][0]
                * (ks[0][0] * kzs[1][1] - ks[1][1] * kzs[0][0])
            )
            * zs[0]
            * 4
            * pref,
            (zs[0] + zs[1]) * ks[0][1] * kzs[1][1] * cp * zs[0] * 4 * pref,
        ]
        res = np.array([[[a000, a001], [a010, a011]], [[a100, a101], [a110, a111]]])

        final.append(res)
    final = np.stack(final, axis=0)

    return final
