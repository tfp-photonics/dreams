#!/usr/bin/env python
import numpy as anp
import treams
import treams.lattice as la
from numpy import arctan2, pi, sqrt
from numpy import power as pow
from scipy.special import loggamma
from treams import pw
import jax.numpy as np
from jax import config
from dreams.jax_la_bounded import lsumsw2d_shift_vmap
from dreams.jax_coeffs import fresnel
from dreams.jax_op import pw_translate
from dreams.jax_waves import pi_fun, tau_fun
from dreams.jax_misc import (
    refractive_index,
    defaultmodes,
    sdefaultmodes,
    wave_vec_z,
    wigner3j,
)

config.update("jax_enable_x64", True)


def interface(eps1, eps2, k0, modes, mu=1, kappa=0):
    """Planar interface between two media.

    Args:
        basis (PlaneWaveBasisByComp): Basis definitions.
        k0 (float): Wave number in vacuum
        materials (Sequence[Material]): Material definitions.
        poltype (str, optional): Polarization type (:ref:`params:Polarizations`).

    Returns:
        SMatrix
    """

    kx = modes[0]
    ky = modes[1]
    pol = modes[2]
    choice = pol == 0
    eps = np.array([eps1, eps2])
    kxs = kx[choice], kx[~choice]
    kys = ky[choice], ky[~choice]
    qs = np.zeros((2, 2, len(kx), len(kx)), complex)
    zs = np.sqrt(mu / eps)
    pls = np.array([[0, 1]])
    ks = np.array([k0 * refractive_index(e, mu, kappa) for e in eps])
    if all(kxs[0] == kxs[1]) and all(kys[0] == kys[1]):
        kzs = np.stack(
            [
                wave_vec_z(
                    kxs[0][:, None],
                    kys[0][:, None],
                    (k0 * refractive_index(e, mu, kappa))[pls],
                )
                for e in eps
            ],
            -2,
        )
        vals = fresnel(ks, kzs, zs)
        qs = qs.at[:, :, choice, choice].set(np.moveaxis(vals[:, :, :, 0, 0], 0, -1))
        qs = qs.at[:, :, choice, ~choice].set(np.moveaxis(vals[:, :, :, 0, 1], 0, -1))
        qs = qs.at[:, :, ~choice, choice].set(np.moveaxis(vals[:, :, :, 1, 0], 0, -1))
        qs = qs.at[:, :, ~choice, ~choice].set(np.moveaxis(vals[:, :, :, 1, 1], 0, -1))
    else:
        for i, (kx, ky, pol) in enumerate(modes):
            for ii, (kx2, ky2, pol2) in enumerate(modes):
                if kx != kx2 or ky != ky2:
                    continue
                kzs = np.array(
                    [
                        wave_vec_z(kx, ky, k0 * refractive_index(e, mu, kappa)[pol])
                        for e in eps
                    ]
                )
                vals = fresnel(ks, kzs, zs)
                qs = qs.at[:, :, ii, i].set(vals[:, :, pol2, pol])
    return qs


def propagation(epsilon, r, k0, modes, mu=1, kappa=0, modetype="up"):
    """S-matrix for the propagation along a distance.

    This S-matrix translates the reference origin along `r`.

    Args:
        r (float, (3,)-array): Translation vector.
        k0 (float): Wave number in vacuum.
        basis (PlaneWaveBasis): Basis definition.
        material (Material, optional): Material definition.
        poltype (str, optional): Polarization type (:ref:`params:Polarizations`).

    Returns:
        SMatrix
    """
    kx, ky = modes[:-1]
    pol = modes[-1]
    ks = k0 * refractive_index(epsilon, mu, kappa)[modes[-1]]
    kzs = wave_vec_z(kx, ky, ks, nondifsum=True) * (2 * (modetype == "up") - 1)
    where = (
        (np.abs(kx[:, None] - kx[None, :]) < 1e-14)
        & (np.abs(ky[:, None] - ky[None, :]) < 1e-14)
        & (np.abs(kzs[:, None] - kzs[None, :]) < 1e-14)
        & (pol[:, None] == pol)
    )
    sup = pw_translate(
        kx,
        ky,
        kzs,
        r[..., None, None, 0],
        r[..., None, None, 1],
        r[..., None, None, 2],
        where=where,
    )

    sdown = pw_translate(
        kx,
        ky,
        -kzs,
        -r[..., None, None, 0],
        -r[..., None, None, 1],
        -r[..., None, None, 2],
        where=where,
    )
    zero = np.zeros_like(sup)
    res = np.array([[sup, zero], [zero, sdown]])
    return res


def stack(items):
    q = items[0]
    for i, _item in enumerate(items[:-1]):
        q = add(q, items[i + 1])
    return q


def add(scur, sadd, check_materials=True, check_modes=False):
    """
    Couple another S-matrix on top of the current one

    Args:
        items (SMatrix): S-matrices in their intended order
        check_materials (bool, optional): Check for matching material parameters
            at each S-matrix
        check_materials (bool, optional): Check for matching modes at each Q-matrix

    Returns:
        SMatrix
    """

    scur = np.array(scur, dtype="complex128")
    sadd = np.array(sadd, dtype="complex128")
    dim = scur.shape[2]
    s_tmp = np.linalg.solve(
        np.eye(dim, dtype="complex128") - scur[0, 1, :, :] @ sadd[1, 0, :, :],
        scur[0, 0, :, :],
    )

    s00 = sadd[0, 0, :, :] @ s_tmp
    s10 = scur[1, 0, :, :] + scur[1, 1, :, :] @ sadd[1, 0, :, :] @ s_tmp
    s_tmp = np.linalg.solve(
        np.eye(dim, dtype="complex128") - sadd[1, 0, :, :] @ scur[0, 1, :, :],
        sadd[1, 1, :, :],
    )
    s11 = scur[1, 1, :, :] @ s_tmp
    s01 = sadd[0, 1, :, :] + sadd[0, 0, :, :] @ scur[0, 1, :, :] @ s_tmp

    sc1 = np.stack((s00, s01), axis=0)
    sc2 = np.stack((s10, s11), axis=0)
    snew = np.stack((sc1, sc2), axis=0)
    return snew


def array2dt(
    t,
    lmax,
    helicity,
    kx,
    ky,
    kz,
    pwpol,
    a,
    epsilon_medium,
    origin=None,
    eta=0.0,
    positions=None,
    nondifsum=True,
):
    """
    Convert a two-dimensional array of T-matrices into an S-matrix

    Unlike for the 1d-case there is no local S-matrix used, so the result is taken
    with respect to the reference origin.

    Args:
        kx (float, array_like): X component of the plane wave
        ky (float, array_like): Y component of the plane wave
        kz (float, array_like): Z component of the plane wave
        pwpol (int, array_like): Plane wave polarizations
        a (float, (2,2)-array): Lattice vectors
        origin (float, (3,)-array, optional): Reference origin of the result
        eta (float or complex, optional): Splitting parameter in the lattice summation

    Returns:
        complex, array

    """
    if positions is not None:
        num = len(positions)
    else:
        num = 1
        positions = anp.zeros((1, 3))
    modes = defaultmodes(lmax, num)

    if len(modes) == 4:
        pidx, l, m, pol = modes
    if len(modes) == 3:
        l, m, pol = modes
        pidx = anp.zeros_like(l)
        modes = anp.array([pidx, l, m, pol])
    if nondifsum:
        kpar = firstbrillouin2d(
            anp.array([kx[0], ky[0]]), la.reciprocal(a), nondifsum=nondifsum
        )
        kpar = anp.reshape(kpar, (-1, 2))
        ks = anp.sqrt(
            kx.flatten()[:2] * kx.flatten()[:2]
            + ky.flatten()[:2] * ky.flatten()[:2]
            + kz.flatten()[:2] * kz.flatten()[:2]
        )
    else:
        kpar = firstbrillouin2d(np.array([kx[0], ky[0]]), la.reciprocal(a), nondifsum)
        kpar = kpar.reshape((-1, 2))
        ks = np.sqrt(
            kx.flatten()[:2] * kx.flatten()[:2]
            + ky.flatten()[:2] * ky.flatten()[:2]
            + kz.flatten()[:2] * kz.flatten()[:2]
        )
    interaction = np.linalg.solve(
        latticecoupling(
            ks, kpar, a, positions, modes, helicity, t, eta, nondifsum=nondifsum
        ),
        t,
    )

    if origin is None:
        origin = anp.zeros((3,))
    posdiff = positions - origin

    tout = pw_translate(
        kx[:, None],
        ky[:, None],
        kz[:, None],
        -posdiff[pidx, 0],
        -posdiff[pidx, 1],
        -posdiff[pidx, 2],
    )
    ain = illuminate_pw(
        kx,
        ky,
        kz,
        anp.array(pwpol),
        anp.array(pidx),
        helicity,
        positions,
        modes,
        nondifsum,
    )
    pout = periodic_to_pw(
        kx[:, None],
        ky[:, None],
        kz[:, None],
        pwpol[:, None],
        *modes[1:],
        la.area(a),
        helicity=helicity,
        nondifsum=nondifsum,
    )
    sca = tout * pout
    ans = sca @ interaction @ ain
    return ans


def _check_modes(modes):
    """_check_modes"""
    if len(modes) < 3 or len(modes) > 4:
        raise ValueError(
            f"invalid length of variable modes {len(modes)}, must be 3 or 4"
        )
    modes = (*(np.array(a) for a in modes),)
    if len(modes) == 3:
        modes = (np.zeros_like(modes[0]),) + modes
    if not np.all([m.ndim == 1 for m in modes]):
        raise ValueError("invalid dimensions of modes")
    if not np.all([m.size == modes[0].size for m in modes[1:]]):
        raise ValueError("all modes need equal size")
    return modes


def periodic_to_pw(
    kx, ky, kz, pol, l, m, qol, area, posout=0, posin=0, helicity=True, nondifsum=False
):
    """
    periodic_to_pw(kx, ky, kz, pol, l, m, qol, area, posout=0, posin=0, helicity=True)

    Convert periodic spherical wave to plane wave

    Returns the coefficient for the basis change in a periodic arrangement of spherical
    modes to plane waves. For multiple positions only diagonal values (with respect to
    the position) are returned. A correct phase factor is still necessary for the full
    result.

    The polarization values `0` and `1` refer to negative and positive helicity
    waves or, if ``helicity == False``, to TE and TM parity.

    Args:
        kx (float, array_like): X component of destination mode wave vector
        ky (float, array_like): Y component of destination mode wave vector
        kz (float or complex, array_like): Z component of destination mode wave vector
        pol (int, array_like): Polarization of the destination mode
        l (int, array_like): Degree of the source mode
        m (int, array_like): Order of the source mode
        qol (int, array_like): Polarization of the source mode
        area (float, array_like): Unit cell area
        posout (int, optional): Output positions
        posin (int, optional): Input positions
        helicity (bool, optional): If true, helicity basis is assumed, else parity basis.
            Defaults to ``True``.

    Returns:
        complex
    """
    kx, ky, kz, pol, l, m, qol = (
        kx + anp.zeros_like(l),
        ky + anp.zeros_like(l),
        kz + anp.zeros_like(l),
        pol + anp.zeros_like(l),
        l + anp.zeros_like(pol),
        m + anp.zeros_like(pol),
        qol + anp.zeros_like(pol),
    )
    if helicity:
        return _periodic_to_pw_h(
            kx,
            ky,
            kz,
            pol.astype(int),
            l.astype(int),
            m.astype(int),
            qol.astype(int),
            area,
            posout,
            posin,
            nondifsum,
        )
    return _periodic_to_pw_p(
        kx,
        ky,
        kz,
        pol.astype(int),
        l.astype(int),
        m.astype(int),
        qol.astype(int),
        area,
        posout,
        posin,
        nondifsum,
    )


def _periodic_to_pw_h(
    kx, ky, kz, polpw, l, m, polvsw, area, posout, posin, nondifsum=False
):
    if nondifsum:
        k = anp.sqrt(kx * kx + ky * ky + kz * kz)
        phi = arctan2(ky, kx)
    else:
        k = np.sqrt(kx * kx + ky * ky + kz * kz)
        phi = np.arctan2(ky, kx)
    costheta = kz / k

    kz_s = kz

    conds = [
        (kz == 0.0),
        ((anp.imag(kz_s) < 0) | ((anp.imag(kz_s) == 0) & (anp.real(kz_s) < 0))),
    ]
    choices = [1e-20 + 1e-20j, -kz_s]
    kz_s = np.select(conds, choices, kz_s)
    if nondifsum:

        ans0 = (
            sqrt(pi * (2 * l + 1) / (l * (l + 1)))
            * anp.exp((loggamma(l - m + 1) - loggamma(l + m + 1)) * 0.5)
            * np.exp(1j * m * phi)
            * pow(-1j, l)
            * (
                treams.special.tau_fun(l, m, costheta)
                + (2 * polpw - 1) * treams.special.pi_fun(l, m, costheta)
            )
            / (area * k * kz_s)
        )
    else:
        ans0 = (
            sqrt(pi * (2 * l + 1) / (l * (l + 1)))
            * anp.exp((loggamma(l - m + 1) - loggamma(l + m + 1)) * 0.5)
            * np.exp(1j * m * phi)
            * pow(-1j, l)
            * (tau_fun(l, m, costheta) + (2 * polpw - 1) * pi_fun(l, m, costheta))
            / (area * k * kz_s)
        )
    answer = np.where(polvsw != polpw, 0.0j, ans0)
    return answer


def _periodic_to_pw_p(
    kx, ky, kz, polpw, l, m, polvsw, area, posout, posin, nondifsum=False
):
    if nondifsum:
        k = anp.sqrt(kx * kx + ky * ky + kz * kz)
    else:
        k = np.sqrt(kx * kx + ky * ky + kz * kz)

    costheta = kz / k
    phi = arctan2(ky, kx)
    kz_s = kz

    conds = [
        (kz == 0.0),
        ((anp.imag(kz_s) < 0) | ((anp.imag(kz_s) == 0) & (anp.real(kz_s) < 0))),
    ]
    choices = [1e-20 + 1e-20j, -kz_s]
    kz_s = anp.select(conds, choices, kz_s)

    prefactor = (
        sqrt(pi * (2 * l + 1) / (l * (l + 1)).astype(float))
        * anp.exp((loggamma(l - m + 1) - loggamma(l + m + 1)) * 0.5)
        * anp.exp(1j * m * phi)
        * anp.power(-1j, l)
        / (area * k * kz_s)
    )

    answer = anp.where(
        polvsw == polpw,
        prefactor * treams.special.tau_fun(l, m, costheta),
        prefactor * treams.special.pi_fun(l, m, costheta),
    )

    return answer


def to_sw(l, m, polsw, kx, ky, kz, polpw, poltype=None, *args, **kwargs):
    """to_sw(l, m, polsw, kx, ky, kz, polpw, helicity=True)

    Coefficient for the expansion of a plane wave in spherical waves

    Returns the coefficient for the basis change from a plane wave to a spherical wave.
    For multiple positions only diagonal values (with respect to the position) are
    returned.

    The polarization values `0` and `1` refer to negative and positive helicity
    waves or, if ``helicity == False``, to TE and TM parity.

    Args:
        l (int, array_like): Degree of the spherical wave
        m (int, array_like): Order of the spherical wave
        polsw (int, array_like): Polarization of the destination mode
        kx (float, array_like): X component of plane wave's wave vector
        ky (float, array_like): Y component of plane wave's wave vector
        kz (float, array_like): Z component of plane wave's wave vector
        polpw (int, array_like): Polarization of the plane wave
        helicity (bool, optional): If true, helicity basis is assumed, else parity basis.
            Defaults to ``True``.

    Returns:
        complex
    """
    if poltype == "helicity":
        return to_sw_h(l, m, polsw, kx, ky, kz, polpw)
    elif poltype == "parity":
        return to_sw_p(l, m, polsw, kx, ky, kz, polpw)
    raise ValueError(f"invalid poltype")


def to_sw_p(l, m, polvsw, kx, ky, kz, polpw):
    kxy = sqrt(kx * kx + ky * ky)
    pref = np.where(kxy == 0, 1, np.power((kx - 1j * ky) / kxy, m))
    k = np.sqrt(kx * kx + ky * ky + kz * kz)
    costheta = kz / k
    pref = pref * (
        2
        * np.sqrt(np.pi * (2 * l + 1) / (l * (l + 1)))
        * np.exp(0.5 * (loggamma(l - m + 1) - loggamma(l + m + 1)))
        * np.power(1j, l)
    )
    return np.where(
        polvsw == polpw,
        pref * treams.special.tau_fun(l, m, costheta),
        pref * pi_fun(l, m, costheta),
    )


def to_sw_h(l, m, polvsw, kx, ky, kz, polpw):
    kxy = np.sqrt(kx * kx + ky * ky)
    pref = np.where(kxy == 0, 1, np.power((kx - 1j * ky) / kxy, m))

    def inner_fun():
        k = np.sqrt(kx * kx + ky * ky + kz * kz)
        costheta = kz / k
        return (
            2
            * np.sqrt(np.pi * (2 * l + 1) / (l * (l + 1)))
            * np.exp(0.5 * (loggamma(l - m + 1) - loggamma(l + m + 1)))
            * np.power(1j, l)
            * pref
        ) * (
            treams.special.tau_fun(l, m, costheta)
            + (2 * polpw - 1) * pi_fun(l, m, costheta)
        )

    ans = np.where(polvsw != polpw, 0.0j, inner_fun())
    return ans


def illuminate_pw(kx, ky, kz, pol, pidx, helicity, positions, modes, nondif=True):
    """
    Illuminate with a plane wave

    Args:
        kx (float, scalar or (N,)-array): X component of the wave vector
        ky (float, scalar or (N,)-array): Y component of the wave vector
        kz (float or complex, scalar or (N,)-array): Z component of the wave vector
        pol (int, scalar or (N,)-array): Polarization of wave, corresponding to
            the attribute `TMatrix.helicity`
    """
    pos = (*(i[:, None] for i in positions[pidx, :].T),)

    if helicity is True:
        poltype = "helicity"
    else:
        poltype = "parity"
    if nondif:
        pwsw = pw.to_sw(
            *(m[:, None] for m in modes[1:]), kx, ky, kz, pol, poltype=poltype
        )
        pwtr = pw_translate(kx, ky, kz, *pos)
        res = pwsw * pwtr

    else:
        res = to_sw(
            *(m[:, None] for m in modes[1:]), kx, ky, kz, pol, poltype=poltype
        ) * pw_translate(kx, ky, kz, *pos)
    return res


def firstbrillouin2d(kpar, b, n=2, nondifsum=False):
    """
    Reduce the 2d wave vector to the first Brillouin zone.

    The reduction to the first Brillouin zone is first approximated roughly. From this
    approximated vector and its 8 neighbours, the shortest one is picked. As a
    sufficient approximation is not guaranteed (especially for extreme geometries),
    this process is iterated `n` times.

    Args:
        kpar (1d-array): parallel wave vector
        b (2d-array): reciprocal lattice vectors
        n (int): number of iterations

    Returns:
        (1d-array)
    """
    kparstart = kpar
    b1 = b[0, :]
    b2 = b[1, :]
    normsq1 = b1 @ b1
    normsq2 = b2 @ b2
    normsqp = (b1 + b2) @ (b1 + b2)
    normsqm = (b1 - b2) @ (b1 - b2)
    if (
        normsqp < normsq1 - 1e-14
        or normsqp < normsq2 - 1e-14
        or normsqm < normsq1 - 1e-14
        or normsqm < normsq2 - 1e-14
    ):
        raise ValueError("Lattice vectors are not of minimal length")
    if nondifsum:
        kpar = kpar - b1 * anp.round((kpar @ b1) / normsq1)
        kpar = kpar - b2 * anp.round((kpar @ b2) / normsq2)
    else:
        kpar = kpar - b1 * np.round((kpar @ b1) / normsq1)
        kpar = kpar - b2 * np.round((kpar @ b2) / normsq2)
    options = kpar + la.cube(2, 1) @ b
    for _i, option in enumerate(options):
        if nondifsum:
            if option @ option < kpar @ kpar:
                kpar = option
        else:
            kpar = np.where(option @ option < kpar @ kpar, option, kpar)
    if n == 0 or anp.array_equal(kpar, kparstart):
        return kpar
    return firstbrillouin2d(kpar, b, n - 1, nondifsum)


def latticecoupling(
    ks, kpar, a, positions, fullmodes, helicity, t, eta=0.0, nondifsum=True
):
    r"""
    The coupling of the T-matrix in a lattice

    Returns

    .. math::

        \mathbb 1 - T C

    The inverse of this multiplied to the T-matrix in `latticeinteract`. The lattice
    type is inferred from `kpar`.

    Args:
        kpar (float): The parallel component of the T-matrix
        a (array): Definition of the lattice
        eta (float or complex, optional): Splitting parameter in the lattice summation

    Returns:
        complex, array
    """
    m = translate_periodic(
        ks,
        kpar,
        a,
        positions,
        fullmodes,
        helicity=helicity,
        eta=eta,
        nondifsum=nondifsum,
    )
    ans = np.eye(t.shape[0]) - t @ m
    return ans


def translate_periodic(
    ks, kpar, a, rs, out, in_=None, rsin=None, helicity=True, eta=0.0, nondifsum=True
):
    """
    translate_periodic(ks, kpar, a, rs, out, in_=None, rsin=None, helicity=True, eta=0)

    Translation coefficients in a lattice

    Returns the translation coefficients for the given modes in a lattice. The calculation
    uses the fast converging sums of :mod:`ptsa.lattice`.

    The polarization values `0` and `1` refer to negative and positive helicity
    waves or, if ``helicity == False``, to TE and TM parity.

    Args:
        ks (float or complex, scalar or (2,)-array): Wave number(s) in the medium, use
            two values in chiral media, indexed analogous to the polarization values
        kpar (float, (D,)-array): Parallel component of the wave, defines the dimension with `1 <= D <= 3`
        a (float, (D,D)-array): Lattice vectors in each row of the array
        rs (float, (M, 3)-array): Shift vectors with respect to one lattice point
        out (3- or 4-tuple of integer arrays): Output modes
        in_ (3- or 4-tuple of integer arrays): Input modes, if none are given equal to
            the output modes
        rsin (float): Shift vectors to use with the input modes, if non are given equal
            to `rs`
        helicity (bool, optional): If true, helicity basis is assumed, else parity basis.
            Defaults to ``True``.
        eta (float or complex, optional): Cut between real and reciprocal space
            summation, if equal to zero, an estimation for the optimal value is done.

    Returns:
        complex array
    """
    if in_ is None:
        in_ = out
    out = (*(anp.array(o) for o in out),)
    in_ = (*(anp.array(i) for i in in_),)
    if len(out) < 3 or len(out) > 4:
        raise ValueError(f"invalid length of output modes {len(out)}, must be 3 or 4")
    if len(out) == 3:
        out = (anp.zeros_like(out[0]),) + out
    if len(in_) < 3 or len(in_) > 4:
        raise ValueError(f"invalid length of input modes {len(in_)}, must be 3 or 4")
    elif len(in_) == 3:
        in_ = (anp.zeros_like(in_[0]),) + in_
    if rsin is None:
        rsin = rs

    modes = anp.array(
        [
            [l, m]
            for l in range(anp.max(out[1]) + anp.max(in_[1]) + 1)
            for m in range(-l, l + 1)
        ]
    )

    ks = ks.reshape((-1, 1))
    if nondifsum:
        if ks.shape[0] == 2 and ks[0, 0] == ks[1, 0]:
            ks = ks[:1, :]
        kpar = anp.array(kpar)
        rsin = anp.array(rsin)
    else:
        ks = np.where((ks.shape[0] == 2) & (ks[0, 0] == ks[1, 0]), ks[:1, :], ks)

        kpar = np.array(kpar)
        rsin = np.array(rsin)

    if anp.ndim(rs) == 1:
        rs = anp.reshape(rs, (1, -1))

    if anp.ndim(rsin) == 1:
        rsin = anp.reshape(rsin, (1, -1))
    rsdiff = -rs[:, None, None, None, :] + rsin[:, None, None, :]
    if kpar.ndim == 0 or kpar.shape[-1] == 1:
        dlms = la.lsumsw1d_shift(modes[:, 0], modes[:, 1], ks, kpar, a, rsdiff, eta)
    elif kpar.shape[-1] == 2:
        if nondifsum == True:
            dlms = la.lsumsw2d_shift(modes[:, 0], modes[:, 1], ks, kpar, a, rsdiff, eta)
        else:
            dlms = lsumsw2d_shift_vmap(
                modes[:, 0], modes[:, 1], ks, kpar.astype(float), a, rsdiff, eta
            )
    else:
        dlms = la.lsumsw3d(modes[:, 0], modes[:, 1], ks, kpar, a, rsdiff, eta)

    if nondifsum is False:
        if helicity:
            ans_h = translate_periodic_h(
                *(o[:, None] for o in out[1:]),
                *in_[1:],
                dlms[out[0][:, None], in_[0], 0, :],
                dlms[out[0][:, None], in_[0], ks.shape[0] - 1, :],
            )

            return ans_h 
        ans_p = translate_periodic_p(
            *(o[:, None] for o in out[1:]),
            *in_[1:],
            dlms[out[0][:, None], in_[0], 0, :],
        )

        return ans_p

    if helicity:
        ans_h = treams.sw._translate_periodic_h(
            *(o[:, None] for o in out[1:]),
            *in_[1:],
            dlms[out[0][:, None], in_[0], 0, :],
            dlms[out[0][:, None], in_[0], ks.shape[0] - 1, :],
        )
        return ans_h

    ans_p = treams.sw._translate_periodic_p(
        *(o[:, None] for o in out[1:]),
        *in_[1:],
        dlms[out[0][:, None], in_[0], 0, :],
    )
    return ans_p


def translate_periodic_p(lambda_, mu, pol1, l, m, pol2, dlms):
    mask = pol1 == pol2
    ans = mask * _transl_A_lattice(lambda_, mu, l, m, dlms) + (
        mask == False
    ) * _transl_B_lattice(lambda_, mu, l, m, dlms)
    return ans


def translate_periodic_h(lambda_, mu, pol1, l, m, pol2, *dlms):
    dlms1, dlms2 = dlms
    dlms = np.where(pol1 == 0, dlms1, dlms2)
    ans = (pol1 == pol2) * (
        _transl_A_lattice(lambda_, mu, l, m, dlms)
        + (2 * pol1 - 1) * _transl_B_lattice(lambda_, mu, l, m, dlms)
    )
    return ans


def _transl_A_lattice(lambda_, mu, l, m, dlms):

    pref = (
        (-1.0) ** (m)
        * anp.sqrt(
            pi
            * (2 * l + 1)
            * (2 * lambda_ + 1)
            / anp.array(l * (l + 1) * lambda_ * (lambda_ + 1), dtype=float)
        )
        * anp.power(1.0j, lambda_ - l)
    )

    res = 0
    p = l + lambda_
    fin = anp.maximum(anp.abs(lambda_ - l), anp.abs(mu - m)) - 1

    for _ in range(
        0,
        anp.max(
            (l + lambda_) - (anp.maximum(anp.abs(lambda_ - l), anp.abs(m - mu)) - 1)
        ),
        2,
    ):
        i, j = np.ogrid[: len(dlms), : len(dlms)]
        dlm = dlms[i, j, (p * (p + 1) + m - mu)]
        res = res + (
            dlm
            * anp.power(1.0j, p)
            * (2 * p + 1 + 0.0j) ** 0.5
            * wigner3j(l, lambda_, p, m, -mu, -m + mu)
            * wigner3j(l, lambda_, p, 0, 0, 0)
            * (l * (l + 1) + lambda_ * (lambda_ + 1) - p * (p + 1))
        ) * ((p - fin) > 0)
        p = p - 2

    return res * pref


def _transl_B_lattice(lambda_, mu, l, m, dlms):
    pref = (
        (-1.0) ** (m)
        * np.sqrt(
            pi
            * (2 * l + 1)
            * (2 * lambda_ + 1)
            / (l * (l + 1) * lambda_ * (lambda_ + 1)).astype(float)
        )
        * anp.power(1.0j, lambda_ - l)
    )
    res = 0
    p = l + lambda_ - 1

    i, j = np.ogrid[: len(dlms), : len(dlms)]

    for _ in range(
        0,
        anp.max(
            (lambda_ + l - 1) - (anp.maximum(abs(lambda_ - l) + 1, anp.abs(m - mu)) - 1)
        ),
        2,
    ):
        dlm = dlms[i, j, (p * (p + 1) + m - mu)]
        res = res + (
            dlm
            * anp.power(1.0j, p)
            * (2 * p + 1 + 0.0j) ** 0.5
            * wigner3j(l, lambda_, p, m, -mu, -m + mu)
            * wigner3j(l, lambda_, p - 1, 0, 0, 0)
            * anp.sqrt(
                (l + lambda_ + 1 + p)
                * (l + lambda_ + 1 - p)
                * (p - lambda_ + l)
                * (p + lambda_ - l)
            )
        )
    return res * pref


def arrayt(
    tmat,
    lmax,
    helicity,
    kpars,
    k0,
    a,
    epsilon_medium,
    origin=None,
    eta=0,
    mu=1,
    kappa=0,
    positions=None,
    nondifsum=True,
):
    """
    S-matrix from an array of T-matrices

    Create a S-matrix for a two-dimensional array of objects described by the
    T-Matrix or an one-dimensional array of objects described by a cylindrical
    T-matrix.

    Args:
        tmat (TMatrix or TMatrixC): (Cylindrical) T-matrix to put in the array
        kpars (float, (N, 2)-array): Tangential components of the wave vector
        a (array): Definition of the lattice
        eta (float or complex, optional): Splitting parameter in the lattice summation

    Returns:
        SMatrix
    """
    modes = sdefaultmodes(kpars, nondifsum)
    ks = k0 * refractive_index(epsilon_medium)
    kzs = wave_vec_z(*modes[:2], ks[modes[2]], nondifsum)
    if nondifsum:
        allpw = (
            anp.hstack((modes[0], modes[0])),
            anp.hstack((modes[1], modes[1])),
            anp.hstack((kzs, -kzs)),
            anp.hstack((modes[2], modes[2])),
        )
    else:
        allpw = (
            np.hstack((modes[0], modes[0])),
            np.hstack((modes[1], modes[1])),
            np.hstack((kzs, -kzs)),
            np.hstack((modes[2], modes[2])),
        )
    if a.ndim == 2:
        res = array2dt(
            tmat,
            lmax,
            helicity,
            *allpw,
            a,
            epsilon_medium,
            origin=origin,
            eta=eta,
            positions=positions,
            nondifsum=nondifsum,
        )

        dim = int(res.shape[0] / 2)
        dim2 = int(res.shape[1] / 2)
        q00 = res[0:dim, 0:dim2] + np.eye(dim, dim2)
        q01 = res[0:dim, dim2 : 2 * dim2]
        q10 = res[dim : 2 * dim, 0:dim2]
        q11 = res[dim : 2 * dim, dim2 : 2 * dim2] + np.eye(dim, dim2)
        qs = np.array([[q00, q01], [q10, q11]])
        return qs, modes
