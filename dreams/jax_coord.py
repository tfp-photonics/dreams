import numpy as anp
import jax
import jax.numpy as np
from jax import config

config.update("jax_enable_x64", True)


def car2pol(car):
    """
    Convert Cartesian to polar coordinates.

    This works on arrays of shape (..., 2), where the last axis is [x, y].

    Args:
        car (array_like): Cartesian coordinates [..., 2] with components [x, y].

    Returns:
        array: Polar coordinates [..., 2] with components [r, phi],
               r >= 0 and phi in [-pi, pi].
    """
    car = np.asarray(car)
    x = car[..., 0]
    y = car[..., 1]
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return np.stack([r, phi], axis=-1)


def subc2s(th):
    """
    th: (x, y, z) -> (r, theta, phi)
    double where trick:avoid nans even in the branch that is not chosen
    """
    cond0 = (th[0] == 0.0) & (th[1] == 0.0)
    denomx = np.where(cond0, 1.0, th[0])
    denomz = np.where(cond0, 1.0, th[2])
    xy_safe = np.hypot(th[1], denomx)
    a1 = np.hypot(xy_safe, denomz)
    a2 = np.abs(np.arctan2(xy_safe, denomz))
    a3 = np.arctan2(th[1], denomx)
    a = np.array([a1, a2, a3])
    anulp = np.array([np.abs(th[2]), 0.0, 0])
    # Note: this explicit z-axis handling (x = y = 0) differs from treams.car2sph,
    # so values on the z-axis are not bitwise-identical to treams. It is required
    # to avoid NaNs / gradient issues with JAX at the origin.
    anuln = np.array([np.abs(th[2]), np.pi, 0])
    conds = [(th[2] < 0) & cond0, (th[2] >= 0) & cond0]
    ans = np.select(conds, [anuln, anulp], a)
    return ans

@jax.jit
def car2sph(thg):
    r"""
    Vectorized Cartesian → spherical coordinate transform.

    Args:
        thg (ndarray):
            Coordinates in Cartesian form. Accepted shapes:

            - ``(3,)`` for a single point,
            - ``(N, 3)`` for a list of points,
            - ``(M, N, 3)`` for a grid of points.
        thg[..., 0] = x
        thg[..., 1] = y
        thg[..., 2] = z

    Returns:
        ndarray:
            Same shape as ``thg``, but with the last axis holding
            spherical coordinates ``(r, theta, phi)`` instead of
            ``(x, y, z)``.
    """

    x = thg[..., 0]
    y = thg[..., 1]
    z = thg[..., 2]

    cond0 = (x == 0.0) & (y == 0.0)

    denomx = np.where(cond0, 1.0, x)
    denomz = np.where(cond0, 1.0, z)

    xy_safe = np.hypot(y, denomx)
    a1 = np.hypot(xy_safe, denomz)
    a2 = np.abs(np.arctan2(xy_safe, denomz))
    a3 = np.arctan2(y, denomx)

    a = np.stack([a1, a2, a3], axis=-1)

    zeros = np.zeros_like(z)
    anulp = np.stack([np.abs(z), zeros, zeros], axis=-1)
    anuln = np.stack([np.abs(z), np.full_like(z, np.pi), zeros], axis=-1)

    ans = np.where((cond0 & (z >= 0))[..., None], anulp, a)
    ans = np.where((cond0 & (z < 0))[..., None], anuln, ans)

    return ans


def vsph2car(iv, ip):
    """Convert vector from spherical to cartesian basis"""

    # def v2c(iv, ip)
    ip1 = ip[:, :, 1]
    ip0 = ip[:, :, 0]
    ip2 = ip[:, :, 2]
    iv1 = iv[:, :, 1]
    iv0 = iv[:, :, 0]
    iv2 = iv[:, :, 2]
    vy = (
        np.sin(ip1) * np.sin(ip2) * iv0
        + np.cos(ip1) * np.sin(ip2) * iv1
        + np.cos(ip2) * iv2
    )
    vz = np.cos(ip1) * iv0 - np.sin(ip1) * iv1
    return np.array(
        [
            np.sin(ip1) * np.cos(ip2) * iv0
            + np.cos(ip1) * np.cos(ip2) * iv1
            - np.sin(ip2) * iv2,
            vy,
            vz,
        ]
    ).transpose(1, 2, 0)
