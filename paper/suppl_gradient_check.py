import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from func_helper import fob
from jax import grad

jax.config.update("jax_enable_x64", True)


def build_problem(num, lmax, lmax_glob):
    eps_emb = 1.0
    eps_obj = 2.5**2
    epsilons = np.tile(np.array([eps_obj, eps_emb]), (num, 1))

    radius = 80.0
    R = 400.0
    ni = np.linspace(0, 360.0, num + 1)[:-1]
    x = R * np.cos(ni * np.pi / 180.0)
    y = R * np.sin(ni * np.pi / 180.0)
    z = np.zeros_like(x)
    positions = np.array([x, y, z]).T
    radii = np.ones(num) * radius

    wl1 = 800.0
    k0 = 2 * np.pi / wl1

    Sgn = 1

    if Sgn == 1:
        tetamin = 0.0
        tetamax = np.pi / 2
    else:
        tetamin = np.pi / 2
        tetamax = np.pi

    phimin = 0.0
    phimax = 2 * np.pi
    Nteta = 20
    Nphi = 36

    dteta = (tetamax - tetamin) / float(Nteta - 1)
    dphi = (phimax - phimin) / float(Nphi)

    r = 100000.0

    tetalist = np.ones((Nteta, Nphi)) * np.linspace(tetamin, tetamax, Nteta)[:, None]
    philist = (
        np.ones((Nteta, Nphi))
        * np.linspace(phimin, phimax, Nphi, endpoint=False)[None, :]
    )

    xff = (r * np.sin(tetalist) * np.cos(philist)).flatten()
    yff = (r * np.sin(tetalist) * np.sin(philist)).flatten()
    zff = (r * np.cos(tetalist)).flatten()

    d_solid_surf = (r**2 * np.sin(tetalist) * dteta * dphi).flatten()

    grid_f = np.transpose(np.array([xff, yff, zff]))
    grid_b = -grid_f

    cfg = {
        "num": num,
        "lmax": lmax,
        "k0": k0,
        "lmax_glob": lmax_glob,
        "eps_emb": eps_emb,
        "eps_obj": eps_obj,
        "epsilons": epsilons,
        "pol": (0, 1, 0),
        "Sgn": Sgn,
        "grid_f": grid_f,
        "grid_b": grid_b,
        "d_solid_surf": d_solid_surf,
        "poltype": "parity",
    }

    param = np.append(positions.flatten(), radii)
    return cfg, param


cfg, param = build_problem(num=6, lmax=3, lmax_glob=6)

param = jnp.array(param)

# choose index to test
idx_position = 1
idx_radius = len(param) - 1

g_ad = grad(fob)(param, cfg)

print("AD gradient position:", g_ad)
print("AD gradient radius:", g_ad)


def fd_grad(p, idx, h):
    e = jnp.zeros_like(p).at[idx].set(1.0)
    return (fob(p + h * e, cfg) - fob(p - h * e, cfg)) / (2 * h)


hs = np.logspace(-6, 0, 30)
# hs = np.logspace(-8, -1, 20)#20)

errors_pos = []
errors_rad = []
print("g_ad[idx_position] =", float(g_ad[idx_position]))
print("g_ad[idx_radius]   =", float(g_ad[idx_radius]))
for h in hs:
    print("h", h)
    g_fd_pos = fd_grad(param, idx_position, h)
    g_fd_rad = fd_grad(param, idx_radius, h)
    print(
        "vals",
        g_fd_pos,
        g_ad[idx_position],
        "diff ",
        abs(g_fd_pos - g_ad[idx_position]),
    )
    err_pos = abs(g_fd_pos - g_ad[idx_position]) 
    err_rad = abs(g_fd_rad - g_ad[idx_radius]) 

    errors_pos.append(err_pos)
    errors_rad.append(err_rad)

plt.loglog(hs, errors_pos, label="Position coordinate")
plt.loglog(hs, errors_rad, label="Radius")
plt.xlabel("FD step size h")
plt.ylabel("Absolute gradient error")
plt.legend()
plt.grid(True, which="both")
plt.savefig("gradient_verification.png", dpi=300)
plt.show()
