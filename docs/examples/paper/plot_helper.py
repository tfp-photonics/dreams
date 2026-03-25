import re
from pathlib import Path

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d.axes3d import Axes3D


def plot_res(
    wavelengths,
    fob_init,
    fob_final,
    forward_init,
    backward_init,
    forward_final,
    backward_final,
    va,
    positions,
    radii,
    posf,
    radf,
    wl_at,
    label,
    fname,
    axs=None,
):

    if axs is None:
        fig, ((ax1, ax2), (ax5, ax6)) = plt.subplots(2, 2, figsize=(20, 17))
    else:
        (ax1, ax2), (ax5, ax6) = axs
        fig = ax1.figure

    ax1.scatter(range(len(va)), va)
    ax1.set_yscale("log")
    ax1.set_ylabel(label)
    ax1.set_xlabel("Steps")

    plot_3d_with_depth(positions, radii, ax=ax2, color="red", label="Initial")
    plot_3d_with_depth(posf, radf, ax=ax2, color="blue", label="Optimized")
    set_axes_not_equal(ax2)

    # z = 0 plane
    xx, yy = np.meshgrid(
        np.linspace(-400, 400, 10),
        np.linspace(-400, 400, 10),
    )
    zz = np.zeros_like(xx)
    ax2.plot_surface(xx, yy, zz, color="gray", alpha=0.3, linewidth=0, zorder=0)
    ax5.plot(wavelengths, fob_init, color="r", label="F/B init")
    ax5.plot(wavelengths, fob_final, color="b", label="F/B final")
    ax5.set_ylabel("F/B")
    ax5.set_xlabel(r"$\lambda$ (nm)")
    ax5.axvline(x=wl_at, linestyle="--")
    ax5.legend()

    ax6.plot(wavelengths, forward_init, color="b", ls="--")
    ax6.plot(wavelengths, backward_init, color="r", ls="--")
    ax6.plot(wavelengths, forward_final, color="b", label="Forward")
    ax6.plot(wavelengths, backward_final, color="r", label="Backward")

    ax6.set_yscale("log")
    ax6.set_ylim(5e2, 5e6)
    ax6.set_ylabel("Integrated intensity (a.u.)")
    ax6.set_xlabel(r"$\lambda$ (nm)")
    ax6.axvline(x=wl_at, linestyle="--")
    ax6.legend()

    fig.suptitle(fname)
    return fig, ((ax1, ax2), (ax5, ax6))


def set_axes_not_equal(ax):
    """Enforces equal aspect ratio for 3D plots."""
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_zlim(-200, 200)

    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()
    sx, sy, sz = (x1 - x0), (y1 - y0), (z1 - z0)
    ax.set_box_aspect((sx, sy, sz))


def set_axes_equal(ax):
    """Enforce equal aspect for 3D axes."""
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)

    mean_x = np.mean(x_limits)
    mean_y = np.mean(y_limits)
    mean_z = np.mean(z_limits)

    ax.set_xlim(mean_x - max_range / 2, mean_x + max_range / 2)
    ax.set_ylim(mean_y - max_range / 2, mean_y + max_range / 2)
    ax.set_zlim(mean_z - max_range / 2, mean_z + max_range / 2)


def plot_3d_with_depth(
    centers, radii, depth_factor=1.0, ax=None, color="red", label=None, norm=None
):
    """


    Parameters
    ----------
    centers : (N, 3) array-like
        Sphere centers [x, y, z].
    radii : (N,) array-like
        Sphere radii.
    depth_factor : float
        Currently unused (kept for API compatibility).
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        Existing 3D axis. If None, a new figure/axis is created.
    color : str or color
        Surface color of the spheres.
    label : str, optional
        Kept for API compatibility (not used inside).
    norm : float, optional

    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    min_r = np.min(np.sqrt(centers[:, 0] ** 2 + centers[:, 1] ** 2))
    max_r = np.max(np.sqrt(centers[:, 0] ** 2 + centers[:, 1] ** 2))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.xaxis.set_tick_params(pad=1)
    ax.yaxis.set_tick_params(pad=1)
    ax.zaxis.set_tick_params(pad=1)

    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(3))

    # Opacity scaling along z
    if norm is None:
        if centers.size == 0:
            norm = 1.0
        else:
            norm = float(np.max(centers[:, 2]) or 1.0)

    # Precompute sphere mesh
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    cos_u = np.cos(u)
    sin_u = np.sin(u)
    sin_v = np.sin(v)
    cos_v = np.cos(v)

    # Build spheres
    for i, center in enumerate(centers):
        cx, cy, cz = center
        r = radii[i]

        x = r * np.outer(cos_u, sin_v) + cx
        y = r * np.outer(sin_u, sin_v) + cy
        z = r * np.outer(np.ones_like(u), cos_v) + cz

        # # Map z position to alpha
        #alpha_value = np.interp(cz, [0.0, norm], [0.3, 0.6])
        r_xy = np.sqrt(cx**2 + cy**2) 
        alpha_value = np.interp(r_xy, [min_r, max_r], [0.7, 0.2])
        ax.plot_surface(
            x,
            y,
            z,
            color=color,
            alpha=float(alpha_value),
            edgecolor="none",
            linewidth=0,
            antialiased=True,
            shade=True,
        )

    return ax


def plot_3ds(pos, radii, pos_opt, radii_opt, ax=None):
    """
    3D overlay of initial (red) and optimized (blue) sphere arrays.

    Parameters
    ----------
    pos, radii      : initial centers (N,3) and radii (N,)
    pos_opt, radii_opt : optimized centers (N,3) and radii (N,)
    pitch           : not used here (kept for API consistency)
    ax              : optional existing 3D axis

    Returns
    -------
    ax : matplotlib 3D axis
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    vm = max(np.abs(c[2]) for ps in [pos_opt, pos] for c in ps)
    _ = Normalize(vmin=-vm, vmax=vm)

    colors = ["red", "blue"]
    labels = ["Initial", "Optimized"]
    ax = plot_3d_with_depth(pos, radii, ax=ax, color=colors[0], label=labels[0])
    set_axes_equal(ax)

    ax = plot_3d_with_depth(pos_opt, radii_opt, ax=ax, color=colors[1], label=labels[1])
    set_axes_equal(ax)
    ax.view_init(elev=20, azim=45)
    ax.grid()
    return ax


def plot_plane(plane_mask, vfa, vba, vf2a, vb2a, x, y, z, ttl, ax):

    a_plane = x[plane_mask] if ("XZ" in ttl) else y[plane_mask]
    z_plane = z[plane_mask]

    # Extract fields on this plane
    vf = vfa[plane_mask]
    vb = vba[plane_mask]
    vf2 = vf2a[plane_mask]
    vb2 = vb2a[plane_mask]

    # Angle: backward is forward + pi
    theta = np.arctan2(a_plane, z_plane)
    theta2 = (theta + np.pi) % (2 * np.pi)

    # magnitudes positive for log scale
    vf_p, vb_p = np.abs(vf), np.abs(vb)
    vf2_p, vb2_p = np.abs(vf2), np.abs(vb2)
    vmax = np.max(
        [
            vf_p.max(initial=0),
            vb_p.max(initial=0),
            vf2_p.max(initial=0),
            vb2_p.max(initial=0),
        ]
    )
    eps = max(1e-15, 1e-12 * vmax) if vmax > 0 else 1e-15
    vf_p = np.clip(vf_p, eps, None)
    vb_p = np.clip(vb_p, eps, None)
    vf2_p = np.clip(vf2_p, eps, None)
    vb2_p = np.clip(vb2_p, eps, None)

    #  for plotting
    order = np.argsort(theta)
    th_f, th_b = theta[order], theta2[order]
    vf_p, vb_p = vf_p[order], vb_p[order]
    vf2_p, vb2_p = vf2_p[order], vb2_p[order]

    ax.scatter(th_f, vf_p, color="r", s=8, label="Initial")
    ax.scatter(th_b, vb_p, color="r", s=8)
    ax.scatter(th_f, vf2_p, color="b", s=8, label="Final")
    ax.scatter(th_b, vb2_p, color="b", s=8)

    ax.set_title(ttl)
    ax.set_theta_zero_location("N")
    ax.set_yscale("log")


def plot_2d_field(vfa, vba, vf2a, vb2a, x, y, z, title=None, axs=None, fig=None):
    vfa = np.asarray(vfa).ravel()
    vba = np.asarray(vba).ravel()
    vf2a = np.asarray(vf2a).ravel()
    vb2a = np.asarray(vb2a).ravel()
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()

    all_vs = np.concatenate((vfa, vba, vf2a, vb2a))
    ymin = np.nanmin(all_vs)
    ymax = np.nanmax(all_vs)

    tolerance = 0.1

    if axs is None:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(20, 10), subplot_kw={"projection": "polar"}
        )
    else:
        ax1, ax2 = axs

    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    mask_xz = np.abs(y) < tolerance  # XZ plane
    mask_yz = np.abs(x) < tolerance  # YZ plane

    plot_plane(mask_xz, vfa, vba, vf2a, vb2a, x, y, z, "XZ plane", ax=ax1)
    plot_plane(mask_yz, vfa, vba, vf2a, vb2a, x, y, z, "YZ plane", ax=ax2)

    ax1.legend(loc="upper left", markerscale=4, bbox_to_anchor=(0.95, 0.95))

    if title is not None:
        plt.title(title)


def plot_multipole(xss, wls, wl, ax, title):
    wls = np.asarray(wls)
    idx = np.where(np.isclose(wls, wl))[0]
    if idx.size != 1:
        raise ValueError(f"Expected exactly one match for wl={wl}, got {idx.size}")
    idx = idx[0]

    xi_names = []
    xi_vals = []

    for key, value in xss.items():
        arr = np.asarray(value)
        xi_names.append(key)
        xi_vals.append(arr[idx])

    base_colors = ["magenta", "cyan"]
    repeat = (len(xi_names) + len(base_colors) - 1) // len(base_colors)
    colors = (base_colors * repeat)[: len(xi_names)]
    ax.bar(xi_names, xi_vals, color=colors)
    ax.set_title(title)
    ax.set_ylabel(r"$\sigma_{sc}$ (nm$^2$)")
    ax.tick_params(axis="x", labelsize=11)


def draw_chain(ax, f_1, base_width=0.01, reference_ax=None):
    """
    Draw chained phasors for each component and return concatenated x/y
    coordinates for setting global axis limits.

    Parameters
    ----------
    ax : matplotlib Axes
    f_1 : array-like, shape (C, M)
        Complex phasors per component (C) and segment (M). Typically f_1 = vec[idx].T
    base_width : float
        Desired shaft width (inches) for the reference_ax.
    reference_ax : Axes or None
        Axes used as baseline for relative width scaling (default: ax).
    colors : sequence
        Two colors per component (even/odd segments). Length >= 2*C.
    labels : sequence
        Two labels per component (even/odd). Length >= 2*C.

    Returns
    -------
    all_x : np.ndarray
        Concatenated start/tip x coordinates across all components.
    all_y : np.ndarray
        Concatenated start/tip y coordinates across all components.
    """
    print("draw chain", f_1.shape)
    fig = ax.get_figure()
    colors = np.array(["gray", "magenta", "blue", "orange", "black", "green"])
    labels = [
        "r$_x$, E-multipoles",
        "r$_x$, M-multipoles",
        "r$_y$, E-multipoles",
        "r$_y$, M-multipoles",
        "z-pol, E-multipoles",
        "z-pol, M-multipoles",
    ]
    # current and reference axes sizes (inches)
    bbox_ax = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    size_ax = (
        max(bbox_ax.width, bbox_ax.height)
        if (bbox_ax.width > 0 and bbox_ax.height > 0)
        else 1.0
    )

    if reference_ax is None:
        reference_ax = ax
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_size_in = max(bbox.width, bbox.height)
    base_visual_width = 0.08  # inches

    width = base_visual_width / ax_size_in

    all_x_parts, all_y_parts = [], []

    for ee, ecomp in enumerate(f_1):
        U = np.asarray(ecomp).real
        V = np.asarray(ecomp).imag

        # chained tails
        X0 = np.concatenate([[0.0], np.cumsum(U)[:-1]])
        Y0 = np.concatenate([[0.0], np.cumsum(V)[:-1]])


        all_x_parts.extend((X0, X0 + U))
        all_y_parts.extend((Y0, Y0 + V))

        # alternating colors
        color_pair = [colors[2 * ee + 1], colors[2 * ee]]
        ln = len(X0)
        for remain in (0, 1):
            mask = (np.arange(ln) % 2) == remain
            ax.quiver(
                X0[mask],
                Y0[mask],
                U[mask],
                V[mask],
                angles="xy",
                scale_units="xy",
                scale=1,
                label=labels[2 * ee + remain],
                color=color_pair[remain],
                width=width,
                units="inches",
            )

        # axes & final tip
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.grid(True)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.plot(X0[-1] + U[-1], Y0[-1] + V[-1], "*", color="black", markersize=7)
        ax.set_aspect("equal", "box")

    all_x = np.hstack(all_x_parts) if all_x_parts else np.array([])
    all_y = np.hstack(all_y_parts) if all_y_parts else np.array([])
    return all_x, all_y


def get_first(f: h5py.File, *keys: str) -> np.ndarray:
    """Return the first existing dataset among keys as a numpy array."""
    for k in keys:
        if k in f:
            return np.array(f[k][...])
    raise KeyError(f"None of {keys} found in {f.filename}")


def read_run(path: Path) -> dict:
    """Read one optimization run into a simple dict."""
    with h5py.File(path, "r") as f:
        run = dict(
            pos_init=np.array(f["pos_init"][...]),
            radii_init=np.array(f["radii_init"][...]),
            pos_final=np.array(f["pos_final"][...]),
            radii_final=np.array(f["radii_final"][...]),
            pitch=float(np.array(f["pitch"][...])),
            wls_axis=get_first(f, "wls_150_range", "wls_range", "wls_axis"),
            R_x=get_first(f, "r1_final_wls_150_lm15", "r1_final", "Rx"),
            R_y=get_first(f, "r2_final_wls_150_lm15", "r2_final", "Ry"),
            opt_wls=[],
        )
        # optimisation wavelengths
        if "wl" in f:
            run["opt_wls"] = [float(np.array(f["wl"][...]))]
        elif "wls" in f:
            run["opt_wls"] = list(np.array(f["wls"][...]).ravel())
        else:
            name = os.path.basename(path.as_posix())
            m = re.search(r"wls-([0-9.]+)-([0-9.]+)", name)
            if m:
                run["opt_wls"] = [float(m.group(1)), float(m.group(2))]
            else:
                m = re.search(r"wl-([0-9.]+)", name)
                if m:
                    run["opt_wls"] = [float(m.group(1))]

        return run


def plot_one_column(
    fig: plt.Figure,
    ax3d: Axes3D,
    axsp: plt.Axes,
    run: dict,
    add_3d_legend: bool = False,
) -> None:
    """
    Plot one column: top 3D structure (initial/final), bottom spectrum.

    This is essentially your first script's plot_one_column,
    just without the debug prints.
    """
    # --- TOP: 3D geometry ---
    ax3d = plot_3ds(
        run["pos_init"],
        run["radii_init"],
        run["pos_final"],
        run["radii_final"],
        ax=ax3d,
    )
    if add_3d_legend:
        legend_patches = [
            mpatches.Patch(color="red", label="Initial"),
            mpatches.Patch(color="blue", label="Optimized"),
        ]
        ax3d.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(0.1, 1.1))

    # --- BOTTOM: spectra ---
    axsp.plot(run["wls_axis"], run["R_x"], label=r"$R_x$")
    axsp.plot(run["wls_axis"], run["R_y"], label=r"$R_y$")

    for wl0 in run["opt_wls"]:
        axsp.axvline(wl0, color="red", ls="--", lw=1)

    axsp.set_xlabel(r"$\lambda$ (nm)")
    axsp.set_ylabel("R")
    ax3d.set_xlabel("x (nm)", labelpad=-1)
    ax3d.set_ylabel("y (nm)", labelpad=-1)
    ax3d.set_zlabel("z (nm)", labelpad=-4.0)


def plot_spectra(runs, out_path="final_plots/"):
    # # --- figure + layout (3×2) ---
    fig = plt.figure(figsize=(7.0, 9.0))
    gs = fig.add_gridspec(3, 2)

    ax3d_1 = fig.add_subplot(gs[0, 0], projection="3d")
    axsp_1 = fig.add_subplot(gs[0, 1])
    ax3d_2 = fig.add_subplot(gs[1, 0], projection="3d")
    axsp_2 = fig.add_subplot(gs[1, 1])
    ax3d_3 = fig.add_subplot(gs[2, 0], projection="3d")
    axsp_3 = fig.add_subplot(gs[2, 1])

    # # --- plot columns ---
    plot_one_column(fig, ax3d_1, axsp_1, runs[0], add_3d_legend=True)
    plot_one_column(fig, ax3d_2, axsp_2, runs[1], add_3d_legend=False)
    plot_one_column(fig, ax3d_3, axsp_3, runs[2], add_3d_legend=False)

    # legend only on the first spectrum axis
    axsp_1.legend()

    panel_axes = [ax3d_1, axsp_1, ax3d_2, axsp_2, ax3d_3, axsp_3]
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for a, lab in zip(panel_axes, panel_labels):
        if getattr(a, "name", "") == "3d":  # 3D axes
            a.text2D(
                0.05,
                1.15,
                lab,
                transform=a.transAxes,
                ha="left",
                va="top",
                fontsize=12,
                fontweight="bold",
            )
        else:
            a.text(
                0.05,
                1.15,
                lab,
                transform=a.transAxes,
                ha="left",
                va="top",
                fontsize=12,
                fontweight="bold",
            )

    plt.subplots_adjust(wspace=0.1, hspace=0.4, left=0.1)
    for ext in ["pdf", "png"]:
        save_kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            save_kwargs["dpi"] = 300
        fig.savefig(out_path + f"fig4abcd.{ext}", **save_kwargs)
    plt.show()


def plot_phasor_grid(folder, files_map, out_path="final_plots/"):
    if isinstance(files_map, dict):
        filenames = list(files_map.keys())
    else:
        filenames = list(files_map)

    fig1, axs1 = plt.subplots(2, 2, figsize=(7, 5.5), sharex="row", sharey="row")
    fig2, axs2 = plt.subplots(2, 2, figsize=(7, 8.0), sharex="row", sharey="row")
    titles = ["x", "y"]
    all_x, all_y = [], []
    all_x2, all_y2 = [], []

    for fname in filenames:
        if "1r" in fname:
            double = True
            fig = fig1
            axs = axs1
        else:
            double = False
            fig = fig2
            axs = axs2
        with h5py.File(folder / fname, "r") as f:
            phasor_x = np.asarray(f["phasor_x"][...])
            phasor_y = np.asarray(f["phasor_y"][...])
            
            if "wls" in f:
                wls = np.asarray(f["wls"][...]).ravel()
            else:
                wls = np.array([float(f["wl"][()])])
            wls_range = np.asarray(f["wls_150_range"][...]).ravel()
            mask = np.isin(wls_range, wls)
            phasor_x = phasor_x[mask]
            phasor_y = phasor_y[mask]
            pitch = f["pitch"][...]
            eps_medium = f["eps_emb"][...]
            lmax = f["lmax"][...]
            radf = f["radii_final"][...]
            posf = f["pos_final"][...]
            radii = f["radii_init"][...]
            positions = f["pos_init"][...]
        params = np.concatenate((np.array(posf).flatten(), np.array(radf)))
        cfg = {
            "lmax": 15,
            "lmax_glob": 15,
            "pitch": pitch,
            "eps_medium": eps_medium,
            "rmax_coef": 1,
            "helicity": False,
            "kx": 0.0,
            "ky": 0.0,
        }

        ansx = np.array(phasor_x).squeeze()
        ansy = np.array(phasor_y).squeeze()
        for row_idx, vec in enumerate((ansx, ansy)):
            for ind, wl in enumerate(wls):
                if np.isclose(wl, 950.0):
                    col = 0
                elif np.isclose(wl, 1050.0):
                    col = 1
                else:
                    continue
                if double:
                    raw = np.asarray(vec)[ind]
                else:
                    raw = np.asarray(vec)
                ax = axs[row_idx, col]
                esum = np.sum(raw, axis=-2)
                f_1 = raw.T
                refl = np.real(esum.conj() @ esum)
                if refl < 1e-3:
                    refl_str = f"{refl:.1e}"
                else:
                    refl_str = f"{refl:.3f}"

                ax.set_title(
                    f"At {wl:.0f} nm: {titles[row_idx]}-illumination\n"
                    f"Reflectance: {refl_str}"
                )

                x_part, y_part = draw_chain(ax, f_1, base_width=0.01)
                if not double:
                    if row_idx == 0:
                        axins = inset_axes(
                            ax, "70%", "70%", loc="upper left", borderpad=0.5
                        )
                        axins.set_aspect("equal", "box")
                        
                        draw_chain(axins, f_1, base_width=0.1)
                        axins.tick_params(labelsize=8, pad=2)
                        ax.indicate_inset_zoom(axins, edgecolor="red")
                        axins.set_xticks([])
                        axins.set_yticks([])
                if double:
                    all_x.append(x_part)
                    all_y.append(y_part)
                else:
                    all_x2.append(x_part)
                    all_y2.append(y_part)

    def build_lims(all_x, all_y, axs, fig):
        all_x = np.hstack(all_x)
        all_y = np.hstack(all_y)
        dx = all_x.max() - all_x.min()
        dy = all_y.max() - all_y.min()
        xmin, xmax = all_x.min() - 0.1 * dx, all_x.max() + 0.1 * dx
        ymin, ymax = all_y.min() - 0.1 * dy, all_y.max() + 0.1 * dy
        for i in range(2):
            for j in range(2):
                axs[i, j].set_xlim(xmin, xmax)
                axs[i, j].set_ylim(ymin, ymax)
        for ax, lab in zip(axs.ravel(), ["(a)", "(b)", "(c)", "(d)"]):
            ax.annotate(
                lab,
                xy=(0.0, 1.10),
                xycoords="axes fraction",
                fontsize=12,
                fontweight="bold",
            )

        handles, labels = [], []
        for ax in axs.ravel():
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        uniq = {}
        for h, l in zip(handles, labels):
            if l and l not in uniq:
                uniq[l] = h

        if uniq:
            fig.legend(
                uniq.values(),
                uniq.keys(),
                loc="lower center",
                ncol=2,
                frameon=False,
                bbox_to_anchor=(0.5, 0.95),
            )

    build_lims(all_x, all_y, axs1, fig1)
    build_lims(all_x2, all_y2, axs2, fig2)
    Path(out_path).mkdir(parents=True, exist_ok=True)
    pairs = [
        (fig1, "fig6abcd"),
        (fig2, "fig5abcd"),
    ]

    for fig, name in pairs:
        fig.savefig(out_path + f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out_path + f"{name}.png", dpi=300, bbox_inches="tight")
