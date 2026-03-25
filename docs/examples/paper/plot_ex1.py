import h5py
import matplotlib.pyplot as plt
import numpy as np
from plot_helper import plot_2d_field, plot_multipole, plot_res

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "font.family": "sans-serif",
        "lines.markersize": 2,
        "axes.titleweight": "bold",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "bf",
    }
)


# ---------- 2×2 figure (polar fields + multipole bars) ----------
def render_summary_panel(
    xssi, xssf, wls, wl, vf, vb, vf2, vb2, xff, yff, zff, figsize=(7.5, 5.0), dpi=300
):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax1 = fig.add_subplot(2, 2, 1, projection="polar")
    ax2 = fig.add_subplot(2, 2, 2, projection="polar")
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # top: far-field init vs final

    plot_2d_field(vf, vb, vf2, vb2, xff, yff, zff, axs=(ax1, ax2), fig=fig)

    plot_multipole(xssi, wls, wl, ax3, "Initial")
    plot_multipole(xssf, wls, wl, ax4, "Final")

    # panel letters
    for ax, lab in zip((ax1, ax2, ax3, ax4), ("(a)", "(b)", "(c)", "(d)")):
        ax.annotate(
            lab,
            xy=(0.02, 1.06),
            xycoords="axes fraction",
            fontsize=11,
            fontweight="bold",
        )

    fig.subplots_adjust(wspace=0.5, hspace=0.3)
    # fig.tight_layout()
    return fig, (ax1, ax2, ax3, ax4)

path = "paper_results/num-6-parity-R-400.0-local-pos-rad-800.0-lmax-3-pol-0-nsteps-100-rad-80-nph-36-nth-20.h5"
with h5py.File(path, "r") as f:
    need = lambda k: np.asarray(f[k][...])

    # wavelengths
    wls = need("wls")
    wl = np.asarray(f["wl_at"][...]) if "wl_at" in f else need("wl")
    va = need("values")

    # scalars / metrics
    fob_init = need("fob_init")
    fob_final = need("fob_final")
    forward_final = need("fwd_final")
    backward_final = need("bwd_final")
    forward_init = need("fwd_init")
    backward_init = need("bwd_init")
    lmax = int(need("lmax"))
    lmax_glob = int(need("lmax_glob"))
    sum_final = need("sum_final")
    sum_init = need("sum_init")
    # geometry
    pos_init = need("pos_init")
    pos_final = need("pos_final").reshape((-1, 3))
    radii_init = need("radii_init")
    radii_final = need("radii_final")

    labels_alt = [
        rf"E$_{l}$" if p == "E" else rf"M$_{l}$"
        for l in range(1, lmax_glob + 1)
        for p in ("E", "M")
    ]

    xssi = {lab: np.asarray(f[lab + "_init"][...]) for lab in labels_alt}
    xssf = {lab: np.asarray(f[lab + "_final"][...]) for lab in labels_alt}

    vf = need("fwd_field_init").flatten()
    vb = need("bwd_field_init").flatten()
    vf2 = need("fwd_field_final").flatten()
    vb2 = need("bwd_field_final").flatten()
    xff = need("xff")
    yff = need("yff")
    zff = need("zff")


fig, axes = render_summary_panel(xssi, xssf, wls, wl, vf, vb, vf2, vb2, xff, yff, zff)
fig.savefig("final_plots/fig2abcd.png", dpi=300, bbox_inches="tight")
fig.savefig("final_plots/fig2abcd.pdf", bbox_inches="tight")
plt.show()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 6))

# make ax2 3D
ax2.remove()
ax2 = fig.add_subplot(2, 2, 2, projection="3d")

ylabel = "F/B"

plot_res(
    wls,
    fob_init,
    fob_final,
    forward_init,
    backward_init,
    forward_final,
    backward_final,
    va,
    pos_init,
    radii_init,
    pos_final,
    radii_final,
    wl,
    ylabel,
    "",
    ((ax1, ax2), (ax3, ax4)),
)

fig.subplots_adjust(wspace=0.5, hspace=0.4)
# enlarge/shift the 3D panel
delta = 0.15
x0, y0, w, h = ax2.get_position().bounds
ax2.set_position([x0 - delta, y0 - delta / 2, w + delta, h + delta])

fig.text(0.05, 0.96, "(a)", fontsize=11, fontweight="bold")  # upper-left
fig.text(0.53, 0.96, "(b)", fontsize=11, fontweight="bold")  # upper-right
fig.text(0.05, 0.46, "(c)", fontsize=11, fontweight="bold")  # lower-left
fig.text(0.53, 0.46, "(d)", fontsize=11, fontweight="bold")  # lower-right

fig.savefig("final_plots/fig1abcd.pdf", bbox_inches="tight")
fig.savefig("final_plots/fig1abcd.png", dpi=300, bbox_inches="tight")
plt.show()
