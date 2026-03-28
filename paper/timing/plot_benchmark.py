import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
# Files
# ----------------------------

scatter_file = "benchmarks/scaling_scatterers_last_jit_num_2_50_lmax_1_10_fob.csv"
lmax_file = "benchmarks/scaling_lmax_last_jit_num_2_50_lmax_1_10_fob.csv"


df_N = pd.read_csv(scatter_file)
df_L = pd.read_csv(lmax_file)

plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 20,
        "lines.linewidth": 2.0,
        "lines.markersize": 7.0,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def panel_label(ax, label):
    ax.text(
        -0.1,
        1.1,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
    )


# ----------------------------
# Figure
# ----------------------------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# ---------------------------------------------------
# (a) Scaling vs number of scatterers
# ---------------------------------------------------
print("df_N columns:", df_N.columns)

N = df_N["num"].values
fwd = df_N["fwd"].values
bwd = df_N["fb"].values
print("N", N, fwd, bwd)
ax1.plot(N, fwd, "o-", label="Forward")
ax1.plot(N, bwd, "s-", label="Forward+Backward")
# ax1.set_title("Scaling vs N")
ax1.set_xlabel("Number of scatterers")
ax1.set_ylabel("Time (s)")
ax1.grid(True, which="both")
ax1.legend()
panel_label(ax1, "(a)")

# ---------------------------------------------------
# (b) Memory vs number of scatterers
# ---------------------------------------------------
# mem = df_N["peak_rss_gb"].values
mem = df_N["peak_exec_rss_gb"].values
print("mem", mem)
ax2.plot(N, mem, "o-")
# ax2.set_title("Memory vs N")
ax2.set_xlabel("Number of scatterers")
ax2.set_ylabel("Memory (GB)")
ax2.grid(True)
panel_label(ax2, "(b)")

# ---------------------------------------------------
# (c) Scaling vs lmax
# ---------------------------------------------------
print("df_L columns:", df_L.columns)

lmax = df_L["lmax"].values
fwd_L = df_L["fwd"].values
bwd_L = df_L["fb"].values

# fwd_L = df_L["forward_time"].values
# bwd_L = df_L["forward_backward_time"].values

ax3.plot(lmax, fwd_L, "o-", label="Forward")
ax3.plot(lmax, bwd_L, "s-", label="Forward+Backward")
# ax3.set_title("Scaling vs $l_{\\max}$")
ax3.set_xlabel("Multipole order $l_{\\max}$")
ax3.set_ylabel("Time (s)")
ax3.grid(True)
ax3.legend()
panel_label(ax3, "(c)")

# ---------------------------------------------------
# (d) Memory vs lmax
# ---------------------------------------------------
# mem_L = df_L["peak_rss_gb"].values
mem_L = df_L["peak_exec_rss_gb"].values

ax4.plot(lmax, mem_L, "o-")
# ax4.set_title("Memory vs $l_{\\max}$")
ax4.set_xlabel("Multipole order $l_{\\max}$")
ax4.set_ylabel("Memory (GB)")
ax4.grid(True)
panel_label(ax4, "(d)")
# plt.suptitle("Example 1")
fig.tight_layout()
fig.savefig("benchmarks/scaling_plots_last_jit_fob.png", dpi=600)
plt.show()
