import os
import re
from pathlib import Path

import h5py
import matplotlib as mpl
import numpy as np
from plot_helper import plot_phasor_grid, plot_spectra

# =========================
# configuration
# =========================

FOLDER = Path("paper_results")

FILES = [
    "parallel-si-safe-5.0-rcd-circle-R-170.0-zshift-0.0-nosub-num-5-randr-0.0-randpos-0.0-rinit-10.0-lmax-7-7-nsteps-200-wl-950.0-pitch-600.0-rmax-1-rl-5.0-with-limit.h5",
    "parallel-si-safe-5.0-rcd-circle-R-170.0-zshift-0.0-nosub-num-5-randr-0.0-randpos-0.0-rinit-10.0-lmax-7-7-nsteps-200-wl-1050.0-pitch-600.0-rmax-1-rl-5.0-with-limit.h5",
    "1r-balanced-epsminxy-zshift-0.0-numwls-2-safecor-5.0-notnorm-si-rcd-circle-adapt-R-170.0-nosub-num-5-randr-0.0-randpos-0.0-rinit-10.0-lmax-7-8-nsteps-250-wls-950.0-1050.0-pitch-600.0-rmax-1-rl-5.0-with-limit-seed-1177611817-jit.h5"
 ]

# =========================
# I/O helpers
# =========================


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
            # fall back to parsing the filename
            name = os.path.basename(path.as_posix())
            m = re.search(r"wls-([0-9.]+)-([0-9.]+)", name)
            if m:
                run["opt_wls"] = [float(m.group(1)), float(m.group(2))]
            else:
                m = re.search(r"wl-([0-9.]+)", name)
                if m:
                    run["opt_wls"] = [float(m.group(1))]
        return run


# =========================
# plotting helpers
# =========================

mpl.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    }
)

runs = [read_run(FOLDER / fn) for fn in FILES]

plot_spectra(runs)
plot_phasor_grid(FOLDER, FILES, out_path="final_plots/")
