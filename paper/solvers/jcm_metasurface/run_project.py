import sys
import os
import  h5py
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
jcm_root = '/scratch/local/nasadova/JCMsuite' # -> set your JCMROOT installation directory
sys.path.append(os.path.join(jcm_root, 'ThirdPartySupport', 'Python'))
import numpy as np
import jcmwave

def objective_from_results(results):
    pf = results[2]   # ConvertToPowerFlux result
    F = pf[flux_key]
    R0 = np.sum(np.real(F[0][:, 2]))
    R1 = np.sum(np.real(F[1][:, 2]))
    return abs(R0 - R1)

def gradient_component_from_results(results, p):
    pf = results[2]
    F = pf[flux_key]
    dF = pf[f"d_{p}"][flux_key]
    R0 = np.sum(np.real(F[0][:, 2]))
    R1 = np.sum(np.real(F[1][:, 2]))
    D = R0 - R1
    sgn = np.sign(D) if D != 0 else 0.0

    dR0 = np.sum(np.real(dF[0][:, 2]))
    dR1 = np.sum(np.real(dF[1][:, 2]))
    return sgn * (dR0 - dR1)

def fd_grad(keys, p, h):
    k_plus = dict(keys)
    k_minus = dict(keys)
    k_plus[p] += h
    k_minus[p] -= h

    res_plus = jcmwave.solve("project.jcmpt", keys=k_plus)
    res_minus = jcmwave.solve("project.jcmpt", keys=k_minus)
    
    J_plus = objective_from_results(res_plus)
    J_minus = objective_from_results(res_minus)

    return (J_plus - J_minus) / (2*h) 

working_dir = 'tmp' 
jcmwave.daemon.shutdown()
num = 5
R = 200.0
zmax = 0.0
rad_init = 80.
ni = np.linspace(0, 360.0, num + 1)[:-1]
x = R * np.cos(ni * np.pi / 180.0)
y = R * np.sin(ni * np.pi / 180.0)
z = np.linspace(-zmax, zmax, x.shape[0])
positions = np.array([x, y, z]).T
radii = np.array([rad_init]*num)
fed = 2
h_bg = 40.0
h_sphere = 15.0
h_sphere_bulk = 25.0
keys = {
    "lambda0": 1050e-9,
    "theta": 0.0,
    "phi": 0.0,
    "fe_degree": fed,
    "precision": 1e-2,

    "eps_medium": 1.5**2,
    "eps_object": 3.5**2,

    "domain_x": 600.0,
    "domain_y": 600.0,
    "z_min": -250.0,
    "cell_thickness": 500.0,

    "h_bg": h_bg,
    "h_sphere": h_sphere,
    "h_sphere_bulk": h_sphere_bulk,
}

for i in range(num):
    keys[f"x{i+1}"] = float(positions[i, 0])
    keys[f"y{i+1}"] = float(positions[i, 1])
    keys[f"z{i+1}"] = float(positions[i, 2])
    keys[f"r{i+1}"] = float(radii[i])
    
t0 = time.perf_counter()
results = jcmwave.solve("project.jcmpt", keys=keys, working_dir=f'{working_dir}/job{i}')

params = [f"{a}{i}" for i in range(1, 6) for a in ("x", "y", "z", "r")]
pf_ref = results[2]
flux_key = "ElectromagneticFieldEnergyFluxDensity"

P_ref_0 = np.sum(np.real(pf_ref[flux_key][0][:, 2]))
P_ref_1 = np.sum(np.real(pf_ref[flux_key][1][:, 2]))

D = P_ref_0 - P_ref_1
J = abs(D)
sgn = np.sign(D)

grad = {}
for p in params:
    dF = pf_ref[f"d_{p}"][flux_key]
    dP0 = np.sum(np.real(dF[0][:, 2]))
    dP1 = np.sum(np.real(dF[1][:, 2]))
    dD = dP0 - dP1
    grad[p] = sgn * dD      
    
t1 = time.perf_counter()
timing = t1-t0
print(f"JCM solve time: {t1 - t0:.3f} s")

coord_params = [f"{a}{i}" for i in range(1, 6) for a in ("x", "y", "z")]
radius_params = [f"r{i}" for i in range(1, 6)]

grad_coords = np.array([grad[p] for p in coord_params], dtype=float)
grad_radii = np.array([grad[p] for p in radius_params], dtype=float)

with h5py.File(f"results_fed_{fed}_hs_{h_bg}_{h_sphere}_{h_sphere_bulk}.h5", "w") as f:
    f["grad_coords"] = grad_coords
    f["grad_radii"] = grad_radii
    f["coord_param_names"] = np.array(coord_params, dtype="S")
    f["radius_param_names"] = np.array(radius_params, dtype="S")
    f["value"] = J
    f["time_full"] = timing
    
R_0 = P_ref_0 
R_1 = P_ref_1 

print("Reflectance, source 0:", R_0)
print("Reflectance, source 1:", R_1)
print("Difference, source 1:", R_1-R_0)


# for p, h in [ ("r1", 0.5)]:
#     g_adj = gradient_component_from_results(results, p)
#     g_fd = fd_grad(keys, p, h)
#     print(p, "adj =", g_adj, "fd =", g_fd, "ratio =", g_adj / g_fd if g_fd != 0 else np.nan)