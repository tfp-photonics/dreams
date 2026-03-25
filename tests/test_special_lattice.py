import os
import math
import jax
import jax.numpy as np
import numpy as onp
import treams.lattice as la
import treams
from treams.special import car2pol as car2pol_ref 
import treams.special as cs
from dreams.jax_waves import plane_wave
from dreams.jax_la_bounded import lsumsw2d_shift_vmap
from dreams.jax_coord import car2pol as car2pol_jax
from dreams.jax_recursive import incgamma, _redincgamma
from dreams.jax_misc import diffr_orders_circle as diffr_orders_circle_jax

def test_diffr_orders_circle_square_exact():
    a = 1.0
    b = 2 * np.pi * np.array([[1 / a, 0.0],
                              [0.0, 1 / a]])
    for rmax in [0.0, 1.0, 4]:
        ref = la.diffr_orders_circle(b, rmax=rmax)
        got = diffr_orders_circle_jax(b, rmax)
        print("ref got", ref, got)
        assert ref.shape == got.shape, (
            f"Shape mismatch for rmax={rmax}: "
            f"ref {ref.shape}, got {got.shape}"
        )
        assert np.array_equal(ref, got), (
            f"Value/order mismatch for rmax={rmax}\n"
            f"ref=\n{ref}\n\n"
            f"got=\n{got}"
        )

def test_plane_wave_partial_parity_matches_treams():
    # --- setup problem ---
    lam = 1.0
    k0 = 2 * onp.pi / lam

    #kpars = onp.array([0.3 * k0, 0.1 * k0])

    eps_bg = 2.25
    mat_bg = treams.Material(eps_bg)

    # treams basis (by components kx, ky, pol)
    
    kpars_array = [
        onp.array([0.0, 0.0, k0]),
        onp.array([0.0      , 0.0      ]),
        onp.array([0.1 * k0 , 0.0      ]),
        onp.array([0.0      , 0.2 * k0 ]),
        onp.array([0.15*k0  , 0.25*k0  ]),
    ]
    # --- reference: treams plane_wave ---
    # x-polarized plane wave [1,0,0] in Cartesian E-field
    for kpars in kpars_array:
        if len(kpars) == 2:
            pwb = treams.PlaneWaveBasisByComp.default([kpars])
        elif len(kpars) == 3:
            pwb = treams.PlaneWaveBasisByUnitVector.default([kpars])
        plw = treams.plane_wave(
            kpars,
            [1, 0, 0],
            k0=k0,
            basis=pwb,
            material=mat_bg,
            poltype="parity"
        )
        coeffs_ref = onp.asarray(plw)  
        # basis array of shape (N, 3): [kx, ky, pol]
        if len(kpars) == 2:
            basis_jax = onp.stack([pwb.kx, pwb.ky, pwb.pol], axis=-1)
        else:
            basis_jax = onp.stack([pwb.qx, pwb.qy, pwb.qz, pwb.pol], axis=-1)
        coeffs_jax, basis_out = plane_wave(
            kpars,                # 2-component k‖
            [1, 0, 0],            # same Cartesian E
            k0=k0,
            basis=basis_jax,
            epsilon=eps_bg,       
            modetype="up",
            poltype="parity",
        )

        coeffs_jax = onp.asarray(coeffs_jax)
        # --- comparison ---
        onp.testing.assert_allclose(
            coeffs_jax,
            coeffs_ref,
            rtol=1e-10,
            atol=1e-12,
        )

def test_lsumsw2d_shift_vmap_matches_ref():
    # --- simple 2D square lattice ---
    a = onp.array([[1.0, 0.0],
                   [0.0, 1.0]])          # lattice vectors
    rs = onp.array([[[0.0, 0.0, 0.0]]])  # shape (n_r1, n_r2, 3)
    k = 2 * onp.pi / 1.5
    ks = onp.array([[k]])                # shape (n_k, 1) 
    kpar = onp.array([0.1 * k, 0.0 * k])
    eta = 0.7

    # modes: ls, ms as 1D arrays
    ls = onp.array([1, 1, 2, 2, 2], int)
    ms = onp.array([-1, 1, -2, 0, 2], int)

    ref = treams.lattice.lsumsw2d_shift(ls, ms, ks.ravel(), kpar, a, rs, eta)

    dlms = lsumsw2d_shift_vmap(ls, ms, ks, kpar, a, rs, eta)
    #dlms = dlms.reshape(-1, dlms.shape[-1])
    dlms = onp.array(dlms)          # materialise
    # shapes should match treams: (n_r1, n_r2, n_k, n_modes)
    assert dlms.shape == ref.shape
    rtol = 1e-10
    atol = 1e-12
    print("DREAMS", dlms, "TREAMS", ref)
    mask = ~onp.isclose(dlms, ref, rtol=rtol, atol=atol)
    print("mismatches:", mask.sum())
    print("total:", mask.size)
    assert onp.allclose(dlms, ref, rtol=1e-10, atol=1e-12)


def _redincgamma_ref(n, z):
    # Python reference mirroring the C code, but using cs.incgamma
    singularity = 1e-7
    twicen = 2.0 * n
    r2 = (z.real * z.real + z.imag * z.imag)
    if r2 < 1e-12:
        if twicen == 0.0:
            return -0.5772156649015329 - onp.log(singularity) + 0.5j * onp.pi
        if twicen > 0.0 and (twicen + 2.0) % 4.0 != 0.0:
            res = onp.math.gamma(n) * (1.0 + onp.exp(-1j * onp.pi * n)) / (2.0 * singularity)
        else:
            res = 0.0 + 0.0j
        return res - (-1j) ** (2.0 * n) / n

    if onp.isclose(z.imag, 0.0):
        z = complex(z.real, -0.0)
    return cs.incgamma(n, z) * ((-z) ** (-n))

def test_redincgamma_matches_ref():
    size_loop = 20  
    ns = [-1.5, 0.5,  -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]
    zs = [
        -1.98, 
        1e-3,
        0.001,
       # -1.01,
        1.0,
        -1e-3 + 1.j,
        -2.0 + 0.0j,
        0.1 + 0.2j,
        -0.1 + 0.2j,
    ]

    for n in ns:
        for z in zs:
            # if onp.isclose(z.imag, 0.0):# thie igves matching res
            #    z = complex(z.real, -0.0)
            jax_val = _redincgamma(
                np.array(n, dtype=np.float64),
                np.array(z, dtype=np.complex128),
                size_loop,
            )
            
            ref_val = _redincgamma_ref(n, complex(z))

            if not onp.allclose(
                onp.array(jax_val),
                onp.array(ref_val),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise AssertionError(
                    f"redincgamma mismatch for n={n}, z={z}:\n"
                    f"  JAX = {jax_val}\n"
                    f"  REF = {ref_val}"
                )



def test_incgamma_matches_ref():
    size_loop = 20
    ns = onp.array([
       -0.5,
       -1.5,
       -1,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
    ])

    zs = [
        1.0,
        -0.001-0.j,
      #  -1.,
        0.001,
        3.0,
        -0.1,              # negative real
        -2.,              # more negative
        0.5 + 0.2j,
        -0.3 + 0.7j,
    ]

    for n in ns:
        for z in zs:
            jax.debug.print("z in for loops {}", z)
            #z_c = complex(z)
            if onp.isclose(z.imag, 0.0):
                print("z imag", z.imag, z)
                z_c = complex(z.real, -0.0)
            else:
                z_c = z
            jax.debug.print("comapre z c {} and z {}", z_c, z)
            # Reference Cython implementation
            ref = cs.incgamma(float(n), z_c)
            #ref = z * np.sqrt(np.pi)
            # JAX implementation
            #z_j = np.array(z_c, dtype=np.complex128)

   
            # Option 1: explicit tiny negative imag
            
            #z = z + np.where(is_on_neg_real, -1e-30, 0.0) * 1j
            val = incgamma(n, z, size_loop)
            val_np = onp.array(val)
      
            assert onp.allclose(
                val_np,
                ref,
                rtol=1e-12,
                atol=1e-14,
            ), (
                f"incgamma mismatch for n={n}, z={z_c}:\n"
                f"  JAX   = {val_np}\n"
                f"  Cython= {ref}"
            )


def test_car2pol():
    """Compare dreams.jax_coord.car2pol with treams.special.car2pol
    for scalars, batches, higher dim, and also check JIT equality.
    """
    rng = onp.random.default_rng(1234)

    # --- define test cases (different shapes) ---
    cases = {
        "origin":          onp.array([0.0, 0.0]),               # (2,)
        "unit_x":          onp.array([1.0, 0.0]),               # (2,)
        "unit_y":          onp.array([0.0, 1.0]),               # (2,)
        "random_batch":    rng.normal(size=(100, 2)),           # (100, 2)
        "grid_4x5":        onp.linspace(-2.0, 2.0, 40).reshape(4, 5, 2),  # (4, 5, 2)
    }

    f_jit = jax.jit(car2pol_jax)

    for name, pts in cases.items():
        pts_jax = np.asarray(pts)

        # JAX eager and JIT results
        got_eager = onp.asarray(car2pol_jax(pts_jax))
        got_jit   = onp.asarray(f_jit(pts_jax))

        # Reference treams implementation: loop over last axis = 2
        pts_flat = pts.reshape(-1, 2)
        ref_flat = onp.stack(
            [onp.asarray(car2pol_ref(p)) for p in pts_flat],
            axis=0,
        )
        ref = ref_flat.reshape(pts.shape)

        # Compare JAX vs reference
        onp.testing.assert_allclose(
            got_eager,
            ref,
            rtol=1e-13,
            atol=1e-13,
            err_msg=f"JAX vs treams mismatch in case '{name}'",
        )
        
        # Compare JIT vs eager
        onp.testing.assert_allclose(
            got_jit,
            got_eager,
            rtol=1e-13,
            atol=1e-13,
            err_msg=f"JIT vs eager mismatch in case '{name}'",
        )


if __name__ == "__main__":
    #test_redincgamma_matches_ref()
    #test_incgamma_matches_ref()
    test_lsumsw2d_shift_vmap_matches_ref()
    #test_car2pol_all()
    #test_s_mat_local()
    #test_t_mat_coreshell_parity()
    #test_s_mat_global()
    #test_plane_wave_partial_parity_matches_treams()
    #test_diffr_orders_circle_square_exact()

    #test_t_mat_sphere_helicity()


