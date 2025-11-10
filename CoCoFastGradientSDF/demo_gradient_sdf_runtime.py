# -*- coding: utf-8 -*-
"""
Mini demo for Gradient-SDF runtime
- No CLI: parameters are defined at the top of this file.
- Verifies:
  (1) Dense single-voxel Taylor query correctness (centers & batch consistency)
  (2) Nearest-surface-point identity under the Taylor model
  (3) Hashed storage equals dense within narrow band; behaves with/without fallback

Prereq: you already generated
  <prefix>_sdf.npy, <prefix>_grad.npy, <prefix>_meta.json
with gradient_sdf_tools_cpu_bk1.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

# ----------------------- USER PARAMETERS -----------------------
PREFIX = Path("./gradient_outputs/gear")   # prefix without suffix; will look for *_sdf.npy/_grad.npy/_meta.json
RNG_SEED = 42
N_RANDOM = 200               # random samples for checks
BLOCK_SIZE = 8
TAU_VOXELS = 6.0            # narrow band half-width in voxels (converted to world units)
CENTER_EPS = 1e-9           # numerical tolerance for center checks
UNIT_EPS = 1e-6             # ||g|| ≈ 1 tolerance
TAYLOR_EPS = 1e-7           # Taylor equalities tolerance on doubles

# ----------------------- IMPORT RUNTIME ------------------------
from gradient_sdf_runtime import (
    load_dense_by_prefix,
    HashedGradientSDF,
)


def _rand_points_in_box(rng, bmin, bmax, n):
    return bmin + rng.random((n, 3)) * (bmax - bmin)


def check_centers_and_batch(vol):
    """At voxel centers, Taylor d==psi & g==grad. Also batch==scalar outputs."""
    rng = np.random.default_rng(RNG_SEED)
    nx, ny, nz = vol.spec.shape
    # pick up to 50 random indices well inside the grid
    I = rng.integers(0, nx, size=50)
    J = rng.integers(0, ny, size=50)
    K = rng.integers(0, nz, size=50)
    ijk = np.stack([I, J, K], axis=1)
    vj = vol.spec.center_from_index(ijk)

    # scalar checks
    max_d_err = 0.0
    max_g_err = 0.0
    for n in range(len(ijk)):
        i, j, k = int(ijk[n, 0]), int(ijk[n, 1]), int(ijk[n, 2])
        d, g, vj_n, ijk_n = vol.taylor_query(vj[n])
        psi = float(vol.sdf[i, j, k])
        grad = vol.grad[i, j, k]
        max_d_err = max(max_d_err, abs(d - psi))
        max_g_err = max(max_g_err, float(np.linalg.norm(g - grad, ord=np.inf)))
        assert np.allclose(vj_n, vj[n], atol=CENTER_EPS)
        assert np.array_equal(ijk_n, ijk[n])
    print(f"[dense] center check: max |d-psi|={max_d_err:.3e}, max |g-grad|_inf={max_g_err:.3e}")
    assert max_d_err < TAYLOR_EPS and max_g_err < TAYLOR_EPS

    # batch vs scalar
    P = _rand_points_in_box(rng, vol.spec.bmin, vol.spec.bmax, N_RANDOM)
    d_b, g_b, vj_b, ijk_b, mask = vol.taylor_query_batch(P)
    # compute scalar for 'valid' only
    idxs = np.where(mask)[0]
    d_s = np.empty_like(d_b)
    d_s[:] = np.nan
    g_s = np.empty_like(g_b)
    g_s[:] = np.nan
    for n in idxs:
        dn, gn, _, _ = vol.taylor_query(P[n])
        d_s[n], g_s[n] = dn, gn
    max_d = np.nanmax(np.abs(d_s - d_b))
    max_g = np.nanmax(np.linalg.norm(g_s - g_b, axis=1))
    print(f"[dense] batch vs scalar: max |Δd|={max_d:.3e}, max ||Δg||={max_g:.3e}")
    assert max_d < 1e-10 and max_g < 1e-10

    # unit length
    g = vol.grad.reshape(-1, 3)
    norms = np.linalg.norm(g, axis=1)
    max_unit_err = float(np.max(np.abs(norms - 1.0)))
    print(f"[dense] unit gradient: max |‖g‖-1|={max_unit_err:.3e}")
    assert max_unit_err < UNIT_EPS


def check_nearest_surface_point(vol):
    """For voxels with small |psi|, p_s=v-psi*g should yield Taylor d≈0 with same ijk."""
    # pick voxels with |psi| <= alpha * min(voxel)
    alpha = 0.2
    vmin = float(np.min(vol.spec.voxel))
    thresh = alpha * vmin
    sdf = vol.sdf
    cand = np.argwhere(np.abs(sdf) <= thresh)
    if len(cand) == 0:
        print("[dense] nearest-surface: skipped (no small-psi voxels; try a looser alpha)")
        return
    rng = np.random.default_rng(RNG_SEED + 1)
    idx = cand[rng.integers(0, len(cand), size=min(50, len(cand)))]

    ok = 0
    max_abs_d = 0.0
    for n in range(len(idx)):
        ps = vol.nearest_surface_point_from_index(idx[n])
        d, g, vj, ijk = vol.taylor_query(ps)
        same = np.array_equal(ijk, idx[n])
        if same:
            ok += 1
            max_abs_d = max(max_abs_d, abs(d))
    print(f"[dense] nearest-surface: same-ijk count={ok}/{len(idx)}, max |d(ps)|={max_abs_d:.3e}")
    # Don't hard fail on same-ijk ratio (geometry dependent), but Taylor value should be tiny if same cell.
    assert max_abs_d < 1e-8 or ok == 0


def check_hash_consistency(vol):
    """Inside narrow band, hashed query equals dense Taylor; outside, fallback works."""
    vmin = float(np.min(vol.spec.voxel))
    tau = TAU_VOXELS * vmin
    h = HashedGradientSDF.build_from_dense(vol, tau=tau, block_size=BLOCK_SIZE, dtype="float64")

    # points near centers in-band
    idx_in = np.argwhere(np.abs(vol.sdf) <= tau)
    if len(idx_in) == 0:
        print("[hash] in-band: skipped (no voxels within tau)")
    else:
        rng = np.random.default_rng(RNG_SEED + 2)
        M = min(100, len(idx_in))
        pick = idx_in[rng.integers(0, len(idx_in), size=M)]
        # jitter within the same voxel cell to ensure same ijk
        offs = (rng.random((M, 3)) - 0.5) * (vol.spec.voxel * 0.9)
        vj = vol.spec.center_from_index(pick)
        P = vj + offs
        d_d, g_d, _, _, _ = vol.taylor_query_batch(P)
        d_h, g_h, _, _, hit = h.taylor_query_batch(P, allow_fallback=False)
        assert hit.all(), "in-band points must hit hashed storage"
        max_dd = float(np.max(np.abs(d_d - d_h)))
        max_gg = float(np.max(np.linalg.norm(g_d - g_h, axis=1)))
        print(f"[hash] in-band equality: max |Δd|={max_dd:.3e}, max ||Δg||={max_gg:.3e}")
        assert max_dd < 1e-10 and max_gg < 1e-10

    # points out of band
    idx_out = np.argwhere(np.abs(vol.sdf) > tau)
    if len(idx_out) == 0:
        print("[hash] out-of-band: skipped (all voxels inside tau)")
        return
    rng = np.random.default_rng(RNG_SEED + 3)
    M = min(50, len(idx_out))
    pick = idx_out[rng.integers(0, len(idx_out), size=M)]
    vj = vol.spec.center_from_index(pick)
    # stay within the cell
    offs = (rng.random((M, 3)) - 0.5) * (vol.spec.voxel * 0.9)
    P = vj + offs

    from gradient_sdf_runtime import GradientSDFVolume  # type: ignore
    h.attach_dense_fallback(vol)

    # no fallback
    d_h1, g_h1, _, _, hit1 = h.taylor_query_batch(P, allow_fallback=False)
    assert (~hit1).all(), "out-of-band should miss when fallback=False"

    # with fallback equals dense
    d_d, g_d, _, _, _ = vol.taylor_query_batch(P)
    d_h2, g_h2, _, _, hit2 = h.taylor_query_batch(P, allow_fallback=True)
    assert hit2.all(), "fallback should recover all"
    max_dd = float(np.nanmax(np.abs(d_d - d_h2)))
    max_gg = float(np.nanmax(np.linalg.norm(g_d - g_h2, axis=1)))
    print(f"[hash] out-of-band fallback: max |Δd|={max_dd:.3e}, max ||Δg||={max_gg:.3e}")
    assert max_dd < 1e-10 and max_gg < 1e-10


if __name__ == "__main__":
    vol = load_dense_by_prefix(PREFIX)
    print(f"Grid shape={vol.spec.shape}, voxel={vol.spec.voxel}, bmin={vol.spec.bmin}, bmax={vol.spec.bmax}")

    check_centers_and_batch(vol)
    check_nearest_surface_point(vol)
    check_hash_consistency(vol)

    print("All checks passed ✓")

