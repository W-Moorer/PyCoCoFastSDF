# -*- coding: utf-8 -*-
from __future__ import annotations

"""
benchmark_gsdf_vs_trilinear.py
------------------------------
对比 Taylor(dense) / Trilinear(ψ) / Taylor(hashed) 的速度与精度；
参数在文件开头配置。
"""

import time
import numpy as np
from gradient_sdf import GradientSDF, DenseGradientSDF, HashedGradientSDF, GridSpec, taylor_query_batch_dense

# ===================== 配置区 =====================
OBJ_PATH = None   # 若提供 DENSE_PREFIX，可更快
DENSE_PREFIX = "../gradient_outputs/gear"
NPZ_PATH = "../gradient_outputs/gear.npz"                       # 仅 .npz 不足以完成全部对比，此处会改为从 OBJ 构建

TAU_VOXELS = 6.0
BLOCK_SIZE = 8
DTYPE = "float64"
RESOLUTION = 512

N_SAMPLES = 120000
REPEATS = 3
# ==================================================

def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x*1024.0:.2f} TiB"

def trilinear_value_and_grad(sdf: np.ndarray, spec: GridSpec, P: np.ndarray):
    P = np.asarray(P, dtype=np.float64)
    voxel = spec.voxel
    U = (P - spec.bmin) / voxel - 0.5
    i0 = np.floor(U[:, 0]).astype(np.int64)
    j0 = np.floor(U[:, 1]).astype(np.int64)
    k0 = np.floor(U[:, 2]).astype(np.int64)
    tx = U[:, 0] - i0; ty = U[:, 1] - j0; tz = U[:, 2] - k0
    i1 = i0 + 1; j1 = j0 + 1; k1 = k0 + 1
    nx, ny, nz = sdf.shape
    mask = (i0 >= 0) & (i1 < nx) & (j0 >= 0) & (j1 < ny) & (k0 >= 0) & (k1 < nz)
    d = np.full(len(P), np.nan); g = np.full((len(P), 3), np.nan)
    if not np.any(mask):
        return d, g, mask
    idx = np.where(mask)[0]
    i0_, j0_, k0_ = i0[idx], j0[idx], k0[idx]
    i1_, j1_, k1_ = i1[idx], j1[idx], k1[idx]
    tx_, ty_, tz_ = tx[idx], ty[idx], tz[idx]
    c000 = sdf[i0_, j0_, k0_]; c100 = sdf[i1_, j0_, k0_]
    c010 = sdf[i0_, j1_, k0_]; c110 = sdf[i1_, j1_, k0_]
    c001 = sdf[i0_, j0_, k1_]; c101 = sdf[i1_, j0_, k1_]
    c011 = sdf[i0_, j1_, k1_]; c111 = sdf[i1_, j1_, k1_]
    wx0 = 1.0 - tx_; wy0 = 1.0 - ty_; wz0 = 1.0 - tz_
    wx1 = tx_;       wy1 = ty_;       wz1 = tz_
    d_loc = (
        c000*wx0*wy0*wz0 + c100*wx1*wy0*wz0 + c010*wx0*wy1*wz0 + c110*wx1*wy1*wz0 +
        c001*wx0*wy0*wz1 + c101*wx1*wy0*wz1 + c011*wx0*wy1*wz1 + c111*wx1*wy1*wz1
    )
    ddx = ((wy0*wz0)*(c100 - c000) + (wy1*wz0)*(c110 - c010) + (wy0*wz1)*(c101 - c001) + (wy1*wz1)*(c111 - c011)) / voxel[0]
    ddy = ((wx0*wz0)*(c010 - c000) + (wx1*wz0)*(c110 - c100) + (wx0*wz1)*(c011 - c001) + (wx1*wz1)*(c111 - c101)) / voxel[1]
    ddz = ((wx0*wy0)*(c001 - c000) + (wx1*wy0)*(c101 - c100) + (wx0*wy1)*(c011 - c010) + (wx1*wy1)*(c111 - c110)) / voxel[2]
    d[idx] = d_loc; g[idx,0]=ddx; g[idx,1]=ddy; g[idx,2]=ddz
    return d, g, mask

def sample_points_in_cells(dense: DenseGradientSDF, n_samples: int, tau_world: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    sdf = dense.sdf; nx, ny, nz = sdf.shape
    mask = (np.abs(sdf) <= tau_world)
    valid = mask[:nx-1, :ny-1, :nz-1]
    cand = np.argwhere(valid)
    if len(cand) == 0:
        cand = np.argwhere(np.ones((nx-1, ny-1, nz-1), dtype=bool))
    sel = cand[rng.integers(0, len(cand), size=min(n_samples, len(cand)))] if len(cand)>0 else np.empty((0,3),dtype=int)
    t = rng.random((len(sel), 3)) if len(sel)>0 else np.empty((0,3))
    P = dense.spec.bmin + dense.spec.voxel * (sel.astype(np.float64) + t + 0.5) if len(sel)>0 else dense.spec.bmin.reshape(1,3)
    return P

def bench(fn, *args, repeats=3, **kwargs):
    times = []; out = None
    for _ in range(repeats):
        t0 = time.perf_counter(); out = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return min(times), out

def main():
    # ---- build/load with dense
    if DENSE_PREFIX:
        g = GradientSDF.from_dense_prefix(DENSE_PREFIX, tau=TAU_VOXELS,
                                          block_size=BLOCK_SIZE, dtype=DTYPE,
                                          attach_dense=True, verbose=True)
    elif NPZ_PATH:
        g = GradientSDF.from_obj(OBJ_PATH, tau=TAU_VOXELS, block_size=BLOCK_SIZE,
                                 dtype=DTYPE, target_resolution=RESOLUTION,
                                 max_resolution=RESOLUTION, verbose=True)
    else:
        g = GradientSDF.from_obj(OBJ_PATH, tau=TAU_VOXELS, block_size=BLOCK_SIZE,
                                 dtype=DTYPE, target_resolution=RESOLUTION,
                                 max_resolution=RESOLUTION, verbose=True)

    dense = g.core._dense
    if dense is None:
        raise SystemExit("需要稠密数据用于对比，请从 OBJ 构建并 keep_dense_in_memory=True 或使用 DENSE_PREFIX。")

    spec = dense.spec
    voxel_min = float(np.min(spec.voxel)); tau_world = TAU_VOXELS * voxel_min
    print(f"[grid] shape={spec.shape}, voxel={spec.voxel}, tau(world)={tau_world:g}")

    # payload stats
    h = HashedGradientSDF.build_from_dense(dense, tau=tau_world, block_size=BLOCK_SIZE, dtype="float64")
    payload = 0; voxels = 0
    for blk in h.blocks.values():
        payload += blk.locs.nbytes + blk.psi.nbytes + blk.g.nbytes + np.asarray(blk.base, dtype=np.int32).nbytes
        voxels += len(blk.psi)
    total = int(np.prod(spec.shape))
    print("\n================ 资源开销 ================")
    print(f"Dense memory (runtime arrays): {human_bytes(dense.sdf.nbytes + dense.grad.nbytes)}")
    print(f"Hashed payload (ψ,ĝ,locs,bases): {human_bytes(payload)}")
    print(f"Hashed voxels: {voxels} / {total}  ({voxels/total*100:.4f}% of dense)")

    # sample points
    P = sample_points_in_cells(dense, n_samples=N_SAMPLES, tau_world=tau_world, seed=0)
    if len(P)==0:
        raise RuntimeError("没有有效采样点，请增大 TAU_VOXELS 或降低分辨率。")

    # benchmarks
    t_taylor, (d_t, g_t, *_ ) = bench(taylor_query_batch_dense, dense, P, repeats=REPEATS)
    t_tri, (d_tri, g_tri, mask_tri) = bench(trilinear_value_and_grad, dense.sdf, spec, P, repeats=REPEATS)
    t_hash, (d_h, g_h, *_ ) = bench(h.taylor_query_batch, P, repeats=REPEATS, allow_fallback=False)

    valid = np.isfinite(d_t) & mask_tri
    if not np.any(valid):
        raise RuntimeError("没有有效采样点，用 TAU_VOXELS 调大或 N_SAMPLES 减小。")

    def deg(a,b):
        a = a / np.maximum(1e-12, np.linalg.norm(a, axis=1, keepdims=True))
        b = b / np.maximum(1e-12, np.linalg.norm(b, axis=1, keepdims=True))
        cos = np.sum(a*b, axis=1).clip(-1.0, 1.0)
        return np.degrees(np.arccos(cos))

    dv = np.abs(d_tri[valid] - d_t[valid])
    ang = deg(g_tri[valid], g_t[valid])

    def stats(x):
        return dict(mean=float(np.mean(x)), median=float(np.median(x)), p95=float(np.percentile(x,95)), max=float(np.max(x)))

    dv_s = stats(dv); ang_s = stats(ang)

    print("\n================ 查询效率 ================")
    qps_taylor = len(P)/t_taylor; qps_tri = len(P)/t_tri; qps_hash = len(P)/t_hash
    print(f"Taylor (dense)  : {t_taylor:.4f}s  -> {qps_taylor:,.0f} q/s  (1.00x)")
    print(f"Trilinear (ψ)   : {t_tri:.4f}s  -> {qps_tri:,.0f} q/s  ({qps_tri/qps_taylor:.2f}x vs Taylor)")
    print(f"Taylor (hashed) : {t_hash:.4f}s  -> {qps_hash:,.0f} q/s  ({qps_hash/qps_taylor:.2f}x vs Taylor)")

    print("\n================ 查询精度 ================")
    print(f"Value |d_tri - d_taylor|  (on {valid.sum()} pts): mean={dv_s['mean']:.3e}, median={dv_s['median']:.3e}, p95={dv_s['p95']:.3e}, max={dv_s['max']:.3e}")
    print(f"Grad  angle(tri, taylor)° (on {valid.sum()} pts): mean={ang_s['mean']:.3f}, median={ang_s['median']:.3f}, p95={ang_s['p95']:.3f}, max={ang_s['max']:.3f}")

if __name__ == "__main__":
    main()
