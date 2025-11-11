# -*- coding: utf-8 -*-
"""
demo_gradient_sdf_all_in_one.py
================================
与合并后的 gradient_sdf.py API 对齐的端到端 demo
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from gradient_sdf import (
    GradientSDF, DenseGradientSDF, HashedGradientSDF,
    GridSpec, taylor_query_batch_dense
)

# ------------------------ 配置参数 ------------------------
OBJ_PATH = "../obj_library/gear.obj"
DENSE_PREFIX = None
NPZ_PATH = None

OUT_NPZ = "../gradient_outputs/gear_sparse.npz"
OUT_DENSE_PREFIX = "../gradient_outputs/gear"
ATTACH_DENSE = True

TAU_VOXELS = 6.0
BLOCK_SIZE = 8
DTYPE = "float64"
REVOLUTION = 512

N_SAMPLES = 120000
TIME_REPEATS = 3

ENABLE_VIZ = True

def main():
    from time import perf_counter

    g = None
    dense = None

    if OBJ_PATH is not None:
        print("[build] from OBJ...")
        g = GradientSDF.from_obj(
            OBJ_PATH,
            out_npz=OUT_NPZ,
            out_dense_prefix=OUT_DENSE_PREFIX,
            keep_dense_in_memory=ATTACH_DENSE or (OUT_DENSE_PREFIX is not None),
            tau=TAU_VOXELS, block_size=BLOCK_SIZE,
            dtype=DTYPE,
            target_resolution=REVOLUTION,
            max_resolution=REVOLUTION,
            verbose=True
        )
        dense = g.core._dense
    elif DENSE_PREFIX is not None:
        print("[load] from dense prefix...")
        g = GradientSDF.from_dense_prefix(
            DENSE_PREFIX,
            tau=TAU_VOXELS, block_size=BLOCK_SIZE,
            dtype=DTYPE,
            save_npz=OUT_NPZ,
            attach_dense=ATTACH_DENSE,
            verbose=True
        )
        dense = g.core._dense
    elif NPZ_PATH is not None:
        print("[load] from sparse .npz...")
        g = GradientSDF.from_npz(NPZ_PATH, verbose=True)
        dense = None
    else:
        raise SystemExit("必须至少提供 OBJ_PATH 或 DENSE_PREFIX 或 NPZ_PATH 之一。")

    spec = g.grid_spec
    print(f"[grid] shape={spec.shape}, voxel={spec.voxel}, bmin={spec.bmin}, bmax={spec.bmax}")
    print(f"[hash] block_size={g.block_size}, backend=block_ptr")
    g.set_query_backend("block_ptr")
    g.warmup(ptr_table=True)

    # 稠密相关校验与基准
    voxel_min = float(np.min(spec.voxel)); tau_world = TAU_VOXELS * voxel_min
    if dense is not None:
        from math import isfinite

        # 简要一致性自检
        rng = np.random.default_rng(0)
        idx = rng.integers(0, np.prod(spec.shape), size=16)
        I = idx // (spec.shape[1]*spec.shape[2])
        J = (idx // spec.shape[2]) % spec.shape[1]
        K = idx % spec.shape[2]
        VJ = dense.spec.center_from_index(np.stack([I, J, K], axis=1))
        d1, g1, *_ = taylor_query_batch_dense(dense, VJ)
        print("[check] center pts ok? max|psi-d|=", float(np.max(np.abs(dense.sdf[I,J,K]-d1))))

    # 统一 API 快速测试
    print("\n[api] quick check on GradientSDF.query_points...")
    P = np.array([[0,0,0], spec.bmin + 0.25*(spec.bmax-spec.bmin), spec.bmax - 0.25*(spec.bmax-spec.bmin)], dtype=np.float64)
    d, n, hit = g.query_points(P, allow_fallback=(dense is not None))
    print("depth=", d); print("||normal||=", np.linalg.norm(n, axis=1)); print("hit=", hit.astype(int).tolist())

    # 可视化（可选）
    if ENABLE_VIZ:
        try:
            import pyvista as pv
            from time import perf_counter as _t
            imin = np.array([+1<<30]*3, dtype=np.int64); imax = np.array([-1<<30]*3, dtype=np.int64)
            for blk in g.core.blocks.values():
                if blk.locs.size == 0: continue
                base = np.asarray(blk.base, dtype=np.int64)
                ijk = base[None,:] + blk.locs.astype(np.int64)
                imin = np.minimum(imin, ijk.min(axis=0)); imax = np.maximum(imax, ijk.max(axis=0))
            imin = np.maximum(imin - 2, 0); imax = np.minimum(imax + 1 + 2, np.array(g.core.spec.shape, dtype=np.int64))

            spec = g.core.spec; voxel = spec.voxel
            ii = np.arange(int(imin[0]), int(imax[0]), 1, dtype=np.int64)
            jj = np.arange(int(imin[1]), int(imax[1]), 1, dtype=np.int64)
            kk = np.arange(int(imin[2]), int(imax[2]), 1, dtype=np.int64)
            X, Y, Z = np.meshgrid(ii, jj, kk, indexing='ij')
            ijk_nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)
            P_world = spec.bmin + voxel * ijk_nodes

            g.core.set_query_backend("index_map"); g.core.warmup(index_map=True)
            t0 = _t()
            d, *_ , hit = g.core.taylor_query_batch(P_world, allow_fallback=False)
            print(f"[viz] sample nodes={len(P_world):,}, time={_t()-t0:.3f}s, hit={hit.mean()*100:.2f}%")
            grid = d.reshape(len(ii), len(jj), len(kk))
            grid[~hit.reshape(grid.shape)] = abs(tau_world)

            ug = pv.ImageData()
            ug.dimensions = grid.shape
            ug.origin = tuple((spec.bmin + voxel * imin).astype(float))
            ug.spacing = tuple((voxel * 1).astype(float))
            ug.point_data["sdf"] = grid.ravel(order="F")

            surf = ug.contour(isosurfaces=[0.0], scalars="sdf")
            pl = pv.Plotter(window_size=(1200, 900))
            pl.add_mesh(surf, color="white", smooth_shading=True, specular=0.1, specular_power=10)
            pl.add_axes(); pl.show_bounds(grid="front", location="outer", all_edges=True)
            pl.add_text("GSDF Zero-Isosurface (level=0)", font_size=12)
            pl.show()
        except Exception as e:
            print("[viz] 跳过可视化：", e)

    print("\\nAll checks done ✓")

if __name__ == "__main__":
    main()
