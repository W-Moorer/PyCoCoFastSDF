# -*- coding: utf-8 -*-
from __future__ import annotations

"""
demo_gradient_sdf_runtime.py
----------------------------
构建/加载 GradientSDF 后，测试不同查询后端的批量查询效率。
所有参数在文件开头配置，而非命令行。
"""

import time
import numpy as np
from gradient_sdf import GradientSDF

# ===================== 配置区 =====================
OBJ_PATH = None         # 三选一：OBJ / DENSE_PREFIX / NPZ_PATH
DENSE_PREFIX = None
NPZ_PATH = "../gradient_outputs/gear_sparse.npz"

OUT_NPZ = "../gradient_outputs/gear_sparse.npz"
OUT_DENSE_PREFIX = "../gradient_outputs/gear"
ATTACH_DENSE = True

TAU_VOXELS = 6.0
BLOCK_SIZE = 8
DTYPE = "float64"
RESOLUTION = 512

BACKEND = "block_ptr"                       # "index_map" | "block_ptr" | "block_atlas"
N_SAMPLES = 500_000
SEED = 0
# ==================================================

def main():
    # ---- build/load
    if OBJ_PATH is not None:
        print("[build] from OBJ ...")
        g = GradientSDF.from_obj(
            OBJ_PATH,
            out_npz=OUT_NPZ,
            out_dense_prefix=OUT_DENSE_PREFIX,
            keep_dense_in_memory=ATTACH_DENSE or (OUT_DENSE_PREFIX is not None),
            tau=TAU_VOXELS, block_size=BLOCK_SIZE,
            dtype=DTYPE,
            target_resolution=RESOLUTION, max_resolution=RESOLUTION,
            verbose=True,
        )
    elif DENSE_PREFIX is not None:
        print("[load] from dense prefix ...")
        g = GradientSDF.from_dense_prefix(
            DENSE_PREFIX,
            tau=TAU_VOXELS, block_size=BLOCK_SIZE,
            dtype=DTYPE, save_npz=OUT_NPZ,
            attach_dense=ATTACH_DENSE, verbose=True
        )
    elif NPZ_PATH is not None:
        print("[load] from sparse .npz ...")
        g = GradientSDF.from_npz(NPZ_PATH, verbose=True)
    else:
        raise SystemExit("必须提供 OBJ_PATH 或 DENSE_PREFIX 或 NPZ_PATH 之一")

    # ---- backend & warmup
    g.set_query_backend(BACKEND)
    if BACKEND == "index_map":
        g.warmup(index_map=True)
    elif BACKEND == "block_ptr":
        g.warmup(ptr_table=True)
    else:
        g.warmup(atlas=True)

    spec = g.grid_spec
    print(f"[grid] shape={spec.shape}, voxel={spec.voxel}, AABB=({spec.bmin} ~ {spec.bmax})")
    print(f"[hash] block_size={g.block_size}, backend={BACKEND}")

    # ---- random queries
    rng = np.random.default_rng(SEED)
    P = spec.bmin + rng.random((N_SAMPLES, 3)) * (spec.bmax - spec.bmin)

    t0 = time.perf_counter()
    d, n, hit = g.query_points(P, allow_fallback=True)
    dt = time.perf_counter() - t0
    qps = len(P) / max(dt, 1e-12)

    print(f"[query] time={dt:.4f}s, N={len(P):,}, QPS={qps:,.0f}, hit={hit.mean()*100:.2f}%")
    print("depth stats: min={:.3e}, max={:.3e}".format(np.nanmin(d), np.nanmax(d)))
    print("normal L2 mean={:.6f}".format(float(np.nanmean(np.linalg.norm(n, axis=1)))))

if __name__ == "__main__":
    main()
