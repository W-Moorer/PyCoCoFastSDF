# -*- coding: utf-8 -*-
from __future__ import annotations

"""
demo_gradient_sdf_tools_cpu_base_func.py
----------------------------------------
从 OBJ 生成稠密 SDF+梯度并保存三件套，同时构建稀疏；
验证体素中心处 Taylor 与存储一致。参数在文件开头配置。
"""

import numpy as np
from gradient_sdf import GradientSDF, taylor_query_batch_dense

# ===================== 配置区 =====================
OBJ_PATH = "../obj_library/gear.obj"
OUT_DENSE_PREFIX = "../gradient_outputs/gear"
OUT_NPZ = "../gradient_outputs/gear_sparse.npz"

TAU_VOXELS = 6.0
BLOCK_SIZE = 8
DTYPE = "float64"
RESOLUTION = 512

SAMPLE_CENTERS = 200
# ==================================================

def main():
    print("[build] from OBJ -> dense + grad + hashed ...")
    g = GradientSDF.from_obj(
        OBJ_PATH,
        out_npz=OUT_NPZ,
        out_dense_prefix=OUT_DENSE_PREFIX,
        keep_dense_in_memory=True,
        tau=TAU_VOXELS, block_size=BLOCK_SIZE,
        dtype=DTYPE,
        target_resolution=RESOLUTION, max_resolution=RESOLUTION,
        verbose=True,
    )
    dense = g.core._dense
    spec = dense.spec
    print(f"[grid] shape={spec.shape}, voxel={spec.voxel}, AABB=({spec.bmin} ~ {spec.bmax})")

    # 体素中心一致性检查
    rng = np.random.default_rng(0)
    nx, ny, nz = spec.shape
    I = rng.integers(0, nx, size=min(SAMPLE_CENTERS, nx))
    J = rng.integers(0, ny, size=min(SAMPLE_CENTERS, ny))
    K = rng.integers(0, nz, size=min(SAMPLE_CENTERS, nz))
    ijk = np.stack([I, J, K], axis=1)
    VJ = spec.center_from_index(ijk)
    d, gvec, vj, ijk_out, hit = taylor_query_batch_dense(dense, VJ)

    max_d_err = np.max(np.abs(d - dense.sdf[I, J, K]))
    max_g_err = np.max(np.linalg.norm(gvec - dense.grad[I, J, K], axis=1))
    print(f"[check] center: max|d-psi|={max_d_err:.3e}, max||g-grad||={max_g_err:.3e}")

    print("[io] saved:\n  - dense: {}_{{sdf,grad,meta}}.*\n  - sparse npz: {}".format(
        OUT_DENSE_PREFIX, OUT_NPZ))
    print("done ✓")

if __name__ == "__main__":
    main()
