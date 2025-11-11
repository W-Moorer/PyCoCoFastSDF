# -*- coding: utf-8 -*-
from __future__ import annotations

"""
demo_gsdf_zero_isosurface.py
----------------------------
仅用稀疏哈希在窄带包围盒上采样体节点，提取 0 等值面进行可视化（需要安装 pyvista）。
所有参数在文件开头配置。
"""

import numpy as np
from gradient_sdf import GradientSDF

# ===================== 配置区 =====================
OBJ_PATH = "../obj_library/gear.obj"     # 三选一：OBJ / DENSE_PREFIX / NPZ_PATH
DENSE_PREFIX = None
NPZ_PATH = None

TAU_VOXELS = 6.0
BLOCK_SIZE = 8
DTYPE = "float64"
RESOLUTION = 512

STEP = 1              # 节点采样步长
MARGIN = 2            # 窄带包围盒外扩（体素）
# ==================================================

def main():
    # 构建/加载
    if NPZ_PATH:
        g = GradientSDF.from_npz(NPZ_PATH, verbose=True)
    elif DENSE_PREFIX:
        g = GradientSDF.from_dense_prefix(DENSE_PREFIX, tau=TAU_VOXELS,
                                          block_size=BLOCK_SIZE, dtype=DTYPE,
                                          attach_dense=True, verbose=True)
    else:
        g = GradientSDF.from_obj(
            OBJ_PATH,
            out_npz=None, out_dense_prefix=None, keep_dense_in_memory=False,
            tau=TAU_VOXELS, block_size=BLOCK_SIZE, dtype=DTYPE,
            target_resolution=RESOLUTION, max_resolution=RESOLUTION,
            verbose=True)

    core = g.core
    core.set_query_backend("index_map"); core.warmup(index_map=True)

    try:
        import pyvista as pv
    except Exception as e:
        print("[viz] PyVista 未安装；pip install pyvista")
        return

    # 计算窄带包围盒
    imin = np.array([+1<<30]*3, dtype=np.int64); imax = np.array([-1<<30]*3, dtype=np.int64)
    for blk in core.blocks.values():
        if blk.locs.size == 0: continue
        base = np.asarray(blk.base, dtype=np.int64)
        ijk = base[None,:] + blk.locs.astype(np.int64)
        imin = np.minimum(imin, ijk.min(axis=0)); imax = np.maximum(imax, ijk.max(axis=0))
    imin = np.maximum(imin - MARGIN, 0); imax = np.minimum(imax + 1 + MARGIN, np.array(core.spec.shape, dtype=np.int64))

    # 节点采样
    spec = core.spec; voxel = spec.voxel
    ii = np.arange(int(imin[0]), int(imax[0]), int(STEP), dtype=np.int64)
    jj = np.arange(int(imin[1]), int(imax[1]), int(STEP), dtype=np.int64)
    kk = np.arange(int(imin[2]), int(imax[2]), int(STEP), dtype=np.int64)
    X, Y, Z = np.meshgrid(ii, jj, kk, indexing='ij')
    ijk_nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)
    P_world = spec.bmin + voxel * ijk_nodes

    d, *_ , hit = core.taylor_query_batch(P_world, allow_fallback=False)
    grid = d.reshape(len(ii), len(jj), len(kk))
    grid[~hit.reshape(grid.shape)] = np.abs(TAU_VOXELS) * float(np.min(voxel))  # miss 处放远

    ug = pv.ImageData()
    ug.dimensions = grid.shape
    ug.origin = tuple((spec.bmin + voxel * imin).astype(float))
    ug.spacing = tuple((voxel * STEP).astype(float))
    ug.point_data["sdf"] = grid.ravel(order="F")

    surf = ug.contour(isosurfaces=[0.0], scalars="sdf")
    pl = pv.Plotter(window_size=(1200, 900))
    pl.add_mesh(surf, color="white", smooth_shading=True, specular=0.1, specular_power=10)
    pl.add_axes(); pl.show_bounds(grid="front", location="outer", all_edges=True)
    pl.add_text("GSDF Zero-Isosurface", font_size=12)
    pl.show()

if __name__ == "__main__":
    main()
