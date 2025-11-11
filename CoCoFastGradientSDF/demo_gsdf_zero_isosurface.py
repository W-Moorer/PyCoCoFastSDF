# -*- coding: utf-8 -*-
"""
零等值面可视化（PyVista 交互版）
================================
- 仅使用 *压缩的梯度SDF稀疏哈希* 做查询（默认不回退稠密）
- 在窄带包围盒内构建规则采样网格，调用 `taylor_query_batch` 取值
- 用 PyVista 的 contour 交互显示等值面（含滑条实时调阈值）

依赖：numpy, pyvista  (若缺少: pip install pyvista)
可选：scipy(非必须)
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Tuple
import numpy as np

import pyvista as pv

from gradient_sdf_runtime import (
    load_dense_by_prefix,
    HashedGradientSDF,
)

# ------------------------- 参数（无 CLI） -------------------------
# 数据前缀：假设存在 <prefix>_sdf.npy, <prefix>_grad.npy, <prefix>_meta.json
PREFIX = Path("./gradient_outputs/gear")    # ← 改成你的前缀（不带 _sdf.npy）

# 构建哈希的参数（保持与你基准一致）
TAU = 10.0
BLOCK_SIZE = 8
HASH_DTYPE = "float64"  # 建议 float32：更小载荷，交互更流畅（必要时可改为 "float64"）

# 采样参数
MC_STEP = 1         # 节点降采样步长；>1 可加速但表面更粗
MARGIN_VOX = 2      # 在窄带 bbox 基础上向外扩（体素数）
ALLOW_FALLBACK = False  # 是否允许 miss 回退到稠密（只看压缩内容则 False）

# ------------------------- 工具函数 -------------------------

def compute_band_bbox(h: HashedGradientSDF) -> Tuple[np.ndarray, np.ndarray]:
    """返回窄带体素的 [imin,jmin,kmin], [imax,jmax,kmax]（*节点*索引，右开区间）。"""
    imin = np.array([+1<<30, +1<<30, +1<<30], dtype=np.int64)
    imax = np.array([-1<<30, -1<<30, -1<<30], dtype=np.int64)
    for blk in h.blocks.values():
        if blk.locs.size == 0:
            continue
        base = np.asarray(blk.base, dtype=np.int64)
        locs = blk.locs.astype(np.int64)
        ijk = base[None, :] + locs  # (M,3) 绝对体素中心索引
        imin = np.minimum(imin, ijk.min(axis=0))
        imax = np.maximum(imax, ijk.max(axis=0))
    # 扩成节点索引范围：[min, max+1]，再加 margin，并截断到全局 shape
    imin = np.maximum(imin - MARGIN_VOX, 0)
    imax = imax + 1 + MARGIN_VOX
    imax = np.minimum(imax, np.array(h.spec.shape, dtype=np.int64))
    return imin, imax


def sample_scalar_field(h: HashedGradientSDF, imin: np.ndarray, imax: np.ndarray, step: int):
    """在规则 *节点* 网格上采样 SDF 值。
    返回 (grid[nx,ny,nz], origin[3], spacing[3])，均为世界坐标系。
    """
    spec = h.spec
    voxel = spec.voxel

    ii = np.arange(int(imin[0]), int(imax[0]), int(step), dtype=np.int64)
    jj = np.arange(int(imin[1]), int(imax[1]), int(step), dtype=np.int64)
    kk = np.arange(int(imin[2]), int(imax[2]), int(step), dtype=np.int64)
    nx, ny, nz = len(ii), len(jj), len(kk)

    # 节点的世界坐标（注意：节点坐标=体素左下角，不是中心）
    origin = (spec.bmin + voxel * np.array([ii[0], jj[0], kk[0]], dtype=np.float64)).astype(np.float64)
    spacing = (voxel * step).astype(np.float64)

    # 构造所有节点的世界坐标批量查询
    X, Y, Z = np.meshgrid(ii, jj, kk, indexing='ij')
    ijk_nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)
    P_world = spec.bmin + voxel * ijk_nodes

    # 查询（默认仅压缩内容，不回退）
    h.set_query_backend('index_map')
    h.warmup(index_map=True)
    t0 = time.time()
    d, _, _, _, hit = h.taylor_query_batch(P_world, allow_fallback=ALLOW_FALLBACK)
    t1 = time.time()
    print(f"[query] nodes={len(P_world):,}, time={t1-t0:.3f}s, hit={hit.mean()*100:.2f}%")

    grid = d.reshape(nx, ny, nz)
    if not ALLOW_FALLBACK:
        # 对 miss 赋一个远离 0 的正值，避免虚假零交叉
        grid[~hit.reshape(nx, ny, nz)] = abs(TAU)
    return grid, origin, spacing


# ------------------------- PyVista 可视化 -------------------------

def visualize_with_pyvista(grid: np.ndarray, origin, spacing):
    """把采样出来的标量场塞进 ImageData，然后用 contour(0) 交互显示。"""
    nx, ny, nz = grid.shape
    ug = pv.ImageData()
    ug.dimensions = (nx, ny, nz)
    ug.origin = tuple(origin)
    ug.spacing = tuple(spacing)

    ug.point_data.clear()
    ug.point_data['sdf'] = grid.ravel(order='F')

    surf = ug.contour(isosurfaces=[0.0], scalars='sdf')

    pl = pv.Plotter(window_size=(1200, 900))
    pl.add_mesh(surf, color='white', smooth_shading=True, specular=0.1, specular_power=10)
    pl.add_axes()
    pl.show_bounds(grid='front', location='outer', all_edges=True)
    
    # 使用 Times 字体显示标题
    pl.add_text('GSDF Zero-Isosurface (level=0)', font_size=12, font='times')
    
    # 隐藏方向指示器（尝试1）
    pl.hide_axes()  # 这会隐藏所有轴，包括方向指示器

    pl.camera_position = 'xy'
    pl.show()


# ------------------------- 主流程 -------------------------
if __name__ == "__main__":
    # 1) 读取稠密构建哈希（也可以替换为你现成的哈希加载）
    vol = load_dense_by_prefix(PREFIX)
    h = HashedGradientSDF.build_from_dense(vol, tau=TAU, block_size=BLOCK_SIZE, dtype=HASH_DTYPE)

    # 2) 取窄带包围盒并采样标量场
    imin, imax = compute_band_bbox(h)
    print(f"band node bbox: imin={imin.tolist()}, imax={imax.tolist()}, step={MC_STEP}")
    grid, origin, spacing = sample_scalar_field(h, imin, imax, step=MC_STEP)
    print(f"grid shape={grid.shape}, origin={origin}, spacing={spacing}")

    # 3) PyVista 交互可视化
    visualize_with_pyvista(grid, origin, spacing)
