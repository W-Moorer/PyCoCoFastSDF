# -*- coding: utf-8 -*-
"""
gradient_sdf_tools_gpu.py — 梯度 SDF 专用实现（始终输出 grad_grid）
---------------------------------------------------------------
- 保留原有函数名/返回签名：parse_obj / compute_sdf_grid / save_sdf_and_meta / pyvista_visualize_isosurface / _main
- SDF 约定：内部为负、外部为正；grad_grid 为单位外法向（∇SDF）
- 优先 igl.signed_distance；否则 AABB 最近点 + FWN 判号
- 始终保存 *_grad.npy，并在 *_meta.json 写 has_grad = true
"""

from __future__ import annotations
import os
import json
import time
import argparse
from typing import Optional, Tuple, Dict

import numpy as np

# -------------------- 依赖探测 --------------------
try:
    import igl  # libigl
    _HAS_IGL = True
    _HAS_SD = hasattr(igl, "signed_distance")
    _HAS_SD_FAST = hasattr(igl, "SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER")
    _HAS_SD_WIND = hasattr(igl, "SIGNED_DISTANCE_TYPE_WINDING_NUMBER")
    _HAS_SD_PSEUDO = hasattr(igl, "SIGNED_DISTANCE_TYPE_PSEUDONORMAL")
    _HAS_FWN = hasattr(igl, "fast_winding_number_for_meshes")
except Exception:
    _HAS_IGL = False
    _HAS_SD = False
    _HAS_SD_FAST = _HAS_SD_WIND = _HAS_SD_PSEUDO = False
    _HAS_FWN = False

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

try:
    from skimage import measure as _measure
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

_AABB_BATCH = 200_000
_EPS = 1e-12

# -------------------- 模块级状态：保存时使用 --------------------
_GRAD_GRID: Optional[np.ndarray] = None  # (nx,ny,nz,3)

# -------------------- 工具函数 --------------------
def _ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def parse_obj(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """极简 OBJ 读取（v/f 三角）"""
    vs, fs = [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            sp = line.strip().split()
            if not sp:
                continue
            if sp[0] == 'v' and len(sp) >= 4:
                vs.append([float(sp[1]), float(sp[2]), float(sp[3])])
            elif sp[0] == 'f' and len(sp) >= 4:
                tri = []
                for t in sp[1:4]:
                    tri.append(int(t.split('/')[0]) - 1)
                fs.append(tri)
    V = np.asarray(vs, dtype=np.float64)
    F = np.asarray(fs, dtype=np.int32)
    return V, F

def _compute_bounds(V: np.ndarray, padding: float) -> Tuple[np.ndarray, np.ndarray]:
    vmin = V.min(axis=0)
    vmax = V.max(axis=0)
    diag = float(np.linalg.norm(vmax - vmin))
    pad = padding * diag
    return (vmin - pad).astype(np.float64), (vmax + pad).astype(np.float64)

def _voxel_axes(bmin: np.ndarray, bmax: np.ndarray,
                voxel_size: Optional[float],
                target_resolution: Optional[int],
                max_resolution: int):
    size = (bmax - bmin).astype(np.float64)
    if voxel_size is None:
        if target_resolution is None:
            target_resolution = min(192, max_resolution)
        longest = float(size.max())
        voxel = longest / float(target_resolution)
    else:
        voxel = float(voxel_size)

    nx, ny, nz = np.maximum(1, np.ceil(size / voxel).astype(int))
    xs = np.linspace(bmin[0] + 0.5*voxel, bmin[0] + (nx-0.5)*voxel, nx, dtype=np.float64)
    ys = np.linspace(bmin[1] + 0.5*voxel, bmin[1] + (ny-0.5)*voxel, ny, dtype=np.float64)
    zs = np.linspace(bmin[2] + 0.5*voxel, bmin[2] + (nz-0.5)*voxel, nz, dtype=np.float64)
    return xs, ys, zs, np.array([voxel, voxel, voxel], dtype=np.float64)

def _grid_points(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)

# -------------------- AABB 最近点 + 三角索引 --------------------
def _aabb_dist_closest_idx(pts: np.ndarray, V: np.ndarray, F: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用 libigl.AABB 批量计算：返回 无符号距离 d, 最近点 C, 最近三角索引 I"""
    if not _HAS_IGL:
        raise RuntimeError("需要 libigl（AABB）")
    Vd = np.ascontiguousarray(V, dtype=np.float64)
    Fi = np.ascontiguousarray(F, dtype=np.int32)
    P  = np.ascontiguousarray(pts, dtype=np.float64)
    tree = igl.AABB()
    tree.init(Vd, Fi)

    M = P.shape[0]
    d = np.empty((M,), dtype=np.float64)
    C = np.empty((M, 3), dtype=np.float64)
    I = np.empty((M,), dtype=np.int32)

    for st in range(0, M, _AABB_BATCH):
        ed = min(M, st + _AABB_BATCH)
        sqrD, Ii, Ci = tree.squared_distance(Vd, Fi, P[st:ed])
        d[st:ed] = np.sqrt(np.maximum(sqrD, 0.0))
        C[st:ed] = Ci
        I[st:ed] = Ii
    return d, C, I

def _face_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    fN = np.cross(V[F][:,1,:] - V[F][:,0,:],
                  V[F][:,2,:] - V[F][:,0,:])
    nlen = np.linalg.norm(fN, axis=1, keepdims=True)
    nlen[nlen == 0.0] = 1.0
    return (fN / nlen).astype(np.float64)

# -------------------- 主函数：计算 SDF（内部负）与梯度（单位外法向） --------------------
def compute_sdf_grid(vertices: np.ndarray,
                     faces: np.ndarray,
                     padding: float = 0.1,
                     voxel_size: Optional[float] = None,
                     target_resolution: Optional[int] = None,
                     max_resolution: int = 512,
                     show_progress: bool = True,
                     sdf_backend: str = 'auto',     # 保留参数名；'auto' -> 优先 signed_distance
                     workers: int = -1,             # 保留参数名，未用
                     robust_sign: bool = True,      # 保留参数名，未用
                     ambiguous_deg: float = 22.5    # 保留参数名，未用
                     ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, Dict[str, float], Tuple[float, float, float]]:
    """
    返回（保持原签名）：
      sdf_grid: (nx,ny,nz)  —— 内部负、外部正
      bounds: (bmin,bmax)
      centers: (F,3)  —— 三角中心（占位/兼容）
      normals: (F,3)  —— 三角法向（占位/兼容）
      timings: dict
      voxel_step: (vx,vy,vz)
    另外：模块级 _GRAD_GRID 会存储 (nx,ny,nz,3) 的单位外法向供保存逻辑使用。
    """
    if not _HAS_IGL:
        raise RuntimeError("需要 libigl：pip install igl")

    global _GRAD_GRID
    _GRAD_GRID = None

    t0 = time.time()
    bmin, bmax = _compute_bounds(vertices, padding)
    xs, ys, zs, vstep = _voxel_axes(bmin, bmax, voxel_size, target_resolution, max_resolution)
    pts = _grid_points(xs, ys, zs)
    centers = vertices[faces].mean(axis=1).astype(np.float64)
    fN = _face_normals(vertices, faces)

    timings: Dict[str, float] = {}
    t_grid = time.time()
    timings["grid_setup"] = t_grid - t0

    # --- 优先：signed_distance（含最近点与面法向） ---
    if _HAS_SD and (_HAS_SD_FAST or _HAS_SD_WIND or _HAS_SD_PSEUDO):
        Vd = np.ascontiguousarray(vertices, dtype=np.float64)
        Fi = np.ascontiguousarray(faces, dtype=np.int32)
        Q  = np.ascontiguousarray(pts, dtype=np.float64)
        if _HAS_SD_FAST:
            stype = igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
        elif _HAS_SD_WIND:
            stype = igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER
        else:
            stype = igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL

        S, I, C, N = igl.signed_distance(Q, Vd, Fi, stype)  # S: 外正内负
        S = np.asarray(S, dtype=np.float64).reshape(-1)
        C = np.asarray(C, dtype=np.float64)
        N = np.asarray(N, dtype=np.float64)

        t_q = time.time()
        timings["signed_distance"] = t_q - t_grid

        # 梯度：sgn(S) * (p - C) / ||p-C||
        pc = pts - C
        norm = np.linalg.norm(pc, axis=1, keepdims=True)
        sgn = np.sign(S).reshape(-1, 1)  # 内负 / 外正
        sgn[sgn == 0.0] = 1.0
        g = sgn * pc / np.maximum(norm, _EPS)
        near = (norm.reshape(-1) < 1e-9)
        if np.any(near):
            g[near] = sgn[near] * N[near]

        sdf_grid = S.reshape(len(xs), len(ys), len(zs))
        _GRAD_GRID = g.reshape(len(xs), len(ys), len(zs), 3)

        t_pack = time.time()
        timings["pack"] = t_pack - t_q
        timings["total"] = t_pack - t0

    else:
        # --- 回退：AABB 最近点 + FWN 判号 ---
        d_unsigned, C, tri_idx = _aabb_dist_closest_idx(pts, vertices, faces)
        if not _HAS_FWN:
            raise RuntimeError("缺少 fast_winding_number_for_meshes（FWN），无法判号；请安装带 FWN 的 libigl。")

        Vd = np.ascontiguousarray(vertices, dtype=np.float64)
        Fi = np.ascontiguousarray(faces, dtype=np.int32)
        Q  = np.ascontiguousarray(pts, dtype=np.float64)
        W = igl.fast_winding_number_for_meshes(Vd, Fi, Q)[0]  # (M,)
        W = np.asarray(W, dtype=np.float64).reshape(-1)

        t_q = time.time()
        timings["aabb+fwn"] = t_q - t_grid

        # SDF：内部负、外部正
        sgnS = np.where(W > 0.5, -1.0, +1.0)   # inside -> -1
        S = d_unsigned * sgnS

        # 梯度：sgn(S) * (p - C) / ||p-C||，近零用面法向回退
        pc = pts - C
        norm = np.linalg.norm(pc, axis=1, keepdims=True)
        g = sgnS.reshape(-1, 1) * pc / np.maximum(norm, _EPS)
        near = (norm.reshape(-1) < 1e-9)
        if np.any(near):
            tri_n = fN[tri_idx[near]]
            g[near] = sgnS.reshape(-1, 1)[near] * tri_n

        sdf_grid = S.reshape(len(xs), len(ys), len(zs))
        _GRAD_GRID = g.reshape(len(xs), len(ys), len(zs), 3)

        t_pack = time.time()
        timings["pack"] = t_pack - t_q
        timings["total"] = t_pack - t0

    return sdf_grid, (bmin, bmax), centers, fN, timings, (float(vstep[0]), float(vstep[1]), float(vstep[2]))

# -------------------- 可视化与保存 --------------------
def _marching_cubes_world(sdf_grid: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray]):
    if not (_HAS_SKIMAGE):
        return None
    bmin, bmax = bounds
    nx, ny, nz = sdf_grid.shape
    xs = np.linspace(bmin[0], bmax[0], int(nx), dtype=np.float64)
    ys = np.linspace(bmin[1], bmax[1], int(ny), dtype=np.float64)
    zs = np.linspace(bmin[2], bmax[2], int(nz), dtype=np.float64)
    spacing = (xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0])
    verts, faces, normals, values = _measure.marching_cubes(sdf_grid, level=0.0, spacing=spacing)
    verts_world = verts + np.array(bmin, dtype=np.float64)[None, :]
    return verts_world, faces

def visualize_zero_isosurface(sdf_grid: np.ndarray,
                              bounds: Tuple[np.ndarray, np.ndarray],
                              out_path: str):
    if not (_HAS_MPL and _HAS_SKIMAGE):
        return
    res = _marching_cubes_world(sdf_grid, bounds)
    if res is None:
        return
    verts_world, faces = res
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts_world[:, 0], verts_world[:, 1], Z=verts_world[:, 2],
                    triangles=faces, linewidth=0.0, antialiased=True, alpha=0.85)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Zero Isosurface")
    _ensure_dir_for(out_path)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

def save_timings_pie(timings: Dict[str, float], out_path: str):
    if not _HAS_MPL:
        return
    labels = [k for k in timings.keys() if k != "total"]
    sizes = [timings[k] for k in labels]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.axis('equal')
    _ensure_dir_for(out_path)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

def save_sdf_and_meta(sdf_grid: np.ndarray,
                      bounds: Tuple[np.ndarray, np.ndarray],
                      obj_path: str,
                      voxel_step: Tuple[float, float, float],
                      padding: float,
                      timings: Dict[str, float],
                      out_prefix: str) -> Tuple[str, str, str, str]:
    """
    保持原签名；本梯度版**总是**保存 *_grad.npy，并在 meta.json 写 has_grad=true。
    返回：(isosurface_png, timings_pie_png, sdf_npy, grad_npy, meta_json)
    """
    global _GRAD_GRID
    if _GRAD_GRID is None:
        raise RuntimeError("内部错误：梯度网格尚未生成。")

    npy_path  = f"{out_prefix}_sdf.npy"
    grad_path = f"{out_prefix}_grad.npy"
    meta_path = f"{out_prefix}_meta.json"
    png_path  = f"{out_prefix}_isosurface.png"
    pie_path  = f"{out_prefix}_timings_pie.png"

    _ensure_dir_for(npy_path)
    np.save(npy_path,  sdf_grid)
    np.save(grad_path, _GRAD_GRID)

    bmin, bmax = bounds
    meta = dict(
        obj=os.path.abspath(obj_path),
        bmin=np.asarray(bmin, dtype=np.float64).tolist(),
        bmax=np.asarray(bmax, dtype=np.float64).tolist(),
        shape=list(map(int, sdf_grid.shape)),
        grad_shape=list(map(int, _GRAD_GRID.shape)),
        voxel_step=list(map(float, voxel_step)),
        padding=float(padding),
        timings={k: float(v) for k, v in timings.items()},
        has_grad=True
    )
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    try:
        visualize_zero_isosurface(sdf_grid, bounds, out_path=png_path)
    except Exception as e:
        png_path = f"{png_path}.skipped"
        with open(png_path, 'w', encoding='utf-8') as f:
            f.write(f"isosurface skipped: {e}")
    try:
        save_timings_pie(timings, pie_path)
    except Exception as e:
        pie_path = f"{pie_path}.skipped"
        with open(pie_path, 'w', encoding='utf-8') as f:
            f.write(f"pie skipped: {e}")

    return png_path, pie_path, npy_path, meta_path

# -------------------- 可选：PyVista 交互可视化（保留函数名） --------------------
def pyvista_visualize_isosurface(sdf_grid,
                                 bounds,
                                 show=True,
                                 out_png: Optional[str] = None):
    """需要 pyvista；若未安装则抛错。"""
    try:
        import pyvista as pv
    except Exception as e:
        raise RuntimeError("需要 pyvista 才能使用该函数") from e

    res = _marching_cubes_world(sdf_grid, bounds)
    if res is None:
        raise RuntimeError("需要 scikit-image 才能进行 marching_cubes 网格化")
    verts_world, faces = res
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]).ravel()
    mesh = pv.PolyData(verts_world, faces_pv)

    plotter = pv.Plotter(window_size=[900, 700])
    plotter.add_mesh(mesh, color="lightsteelblue", opacity=0.8, smooth_shading=True)
    plotter.show_axes()
    if out_png:
        plotter.screenshot(out_png)
    if show:
        plotter.show()

# -------------------- CLI --------------------
def _main():
    ap = argparse.ArgumentParser("Gradient SDF Grid Generator (always outputs grad_grid)")
    ap.add_argument("--obj", required=True, help="输入 .obj 网格路径")
    ap.add_argument("--out", required=True, help="输出前缀，例如 out/scene")
    ap.add_argument("--padding", type=float, default=0.1)
    ap.add_argument("--voxel_size", type=float, default=None, help="体素尺寸（优先于 target_resolution）")
    ap.add_argument("--target_resolution", type=int, default=192)
    ap.add_argument("--max_resolution", type=int, default=512)
    args = ap.parse_args()

    V, F = parse_obj(args.obj)
    sdf_grid, bounds, centers, normals, timings, voxel_step = compute_sdf_grid(
        V, F,
        padding=args.padding,
        voxel_size=args.voxel_size,
        target_resolution=args.target_resolution,
        max_resolution=args.max_resolution,
        show_progress=True,
        sdf_backend='auto'
    )

    png, pie, npy, meta = save_sdf_and_meta(
        sdf_grid, bounds=bounds, obj_path=args.obj,
        voxel_step=voxel_step, padding=args.padding,
        timings=timings, out_prefix=args.out
    )

    print(json.dumps({
        "isosurface_png": png,
        "timings_pie": pie,
        "sdf_npy": npy,
        "grad_npy": f"{args.out}_grad.npy",
        "meta_json": meta
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    _main()
