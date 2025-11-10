#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 sdf_tools_gpu.py 生成的 .npy（SDF 体素）与 .json（meta）文件，
提取零等值面并将其法向与输入 OBJ 网格法向进行逐点对齐比较，
输出交互式 PyVista 热力图（标注角度误差，单位：°）。

要点：
- 使用 pyvista.ImageData（vtkImageData）。
- SDF 以 cell_data 写入（dims = sdf.shape + 1），先 cell→point，再 contour 取 0 等值面。
- “最近表面点 + 重心插值点法向”（或可选面法向）做对应，减小“最近顶点”导致的误差放大。
- 兼容旧版 PyVista：triangulate 与 compute_normals 的参数差异用包装函数消除。

直接修改脚本顶部常量后运行：
    python demo_sdf_normal_error_heatmap.py
"""

from __future__ import annotations

import json
from typing import Tuple, Optional

import numpy as np

# ---------- PyVista / VTK ----------
try:
    import pyvista as pv
    from pyvista import _vtk as vtk
except Exception as e:
    raise RuntimeError("需要安装 pyvista：pip install pyvista") from e

# ---------- 可选 SciPy（非必须） ----------
try:
    from scipy.spatial import cKDTree   # 这里没有用到最近“顶点”，仅保留以备他处
    _HAS_KDTREE = True
except Exception:
    _HAS_KDTREE = False


# ==================== 参数定义 ====================
SDF_NPY_PATH = "./traditional_outputs/gear_sdf.npy"        # SDF 体素文件路径
META_JSON_PATH = "./traditional_outputs/gear_meta.json"    # 元数据文件路径
OBJ_PATH = "./obj_library/gear.obj"            # OBJ 网格文件路径

ABS_ANGLE = True            # True: 方向无关角度（推荐）；False: 方向敏感
USE_FACE_NORMALS = False    # True: 使用三角面法向；False: 重心插值点法向（推荐）
CLIM: Optional[Tuple[float, float]] = None   # 例如 (0, 45)；None 自动
SCREENSHOT_PATH: Optional[str] = None        # 例如 "heatmap.png" 或 None
# =================================================


# ----------------- 兼容封装 -----------------
def triangulate_compat(poly: pv.PolyData) -> pv.PolyData:
    """兼容不同版本的 PyVista triangulate。"""
    try:
        return poly.triangulate(inplace=False)  # 新/常见签名
    except TypeError:
        return poly.triangulate()               # 旧签名无参数


def compute_normals_compat(poly: pv.PolyData,
                           point_normals: bool = True,
                           cell_normals: bool = False,
                           feature_angle: float = 60.0,
                           want_consistent: bool = True,
                           want_auto_orient: bool = True,
                           want_split_vertices: bool = True) -> pv.PolyData:
    """
    兼容不同版本的 compute_normals 参数：
    - 有的版本用 split_vertices=True；
    - 有的版本用 splitting=True；
    - 部分版本没有 consistent_normals / auto_orient_normals。
    逐个尝试可用的参数组合，直到成功为止。
    """
    candidates = []

    # 最完整（优先）
    candidates.append(dict(point_normals=point_normals, cell_normals=cell_normals,
                           feature_angle=feature_angle,
                           consistent_normals=want_consistent,
                           auto_orient_normals=want_auto_orient,
                           split_vertices=want_split_vertices))
    candidates.append(dict(point_normals=point_normals, cell_normals=cell_normals,
                           feature_angle=feature_angle,
                           consistent_normals=want_consistent,
                           auto_orient_normals=want_auto_orient,
                           splitting=want_split_vertices))
    # 去掉 consistent
    candidates.append(dict(point_normals=point_normals, cell_normals=cell_normals,
                           feature_angle=feature_angle,
                           auto_orient_normals=want_auto_orient,
                           split_vertices=want_split_vertices))
    candidates.append(dict(point_normals=point_normals, cell_normals=cell_normals,
                           feature_angle=feature_angle,
                           auto_orient_normals=want_auto_orient,
                           splitting=want_split_vertices))
    # 只保留角度与 split
    candidates.append(dict(point_normals=point_normals, cell_normals=cell_normals,
                           feature_angle=feature_angle,
                           split_vertices=want_split_vertices))
    candidates.append(dict(point_normals=point_normals, cell_normals=cell_normals,
                           feature_angle=feature_angle,
                           splitting=want_split_vertices))
    # 最简
    candidates.append(dict(point_normals=point_normals, cell_normals=cell_normals,
                           feature_angle=feature_angle))
    candidates.append(dict(point_normals=point_normals, cell_normals=cell_normals))

    last_err = None
    for kw in candidates:
        try:
            return poly.compute_normals(**kw)
        except TypeError as e:
            last_err = e
            continue
    # 如果全都不行，抛出最后一个错误
    raise last_err if last_err else RuntimeError("compute_normals 调用失败（未知原因）")


# ----------------- I/O -----------------
def load_meta(meta_json_path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    with open(meta_json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    bmin = np.array(meta.get('bounds_min'), dtype=float)
    bmax = np.array(meta.get('bounds_max'), dtype=float)
    voxel_step = tuple(float(x) for x in meta.get('voxel_step'))
    return bmin, bmax, voxel_step


def build_image_data(sdf_grid: np.ndarray,
                     bmin: np.ndarray,
                     voxel_step: Tuple[float, float, float]) -> pv.ImageData:
    """
    将 SDF 作为 **cell_data** 写入 ImageData，维度 = shape + 1（cell-centered 布局）。
    与原实现一致，采用 Fortran 顺序展平（order='F'）。
    """
    nx, ny, nz = sdf_grid.shape
    grid = pv.ImageData()
    grid.dimensions = np.array([nx, ny, nz]) + 1
    grid.origin = bmin.astype(float)
    grid.spacing = voxel_step
    grid.point_data.clear()
    grid.cell_data.clear()
    grid.cell_data["sdf"] = sdf_grid.ravel(order="F")
    return grid


def extract_zero_isosurface(grid_with_point_sdf: pv.ImageData) -> pv.PolyData:
    """从包含**点数据** SDF 的 ImageData 中提取零等值面，并计算几何点法向。"""
    surf = grid_with_point_sdf.contour([0.0], scalars="sdf")
    surf = triangulate_compat(surf)
    surf = compute_normals_compat(surf, point_normals=True, cell_normals=False)
    return surf


def load_obj_with_point_normals(obj_path: str) -> pv.PolyData:
    mesh = pv.read(obj_path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = triangulate_compat(mesh)
    # 为一致性重算点法向
    mesh = compute_normals_compat(mesh, point_normals=True, cell_normals=False)
    return mesh


# ----------------- 表面采样：最近表面点 + 重心插值法向 -----------------
def sample_obj_normals_on_surface(obj: pv.PolyData,
                                  query_pts: np.ndarray,
                                  use_face_normals: bool = False) -> np.ndarray:
    """
    对每个查询点（等值面顶点）：
      1) 找到 OBJ 上最近的三角面上的最近点；
      2) 若 use_face_normals=False：对该三角的三个顶点点法向做重心插值；
         若 use_face_normals=True：直接使用该三角的面法向。
    """
    m = obj
    if not isinstance(m, pv.PolyData):
        m = m.extract_surface()
    m = triangulate_compat(m)

    # 准备法向数据
    if use_face_normals:
        m = compute_normals_compat(m, point_normals=False, cell_normals=True)
        cell_normals = np.asarray(m.cell_data["Normals"])
    else:
        if "Normals" not in m.point_data:
            m = compute_normals_compat(m, point_normals=True, cell_normals=False)
        point_normals = np.asarray(m.point_data["Normals"])

    pts_arr = m.points
    faces_tri = m.faces.reshape(-1, 4)[:, 1:4]  # (n_cells, 3)

    # Cell 级定位器：最近表面点
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(m)
    locator.BuildLocator()

    # vtk 的可变输出参数适配
    try:
        mutable = vtk.mutable
    except AttributeError:
        mutable = vtk.reference

    out = np.empty((query_pts.shape[0], 3), dtype=float)

    def _barycentric(p, a, b, c):
        v0, v1, v2 = b - a, c - a, p - a
        d00 = np.dot(v0, v0); d01 = np.dot(v0, v1); d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0); d21 = np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        if denom <= 1e-20:
            return np.array([1.0, 0.0, 0.0])
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return np.array([u, v, w])

    for i, p in enumerate(query_pts):
        closest = [0.0, 0.0, 0.0]
        cell_id = mutable(0)
        sub_id = mutable(0)
        dist2 = mutable(0.0)
        locator.FindClosestPoint(p, closest, cell_id, sub_id, dist2)
        cid = int(cell_id)

        tri = faces_tri[cid]
        a, b, c = pts_arr[tri[0]], pts_arr[tri[1]], pts_arr[tri[2]]
        if use_face_normals:
            n = cell_normals[cid]
        else:
            w = _barycentric(np.asarray(closest), a, b, c)
            n = (w[0] * point_normals[tri[0]] +
                 w[1] * point_normals[tri[1]] +
                 w[2] * point_normals[tri[2]])

        n_norm = np.linalg.norm(n)
        if n_norm > 1e-12:
            n = n / n_norm
        out[i] = n

    return out


# ----------------- 误差计算 -----------------
def unitize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.clip(n, eps, None)
    return v / n


def normal_error_degrees(n_iso: np.ndarray, n_obj: np.ndarray, abs_angle: bool = True) -> np.ndarray:
    n_iso_u = unitize(n_iso)
    n_obj_u = unitize(n_obj)
    dots = np.einsum('ij,ij->i', n_iso_u, n_obj_u)
    if abs_angle:
        dots = np.abs(dots)
    dots = np.clip(dots, -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    return ang


# ----------------- 可视化 -----------------
def plot_heatmap(surf: pv.PolyData,
                 obj: Optional[pv.PolyData] = None,
                 clim: Optional[Tuple[float, float]] = None,
                 screenshot: Optional[str] = None):
    p = pv.Plotter(window_size=[1000, 800])
    p.set_background('white')

    scal_name = "normal_error_deg"
    if clim is None:
        upper = float(np.nanpercentile(np.asarray(surf[scal_name]), 95))
        clim = (0.0, max(5.0, upper))

    p.add_mesh(surf, scalars=scal_name, cmap='turbo', clim=clim,
               show_edges=False, smooth_shading=True, scalar_bar_args={
                   'title': 'Normal Error (°)',
                   'title_font_size': 12,
                   'label_font_size': 10
               })

    if obj is not None:
        p.add_mesh(obj, color='black', style='wireframe', opacity=0.15)

    p.add_axes()
    p.show_bounds(grid='front', location='outer', all_edges=True)
    p.view_isometric()

    if screenshot:
        p.screenshot(screenshot)
    p.show()


# ----------------- 主流程 -----------------
def main():
    # 1) 载入 SDF 与 Meta
    sdf_grid = np.load(SDF_NPY_PATH)
    bmin, bmax, voxel_step = load_meta(META_JSON_PATH)

    # 2) ImageData & 等值面
    grid = build_image_data(sdf_grid, bmin=bmin, voxel_step=voxel_step)
    grid_point_data = grid.cell_data_to_point_data()  # contour 需要点数据
    surf = extract_zero_isosurface(grid_point_data)

    # 3) 载入 OBJ 并获取点法向（供重心插值用）
    obj = load_obj_with_point_normals(OBJ_PATH)

    # 4) 将 OBJ 法向通过“最近表面点 + 重心插值/面法向”采样到等值面顶点
    obj_n_at_iso = sample_obj_normals_on_surface(
        obj, surf.points, use_face_normals=USE_FACE_NORMALS
    )

    # 5) 计算角度误差（°）并写入等值面属性
    n_iso = np.asarray(surf.point_normals)  # 几何法向；也可替换为 SDF 梯度方向
    ang_deg = normal_error_degrees(n_iso, obj_n_at_iso, abs_angle=ABS_ANGLE)
    surf["normal_error_deg"] = ang_deg

    # 6) 可视化
    plot_heatmap(surf, obj=obj, clim=CLIM, screenshot=SCREENSHOT_PATH)


if __name__ == '__main__':
    main()
