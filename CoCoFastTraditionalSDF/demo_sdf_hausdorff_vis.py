#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 sdf_tools_gpu.py 生成的 *_sdf.npy 与 *_meta.json 中提取零等值面，
计算输入 OBJ 表面到零等值面的单向 Hausdorff 距离，并用 PyVista 交互式可视化：
- 颜色：OBJ 点到零等值面的最短距离（单位：与数据一致）
- 文本：报告 HD (最大距离), HD95 (95分位距离), ASD (平均距离) 等指标

用法：
    python sdf_hausdorff_vis.py

说明：
- 本脚本按 *_meta.json* 的 `bounds_min/bounds_max/voxel_step` 构造 UniformGrid，
  将 *_sdf.npy 作为**cell_data** 装载并提取 0 等值面。
- 只计算 OBJ 到 Iso 的单向距离。
- 交互界面在右上角显示距离热力图和统计指标。
"""
from __future__ import annotations

import json
from typing import Tuple, Optional

import numpy as np

# -------------------- PyVista / VTK --------------------
try:
    import pyvista as pv
    from pyvista import _vtk as vtk
except Exception as e:
    raise RuntimeError("需要安装 pyvista：pip install pyvista") from e

# ==================== 参数定义 ====================
# 在这里修改参数，无需通过命令行输入
SDF_NPY_PATH = "./traditional_outputs/gear_sdf.npy"  # SDF 体素文件路径
META_JSON_PATH = "./traditional_outputs/gear_meta.json"  # 元数据文件路径
OBJ_PATH = "./obj_library/gear.obj"  # OBJ 网格文件路径
CLIM = None  # 色标范围，例如 (0, 5) 或 None 表示自动
PCTL = 95.0  # HDp 的分位数（默认 95）
SCREENSHOT_PATH = None  # 截图保存路径，例如 "hd.png" 或 None


# =================================================


# -------------------- I/O --------------------

def load_meta(meta_json_path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    with open(meta_json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    bmin = np.array(meta.get('bounds_min'), dtype=float)
    bmax = np.array(meta.get('bounds_max'), dtype=float)
    voxel_step = tuple(float(x) for x in meta.get('voxel_step'))
    return bmin, bmax, voxel_step


def build_image_data_from_cells(sdf_grid: np.ndarray,
                                bmin: np.ndarray,
                                voxel_step: Tuple[float, float, float]) -> pv.ImageData:
    """使用 PyVista 的 ImageData（vtkImageData）装载 SDF，并作为 **cell_data** 提取等值面。"""
    nx, ny, nz = sdf_grid.shape
    spacing = tuple(float(x) for x in voxel_step)

    grid = pv.ImageData()
    grid.dimensions = (nx + 1, ny + 1, nz + 1)
    grid.origin = bmin.astype(float)
    grid.spacing = spacing

    grid.point_data.clear();
    grid.cell_data.clear()
    grid.cell_data["sdf"] = np.ascontiguousarray(sdf_grid).ravel(order='F')
    return grid


def extract_zero_isosurface(grid: pv.ImageData) -> pv.PolyData:
    grid_point_data = grid.cell_data_to_point_data()
    surf = grid_point_data.contour([0.0], scalars='sdf')
    surf = surf.triangulate().compute_normals(point_normals=True, cell_normals=False,
                                              auto_orient_normals=True, consistent_normals=True)
    return surf


def load_obj_surface(obj_path: str) -> pv.PolyData:
    m = pv.read(obj_path)
    if not isinstance(m, pv.PolyData):
        m = m.extract_surface()
    m = m.triangulate()
    return m


# -------------------- 距离计算 --------------------

def _implicit_distance_numpy(points: np.ndarray, target: pv.PolyData) -> np.ndarray:
    """使用 VTK 的 vtkImplicitPolyDataDistance 计算点到目标曲面的绝对距离。"""
    ipd = vtk.vtkImplicitPolyDataDistance()
    ipd.SetInput(target)
    out = np.empty((points.shape[0],), dtype=float)
    for i, p in enumerate(points):
        out[i] = ipd.EvaluateFunction(p)
    return np.abs(out)


def distances_A_to_B_surface(A: pv.PolyData, B: pv.PolyData) -> np.ndarray:
    """对 A 的每个点，求到 B 表面的最小距离。"""
    return _implicit_distance_numpy(A.points, B)


# -------------------- 统计指标 --------------------

def summarize_unidirectional_distance(d: np.ndarray, pctl: float = 95.0) -> dict:
    """计算单向距离的统计指标。"""
    hd = float(np.max(d)) if d.size else 0.0
    hd_p = float(np.percentile(d, pctl)) if d.size else 0.0
    asd = float(np.mean(d)) if d.size else 0.0
    return dict(
        HD=hd,
        HDp=hd_p,
        ASD=asd,
    )


# -------------------- 可视化 --------------------

def plot_interactive(iso: pv.PolyData, obj: pv.PolyData,
                     d_obj_to_iso: np.ndarray, metrics: dict,
                     clim: Optional[Tuple[float, float]] = None,
                     screenshot: Optional[str] = None):
    p = pv.Plotter(window_size=[1200, 900])
    p.set_background('white')

    def set_times_font(actor_or_text_prop):
        """为 VTK actor 或 TextProperty 设置 Times 字体"""
        if hasattr(actor_or_text_prop, 'GetTextProperty'):
            text_prop = actor_or_text_prop.GetTextProperty()
            text_prop.SetFontFamilyToTimes()
        elif hasattr(actor_or_text_prop, 'SetFontFamilyToTimes'):
            actor_or_text_prop.SetFontFamilyToTimes()
        else:
            print(f"Warning: Cannot set Times font for object of type {type(actor_or_text_prop)}")

    # 自动色标现在完全跟随数据的最小值和最大值
    def _auto_clim(arr: np.ndarray):
        if clim is not None:
            return tuple(clim)
        # 使用数据的最小值和最大值作为颜色范围
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        # 为了避免颜色条范围过小（例如所有点距离都一样），可以加一个小的容差
        if np.isclose(min_val, max_val):
            max_val = min_val + 1e-6
        return (min_val, max_val)

    # 添加 OBJ 网格的距离热力图，明确禁用默认 scalar bar
    obj_cmap = 'viridis'
    obj["d_to_iso"] = d_obj_to_iso
    clim_obj = _auto_clim(d_obj_to_iso)

    # 修改点：捕获 add_mesh 返回的 actor，以便获取其 lookup table
    actor_obj = p.add_mesh(obj, scalars="d_to_iso", cmap=obj_cmap, clim=clim_obj,
                           smooth_shading=True, show_edges=False, show_scalar_bar=False)

    # 添加 Iso 网格作为半透明线框参考
    p.add_mesh(iso, color='black', style='wireframe', opacity=0.2, line_width=1)

    # 手动创建 scalar bar，并设置 Times 字体
    scalar_bar = p.add_scalar_bar(title='OBJ → Iso Distance',
                                  position_x=0.15, position_y=0.02,  # 修改位置到底部
                                  width=0.7, height=0.05,  # 修改为水平扁条形
                                  n_labels=5, label_font_size=10,
                                  title_font_size=12)

    # 修改点：从网格的 mapper 获取 lookup table 并设置给 scalar bar，确保颜色同步
    lookup_table = actor_obj.mapper.lookup_table
    scalar_bar.SetLookupTable(lookup_table)

    scalar_bar.SetOrientation(0)  # 0 for horizontal, 1 for vertical

    set_times_font(scalar_bar.GetTitleTextProperty())
    set_times_font(scalar_bar.GetLabelTextProperty())

    # 在右上角显示指标文本
    txt = (f"HD = {metrics['HD']:.6g}\n"
           f"HD{PCTL:.0f} = {metrics['HDp']:.6g}\n"
           f"ASD = {metrics['ASD']:.6g}")
    metrics_actor = p.add_text(txt, position='upper_right', font_size=12, color='black', name='metrics')
    set_times_font(metrics_actor)

    p.add_axes()
    p.show_bounds(grid='front', location='outer', all_edges=True)
    p.view_isometric()

    if screenshot:
        p.screenshot(screenshot)
    p.show()


# -------------------- 主流程 --------------------

def main():
    print("=" * 80)
    print("OBJ 到 SDF 零等值面单向距离评估工具")
    print("=" * 80)

    # 打印参数信息
    print("\n【参数配置】")
    print(f"  SDF_NPY_PATH: {SDF_NPY_PATH}")
    print(f"    - SDF 体素文件路径，包含符号距离场数据")
    print(f"  META_JSON_PATH: {META_JSON_PATH}")
    print(f"    - 元数据文件路径，包含边界和体素步长信息")
    print(f"  OBJ_PATH: {OBJ_PATH}")
    print(f"    - 参考 OBJ 网格文件路径")
    print(f"  CLIM: {CLIM}")
    print(f"    - 色标范围，None 表示自动计算")
    print(f"  PCTL: {PCTL}")
    print(f"    - HDp 的分位数，用于计算 HD{PCTL:.0f}")
    print(f"  SCREENSHOT_PATH: {SCREENSHOT_PATH}")
    print(f"    - 截图保存路径，None 表示不保存")

    # 1) 载入 SDF 与 Meta
    print("\n【数据加载】")
    print(f"  正在加载 SDF 文件: {SDF_NPY_PATH}")
    sdf_grid = np.load(SDF_NPY_PATH)
    print(f"  SDF 网格尺寸: {sdf_grid.shape}")

    print(f"  正在加载元数据文件: {META_JSON_PATH}")
    bmin, bmax, voxel_step = load_meta(META_JSON_PATH)
    print(f"  边界范围: [{bmin}, {bmax}]")
    print(f"  体素步长: {voxel_step}")

    # 2) ImageData & 零等值面
    print("\n【等值面提取】")
    print("  正在构建 ImageData 网格...")
    grid = build_image_data_from_cells(sdf_grid, bmin=bmin, voxel_step=voxel_step)
    print("  正在提取零等值面...")
    iso = extract_zero_isosurface(grid)
    print(f"  等值面顶点数: {iso.n_points}")
    print(f"  等值面三角面数: {iso.n_cells}")

    # 3) 载入 OBJ 表面
    print("\n【OBJ 加载】")
    print(f"  正在加载 OBJ 文件: {OBJ_PATH}")
    obj = load_obj_surface(OBJ_PATH)
    print(f"  OBJ 顶点数: {obj.n_points}")
    print(f"  OBJ 三角面数: {obj.n_cells}")

    # 4) 计算 OBJ 到 Iso 的距离
    print("\n【距离计算】")
    print("  正在计算 OBJ → Iso 距离...")
    d_obj_to_iso = distances_A_to_B_surface(obj, iso)

    # 5) 统计单向距离指标
    print("\n【统计指标】")
    metrics = summarize_unidirectional_distance(d_obj_to_iso, pctl=PCTL)

    print("\n【评估结果】")
    print(f"  HD (Hausdorff 距离): {metrics['HD']:.6g}")
    print(f"    - 定义: max_x d(x, Iso)")
    print(f"    - 说明: OBJ 曲面到 Iso 曲面的最大距离")
    print(f"  HD{PCTL:.0f} ({PCTL:.0f}% Hausdorff 距离): {metrics['HDp']:.6g}")
    print(f"    - 定义: pctl(d(OBJ, Iso))")
    print(f"    - 说明: 排除离群点后的稳健距离度量")
    print(f"  ASD (平均表面距离): {metrics['ASD']:.6g}")
    print(f"    - 定义: mean(d(OBJ, Iso))")
    print(f"    - 说明: OBJ 曲面到 Iso 曲面的平均距离")

    # 6) 交互式可视化
    print("\n【可视化】")
    print("  正在启动交互式可视化界面...")
    print("  使用说明:")
    print("    - 鼠标左键旋转，右键缩放，中键平移")
    print("    - 滚轮调整缩放")

    plot_interactive(iso, obj, d_obj_to_iso,
                     metrics, clim=CLIM, screenshot=SCREENSHOT_PATH)

    print("\n程序执行完成！")


if __name__ == '__main__':
    main()
