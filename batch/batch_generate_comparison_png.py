#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成原始网格与零等值面对比可视化PNG图片
右侧：零等值面模型（从SDF提取）
左侧：原始OBJ网格模型
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np

# PyVista / VTK
try:
    import pyvista as pv
except Exception as e:
    raise RuntimeError("需要安装 pyvista：pip install pyvista") from e


# ==================== 参数定义 ====================
BATCH_RESULTS_DIR = "../batch_results"      # SDF结果目录
OBJ_MODEL_DIR = "../obj_model"              # OBJ模型目录
OUTPUT_DIR = "../comparison_png_results"    # PNG输出目录

ISO_COLOR = (0.6, 0.8, 0.9)       # 零等值面颜色（浅青色）
ISO_OPACITY = 0.7                  # 零等值面透明度
OBJ_COLOR = (0.3, 0.4, 0.7)        # 原始模型颜色（深蓝色）
OBJ_OPACITY = 0.7                  # 原始模型透明度
SHOW_EDGES = False                 # 是否显示网格边线
# =================================================


def load_meta(meta_json_path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """
    加载元数据文件

    Args:
        meta_json_path: 元数据JSON文件路径

    Returns:
        (bounds_min, bounds_max, voxel_step)
    """
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
    将SDF数据构建为PyVista ImageData对象

    Args:
        sdf_grid: SDF体素数据 (nx, ny, nz)
        bmin: 边界最小值
        voxel_step: 体素步长

    Returns:
        PyVista ImageData对象
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
    """
    从ImageData中提取零等值面

    Args:
        grid_with_point_sdf: 包含点数据SDF的ImageData

    Returns:
        零等值面PolyData对象
    """
    surf = grid_with_point_sdf.contour([0.0], scalars="sdf")
    surf = surf.triangulate()
    return surf


def load_obj_model(obj_path: str) -> pv.PolyData:
    """
    加载OBJ模型

    Args:
        obj_path: OBJ文件路径

    Returns:
        PyVista PolyData对象
    """
    mesh = pv.read(obj_path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.triangulate()
    return mesh


def plot_comparison_and_save(obj_mesh: pv.PolyData,
                             iso_surface: pv.PolyData,
                             output_png_path: str):
    """
    创建对比可视化并保存为PNG：左侧原始模型，右侧零等值面

    Args:
        obj_mesh: 原始OBJ网格模型
        iso_surface: 零等值面模型
        output_png_path: PNG输出文件路径
    """
    # 创建绘图器，使用双视图布局
    plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800], off_screen=True)
    plotter.set_background('white')

    # ===== 左侧视图：原始OBJ模型 =====
    plotter.subplot(0, 0)
    
    # 添加原始模型
    plotter.add_mesh(obj_mesh, 
                     color=OBJ_COLOR, 
                     opacity=OBJ_OPACITY,
                     show_edges=SHOW_EDGES)
    
    plotter.view_isometric()

    # ===== 右侧视图：零等值面模型 =====
    plotter.subplot(0, 1)
    
    # 添加零等值面
    plotter.add_mesh(iso_surface,
                     color=ISO_COLOR,
                     opacity=ISO_OPACITY,
                     show_edges=SHOW_EDGES)
    
    plotter.view_isometric()

    # 保存为PNG格式
    plotter.screenshot(output_png_path)
    print(f"    PNG已保存到: {output_png_path}")


def process_single_model(model_name: str, rel_path: str, 
                        sdf_path: str, meta_path: str, obj_path: str, 
                        output_dir: str) -> bool:
    """
    处理单个模型，生成对比可视化SVG

    Args:
        model_name: 模型名称
        rel_path: 相对路径
        sdf_path: SDF文件路径
        meta_path: 元数据文件路径
        obj_path: OBJ文件路径
        output_dir: 输出目录

    Returns:
        是否处理成功
    """
    try:
        print(f"  处理模型: {model_name}")
        
        # 1) 加载SDF数据
        sdf_grid = np.load(sdf_path)
        
        # 2) 加载元数据
        bmin, bmax, voxel_step = load_meta(meta_path)
        
        # 3) 构建ImageData并提取零等值面
        grid = build_image_data(sdf_grid, bmin=bmin, voxel_step=voxel_step)
        grid_point_data = grid.cell_data_to_point_data()
        iso_surface = extract_zero_isosurface(grid_point_data)
        
        # 4) 加载OBJ模型
        obj_mesh = load_obj_model(obj_path)
        
        # 5) 创建输出目录
        model_output_dir = os.path.join(output_dir, rel_path)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 6) 生成对比可视化并保存PNG
        png_output_path = os.path.join(model_output_dir, f"{model_name}_comparison.png")
        plot_comparison_and_save(obj_mesh, iso_surface, png_output_path)
        
        return True
    except Exception as e:
        print(f"  处理失败: {model_name}, 错误: {e}")
        return False


def discover_models(batch_results_dir: str, obj_model_dir: str):
    """
    递归发现所有模型及其对应的文件路径

    Args:
        batch_results_dir: SDF结果目录
        obj_model_dir: OBJ模型目录

    Yields:
        (model_name, rel_path, sdf_path, meta_path, obj_path)
    """
    batch_results_path = Path(batch_results_dir)
    
    # 递归遍历batch_results目录
    for sdf_file in batch_results_path.rglob("*_sdf.npy"):
        # 获取相对路径
        rel_path = sdf_file.parent.relative_to(batch_results_path)
        
        # 提取模型名称（去掉_sdf.npy后缀）
        model_name = sdf_file.stem.replace("_sdf", "")
        
        # 构建其他文件路径
        meta_path = sdf_file.parent / f"{model_name}_meta.json"
        obj_path = Path(obj_model_dir) / rel_path / f"{model_name}.obj"
        
        # 检查所有必需文件是否存在
        if not meta_path.exists():
            print(f"  跳过 {model_name}: 元数据文件不存在 {meta_path}")
            continue
        if not obj_path.exists():
            print(f"  跳过 {model_name}: OBJ文件不存在 {obj_path}")
            continue
        
        yield model_name, str(rel_path), str(sdf_file), str(meta_path), str(obj_path)


def main():
    """
    主函数：批量生成对比可视化SVG图片
    """
    print("=" * 60)
    print("批量生成原始网格与零等值面对比可视化SVG图片")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 统计变量
    total_count = 0
    success_count = 0
    fail_count = 0
    
    # 遍历所有模型
    print(f"\n开始处理模型...")
    print(f"SDF结果目录: {BATCH_RESULTS_DIR}")
    print(f"OBJ模型目录: {OBJ_MODEL_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    
    for model_name, rel_path, sdf_path, meta_path, obj_path in discover_models(BATCH_RESULTS_DIR, OBJ_MODEL_DIR):
        total_count += 1
        
        # 处理单个模型
        success = process_single_model(model_name, rel_path, sdf_path, meta_path, obj_path, OUTPUT_DIR)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        print()
    
    # 输出统计结果
    print("=" * 60)
    print("批量处理完成！")
    print(f"总模型数: {total_count}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
