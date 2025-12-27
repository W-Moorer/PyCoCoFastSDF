#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批处理脚本：批量计算 batch_results 中所有模型的 Hausdorff 距离误差，
并将结果保存为 ParaView 可视化的 VTK 文件。

用法：
    python batch_compute_hausdorff_error.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

# 添加项目路径到 sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# 导入 PyVista
try:
    import pyvista as pv
    from pyvista import _vtk as vtk
except ImportError as e:
    print(f"错误：需要安装 pyvista: pip install pyvista")
    sys.exit(1)


def load_meta(meta_json_path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """
    加载元数据文件
    
    参数:
        meta_json_path: 元数据 JSON 文件路径
        
    返回:
        (bmin, bmax, voxel_step): 边界最小值、最大值和体素步长
    """
    with open(meta_json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    bmin = np.array(meta.get('bounds_min'), dtype=float)
    bmax = np.array(meta.get('bounds_max'), dtype=float)
    voxel_step = tuple(float(x) for x in meta.get('voxel_step'))
    return bmin, bmax, voxel_step


def build_image_data_from_cells(sdf_grid: np.ndarray,
                                bmin: np.ndarray,
                                voxel_step: Tuple[float, float, float]) -> pv.ImageData:
    """
    使用 PyVista 的 ImageData 装载 SDF，并作为 cell_data 提取等值面
    
    参数:
        sdf_grid: SDF 网格数据
        bmin: 边界最小值
        voxel_step: 体素步长
        
    返回:
        pv.ImageData: PyVista 图像数据对象
    """
    nx, ny, nz = sdf_grid.shape
    spacing = tuple(float(x) for x in voxel_step)

    grid = pv.ImageData()
    grid.dimensions = (nx + 1, ny + 1, nz + 1)
    grid.origin = bmin.astype(float)
    grid.spacing = spacing

    grid.point_data.clear()
    grid.cell_data.clear()
    grid.cell_data["sdf"] = np.ascontiguousarray(sdf_grid).ravel(order='F')
    return grid


def extract_zero_isosurface(grid: pv.ImageData) -> pv.PolyData:
    """
    从 SDF 网格中提取零等值面
    
    参数:
        grid: PyVista 图像数据对象
        
    返回:
        pv.PolyData: 零等值面网格
    """
    grid_point_data = grid.cell_data_to_point_data()
    surf = grid_point_data.contour([0.0], scalars='sdf')
    surf = surf.triangulate().compute_normals(point_normals=True, cell_normals=False,
                                              auto_orient_normals=True, consistent_normals=True)
    return surf


def load_obj_surface(obj_path: str) -> pv.PolyData:
    """
    加载 OBJ 表面网格
    
    参数:
        obj_path: OBJ 文件路径
        
    返回:
        pv.PolyData: PyVista 多边形数据对象
    """
    m = pv.read(obj_path)
    if not isinstance(m, pv.PolyData):
        m = m.extract_surface()
    m = m.triangulate()
    return m


def _implicit_distance_numpy(points: np.ndarray, target: pv.PolyData) -> np.ndarray:
    """
    使用 VTK 的 vtkImplicitPolyDataDistance 计算点到目标曲面的绝对距离
    
    参数:
        points: 点坐标数组
        target: 目标曲面
        
    返回:
        np.ndarray: 每个点到目标曲面的距离
    """
    ipd = vtk.vtkImplicitPolyDataDistance()
    ipd.SetInput(target)
    out = np.empty((points.shape[0],), dtype=float)
    for i, p in enumerate(points):
        out[i] = ipd.EvaluateFunction(p)
    return np.abs(out)


def distances_A_to_B_surface(A: pv.PolyData, B: pv.PolyData) -> np.ndarray:
    """
    计算 A 的每个点到 B 表面的最小距离
    
    参数:
        A: 源网格
        B: 目标网格
        
    返回:
        np.ndarray: 距离数组
    """
    return _implicit_distance_numpy(A.points, B)


def summarize_unidirectional_distance(d: np.ndarray, pctl: float = 95.0) -> dict:
    """
    计算单向距离的统计指标
    
    参数:
        d: 距离数组
        pctl: 分位数（默认 95）
        
    返回:
        dict: 包含 HD, HDp, ASD 的字典
    """
    hd = float(np.max(d)) if d.size else 0.0
    hd_p = float(np.percentile(d, pctl)) if d.size else 0.0
    asd = float(np.mean(d)) if d.size else 0.0
    return dict(
        HD=hd,
        HDp=hd_p,
        ASD=asd,
    )


def save_vtk_polydata(mesh: pv.PolyData, output_path: str):
    """
    保存 PyVista PolyData 为 VTK XML PolyData (.vtp) 文件，可用于 ParaView 查看
    
    参数:
        mesh: PyVista 多边形数据对象
        output_path: 输出文件路径
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    mesh.save(output_path, binary=True)
    print(f"      VTK 文件已保存到: {output_path}")


def save_metrics_to_json(metrics: dict, output_path: str):
    """
    将统计指标保存为 JSON 文件
    
    参数:
        metrics: 统计指标字典
        output_path: 输出文件路径
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"      指标文件已保存到: {output_path}")


def find_all_models(batch_results_dir: str, obj_model_dir: str):
    """
    查找所有需要处理的模型
    
    参数:
        batch_results_dir: batch_results 目录
        obj_model_dir: obj_model 目录
        
    返回:
        List[Tuple]: (模型名称, 相对子路径, OBJ路径, SDF路径, Meta路径) 的列表
    """
    models = []
    batch_results_path = Path(batch_results_dir)
    obj_model_path = Path(obj_model_dir)
    
    if not batch_results_path.exists():
        print(f"错误：batch_results 目录不存在: {batch_results_dir}")
        return models
    
    # 遍历 batch_results 中的所有子文件夹
    for meta_file in batch_results_path.rglob("*_meta.json"):
        # 提取模型名称（去掉 _meta.json 后缀）
        model_name = meta_file.stem.replace("_meta", "")
        
        # 计算相对子路径
        rel_path = meta_file.relative_to(batch_results_path).parent
        
        # 构建对应的 OBJ 文件路径
        obj_path = obj_model_path / rel_path / f"{model_name}.obj"
        
        # 检查 OBJ 文件是否存在
        if not obj_path.exists():
            print(f"      警告：找不到对应的 OBJ 文件: {obj_path}")
            continue
        
        # 构建 SDF 文件路径
        sdf_path = meta_file.parent / f"{model_name}_sdf.npy"
        
        if not sdf_path.exists():
            print(f"      警告：找不到对应的 SDF 文件: {sdf_path}")
            continue
        
        models.append((
            model_name,
            str(rel_path),
            str(obj_path),
            str(sdf_path),
            str(meta_file)
        ))
    
    return models


def process_single_model(model_name: str, rel_path: str, obj_path: str, 
                          sdf_path: str, meta_path: str, output_dir: str, 
                          pctl: float = 95.0) -> bool:
    """
    处理单个模型，计算 Hausdorff 距离误差并保存结果
    
    参数:
        model_name: 模型名称
        rel_path: 相对子路径
        obj_path: OBJ 文件路径
        sdf_path: SDF 文件路径
        meta_path: 元数据文件路径
        output_dir: 输出目录
        pctl: 分位数（默认 95）
        
    返回:
        bool: 处理是否成功
    """
    print(f"\n{'='*80}")
    print(f"正在处理: {rel_path}/{model_name}")
    print(f"{'='*80}")
    
    try:
        # 1) 加载 SDF 与元数据
        print("[1/5] 正在加载数据...")
        print(f"      SDF 文件: {sdf_path}")
        print(f"      Meta 文件: {meta_path}")
        
        sdf_grid = np.load(sdf_path)
        print(f"      SDF 网格尺寸: {sdf_grid.shape}")
        
        bmin, bmax, voxel_step = load_meta(meta_path)
        print(f"      边界范围: [{bmin}, {bmax}]")
        print(f"      体素步长: {voxel_step}")
        
        # 2) 构建网格并提取零等值面
        print("[2/5] 正在提取零等值面...")
        grid = build_image_data_from_cells(sdf_grid, bmin=bmin, voxel_step=voxel_step)
        iso = extract_zero_isosurface(grid)
        print(f"      等值面顶点数: {iso.n_points}")
        print(f"      等值面三角面数: {iso.n_cells}")
        
        # 3) 加载 OBJ 表面
        print("[3/5] 正在加载 OBJ 文件...")
        print(f"      OBJ 文件: {obj_path}")
        obj = load_obj_surface(obj_path)
        print(f"      OBJ 顶点数: {obj.n_points}")
        print(f"      OBJ 三角面数: {obj.n_cells}")
        
        # 4) 计算 OBJ 到 Iso 的距离
        print("[4/5] 正在计算 Hausdorff 距离...")
        d_obj_to_iso = distances_A_to_B_surface(obj, iso)
        
        # 将距离数据添加到 OBJ 网格中
        obj["d_to_iso"] = d_obj_to_iso
        
        # 5) 统计指标
        print("[5/5] 正在计算统计指标...")
        metrics = summarize_unidirectional_distance(d_obj_to_iso, pctl=pctl)
        
        print(f"\n【评估结果】")
        print(f"  HD (Hausdorff 距离): {metrics['HD']:.6g}")
        print(f"  HD{pctl:.0f} ({pctl:.0f}% Hausdorff 距离): {metrics['HDp']:.6g}")
        print(f"  ASD (平均表面距离): {metrics['ASD']:.6g}")
        
        # 6) 保存结果
        print(f"\n【保存结果】")
        
        # 构建输出路径
        model_output_dir = os.path.join(output_dir, rel_path)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 保存 VTK 文件（用于 ParaView）
        vtk_output_path = os.path.join(model_output_dir, f"{model_name}_hausdorff_error.vtp")
        save_vtk_polydata(obj, vtk_output_path)
        
        # 保存指标文件
        metrics_output_path = os.path.join(model_output_dir, f"{model_name}_metrics.json")
        save_metrics_to_json(metrics, metrics_output_path)
        
        print(f"\n✓ 处理完成: {model_name}")
        return True
        
    except Exception as e:
        print(f"\n✗ 处理失败: {model_name}")
        print(f"  错误信息: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主函数：批量计算所有模型的 Hausdorff 距离误差
    """
    print("="*80)
    print("批量计算 Hausdorff 距离误差工具")
    print("="*80)
    
    # 配置参数
    BATCH_RESULTS_DIR = os.path.join(project_root, "batch_results")
    OBJ_MODEL_DIR = os.path.join(project_root, "obj_model")
    OUTPUT_DIR = os.path.join(project_root, "batch_hausdorff_results")
    PCTL = 95.0
    
    print(f"\n【配置参数】")
    print(f"  Batch Results 目录: {BATCH_RESULTS_DIR}")
    print(f"  OBJ Model 目录: {OBJ_MODEL_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  HDp 分位数: {PCTL}")
    
    # 查找所有模型
    print(f"\n【扫描模型】")
    print(f"  正在扫描 {BATCH_RESULTS_DIR} 目录...")
    models = find_all_models(BATCH_RESULTS_DIR, OBJ_MODEL_DIR)
    
    if not models:
        print(f"  未找到任何模型")
        return
    
    print(f"  找到 {len(models)} 个模型:")
    for model_name, rel_path, _, _, _ in models:
        print(f"    - {rel_path}/{model_name}")
    
    # 批处理所有模型
    print(f"\n【开始批处理】")
    print(f"  总共需要处理 {len(models)} 个模型")
    
    success_count = 0
    fail_count = 0
    
    for idx, (model_name, rel_path, obj_path, sdf_path, meta_path) in enumerate(models, 1):
        print(f"\n进度: [{idx}/{len(models)}]")
        
        success = process_single_model(
            model_name=model_name,
            rel_path=rel_path,
            obj_path=obj_path,
            sdf_path=sdf_path,
            meta_path=meta_path,
            output_dir=OUTPUT_DIR,
            pctl=PCTL
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"批处理完成！")
    print(f"{'='*80}")
    print(f"  成功处理: {success_count} 个")
    print(f"  失败处理: {fail_count} 个")
    print(f"  结果保存在: {OUTPUT_DIR}")
    print(f"\n使用 ParaView 打开 .vtp 文件进行可视化:")
    print(f"  1. 打开 ParaView")
    print(f"  2. File -> Open -> 选择 .vtp 文件")
    print(f"  3. 在 Color by 中选择 'd_to_iso' 查看距离热力图")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
