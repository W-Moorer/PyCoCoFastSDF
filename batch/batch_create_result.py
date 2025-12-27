#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批处理脚本：递归遍历 obj_model 文件夹中的所有 OBJ 模型，
计算 SDF 并将结果保存到 batch_results/[模型名称]/ 目录中。

用法：
    python batch_create_result.py
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# 添加项目路径到 sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# 导入 SDF 计算工具
try:
    from CoCoFastTraditionalSDF.sdf_tools_gpu import (
        parse_obj, compute_sdf_grid, save_sdf_and_meta
    )
except ImportError as e:
    print(f"错误：无法导入 SDF 工具模块: {e}")
    sys.exit(1)


def find_all_obj_files(root_dir: str) -> List[Tuple[str, str]]:
    """
    递归查找指定目录下的所有 .obj 文件
    
    参数:
        root_dir: 根目录路径
        
    返回:
        List[Tuple[str, str]]: (obj文件路径, 相对子路径) 的列表
    """
    obj_files = []
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"错误：目录不存在: {root_dir}")
        return obj_files
    
    for obj_file in root_path.rglob("*.obj"):
        # 计算相对于根目录的子路径
        rel_path = obj_file.relative_to(root_path).parent
        obj_files.append((str(obj_file), str(rel_path)))
    
    return obj_files


def process_single_obj(obj_path: str, output_dir: str, target_resolution: int = 512, max_resolution: int = 512) -> bool:
    """
    处理单个 OBJ 文件，计算 SDF 并保存结果
    
    参数:
        obj_path: OBJ 文件路径
        output_dir: 输出目录路径
        target_resolution: 目标分辨率
        max_resolution: 最大分辨率
        
    返回:
        bool: 处理是否成功
    """
    print(f"\n{'='*80}")
    print(f"正在处理: {obj_path}")
    print(f"{'='*80}")
    
    try:
        # 1) 读取 OBJ 文件
        print("[1/4] 正在读取 OBJ 文件...")
        V, F = parse_obj(obj_path)
        print(f"      顶点数: {len(V)}, 面数: {len(F)}")
        
        # 2) 计算 SDF
        print("[2/4] 正在计算 SDF...")
        t0 = time.time()
        
        # 尝试使用 GPU 后端，失败则回退到 CPU
        backend = "torch3d_fwn"
        try:
            import torch
            from pytorch3d.structures import Meshes
            from pytorch3d.loss import point_mesh_distance
            if not torch.cuda.is_available():
                backend = "fwn_aabb"
        except ImportError:
            backend = "fwn_aabb"
        
        print(f"      使用后端: {backend}")
        
        try:
            sdf, bounds, _, _, timings, voxel_step = compute_sdf_grid(
                V, F,
                padding=0.1,
                voxel_size=None,
                target_resolution=target_resolution,
                max_resolution=max_resolution,
                show_progress=True,
                workers=(-1 if backend != "torch3d_fwn" else 1),
                sdf_backend=backend,
            )
        except Exception as e:
            if backend == "torch3d_fwn":
                print(f"      GPU 后端失败，回退到 CPU: {e}")
                backend = "fwn_aabb"
                sdf, bounds, _, _, timings, voxel_step = compute_sdf_grid(
                    V, F,
                    padding=0.1,
                    voxel_size=None,
                    target_resolution=target_resolution,
                    max_resolution=max_resolution,
                    show_progress=True,
                    workers=-1,
                    sdf_backend="fwn_aabb",
                )
            else:
                raise
        
        elapsed = time.time() - t0
        print(f"      SDF 计算完成，耗时: {elapsed:.2f} 秒")
        
        # 3) 保存 SDF 和元数据
        print("[3/4] 正在保存结果...")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 从 OBJ 文件路径提取模型名称（不含扩展名）
        obj_filename = Path(obj_path).stem
        output_prefix = os.path.join(output_dir, obj_filename)
        
        save_sdf_and_meta(
            sdf_grid=sdf,
            bounds=bounds,
            obj_path=obj_path,
            voxel_step=voxel_step,
            padding=0.1,
            timings=timings,
            out_prefix=output_prefix
        )
        
        print(f"      结果已保存到: {output_prefix}")
        print(f"      生成的文件:")
        print(f"        - {output_prefix}_sdf.npy")
        print(f"        - {output_prefix}_meta.json")
        print(f"        - {output_prefix}_isosurface.png")
        print(f"        - {output_prefix}_timings_pie.png")
        
        # 4) 打印统计信息
        print("[4/4] 统计信息:")
        print(f"      SDF 网格形状: {sdf.shape}")
        print(f"      体素步长: {voxel_step}")
        print(f"      边界范围: {bounds}")
        for k, v in timings.items():
            if k != 'total':
                print(f"      {k}: {v:.3f}s")
        
        print(f"\n✓ 处理完成: {obj_path}")
        return True
        
    except Exception as e:
        print(f"\n✗ 处理失败: {obj_path}")
        print(f"  错误信息: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主函数：批处理所有 OBJ 模型
    """
    print("="*80)
    print("批处理 SDF 计算工具")
    print("="*80)
    
    # 配置参数
    OBJ_MODEL_DIR = os.path.join(project_root, "obj_model")
    BATCH_RESULTS_DIR = os.path.join(project_root, "batch_results")
    TARGET_RESOLUTION = 512
    MAX_RESOLUTION = 512
    
    print(f"\n【配置参数】")
    print(f"  OBJ 模型目录: {OBJ_MODEL_DIR}")
    print(f"  结果输出目录: {BATCH_RESULTS_DIR}")
    print(f"  目标分辨率: {TARGET_RESOLUTION}")
    print(f"  最大分辨率: {MAX_RESOLUTION}")
    
    # 查找所有 OBJ 文件
    print(f"\n【扫描模型文件】")
    print(f"  正在扫描 {OBJ_MODEL_DIR} 目录...")
    obj_files = find_all_obj_files(OBJ_MODEL_DIR)
    
    if not obj_files:
        print(f"  未找到任何 .obj 文件")
        return
    
    print(f"  找到 {len(obj_files)} 个 .obj 文件:")
    for obj_path, rel_path in obj_files:
        print(f"    - {rel_path}/{Path(obj_path).name}")
    
    # 批处理所有模型
    print(f"\n【开始批处理】")
    print(f"  总共需要处理 {len(obj_files)} 个模型")
    
    success_count = 0
    fail_count = 0
    total_start_time = time.time()
    
    for idx, (obj_path, rel_path) in enumerate(obj_files, 1):
        print(f"\n进度: [{idx}/{len(obj_files)}]")
        
        # 构建输出目录：batch_results/[相对子路径]/
        output_dir = os.path.join(BATCH_RESULTS_DIR, rel_path)
        
        # 处理单个模型
        success = process_single_obj(
            obj_path=obj_path,
            output_dir=output_dir,
            target_resolution=TARGET_RESOLUTION,
            max_resolution=MAX_RESOLUTION
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # 打印总结
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*80}")
    print(f"批处理完成！")
    print(f"{'='*80}")
    print(f"  总处理时间: {total_elapsed:.2f} 秒 ({total_elapsed/60:.2f} 分钟)")
    print(f"  成功处理: {success_count} 个")
    print(f"  失败处理: {fail_count} 个")
    print(f"  结果保存在: {BATCH_RESULTS_DIR}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
