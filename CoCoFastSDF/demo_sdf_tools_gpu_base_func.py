import os
import time

from sdf_tools_gpu import (
    parse_obj, compute_sdf_grid,
    visualize_zero_isosurface, save_timings_pie, save_sdf_and_meta
)

# -------------------- 环境/参数（可按需调整） --------------------
# FWN 未决带（越小越少回退；1e-5~1e-4 之间调）
os.environ.setdefault("SDF_FWN_TAU", "1e-5")
# AABB/FWN 分批大小（CPU 内存足时可加大以提速）
os.environ.setdefault("SDF_AABB_BATCH", "600000")
os.environ.setdefault("SDF_FWN_BATCH", "3000000")
# GPU 距离（torch3d）一次送入的点数（依显存调整；越大越快）
os.environ.setdefault("SDF_TORCH3D_POINTS_CHUNK", "2000000")

# -------------------- 后端探测：优先用 GPU，失败回退 CPU --------------------
def pick_backend() -> str:
    try:
        import torch
        from pytorch3d.structures import Meshes  # noqa: F401
        from pytorch3d.loss import point_mesh_distance  # noqa: F401
        if torch.cuda.is_available():
            return "torch3d_fwn"
    except Exception:
        pass
    return "fwn_aabb"

# 1) 读入网格
# 获取当前脚本所在目录的父目录（项目根目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
obj_path = os.path.join(project_root, "obj_library", "sphere.obj")

# 从obj文件路径中提取文件名（不含扩展名）作为输出前缀
obj_filename = os.path.splitext(os.path.basename(obj_path))[0]
output_prefix = os.path.join("outputs", obj_filename)

V, F = parse_obj(obj_path)

# 2) 选择后端并计算 SDF
backend = pick_backend()
print(f"[Info] Selected backend: {backend}")

t0 = time.time()
try:
    sdf, bounds, _, _, timings, voxel_step = compute_sdf_grid(
        V, F,
        padding=0.1,
        voxel_size=None,          # 留空则按 target_resolution / max_resolution 推断体素步长
        target_resolution=512,    # 建议 128/256/512；或置 None 用 max_resolution
        max_resolution=512,
        show_progress=True,
        workers=(-1 if backend != "torch3d_fwn" else 1),  # GPU 距离不吃 CPU 线程；CPU 路径开满线程
        sdf_backend=backend,
    )
except Exception as e:
    if backend == "torch3d_fwn":
        print("[Warn] GPU 路径失败，回退到 fwn_aabb（CPU）:", e)
        backend = "fwn_aabb"
        sdf, bounds, _, _, timings, voxel_step = compute_sdf_grid(
            V, F,
            padding=0.1,
            voxel_size=None,
            target_resolution=512,
            max_resolution=512,
            show_progress=True,
            workers=-1,
            sdf_backend="fwn_aabb",
        )
    else:
        raise
    
# 3) 打印/保存耗时（饼图已自动忽略 'total'）
for k, v in timings.items():
    print(f"{k:>20s}: {v:.3f}s")
save_timings_pie(timings, f"{output_prefix}_timings_pie.png")

# 4) 保存零等值面可视化（紧包、无坐标轴/网格、光照+透明度）
visualize_zero_isosurface(sdf, bounds, out_path=f"{output_prefix}_isosurface.png")

# 5) 保存 SDF 和元信息
save_sdf_and_meta(
    sdf_grid=sdf,
    bounds=bounds,
    obj_path=obj_path,
    voxel_step=voxel_step,
    padding=0.1,
    timings=timings,
    out_prefix=output_prefix
)

# 6) （可选）PyVista 交互可视化
try:
    from sdf_tools_gpu import pyvista_visualize_isosurface
    pyvista_visualize_isosurface(sdf, bounds, show=True, out_png=None)
except Exception as e:
    print("[Info] PyVista 可视化跳过：", e)
