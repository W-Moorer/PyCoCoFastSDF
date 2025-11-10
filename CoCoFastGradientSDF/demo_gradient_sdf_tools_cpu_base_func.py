import os
import time

from gradient_sdf_tools_cpu import (   # ← 仍然从梯度版导入
    parse_obj, compute_sdf_grid,
    visualize_zero_isosurface,
    save_timings_pie,  # ← 用别名保持你的调用不变
    save_sdf_and_meta,
)

# -------------------- 环境/参数（可按需调整） --------------------
# 说明：这些环境变量在梯度版里并不会被读取，留着也无妨，删掉更干净。
os.environ.setdefault("SDF_FWN_TAU", "1e-5")
os.environ.setdefault("SDF_AABB_BATCH", "600000")
os.environ.setdefault("SDF_FWN_BATCH", "3000000")
os.environ.setdefault("SDF_TORCH3D_POINTS_CHUNK", "2000000")

# -------------------- 后端选择 --------------------
def pick_backend() -> str:
    """
    在梯度版 gradient_sdf_tools_gpu.py 里，compute_sdf_grid 的 sdf_backend 参数会被忽略，
    实际会自动优先使用 libigl.signed_distance；失败时回退到 AABB+FWN。
    这里保留该函数只是为了和你原始示例对齐。
    """
    try:
        import torch
        from pytorch3d.structures import Meshes  # noqa: F401
        from pytorch3d.loss import point_mesh_distance  # noqa: F401
        if torch.cuda.is_available():
            return "torch3d_fwn"  # ← 将被忽略，不影响运行
    except Exception:
        pass
    return "fwn_aabb"  # ← 同样会被忽略

# 1) 读入网格
# 获取当前脚本所在目录的父目录（项目根目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
obj_path = os.path.join(project_root, "obj_library", "gear.obj")

# 从obj文件路径中提取文件名（不含扩展名）作为输出前缀
obj_filename = os.path.splitext(os.path.basename(obj_path))[0]
output_prefix = os.path.join("gradient_outputs", obj_filename)

V, F = parse_obj(obj_path)

# 2) 选择后端并计算 SDF
backend = pick_backend()
print(f"[Info] Selected backend: {backend} (梯度版会自动选择实际可用的 CPU 路径)")

t0 = time.time()
try:
    sdf, bounds, _, _, timings, voxel_step = compute_sdf_grid(
        V, F,
        padding=0.1,
        voxel_size=None,          # 留空则按 target_resolution / max_resolution 推断体素步长
        target_resolution=512,    # 建议 128/256/512；或置 None 用 max_resolution
        max_resolution=512,
        show_progress=True,
        workers=(-1 if backend != "torch3d_fwn" else 1),  # 保留原写法；梯度版里该参数不会用到
        sdf_backend=backend,      # 将被忽略，但保留也不影响
    )
except Exception as e:
    # 正常情况下不会走到这里；保留原有回退逻辑以兼容你的旧脚本结构
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

# 5) 保存 SDF 和元信息（梯度版会始终额外写出 *_grad.npy）
save_sdf_and_meta(
    sdf_grid=sdf,
    bounds=bounds,
    obj_path=obj_path,
    voxel_step=voxel_step,
    padding=0.1,
    timings=timings,
    out_prefix=output_prefix
)
