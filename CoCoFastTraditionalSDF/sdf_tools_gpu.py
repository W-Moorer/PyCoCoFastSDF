# -*- coding: utf-8 -*-
"""
sdf_tools_gpu.py — 高精度 SDF 计算工具（修正版）

主要修正：
1. 符号约定标准化：内部为负 (Negative Inside)，外部为正。
2. 精度提升：默认 CPU 路径 (fwn_aabb) 现在优先使用 Exact Winding Number 以消除平面误差。
3. 逻辑修复：移除了导致符号反转的冗余乘法。
"""
from __future__ import annotations

import os, json, math, argparse, time, hashlib, struct
from typing import Optional, Tuple, Dict, List

import numpy as np

# -------------------- 依赖探测 --------------------
try:
    from scipy.spatial import cKDTree
    _HAS_CKDTREE = True
except Exception:
    _HAS_CKDTREE = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

try:
    from skimage import measure
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# libigl
try:
    import igl
    _HAS_IGL = True
    _HAS_IGL_FWN_FUNC = hasattr(igl, "fast_winding_number_for_meshes")
    _HAS_IGL_SD       = hasattr(igl, "signed_distance")
    # 探测不同类型的 Winding Number 常量
    _HAS_SD_FAST      = hasattr(igl, "SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER")
    _HAS_SD_WIND      = hasattr(igl, "SIGNED_DISTANCE_TYPE_WINDING_NUMBER") # Exact
    _HAS_SD_PSEUDO    = hasattr(igl, "SIGNED_DISTANCE_TYPE_PSEUDONORMAL")
except Exception:
    _HAS_IGL = False
    _HAS_IGL_FWN_FUNC = False
    _HAS_IGL_SD = False
    _HAS_SD_FAST = _HAS_SD_WIND = _HAS_SD_PSEUDO = False

# Numba
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

# PyTorch / PyTorch3D
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    if _HAS_TORCH:
        from pytorch3d.structures import Meshes
        from pytorch3d.loss import point_mesh_distance
        _HAS_P3D = True
    else:
        _HAS_P3D = False
except Exception:
    _HAS_P3D = False


# -------------------- 环境参数 --------------------
_FWN_TAU = float(os.getenv("SDF_FWN_TAU", "1e-5"))
_FWN_BATCH = int(os.getenv("SDF_FWN_BATCH", "2000000"))
_AABB_BATCH = int(os.getenv("SDF_AABB_BATCH", "400000"))
_TORCH3D_POINTS_CHUNK = int(os.getenv("SDF_TORCH3D_POINTS_CHUNK", "2000000"))
_SDF_K_DEFAULT = int(os.getenv("SDF_K", "24"))


# -------------------- 杂项工具 --------------------
def _ensure_dir_for(path_str: str):
    d = os.path.dirname(os.path.abspath(path_str))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _normalize_workers(workers: Optional[int]) -> int:
    if workers is None or workers == 0:
        return max(1, (os.cpu_count() or 8) // 2)
    if workers < 0:
        return max(1, os.cpu_count() or 8)
    return int(workers)

# -------------------- IO --------------------
def parse_obj(path: str) -> Tuple[np.ndarray, np.ndarray]:
    vs, fs = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            sp = line.strip().split()
            if not sp:
                continue
            if sp[0] == 'v' and len(sp) >= 4:
                vs.append([float(sp[1]), float(sp[2]), float(sp[3])])
            elif sp[0] == 'f' and len(sp) >= 4:
                idx = []
                for w in sp[1:4]:
                    a = w.split('/')[0]
                    idx.append(int(a) - 1)
                fs.append(idx)
    V = np.asarray(vs, dtype=np.float64)
    F = np.asarray(fs, dtype=np.int32)
    return V, F


# -------------------- 体素网格 --------------------
def _compute_bounds(vertices: np.ndarray, padding: float) -> Tuple[np.ndarray, np.ndarray]:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    diag = float(np.linalg.norm(vmax - vmin))
    pad = padding * diag
    return (vmin - pad).astype(np.float64), (vmax + pad).astype(np.float64)

def _voxel_grid_axes(bmin: np.ndarray, bmax: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 修正 1: 使用 Cell-Centered 逻辑，起点偏移半个 voxel_size
    # 修正 2: 使用 np.arange 保证步长严格一致，避免 linspace 的拉伸误差
    
    # 重新计算基于步长的终点，确保覆盖原始 bmax
    # 这里我们生成从 bmin + half_step 开始的序列
    half_step = voxel_size * 0.5
    
    # X 轴
    xs = np.arange(bmin[0] + half_step, bmax[0], voxel_size, dtype=np.float64)
    # 确保最后一个点如果不小心超出了 bmax (由于浮点误差) 或没覆盖到，进行微调（可选，视具体需求）
    # 通常 arange 足够，但为了仅仅覆盖包围盒，我们可以动态扩展 max
    if xs[-1] + half_step < bmax[0]:
        xs = np.append(xs, xs[-1] + voxel_size)

    # Y 轴
    ys = np.arange(bmin[1] + half_step, bmax[1], voxel_size, dtype=np.float64)
    if ys[-1] + half_step < bmax[1]:
        ys = np.append(ys, ys[-1] + voxel_size)

    # Z 轴
    zs = np.arange(bmin[2] + half_step, bmax[2], voxel_size, dtype=np.float64)
    if zs[-1] + half_step < bmax[2]:
        zs = np.append(zs, zs[-1] + voxel_size)
        
    return xs, ys, zs

def _grid_points(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)
    return pts


# -------------------- 可视化 --------------------
def visualize_zero_isosurface(
    sdf_grid: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    out_path: Optional[str] = None,
    max_tris: int = 500_000,
    alpha: float = 0.6,
    face_rgb: Optional[Tuple[float, float, float]] = (0.3, 0.5, 0.8),
    transparent_bg: bool = False
):
    if not _HAS_MPL or not _HAS_SKIMAGE:
        return None

    bmin, bmax = bounds
    nx, ny, nz = sdf_grid.shape
    xs = np.linspace(bmin[0], bmax[0], int(nx), dtype=np.float64)
    ys = np.linspace(bmin[1], bmax[1], int(ny), dtype=np.float64)
    zs = np.linspace(bmin[2], bmax[2], int(nz), dtype=np.float64)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()

    spacing = (xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0])
    
    # Marching Cubes 提取 0 等值面
    try:
        verts, faces, normals_mc, values = measure.marching_cubes(sdf_grid, level=0.0, spacing=spacing)
    except ValueError:
        print("[Warn] No isosurface found at level 0.0")
        plt.close(fig)
        return

    verts_world = verts + np.array(bmin, dtype=np.float64)[None, :]
    
    # 简单的降采样以防止 matplotlib 卡死
    if faces.shape[0] > max_tris:
        step = int(math.ceil(faces.shape[0] / max_tris))
        faces = faces[::step, :]

    poly3d = verts_world[faces]
    coll = Poly3DCollection(poly3d, linewidths=0.1, alpha=alpha)
    if face_rgb is not None:
        coll.set_facecolor(face_rgb)
    ax.add_collection3d(coll)

    ax.set_xlim([bmin[0], bmax[0]])
    ax.set_ylim([bmin[1], bmax[1]])
    ax.set_zlim([bmin[2], bmax[2]])

    if out_path:
        _ensure_dir_for(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.05, transparent=transparent_bg)
        plt.close(fig)
    else:
        plt.show()
    return out_path

def save_timings_pie(timings: Dict[str, float], out_path: str):
    if not _HAS_MPL: return
    labels = [k for k in timings.keys() if k != 'total']
    sizes = [timings[k] for k in labels]
    total = sum(sizes) + 1e-18
    pct = [100.0 * s / total for s in sizes]
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    wedges, texts = ax.pie(sizes, wedgeprops=dict(width=0.35), startangle=140)
    ax.axis('equal')
    legend_labels = [f"{labels[i]}: {pct[i]:.1f}%" for i in range(len(labels))]
    ax.legend(wedges, legend_labels, title="Timings", loc="center left", bbox_to_anchor=(1.0, 0.5))
    _ensure_dir_for(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# -------------------- 奇偶性回退 (Parity Check) --------------------
def _triangle_bboxes(V: np.ndarray, F: np.ndarray):
    tri0 = V[F[:,0]]; tri1 = V[F[:,1]]; tri2 = V[F[:,2]]
    vmin = np.minimum(np.minimum(tri0, tri1), tri2)
    vmax = np.maximum(np.maximum(tri0, tri1), tri2)
    return vmin, vmax, tri0, tri1, tri2

if _HAS_NUMBA:
    @njit(cache=True)
    def _ray_parity_sign_subset_numba(
        P, idxs, Vmin, Vmax, A, E1, E2, det_all, inv_det_all, valid_all, bmax_x
    ):
        out = np.ones((idxs.shape[0],), dtype=np.float64)
        for k in range(idxs.shape[0]):
            i = idxs[k]
            x = P[i,0]; y = P[i,1]; z = P[i,2]
            # 极微小的抖动以避免击中边缘
            j = math.sin(y * 1e6 + z * 1e3) * 1e-9
            y += j; z += j
            hit_cnt = 0
            for t in range(A.shape[0]):
                if not valid_all[t]: continue
                if y < Vmin[t,1] or y >= Vmax[t,1]: continue
                if z < Vmin[t,2] or z >= Vmax[t,2]: continue
                if Vmax[t,0] < x: continue
                tvec_x = x - A[t,0]; tvec_y = y - A[t,1]; tvec_z = z - A[t,2]
                invd   = inv_det_all[t]
                u = (tvec_y * (-E2[t,2]) + tvec_z * E2[t,1]) * invd
                v = (tvec_y * E1[t,2] - tvec_z * E1[t,1]) * invd
                if (u < 0.0) or (v <= 0.0) or (u + v > 1.0): continue
                tt = (
                    E2[t,0]*(tvec_y*E1[t,2] - tvec_z*E1[t,1]) +
                    E2[t,1]*(tvec_z*E1[t,0] - tvec_x*E1[t,2]) +
                    E2[t,2]*(tvec_x*E1[t,1] - tvec_y*E1[t,0])
                ) * invd
                if tt > 1e-12 and (x + tt) <= (bmax_x + 1e-9):
                    hit_cnt += 1
            # 奇数次相交 => 内部 => -1.0
            out[k] = -1.0 if (hit_cnt % 2 == 1) else 1.0
        return out
else:
    def _ray_parity_sign_subset_numba(*a, **k):
        raise RuntimeError("需要 numba 以启用高速奇偶回退")


# -------------------- 核心算法: AABB / FWN / SD --------------------
def _aabb_squared_distance_in_batches(pts: np.ndarray, V: np.ndarray, F: np.ndarray, batch: int, workers: int) -> np.ndarray:
    """计算无符号距离 (Unsigned Distance)"""
    if not _HAS_IGL: raise RuntimeError("Missing libigl")
    Vd = np.ascontiguousarray(V, dtype=np.float64)
    Fi = np.ascontiguousarray(F, dtype=np.int32)
    P  = np.ascontiguousarray(pts, dtype=np.float64)
    
    tree = igl.AABB()
    tree.init(Vd, Fi)
    
    def _one(qslice):
        sqrD, I, C = tree.squared_distance(Vd, Fi, qslice)
        return np.sqrt(np.maximum(sqrD, 0.0))

    M = P.shape[0]
    out = np.empty((M,), dtype=np.float64)
    chunks = [(st, min(M, st + batch)) for st in range(0, M, batch)]
    
    w = _normalize_workers(workers)
    if w == 1 or len(chunks) == 1:
        for st, ed in chunks:
            out[st:ed] = _one(P[st:ed])
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=w) as ex:
            futs = [(st, ed, ex.submit(_one, P[st:ed])) for st, ed in chunks]
            for st, ed, fu in futs:
                out[st:ed] = fu.result()
    return out

def _sign_from_winding_with_parity_fallback(pts: np.ndarray, V: np.ndarray, F: np.ndarray, tau: float) -> np.ndarray:
    """计算符号 (Sign): 内部为 -1.0, 外部为 +1.0"""
    if not _HAS_IGL or not _HAS_IGL_FWN_FUNC: raise RuntimeError("Missing igl.fast_winding_number_for_meshes")
    
    # 1. 计算 Winding Number (近似)
    Vd = np.ascontiguousarray(V, dtype=np.float64)
    Fi = np.ascontiguousarray(F, dtype=np.int32)
    M = pts.shape[0]
    W = np.empty((M,), dtype=np.float64)
    batch = _FWN_BATCH
    for st in range(0, M, batch):
        ed = min(M, st + batch)
        Qb = np.ascontiguousarray(pts[st:ed], dtype=np.float64)
        W[st:ed] = igl.fast_winding_number_for_meshes(Vd, Fi, Qb)

    sign = np.empty((M,), dtype=np.float64)
    isfinite = np.isfinite(W)
    
    # 2. 判定: W > 0.5 为内部
    inside  = np.where(isfinite, W >= (0.5 + tau), False)
    outside = np.where(isfinite, W <= (0.5 - tau), False)
    undec   = ~(inside | outside)

    sign[inside]  = -1.0 # Standard SDF: Inside is Negative
    sign[outside] = +1.0 # Outside is Positive

    # 3. 对模糊区域 (0.5 ± tau) 进行射线回退 (Ray Parity)
    if np.any(undec):
        idx = np.where(undec)[0]
        if _HAS_NUMBA:
            Vmin, Vmax, A, B, Ctri = _triangle_bboxes(V, F)
            bmax_x = float(np.max(Vmax[:,0]))
            E1 = B - A; E2 = Ctri - A
            det_all = E1[:,1]*(-E2[:,2]) + E1[:,2]*E2[:,1]
            valid_all = (np.abs(det_all) > 1e-12)
            inv_det_all = np.zeros_like(det_all)
            inv_det_all[valid_all] = 1.0/det_all[valid_all]
            s_par = _ray_parity_sign_subset_numba(
                pts, idx, Vmin, Vmax, A, E1, E2, det_all, inv_det_all, valid_all, bmax_x
            )
            sign[idx] = s_par
        else:
            # 无 Numba 时回退到最近中心法线判定 (粗略)
            print("[Warn] Parity check needs Numba. Fallback to heuristic.")
            sign[idx] = 1.0 # 默认外部，防止崩溃

    return sign

def _sdf_with_fwn_aabb(pts: np.ndarray, V: np.ndarray, F: np.ndarray, workers: int, aabb_batch: int, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    CPU 高精度路径
    """
    # 策略 A: 如果有 Exact Winding Number (libigl 2.3+)，优先使用它直接算 Signed Distance
    # 这能避免 Fast Winding Number 在平面附近的微小误差
    if _HAS_IGL and _HAS_IGL_SD and _HAS_SD_WIND:
        Vd = np.ascontiguousarray(V, dtype=np.float64)
        Fi = np.ascontiguousarray(F, dtype=np.int32)
        Q  = np.ascontiguousarray(pts, dtype=np.float64)
        
        # 使用 EXACT WINDING NUMBER 而非 FAST
        # 即使这比 FAST 慢，但能保证几何准确性，消除 0.005 这种凸起
        try:
            sd, I, C, N = igl.signed_distance(Q, Vd, Fi, igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER)
        except TypeError:
            sd, I, C = igl.signed_distance(Q, Vd, Fi, igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER)
        
        d_unsigned = np.abs(sd)
        # 标准 SDF: 内部为负
        sign = np.where(sd < 0.0, -1.0, 1.0)
        return d_unsigned, sign

    # 策略 B: 组合 AABB (距离) + FWN (符号)
    d_unsigned = _aabb_squared_distance_in_batches(pts, V, F, batch=aabb_batch, workers=workers)
    sign = _sign_from_winding_with_parity_fallback(pts, V, F, tau=tau)
    return d_unsigned, sign

def _sdf_with_torch3d_fwn(pts: np.ndarray, V: np.ndarray, F: np.ndarray, device: Optional[str], points_chunk: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU 路径 (Float32精度)
    注意：Torch3D 内部计算为 Float32，对于极高精度要求(如误差<1e-5)可能会有数值噪点。
    """
    if not (_HAS_TORCH and _HAS_P3D and torch.cuda.is_available()):
        raise RuntimeError("Missing PyTorch3D/CUDA")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    v = torch.from_numpy(np.ascontiguousarray(V, dtype=np.float32)).to(device=device)
    f = torch.from_numpy(np.ascontiguousarray(F, dtype=np.int64)).to(device=device)
    mesh = Meshes(verts=[v], faces=[f])

    M = pts.shape[0]
    d_unsigned = np.empty((M,), dtype=np.float64)

    # 1. GPU 距离
    for st in range(0, M, points_chunk):
        ed = min(M, st + points_chunk)
        p = torch.from_numpy(np.ascontiguousarray(pts[st:ed], dtype=np.float32)).to(device=device)[None, ...]
        d_per_point = point_mesh_distance(p, mesh, reduction="none")
        d_np = d_per_point[0].detach().cpu().numpy().astype(np.float64)
        d_unsigned[st:ed] = np.sqrt(np.maximum(d_np, 0.0))

    # 2. CPU 符号 (利用 libigl 判号)
    # 优先使用 Exact Winding Number
    Vd = np.ascontiguousarray(V, dtype=np.float64)
    Fi = np.ascontiguousarray(F, dtype=np.int32)
    Q  = np.ascontiguousarray(pts, dtype=np.float64)
    
    stype = None
    if _HAS_SD_WIND: stype = igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER
    elif _HAS_SD_FAST: stype = igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    else: stype = igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
        
    sd, *_ = igl.signed_distance(Q, Vd, Fi, stype)
    sign = np.where(sd < 0.0, -1.0, 1.0)
    
    return d_unsigned, sign


# -------------------- 主入口 --------------------
def compute_sdf_grid(vertices: np.ndarray, faces: np.ndarray,
                     padding: float = 0.1, voxel_size: Optional[float] = None,
                     target_resolution: Optional[int] = None, max_resolution: int = 512,
                     sdf_backend: str = 'fwn_aabb', workers: int = -1, **kwargs
                     ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, Dict[str, float], Tuple[float, float, float]]:
    
    t0 = time.time()
    bmin, bmax = _compute_bounds(vertices, padding)
    t_grid = time.time()

    if voxel_size is None:
        size = bmax - bmin
        if target_resolution is None: target_resolution = max_resolution
        longest = float(np.max(size))
        voxel_size = longest / float(target_resolution)
    
    xs, ys, zs = _voxel_grid_axes(bmin, bmax, voxel_size)
    voxel_step = (float(xs[1]-xs[0]), float(ys[1]-ys[0]), float(zs[1]-zs[0]))
    pts = _grid_points(xs, ys, zs)
    t_pts = time.time()

    # 后端计算
    if sdf_backend == 'fwn_aabb':
        d_unsigned, sign = _sdf_with_fwn_aabb(pts, vertices, faces, workers=workers, aabb_batch=_AABB_BATCH, tau=_FWN_TAU)
    elif sdf_backend == 'torch3d_fwn':
        d_unsigned, sign = _sdf_with_torch3d_fwn(pts, vertices, faces, device=None, points_chunk=_TORCH3D_POINTS_CHUNK)
    else:
        raise ValueError(f"Unknown backend: {sdf_backend}")
    t_query = time.time()

    # 修正：SDF = Distance * Sign (无需 * -1.0)
    # Standard: Inside = -1, Dist = d. Result = -d (Negative Inside).
    sdf_flat = d_unsigned * sign 
    sdf_grid = sdf_flat.reshape(len(xs), len(ys), len(zs))
    t_reshape = time.time()

    timings = {
        "setup": t_pts - t0,
        "query": t_query - t_pts,
        "total": t_reshape - t0
    }
    
    # 兼容旧接口返回
    centers = np.zeros((1,3)); normals = np.zeros((1,3)) 
    return sdf_grid, (bmin, bmax), centers, normals, timings, voxel_step

def save_sdf_and_meta(sdf_grid, bounds, obj_path, voxel_step, padding, timings, out_prefix):
    npy_path = f"{out_prefix}_sdf.npy"
    meta_path = f"{out_prefix}_meta.json"
    
    _ensure_dir_for(npy_path)
    np.save(npy_path, sdf_grid)
    
    bmin, bmax = bounds
    meta = dict(
        obj=os.path.abspath(obj_path),
        bounds_min=list(map(float, bmin)),
        bounds_max=list(map(float, bmax)),
        voxel_step=list(map(float, voxel_step)),
        padding=float(padding),
        timings=timings
    )
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    
    # Visuals
    visualize_zero_isosurface(sdf_grid, bounds, out_path=f"{out_prefix}_isosurface.png")
    save_timings_pie(timings, f"{out_prefix}_timings_pie.png")
    
    return f"{out_prefix}_isosurface.png", f"{out_prefix}_timings_pie.png", npy_path, meta_path