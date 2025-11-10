# -*- coding: utf-8 -*-
"""
sdf_tools_gpu.py — 高效而鲁棒的 SDF 计算工具（默认：FWN/SD + AABB；含 GPU 距离选项、可视化与 CLI）

默认后端 fwn_aabb：
- 距离：libigl.AABB 的“真最近距离”（分批，多线程）
- 符号：优先使用 libigl.signed_distance 的 FAST_WINDING_NUMBER/WINDING_NUMBER 一步到位；
        若不可用，则用 FWN（fast_winding_number_for_meshes）计算绕数并对 0.5 极窄带回退一次奇偶射线。
- 彻底避开“伪法线 + 角度阈值”的易错逻辑，从算法上解决齿轮等尖锐结构的“气泡”问题。

可选 GPU 后端 torch3d_fwn（若 PyTorch3D+CUDA 可用）：
- 距离：GPU 上的 point–mesh 最近距离核（PyTorch3D，要求支持 reduction='none'）
- 符号：仍用 libigl 的 signed_distance（FWN/WINDING），保证几何正确。
- 若 GPU 不可用或 API 不匹配，会自动回退到 fwn_aabb（CPU）。

保留旧后端 nn 作为对照（kNN 最近中心，非默认）。
"""
from __future__ import annotations

import os, json, math, argparse, time, hashlib
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

# libigl（AABB/SD/FWN）
try:
    import igl
    _HAS_IGL = True
    # ---- libigl 能力探测（不同 Wheel/版本导出不一）----
    _HAS_IGL_FWN_FUNC = hasattr(igl, "fast_winding_number_for_meshes")
    _HAS_IGL_SD       = hasattr(igl, "signed_distance")
    _HAS_SD_FAST      = hasattr(igl, "SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER")
    _HAS_SD_WIND      = hasattr(igl, "SIGNED_DISTANCE_TYPE_WINDING_NUMBER")
    _HAS_SD_PSEUDO    = hasattr(igl, "SIGNED_DISTANCE_TYPE_PSEUDONORMAL")
except Exception:
    _HAS_IGL = False
    _HAS_IGL_FWN_FUNC = False
    _HAS_IGL_SD = False
    _HAS_SD_FAST = _HAS_SD_WIND = _HAS_SD_PSEUDO = False

# 可选：Numba（用于奇偶回退）
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

# 可选：PyTorch / PyTorch3D（GPU 距离）
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
# FWN“未决带”（越小越少回退；建议 1e-5～1e-4）
_FWN_TAU = float(os.getenv("SDF_FWN_TAU", "1e-5"))
# FWN / AABB 批大小（根据内存调）
_FWN_BATCH = int(os.getenv("SDF_FWN_BATCH", "2000000"))
_AABB_BATCH = int(os.getenv("SDF_AABB_BATCH", "400000"))
# GPU 距离（PyTorch3D）一次送入的点数
_TORCH3D_POINTS_CHUNK = int(os.getenv("SDF_TORCH3D_POINTS_CHUNK", "2000000"))
# 旧后端 kNN 默认
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

def _hash_jitter(y, z) -> float:
    s = f"{float(y):.12e},{float(z):.12e}".encode('utf-8')
    h = hashlib.blake2b(s, digest_size=8).digest()
    val = int.from_bytes(h, 'little')
    return ((val % 1024) / 512.0 - 1.0) * 1e-9


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
    size = (bmax - bmin)
    nx = max(2, int(math.ceil(size[0] / voxel_size)))
    ny = max(2, int(math.ceil(size[1] / voxel_size)))
    nz = max(2, int(math.ceil(size[2] / voxel_size)))
    xs = np.linspace(bmin[0], bmax[0], nx, dtype=np.float64)
    ys = np.linspace(bmin[1], bmax[1], ny, dtype=np.float64)
    zs = np.linspace(bmin[2], bmax[2], nz, dtype=np.float64)
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
    if not _HAS_MPL:
        raise RuntimeError("matplotlib 不可用")
    if not _HAS_SKIMAGE:
        raise RuntimeError("需要 scikit-image：pip install scikit-image")

    try:
        matplotlib.rcParams['font.family'] = 'Times New Roman'
    except Exception:
        pass

    bmin, bmax = bounds
    nx, ny, nz = sdf_grid.shape
    xs = np.linspace(bmin[0], bmax[0], int(nx), dtype=np.float64)
    ys = np.linspace(bmin[1], bmax[1], int(ny), dtype=np.float64)
    zs = np.linspace(bmin[2], bmax[2], int(nz), dtype=np.float64)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()

    spacing = (xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0])
    level = 0.0
    verts, faces, normals_mc, values = measure.marching_cubes(sdf_grid, level=level, spacing=spacing)
    verts_world = verts + np.array(bmin, dtype=np.float64)[None, :]
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
        if transparent_bg:
            fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.05, transparent=True)
        else:
            fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
    else:
        plt.show()

    return out_path

def save_timings_pie(timings: Dict[str, float], out_path: str):
    if not _HAS_MPL:
        raise RuntimeError("matplotlib 不可用")
    labels = [k for k in timings.keys() if k != 'total']
    sizes = [timings[k] for k in labels]
    total = sum(sizes) + 1e-18
    pct = [100.0 * s / total for s in sizes]
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    wedges, texts = ax.pie(sizes, wedgeprops=dict(width=0.35), startangle=140, autopct=None)
    ax.axis('equal')
    legend_labels = [f"{labels[i]}: {pct[i]:.1f}%" for i in range(len(labels))]
    ax.legend(wedges, legend_labels, title="Timings", loc="center left", bbox_to_anchor=(1.0, 0.5))
    _ensure_dir_for(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return out_path


# -------------------- 旧工具（仅回退/兜底会用到） --------------------
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
            j = math.sin(y * 1e6 + z * 1e3) * 1e-9
            y += j; z += j
            hit_cnt = 0
            for t in range(A.shape[0]):
                if not valid_all[t]:
                    continue
                if y < Vmin[t,1] or y >= Vmax[t,1]:
                    continue
                if z < Vmin[t,2] or z >= Vmax[t,2]:
                    continue
                if Vmax[t,0] < x:
                    continue
                tvec_x = x - A[t,0]; tvec_y = y - A[t,1]; tvec_z = z - A[t,2]
                invd   = inv_det_all[t]
                u = (tvec_y * (-E2[t,2]) + tvec_z * E2[t,1]) * invd
                v = (tvec_y * E1[t,2] - tvec_z * E1[t,1]) * invd
                if (u < 0.0) or (v <= 0.0) or (u + v > 1.0):
                    continue
                tt = (
                    E2[t,0]*(tvec_y*E1[t,2] - tvec_z*E1[t,1]) +
                    E2[t,1]*(tvec_z*E1[t,0] - tvec_x*E1[t,2]) +
                    E2[t,2]*(tvec_x*E1[t,1] - tvec_y*E1[t,0])
                ) * invd
                if tt > 1e-12 and (x + tt) <= (bmax_x + 1e-9):
                    hit_cnt += 1
            out[k] = -1.0 if (hit_cnt % 2 == 1) else 1.0
        return out
else:
    def _ray_parity_sign_subset_numba(*a, **k):
        raise RuntimeError("需要 numba 以启用高速奇偶回退；请 pip install numba")


# -------------------- FWN/SD/AABB 主体 --------------------
def _eval_fwn_in_batches(pts: np.ndarray, V: np.ndarray, F: np.ndarray,
                         batch: int) -> np.ndarray:
    """
    分批评估 FWN。若当前 libigl 不含 FWN 函数，抛出，让调用方走 signed_distance 路径。
    """
    if not _HAS_IGL:
        raise RuntimeError("需要 libigl：pip install libigl")
    if not _HAS_IGL_FWN_FUNC:
        raise RuntimeError("libigl 缺少 fast_winding_number_for_meshes")

    Vd = np.ascontiguousarray(V, dtype=np.float64)
    Fi = np.ascontiguousarray(F, dtype=np.int32)
    M  = pts.shape[0]
    W  = np.empty((M,), dtype=np.float64)

    for st in range(0, M, batch):
        ed = min(M, st + batch)
        Qb = np.ascontiguousarray(pts[st:ed], dtype=np.float64)
        W[st:ed] = igl.fast_winding_number_for_meshes(Vd, Fi, Qb)
    return W


def _aabb_squared_distance_in_batches(
    pts: np.ndarray, V: np.ndarray, F: np.ndarray, batch: int, workers: int
) -> np.ndarray:
    """
    libigl.AABB 真最近距离，分批 + 线程池。
    - 正确用法：AABB 需先空构造，再 init(V,F)；且确保 V(float64)/F(int32) 且连续。
    """
    if not _HAS_IGL:
        raise RuntimeError("需要 libigl：pip install libigl")

    # 保证类型与内存布局（libigl 这边最稳的是 float64 / int32 且 C-contiguous）
    Vd = np.ascontiguousarray(V, dtype=np.float64)
    Fi = np.ascontiguousarray(F, dtype=np.int32)
    P  = np.ascontiguousarray(pts, dtype=np.float64)

    # 正确用法：先构造，再 init
    tree = igl.AABB()
    tree.init(Vd, Fi)

    def _one(qslice: np.ndarray):
        sqrD, I, C = tree.squared_distance(Vd, Fi, qslice)
        return np.sqrt(np.maximum(sqrD, 0.0))

    M = P.shape[0]
    chunks = [(st, min(M, st + batch)) for st in range(0, M, batch)]
    out = np.empty((M,), dtype=np.float64)

    from concurrent.futures import ThreadPoolExecutor
    w = _normalize_workers(workers)
    if w == 1 or len(chunks) == 1:
        for st, ed in chunks:
            out[st:ed] = _one(P[st:ed])
        return out

    # 注：部分平台/轮子线程安全性差，若遇到崩溃/异常结果请将 workers=1
    with ThreadPoolExecutor(max_workers=w) as ex:
        futs = [(st, ed, ex.submit(_one, P[st:ed])) for st, ed in chunks]
        for st, ed, fu in futs:
            out[st:ed] = fu.result()
    return out


def _sign_from_winding_with_parity_fallback(
    pts: np.ndarray, V: np.ndarray, F: np.ndarray,
    tau: float
) -> np.ndarray:
    """
    先 FWN 判号；对 |w-0.5|<=tau 的极少数点，用奇偶回退一次。
    若 FWN 函数缺失，抛出异常交由上层使用 signed_distance。
    """
    W = _eval_fwn_in_batches(pts, V, F, batch=_FWN_BATCH)
    sign = np.empty((pts.shape[0],), dtype=np.float64)

    isfinite = np.isfinite(W)
    inside  = np.where(isfinite, W >= (0.5 + tau), False)
    outside = np.where(isfinite, W <= (0.5 - tau), False)
    undec   = ~(inside | outside)

    sign[inside]  = -1.0
    sign[outside] = +1.0

    if np.any(undec):
        idx = np.where(undec)[0]
        Vmin, Vmax, A, B, Ctri = _triangle_bboxes(V, F)
        bmax_x = float(np.max(Vmax[:,0]))
        E1 = B - A; E2 = Ctri - A
        det_all = E1[:,1]*(-E2[:,2]) + E1[:,2]*E2[:,1]
        valid_all = (np.abs(det_all) > 1e-12)
        inv_det_all = np.zeros_like(det_all); inv_det_all[valid_all] = 1.0/det_all[valid_all]
        s_par = _ray_parity_sign_subset_numba(
            pts, idx, Vmin, Vmax, A, E1, E2, det_all, inv_det_all, valid_all, bmax_x
        )
        sign[idx] = s_par

    return sign


def _sdf_with_fwn_aabb(
    pts: np.ndarray, V: np.ndarray, F: np.ndarray,
    workers: int = -1,
    aabb_batch: int = _AABB_BATCH,
    tau: float = _FWN_TAU
) -> Tuple[np.ndarray, np.ndarray]:
    """
    默认路径：
    (A) 若 libigl 暴露了 signed_distance + FAST/WINDING，直接一步得到带符号距离 → 最快最稳。
    (B) 否则：AABB 真最近距离 + FWN 判号 + 0.5 带奇偶回退。
    """
    # ---------- (A) 优先 signed_distance（FAST/WINDING） ----------
    if _HAS_IGL and _HAS_IGL_SD and (_HAS_SD_FAST or _HAS_SD_WIND or _HAS_SD_PSEUDO):
        Vd = np.ascontiguousarray(V, dtype=np.float64)
        Fi = np.ascontiguousarray(F, dtype=np.int32)
        Q  = np.ascontiguousarray(pts, dtype=np.float64)

        # 优先 FAST_WINDING_NUMBER；其次 WINDING_NUMBER；最后 PSEUDONORMAL（保底）
        if _HAS_SD_FAST:
            stype = igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
        elif _HAS_SD_WIND:
            stype = igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER
        else:
            stype = igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL

        try:
            sd, I, C, N = igl.signed_distance(Q, Vd, Fi, stype)
        except TypeError:
            sd, I, C = igl.signed_distance(Q, Vd, Fi, stype)

        d_unsigned = np.abs(sd)                    # 无符号距离
        sign = np.where(sd < 0.0, -1.0, 1.0)       # 内部为 -1
        return d_unsigned, sign

    # ---------- (B) 常规路径：AABB + FWN + 0.5 带回退 ----------
    d_unsigned = _aabb_squared_distance_in_batches(
        pts, V, F, batch=aabb_batch, workers=workers
    )

    # 符号：FWN；若当前 libigl 没有 FWN 函数，则再兜底尝试 signed_distance
    try:
        sign = _sign_from_winding_with_parity_fallback(pts, V, F, tau=tau)
    except RuntimeError as e:
        if "fast_winding_number_for_meshes" in str(e) and _HAS_IGL_SD:
            Vd = np.ascontiguousarray(V, dtype=np.float64)
            Fi = np.ascontiguousarray(F, dtype=np.int32)
            Q  = np.ascontiguousarray(pts, dtype=np.float64)
            # 尽量选 WIND/FAST；没有就 PSEUDONORMAL
            if _HAS_SD_FAST:
                stype = igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
            elif _HAS_SD_WIND:
                stype = igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER
            else:
                stype = igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
            try:
                sd, *_ = igl.signed_distance(Q, Vd, Fi, stype)
                sign = np.where(sd < 0.0, -1.0, 1.0)
            except Exception as ee:
                raise ee
        else:
            raise e

    return d_unsigned, sign


# -------------------- 可选：GPU 距离（PyTorch3D） + SD/FWN 符号 --------------------
def _sdf_with_torch3d_fwn(
    pts: np.ndarray, V: np.ndarray, F: np.ndarray,
    tau: float = _FWN_TAU,
    device: Optional[str] = None,
    points_chunk: int = _TORCH3D_POINTS_CHUNK
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU 距离（PyTorch3D） + libigl 符号（signed_distance 的 FAST/WINDING/PSEUDO）。
    若缺依赖/无 CUDA/不支持 per-point，则抛出异常由上层回退到 fwn_aabb。
    """
    if not (_HAS_TORCH and _HAS_P3D and torch.cuda.is_available()):
        raise RuntimeError("需要 PyTorch(+CUDA) 与 PyTorch3D 方可使用 torch3d_fwn 后端")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    v = torch.from_numpy(np.ascontiguousarray(V, dtype=np.float32)).to(device=device)
    f = torch.from_numpy(np.ascontiguousarray(F, dtype=np.int64)).to(device=device)
    mesh = Meshes(verts=[v], faces=[f])

    M = pts.shape[0]
    d_unsigned = np.empty((M,), dtype=np.float64)

    # 分块将点送入 GPU 计算 per-point 距离
    for st in range(0, M, points_chunk):
        ed = min(M, st + points_chunk)
        p = torch.from_numpy(np.ascontiguousarray(pts[st:ed], dtype=np.float32)).to(device=device)[None, ...]  # [1,P,3]
        try:
            d_per_point = point_mesh_distance(p, mesh, reduction="none")  # [1,P] 每点平方距离
            d_np = d_per_point[0].detach().cpu().numpy().astype(np.float64)
        except TypeError as e:
            raise RuntimeError("当前 PyTorch3D 版本不支持 per-point 距离（reduction='none'）。请升级，或改用 fwn_aabb 后端。") from e
        d_unsigned[st:ed] = np.sqrt(np.maximum(d_np, 0.0))

    # 符号用 libigl.signed_distance（与默认后端一致）
    # 若没有 FAST/WINDING 则退到 PSEUDONORMAL（不推荐，但保底）
    if not (_HAS_IGL and _HAS_IGL_SD):
        raise RuntimeError("需要 libigl.signed_distance 以判号；请安装 libigl。")

    Vd = np.ascontiguousarray(V, dtype=np.float64)
    Fi = np.ascontiguousarray(F, dtype=np.int32)
    Q  = np.ascontiguousarray(pts, dtype=np.float64)
    if _HAS_SD_FAST:
        stype = igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    elif _HAS_SD_WIND:
        stype = igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER
    else:
        stype = igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
    try:
        sd, *_ = igl.signed_distance(Q, Vd, Fi, stype)
        sign = np.where(sd < 0.0, -1.0, 1.0)
    except Exception as e:
        raise e

    return d_unsigned, sign


# -------------------- 旧后端（保留以备参考/对照） --------------------
def _sample_tri_centers_normals(V: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tri0 = V[F[:, 0], :]
    tri1 = V[F[:, 1], :]
    tri2 = V[F[:, 2], :]
    centers = (tri0 + tri1 + tri2) / 3.0
    nvec = np.cross(tri1 - tri0, tri2 - tri0)
    nlen = np.linalg.norm(nvec, axis=1, keepdims=True)
    nlen[nlen == 0] = 1.0
    normals = nvec / nlen
    return centers.astype(np.float64), normals.astype(np.float64)

def _nn_unsigned_and_sign(points: np.ndarray,
                          centers: np.ndarray,
                          normals: np.ndarray,
                          batch: int = 200_000,
                          workers: int = 1,
                          show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if not _HAS_CKDTREE:
        raise RuntimeError("需要 SciPy：pip install scipy")
    kdt = cKDTree(centers)
    M = points.shape[0]
    ranges = list(range(0, M, batch))
    dists = np.empty((M,), dtype=np.float64)
    sign = np.empty((M,), dtype=np.float64)
    _iter = ranges
    if show_progress:
        try:
            from tqdm import tqdm
            _iter = tqdm(ranges, desc="[NN-center]")
        except Exception:
            pass
    for st in _iter:
        ed = min(st+batch, M)
        dd, jj = kdt.query(points[st:ed], k=1, workers=_normalize_workers(workers))
        c = centers[jj]; n = normals[jj]
        dists[st:ed] = np.linalg.norm(points[st:ed] - c, axis=1)
        sign[st:ed] = np.sign(np.sum((points[st:ed] - c) * n, axis=1) + 1e-18)
    return dists, sign


# -------------------- 公开主流程 --------------------
def compute_sdf_grid(vertices: np.ndarray,
                     faces: np.ndarray,
                     padding: float = 0.1,
                     voxel_size: Optional[float] = None,
                     target_resolution: Optional[int] = None,
                     max_resolution: int = 512,
                     show_progress: bool = True,
                     sdf_backend: str = 'fwn_aabb',   # << 默认后端
                     workers: int = -1,
                     robust_sign: bool = True,         # 旧后端用；新后端忽略
                     ambiguous_deg: float = 22.5,      # 旧后端用；新后端忽略
                     ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, Dict[str, float], Tuple[float, float, float]]:
    if not _HAS_IGL:
        raise RuntimeError("需要 libigl（AABB/SD/FWN）以运行默认后端：pip install libigl")

    t0 = time.time()
    bmin, bmax = _compute_bounds(vertices, padding)
    t_grid_setup = time.time()

    if voxel_size is None:
        size = bmax - bmin
        if target_resolution is None:
            target_resolution = max_resolution
        longest = float(np.max(size))
        voxel_size = longest / float(target_resolution)
    xs, ys, zs = _voxel_grid_axes(bmin, bmax, voxel_size)
    voxel_step = (float(xs[1]-xs[0]), float(ys[1]-ys[0]), float(zs[1]-zs[0]))
    t_axes = time.time()

    pts = _grid_points(xs, ys, zs)
    t_pts = time.time()

    # 旧后端需要的采样（新后端无用，但保留变量以兼容返回值）
    centers, normals = _sample_tri_centers_normals(vertices, faces)
    t_nn_sample = time.time()

    # ---------------- 路径选择 ----------------
    backend = sdf_backend
    d_unsigned: np.ndarray
    sign: np.ndarray

    if backend == 'fwn_aabb':
        d_unsigned, sign = _sdf_with_fwn_aabb(
            pts, vertices, faces,
            workers=workers,
            aabb_batch=_AABB_BATCH,
            tau=_FWN_TAU
        )
    elif backend == 'torch3d_fwn':
        d_unsigned, sign = _sdf_with_torch3d_fwn(
            pts, vertices, faces,
            tau=_FWN_TAU,
            device=None,
            points_chunk=_TORCH3D_POINTS_CHUNK
        )
    elif backend == 'nn':
        d_unsigned, sign = _nn_unsigned_and_sign(
            pts, centers, normals,
            workers=_normalize_workers(workers),
            show_progress=show_progress
        )
    else:
        raise ValueError(f"未知后端：{backend}. 可选：fwn_aabb（默认）/ torch3d_fwn / nn")

    t_query = time.time()

    sdf_flat = d_unsigned * sign * (-1.0)  # 内部为负
    sdf_grid = sdf_flat.reshape(len(xs), len(ys), len(zs))
    t_reshape = time.time()

    timings = {
        "grid_setup": t_grid_setup - t0,
        "grid_axes": t_axes - t_grid_setup,
        "grid_points": t_pts - t_axes,
        "nn_sample": t_nn_sample - t_pts,
        "query": t_query - t_nn_sample,
        "reshape": t_reshape - t_query,
        "total": t_reshape - t0
    }

    return sdf_grid, (bmin, bmax), centers, normals, timings, voxel_step


# -------------------- 保存 --------------------
def save_sdf_and_meta(sdf_grid: np.ndarray,
                      bounds: Tuple[np.ndarray, np.ndarray],
                      obj_path: str,
                      voxel_step: Tuple[float, float, float],
                      padding: float,
                      timings: Dict[str, float],
                      out_prefix: str) -> Tuple[str, str, str, str]:
    npy_path = f"{out_prefix}_sdf.npy"
    meta_path = f"{out_prefix}_meta.json"
    png_path = f"{out_prefix}_isosurface.png"
    pie_path = f"{out_prefix}_timings_pie.png"

    _ensure_dir_for(npy_path)
    np.save(npy_path, sdf_grid)

    meta = dict(
        obj=os.path.abspath(obj_path),
        bounds_min=list(map(float, bounds[0])),
        bounds_max=list(map(float, bounds[1])),
        voxel_step=list(map(float, voxel_step)),
        padding=float(padding),
        timings={k: float(v) for k, v in timings.items()}
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


# -------------------- PyVista 交互可视化（可选） --------------------
def pyvista_visualize_isosurface(sdf_grid,
                                 bounds,
                                 show=True,
                                 out_png: Optional[str] = None):
    """
    如果存在 pv.UniformGrid 则走结构网格分支；否则降级为 marching_cubes + PolyData。
    """
    try:
        import pyvista as pv
    except Exception as e:
        raise RuntimeError("需要 pyvista 才能使用该函数") from e

    bmin, bmax = bounds
    nx, ny, nz = sdf_grid.shape
    xs = np.linspace(bmin[0], bmax[0], int(nx), dtype=np.float64)
    ys = np.linspace(bmin[1], bmax[1], int(ny), dtype=np.float64)
    zs = np.linspace(bmin[2], bmax[2], int(nz), dtype=np.float64)
    spacing = (xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0])

    UG = getattr(pv, "UniformGrid", None)
    plotter = pv.Plotter(window_size=[900, 700])
    if UG is not None:
        grid = UG()
        grid.dimensions = np.array(sdf_grid.shape) + 1
        grid.spacing = spacing
        grid.origin = bmin
        grid.cell_data["values"] = sdf_grid.ravel(order="F")
        surf = grid.contour([0.0], scalars="values")
        plotter.add_mesh(surf, color="lightsteelblue", opacity=0.75, smooth_shading=True)
    else:
        from skimage import measure as _measure
        verts, faces, normals_mc, values = _measure.marching_cubes(sdf_grid, level=0.0, spacing=spacing)
        verts_world = verts + np.array(bmin, dtype=np.float64)[None, :]
        import numpy as _np
        faces_pv = _np.hstack([_np.full((faces.shape[0], 1), 3, dtype=_np.int64), faces.astype(_np.int64)]).ravel()
        mesh = pv.PolyData(verts_world, faces_pv)
        plotter.add_mesh(mesh, color="lightsteelblue", opacity=0.75, smooth_shading=True)

    plotter.add_axes()
    plotter.show_bounds(grid='front', location='outer', all_edges=True)
    plotter.view_isometric()

    if out_png:
        _ensure_dir_for(out_png)
        plotter.screenshot(out_png)
    if show:
        plotter.show()
    else:
        plotter.close()


# -------------------- CLI --------------------
def _main():
    ap = argparse.ArgumentParser("sdf_tools_gpu (default fwn_aabb; GPU distance optional via torch3d_fwn)")
    ap.add_argument('--obj', type=str, required=True)
    ap.add_argument('--out', type=str, required=True, help='输出前缀（不含扩展名）')
    ap.add_argument('--padding', type=float, default=0.1)
    ap.add_argument('--voxel_size', type=float, default=None)
    ap.add_argument('--target_resolution', type=int, default=256)
    ap.add_argument('--max_resolution', type=int, default=512)
    ap.add_argument('--workers', type=int, default=-1, help='用于 AABB 距离的线程数；-1=CPU核数；1=单线程')
    ap.add_argument('--backend', type=str, default='fwn_aabb',
                    choices=['fwn_aabb','torch3d_fwn','nn'],
                    help="SDF 后端：fwn_aabb=默认；torch3d_fwn=GPU距离+SD符号；nn=旧版最近中心")
    args = ap.parse_args()

    V, F = parse_obj(args.obj)
    sdf, bounds, surf_pts, surf_nrm, timings, voxel_step = compute_sdf_grid(
        V, F,
        padding=args.padding,
        voxel_size=args.voxel_size,
        target_resolution=args.target_resolution,
        max_resolution=args.max_resolution,
        show_progress=True,
        sdf_backend=args.backend,
        workers=args.workers
    )

    png, pie, npy, meta = save_sdf_and_meta(
        sdf, bounds=bounds,
        obj_path=args.obj,
        voxel_step=voxel_step, padding=args.padding,
        timings=timings, out_prefix=args.out
    )

    print(json.dumps({
        "isosurface_png": png,
        "timings_pie": pie,
        "sdf_npy": npy,
        "meta_json": meta
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    _main()
