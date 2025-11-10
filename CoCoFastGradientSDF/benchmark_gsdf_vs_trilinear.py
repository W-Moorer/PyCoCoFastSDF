# -*- coding: utf-8 -*-
"""
Gradient‑SDF vs 传统 SDF（三线性）对比基准（修正版）
=================================================
- 无 CLI：在文件顶部直接改参数
- 打印：
  1) 稠密网格内存/磁盘开销 vs 稀疏哈希开销（数量、比例、字节数）
  2) 查询效率：Gradient‑SDF（单体素泰勒） vs 传统 SDF（三线性值+解析梯度）吞吐
  3) 查询精度：值差 |d_tri - d_taylor|，梯度夹角误差（°）

前提：已经用 gradient_sdf_tools_cpu_bk1.py 生成 <prefix>_sdf.npy/_grad.npy/_meta.json
可选：先用 gradient_sdf_runtime.build_hash_from_prefix 保存一个 .npz（本脚本也会现建）
"""
from __future__ import annotations

from pathlib import Path
import os
import time
import math
import gc
import numpy as np

# ---------------------- 用户参数（改这里） ----------------------
PREFIX = Path("./gradient_outputs/gear")   # 不含后缀；会自动找 *_sdf.npy/_grad.npy/_meta.json
BLOCK_SIZE = 8                # 哈希块边长（体素）
HASH_DTYPE = "float64"        # "float64"（数值等价）或 "float32"（省内存）
TAU_IN_VOXELS = 6.0           # 窄带厚度（单位：体素），实际用 tau = TAU_IN_VOXELS * min(voxel)
N_SAMPLES = 150000            # 采样点数量（越多统计越稳、也越慢）
TIME_REPEATS = 3              # 重复计时次数取最小值（避免冷启动抖动）
SAVE_HASH_PATH = None         # 例如 Path("out/scene_hash.npz")，None 表示不落盘

# ---------------------- 导入运行时代码 --------------------------
from gradient_sdf_runtime import (
    load_dense_by_prefix,
    HashedGradientSDF,
)

# ---------------------- 工具函数 -------------------------------

def human_bytes(n: int) -> str:
    units = ["B","KiB","MiB","GiB","TiB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x*1024.0:.2f} TiB"


def build_hashed(vol, tau_world: float, block_size: int, dtype: str):
    h = HashedGradientSDF.build_from_dense(vol, tau=tau_world, block_size=block_size, dtype=dtype)
    return h


def hash_payload_nbytes(h: HashedGradientSDF) -> int:
    total = 0
    for blk in h.blocks.values():
        total += blk.locs.nbytes
        total += blk.psi.nbytes
        total += blk.g.nbytes
        total += np.asarray(blk.base, dtype=np.int32).nbytes
    # 不含 Python dict/对象开销，仅统计矩阵 payload
    return total


def gather_dense_disk_sizes(prefix: Path):
    sdf_p = prefix.with_name(prefix.name + "_sdf.npy")
    grad_p = prefix.with_name(prefix.name + "_grad.npy")
    sizes = {}
    for p in [sdf_p, grad_p]:
        sizes[p.name] = os.path.getsize(p) if p.exists() else 0
    return sizes

# ---------------------- 传统 SDF 三线性插值（解析梯度） ---------

def trilinear_value_and_grad(sdf: np.ndarray, spec, P: np.ndarray):
    """在体素中心网格上做三线性插值，返回 (d, grad, mask)。
    - P: (N,3) 世界坐标
    - mask: True 表示 i0..i1 等 8 个角点都在栅格范围内
    解析梯度：对三线性表达式按 x/y/z 求导，并按体素步长缩放。
    """
    P = np.asarray(P, dtype=np.float64)
    voxel = spec.voxel
    # 连续索引（以体素中心为整数+0.5 的网格）
    U = (P - spec.bmin) / voxel - 0.5  # (N,3)
    i0 = np.floor(U[:,0]).astype(np.int64)
    j0 = np.floor(U[:,1]).astype(np.int64)
    k0 = np.floor(U[:,2]).astype(np.int64)
    tx = U[:,0] - i0
    ty = U[:,1] - j0
    tz = U[:,2] - k0

    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1

    nx, ny, nz = sdf.shape
    mask = (i0>=0)&(i1<nx)&(j0>=0)&(j1<ny)&(k0>=0)&(k1<nz)
    if not np.any(mask):
        return (np.full(len(P), np.nan), np.full((len(P),3), np.nan), mask)

    # 只在有效点上取样，避免越界
    idx = np.where(mask)[0]
    i0_, j0_, k0_ = i0[idx], j0[idx], k0[idx]
    i1_, j1_, k1_ = i1[idx], j1[idx], k1[idx]
    tx_, ty_, tz_ = tx[idx], ty[idx], tz[idx]

    c000 = sdf[i0_, j0_, k0_]
    c100 = sdf[i1_, j0_, k0_]
    c010 = sdf[i0_, j1_, k0_]
    c110 = sdf[i1_, j1_, k0_]
    c001 = sdf[i0_, j0_, k1_]
    c101 = sdf[i1_, j0_, k1_]
    c011 = sdf[i0_, j1_, k1_]
    c111 = sdf[i1_, j1_, k1_]

    wx0 = 1.0 - tx_
    wy0 = 1.0 - ty_
    wz0 = 1.0 - tz_
    wx1 = tx_
    wy1 = ty_
    wz1 = tz_

    # 值：标准三线性公式
    d = (
        c000*wx0*wy0*wz0 + c100*wx1*wy0*wz0 + c010*wx0*wy1*wz0 + c110*wx1*wy1*wz0 +
        c001*wx0*wy0*wz1 + c101*wx1*wy0*wz1 + c011*wx0*wy1*wz1 + c111*wx1*wy1*wz1
    )

    # 解析梯度：对三线性权重求偏导，再除以体素步长
    ddx = (
        (wy0*wz0)*(c100 - c000) + (wy1*wz0)*(c110 - c010) + (wy0*wz1)*(c101 - c001) + (wy1*wz1)*(c111 - c011)
    ) / voxel[0]
    ddy = (
        (wx0*wz0)*(c010 - c000) + (wx1*wz0)*(c110 - c100) + (wx0*wz1)*(c011 - c001) + (wx1*wz1)*(c111 - c101)
    ) / voxel[1]
    ddz = (
        (wx0*wy0)*(c001 - c000) + (wx1*wy0)*(c101 - c100) + (wx0*wy1)*(c011 - c010) + (wx1*wy1)*(c111 - c110)
    ) / voxel[2]

    d_full = np.full(len(P), np.nan)
    g_full = np.full((len(P),3), np.nan)
    d_full[idx] = d
    g_full[idx,0] = ddx
    g_full[idx,1] = ddy
    g_full[idx,2] = ddz
    return d_full, g_full, mask

# ---------------------- 采样点（窄带内的单元格） ----------------

def sample_points_in_cells(vol, n_samples: int, tau_world: float, rng: np.random.Generator):
    sdf = vol.sdf
    nx, ny, nz = sdf.shape
    # 候选单元：以 i0 为左下近角的 cell（需要 i0..i1 都在范围）且 |ψ(i0)|<=tau
    mask = (np.abs(sdf) <= tau_world)
    valid = mask[:nx-1, :ny-1, :nz-1]  # 保证 i1/j1/k1 存在
    cand = np.argwhere(valid)
    if len(cand) == 0:
        # 退化：全体内点里采样
        cand = np.argwhere(np.ones((nx-1,ny-1,nz-1), dtype=bool))
    sel = cand[rng.integers(0, len(cand), size=min(n_samples, len(cand)))]
    # 在每个 cell 内随机 t∈[0,1)^3，然后转世界坐标：p = bmin + voxel*(i0+t + 0.5)
    t = rng.random((len(sel), 3))
    ijk0 = sel.astype(np.int64)
    P = vol.spec.bmin + vol.spec.voxel * (ijk0 + t + 0.5)
    return P

# ---------------------- 主流程 ---------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    vol = load_dense_by_prefix(PREFIX)
    voxel = vol.spec.voxel
    tau_world = float(np.min(voxel) * TAU_IN_VOXELS)

    # ----- 统计稠密内存/磁盘开销 -----
    dense_mem = vol.sdf.nbytes + vol.grad.nbytes
    dense_sizes = gather_dense_disk_sizes(PREFIX)

    # ----- 构建稀疏哈希并统计开销 -----
    h = build_hashed(vol, tau_world=tau_world, block_size=BLOCK_SIZE, dtype=HASH_DTYPE)
    hashed_payload = hash_payload_nbytes(h)
    hashed_voxels = int(sum(len(blk.psi) for blk in h.blocks.values()))
    total_voxels = int(np.prod(vol.spec.shape))

    hash_disk = None
    if SAVE_HASH_PATH is not None:
        h.save(Path(SAVE_HASH_PATH))
        hash_disk = os.path.getsize(SAVE_HASH_PATH)

    # ----- 准备采样点（窄带内的单元格） -----
    P = sample_points_in_cells(vol, N_SAMPLES, tau_world, rng)

    # ----- 计时工具（支持 kwargs） -----
    def bench(fn, *args, repeats=TIME_REPEATS, **kwargs):
        times = []
        outs = None
        for _ in range(repeats):
            gc.collect()
            t0 = time.perf_counter()
            outs = fn(*args, **kwargs)
            dt = time.perf_counter() - t0
            times.append(dt)
        return min(times), outs

    # ----- 基准 1：Gradient‑SDF（稠密单体素泰勒） -----
    t_taylor, (d_t, g_t, _, _, mask_t) = bench(vol.taylor_query_batch, P)

    # ----- 基准 2：传统 SDF（三线性 + 解析梯度） -----
    t_tri, (d_tri, g_tri, mask_tri) = bench(trilinear_value_and_grad, vol.sdf, vol.spec, P)

    # -----（可选）基准 3：稀疏哈希（命中窄带，不开回退） -----
    t_hash, (d_h, g_h, _, _, hit_h) = bench(h.taylor_query_batch, P, allow_fallback=False)

    # ----- 统计精度：仅对两者都有效的点 -----
    valid = mask_t & mask_tri
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        raise RuntimeError("没有有效的采样点，请调大 TAU_IN_VOXELS 或降低 N_SAMPLES。")

    dv = np.abs(d_tri[valid] - d_t[valid])
    def deg_angle(a, b):
        a = a / np.maximum(1e-12, np.linalg.norm(a, axis=1, keepdims=True))
        b = b / np.maximum(1e-12, np.linalg.norm(b, axis=1, keepdims=True))
        cos = np.sum(a*b, axis=1).clip(-1.0, 1.0)
        return np.degrees(np.arccos(cos))

    ang = deg_angle(g_tri[valid], g_t[valid])

    def stats(x):
        x = np.asarray(x)
        return dict(mean=float(np.mean(x)), median=float(np.median(x)), p95=float(np.percentile(x,95)), max=float(np.max(x)))

    dv_s = stats(dv)
    ang_s = stats(ang)

    # ----- 打印报告 -----
    print("\n================ 资源开销 ================")
    print(f"Dense memory (runtime arrays): {human_bytes(dense_mem)}")
    for name, sz in dense_sizes.items():
        print(f"Dense disk {name:>14}: {human_bytes(sz)}")
    print(f"Hashed payload (ψ,ĝ,locs,bases): {human_bytes(hashed_payload)}  (dtype={HASH_DTYPE})")
    if hash_disk is not None:
        print(f"Hashed disk file (.npz):         {human_bytes(hash_disk)}")
    print(f"Hashed voxels: {hashed_voxels} / {total_voxels}  ({hashed_voxels/total_voxels*100:.4f}% of dense)")

    print("\n================ 查询效率 ================")
    qps_taylor = len(P)/t_taylor
    qps_tri = len(P)/t_tri
    qps_hash = len(P)/t_hash
    print(f"Taylor (dense)  : {t_taylor:.4f}s  -> {qps_taylor:,.0f} q/s  (1.00x)")
    print(f"Trilinear (ψ)   : {t_tri:.4f}s  -> {qps_tri:,.0f} q/s  ({qps_tri/qps_taylor:.2f}x vs Taylor)")
    print(f"Taylor (hashed) : {t_hash:.4f}s  -> {qps_hash:,.0f} q/s  ({qps_hash/qps_taylor:.2f}x vs Taylor)")
    if isinstance(hit_h, np.ndarray):
        print(f"Hashed hit ratio: {np.count_nonzero(hit_h)}/{len(P)} = {np.count_nonzero(hit_h)/len(P)*100:.2f}%")

    print("\n================ 查询精度 ================")
    print(f"Value |d_tri - d_taylor|  (on {n_valid} pts): mean={dv_s['mean']:.3e}, median={dv_s['median']:.3e}, p95={dv_s['p95']:.3e}, max={dv_s['max']:.3e}")
    print(f"Grad  angle(tri, taylor)° (on {n_valid} pts): mean={ang_s['mean']:.3f}, median={ang_s['median']:.3f}, p95={ang_s['p95']:.3f}, max={ang_s['max']:.3f}")
    print("\n说明：")
    print("- Taylor：单体素一阶泰勒（论文的 Gradient‑SDF 查询）")
    print("- Trilinear：传统 SDF 的三线性插值值 + 解析梯度；角度越小代表方向越贴近存储的单位外法向")
    print("- Hashed：命中窄带时的哈希查询；Hit ratio 提示采样点有多少落在窄带内。")
