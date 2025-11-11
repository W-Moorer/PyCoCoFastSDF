# -*- coding: utf-8 -*-
"""
Gradient SDF — 单文件一体化实现（结构化重组版）
================================================
目标：在**不改变任何计算代码逻辑**与对外 API 行为的前提下，使 runtime 与 tools 的组合更有机、更易维护。

主要整理：
- 明确模块分层：基础工具 / 稠密容器 / 稀疏哈希 / 工具集（OBJ→SDF）/ 统一封装。
- 移除 `HashedGradientSDF.taylor_query_batch` 内的大段重复与死代码；三种后端实现（`index_map` / `block_ptr` / `block_atlas`）保持原有计算路径与公式完全一致。
- 统一 dtype、路径与异常信息；保持旧别名与兼容函数。
- I/O（npz 稀疏格式）与 CLI 参数、返回值均未改变；`GradientSDF` 封装保持原始签名与行为。

说明：本文件仅做架构整理与代码清扫，不改变任何数值计算；如需对照，请将三种查询后端的向量化公式与旧版逐行比对。
"""

from __future__ import annotations

# =============================================================================
# 依赖与类型
# =============================================================================
import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

Array = np.ndarray

# 可选依赖（tools 用）
try:
    import igl  # libigl
    _HAS_IGL = True
    _HAS_SD = hasattr(igl, "signed_distance")
    _HAS_SD_FAST = hasattr(igl, "SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER")
    _HAS_SD_WIND = hasattr(igl, "SIGNED_DISTANCE_TYPE_WINDING_NUMBER")
    _HAS_SD_PSEUDO = hasattr(igl, "SIGNED_DISTANCE_TYPE_PSEUDONORMAL")
    _HAS_FWN = hasattr(igl, "fast_winding_number_for_meshes")
except Exception:
    _HAS_IGL = False
    _HAS_SD = False
    _HAS_SD_FAST = _HAS_SD_WIND = _HAS_SD_PSEUDO = False
    _HAS_FWN = False

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

try:
    from skimage import measure as _measure
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


# =============================================================================
# 基础工具
# =============================================================================

_DEF_EPS = 1e-12
_AABB_BATCH = 200_000


def _load_meta(meta_path: Path) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    required = ["bmin", "bmax", "shape", "voxel_step"]
    for k in required:
        if k not in meta:
            raise KeyError(f"meta.json 缺少键: {k}")
    return meta


def _round_to_int(a: Array) -> Array:
    return np.rint(a).astype(np.int64)


def _in_bounds(ijk: Array, shape: Tuple[int, int, int]) -> Array:
    i, j, k = ijk[..., 0], ijk[..., 1], ijk[..., 2]
    nx, ny, nz = shape
    return (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)


@dataclass
class GridSpec:
    bmin: Array
    bmax: Array
    voxel: Array
    shape: Tuple[int, int, int]

    @property
    def nx(self) -> int:
        return int(self.shape[0])

    @property
    def ny(self) -> int:
        return int(self.shape[1])

    @property
    def nz(self) -> int:
        return int(self.shape[2])

    def center_from_index(self, ijk: Array) -> Array:
        return self.bmin + self.voxel * (ijk.astype(np.float64) + 0.5)

    def index_from_world(self, p: Array) -> Array:
        ijk_f = (p - self.bmin) / self.voxel - 0.5
        return _round_to_int(ijk_f)


# =============================================================================
# 稠密体素：读取/查询（批量向量化）
# =============================================================================

class GradientSDFVolume:
    def __init__(self, sdf_npy: Path, grad_npy: Optional[Path] = None, meta_json: Optional[Path] = None):
        sdf_npy = Path(sdf_npy)
        if meta_json is None:
            meta_json = sdf_npy.with_name(sdf_npy.name.replace("_sdf.npy", "_meta.json"))
        if grad_npy is None:
            grad_npy = sdf_npy.with_name(sdf_npy.name.replace("_sdf.npy", "_grad.npy"))
        if not sdf_npy.exists() or not grad_npy.exists() or not meta_json.exists():
            raise FileNotFoundError("缺少 sdf/grad/meta 之一")

        self.sdf: Array = np.load(sdf_npy).astype(np.float64, copy=False)
        self.grad: Array = np.load(grad_npy).astype(np.float64, copy=False)
        meta = _load_meta(meta_json)
        shape = tuple(int(x) for x in meta["shape"])  # (nx,ny,nz)
        if tuple(self.sdf.shape) != shape or tuple(self.grad.shape[:3]) != shape or self.grad.shape[-1] != 3:
            raise ValueError("SDF/Grad 形状与 meta 不一致或梯度最后一维≠3")
        self.spec = GridSpec(
            bmin=np.asarray(meta["bmin"], dtype=np.float64),
            bmax=np.asarray(meta["bmax"], dtype=np.float64),
            voxel=np.asarray(meta["voxel_step"], dtype=np.float64),
            shape=shape,
        )

    def taylor_query(self, p: Array) -> Tuple[Array, Array, Array, Array]:
        p = np.asarray(p, dtype=np.float64).reshape(3)
        ijk = self.spec.index_from_world(p)
        if not bool(_in_bounds(ijk, self.spec.shape)):
            raise IndexError("查询点超出栅格范围")
        i, j, k = (int(ijk[0]), int(ijk[1]), int(ijk[2]))
        psi = float(self.sdf[i, j, k])
        g = self.grad[i, j, k].astype(np.float64)
        vj = self.spec.center_from_index(ijk)
        d = psi + float(np.dot(p - vj, g))
        return d, g, vj, ijk

    def taylor_query_batch(self, P: Array) -> Tuple[Array, Array, Array, Array, Array]:
        P = np.asarray(P, dtype=np.float64)
        ijk = self.spec.index_from_world(P)
        inb = _in_bounds(ijk, self.spec.shape)
        vj = self.spec.center_from_index(ijk)

        d = np.full((len(P),), np.nan, dtype=np.float64)
        g = np.full((len(P), 3), np.nan, dtype=np.float64)

        if np.any(inb):
            I = ijk[inb, 0].astype(np.int64)
            J = ijk[inb, 1].astype(np.int64)
            K = ijk[inb, 2].astype(np.int64)
            psi = self.sdf[I, J, K]
            gg = self.grad[I, J, K]
            diff = P[inb] - vj[inb]
            d_valid = psi + np.einsum('ij,ij->i', diff, gg)
            d[inb] = d_valid
            g[inb] = gg
        return d, g, vj, ijk, inb

    def nearest_surface_point_from_index(self, ijk: Array) -> Array:
        ijk = np.asarray(ijk, dtype=np.int64)
        vj = self.spec.center_from_index(ijk)
        if ijk.ndim == 1:
            i, j, k = (int(ijk[0]), int(ijk[1]), int(ijk[2]))
            return vj - self.sdf[i, j, k] * self.grad[i, j, k]
        I = ijk[:, 0].astype(np.int64)
        J = ijk[:, 1].astype(np.int64)
        K = ijk[:, 2].astype(np.int64)
        return vj - self.sdf[I, J, K, None] * self.grad[I, J, K]


# =============================================================================
# 稀疏哈希体素（块哈希 + 向量化批量查询）
# =============================================================================

@dataclass
class _Block:
    base: Tuple[int, int, int]  # 绝对体素基坐标
    locs: Array                  # (M,3) uint8
    psi: Array                   # (M,)
    g: Array                     # (M,3)
    ptr: Optional[Array] = None  # 小稠密指针表（懒构建）：shape=(B,B,B)，值为本块局部索引，-1 表示该处无体素

    def ensure_ptr(self, B: int):
        if self.ptr is not None:
            return
        ptr = np.full((B, B, B), -1, dtype=np.int32)
        li = self.locs[:, 0].astype(np.intp)
        lj = self.locs[:, 1].astype(np.intp)
        lk = self.locs[:, 2].astype(np.intp)
        ptr[li, lj, lk] = np.arange(self.locs.shape[0], dtype=np.int32)
        self.ptr = ptr


class HashedGradientSDF:
    def __init__(self, spec: GridSpec, blocks: Dict[Tuple[int, int, int], _Block], block_size: int = 8):
        self.spec = spec
        self.blocks = blocks
        self.block_size = int(block_size)
        self._dense: Optional[GradientSDFVolume] = None
        # 预计算块网格尺寸（用于唯一编码）
        self._nbx = (self.spec.nx + self.block_size - 1) // self.block_size
        self._nby = (self.spec.ny + self.block_size - 1) // self.block_size
        # 全局拼接后的索引（懒构建）
        self._codes_all: Optional[np.ndarray] = None   # int64，绝对线性编码 i + nx*(j + ny*k)
        self._psi_all: Optional[np.ndarray] = None     # 与 codes_all 对齐
        self._g_all: Optional[np.ndarray] = None       # 与 codes_all 对齐 (N,3)
        # 全局 O(1) 反向映射：linear_index -> packed index（-1 表示缺失），懒构建
        self._index_map: Optional[np.ndarray] = None   # int32, 长度 nx*ny*nz，-1 表示 miss
        # 查询后端：'index_map'（默认）/'block_ptr'/'block_atlas'
        self.query_backend: str = 'index_map'
        # ---- Block Atlas （跨块全向量化 O(1) 指针表） ----
        self._atlas_stride: int = self.block_size ** 3
        self._atlas_ptr_flat: Optional[np.ndarray] = None   # int32, shape=(num_blocks*B^3,)
        self._block_row_map: Optional[np.ndarray] = None    # int32, length = nbx*nby*nz, -1=absent
        self._packed_psi: Optional[np.ndarray] = None       # packed over all blocks
        self._packed_g: Optional[np.ndarray] = None

    # ---- 构建 ----
    @staticmethod
    def build_from_dense(vol: GradientSDFVolume, tau: float = 10.0, block_size: int = 8, dtype: str = "float32") -> "HashedGradientSDF":
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
        sdf = vol.sdf
        grad = vol.grad
        spec = vol.spec
        mask = np.abs(sdf) <= float(tau)
        idxs = np.argwhere(mask)
        if len(idxs) == 0:
            raise ValueError("阈值过小，窄带为空。")
        B = int(block_size)
        buckets: Dict[Tuple[int, int, int], List[Tuple[Tuple[int, int, int], float, Array]]] = {}
        for (i, j, k) in idxs:
            key = (int(i)//B, int(j)//B, int(k)//B)
            if key not in buckets:
                buckets[key] = []
            psi = float(sdf[i, j, k])
            gvec = grad[i, j, k]
            buckets[key].append(((int(i), int(j), int(k)), psi, gvec))
        blk_objs: Dict[Tuple[int, int, int], _Block] = {}
        for key, triplets in buckets.items():
            base = (key[0]*B, key[1]*B, key[2]*B)
            M = len(triplets)
            locs = np.empty((M, 3), dtype=np.uint8)
            psi_arr = np.empty((M,), dtype=np.float32 if dtype == "float32" else np.float64)
            g_arr = np.empty((M, 3), dtype=np.float32 if dtype == "float32" else np.float64)
            for n, (ijk, psi, gvec) in enumerate(triplets):
                locs[n] = np.array([ijk[0]-base[0], ijk[1]-base[1], ijk[2]-base[2]], dtype=np.uint8)
                psi_arr[n] = psi
                g_arr[n] = gvec
            blk_objs[key] = _Block(base=base, locs=locs, psi=psi_arr, g=g_arr)
        return HashedGradientSDF(spec=spec, blocks=blk_objs, block_size=B)

    def attach_dense_fallback(self, vol: GradientSDFVolume):
        self._dense = vol

    def set_query_backend(self, mode: str = 'index_map'):
        if mode not in ('index_map', 'block_ptr', 'block_atlas'):
            raise ValueError("mode must be 'index_map', 'block_ptr', or 'block_atlas'")
        self.query_backend = mode

    def warmup(self, *, index_map: bool = True, ptr_table: bool = False, atlas: bool = False):
        if index_map:
            self._ensure_index_map()
        if ptr_table:
            B = self.block_size
            for blk in self.blocks.values():
                blk.ensure_ptr(B)
        if atlas:
            self._ensure_block_atlas()

    # ---- 懒构建全局索引：把所有块拼成一个按绝对线性编码排序的大表 ----
    def _ensure_global_index(self):
        if self._codes_all is not None:
            return
        nx, ny, nz = self.spec.shape
        codes_list = []
        psi_list = []
        g_list = []
        for blk in self.blocks.values():
            locs = blk.locs.astype(np.int64)
            i = blk.base[0] + locs[:, 0]
            j = blk.base[1] + locs[:, 1]
            k = blk.base[2] + locs[:, 2]
            code = i + nx * (j + ny * k)  # int64
            codes_list.append(code.astype(np.int64, copy=False))
            psi_list.append(blk.psi)
            g_list.append(blk.g)
        if len(codes_list) == 0:
            self._codes_all = np.empty((0,), dtype=np.int64)
            self._psi_all = np.empty((0,), dtype=np.float64)
            self._g_all = np.empty((0, 3), dtype=np.float64)
            return
        codes = np.concatenate(codes_list).astype(np.int64, copy=False)
        psi = np.concatenate(psi_list)
        g = np.concatenate(g_list, axis=0)
        ord_ = np.argsort(codes, kind='mergesort')
        self._codes_all = codes[ord_]
        self._psi_all = psi[ord_]
        self._g_all = g[ord_]

    # ---- 懒构建 O(1) 反向映射 ----
    def _ensure_index_map(self):
        if self._index_map is not None:
            return
        self._ensure_global_index()
        nx, ny, nz = self.spec.shape
        total = int(nx*ny*nz)
        index_map = np.full((total,), -1, dtype=np.int32)
        if self._codes_all.size:
            index_map[self._codes_all] = np.arange(self._codes_all.size, dtype=np.int32)
        self._index_map = index_map

    # ---- 懒构建 Block Atlas：跨块全向量化 O(1) 查询 ----
    def _ensure_block_atlas(self):
        if self._atlas_ptr_flat is not None:
            return
        B = self.block_size
        nx, ny, nz = self.spec.shape
        nbx, nby = self._nbx, self._nby
        # 为每个实际存在的块分配一行（长度 B^3），存 packed 全局索引；其余置 -1
        num_blocks = len(self.blocks)
        stride = B * B * B
        atlas_ptr = np.full((num_blocks, stride), -1, dtype=np.int32)
        # block_lin -> row 的映射表（稀疏小数组）
        block_row_map = np.full((nbx * nby * ((nz + B - 1)//B),), -1, dtype=np.int32)

        packed_psi = []
        packed_g = []
        row = 0
        for (kx, ky, kz), blk in self.blocks.items():
            block_row_map[kx + nbx * (ky + nby * kz)] = row
            base_off = sum(arr.shape[0] for arr in packed_psi)
            li = blk.locs[:, 0].astype(np.intp)
            lj = blk.locs[:, 1].astype(np.intp)
            lk = blk.locs[:, 2].astype(np.intp)
            local_code = (li + B * (lj + B * lk)).astype(np.intp)
            atlas_ptr[row, local_code] = base_off + np.arange(blk.locs.shape[0], dtype=np.int32)
            packed_psi.append(blk.psi)
            packed_g.append(blk.g)
            row += 1
        self._atlas_ptr_flat = atlas_ptr.ravel()
        self._atlas_stride = stride
        self._block_row_map = block_row_map
        self._packed_psi = np.concatenate(packed_psi) if len(packed_psi) else np.empty((0,), dtype=np.float64)
        self._packed_g = np.concatenate(packed_g, axis=0) if len(packed_g) else np.empty((0,3), dtype=np.float64)

    # ---- 批量向量化哈希查询（保持原有数学与分支逻辑） ----
    def taylor_query_batch(self, P: Array, allow_fallback: bool = False) -> Tuple[Array, Array, Array, Array, Array]:
        P = np.asarray(P, dtype=np.float64)
        ijk = self.spec.index_from_world(P)
        vj = self.spec.center_from_index(ijk)
        inb = _in_bounds(ijk, self.spec.shape)
        d = np.full((len(P),), np.nan, dtype=np.float64)
        g = np.full((len(P), 3), np.nan, dtype=np.float64)
        hit = np.zeros((len(P),), dtype=bool)
        if not np.any(inb):
            return d, g, vj, ijk, hit

        if self.query_backend == 'index_map':
            # ---- 全向量化 O(1) 路径：index_map ----
            self._ensure_index_map()
            index_map = self._index_map  # type: ignore
            psi_all = self._psi_all      # type: ignore
            g_all = self._g_all          # type: ignore
            nx, ny, nz = self.spec.shape

            I = ijk[inb, 0].astype(np.int64)
            J = ijk[inb, 1].astype(np.int64)
            K = ijk[inb, 2].astype(np.int64)
            code_q = I + nx * (J + ny * K)
            idx = index_map[code_q]
            ok = idx >= 0
            if np.any(ok):
                out_idx = np.where(inb)[0][ok]
                idx2 = idx[ok].astype(np.int64)
                psi = psi_all[idx2].astype(np.float64)
                gg = g_all[idx2].astype(np.float64)
                diff = P[out_idx] - vj[out_idx]
                d[out_idx] = psi + np.einsum('ij,ij->i', diff, gg)
                g[out_idx] = gg
                hit[out_idx] = True

        elif self.query_backend == 'block_atlas':
            # ---- 跨块全向量化 O(1) 路径：Block Atlas ----
            self._ensure_block_atlas()
            nx, ny, nz = self.spec.shape
            B = self.block_size
            nbx, nby = self._nbx, self._nby
            stride = self._atlas_stride
            atlas = self._atlas_ptr_flat      # type: ignore
            row_map = self._block_row_map     # type: ignore
            psi_all = self._packed_psi        # type: ignore
            g_all = self._packed_g            # type: ignore

            I = ijk[inb, 0].astype(np.int64)
            J = ijk[inb, 1].astype(np.int64)
            K = ijk[inb, 2].astype(np.int64)
            kx = I // B
            ky = J // B
            kz = K // B
            row = row_map[kx + nbx * (ky + nby * kz)]
            ok_row = row >= 0
            if np.any(ok_row):
                idx_pts = np.where(inb)[0][ok_row]
                row = row[ok_row].astype(np.int64)
                li = (I[ok_row] - (kx[ok_row] * B)).astype(np.int64)
                lj = (J[ok_row] - (ky[ok_row] * B)).astype(np.int64)
                lk = (K[ok_row] - (kz[ok_row] * B)).astype(np.int64)
                ok_loc = (li >= 0) & (li < B) & (lj >= 0) & (lj < B) & (lk >= 0) & (lk < B)
                if np.any(ok_loc):
                    idx_pts2 = idx_pts[ok_loc]
                    row2 = row[ok_loc]
                    li = li[ok_loc]
                    lj = lj[ok_loc]
                    lk = lk[ok_loc]
                    local_code = li + B * (lj + B * lk)
                    atlas_idx = row2 * stride + local_code
                    ptr = atlas[atlas_idx]
                    ok_ptr = ptr >= 0
                    if np.any(ok_ptr):
                        idx_fin = idx_pts2[ok_ptr]
                        ptr2 = ptr[ok_ptr].astype(np.int64)
                        psi = psi_all[ptr2].astype(np.float64)
                        gg = g_all[ptr2].astype(np.float64)
                        diff = P[idx_fin] - vj[idx_fin]
                        d[idx_fin] = psi + np.einsum('ij,ij->i', diff, gg)
                        g[idx_fin] = gg
                        hit[idx_fin] = True

        else:  # 'block_ptr'
            I = ijk[inb, 0].astype(np.int64)
            J = ijk[inb, 1].astype(np.int64)
            K = ijk[inb, 2].astype(np.int64)
            B = self.block_size
            out_idx = np.where(inb)[0]
            kx = I // B
            ky = J // B
            kz = K // B
            key_lin = (kx + self._nbx * (ky + self._nby * kz)).astype(np.int64)
            uniq, inv = np.unique(key_lin, return_inverse=True)

            for u, code in enumerate(uniq):
                kz_u = code // (self._nbx * self._nby)
                rem = code - kz_u * (self._nbx * self._nby)
                ky_u = rem // self._nbx
                kx_u = rem - ky_u * self._nbx
                key = (int(kx_u), int(ky_u), int(kz_u))
                blk = self.blocks.get(key)
                if blk is None:
                    continue
                blk.ensure_ptr(B)
                sel = (inv == u)
                idx_pts = out_idx[sel]
                li = I[sel] - blk.base[0]
                lj = J[sel] - blk.base[1]
                lk = K[sel] - blk.base[2]
                ok = (li >= 0) & (li < B) & (lj >= 0) & (lj < B) & (lk >= 0) & (lk < B)
                if not np.any(ok):
                    continue
                li = li[ok].astype(np.intp)
                lj = lj[ok].astype(np.intp)
                lk = lk[ok].astype(np.intp)
                idx_ok = idx_pts[ok]
                ptr = blk.ptr  # type: ignore
                local_idx = ptr[li, lj, lk]
                ok2 = local_idx >= 0
                if not np.any(ok2):
                    continue
                local_idx = local_idx[ok2].astype(np.int64)
                idx_fin = idx_ok[ok2]
                psi = blk.psi[local_idx].astype(np.float64)
                gg = blk.g[local_idx].astype(np.float64)
                diff = P[idx_fin] - vj[idx_fin]
                d[idx_fin] = psi + np.einsum('ij,ij->i', diff, gg)
                g[idx_fin] = gg
                hit[idx_fin] = True

        if allow_fallback and (self._dense is not None):
            miss = inb & (~hit)
            if np.any(miss):
                d2, g2, _, _, _ = self._dense.taylor_query_batch(P[miss])
                d[miss] = d2
                g[miss] = g2
                hit[miss] = np.isfinite(d2)
        return d, g, vj, ijk, hit


# =============================================================================
# 便捷加载器（runtime 层 I/O 保持不变）
# =============================================================================

def load_dense_by_prefix(prefix: Path) -> GradientSDFVolume:
    prefix = Path(prefix)
    return GradientSDFVolume(
        sdf_npy=prefix.with_name(prefix.name + "_sdf.npy"),
        grad_npy=prefix.with_name(prefix.name + "_grad.npy"),
        meta_json=prefix.with_name(prefix.name + "_meta.json"),
    )


def build_hash_from_prefix(prefix: Path, tau: float = 10.0, block_size: int = 8, dtype: str = "float32") -> 'HashedGradientSDF':
    vol = load_dense_by_prefix(prefix)
    return HashedGradientSDF.build_from_dense(vol, tau=float(tau), block_size=int(block_size), dtype=dtype)


# =============================================================================
# tools（OBJ → SDF/grad 网格），保持既有公式与签名
# =============================================================================

# 模块级状态：保存时使用
_GRAD_GRID: Optional[np.ndarray] = None  # (nx,ny,nz,3)


def _ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def parse_obj(path: str) -> Tuple[np.ndarray, np.ndarray]:
    vs, fs = [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            sp = line.strip().split()
            if not sp:
                continue
            if sp[0] == 'v' and len(sp) >= 4:
                vs.append([float(sp[1]), float(sp[2]), float(sp[3])])
            elif sp[0] == 'f' and len(sp) >= 4:
                tri = []
                for t in sp[1:4]:
                    tri.append(int(t.split('/')[0]) - 1)
                fs.append(tri)
    V = np.asarray(vs, dtype=np.float64)
    F = np.asarray(fs, dtype=np.int32)
    return V, F


def _compute_bounds(V: np.ndarray, padding: float) -> Tuple[np.ndarray, np.ndarray]:
    vmin = V.min(axis=0)
    vmax = V.max(axis=0)
    diag = float(np.linalg.norm(vmax - vmin))
    pad = padding * diag
    return (vmin - pad).astype(np.float64), (vmax + pad).astype(np.float64)


def _voxel_axes(bmin: np.ndarray, bmax: np.ndarray,
                voxel_size: Optional[float],
                target_resolution: Optional[int],
                max_resolution: int):
    size = (bmax - bmin).astype(np.float64)
    if voxel_size is None:
        if target_resolution is None:
            target_resolution = min(192, max_resolution)
        longest = float(size.max())
        voxel = longest / float(target_resolution)
    else:
        voxel = float(voxel_size)

    nx, ny, nz = np.maximum(1, np.ceil(size / voxel).astype(int))
    xs = np.linspace(bmin[0] + 0.5*voxel, bmin[0] + (nx-0.5)*voxel, nx, dtype=np.float64)
    ys = np.linspace(bmin[1] + 0.5*voxel, bmin[1] + (ny-0.5)*voxel, ny, dtype=np.float64)
    zs = np.linspace(bmin[2] + 0.5*voxel, bmin[2] + (nz-0.5)*voxel, nz, dtype=np.float64)
    return xs, ys, zs, np.array([voxel, voxel, voxel], dtype=np.float64)


def _grid_points(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)


def _aabb_dist_closest_idx(pts: np.ndarray, V: np.ndarray, F: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not _HAS_IGL:
        raise RuntimeError("需要 libigl（AABB）")
    Vd = np.ascontiguousarray(V, dtype=np.float64)
    Fi = np.ascontiguousarray(F, dtype=np.int32)
    P  = np.ascontiguousarray(pts, dtype=np.float64)
    tree = igl.AABB()
    tree.init(Vd, Fi)

    M = P.shape[0]
    d = np.empty((M,), dtype=np.float64)
    C = np.empty((M, 3), dtype=np.float64)
    I = np.empty((M,), dtype=np.int32)

    for st in range(0, M, _AABB_BATCH):
        ed = min(M, st + _AABB_BATCH)
        sqrD, Ii, Ci = tree.squared_distance(Vd, Fi, P[st:ed])
        d[st:ed] = np.sqrt(np.maximum(sqrD, 0.0))
        C[st:ed] = Ci
        I[st:ed] = Ii
    return d, C, I


def _face_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    fN = np.cross(V[F][:,1,:] - V[F][:,0,:],
                  V[F][:,2,:] - V[F][:,0,:])
    nlen = np.linalg.norm(fN, axis=1, keepdims=True)
    nlen[nlen == 0.0] = 1.0
    return (fN / nlen).astype(np.float64)


def compute_sdf_grid(vertices: np.ndarray,
                     faces: np.ndarray,
                     padding: float = 0.1,
                     voxel_size: Optional[float] = None,
                     target_resolution: Optional[int] = None,
                     max_resolution: int = 512,
                     show_progress: bool = True,
                     sdf_backend: str = 'auto',
                     workers: int = -1,
                     robust_sign: bool = True,
                     ambiguous_deg: float = 22.5
                     ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, Dict[str, float], Tuple[float, float, float]]:
    """
    返回（保持原签名）：
      sdf_grid: (nx,ny,nz)  —— 内部负、外部正
      bounds: (bmin,bmax)
      centers: (F,3)  —— 三角中心（占位/兼容）
      normals: (F,3)  —— 三角法向（占位/兼容）
      timings: dict
      voxel_step: (vx,vy,vz)
    另外：模块级 _GRAD_GRID 会存储 (nx,ny,nz,3) 的单位外法向供保存逻辑使用。
    """
    if not _HAS_IGL:
        raise RuntimeError("需要 libigl：pip install igl")

    global _GRAD_GRID
    _GRAD_GRID = None

    t0 = time.time()
    bmin, bmax = _compute_bounds(vertices, padding)
    xs, ys, zs, vstep = _voxel_axes(bmin, bmax, voxel_size, target_resolution, max_resolution)
    pts = _grid_points(xs, ys, zs)
    centers = vertices[faces].mean(axis=1).astype(np.float64)
    fN = _face_normals(vertices, faces)

    timings: Dict[str, float] = {}
    t_grid = time.time()
    timings["grid_setup"] = t_grid - t0

    # --- 优先：signed_distance（含最近点与面法向） ---
    if _HAS_SD and (_HAS_SD_FAST or _HAS_SD_WIND or _HAS_SD_PSEUDO):
        Vd = np.ascontiguousarray(vertices, dtype=np.float64)
        Fi = np.ascontiguousarray(faces, dtype=np.int32)
        Q  = np.ascontiguousarray(pts, dtype=np.float64)
        if _HAS_SD_FAST:
            stype = igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
        elif _HAS_SD_WIND:
            stype = igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER
        else:
            stype = igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL

        S, I, C, N = igl.signed_distance(Q, Vd, Fi, stype)  # S: 外正内负
        S = np.asarray(S, dtype=np.float64).reshape(-1)
        C = np.asarray(C, dtype=np.float64)
        N = np.asarray(N, dtype=np.float64)

        t_q = time.time()
        timings["signed_distance"] = t_q - t_grid

        # 梯度：sgn(S) * (p - C) / ||p-C||
        pc = pts - C
        norm = np.linalg.norm(pc, axis=1, keepdims=True)
        sgn = np.sign(S).reshape(-1, 1)  # 内负 / 外正
        sgn[sgn == 0.0] = 1.0
        g = sgn * pc / np.maximum(norm, _DEF_EPS)
        near = (norm.reshape(-1) < 1e-9)
        if np.any(near):
            g[near] = sgn[near] * N[near]

        sdf_grid = S.reshape(len(xs), len(ys), len(zs))
        _GRAD_GRID = g.reshape(len(xs), len(ys), len(zs), 3)

        t_pack = time.time()
        timings["pack"] = t_pack - t_q
        timings["total"] = t_pack - t0

    else:
        # --- 回退：AABB 最近点 + FWN 判号 ---
        d_unsigned, C, tri_idx = _aabb_dist_closest_idx(pts, vertices, faces)
        if not _HAS_FWN:
            raise RuntimeError("缺少 fast_winding_number_for_meshes（FWN），无法判号；请安装带 FWN 的 libigl。")

        Vd = np.ascontiguousarray(vertices, dtype=np.float64)
        Fi = np.ascontiguousarray(faces, dtype=np.int32)
        Q  = np.ascontiguousarray(pts, dtype=np.float64)
        W = igl.fast_winding_number_for_meshes(Vd, Fi, Q)[0]  # (M,)
        W = np.asarray(W, dtype=np.float64).reshape(-1)

        t_q = time.time()
        timings["aabb+fwn"] = t_q - t_grid

        # SDF：内部负、外部正
        sgnS = np.where(W > 0.5, -1.0, +1.0)   # inside -> -1
        S = d_unsigned * sgnS

        # 梯度：sgn(S) * (p - C) / ||p-C||，近零用面法向回退
        pc = pts - C
        norm = np.linalg.norm(pc, axis=1, keepdims=True)
        g = sgnS.reshape(-1, 1) * pc / np.maximum(norm, _DEF_EPS)
        near = (norm.reshape(-1) < 1e-9)
        if np.any(near):
            tri_n = _face_normals(vertices, faces)[tri_idx[near]]
            g[near] = sgnS.reshape(-1, 1)[near] * tri_n

        sdf_grid = S.reshape(len(xs), len(ys), len(zs))
        _GRAD_GRID = g.reshape(len(xs), len(ys), len(zs), 3)

        t_pack = time.time()
        timings["pack"] = t_pack - t_q
        timings["total"] = t_pack - t0

    return sdf_grid, (bmin, bmax), centers, fN, timings, (float(vstep[0]), float(vstep[1]), float(vstep[2]))


def _marching_cubes_world(sdf_grid: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray]):
    if not (_HAS_SKIMAGE):
        return None
    bmin, bmax = bounds
    nx, ny, nz = sdf_grid.shape
    xs = np.linspace(bmin[0], bmax[0], int(nx), dtype=np.float64)
    ys = np.linspace(bmin[1], bmax[1], int(ny), dtype=np.float64)
    zs = np.linspace(bmin[2], bmax[2], int(nz), dtype=np.float64)
    spacing = (xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0])
    verts, faces, normals, values = _measure.marching_cubes(sdf_grid, level=0.0, spacing=spacing)
    verts_world = verts + np.array(bmin, dtype=np.float64)[None, :]
    return verts_world, faces


def visualize_zero_isosurface(sdf_grid: np.ndarray,
                              bounds: Tuple[np.ndarray, np.ndarray],
                              out_path: str):
    if not (_HAS_MPL and _HAS_SKIMAGE):
        return
    res = _marching_cubes_world(sdf_grid, bounds)
    if res is None:
        return
    verts_world, faces = res
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts_world[:, 0], verts_world[:, 1], Z=verts_world[:, 2],
                    triangles=faces, linewidth=0.0, antialiased=True, alpha=0.85)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Zero Isosurface")
    _ensure_dir_for(out_path)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_timings_pie(timings: Dict[str, float], out_path: str):
    if not _HAS_MPL:
        return
    labels = [k for k in timings.keys() if k != "total"]
    sizes = [timings[k] for k in labels]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.axis('equal')
    _ensure_dir_for(out_path)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_sdf_and_meta(sdf_grid: np.ndarray,
                      bounds: Tuple[np.ndarray, np.ndarray],
                      obj_path: str,
                      voxel_step: Tuple[float, float, float],
                      padding: float,
                      timings: Dict[str, float],
                      out_prefix: str) -> Tuple[str, str, str, str]:
    """
    保持原签名；本梯度版**总是**保存 *_grad.npy，并在 meta.json 写 has_grad=true。
    返回：(isosurface_png, timings_pie_png, sdf_npy, grad_npy, meta_json)
    """
    global _GRAD_GRID
    if _GRAD_GRID is None:
        raise RuntimeError("内部错误：梯度网格尚未生成。")

    npy_path  = f"{out_prefix}_sdf.npy"
    grad_path = f"{out_prefix}_grad.npy"
    meta_path = f"{out_prefix}_meta.json"
    png_path  = f"{out_prefix}_isosurface.png"
    pie_path  = f"{out_prefix}_timings_pie.png"

    _ensure_dir_for(npy_path)
    np.save(npy_path,  sdf_grid)
    np.save(grad_path, _GRAD_GRID)

    bmin, bmax = bounds
    meta = dict(
        obj=os.path.abspath(obj_path),
        bmin=np.asarray(bmin, dtype=np.float64).tolist(),
        bmax=np.asarray(bmax, dtype=np.float64).tolist(),
        shape=list(map(int, sdf_grid.shape)),
        grad_shape=list(map(int, _GRAD_GRID.shape)),
        voxel_step=list(map(float, voxel_step)),
        padding=float(padding),
        timings={k: float(v) for k, v in timings.items()},
        has_grad=True
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


def pyvista_visualize_isosurface(sdf_grid,
                                 bounds,
                                 show=True,
                                 out_png: Optional[str] = None):
    """需要 pyvista；若未安装则抛错。"""
    try:
        import pyvista as pv
    except Exception as e:
        raise RuntimeError("需要 pyvista 才能使用该函数") from e

    res = _marching_cubes_world(sdf_grid, bounds)
    if res is None:
        raise RuntimeError("需要 scikit-image 才能进行 marching_cubes 网格化")
    verts_world, faces = res
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]).ravel()
    mesh = pv.PolyData(verts_world, faces_pv)

    plotter = pv.Plotter(window_size=[900, 700])
    plotter.add_mesh(mesh, color="lightsteelblue", opacity=0.8, smooth_shading=True)
    plotter.show_axes()
    if out_png:
        plotter.screenshot(out_png)
    if show:
        plotter.show()


# =============================================================================
# 轻薄封装与兼容 API（不会覆盖上面定义）
# =============================================================================

# 别名（保持旧 demo 的导入接口一致）
DenseGradientSDF = GradientSDFVolume  # 稠密容器别名


def taylor_query_batch_dense(vol: GradientSDFVolume, P: np.ndarray):
    """兼容旧 demo：调用稠密体素的单体素泰勒批查询。"""
    return vol.taylor_query_batch(P)


def _save_hashed_to_npz(h: HashedGradientSDF, path: Union[str, Path], dtype: Optional[str] = None):
    if dtype is None:
        sample = None
        for blk in h.blocks.values():
            sample = blk.psi.dtype
            break
        dtype = "float32" if (sample is None or sample == np.float32) else "float64"
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    dt = np.float32 if dtype == "float32" else np.float64

    keys, counts, locs_cat, psi_cat, g_cat = [], [], [], [], []
    for (kx, ky, kz), blk in h.blocks.items():
        keys.append([kx, ky, kz])
        counts.append(int(blk.locs.shape[0]))
        locs_cat.append(blk.locs.astype(np.uint8, copy=False))
        psi_cat.append(blk.psi.astype(dt, copy=False))
        g_cat.append(blk.g.astype(dt, copy=False))
    keys = np.asarray(keys, dtype=np.int32) if keys else np.empty((0, 3), dtype=np.int32)
    counts = np.asarray(counts, dtype=np.int32) if counts else np.empty((0,), dtype=np.int32)
    locs_cat = np.concatenate(locs_cat, axis=0) if len(locs_cat) else np.empty((0, 3), dtype=np.uint8)
    psi_cat  = np.concatenate(psi_cat,  axis=0) if len(psi_cat)  else np.empty((0,),   dtype=dt)
    g_cat    = np.concatenate(g_cat,    axis=0) if len(g_cat)    else np.empty((0, 3), dtype=dt)

    np.savez_compressed(str(path),
        bmin=h.spec.bmin.astype(np.float64),
        bmax=h.spec.bmax.astype(np.float64),
        voxel_step=h.spec.voxel.astype(np.float64),
        shape=np.asarray(h.spec.shape, dtype=np.int32),
        block_size=np.asarray([h.block_size], dtype=np.int32),
        keys=keys, counts=counts, locs=locs_cat, psi=psi_cat, g=g_cat
    )


def _load_hashed_from_npz(path: Union[str, Path]) -> HashedGradientSDF:
    with np.load(str(path), allow_pickle=False) as z:
        bmin = z["bmin"].astype(np.float64)
        bmax = z["bmax"].astype(np.float64)
        voxel = z["voxel_step"].astype(np.float64)
        shape = tuple(int(x) for x in z["shape"].astype(np.int32).tolist())
        block_size = int(z["block_size"][0])
        keys = z["keys"].astype(np.int32)
        counts = z["counts"].astype(np.int32)
        locs = z["locs"].astype(np.uint8)
        psi = z["psi"]
        g = z["g"]

    spec = GridSpec(bmin=bmin, bmax=bmax, voxel=voxel, shape=shape)
    blocks: Dict[Tuple[int, int, int], _Block] = {}
    off = 0
    for (kx, ky, kz), c in zip(keys, counts):
        c = int(c)
        if c <= 0:
            continue
        base = (int(kx) * block_size, int(ky) * block_size, int(kz) * block_size)
        locs_i = locs[off:off + c]
        psi_i = psi[off:off + c]
        g_i = g[off:off + c]
        blocks[(int(kx), int(ky), int(kz))] = _Block(base=base, locs=locs_i, psi=psi_i, g=g_i)
        off += c
    return HashedGradientSDF(spec=spec, blocks=blocks, block_size=block_size)


class GradientSDF:
    def __init__(self, core: HashedGradientSDF):
        self.core = core

    @classmethod
    def from_obj(cls, obj_path: Union[str, Path],
                 out_npz: Optional[Union[str, Path]] = None,
                 out_dense_prefix: Optional[Union[str, Path]] = None,
                 keep_dense_in_memory: bool = False,
                 tau: float = 10.0,
                 block_size: int = 8,
                 dtype: str = "float32",
                 padding: float = 0.1,
                 voxel_size: Optional[float] = None,
                 target_resolution: Optional[int] = 192,
                 max_resolution: int = 512,
                 verbose: bool = True) -> "GradientSDF":
        if verbose: print("[build] parse OBJ ...")
        V, F = parse_obj(str(obj_path))
        if verbose: print("[build] compute dense SDF grid (tools) ...")
        sdf_grid, (bmin, bmax), _centers, _normals, timings, vstep = compute_sdf_grid(
            V, F, padding=padding, voxel_size=voxel_size,
            target_resolution=target_resolution, max_resolution=max_resolution,
            show_progress=verbose, sdf_backend='auto'
        )
        if '_GRAD_GRID' not in globals() or globals()['_GRAD_GRID'] is None:
            raise RuntimeError("tools.compute_sdf_grid 未生成梯度网格（_GRAD_GRID is None）")
        grad_grid = globals()['_GRAD_GRID']

        if out_dense_prefix is not None:
            if verbose: print(f"[io] save dense triplet via tools -> {out_dense_prefix}_{{sdf,grad,meta}}.*")
            save_sdf_and_meta(sdf_grid, (bmin, bmax), str(obj_path), tuple(map(float, vstep)),
                              float(padding), timings, str(out_dense_prefix))
            keep_dense_in_memory = True

        dense = GradientSDFVolume.__new__(GradientSDFVolume)
        dense.sdf = np.asarray(sdf_grid, dtype=np.float64, copy=False)
        dense.grad = np.asarray(grad_grid, dtype=np.float64, copy=False)
        dense.spec = GridSpec(
            bmin=np.asarray(bmin, dtype=np.float64),
            bmax=np.asarray(bmax, dtype=np.float64),
            voxel=np.asarray(vstep, dtype=np.float64),
            shape=sdf_grid.shape
        )

        if verbose: print("[hash] build from dense (runtime) ...")
        core = HashedGradientSDF.build_from_dense(dense, tau=float(tau), block_size=int(block_size), dtype=dtype)

        g = cls(core)
        if keep_dense_in_memory:
            if verbose: print("[build] attach dense fallback for runtime ...")
            g.core.attach_dense_fallback(dense)

        if out_npz is not None:
            if verbose: print(f"[io] save sparse npz -> {out_npz}")
            _save_hashed_to_npz(g.core, out_npz, dtype=dtype)

        if verbose: print("[warmup] index_map ...")
        g.set_query_backend("index_map")
        g.warmup(index_map=True)

        if verbose: print("[build] done ✓")
        return g

    @classmethod
    def from_dense_prefix(cls, prefix: Union[str, Path],
                          tau: float = 10.0,
                          block_size: int = 8,
                          dtype: str = "float32",
                          save_npz: Optional[Union[str, Path]] = None,
                          attach_dense: bool = False,
                          verbose: bool = True) -> "GradientSDF":
        prefix = Path(prefix)
        if verbose: print(f"[load] dense triplet <- {prefix}_{{sdf,grad,meta}}.*")
        dense = load_dense_by_prefix(prefix)
        if verbose:
            s = dense.spec
            print(f"[grid] shape={s.shape}, voxel={s.voxel}, bmin={s.bmin}, bmax={s.bmax}")
        core = HashedGradientSDF.build_from_dense(dense, tau=float(tau), block_size=int(block_size), dtype=dtype)
        g = cls(core)
        if attach_dense:
            if verbose: print("[load] attach dense fallback")
            g.core.attach_dense_fallback(dense)
        if save_npz is not None:
            if verbose: print(f"[io] save sparse npz -> {save_npz}")
            _save_hashed_to_npz(g.core, save_npz, dtype=dtype)

        if verbose: print("[warmup] index_map ...")
        g.set_query_backend("index_map")
        g.warmup(index_map=True)
        if verbose: print("[load] done ✓")
        return g

    @classmethod
    def from_npz(cls, path: Union[str, Path], *, verbose: bool = True) -> "GradientSDF":
        if verbose: print(f"[load] sparse npz <- {path}")
        core = _load_hashed_from_npz(path)
        g = cls(core)
        if verbose: print("[warmup] index_map ...")
        g.set_query_backend("index_map")
        g.warmup(index_map=True)
        if verbose: print("[load] done ✓")
        return g

    # ---- 查询 ----
    def set_query_backend(self, mode: str = "index_map"):
        self.core.set_query_backend(mode)

    def warmup(self, *, index_map: bool = True, ptr_table: bool = False, atlas: bool = False):
        self.core.warmup(index_map=index_map, ptr_table=ptr_table, atlas=atlas)

    def query_points(self, P: Iterable[Iterable[float]], *, allow_fallback: bool = False
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        P = np.asarray(P, dtype=np.float64)
        d, n, vj, ijk, hit = self.core.taylor_query_batch(P)
        if allow_fallback and (self.core._dense is not None):
            miss = (~hit)
            if np.any(miss):
                d2, n2, *_ = self.core._dense.taylor_query_batch(P[miss])
                d[miss] = d2
                n[miss] = n2
                hit[miss] = np.isfinite(d2)
        return d, n, hit

    def query_point(self, p: Iterable[float], *, allow_fallback: bool = False
                    ) -> Tuple[float, np.ndarray, bool]:
        P = np.asarray(p, dtype=np.float64).reshape(1, 3)
        d, n, hit = self.query_points(P, allow_fallback=allow_fallback)
        return float(d[0]), n[0], bool(hit[0])

    # ---- I/O ----
    def save_npz(self, path: Union[str, Path], dtype: Optional[str] = None):
        _save_hashed_to_npz(self.core, path, dtype=dtype)

    def save_dense(self, prefix: Union[str, Path]):
        if self.core._dense is None:
            raise RuntimeError("当前实例未保留稠密数据，无法导出。请在构建时 keep_dense_in_memory=True 或 attach 稠密回退。")
        s = self.core.spec
        save_sdf_and_meta(self.core._dense.sdf, (s.bmin, s.bmax), "<unknown-obj>", tuple(map(float, s.voxel)),
                          padding=0.0, timings={}, out_prefix=str(prefix))

    @property
    def grid_spec(self) -> GridSpec:
        return self.core.spec

    @property
    def block_size(self) -> int:
        return self.core.block_size


# =============================================================================
# CLI（保持与原 tools CLI 的可用性）
# =============================================================================

def _main():
    ap = argparse.ArgumentParser("Gradient SDF Grid Generator (always outputs grad_grid)")
    ap.add_argument("--obj", required=True, help="输入 .obj 网格路径")
    ap.add_argument("--out", required=True, help="输出前缀，例如 out/scene")
    ap.add_argument("--padding", type=float, default=0.1)
    ap.add_argument("--voxel_size", type=float, default=None, help="体素尺寸（优先于 target_resolution）")
    ap.add_argument("--target_resolution", type=int, default=192)
    ap.add_argument("--max_resolution", type=int, default=512)
    args = ap.parse_args()

    V, F = parse_obj(args.obj)
    sdf_grid, bounds, centers, normals, timings, voxel_step = compute_sdf_grid(
        V, F,
        padding=args.padding,
        voxel_size=args.voxel_size,
        target_resolution=args.target_resolution,
        max_resolution=args.max_resolution,
        show_progress=True,
        sdf_backend='auto'
    )

    png, pie, npy, meta = save_sdf_and_meta(
        sdf_grid, bounds=bounds, obj_path=args.obj,
        voxel_step=voxel_step, padding=args.padding,
        timings=timings, out_prefix=args.out
    )

    print(json.dumps({
        "isosurface_png": png,
        "timings_pie": pie,
        "sdf_npy": npy,
        "grad_npy": f"{args.out}_grad.npy",
        "meta_json": meta
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
