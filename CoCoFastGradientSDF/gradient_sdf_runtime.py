# -*- coding: utf-8 -*-
"""
Gradient-SDF 运行时：单体素泰勒查询 & 稀疏哈希体素存储（向量化+哈希加速，修复版）
================================================================
- 稠密查询：单体素泰勒（全向量化）
- 稀疏哈希：块哈希，批量向量化查询（按块分组 + 代码表 searchsorted），并修复越界索引问题
- I/O：.npz 保存/加载；哈希 (ψ, ĝ) 精度严格跟随构建参数 dtype（float64/float32）

依赖：numpy, json
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

Array = np.ndarray

# -----------------------------------------------------------------------------
# 基础工具
# -----------------------------------------------------------------------------

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
    def nx(self) -> int: return int(self.shape[0])
    @property
    def ny(self) -> int: return int(self.shape[1])
    @property
    def nz(self) -> int: return int(self.shape[2])

    def center_from_index(self, ijk: Array) -> Array:
        return self.bmin + self.voxel * (ijk.astype(np.float64) + 0.5)

    def index_from_world(self, p: Array) -> Array:
        ijk_f = (p - self.bmin) / self.voxel - 0.5
        return _round_to_int(ijk_f)


# -----------------------------------------------------------------------------
# 稠密体素：读取/查询（批量向量化）
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# 稀疏哈希体素（块哈希 + 向量化批量查询）
# -----------------------------------------------------------------------------

@dataclass
class _Block:
    base: Tuple[int, int, int]  # 绝对体素基坐标
    locs: Array                  # (M,3) uint8
    psi: Array                   # (M,)
    g: Array                     # (M,3)
    # 小稠密指针表（懒构建）：shape=(B,B,B)，值为本块局部索引，-1 表示该处无体素
    ptr: Optional[Array] = None  # int32

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
        # 查询后端：'index_map'（默认，最快，O(1) 全向量化）或 'block_ptr'（B×B×B 指针表）
        self.query_backend: str = 'index_map'
        # ---- Block Atlas （跨块全向量化 O(1) 指针表） ----
        self._atlas_stride: int = self.block_size ** 3
        self._atlas_ptr_flat: Optional[np.ndarray] = None   # int32, shape=(num_blocks*B^3,)
        self._block_row_map: Optional[np.ndarray] = None    # int32, length = nbx*nby*nz, -1=absent
        self._packed_psi: Optional[np.ndarray] = None       # packed over all blocks
        self._packed_g: Optional[np.ndarray] = None
    @staticmethod
    def build_from_dense(vol: GradientSDFVolume, tau: float = 10.0, block_size: int = 8, dtype: str = "float32") -> "HashedGradientSDF":
        """从稠密体素提取窄带 |ψ|<=tau 的稀疏哈希并构建对象。
        `dtype` 决定保存到哈希中的 (ψ, ĝ) 精度（float32/float64）。
        """
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

        # 先为每个块确定一个 row，并计算该块在 packed 数组里的偏移
        # 我们把所有块的 (psi, g) 也拼接成 packed_psi/g，便于 O(1) 直接寻址
        packed_psi = []
        packed_g = []
        row = 0
        for (kx, ky, kz), blk in self.blocks.items():
            block_row_map[kx + nbx * (ky + nby * kz)] = row
            # 该块的局部 -> 全局 packed 偏移
            base_off = sum(arr.shape[0] for arr in packed_psi)
            # 确保局部索引存在
            li = blk.locs[:, 0].astype(np.intp)
            lj = blk.locs[:, 1].astype(np.intp)
            lk = blk.locs[:, 2].astype(np.intp)
            local_code = (li + B * (lj + B * lk)).astype(np.intp)
            # 把该块的本地索引映射到 atlas 行
            atlas_ptr[row, local_code] = base_off + np.arange(blk.locs.shape[0], dtype=np.int32)
            # 追加 payload
            packed_psi.append(blk.psi)
            packed_g.append(blk.g)
            row += 1
        self._atlas_ptr_flat = atlas_ptr.ravel()
        self._atlas_stride = stride
        self._block_row_map = block_row_map
        self._packed_psi = np.concatenate(packed_psi) if len(packed_psi) else np.empty((0,), dtype=np.float64)
        self._packed_g = np.concatenate(packed_g, axis=0) if len(packed_g) else np.empty((0,3), dtype=np.float64)

    # -------------------- 批量向量化哈希查询（修复越界） --------------------
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
                # 局部索引合法性（极少数边界问题）
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

        else:  # 'block_ptr'（按块循环）
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
        else:
            # ---- 块内小稠密指针表路径：按块 O(1) 直接寻址 ----
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

        I = ijk[inb, 0].astype(np.int64)
        J = ijk[inb, 1].astype(np.int64)
        K = ijk[inb, 2].astype(np.int64)
        B = self.block_size
        out_idx = np.where(inb)[0]

        # 分块：唯一编码 + 反算 block key
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
            # O(1) 直接寻址：从 ptr 取块内局部索引
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

        self._ensure_index_map()
        codes_all = self._codes_all  # type: ignore
        psi_all = self._psi_all      # type: ignore
        g_all = self._g_all          # type: ignore
        index_map = self._index_map  # type: ignore
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

        if allow_fallback and (self._dense is not None):
            miss = inb & (~hit)
            if np.any(miss):
                d2, g2, _, _, _ = self._dense.taylor_query_batch(P[miss])
                d[miss] = d2
                g[miss] = g2
                hit[miss] = np.isfinite(d2)
        return d, g, vj, ijk, hit

        # ---- 全局索引路径：一次 searchsorted 在大表上 ----
        self._ensure_global_index()
        codes_all = self._codes_all  # type: ignore
        psi_all = self._psi_all      # type: ignore
        g_all = self._g_all          # type: ignore
        nx, ny, nz = self.spec.shape

        I = ijk[inb, 0].astype(np.int64)
        J = ijk[inb, 1].astype(np.int64)
        K = ijk[inb, 2].astype(np.int64)
        code_q = I + nx * (J + ny * K)
        out_idx = np.where(inb)[0]

        # 二分查找
        pos = np.searchsorted(codes_all, code_q, side='left')
        okpos = pos < codes_all.size
        if np.any(okpos):
            pos_ok = pos[okpos]
            cq_ok = code_q[okpos]
            match = codes_all[pos_ok] == cq_ok
            if np.any(match):
                pos_ok2 = pos_ok[match]
                idx_fin = out_idx[okpos][match]
                psi = psi_all[pos_ok2].astype(np.float64)
                gg = g_all[pos_ok2].astype(np.float64)
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

        I = ijk[inb, 0].astype(np.int64)
        J = ijk[inb, 1].astype(np.int64)
        K = ijk[inb, 2].astype(np.int64)
        B = self.block_size
        out_idx = np.where(inb)[0]

        # 计算每点的块键唯一编码
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
            blk.ensure_index(B)

            sel = (inv == u)
            idx_pts = out_idx[sel]
            li = I[sel] - blk.base[0]
            lj = J[sel] - blk.base[1]
            lk = K[sel] - blk.base[2]
            ok = (li >= 0) & (li < B) & (lj >= 0) & (lj < B) & (lk >= 0) & (lk < B)
            if not np.any(ok):
                continue
            li = li[ok].astype(np.uint32)
            lj = lj[ok].astype(np.uint32)
            lk = lk[ok].astype(np.uint32)
            idx_ok = idx_pts[ok]
            qcodes = li + (lj * np.uint32(B)) + (lk * np.uint32(B * B))

            # 安全 searchsorted：严格先筛下标，再取值比较
            codes = blk._codes_sorted
            if codes is None or codes.size == 0:
                continue
            pos = np.searchsorted(codes, qcodes, side='left')
            okpos = pos < codes.size
            if not np.any(okpos):
                continue
            pos_ok = pos[okpos]
            q_ok = qcodes[okpos]
            match = codes[pos_ok] == q_ok
            if not np.any(match):
                continue
            pos_ok = pos_ok[match]
            idx_fin = idx_ok[okpos][match]

            ord2 = blk._ord[pos_ok]
            psi = blk.psi[ord2].astype(np.float64)
            gg = blk.g[ord2].astype(np.float64)
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

        I = ijk[inb, 0].astype(np.int64)
        J = ijk[inb, 1].astype(np.int64)
        K = ijk[inb, 2].astype(np.int64)
        B = self.block_size
        out_idx = np.where(inb)[0]

        # 计算每点的块键唯一编码
        kx = I // B
        ky = J // B
        kz = K // B
        key_lin = (kx + self._nbx*(ky + self._nby*kz)).astype(np.int64)
        uniq, inv = np.unique(key_lin, return_inverse=True)

        for u, code in enumerate(uniq):
            kz_u = code // (self._nbx*self._nby)
            rem = code - kz_u*(self._nbx*self._nby)
            ky_u = rem // self._nbx
            kx_u = rem - ky_u*self._nbx
            key = (int(kx_u), int(ky_u), int(kz_u))
            blk = self.blocks.get(key)
            if blk is None:
                continue
            blk.ensure_index(B)
            sel = (inv == u)
            idx_pts = out_idx[sel]
            li = I[sel] - blk.base[0]
            lj = J[sel] - blk.base[1]
            lk = K[sel] - blk.base[2]
            ok = (li >= 0) & (li < B) & (lj >= 0) & (lj < B) & (lk >= 0) & (lk < B)
            if not np.any(ok):
                continue
            li = li[ok].astype(np.uint32)
            lj = lj[ok].astype(np.uint32)
            lk = lk[ok].astype(np.uint32)
            idx_ok = idx_pts[ok]
            qcodes = li + (lj*np.uint32(B)) + (lk*np.uint32(B*B))

            # 安全 searchsorted：先筛 pos<len，再比较
            pos = np.searchsorted(blk._codes_sorted, qcodes)
            okpos = pos < blk._codes_sorted.size
            if not np.any(okpos):
                continue
            pos_ok = pos[okpos]
            q_ok = qcodes[okpos]
            match = blk._codes_sorted[pos_ok] == q_ok
            if not np.any(match):
                continue
            pos_ok = pos_ok[match]
            idx_fin = idx_ok[okpos][match]

            ord2 = blk._ord[pos_ok]
            psi = blk.psi[ord2].astype(np.float64)
            gg = blk.g[ord2].astype(np.float64)
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


# -----------------------------------------------------------------------------
# 便捷加载器
# -----------------------------------------------------------------------------

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
