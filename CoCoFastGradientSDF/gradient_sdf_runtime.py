# -*- coding: utf-8 -*-
"""
gradient_sdf.py — 稠密/稀疏一体化 Gradient-SDF（性能版）
---------------------------------------------------
- 支持两种存储：
  1) 稀疏梯度 SDF（.npz）：块哈希 + 单体素泰勒一阶（ψ,g）
  2) 稠密三件套（prefix_sdf.npy / prefix_grad.npy / prefix_meta.json）
- 统一 API：from_obj / from_dense_prefix / from_npz / query_points / save_npz / save_dense / set_query_backend
- 查询后端：index_map（默认，最快）/ block_ptr / block_atlas
- 导出工具函数：taylor_query_batch_dense（供回退/对比）

注意：默认稀疏 payload 使用 float32，以匹配你原实现的带宽优势。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

Array = np.ndarray
__all__ = [
    "GridSpec", "DenseGradientSDF", "HashedGradientSDF", "GradientSDF",
    "taylor_query_batch_dense"
]


# ------------------------- 基础 -------------------------

def _round_to_int(a: Array) -> Array:
    return np.rint(a).astype(np.int64)

def _in_bounds(ijk: Array, shape: Tuple[int,int,int]) -> Array:
    i, j, k = ijk[...,0], ijk[...,1], ijk[...,2]
    nx, ny, nz = shape
    return (i>=0)&(i<nx)&(j>=0)&(j<ny)&(k>=0)&(k<nz)


@dataclass
class GridSpec:
    bmin: Array
    bmax: Array
    voxel: Array              # (vx,vy,vz)
    shape: Tuple[int,int,int] # (nx,ny,nz)

    @property
    def nx(self): return int(self.shape[0])
    @property
    def ny(self): return int(self.shape[1])
    @property
    def nz(self): return int(self.shape[2])

    def center_from_index(self, ijk: Array) -> Array:
        return self.bmin + self.voxel * (ijk.astype(np.float64) + 0.5)

    def index_from_world(self, p: Array) -> Array:
        return _round_to_int((p - self.bmin) / self.voxel - 0.5)


# ------------------------- 稠密容器/保存 -------------------------

class DenseGradientSDF:
    """加载 prefix_* 三件套（*_sdf.npy, *_grad.npy, *_meta.json）"""
    def __init__(self, sdf_npy: Union[str,Path], grad_npy: Optional[Union[str,Path]] = None, meta_json: Optional[Union[str,Path]] = None):
        sdf_npy = Path(sdf_npy)
        grad_npy = Path(grad_npy) if grad_npy else sdf_npy.with_name(sdf_npy.name.replace("_sdf.npy", "_grad.npy"))
        meta_json = Path(meta_json) if meta_json else sdf_npy.with_name(sdf_npy.name.replace("_sdf.npy", "_meta.json"))
        if not sdf_npy.exists() or not grad_npy.exists() or not meta_json.exists():
            raise FileNotFoundError("缺少 sdf/grad/meta 之一")
        self.sdf: Array = np.load(sdf_npy).astype(np.float64, copy=False)
        self.grad: Array = np.load(grad_npy).astype(np.float64, copy=False)
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        shape = tuple(int(x) for x in meta["shape"])
        if tuple(self.sdf.shape) != shape or tuple(self.grad.shape[:3]) != shape or self.grad.shape[-1] != 3:
            raise ValueError("SDF/Grad 形状与 meta 不一致或梯度最后一维≠3")
        self.spec = GridSpec(
            bmin=np.asarray(meta["bmin"], dtype=np.float64),
            bmax=np.asarray(meta["bmax"], dtype=np.float64),
            voxel=np.asarray(meta.get("voxel_step", meta.get("voxel", [1,1,1])), dtype=np.float64),
            shape=shape,
        )

def save_dense_triplet(prefix: Union[str,Path], sdf: Array, grad: Array, spec: GridSpec, *, dtype: str = "float32"):
    """保存稠密：prefix_sdf.npy / prefix_grad.npy / prefix_meta.json"""
    if dtype not in ("float32","float64"): raise ValueError("dtype must be float32/float64")
    prefix = Path(prefix)
    np.save(prefix.with_name(prefix.name + "_sdf.npy"),  sdf.astype(np.float32 if dtype=="float32" else np.float64, copy=False))
    np.save(prefix.with_name(prefix.name + "_grad.npy"), grad.astype(np.float32 if dtype=="float32" else np.float64, copy=False))
    meta = dict(bmin=spec.bmin.tolist(), bmax=spec.bmax.tolist(), voxel_step=spec.voxel.tolist(), shape=list(spec.shape))
    with open(prefix.with_name(prefix.name + "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ------------------------- 稀疏（块哈希） -------------------------

@dataclass
class _Block:
    base: Tuple[int,int,int]   # 绝对体素起点
    locs: Array                # (M,3) uint8
    psi: Array                 # (M,)  float32/64
    g:   Array                 # (M,3) float32/64
    ptr: Optional[Array] = None # (B,B,B) int32，-1 表示 miss

    def ensure_ptr(self, B: int):
        if self.ptr is not None: return
        ptr = np.full((B,B,B), -1, dtype=np.int32)
        li, lj, lk = self.locs[:,0].astype(np.intp), self.locs[:,1].astype(np.intp), self.locs[:,2].astype(np.intp)
        ptr[li, lj, lk] = np.arange(self.locs.shape[0], dtype=np.int32)
        self.ptr = ptr


class HashedGradientSDF:
    """稀疏梯度 SDF 的核心。默认 index_map 后端速度最佳。"""
    def __init__(self, spec: GridSpec, blocks: Dict[Tuple[int,int,int], _Block], block_size: int = 8):
        self.spec = spec
        self.blocks = blocks
        self.block_size = int(block_size)
        self._dense: Optional[DenseGradientSDF] = None

        # 块网格尺寸
        self._nbx = (self.spec.nx + self.block_size - 1) // self.block_size
        self._nby = (self.spec.ny + self.block_size - 1) // self.block_size
        self._nbz = (self.spec.nz + self.block_size - 1) // self.block_size

        # 懒构建缓存
        self._codes_all: Optional[np.ndarray] = None
        self._psi_all: Optional[np.ndarray] = None
        self._g_all: Optional[np.ndarray] = None
        self._index_map: Optional[np.ndarray] = None  # 长度 nx*ny*nz, int32, -1=miss

        # Block Atlas
        self._atlas_stride: int = self.block_size ** 3
        self._atlas_ptr_flat: Optional[np.ndarray] = None
        self._block_row_map: Optional[np.ndarray] = None
        self._packed_psi: Optional[np.ndarray] = None
        self._packed_g: Optional[np.ndarray] = None

        self.query_backend: str = "index_map"

    # ---------- 构建 ----------
    @staticmethod
    def build_from_dense(vol: 'DenseGradientSDF', tau: float = 10.0, block_size: int = 8, dtype: str = "float32") -> "HashedGradientSDF":
        if dtype not in ("float32","float64"): raise ValueError("dtype must be float32/float64")
        sdf, grad, spec = vol.sdf, vol.grad, vol.spec
        mask = np.abs(sdf) <= float(tau)
        idxs = np.argwhere(mask)
        if len(idxs) == 0: raise ValueError("阈值过小，窄带为空。")
        B = int(block_size)
        buckets: Dict[Tuple[int,int,int], List[Tuple[Tuple[int,int,int], float, Array]]] = {}
        for (i,j,k) in idxs:
            key = (int(i)//B, int(j)//B, int(k)//B)
            if key not in buckets: buckets[key] = []
            buckets[key].append(((int(i),int(j),int(k)), float(sdf[i,j,k]), grad[i,j,k]))

        blocks: Dict[Tuple[int,int,int], _Block] = {}
        dt = np.float32 if dtype=="float32" else np.float64
        for key, items in buckets.items():
            base = (key[0]*B, key[1]*B, key[2]*B)
            M = len(items)
            locs = np.empty((M,3), dtype=np.uint8)
            psi  = np.empty((M,), dtype=dt)
            g    = np.empty((M,3), dtype=dt)
            for n,(ijk,psi_v,g_v) in enumerate(items):
                locs[n] = np.array([ijk[0]-base[0], ijk[1]-base[1], ijk[2]-base[2]], dtype=np.uint8)
                psi[n]  = psi_v
                g[n]    = g_v
            blocks[key] = _Block(base=base, locs=locs, psi=psi, g=g)
        return HashedGradientSDF(spec=spec, blocks=blocks, block_size=B)

    def attach_dense_fallback(self, vol: 'DenseGradientSDF'):
        self._dense = vol

    # ---------- 后端/预热 ----------
    def set_query_backend(self, mode: str = "index_map"):
        if mode not in ("index_map", "block_ptr", "block_atlas"):
            raise ValueError("mode must be 'index_map'/'block_ptr'/'block_atlas'")
        self.query_backend = mode

    def warmup(self, *, index_map: bool = True, ptr_table: bool = False, atlas: bool = False):
        if index_map: self._ensure_index_map()
        if ptr_table:
            B = self.block_size
            for blk in self.blocks.values():
                blk.ensure_ptr(B)
        if atlas: self._ensure_block_atlas()

    # ---------- 懒构建：全局索引/映射 ----------
    def _ensure_global_index(self):
        if self._codes_all is not None: return
        nx, ny, nz = self.spec.shape
        codes_list, psi_list, g_list = [], [], []
        for blk in self.blocks.values():
            locs = blk.locs.astype(np.int64)
            i = blk.base[0] + locs[:,0]
            j = blk.base[1] + locs[:,1]
            k = blk.base[2] + locs[:,2]
            code = i + nx * (j + ny * k)     # int64
            codes_list.append(code.astype(np.int64, copy=False))
            psi_list.append(blk.psi)          # 注意：保留原 dtype（float32/64）
            g_list.append(blk.g)
        if len(codes_list)==0:
            self._codes_all = np.empty((0,), dtype=np.int64)
            self._psi_all   = np.empty((0,), dtype=np.float32)
            self._g_all     = np.empty((0,3), dtype=np.float32)
            return
        codes = np.concatenate(codes_list).astype(np.int64, copy=False)
        psi   = np.concatenate(psi_list, axis=0)
        g     = np.concatenate(g_list, axis=0)
        ord_  = np.argsort(codes, kind="mergesort")
        self._codes_all = codes[ord_]
        self._psi_all   = psi[ord_]   # dtype 原样保留
        self._g_all     = g[ord_]     # dtype 原样保留

    def _ensure_index_map(self):
        if self._index_map is not None: return
        self._ensure_global_index()
        total = int(self.spec.nx*self.spec.ny*self.spec.nz)
        index_map = np.full((total,), -1, dtype=np.int32)
        if self._codes_all.size:
            index_map[self._codes_all] = np.arange(self._codes_all.size, dtype=np.int32)
        self._index_map = index_map

    # ---------- 懒构建：Block Atlas ----------
    def _ensure_block_atlas(self):
        if self._atlas_ptr_flat is not None: return
        B = self.block_size
        nbx, nby, nbz = self._nbx, self._nby, self._nbz
        stride = B*B*B
        # 为每个块分配一行；row_map 负责 (kx,ky,kz)->row 的线性映射
        row_map = np.full((nbx*nby*nbz,), -1, dtype=np.int32)
        atlas_rows = []
        packed_psi = []
        packed_g   = []
        row = 0
        for (kx,ky,kz), blk in self.blocks.items():
            row_map[kx + nbx*(ky + nby*kz)] = row
            # 本块在拼接数组的起始偏移
            base_off = sum(x.shape[0] for x in packed_psi)
            # 本块的本地 (B^3) 指针行
            row_ptr = np.full((stride,), -1, dtype=np.int32)
            li = blk.locs[:,0].astype(np.intp)
            lj = blk.locs[:,1].astype(np.intp)
            lk = blk.locs[:,2].astype(np.intp)
            lcode = li + B*(lj + B*lk)
            row_ptr[lcode] = base_off + np.arange(blk.locs.shape[0], dtype=np.int32)
            atlas_rows.append(row_ptr)
            packed_psi.append(blk.psi)
            packed_g.append(blk.g)
            row += 1

        self._atlas_ptr_flat = (np.stack(atlas_rows, axis=0).reshape(-1) if len(atlas_rows) else np.empty((0,), dtype=np.int32))
        self._atlas_stride   = stride
        self._block_row_map  = row_map
        self._packed_psi     = (np.concatenate(packed_psi, axis=0) if len(packed_psi) else np.empty((0,), dtype=np.float32))
        self._packed_g       = (np.concatenate(packed_g, axis=0)   if len(packed_g)   else np.empty((0,3), dtype=np.float32))

    # ---------- 稀疏：批量泰勒查询 ----------
    def taylor_query_batch(self, P: Array, allow_fallback: bool = False
                           ) -> Tuple[Array, Array, Array, Array, Array]:
        """
        输入：P(M,3) 世界坐标
        输出：d(M,), g(M,3), vj(M,3), ijk(M,3), hit(M,)bool
        - 计算在 payload dtype（float32/64）上进行，避免 per-call 复制；最后按需写入 float64 输出。
        """
        P = np.asarray(P, dtype=np.float64)
        ijk = self.spec.index_from_world(P)
        vj  = self.spec.center_from_index(ijk)
        inb = _in_bounds(ijk, self.spec.shape)

        d   = np.full((len(P),),  np.nan, dtype=np.float64)
        g   = np.full((len(P),3), np.nan, dtype=np.float64)
        hit = np.zeros((len(P),), dtype=bool)
        if not np.any(inb):
            return d, g, vj, ijk, hit

        if self.query_backend == "index_map":
            self._ensure_index_map()
            nx, ny, nz = self.spec.shape
            I = ijk[inb,0].astype(np.int64)
            J = ijk[inb,1].astype(np.int64)
            K = ijk[inb,2].astype(np.int64)
            code = I + nx*(J + ny*K)
            idx  = self._index_map[code]            # type: ignore
            ok   = idx >= 0
            if np.any(ok):
                out_idx = np.where(inb)[0][ok]
                idx2    = idx[ok].astype(np.int64)
                psi_all = self._psi_all             # type: ignore
                g_all   = self._g_all               # type: ignore
                dt      = psi_all.dtype             # payload dtype
                Pdt  = P.astype(dt, copy=False)
                vjdt = vj.astype(dt, copy=False)

                psi  = psi_all[idx2]                # 不复制
                gg   = g_all[idx2]
                diff = Pdt[out_idx] - vjdt[out_idx]
                dloc = psi + np.einsum("ij,ij->i", diff, gg, optimize=True)
                d[out_idx] = dloc.astype(np.float64, copy=False)
                g[out_idx] = gg.astype(np.float64, copy=False)
                hit[out_idx] = True

        elif self.query_backend == "block_atlas":
            self._ensure_block_atlas()
            B = self.block_size
            nbx, nby = self._nbx, self._nby
            stride   = self._atlas_stride
            row_map  = self._block_row_map        # type: ignore
            atlas    = self._atlas_ptr_flat       # type: ignore
            psi_all  = self._packed_psi           # type: ignore
            g_all    = self._packed_g             # type: ignore
            dt       = psi_all.dtype

            I = ijk[inb,0].astype(np.int64)
            J = ijk[inb,1].astype(np.int64)
            K = ijk[inb,2].astype(np.int64)
            kx = I // B; ky = J // B; kz = K // B
            row = row_map[kx + nbx*(ky + nby*kz)]
            ok_row = row >= 0
            if np.any(ok_row):
                out_idx = np.where(inb)[0][ok_row]
                row2 = row[ok_row].astype(np.int64)
                li = (I[ok_row] - (kx[ok_row]*B)).astype(np.int64)
                lj = (J[ok_row] - (ky[ok_row]*B)).astype(np.int64)
                lk = (K[ok_row] - (kz[ok_row]*B)).astype(np.int64)
                ok_loc = (li>=0)&(li<B)&(lj>=0)&(lj<B)&(lk>=0)&(lk<B)
                if np.any(ok_loc):
                    pts   = out_idx[ok_loc]
                    row3  = row2[ok_loc]
                    li2, lj2, lk2 = li[ok_loc], lj[ok_loc], lk[ok_loc]
                    lcode = li2 + B*(lj2 + B*lk2)
                    ptr   = atlas[row3*stride + lcode]
                    ok_ptr = ptr >= 0
                    if np.any(ok_ptr):
                        pts2 = pts[ok_ptr]
                        ptr2 = ptr[ok_ptr].astype(np.int64)
                        Pdt  = P.astype(dt, copy=False)
                        vjdt = vj.astype(dt, copy=False)
                        psi  = psi_all[ptr2]
                        gg   = g_all[ptr2]
                        diff = Pdt[pts2] - vjdt[pts2]
                        dloc = psi + np.einsum("ij,ij->i", diff, gg, optimize=True)
                        d[pts2] = dloc.astype(np.float64, copy=False)
                        g[pts2] = gg.astype(np.float64, copy=False)
                        hit[pts2] = True

        else:  # block_ptr
            B = self.block_size
            out_idx_all = np.where(inb)[0]
            I = ijk[inb,0].astype(np.int64)
            J = ijk[inb,1].astype(np.int64)
            K = ijk[inb,2].astype(np.int64)
            kx = I // B; ky = J // B; kz = K // B
            key_lin = (kx + self._nbx*(ky + self._nby*kz)).astype(np.int64)
            uniq, inv = np.unique(key_lin, return_inverse=True)
            for u, code in enumerate(uniq):
                kz_u = code // (self._nbx*self._nby)
                rem  = code - kz_u*(self._nbx*self._nby)
                ky_u = rem // self._nbx
                kx_u = rem - ky_u*self._nbx
                key = (int(kx_u), int(ky_u), int(kz_u))
                blk = self.blocks.get(key)
                if blk is None: continue
                blk.ensure_ptr(B)
                sel   = (inv==u)
                pts   = out_idx_all[sel]
                li = I[sel] - blk.base[0]
                lj = J[sel] - blk.base[1]
                lk = K[sel] - blk.base[2]
                ok = (li>=0)&(li<B)&(lj>=0)&(lj<B)&(lk>=0)&(lk<B)
                if not np.any(ok): continue
                li = li[ok].astype(np.intp); lj = lj[ok].astype(np.intp); lk = lk[ok].astype(np.intp)
                pts2 = pts[ok]
                local_idx = blk.ptr[li, lj, lk]     # type: ignore
                ok2 = local_idx >= 0
                if not np.any(ok2): continue
                pts3 = pts2[ok2]
                ptr2 = local_idx[ok2].astype(np.int64)
                dt   = blk.psi.dtype
                Pdt  = P.astype(dt, copy=False)
                vjdt = vj.astype(dt, copy=False)
                psi  = blk.psi[ptr2]
                gg   = blk.g[ptr2]
                diff = Pdt[pts3] - vjdt[pts3]
                dloc = psi + np.einsum("ij,ij->i", diff, gg, optimize=True)
                d[pts3] = dloc.astype(np.float64, copy=False)
                g[pts3] = gg.astype(np.float64, copy=False)
                hit[pts3] = True

        if allow_fallback and (self._dense is not None):
            miss = inb & (~hit)
            if np.any(miss):
                d2, g2, *_ = taylor_query_batch_dense(self._dense, P[miss])
                d[miss] = d2
                g[miss] = g2
                hit[miss] = np.isfinite(d2)
        return d, g, vj, ijk, hit

    # ---------- I/O（.npz） ----------
    def save_npz(self, path: Union[str,Path], dtype: Optional[str] = None):
        """保存稀疏 .npz。若 dtype 指定则重铸（float32/64）。"""
        if dtype is None:
            # 以第一个块的 dtype 推断；默认 float32
            sample = None
            for blk in self.blocks.values():
                sample = blk.psi.dtype; break
            dtype = "float32" if (sample is None or sample==np.float32) else "float64"
        if dtype not in ("float32","float64"): raise ValueError("dtype must be float32/float64")
        dt = np.float32 if dtype=="float32" else np.float64

        keys, counts, locs_cat, psi_cat, g_cat = [], [], [], [], []
        for (kx,ky,kz), blk in self.blocks.items():
            keys.append([kx,ky,kz]); counts.append(int(blk.locs.shape[0]))
            locs_cat.append(blk.locs.astype(np.uint8, copy=False))
            psi_cat.append(blk.psi.astype(dt, copy=False))
            g_cat.append(blk.g.astype(dt, copy=False))
        keys = np.asarray(keys, dtype=np.int32) if keys else np.empty((0,3), dtype=np.int32)
        counts = np.asarray(counts, dtype=np.int32) if counts else np.empty((0,), dtype=np.int32)
        locs_cat = np.concatenate(locs_cat, axis=0) if len(locs_cat) else np.empty((0,3), dtype=np.uint8)
        psi_cat  = np.concatenate(psi_cat, axis=0)  if len(psi_cat)  else np.empty((0,), dtype=dt)
        g_cat    = np.concatenate(g_cat, axis=0)    if len(g_cat)    else np.empty((0,3), dtype=dt)

        np.savez_compressed(str(path),
            bmin=self.spec.bmin.astype(np.float64),
            bmax=self.spec.bmax.astype(np.float64),
            voxel_step=self.spec.voxel.astype(np.float64),
            shape=np.asarray(self.spec.shape, dtype=np.int32),
            block_size=np.asarray([self.block_size], dtype=np.int32),
            keys=keys, counts=counts, locs=locs_cat, psi=psi_cat, g=g_cat
        )

    @staticmethod
    def load_npz(path: Union[str,Path]) -> "HashedGradientSDF":
        with np.load(str(path), allow_pickle=False) as z:
            bmin = z["bmin"].astype(np.float64); bmax = z["bmax"].astype(np.float64)
            voxel = z["voxel_step"].astype(np.float64)
            shape = tuple(int(x) for x in z["shape"].astype(np.int32).tolist())
            block_size = int(z["block_size"][0])
            keys = z["keys"].astype(np.int32); counts = z["counts"].astype(np.int32)
            locs = z["locs"].astype(np.uint8); psi = z["psi"]; g = z["g"]
        spec = GridSpec(bmin=bmin, bmax=bmax, voxel=voxel, shape=shape)
        blocks: Dict[Tuple[int,int,int], _Block] = {}
        off = 0
        for (kx,ky,kz), c in zip(keys, counts):
            c = int(c)
            if c<=0: continue
            base = (int(kx)*block_size, int(ky)*block_size, int(kz)*block_size)
            locs_i = locs[off:off+c]; psi_i = psi[off:off+c]; g_i = g[off:off+c]
            blocks[(int(kx),int(ky),int(kz))] = _Block(base=base, locs=locs_i, psi=psi_i, g=g_i)
            off += c
        return HashedGradientSDF(spec=spec, blocks=blocks, block_size=block_size)


# ------------------------- 稠密一阶泰勒查询（供回退/对比） -------------------------

def taylor_query_batch_dense(vol: 'DenseGradientSDF', P: Array
                             ) -> Tuple[Array, Array, Array, Array, Array]:
    P = np.asarray(P, dtype=np.float64)
    ijk = vol.spec.index_from_world(P)
    inb = _in_bounds(ijk, vol.spec.shape)
    vj  = vol.spec.center_from_index(ijk)
    d   = np.full((len(P),),  np.nan, dtype=np.float64)
    g   = np.full((len(P),3), np.nan, dtype=np.float64)
    if np.any(inb):
        I = ijk[inb,0].astype(np.int64); J = ijk[inb,1].astype(np.int64); K = ijk[inb,2].astype(np.int64)
        psi = vol.sdf[I,J,K]
        gg  = vol.grad[I,J,K]
        diff = P[inb] - vj[inb]
        d_valid = psi + np.einsum("ij,ij->i", diff, gg, optimize=True)
        d[inb] = d_valid
        g[inb] = gg
    hit = np.isfinite(d)
    return d, g, vj, ijk, hit


# ------------------------- （可选）从 OBJ 生成稠密网格 -------------------------

try:
    import igl  # libigl
    _HAS_IGL = True
except Exception:
    _HAS_IGL = False

_EPS = 1e-12

def _parse_obj(path: Union[str,Path]) -> Tuple[np.ndarray, np.ndarray]:
    vs, fs = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"): continue
            sp = line.strip().split()
            if not sp: continue
            if sp[0] == "v" and len(sp) >= 4:
                vs.append([float(sp[1]), float(sp[2]), float(sp[3])])
            elif sp[0] == "f" and len(sp) >= 4:
                tri = []
                for t in sp[1:4]:
                    tri.append(int(t.split("/")[0]) - 1)
                fs.append(tri)
    V = np.asarray(vs, dtype=np.float64)
    F = np.asarray(fs, dtype=np.int32)
    return V, F

def _compute_bounds(V: np.ndarray, padding: float) -> Tuple[np.ndarray, np.ndarray]:
    vmin = V.min(axis=0); vmax = V.max(axis=0)
    diag = float(np.linalg.norm(vmax - vmin)); pad = padding * diag
    return (vmin - pad).astype(np.float64), (vmax + pad).astype(np.float64)

def _voxel_axes(bmin: np.ndarray, bmax: np.ndarray, voxel_size: Optional[float], target_resolution: Optional[int], max_resolution: int):
    size = (bmax - bmin).astype(np.float64)
    if voxel_size is None:
        if target_resolution is None: target_resolution = min(192, max_resolution)
        longest = float(size.max()); voxel = longest / float(target_resolution)
    else:
        voxel = float(voxel_size)
    nx, ny, nz = np.maximum(1, np.ceil(size / voxel).astype(int))
    xs = np.linspace(bmin[0]+0.5*voxel, bmin[0]+(nx-0.5)*voxel, nx, dtype=np.float64)
    ys = np.linspace(bmin[1]+0.5*voxel, bmin[1]+(ny-0.5)*voxel, ny, dtype=np.float64)
    zs = np.linspace(bmin[2]+0.5*voxel, bmin[2]+(nz-0.5)*voxel, nz, dtype=np.float64)
    return xs, ys, zs, np.array([voxel, voxel, voxel], dtype=np.float64)

def _grid_points(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)

def compute_sdf_grid(vertices: np.ndarray, faces: np.ndarray,
                     padding: float = 0.1,
                     voxel_size: Optional[float] = None,
                     target_resolution: Optional[int] = 192,
                     max_resolution: int = 512) -> Tuple[np.ndarray, GridSpec, np.ndarray]:
    """用 libigl.signed_distance 生成稠密 SDF（外正内负）与单位外法向"""
    if not _HAS_IGL:
        raise RuntimeError("需要 libigl（pip install igl）")
    bmin, bmax = _compute_bounds(vertices, padding)
    xs, ys, zs, vstep = _voxel_axes(bmin, bmax, voxel_size, target_resolution, max_resolution)
    pts = _grid_points(xs, ys, zs)

    Vd = np.ascontiguousarray(vertices, dtype=np.float64)
    Fi = np.ascontiguousarray(faces, dtype=np.int32)
    Q  = np.ascontiguousarray(pts, dtype=np.float64)

    # 优先 FWN 的 signed_distance
    if hasattr(igl, "SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER"):
        stype = igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    elif hasattr(igl, "SIGNED_DISTANCE_TYPE_WINDING_NUMBER"):
        stype = igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER
    else:
        stype = igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL

    S, I, C, N = igl.signed_distance(Q, Vd, Fi, stype)  # S: 外正内负
    S = np.asarray(S, dtype=np.float64).reshape(-1)
    C = np.asarray(C, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)

    pc   = pts - C
    norm = np.linalg.norm(pc, axis=1, keepdims=True)
    sgn  = np.sign(S).reshape(-1,1); sgn[sgn==0.0] = 1.0
    g = sgn * pc / np.maximum(norm, _EPS)
    near = (norm.reshape(-1) < 1e-9)
    if np.any(near):
        g[near] = sgn[near] * N[near]

    sdf_grid  = S.reshape(len(xs), len(ys), len(zs))
    grad_grid = g.reshape(len(xs), len(ys), len(zs), 3)
    spec = GridSpec(bmin=bmin, bmax=bmax, voxel=vstep, shape=sdf_grid.shape)
    return sdf_grid, spec, grad_grid


# ------------------------- 统一外部 API -------------------------

class GradientSDF:
    def __init__(self, core: HashedGradientSDF):
        self.core = core

    # ---- 构建 ----
    @classmethod
    def from_obj(cls, obj_path: Union[str,Path],
                 out_npz: Optional[Union[str,Path]] = None,
                 out_dense_prefix: Optional[Union[str,Path]] = None,
                 keep_dense_in_memory: bool = False,
                 tau: float = 10.0,
                 block_size: int = 8,
                 dtype: str = "float32",
                 dense_dtype: str = "float32",
                 padding: float = 0.1,
                 voxel_size: Optional[float] = None,
                 target_resolution: Optional[int] = 192,
                 max_resolution: int = 512) -> "GradientSDF":
        V, F = _parse_obj(obj_path)
        sdf_grid, spec, grad_grid = compute_sdf_grid(V, F,
                                                     padding=padding,
                                                     voxel_size=voxel_size,
                                                     target_resolution=target_resolution,
                                                     max_resolution=max_resolution)
        dense = DenseGradientSDF.__new__(DenseGradientSDF)
        dense.sdf = sdf_grid.astype(np.float64, copy=False)
        dense.grad = grad_grid.astype(np.float64, copy=False)
        dense.spec = spec

        core = HashedGradientSDF.build_from_dense(dense, tau=float(tau), block_size=int(block_size), dtype=dtype)
        g = cls(core)
        if out_npz is not None:
            g.save_npz(out_npz, dtype=dtype)
        if out_dense_prefix is not None:
            save_dense_triplet(out_dense_prefix, sdf_grid, grad_grid, spec, dtype=dense_dtype)
            keep_dense_in_memory = True
        if keep_dense_in_memory:
            g.core.attach_dense_fallback(dense)
        return g

    @classmethod
    def from_dense_prefix(cls, prefix: Union[str,Path],
                          tau: float = 10.0,
                          block_size: int = 8,
                          dtype: str = "float32",
                          save_npz: Optional[Union[str,Path]] = None,
                          attach_dense: bool = False) -> "GradientSDF":
        prefix = Path(prefix)
        dense = DenseGradientSDF(prefix.with_name(prefix.name + "_sdf.npy"),
                                 prefix.with_name(prefix.name + "_grad.npy"),
                                 prefix.with_name(prefix.name + "_meta.json"))
        core = HashedGradientSDF.build_from_dense(dense, tau=float(tau), block_size=int(block_size), dtype=dtype)
        g = cls(core)
        if attach_dense:
            g.core.attach_dense_fallback(dense)
        if save_npz is not None:
            g.save_npz(save_npz, dtype=dtype)
        return g

    @classmethod
    def from_npz(cls, path: Union[str,Path]) -> "GradientSDF":
        return cls(HashedGradientSDF.load_npz(path))

    # ---- 查询 ----
    def set_query_backend(self, mode: str = "index_map"):
        self.core.set_query_backend(mode)

    def warmup(self, *, index_map: bool = True, ptr_table: bool = False, atlas: bool = False):
        self.core.warmup(index_map=index_map, ptr_table=ptr_table, atlas=atlas)

    def query_points(self, P: Iterable[Iterable[float]], *, allow_fallback: bool = False
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        P = np.asarray(P, dtype=np.float64)
        d, n, *_ = self.core.taylor_query_batch(P, allow_fallback=allow_fallback)
        return d, n, _[-1]  # hit

    def query_point(self, p: Iterable[float], *, allow_fallback: bool = False
                    ) -> Tuple[float, np.ndarray, bool]:
        P = np.asarray(p, dtype=np.float64).reshape(1,3)
        d, n, hit = self.query_points(P, allow_fallback=allow_fallback)
        return float(d[0]), n[0], bool(hit[0])

    # ---- I/O ----
    def save_npz(self, path: Union[str,Path], dtype: Optional[str] = None):
        self.core.save_npz(path, dtype=dtype)

    def save_dense(self, prefix: Union[str,Path], dtype: str = "float32"):
        if self.core._dense is None:
            raise RuntimeError("当前实例未保留稠密数据，无法导出。请在构建时 keep_dense_in_memory=True 或 attach 稠密回退。")
        save_dense_triplet(prefix, self.core._dense.sdf, self.core._dense.grad, self.core.spec, dtype=dtype)

    # ---- 属性 ----
    @property
    def grid_spec(self) -> GridSpec:
        return self.core.spec
    @property
    def block_size(self) -> int:
        return self.core.block_size


# ------------------------- CLI（可选） -------------------------

def _cli():
    import argparse, sys, json as _json
    ap = argparse.ArgumentParser("Gradient SDF builder & query")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--obj", required=True)
    b.add_argument("--out_npz", default=None)
    b.add_argument("--out_dense_prefix", default=None)
    b.add_argument("--keep_dense_in_memory", action="store_true", default=False)
    b.add_argument("--tau", type=float, default=10.0)
    b.add_argument("--block_size", type=int, default=8)
    b.add_argument("--dtype", type=str, default="float32", choices=["float32","float64"])
    b.add_argument("--dense_dtype", type=str, default="float32", choices=["float32","float64"])
    b.add_argument("--padding", type=float, default=0.1)
    b.add_argument("--voxel_size", type=float, default=None)
    b.add_argument("--target_resolution", type=int, default=192)
    b.add_argument("--max_resolution", type=int, default=512)

    q = sub.add_parser("query")
    q.add_argument("--npz", required=True)
    q.add_argument("--backend", type=str, default="index_map", choices=["index_map","block_atlas","block_ptr"])
    q.add_argument("--fallback_prefix", type=str, default=None)

    args = ap.parse_args()
    if args.cmd == "build":
        g = GradientSDF.from_obj(args.obj,
                                 out_npz=args.out_npz,
                                 out_dense_prefix=args.out_dense_prefix,
                                 keep_dense_in_memory=args.keep_dense_in_memory,
                                 tau=args.tau, block_size=args.block_size,
                                 dtype=args.dtype, dense_dtype=args.dense_dtype,
                                 padding=args.padding, voxel_size=args.voxel_size,
                                 target_resolution=args.target_resolution, max_resolution=args.max_resolution)
        print(_json.dumps({"grid_shape": g.grid_spec.shape, "block_size": g.block_size}, ensure_ascii=False))
    else:
        g = GradientSDF.from_npz(args.npz)
        g.set_query_backend(args.backend)
        if args.fallback_prefix:
            pref = Path(args.fallback_prefix)
            dense = DenseGradientSDF(pref.with_name(pref.name + "_sdf.npy"),
                                     pref.with_name(pref.name + "_grad.npy"),
                                     pref.with_name(pref.name + "_meta.json"))
            g.core.attach_dense_fallback(dense)
        pts = []
        for line in sys.stdin:
            sp = line.strip().split()
            if len(sp)==3: pts.append([float(sp[0]), float(sp[1]), float(sp[2])])
        d, n, hit = g.query_points(pts, allow_fallback=bool(args.fallback_prefix))
        print(_json.dumps({"depth": d.tolist(), "normal": n.tolist(), "hit": [bool(x) for x in hit]}, ensure_ascii=False))

if __name__ == "__main__":
    try:
        _cli()
    except SystemExit:
        pass
