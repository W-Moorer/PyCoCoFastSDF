#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_viz_point_normal_pyvista_signfix.py
要点：
  • 自动判别 SDF 的“内/外”符号约定（统计体素多数一侧为“外部”）。
  • 若“外部为正”（常见约定）：penetration=max(0,-ds)，外法向 n_out = +∇ds/||∇ds||。
  • 若“外部为负”（你的数据很可能如此）：penetration=max(0, ds)，外法向 n_out = -∇ds/||∇ds||。
  • 统一用 n_raw = ∇ds/||∇ds|| 做零面投影：x <- x - ds * n_raw（对任意符号约定均成立）。
  • 三线性插值使用 cell-centered 半体素偏移 u=(x-bmin)/step - 0.5；contour 前做 cell→point。

用法：在顶部设置路径与 point_position，运行脚本。依赖：pyvista, vtk
"""

# ========== 在这里设置你的文件路径与测试点 ==========
SDF_NPY_FILE = "./traditional_outputs/gear_sdf.npy"
META_JSON_FILE = "./traditional_outputs/gear_meta.json"
SDF_BIN_FILE = None
SCREENSHOT = None
point_position = (22.6024, 0.347387, -1.64363)
# ===========================================

import json, numpy as np

def load_sdf_and_meta(npy_file, meta_file, bin_file=None):
    with open(meta_file, 'r', encoding='utf-8') as f:
        m = json.load(f)
    bmin = np.array(m["bounds_min"], dtype=float)
    bmax = np.array(m["bounds_max"], dtype=float)
    if "voxel_step" in m and m["voxel_step"] is not None:
        step = np.array(m["voxel_step"], dtype=float)
    else:
        step = None
    if bin_file is None and isinstance(m.get("binary"), dict):
        p = m["binary"].get("path")
        if isinstance(p, str) and len(p) > 0:
            bin_file = p
    if bin_file:
        with open(bin_file, 'rb') as f:
            hdr = f.read(58)
            nx = int.from_bytes(hdr[4:8], 'little')
            ny = int.from_bytes(hdr[8:12], 'little')
            nz = int.from_bytes(hdr[12:16], 'little')
            dtype_size = hdr[56]
            total = nx * ny * nz
            if dtype_size == 8:
                arr = np.frombuffer(f.read(total * 8), dtype=np.float64)
            elif dtype_size == 4:
                arr = np.frombuffer(f.read(total * 4), dtype=np.float32).astype(np.float64)
            else:
                raise RuntimeError("unsupported dtype_size")
        sdf = arr.reshape((nx, ny, nz), order='C')
    else:
        sdf = np.load(npy_file)
    if step is None:
        nx, ny, nz = sdf.shape
        step = (bmax - bmin) / np.array([nx, ny, nz], dtype=float)
    return sdf, bmin, bmax, step

def detect_sign_convention(sdf):
    neg_ratio = float(np.mean(sdf < 0.0))
    # 多数为负 ⇒ 外部为负；否则外部为正
    outside_positive = not (neg_ratio > 0.5)
    return outside_positive, neg_ratio

def trilinear_value_and_grad(sdf, bmin, step, x_world):
    """cell-centered 三线性：u=(x-bmin)/step - 0.5"""
    x = np.asarray(x_world, dtype=float)
    u = (x - bmin) / step - 0.5
    i, j, k = np.floor(u).astype(int)
    nx, ny, nz = sdf.shape
    i = int(np.clip(i, 0, nx-2)); j = int(np.clip(j, 0, ny-2)); k = int(np.clip(k, 0, nz-2))
    xi, yi, zi = (u - np.array([i, j, k], dtype=float)).tolist()

    def g(a,b,c): return sdf[i+a, j+b, k+c]
    d000,d100,d010,d110 = g(0,0,0), g(1,0,0), g(0,1,0), g(1,1,0)
    d001,d101,d011,d111 = g(0,0,1), g(1,0,1), g(0,1,1), g(1,1,1)

    wx0,wx1 = 1-xi, xi; wy0,wy1 = 1-yi, yi; wz0,wz1 = 1-zi, zi
    ds = (
        d000*wx0*wy0*wz0 + d100*wx1*wy0*wz0 + d010*wx0*wy1*wz0 + d110*wx1*wy1*wz0 +
        d001*wx0*wy0*wz1 + d101*wx1*wy0*wz1 + d011*wx0*wy1*wz1 + d111*wx1*wy1*wz1
    )
    dd_dxi = (
        (-1)*d000*wy0*wz0 + (+1)*d100*wy0*wz0 +
        (-1)*d010*wy1*wz0 + (+1)*d110*wy1*wz0 +
        (-1)*d001*wy0*wz1 + (+1)*d101*wy0*wz1 +
        (-1)*d011*wy1*wz1 + (+1)*d111*wy1*wz1
    )
    dd_deta = (
        (-1)*d000*wx0*wz0 + (-1)*d100*wx1*wz0 +
        (+1)*d010*wx0*wz0 + (+1)*d110*wx1*wz0 +
        (-1)*d001*wx0*wz1 + (-1)*d101*wx1*wz1 +
        (+1)*d011*wx0*wz1 + (+1)*d111*wx1*wz1
    )
    dd_dzeta = (
        (-1)*(d000*wx0*wy0 + d100*wx1*wy0 + d010*wx0*wy1 + d110*wx1*wy1) +
        (+1)*(d001*wx0*wy0 + d101*wx1*wy0 + d011*wx0*wy1 + d111*wx1*wy1)
    )
    grad = np.array([dd_dxi/step[0], dd_deta/step[1], dd_dzeta/step[2]], dtype=float)
    n_raw = grad / (np.linalg.norm(grad) + 1e-12)
    return float(ds), n_raw, grad

def project_to_surface(sdf, bmin, step, x0, iters=5):
    x = np.asarray(x0, dtype=float)
    ds, n_raw, grad = trilinear_value_and_grad(sdf, bmin, step, x)
    for _ in range(max(1, iters)):
        g = grad
        gn2 = float(np.dot(g, g))
        if gn2 <= 1e-20:
            break
        x1 = x - ds * g / gn2
        ds1, n_raw1, grad1 = trilinear_value_and_grad(sdf, bmin, step, x1)
        if ds * ds1 <= 0.0:
            a = x; va = ds; b = x1; vb = ds1
            for __ in range(24):
                m = (a + b) * 0.5
                vm, _, _ = trilinear_value_and_grad(sdf, bmin, step, m)
                if abs(vm) <= 1e-8:
                    x = m; ds = vm; grad = grad1
                    break
                if va * vm <= 0.0:
                    b = m; vb = vm
                else:
                    a = m; va = vm
            else:
                x = (a + b) * 0.5
                ds, grad = trilinear_value_and_grad(sdf, bmin, step, x)[0:3:2]
            break
        elif abs(ds1) < abs(ds):
            x = x1; ds = ds1; grad = grad1
        else:
            alpha = 0.5
            improved = False
            for __ in range(10):
                xt = x - alpha * ds * g / gn2
                dst, _, gradt = trilinear_value_and_grad(sdf, bmin, step, xt)
                if abs(dst) < abs(ds):
                    x = xt; ds = dst; grad = gradt
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                break
        if abs(ds) <= 1e-8:
            break
    return x

def _refine_tip_along_normal(sdf, bmin, step, x, n_out, ds, grad,
                             tol=1e-9, max_iter=32):
    import numpy as _np
    dirv = _np.asarray(n_out, dtype=float)
    f0 = float(ds)
    t0 = 0.0
    gnorm = float(_np.linalg.norm(grad))
    t1 = float(abs(ds)) / max(gnorm, 1e-12)
    f1, _n1, g1 = trilinear_value_and_grad(sdf, bmin, step, x + dirv * t1)
    if f0 * f1 <= 0.0:
        a, fa = t0, f0
        b, fb = t1, f1
        m = 0.5 * (a + b)
        for _ in range(max_iter):
            m = 0.5 * (a + b)
            fm, _nm, _gm = trilinear_value_and_grad(sdf, bmin, step, x + dirv * m)
            if abs(fm) <= tol or abs(b - a) <= tol:
                return m, x + dirv * m, fm
            if fa * fm <= 0.0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return m, x + dirv * m, fm
    else:
        t = t1
        for _ in range(max_iter):
            ft, _nt, gt = trilinear_value_and_grad(sdf, bmin, step, x + dirv * t)
            dd = float(_np.dot(gt, dirv))
            if abs(dd) <= 1e-14:
                break
            t_next = t - ft / dd
            if t_next < 0.0:
                t_next = 0.5 * t
            if abs(t_next - t) <= tol:
                t = t_next
                break
            t = t_next
            if abs(ft) <= tol:
                break
        ft_final, _n_final, _g_final = trilinear_value_and_grad(sdf, bmin, step, x + dirv * t)
        return t, x + dirv * t, ft_final

# ---- PyVista 可视化 ----
def make_grid_from_sdf(sdf, bmin, step):
    import pyvista as pv, numpy as _np
    nx, ny, nz = sdf.shape
    GridClass = pv.UniformGrid if hasattr(pv, 'UniformGrid') else getattr(pv, 'ImageData', None)
    if GridClass is None:
        raise AttributeError("PyVista 缺少 UniformGrid/ImageData")
    grid = GridClass()
    grid.dimensions = _np.array([nx, ny, nz], dtype=int) + 1
    grid.spacing = tuple(float(x) for x in step)
    grid.origin = tuple(float(x) for x in bmin)
    grid.cell_data["values"] = sdf.ravel(order="F")
    grid = grid.cell_data_to_point_data()
    return grid

def visualize(sdf, bmin, bmax, step, ds, n_out, x, x_proj, inside, outside_positive, grad_norm, screenshot=None):
    import pyvista as pv, numpy as _np
    grid = make_grid_from_sdf(sdf, bmin, step)
    surf = grid.contour([0.0], scalars="values")
    plotter = pv.Plotter(window_size=[1100, 850])
    plotter.add_mesh(surf, opacity=0.7, smooth_shading=True)

    spacing = (bmax - bmin) / _np.array(sdf.shape, dtype=float)
    lunit = float(_np.min(spacing))
    x = _np.array(x, float); x_proj = _np.array(x_proj, float)

    color_in = "red" if inside else "lime"
    plotter.add_mesh(pv.Sphere(radius=0.8*lunit, center=x), color=color_in, name="query_point")
    plotter.add_mesh(pv.Sphere(radius=0.1*lunit, center=x_proj), color="cyan", name="proj_on_iso")
    plotter.add_mesh(pv.Line(x, x_proj), line_width=3, color="white", name="line_to_iso")
    _ds0, _n0, _g0 = trilinear_value_and_grad(sdf, bmin, step, x)
    t_star, x_tip, tip_val = _refine_tip_along_normal(sdf, bmin, step, x, n_out, ds, _g0)
    depth = float(t_star) if inside else 0.0
    dir_vec = x_tip - x
    dir_len = float(_np.linalg.norm(dir_vec))
    dir_hat = dir_vec / (dir_len + 1e-12)
    plotter.add_mesh(pv.Arrow(start=x, direction=dir_hat, scale=max(dir_len, 1e-9)), color="yellow", name="normal_out")
    plotter.add_mesh(pv.Sphere(radius=0.4*lunit, center=x_tip), color="white", name="arrow_tip")
    ds_tip, n_raw_tip, _g_tip = trilinear_value_and_grad(sdf, bmin, step, x_tip)
    n_out_tip = (n_raw_tip if outside_positive else -n_raw_tip)
    plotter.add_mesh(pv.Arrow(start=x_tip, direction=n_out_tip, scale=3.0*lunit), color="orange", name="normal_at_tip")

    txt = (f"SDF convention: outside {'positive' if outside_positive else 'negative'}\n"
           f"ds = {ds:.6g} | penetration(normal) = {depth:.6g} | SDF(tip) = {tip_val:.3e}\n"
           f"n_tip = [{n_out_tip[0]:.4g}, {n_out_tip[1]:.4g}, {n_out_tip[2]:.4g}]  "
           f"({'inside' if inside else 'outside'})")
    plotter.add_text(txt, font_size=12)
    plotter.add_axes()
    plotter.show_bounds(grid='front', location='outer', all_edges=True)
    plotter.view_isometric()

    if screenshot:
        plotter.screenshot(screenshot)
        print(f"[saved] {screenshot}")
        plotter.close()
    else:
        plotter.show()

def main():
    sdf, bmin, bmax, step = load_sdf_and_meta(SDF_NPY_FILE, META_JSON_FILE, SDF_BIN_FILE)

    outside_positive, neg_ratio = detect_sign_convention(sdf)
    print(f"[Sign detection] outside_positive={outside_positive} (neg_ratio={neg_ratio:.3f})")

    xq = point_position if point_position is not None else tuple(((bmin + bmax) * 0.5).tolist())
    ds, n_raw, grad = trilinear_value_and_grad(sdf, bmin, step, xq)

    # 外法向：根据约定选择方向
    n_out = (n_raw if outside_positive else -n_raw)

    inside = (ds < 0.0) if outside_positive else (ds > 0.0)
    t_star, x_tip, tip_val = _refine_tip_along_normal(sdf, bmin, step, np.array(xq, dtype=float), n_out, ds, grad)
    depth = float(t_star) if inside else 0.0

    print(f"[Point] {xq}")
    print(f"[SDF] ds = {ds:.6g}  |  penetration = {depth:.6g}")
    print(f"[Grad] {grad}")
    print(f"[n_raw] {n_raw}")
    print(f"[n_out] {n_out}  | inside={inside}")

    ds_tip, n_raw_tip, _gtip = trilinear_value_and_grad(sdf, bmin, step, x_tip)
    n_out_tip = (n_raw_tip if outside_positive else -n_raw_tip)
    print(f"[Tip] SDF(tip) = {tip_val:.3e}  |  penetration = {depth:.6g}")
    print(f"[n_out_tip] {n_out_tip}")

    x_proj = project_to_surface(sdf, bmin, step, xq, iters=5)

    try:
        visualize(sdf, bmin, bmax, step, ds, n_out, xq, x_proj, inside, outside_positive, grad_norm=float(np.linalg.norm(grad)), screenshot=SCREENSHOT)
    except Exception as e:
        print("PyVista 可视化失败：", e)
        print("请检查环境或安装：conda install -c conda-forge pyvista vtk  或  pip install -U pyvista vtk")

if __name__ == "__main__":
    main()
