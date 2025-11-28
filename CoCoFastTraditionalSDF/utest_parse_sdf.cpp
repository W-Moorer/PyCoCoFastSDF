#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <random>

struct SdfHeader {
    char magic[4];
    uint32_t nx, ny, nz;
    float bmin[3], bmax[3], voxel[3];
    float padding;
    uint8_t dtype_size;
    uint8_t order;
};

bool load_sdf_bin(const std::string& path, SdfHeader& h, std::vector<double>& data) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(h.magic, 4);
    if (std::string(h.magic, 4) != "SDFB") return false;
    f.read(reinterpret_cast<char*>(&h.nx), 4);
    f.read(reinterpret_cast<char*>(&h.ny), 4);
    f.read(reinterpret_cast<char*>(&h.nz), 4);
    f.read(reinterpret_cast<char*>(h.bmin), sizeof(float)*3);
    f.read(reinterpret_cast<char*>(h.bmax), sizeof(float)*3);
    f.read(reinterpret_cast<char*>(h.voxel), sizeof(float)*3);
    f.read(reinterpret_cast<char*>(&h.padding), sizeof(float));
    f.read(reinterpret_cast<char*>(&h.dtype_size), 1);
    f.read(reinterpret_cast<char*>(&h.order), 1);
    if (h.dtype_size != 8 || h.order != 0) return false;
    size_t total = size_t(h.nx) * size_t(h.ny) * size_t(h.nz);
    data.resize(total);
    f.read(reinterpret_cast<char*>(data.data()), total * sizeof(double));
    if (!f) return false;
    return true;
}

inline size_t idx3(size_t i, size_t j, size_t k, const SdfHeader& h) {
    return (i * h.ny + j) * h.nz + k;
}

inline double sdf_at_ijk(const std::vector<double>& d, const SdfHeader& h,
                        uint32_t i, uint32_t j, uint32_t k) {
    return d[idx3(i,j,k,h)];
}



inline bool world_to_cell(const SdfHeader& h, const double p[3],
                          int& i0, int& j0, int& k0, double w[3]) {
    double fx = (p[0] - h.bmin[0]) / h.voxel[0] - 0.5;
    double fy = (p[1] - h.bmin[1]) / h.voxel[1] - 0.5;
    double fz = (p[2] - h.bmin[2]) / h.voxel[2] - 0.5;
    i0 = int(std::floor(fx)); j0 = int(std::floor(fy)); k0 = int(std::floor(fz));
    w[0] = fx - i0; w[1] = fy - j0; w[2] = fz - k0;
    if (i0 < 0 || j0 < 0 || k0 < 0) return false;
    if (i0+1 >= int(h.nx) || j0+1 >= int(h.ny) || k0+1 >= int(h.nz)) return false;
    return true;
}

inline double trilinear(const std::vector<double>& d, const SdfHeader& h,
                       int i0, int j0, int k0, const double w[3]) {
    int i1 = i0 + 1, j1 = j0 + 1, k1 = k0 + 1;
    double c000 = sdf_at_ijk(d,h,i0,j0,k0), c100 = sdf_at_ijk(d,h,i1,j0,k0);
    double c010 = sdf_at_ijk(d,h,i0,j1,k0), c110 = sdf_at_ijk(d,h,i1,j1,k0);
    double c001 = sdf_at_ijk(d,h,i0,j0,k1), c101 = sdf_at_ijk(d,h,i1,j0,k1);
    double c011 = sdf_at_ijk(d,h,i0,j1,k1), c111 = sdf_at_ijk(d,h,i1,j1,k1);
    double wx = w[0], wy = w[1], wz = w[2];
    double c00 = c000*(1-wx) + c100*wx;
    double c01 = c001*(1-wx) + c101*wx;
    double c10 = c010*(1-wx) + c110*wx;
    double c11 = c011*(1-wx) + c111*wx;
    double c0 = c00*(1-wy) + c10*wy;
    double c1 = c01*(1-wy) + c11*wy;
    return c0*(1-wz) + c1*wz;
}

inline void gradient_at_grid(const std::vector<double>& d, const SdfHeader& h,
                             int i, int j, int k, double g[3]) {
    double dvx;
    if (i > 0 && i + 1 < int(h.nx)) dvx = (sdf_at_ijk(d,h,i+1,j,k) - sdf_at_ijk(d,h,i-1,j,k)) / (2.0 * h.voxel[0]);
    else if (i + 1 < int(h.nx)) dvx = (sdf_at_ijk(d,h,i+1,j,k) - sdf_at_ijk(d,h,i,j,k)) / h.voxel[0];
    else if (i > 0) dvx = (sdf_at_ijk(d,h,i,j,k) - sdf_at_ijk(d,h,i-1,j,k)) / h.voxel[0];
    else dvx = 0.0;
    double dvy;
    if (j > 0 && j + 1 < int(h.ny)) dvy = (sdf_at_ijk(d,h,i,j+1,k) - sdf_at_ijk(d,h,i,j-1,k)) / (2.0 * h.voxel[1]);
    else if (j + 1 < int(h.ny)) dvy = (sdf_at_ijk(d,h,i,j+1,k) - sdf_at_ijk(d,h,i,j,k)) / h.voxel[1];
    else if (j > 0) dvy = (sdf_at_ijk(d,h,i,j,k) - sdf_at_ijk(d,h,i,j-1,k)) / h.voxel[1];
    else dvy = 0.0;
    double dvz;
    if (k > 0 && k + 1 < int(h.nz)) dvz = (sdf_at_ijk(d,h,i,j,k+1) - sdf_at_ijk(d,h,i,j,k-1)) / (2.0 * h.voxel[2]);
    else if (k + 1 < int(h.nz)) dvz = (sdf_at_ijk(d,h,i,j,k+1) - sdf_at_ijk(d,h,i,j,k)) / h.voxel[2];
    else if (k > 0) dvz = (sdf_at_ijk(d,h,i,j,k) - sdf_at_ijk(d,h,i,j,k-1)) / h.voxel[2];
    else dvz = 0.0;
    g[0] = dvx; g[1] = dvy; g[2] = dvz;
}

inline bool sdf_value_and_grad_world(const std::vector<double>& d, const SdfHeader& h,
                                     const double p[3], double& val, double g[3]) {
    int i0,j0,k0;
    double wd[3];
    if (!world_to_cell(h, p, i0, j0, k0, wd)) return false;
    int i1 = i0 + 1, j1 = j0 + 1, k1 = k0 + 1;
    double d000 = sdf_at_ijk(d,h,i0,j0,k0);
    double d100 = sdf_at_ijk(d,h,i1,j0,k0);
    double d010 = sdf_at_ijk(d,h,i0,j1,k0);
    double d110 = sdf_at_ijk(d,h,i1,j1,k0);
    double d001 = sdf_at_ijk(d,h,i0,j0,k1);
    double d101 = sdf_at_ijk(d,h,i1,j0,k1);
    double d011 = sdf_at_ijk(d,h,i0,j1,k1);
    double d111 = sdf_at_ijk(d,h,i1,j1,k1);
    double xi = wd[0], yi = wd[1], zi = wd[2];
    double wx0 = 1.0 - xi, wx1 = xi;
    double wy0 = 1.0 - yi, wy1 = yi;
    double wz0 = 1.0 - zi, wz1 = zi;
    val = (
        d000*wx0*wy0*wz0 + d100*wx1*wy0*wz0 + d010*wx0*wy1*wz0 + d110*wx1*wy1*wz0 +
        d001*wx0*wy0*wz1 + d101*wx1*wy0*wz1 + d011*wx0*wy1*wz1 + d111*wx1*wy1*wz1
    );
    double dd_dxi = (
        (-1.0)*d000*wy0*wz0 + (+1.0)*d100*wy0*wz0 +
        (-1.0)*d010*wy1*wz0 + (+1.0)*d110*wy1*wz0 +
        (-1.0)*d001*wy0*wz1 + (+1.0)*d101*wy0*wz1 +
        (-1.0)*d011*wy1*wz1 + (+1.0)*d111*wy1*wz1
    );
    double dd_deta = (
        (-1.0)*d000*wx0*wz0 + (-1.0)*d100*wx1*wz0 +
        (+1.0)*d010*wx0*wz0 + (+1.0)*d110*wx1*wz0 +
        (-1.0)*d001*wx0*wz1 + (-1.0)*d101*wx1*wz1 +
        (+1.0)*d011*wx0*wz1 + (+1.0)*d111*wx1*wz1
    );
    double dd_dzeta = (
        (-1.0)*(d000*wx0*wy0 + d100*wx1*wy0 + d010*wx0*wy1 + d110*wx1*wy1) +
        (+1.0)*(d001*wx0*wy0 + d101*wx1*wy0 + d011*wx0*wy1 + d111*wx1*wy1)
    );
    g[0] = dd_dxi / h.voxel[0];
    g[1] = dd_deta / h.voxel[1];
    g[2] = dd_dzeta / h.voxel[2];
    return true;
}

inline void normalize3(double v[3]) {
    double n = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (n > 0) { v[0] /= n; v[1] /= n; v[2] /= n; }
}

inline bool detect_outside_positive(const std::vector<double>& d) {
    size_t neg = 0;
    for (size_t i = 0; i < d.size(); ++i) if (d[i] < 0.0f) ++neg;
    float neg_ratio = d.empty() ? 0.0f : float(neg) / float(d.size());
    return !(neg_ratio > 0.5f);
}

inline bool refine_tip_along_normal(
    const std::vector<double>& data,
    const SdfHeader& h,
    const double x[3],
    const double n_out[3],
    double ds,
    const double grad[3],
    double tol,
    int max_iter,
    double& t_star,
    double tip[3],
    double& tip_val
) {
    double dir[3] = { n_out[0], n_out[1], n_out[2] };
    normalize3(dir);
    double gnorm = std::sqrt(grad[0]*grad[0] + grad[1]*grad[1] + grad[2]*grad[2]);
    if (gnorm <= 1e-12) { t_star = 0.0; tip[0]=x[0]; tip[1]=x[1]; tip[2]=x[2]; tip_val = ds; return false; }
    double t0 = 0.0;
    double t1 = std::fabs(ds) / gnorm;
    double xt1[3] = { x[0] + dir[0]*t1, x[1] + dir[1]*t1, x[2] + dir[2]*t1 };
    double f1; double g1[3];
    if (!sdf_value_and_grad_world(data, h, xt1, f1, g1)) { return false; }
    double f0 = ds;
    if (f0 * f1 <= 0.0) {
        double a = t0, fa = f0;
        double b = t1, fb = f1;
        for (int it = 0; it < max_iter; ++it) {
            double m = 0.5 * (a + b);
            double xm[3] = { x[0] + dir[0]*m, x[1] + dir[1]*m, x[2] + dir[2]*m };
            double fm; double gm[3];
            if (!sdf_value_and_grad_world(data, h, xm, fm, gm)) break;
            if (std::fabs(fm) <= tol || std::fabs(b - a) <= tol) { t_star = m; tip[0]=xm[0]; tip[1]=xm[1]; tip[2]=xm[2]; tip_val = fm; return true; }
            if (fa * fm <= 0.0) { b = m; fb = fm; } else { a = m; fa = fm; }
        }
        double m = 0.5 * (a + b);
        double xm[3] = { x[0] + dir[0]*m, x[1] + dir[1]*m, x[2] + dir[2]*m };
        double fm; double gm[3];
        sdf_value_and_grad_world(data, h, xm, fm, gm);
        t_star = m; tip[0]=xm[0]; tip[1]=xm[1]; tip[2]=xm[2]; tip_val = fm; return true;
    } else {
        double t = t1;
        for (int it = 0; it < max_iter; ++it) {
            double xt[3] = { x[0] + dir[0]*t, x[1] + dir[1]*t, x[2] + dir[2]*t };
            double ft; double gt[3];
            if (!sdf_value_and_grad_world(data, h, xt, ft, gt)) break;
            double dd = gt[0]*dir[0] + gt[1]*dir[1] + gt[2]*dir[2];
            if (std::fabs(dd) <= 1e-14) break;
            double t_next = t - ft / dd;
            if (t_next < 0.0) t_next = 0.5 * t;
            if (std::fabs(t_next - t) <= tol) { t = t_next; break; }
            t = t_next;
            if (std::fabs(ft) <= tol) break;
        }
        double xt[3] = { x[0] + dir[0]*t, x[1] + dir[1]*t, x[2] + dir[2]*t };
        double ft; double gt[3];
        sdf_value_and_grad_world(data, h, xt, ft, gt);
        t_star = t; tip[0]=xt[0]; tip[1]=xt[1]; tip[2]=xt[2]; tip_val = ft; return true;
    }
}

// 示例：点接触判断
inline bool is_in_contact(const std::vector<double>& d, const SdfHeader& h,
                          const double p[3], double threshold /* e.g. 0 */) {
    double val; double g[3];
    if (!sdf_value_and_grad_world(d,h,p,val,g)) return false;
    return val >= threshold;
}

int main()
{
    std::string path = "e:\\CoCoSim\\data\\input\\sdf_library\\gear_sdf.sdf";
    SdfHeader h{};
    std::vector<double> d;
    bool ok = load_sdf_bin(path, h, d);
    if (!ok) {
        std::cout << "读取失败: " << path << "\n";
        return 1;
    }
    std::cout << "magic " << std::string(h.magic,4)
              << " dims " << h.nx << " " << h.ny << " " << h.nz << "\n";
    std::cout << "bmin " << h.bmin[0] << " " << h.bmin[1] << " " << h.bmin[2] << "\n";
    std::cout << "bmax " << h.bmax[0] << " " << h.bmax[1] << " " << h.bmax[2] << "\n";
    std::cout << "voxel " << h.voxel[0] << " " << h.voxel[1] << " " << h.voxel[2] << "\n";
    std::cout << "dtype_size " << int(h.dtype_size) << " order " << int(h.order) << "\n";
    double v0 = d.empty() ? 0.0 : d[0];
    uint32_t ci = h.nx ? h.nx/2 : 0;
    uint32_t cj = h.ny ? h.ny/2 : 0;
    uint32_t ck = h.nz ? h.nz/2 : 0;
    double vc = d.empty() ? 0.0 : sdf_at_ijk(d, h, ci, cj, ck);
    double vmin = d.empty() ? 0.0 : d[0];
    double vmax = d.empty() ? 0.0 : d[0];
    for (size_t i = 1; i < d.size(); ++i) {
        if (d[i] < vmin) vmin = d[i];
        if (d[i] > vmax) vmax = d[i];
    }
    std::cout << "sample v0 " << v0 << " center " << vc << "\n";
    std::cout << "min " << vmin << " max " << vmax << "\n";
    bool outside_positive = detect_outside_positive(d);
    std::cout << "outside_positive " << (outside_positive?1:0) << "\n";
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> dx(h.bmin[0], h.bmax[0]);
    std::uniform_real_distribution<double> dy(h.bmin[1], h.bmax[1]);
    std::uniform_real_distribution<double> dz(h.bmin[2], h.bmax[2]);
    for (int t = 0; t < 20; ++t) {
        double p[3] = { dx(rng), dy(rng), dz(rng) };
        double sdf; double gtmp[3];
        if (!sdf_value_and_grad_world(d,h,p,sdf,gtmp)) continue;
        double n_raw[3] = { gtmp[0], gtmp[1], gtmp[2] }; normalize3(n_raw);
        double n_out0[3] = { n_raw[0], n_raw[1], n_raw[2] };
        if (!outside_positive) { n_out0[0] = -n_out0[0]; n_out0[1] = -n_out0[1]; n_out0[2] = -n_out0[2]; }
        bool inside = outside_positive ? (sdf < 0.0) : (sdf > 0.0);
        if (!inside) {
            std::cout << "p " << p[0] << " " << p[1] << " " << p[2] << " sdf " << sdf << " contact 0\n";
        } else {
            double t_star, tip[3], tip_val;
            bool ok = refine_tip_along_normal(d, h, p, n_out0, sdf, gtmp, 1e-8, 32, t_star, tip, tip_val);
            double n_tip_raw[3];
            sdf_value_and_grad_world(d, h, tip, tip_val, n_tip_raw);
            normalize3(n_tip_raw);
            double n_out_tip[3] = { n_tip_raw[0], n_tip_raw[1], n_tip_raw[2] };
            if (!outside_positive) { n_out_tip[0] = -n_out_tip[0]; n_out_tip[1] = -n_out_tip[1]; n_out_tip[2] = -n_out_tip[2]; }
            double depth = t_star;
            std::cout << "p " << p[0] << " " << p[1] << " " << p[2]
                      << " sdf " << sdf << " contact 1 depth " << depth
                      << " tip_sdf " << tip_val
                      << " n " << n_out_tip[0] << " " << n_out_tip[1] << " " << n_out_tip[2] << "\n";
        }
    }
    return 0;
}
