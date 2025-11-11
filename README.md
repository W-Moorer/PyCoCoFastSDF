<div align="center">
  <pre style="font-family: 'Courier New', monospace; 
              line-height: 1.2; 
              white-space: pre-wrap;
              display: inline-block;
              padding: 10px;
              border-radius: 4px;
              border: 1px solid #3b0aceff;">
$$$$$$$\             $$$$$$\             $$$$$$\                      
$$  __$$\           $$  __$$\           $$  __$$\                     
$$ |  $$ |$$\   $$\ $$ /  \__| $$$$$$\  $$ /  \__| $$$$$$\            
$$$$$$$  |$$ |  $$ |$$ |      $$  __$$\ $$ |      $$  __$$\           
$$  ____/ $$ |  $$ |$$ |      $$ /  $$ |$$ |      $$ /  $$ |          
$$ |      $$ |  $$ |$$ |  $$\ $$ |  $$ |$$ |  $$\ $$ |  $$ |          
$$ |      \$$$$$$$ |\$$$$$$  |\$$$$$$  |\$$$$$$  |\$$$$$$  |          
\__|       \____$$ | \______/  \______/  \______/  \______/           
          $$\   $$ |                                                  
          \$$$$$$  |                                                  
           \______/                                                   
$$$$$$$$\                       $$\      $$$$$$\  $$$$$$$\  $$$$$$$$\ 
$$  _____|                      $$ |    $$  __$$\ $$  __$$\ $$  _____|
$$ |       $$$$$$\   $$$$$$$\ $$$$$$\   $$ /  \__|$$ |  $$ |$$ |      
$$$$$\     \____$$\ $$  _____|\_$$  _|  \$$$$$$\  $$ |  $$ |$$$$$\    
$$  __|    $$$$$$$ |\$$$$$$\    $$ |     \____$$\ $$ |  $$ |$$  __|   
$$ |      $$  __$$ | \____$$\   $$ |$$\ $$\   $$ |$$ |  $$ |$$ |      
$$ |      \$$$$$$$ |$$$$$$$  |  \$$$$  |\$$$$$$  |$$$$$$$  |$$ |      
\__|       \_______|\_______/    \____/  \______/ \_______/ \__|      
                                                                      
                                                                      
                                                                      
  </pre>
</div>

# PyCoCoFastSDF

**作者：Wang Lijing**

## 简介

PyCoCoFastSDF是一个高效而鲁棒的有符号距离场(SDF)计算工具库，专为三维网格模型设计。该库提供了梯度SDF的完整实现，支持同时计算SDF值和精确的梯度信息，大幅提升查询效率和精度。

## 项目架构

### 核心模块：CoCoSDF

**CoCoSDF文件夹是本项目的正式可用核心模块**，包含了所有必要的功能和API。该模块提供了：

- **梯度SDF计算**：同时计算SDF值和单位外法向梯度(∇SDF)
- **高效查询**：支持单体素泰勒展开的高效查询，比传统三线性插值更快更精确
- **稀疏哈希存储**：可选的稀疏哈希存储结构，大幅减少内存和磁盘占用
- **运行时优化**：提供专门的运行时库，支持大规模场景的高效查询
- **多种查询后端**：支持index_map、block_ptr和block_atlas三种查询后端，可根据场景选择最优方案

### 验证模块：CoCoFastTraditionalSDF和CoCoFastGradientSDF

**CoCoFastTraditionalSDF和CoCoFastGradientSDF文件夹均为算法逻辑验证模块**，这些文件夹中的功能已全部整合至CoCoSDF文件夹中。这些验证模块主要用于：

- 算法原型验证
- 性能对比测试
- 不同实现方案的实验性探索

**实际使用时仅需关注和使用CoCoSDF文件夹**，验证模块仅作为参考和对比使用。

## 主要特性

### 梯度SDF核心功能

- **精确梯度计算**：同时计算SDF值和单位外法向梯度(∇SDF)，提供更完整的几何信息
- **高效查询**：支持单体素泰勒展开的高效查询，比传统三线性插值更快更精确
- **稀疏哈希存储**：可选的稀疏哈希存储结构，大幅减少内存和磁盘占用
- **运行时优化**：提供专门的运行时库，支持大规模场景的高效查询
- **多查询后端**：支持index_map、block_ptr和block_atlas三种查询后端，可根据场景选择最优方案

### 性能优化

- **CPU并行计算效率高于GPU并行计算**：经过测试验证，CPU并行计算模式在多数场景下表现更优，建议优先选择CPU并行计算模式
- **批处理优化**：支持分批处理大规模数据，有效控制内存使用
- **多线程支持**：CPU后端支持多线程并行计算

### 可视化与分析

- **可视化支持**：内置零等值面可视化和性能分析图表生成功能
- **性能分析**：提供详细的计算阶段耗时分析，帮助优化性能瓶颈

## 算法说明

### 梯度SDF算法

梯度SDF在计算SDF值的同时，还计算每个体素点的梯度信息：

1. **SDF计算**：使用libigl的AABB树进行精确的最近距离计算，结合快速绕数算法(FWN)进行符号判定
2. **梯度计算**：
   - 对于非边界点：梯度 = sgn(SDF) * (p - C) / ||p-C||，其中C是最近点
   - 对于近边界点：使用面法向作为梯度回退
3. **存储优化**：提供稀疏哈希存储，仅保存窄带内的体素数据

### 查询算法

支持三种查询后端：

1. **index_map**（默认）：全向量化O(1)查询，适合大多数场景
2. **block_ptr**：基于块指针的查询，内存占用更低
3. **block_atlas**：跨块全向量化O(1)查询，适合大规模场景

## 安装依赖

### 必需依赖

```bash
pip install numpy
pip install libigl
```

### 可选依赖

```bash
# 可视化功能
pip install matplotlib
pip install scikit-image

# 性能优化
pip install scipy
pip install numba

# 交互式可视化
pip install pyvista

# 进度条显示
pip install tqdm
```

### 环境建议

- 若遇到 `igl` 安装困难，优先尝试 Conda 环境。
- Windows 用户注意 VS 运行库；Linux/macOS 通常更顺畅。

## 使用方法

### 基本用法

```python
from CoCoSDF.gradient_sdf import parse_obj, compute_sdf_grid, visualize_zero_isosurface

# 1. 加载OBJ模型
V, F = parse_obj("path/to/your/model.obj")

# 2. 计算SDF和梯度网格
sdf, bounds, _, _, timings, voxel_step = compute_sdf_grid(
    V, F,
    padding=0.1,              # 边界填充比例
    voxel_size=None,          # 体素大小（留空则按分辨率推断）
    target_resolution=256,    # 目标分辨率
    max_resolution=512,       # 最大分辨率
    sdf_backend="auto",       # 自动选择最佳后端
    workers=-1                # 工作线程数（-1为自动检测）
)

# 3. 可视化零等值面
visualize_zero_isosurface(sdf, bounds, out_path="output_isosurface.png")

# 4. 梯度数据会自动保存为 {prefix}_grad.npy
```

### 命令行使用

```bash
python CoCoSDF/gradient_sdf.py \
    --obj path/to/model.obj \
    --out output_prefix \
    --target_resolution 256 \
    --padding 0.1
```

### 高效查询

```python
from CoCoSDF.gradient_sdf import load_dense_by_prefix, HashedGradientSDF

# 1. 加载已计算的SDF和梯度数据
vol = load_dense_by_prefix("path/to/prefix")

# 2. 构建稀疏哈希结构（可选，用于大规模场景）
hash_sdf = HashedGradientSDF.build_from_dense(
    vol, 
    tau=0.1,               # 窄带厚度
    block_size=8,          # 哈希块大小
    dtype="float32"        # 数据类型
)

# 3. 高效查询SDF值和梯度
points = np.array([[x1, y1, z1], [x2, y2, z2], ...])  # 查询点
sdf_values, gradients = hash_sdf.query(points)         # 批量查询
```

### 高级封装

```python
from CoCoSDF.gradient_sdf import GradientSDF

# 从OBJ直接构建
gsdf = GradientSDF.from_obj(
    "model.obj",
    out_dense_prefix="model_output",
    tau=10.0,
    block_size=8,
    target_resolution=256
)

# 查询
points = np.random.rand(1000, 3)
sdf_vals, normals, hits = gsdf.query_points(points)

# 保存
gsdf.save_npz("model_sparse.npz")
```

## 核心概念

### SDF 约定与梯度

- **符号约定**：内部为负、外部为正。
- **梯度（单位外法向）**：`∇SDF` 指向外侧，长度为 1（构建时已归一）。

### GridSpec 与体素索引

- `GridSpec` 定义边界 `bmin/bmax`、体素步长 `voxel` 与网格尺寸 `shape=(nx,ny,nz)`。
- 世界坐标 `p` → 体素索引 `ijk`：`index_from_world(p)`；体素中心坐标 `v_j`：`center_from_index(ijk)`。

### 稠密 vs. 稀疏（哈希）

- 稠密：完整三维数组保存 `ψ` 与 `ĝ`；查询直接索引。
- 稀疏（哈希）：只保存窄带（|ψ| ≤ τ）体素，以块（`block_size`³）组织；显著节省内存并加速批量查询。

### 查询后端

- `index_map`：全局线性编码 → 索引表（最快，一般默认）。
- `block_atlas`：块行指针 + 局部编码（大批量时表现稳定）。
- `block_ptr`：块内小稠密表（代码直观，便于调试）。

## API 参考

### `GradientSDFVolume`（稠密容器）

- **构造**：`GradientSDFVolume(sdf_npy, grad_npy=None, meta_json=None)`
- **方法**：
  - `taylor_query(p) -> (d, g, vj, ijk)`：单点查询（抛出越界异常）。
  - `taylor_query_batch(P) -> (d, g, vj, ijk, inb)`：批量查询（越界返回 `nan`）。
  - `nearest_surface_point_from_index(ijk)`：索引处最近表面点估计 `vj - ψ·ĝ`。

### `HashedGradientSDF`（哈希体素）

- **静态构造**：`build_from_dense(dense, tau=10.0, block_size=8, dtype="float32")`
- **查询**：`taylor_query_batch(P, allow_fallback=False)`
- **行为**：
  - `set_query_backend(mode)`：`index_map`/`block_atlas`/`block_ptr`
  - `warmup(index_map=True, ptr_table=False, atlas=False)`：懒构建加速结构
  - `attach_dense_fallback(dense)`：允许 miss 场景回退稠密

### `GradientSDF`（统一封装）

- **构造**：
  - `from_obj(obj_path, out_npz=None, out_dense_prefix=None, keep_dense_in_memory=False, tau=10.0, block_size=8, dtype="float32", padding=0.1, voxel_size=None, target_resolution=192, max_resolution=512, verbose=True)`
  - `from_dense_prefix(prefix, tau=10.0, block_size=8, dtype="float32", save_npz=None, attach_dense=False, verbose=True)`
  - `from_npz(path, verbose=True)`
- **查询**：
  - `query_points(P, allow_fallback=False) -> (d, n, hit)`
  - `query_point(p, allow_fallback=False) -> (d, n, hit_one)`
- **导出**：
  - `save_npz(path, dtype=None)`
  - `save_dense(prefix)`（需要在构建时保留稠密）
- **其他**：
  - `grid_spec`、`block_size` 属性

### 工具函数（tools 区）

- `parse_obj(path) -> (V, F)`：极简 OBJ（v/f 三角）解析。
- `compute_sdf_grid(V, F, ...) -> (sdf_grid, (bmin,bmax), centers, normals, timings, voxel_step)`：构建稠密 SDF 与梯度（单位外法向）。
- `save_sdf_and_meta(sdf_grid, bounds, obj_path, voxel_step, padding, timings, out_prefix)`：落盘三件套与可视化。
- `pyvista_visualize_isosurface(...)`：交互式可视化（需 `pyvista`）。
- 兼容：`taylor_query_batch_dense(vol, P)` 等价于 `vol.taylor_query_batch(P)`。

## 环境变量配置

可以通过环境变量调整性能参数：

```bash
# FWN未决带（越小越少回退；建议1e-5～1e-4）
export SDF_FWN_TAU=1e-5

# AABB/FWN批大小（根据内存调整）
export SDF_AABB_BATCH=400000
export SDF_FWN_BATCH=2000000
```

## 输出文件

### 标准输出文件

1. **SDF数据**：`{prefix}_sdf.npy` - 三维有符号距离场数组
2. **梯度数据**：`{prefix}_grad.npy` - 三维梯度场数组（单位外法向）
3. **元数据**：`{prefix}_meta.json` - 包含边界、体素步长、时间统计等信息
4. **零等值面图像**：`{prefix}_isosurface.png` - SDF零等值面的可视化图像
5. **性能分析图**：`{prefix}_timings_pie.png` - 各计算阶段耗时分析饼图

### 稀疏哈希输出

6. **哈希数据**（可选）：`{prefix}_hash.npz` - 稀疏哈希存储结构

### 文件格式

#### 稠密三件套

- `*_sdf.npy`：浮点 `(nx,ny,nz)`
- `*_grad.npy`：浮点 `(nx,ny,nz,3)`（单位外法向）
- `*_meta.json`：示例键：

  ```json
  {
    "obj": "<abs path>",
    "bmin": [x,y,z],
    "bmax": [x,y,z],
    "shape": [nx,ny,nz],
    "grad_shape": [nx,ny,nz,3],
    "voxel_step": [vx,vy,vz],
    "padding": 0.1,
    "timings": {"grid_setup": ..., "signed_distance": ..., "pack": ..., "total": ...},
    "has_grad": true
  }
  ```

#### 稀疏 `.npz` 格式

- 关键数组：
  - `bmin`, `bmax`, `voxel_step`, `shape`, `block_size`
  - `keys` `(num_blocks,3)`：块坐标 `(kx,ky,kz)`
  - `counts` `(num_blocks,)`：每块体素数
  - `locs` `(∑counts,3)`：块内 `uint8` 局部坐标
  - `psi` `(∑counts,)`、`g` `(∑counts,3)`：按块拼接的载荷

## 示例

项目提供了完整的示例代码：

### 核心功能示例

- `CoCoSDF/demo_gradient_sdf_tools_cpu_base_func.py` - 基本使用示例
- `CoCoSDF/demo_gradient_sdf_runtime.py` - 运行时查询示例
- `CoCoSDF/demo_gsdf_zero_isosurface.py` - 零等值面可视化示例
- `CoCoSDF/demo_benchmark_gsdf_vs_trilinear.py` - 梯度SDF与传统SDF性能对比
- `CoCoSDF/demo_gradient_sdf_all_in_one.py` - 完整功能演示

### 验证模块示例（仅供参考）

- `CoCoFastTraditionalSDF/demo_sdf_tools_gpu_base_func.py` - 传统SDF基本使用示例
- `CoCoFastTraditionalSDF/demo_sdf_hausdorff_vis.py` - 豪斯多夫距离可视化
- `CoCoFastTraditionalSDF/demo_sdf_normal_error_heatmap.py` - 法向误差热图
- `CoCoFastGradientSDF/demo_gradient_sdf_tools_cpu_base_func.py` - 梯度SDF验证示例
- `CoCoFastGradientSDF/demo_gradient_sdf_runtime.py` - 梯度SDF运行时验证
- `CoCoFastGradientSDF/benchmark_gsdf_vs_trilinear.py` - 性能对比验证

## 性能对比

### 存储效率

- **传统SDF**：仅存储距离值，内存占用 = N×N×N×4字节（float32）
- **梯度SDF**：存储距离值+梯度，内存占用 = N×N×N×16字节（float32）
- **梯度SDF（稀疏哈希）**：仅存储窄带内数据，内存占用可减少90%以上

### 查询效率

- **传统SDF（三线性）**：每次查询需8个体素值插值 + 解析梯度计算
- **梯度SDF（泰勒展开）**：每次查询仅需1个体素值 + 预计算梯度，速度提升2-3倍
- **梯度SDF（稀疏哈希）**：结合哈希查找，适合大规模场景

### 查询精度

- **传统SDF（三线性）**：值精度高，但梯度精度受限于插值误差
- **梯度SDF（泰勒展开）**：值精度与梯度精度均高，特别适合法向敏感的应用

## 性能与调优

### 后端选择

- **默认** `index_map`：全局线性编码反查，普适最快。
- `block_atlas`：批量点查询时表现稳定，减少二级寻址开销。
- `block_ptr`：更易调试，适合核对数据正确性。

### 参数建议

- `tau`（窄带阈值）：越大，哈希覆盖体素越多，命中率更高但稀疏文件更大；常用 5~15。
- `block_size`：常用 8；增大可降低块数量、提高连续性，但局部空洞增多时收益有限。
- `dtype`：`float32` 足以支撑多数实时/交互；需要更高精度时用 `float64`。
- 体素分辨率：优先 `target_resolution`（最长边目标体素数）或显式 `voxel_size`。

### 内存与时间估算

- 稠密内存 ~ `nx*ny*nz*(4+12)` 字节（float32 估算）或翻倍（float64）。
- 稀疏 `.npz` 大小 ≈ 窄带体素数 ×（按 `dtype` 计算的 `psi,g,locs` 载荷） + 元数据开销。

## 性能优化建议

1. **CPU并行计算优先**：CPU并行计算效率高于GPU并行计算，建议优先选择CPU并行计算模式
2. **批处理大小**：根据可用内存调整批处理大小，平衡内存使用和计算效率
3. **梯度SDF**：对于需要梯度信息的应用，直接使用梯度SDF而非后处理计算
4. **稀疏哈希**：对于大规模场景，使用稀疏哈希存储大幅减少内存占用
5. **查询后端选择**：根据场景特点选择合适的查询后端：
   - 一般场景：使用默认的index_map后端
   - 内存受限场景：使用block_ptr后端
   - 大规模场景：使用block_atlas后端
6. **数据类型**：根据精度需求选择float32或float64，平衡精度和内存使用

## 故障排查

- **ImportError: igl**：使用 `conda install -c conda-forge igl`；或在可用平台 `pip install igl`。
- **缺少 scikit-image/matplotlib**：仅影响可视化输出；数值结果不受影响。
- **越界查询**：稠密容器 `taylor_query` 抛越界；批量接口对应项返回 `nan`。请检查点是否在网格边界内或开启 `allow_fallback`。
- **窄带为空**：`tau` 太小；增大 `tau` 或提高分辨率。
- **meta 不一致**：确保三件套来自同一次构建，勿混用不同 OBJ/参数生成的文件。

## FAQ

**Q: SDF 的正负号定义？**
A: 外部为正、内部为负；梯度指向外侧并归一。

**Q: 查询为何采用"单体素泰勒"而非三线性插值？**
A: 该框架围绕体素中心的泰勒近似在窄带内具有良好稳定性，且结合哈希结构可获得更低常数的向量化查询性能。

**Q: 如何对比不同后端性能？**
A: 使用相同点集，分别 `set_query_backend(...)` + `warmup(...)`，计时 `taylor_query_batch` 即可。批量场景可优先尝试 `block_atlas`。

**Q: 我可以仅使用稀疏 `.npz` 而不保留稠密吗？**
A: 可以，`from_obj(..., out_npz=...)` 或 `from_npz(...)` 直接工作；若需要密集导出，再用 `save_dense`（前提是构建时保留了稠密）。

## 许可证

本项目采用MIT许可证，详情请参阅[LICENSE](LICENSE)文件。

## 贡献

欢迎贡献代码！主要贡献者包括：
- **Wang Lijing** - 项目负责人，核心算法实现
- ChatGPT - 辅助代码生成和优化
- GLM - 项目文档编写
- Trae CN - 项目测试和反馈

## 参考文献

- Gradient SDF: A Semi-Implicit Surface Representation for 3D Reconstruction (详见docs目录)