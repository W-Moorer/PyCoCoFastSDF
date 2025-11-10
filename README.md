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

PyCoCoFastSDF是一个高效而鲁棒的有符号距离场(SDF)计算工具库，专为三维网格模型设计。该库提供了两种主要实现：

1. **传统SDF (CoCoFastTraditionalSDF)**：提供多种计算后端，包括基于libigl的CPU实现和基于PyTorch3D的GPU加速实现
2. **梯度SDF (CoCoFastGradientSDF)**：不仅计算SDF值，还同时计算精确的梯度信息，支持更高效的查询和插值

项目采用先进的算法设计，默认使用FWN/SD + AABB后端，彻底避开"伪法线 + 角度阈值"的易错逻辑，从算法上解决齿轮等尖锐结构的"气泡"问题。同时支持GPU加速距离计算，大幅提升大规模网格的处理速度。

## 主要特性

### 传统SDF (CoCoFastTraditionalSDF)

- **多后端支持**：提供fwn_aabb（默认）、torch3d_fwn（GPU加速）和nn（传统方法）三种计算后端
- **高精度算法**：使用libigl的AABB树进行"真最近距离"计算，结合快速绕数算法(FWN)进行符号判定
- **GPU加速**：可选PyTorch3D后端，利用GPU并行计算能力加速大规模点集的距离计算
- **鲁棒性设计**：针对尖锐结构（如齿轮）优化，避免传统方法中的"气泡"问题

### 梯度SDF (CoCoFastGradientSDF)

- **精确梯度计算**：同时计算SDF值和单位外法向梯度(∇SDF)，提供更完整的几何信息
- **高效查询**：支持单体素泰勒展开的高效查询，比传统三线性插值更快更精确
- **稀疏哈希存储**：可选的稀疏哈希存储结构，大幅减少内存和磁盘占用
- **运行时优化**：提供专门的运行时库，支持大规模场景的高效查询

### 共同特性

- **可视化支持**：内置零等值面可视化和性能分析图表生成功能
- **批处理优化**：支持分批处理大规模数据，有效控制内存使用
- **多线程支持**：CPU后端支持多线程并行计算

## 算法说明

### 默认后端：fwn_aabb

默认后端采用两步计算策略：

1. **距离计算**：使用libigl的AABB树进行精确的最近距离计算，确保距离值的准确性
2. **符号判定**：
   - 优先使用libigl的signed_distance函数（FAST_WINDING_NUMBER/WINDING_NUMBER模式）
   - 若不可用，则使用快速绕数算法(FWN)计算绕数，并对0.5极窄带回退一次奇偶射线

这种设计彻底避免了传统方法中"伪法线 + 角度阈值"的易错逻辑，从算法层面解决了齿轮等尖锐结构的"气泡"问题。

### GPU加速后端：torch3d_fwn

GPU加速后端结合了PyTorch3D的GPU计算能力和libigl的符号判定：

1. **距离计算**：使用PyTorch3D的point_mesh_distance在GPU上并行计算点-网格距离
2. **符号判定**：仍使用libigl的signed_distance（FWN/WINDING模式），保证几何正确性

若GPU不可用或API不匹配，系统会自动回退到fwn_aabb（CPU）后端。

### 梯度SDF算法

梯度SDF在计算SDF值的同时，还计算每个体素点的梯度信息：

1. **SDF计算**：与传统SDF相同，计算有符号距离值
2. **梯度计算**：
   - 对于非边界点：梯度 = sgn(SDF) * (p - C) / ||p-C||，其中C是最近点
   - 对于近边界点：使用面法向作为梯度回退
3. **存储优化**：提供稀疏哈希存储，仅保存窄带内的体素数据

## 安装依赖

### 必需依赖

```bash
pip install numpy
pip install libigl
```

### 可选依赖

```bash
# GPU加速支持
pip install torch torchvision  # 需要CUDA版本
pip install pytorch3d

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

## 使用方法

### 传统SDF基本用法

```python
from CoCoFastTraditionalSDF.sdf_tools_gpu import parse_obj, compute_sdf_grid, visualize_zero_isosurface

# 1. 加载OBJ模型
V, F = parse_obj("path/to/your/model.obj")

# 2. 计算SDF网格
sdf, bounds, _, _, timings, voxel_step = compute_sdf_grid(
    V, F,
    padding=0.1,              # 边界填充比例
    voxel_size=None,          # 体素大小（留空则按分辨率推断）
    target_resolution=256,    # 目标分辨率
    max_resolution=512,       # 最大分辨率
    sdf_backend="fwn_aabb",   # 计算后端
    workers=-1                # 工作线程数（-1为自动检测）
)

# 3. 可视化零等值面
visualize_zero_isosurface(sdf, bounds, out_path="output_isosurface.png")
```

### 梯度SDF基本用法

```python
from CoCoFastGradientSDF.gradient_sdf_tools_cpu import parse_obj, compute_sdf_grid, visualize_zero_isosurface

# 1. 加载OBJ模型
V, F = parse_obj("path/to/your/model.obj")

# 2. 计算SDF和梯度网格
sdf, bounds, _, _, timings, voxel_step = compute_sdf_grid(
    V, F,
    padding=0.1,              # 边界填充比例
    voxel_size=None,          # 体素大小（留空则按分辨率推断）
    target_resolution=256,    # 目标分辨率
    max_resolution=512,       # 最大分辨率
    sdf_backend="auto",        # 自动选择最佳后端
    workers=-1                # 工作线程数（-1为自动检测）
)

# 3. 可视化零等值面
visualize_zero_isosurface(sdf, bounds, out_path="output_isosurface.png")

# 4. 梯度数据会自动保存为 {prefix}_grad.npy
```

### 命令行使用

```bash
# 传统SDF
python CoCoFastTraditionalSDF/sdf_tools_gpu.py \
    --obj path/to/model.obj \
    --out output_prefix \
    --backend fwn_aabb \
    --target_resolution 256 \
    --padding 0.1

# 梯度SDF
python CoCoFastGradientSDF/gradient_sdf_tools_cpu.py \
    --obj path/to/model.obj \
    --out output_prefix \
    --target_resolution 256 \
    --padding 0.1
```

### GPU加速使用

```python
# 自动选择最佳后端（优先GPU）
from CoCoFastTraditionalSDF.demo_sdf_tools_gpu_base_func import pick_backend

backend = pick_backend()  # 自动检测GPU支持
print(f"使用后端: {backend}")

sdf, bounds, _, _, timings, voxel_step = compute_sdf_grid(
    V, F,
    sdf_backend=backend,    # 使用自动选择的后端
    # ...其他参数
)
```

### 梯度SDF高效查询

```python
from CoCoFastGradientSDF.gradient_sdf_runtime import load_dense_by_prefix, HashedGradientSDF

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

## 环境变量配置

可以通过环境变量调整性能参数：

```bash
# FWN未决带（越小越少回退；建议1e-5～1e-4）
export SDF_FWN_TAU=1e-5

# AABB/FWN批大小（根据内存调整）
export SDF_AABB_BATCH=400000
export SDF_FWN_BATCH=2000000

# GPU距离一次送入的点数（根据显存调整）
export SDF_TORCH3D_POINTS_CHUNK=2000000
```

## 输出文件

### 传统SDF输出文件

1. **SDF数据**：`{prefix}_sdf.npy` - 三维有符号距离场数组
2. **元数据**：`{prefix}_meta.json` - 包含边界、体素步长、时间统计等信息
3. **零等值面图像**：`{prefix}_isosurface.png` - SDF零等值面的可视化图像
4. **性能分析图**：`{prefix}_timings_pie.png` - 各计算阶段耗时分析饼图

### 梯度SDF额外输出

5. **梯度数据**：`{prefix}_grad.npy` - 三维梯度场数组（单位外法向）
6. **哈希数据**（可选）：`{prefix}_hash.npz` - 稀疏哈希存储结构

## 示例

项目提供了完整的示例代码：

### 传统SDF示例

- `CoCoFastTraditionalSDF/demo_sdf_tools_gpu_base_func.py` - 基本使用示例
- `CoCoFastTraditionalSDF/demo_sdf_hausdorff_vis.py` - 豪斯多夫距离可视化
- `CoCoFastTraditionalSDF/demo_sdf_normal_error_heatmap.py` - 法向误差热图

### 梯度SDF示例

- `CoCoFastGradientSDF/demo_gradient_sdf_tools_cpu_base_func.py` - 基本使用示例
- `CoCoFastGradientSDF/demo_gradient_sdf_runtime.py` - 运行时查询示例
- `CoCoFastGradientSDF/benchmark_gsdf_vs_trilinear.py` - 梯度SDF与传统SDF性能对比

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

## 性能优化建议

1. **GPU加速**：对于大规模网格，优先使用torch3d_fwn后端
2. **批处理大小**：根据可用内存调整批处理大小，平衡内存使用和计算效率
3. **梯度SDF**：对于需要梯度信息的应用，直接使用梯度SDF而非后处理计算
4. **稀疏哈希**：对于大规模场景，使用稀疏哈希存储大幅减少内存占用
5. **数据类型**：根据精度需求选择float32或float64，平衡精度和内存使用

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