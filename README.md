<div style="text-align: center; margin-bottom: 20px;">
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

PyCoCoFastSDF是一个高效而鲁棒的有符号距离场(SDF)计算工具库，专为三维网格模型设计。该库提供了多种计算后端，包括基于libigl的CPU实现和基于PyTorch3D的GPU加速实现，能够快速准确地计算复杂几何体的有符号距离场。

项目采用先进的算法设计，默认使用FWN/SD + AABB后端，彻底避开"伪法线 + 角度阈值"的易错逻辑，从算法上解决齿轮等尖锐结构的"气泡"问题。同时支持GPU加速距离计算，大幅提升大规模网格的处理速度。

## 主要特性

- **多后端支持**：提供fwn_aabb（默认）、torch3d_fwn（GPU加速）和nn（传统方法）三种计算后端
- **高精度算法**：使用libigl的AABB树进行"真最近距离"计算，结合快速绕数算法(FWN)进行符号判定
- **GPU加速**：可选PyTorch3D后端，利用GPU并行计算能力加速大规模点集的距离计算
- **鲁棒性设计**：针对尖锐结构（如齿轮）优化，避免传统方法中的"气泡"问题
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

### 传统后端：nn

保留的传统后端，基于kNN最近中心方法，主要用于对比和参考。

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

### 基本用法

```python
from sdf_tools_gpu import parse_obj, compute_sdf_grid, visualize_zero_isosurface

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

### 命令行使用

```bash
python sdf_tools_gpu.py \
    --obj path/to/model.obj \
    --out output_prefix \
    --backend fwn_aabb \
    --target_resolution 256 \
    --padding 0.1
```

### GPU加速使用

```python
# 自动选择最佳后端（优先GPU）
from demo_sdf_tools_gpu_base_func import pick_backend

backend = pick_backend()  # 自动检测GPU支持
print(f"使用后端: {backend}")

sdf, bounds, _, _, timings, voxel_step = compute_sdf_grid(
    V, F,
    sdf_backend=backend,    # 使用自动选择的后端
    # ...其他参数
)
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

项目会生成以下输出文件：

1. **SDF数据**：`{prefix}_sdf.npy` - 三维有符号距离场数组
2. **元数据**：`{prefix}_meta.json` - 包含边界、体素步长、时间统计等信息
3. **零等值面图像**：`{prefix}_isosurface.png` - SDF零等值面的可视化图像
4. **性能分析图**：`{prefix}_timings_pie.png` - 各计算阶段耗时分析饼图

## 示例

项目提供了完整的示例代码，位于`demo_sdf_tools_gpu_base_func.py`，展示了如何：

1. 自动选择最佳计算后端
2. 处理OBJ模型并计算SDF
3. 生成可视化结果
4. 保存结果和元数据
5. 进行性能分析

## 性能优化建议

1. **GPU加速**：对于大规模网格，优先使用torch3d_fwn后端
2. **批处理大小**：根据可用内存调整批处理大小，平衡内存使用和计算效率
3. **多线程**：CPU后端充分利用多核并行计算
4. **分辨率选择**：根据实际需求选择合适的分辨率，避免不必要的计算开销

## 许可证

请参阅LICENSE文件了解项目的许可证信息。

## 贡献

欢迎提交问题报告和改进建议。