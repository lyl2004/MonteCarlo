# MonteCarlo

`MonteCarlo` 是一个面向云雾/气溶胶光散射问题的异构仿真工作台。项目通过 GUI 统一调度两条后端：

- `Mie + Monte Carlo + Numba`
  - 面向球形粒子和等效粒径分布。
  - 适合快速基线、参数扫描、工程诊断。
- `IITM/T-Matrix + Monte Carlo + Julia`
  - 面向 `sphere / cylinder / spheroid`。
  - 适合非球形粒子、solver 诊断和更严格的散射物理分析。

当前日期状态：`2026-04-27`。

## 当前状态

- GUI 已支持 `mie` 和 `iitm` 两个后端切换。
- 两套后端均导出同构数据包：`density.npz + render_main.html + render_top.html + render_front.html`。
- 两套后端均支持场族：
  - `proxy`：代理场，用于快速预览、参数扫描和工程诊断。
  - `exact`：detector-conditioned response field，由 Monte Carlo 散射事件处的探测锥响应累积得到。
- `field_compute_mode` 支持：
  - `proxy_only`
  - `exact_only`
  - `both`
- 后端运行结果会返回：
  - `field_catalog`
  - `available_field_families`
  - `requested_field_compute_mode`
  - `effective_field_compute_mode`
  - `primary_field_family`

## 后端能力矩阵

| 能力 | Mie 后端 | Julia/IITM 后端 |
| --- | --- | --- |
| 球形粒子散射 | 支持 | 支持 |
| 非球形粒子散射 | 不支持 | 支持 `cylinder / spheroid / sphere` |
| 对数正态粒径聚合 | 支持 | 支持 |
| Mueller 偏振表 | 支持 | 支持 |
| Monte Carlo 传输 | Numba | Julia threads |
| 3D density-aware MC | 支持 | 支持 |
| z-slab majorant | 支持 | 支持 |
| `proxy` 三维场 | 支持 | 支持 |
| `exact` 三维场 | 支持 | 支持 |
| `field_compute_mode=both` | 支持 | 支持 |
| 输出格式 | NPZ + 浏览器 Plotly | NPZ + 浏览器 Plotly |
| solver 诊断 | 不适用 | 支持 |

## 目录结构

```text
MonteCarlo/
├─ inputs/
│  ├─ mie/
│  └─ iitm/
├─ outputs/
│  ├─ mie/
│  └─ iitm/
├─ temp/
├─ log/
├─ tests/
│  ├─ test_mie_contracts.py
│  ├─ julia_field_contract.jl
│  └─ julia_server_contract.jl
├─ src/
│  ├─ gui.py
│  ├─ mie_core.py
│  ├─ mie_numba.py
│  ├─ mie_worker.py
│  ├─ iitm_http_worker.py
│  ├─ iitm_renderer.py
│  └─ julia/
│     ├─ iitm_physics.jl
│     ├─ iitm_server.jl
│     ├─ Project.toml
│     └─ Manifest.toml
├─ pixi.toml
├─ pixi.lock
├─ Readme.md
└─ todo.md
```

## GUI 调度流程

GUI 主文件是 `src/gui.py`。

1. 用户选择后端：`mie` 或 `iitm`。
2. GUI 使用统一配置表展示参数。
3. 项目配置保存到：
   - `inputs/mie/<project>.json`
   - `inputs/iitm/<project>.json`
4. 点击运行后，GUI 根据后端启动 worker：
   - Mie：`pixi run -e mie python -u src/mie_worker.py ...`
   - Julia/IITM：`pixi run -e gui python src/iitm_http_worker.py ...`
5. worker 将日志写到 stdout，GUI 实时显示。
6. worker 返回 JSON：
   - `status`
   - `metrics`
   - `artifacts`
   - `field_catalog`
   - `requested_field_compute_mode`
   - `effective_field_compute_mode`
7. GUI iframe 加载输出目录下的 HTML。
8. GUI 通过 URL query 或 `postMessage` 切换 `family + field`。

## Mie 后端详细流程

核心文件：

- `src/mie_core.py`
- `src/mie_numba.py`
- `src/mie_worker.py`

入口：

```powershell
pixi run -e mie python -u src/mie_worker.py --project_name test1 --config "{...}" --cpu_limit 4
```

### 1. 配置与目录

`mie_worker.py` 合并 `DEFAULT_CONFIG` 和用户 JSON，并强制：

- `grid_dim -> int`
- `photons -> int`

目录行为：

| 路径 | 行为 |
| --- | --- |
| `temp/mie/<project>` | 每次运行前清空重建 |
| `outputs/mie/<project>` | 创建或复用，覆盖本次输出 |
| `log/mie_<project>_<timestamp>.log` | 写运行日志 |

### 2. Mie 光学层构建

函数：`build_mie_layers(config)`。

流程：

1. 读取 `L_size / r_bottom / r_top / sigma_ln / wavelength_um / m_real / m_imag / visibility_km`。
2. 确定层数：
   - `mie_layer_count` 显式给定时使用该值。
   - 否则默认 `ceil(grid_dim / 8)`，并限制在 `6..16`。
3. 确定粒径分布：
   - `sigma_ln <= 1e-6`：`mono`
   - 否则：`lognormal`
4. 每个 z-layer 计算代表半径。
5. 调用 `mie_core.mie_effective_polarized()` 计算：
   - `sigma_ext`
   - `sigma_sca`
   - `omega0`
   - `g`
   - `M11 / M12 / M33 / M34`
6. 调用 `mie_scatter_observables()` 得到：
   - `sigma_back_ref`
   - `sigma_forward_ref`
   - `forward_back_ratio`
   - `depol_back`
   - `depol_forward`
7. 由 visibility 换算体消光系数，并生成：
   - `beta_back_profile`
   - `beta_forward_profile`
   - `depol_profile`
   - `beta_ext_profile`

### 3. 三维密度场与 proxy 场

函数：`generate_field(config, temp_dir, optical_layers)`。

物理/工程模型：

- 多尺度随机噪声模拟不均匀结构。
- 垂直 Gaussian profile 模拟云层中心和厚度。
- `density_norm` 被裁剪到 `[0, 1]`。
- proxy 场按 `density * z-LUT` 展开。

输出字段：

- `density_norm`
- `beta_ext`
- `beta_back`
- `beta_forward`
- `depol_ratio`
- `lut_back`
- `lut_forward`
- `lut_depol`

临时写入：

```text
temp/mie/<project>/field_data.npz
```

### 4. Numba Monte Carlo

入口：`mie_numba.run_advanced_simulation()`。

主要内核：

| 内核 | 适用路径 |
| --- | --- |
| `mc_kernel_advanced_fast` | 无空间/角度记录、无 exact 场 |
| `mc_kernel_advanced_exact` | `exact_only` 或 `both` |
| `mc_kernel_advanced` | 空间记录或角度直方图慢路径 |

当前优化：

- 3D density 使用 z-slab local majorant，避免稀疏云层下全局 majorant 导致大量 null collision。
- fast/exact batch size 为 `8192`。
- exact voxel arrays 由 Python 侧预分配，Numba 内核中原地累积。
- exact 的 `event_count` 保留原始散射事件计数；Stokes response 按 photon 数归一化。

Monte Carlo 输出：

- `R_back`
- `R_trans`
- `R_abs`
- `depol_ratio`
- `omega0`
- `g`
- `beta_back_ref`
- `beta_forward_ref`
- `forward_back_ratio`
- `depol_back`
- `depol_forward`
- `arrays.voxel_fields`，当请求 exact 时存在。

### 5. Mie exact 场

`exact` 场定义为 detector-conditioned response field。

在 accepted scattering event 处：

1. 找到事件所在 voxel。
2. 对 forward detector cone 每个方向：
   - 计算该方向相对当前 photon 方向的散射角。
   - 插值 Mueller matrix。
   - 旋转 Stokes。
   - 计算到边界的 escape transmittance。
   - 累积 `forward_I/Q/U/V`。
3. 对 back detector cone 重复同样过程。
4. 累积 `event_count`。

导出字段：

- `exact_beta_back`
- `exact_beta_forward`
- `exact_depol_ratio`
- `exact_event_count`
- `exact_back_Q/U/V`
- `exact_forward_Q/U/V`
- `exact_summary`

## Julia/IITM 后端详细流程

核心文件：

- `src/julia/iitm_physics.jl`
- `src/julia/iitm_server.jl`
- `src/iitm_http_worker.py`

启动服务：

```powershell
pixi run -e julia julia --project=src/julia src/julia/iitm_server.jl --port 2700 --root .
```

HTTP API：

- `GET /health`
- `POST /simulate`
- `POST /simulate/stream`
- `POST /scatter_only`

GUI 默认使用 `/simulate/stream`。

### 1. 配置拆分

函数：`build_configs(config)`。

生成：

- `scatter_cfg`：T-Matrix/IITM 散射求解参数。
- `mc_cfg`：Monte Carlo 参数。
- `beta_s`：visibility 换算得到的地表/参考消光系数。

代表粒径策略：

1. 如果用户设置 `r_eff > 0`，使用用户显式代表粒径。
2. 否则使用 `sqrt(r_bottom * r_top)`。
3. 默认受 `nmax_safe=20` 约束，避免高阶 T-Matrix 过慢。

这意味着极大 `r_top` 下，Julia 默认结果是受安全策略约束的代表粒径模型，而不是无条件使用最大粒径。

### 2. Step1 密度场

函数：`step1_field()`。

生成：

- `density_norm`
- `axis`
- 相关 field metadata

### 3. Step2 T-Matrix 散射

函数：`step2_scatter()`。

支持形状：

- `sphere`
- `cylinder`
- `spheroid`

solver：

- `auto`
- `iitm`
- `ebcm`

输出：

- `sigma_ext`
- `sigma_sca`
- `omega0`
- `g`
- `M11 / M12 / M33 / M34`
- `sigma_back_ref`
- `sigma_forward_ref`
- `forward_back_ratio`
- `depol_back`
- `depol_forward`
- solver diagnostics

solver diagnostics 包括：

- `solver_requested`
- `solver_used`
- `fallback_used`
- `fallback_reason`
- `solver_path_summary`
- `ebcm_count`
- `iitm_count`
- `nmax_min`
- `nmax_max`
- `ebcm_loss_estimate_max`
- `ebcm_loss_estimate_mean`

### 4. Step3 Julia Monte Carlo

函数：`step3_mc()`，核心在 `run_monte_carlo()`。

行为：

- `proxy_only`：不收集 voxel fields，避免 exact 额外开销。
- `exact_only` / `both`：收集 voxel observables。
- 使用 chunk 并行。
- 每个 chunk 使用独立 seed。
- voxel exact 数据先写入 chunk-local sparse map，再归并。

3D density 支持：

- `mc_use_3d_density=true`
- `mc_density_sampling=nearest`
- `mc_density_sampling=trilinear`

### 5. Step4 场族与渲染

函数：`step4_render()`。

流程：

1. `build_field_bundle()` 生成 `proxy` 和/或 `exact` family。
2. `build_field_catalog()` 生成 GUI 字段目录。
3. `save_field_npz()` 保存 `density.npz`。
4. `render_to_html()` 生成 3 个 HTML。

## 文件输出协议

两套后端最终输出目录：

- `outputs/mie/<project>`
- `outputs/iitm/<project>`

标准产物：

```text
density.npz
render_main.html
render_top.html
render_front.html
```

`density.npz` 常用键：

| 键 | 含义 |
| --- | --- |
| `density` | 三维密度场 |
| `axis` | 坐标轴 |
| `meta` | `[L, N, reserved]` |
| `proxy_beta_back` | proxy 后向场 |
| `proxy_beta_forward` | proxy 前向场 |
| `proxy_depol_ratio` | proxy 退偏场 |
| `proxy_summary` | proxy 汇总 |
| `exact_beta_back` | exact 后向响应场 |
| `exact_beta_forward` | exact 前向响应场 |
| `exact_depol_ratio` | exact 退偏场 |
| `exact_event_count` | exact voxel 采样次数 |
| `exact_summary` | exact 汇总 |
| `beta_back` | primary family legacy alias |
| `beta_forward` | primary family legacy alias |
| `depol_ratio` | primary family legacy alias |
| `summary` | primary family legacy alias |

## 浏览器渲染流程

两套后端当前渲染方式同构。

1. 后端写 `density.npz`。
2. 后端写三份 HTML，每份只包含轻量 JS 模板和默认 camera。
3. HTML 加载：
   - JSZip
   - Plotly
   - 内置 `.npy` parser
4. 浏览器执行 `fetch('./density.npz?t=...')`。
5. JS 从 NPZ 中读取 `density / axis / field arrays`。
6. 根据 `field_catalog` 生成字段按钮。
7. 使用 Plotly `volume` trace 渲染。
8. GUI iframe 通过 query 或 `postMessage` 切换：
   - `family=proxy|exact`
   - `field=beta_back|beta_forward|depol_ratio|density|event_count`

渲染成本随 `grid_dim^3 * 字段数` 增长。`grid_dim=120` 时，单个 Float32 三维场约 `6.9 MB`，多个 proxy/exact 字段会显著增加浏览器内存和 NPZ 大小。

## 参数说明

### 公共参数

| 参数 | Mie | Julia | 含义 |
| --- | --- | --- | --- |
| `L_size` | 支持 | 支持 | 物理盒子厚度/边长 |
| `grid_dim` | 支持 | 支持 | 三维场分辨率 |
| `wavelength_um` | 支持 | 支持 | 波长 |
| `m_real` | 支持 | 支持 | 折射率实部 |
| `m_imag` | 支持 | 支持 | 折射率虚部 |
| `r_bottom` | 支持 | 支持 | 底部粒径 |
| `r_top` | 支持 | 支持 | 顶部粒径 |
| `sigma_ln` | 支持 | 支持 | lognormal 对数标准差 |
| `visibility_km` | 支持 | 支持 | 可见度 |
| `angstrom_q` | 支持 | 支持 | Angstrom 波长修正指数 |
| `cloud_center_z` | 支持 | 支持 | 云中心高度 |
| `cloud_thickness` | 支持 | 支持 | 云厚度 |
| `turbulence_scale` | 支持 | 支持 | 湍流结构尺度 |
| `photons` | 支持 | 支持 | Monte Carlo 光子数 |
| `field_compute_mode` | 支持 | 支持 | `proxy_only/exact_only/both` |
| `field_forward_half_angle_deg` | 支持 | 支持 | exact 前向探测锥半角 |
| `field_back_half_angle_deg` | 支持 | 支持 | exact 后向探测锥半角 |
| `field_quadrature_polar` | 支持 | 支持 | 探测锥极角积分点 |
| `field_quadrature_azimuth` | 支持 | 支持 | 探测锥方位积分点 |

### Mie 侧参数

| 参数 | 含义 |
| --- | --- |
| `mie_layer_count` | 光学层数量 |
| `mie_n_radii` | lognormal 粒径积分采样数 |
| `forward_cone_deg` | proxy 前向参考 cone |
| `cpu_cores` / `--cpu_limit` | Numba/OMP 线程环境变量 |
| `explode_dist` | 保留配置项，当前浏览器渲染不主要依赖 |

### Julia/IITM 侧参数

| 参数 | 含义 |
| --- | --- |
| `shape_type` | `cylinder/spheroid/sphere` |
| `r_eff` | 显式代表粒径 |
| `axis_ratio` | 轴比 |
| `nmax_override` | T-Matrix 阶数覆盖 |
| `n_radii` | 粒径分布积分点数 |
| `Nr` | IITM 径向积分分辨率 |
| `Ntheta` | IITM 角向积分分辨率 |
| `tmatrix_solver` | `auto/iitm/ebcm` |
| `forward_mode` | 前向参考模式 |
| `forward_cone_deg` | 前向 cone 平均角 |
| `ebcm_threshold` | EBCM 阈值 |
| `ebcm_ndgs` | EBCM 参数 |
| `ebcm_maxiter` | EBCM 最大迭代 |
| `ebcm_loss_threshold` | EBCM loss guard |
| `iitm_nr_scale_on_fallback` | fallback 后 Nr 放大 |
| `iitm_ntheta_scale_on_fallback` | fallback 后 Ntheta 放大 |
| `mc_use_3d_density` | 是否使用 3D density MC |
| `mc_density_sampling` | `nearest/trilinear` |
| `scale_height_m` | 1D profile 尺度高度 |

## 物理语义与精度边界

### `proxy` 场

`proxy` 是快速场，不是 MC voxel history inversion。

- Mie proxy：`density * Mie layer LUT`。
- Julia proxy：`density * scatter reference quantities`。
- 适合：
  - 定性展示。
  - 参数扫描。
  - 快速比较。
- 不适合：
  - 直接当作 detector-conditioned 定量响应。
  - 与 exact voxel 逐点做无条件等价比较。

### `exact` 场

`exact` 是 detector-conditioned response field。

- 来源：Monte Carlo accepted scattering events。
- 每个事件按 forward/back detector cone 累积响应。
- `event_count` 是解释 exact 场可信度的关键。
- `event_count=0` 的 voxel 不应做定量解释。

### 退偏比

`depol_ratio` 当前统一钳制在 `[0, 1]`。

- proxy depol：来自 LUT 或散射参考量。
- exact depol：来自 voxel 累积 Stokes 分量：
  - `1 - sqrt(Q^2 + U^2 + V^2) / I`

两者来源不同，当前字段名相同但通过 family 区分。

### `sigma_*_ref` 与 `beta_*_ref`

- `sigma_*_ref`：截面参考量，来自单粒子或粒径分布等效散射。
- `beta_*_ref`：体积系数参考量，Mie Numba 侧用于层平均指标。
- 不应把 `sigma` 和 `beta` 直接混用。

### Monte Carlo 统计误差

比例量 `p` 的标准误差近似为：

```text
sqrt(p * (1 - p) / N)
```

示例：如果 `R_back ≈ 0.003` 且 `N = 80000`，标准误差约为 `0.000193`，相对误差约 `6%`。exact voxel 场的有效样本不是总 photon 数，而是每个 voxel 的 `event_count`。

## 当前验证数据

### Mie 物理一致性

命令：

```powershell
pixi run -e mie python -u src/mie_worker.py --test_only
```

最近一次结果：

| 测试 | 数据 |
| --- | --- |
| TC-E 无吸收 | `R_back=0.000050`, `R_trans=0.999950`, sum=`1.000000` |
| TC-A 有吸收 | `R_back=0.000150`, `R_trans=0.999800`, `R_abs=0.000050`, sum=`1.000000` |
| TC-P 偏振 | `depol=0.219092` |
| TC-R 瑞利波长比 | `R_back@1.55um=1.475e-3`, `R_back@0.55um=8.840e-2`, ratio=`59.932` |
| TC-G 大粒径后向 | `R_back=0.002440` |
| TC-L lognormal 手动积分 | `Bext=7.547191e-13 m^-1` |

### Julia 物理回归

命令：

```powershell
pixi run -e julia julia --threads auto --project=src/julia src/julia/iitm_physics.jl
```

最近一次结果：

```text
79 / 79 通过
```

覆盖内容：

- visibility 和积分工具。
- lognormal PDF 归一化。
- `sphere/cylinder/spheroid` 粒子构造。
- 单粒子截面合理性。
- CDF 单调性。
- `integral M11 sin(theta) dtheta ≈ 2`。
- 无吸收/有吸收 MC 能量守恒。
- 3D density nearest/trilinear MC。
- proxy/exact NPZ 合约。
- 多线程与单线程 MC 回归一致性。

### Mie exact 端到端烟测

最近一次 worker exact 输出：

- `R_back=0.008`
- `R_trans=0.992`
- `R_abs=0`
- `exact_event_count=118`
- `exact_forward_sum≈0.2387`
- `exact_back_sum≈0.1894`

## 当前性能数据

本机热身后小基准：

| 路径 | 近似吞吐 |
| --- | --- |
| Mie proxy fast | `~13 Mphotons/s` |
| Mie exact, `1 x 2` detector cone | `~2.1 Mphotons/s` |
| Mie exact, 默认近似 `2 x 6` detector cone | `~1.1 Mphotons/s` |

解释：

- Mie proxy fast 主要受 photon propagation 和 null collision 影响。
- Mie exact 额外执行 Mueller 插值、Stokes 旋转、探测锥循环和 escape transmittance 积分。
- Julia 完整物理回归最近一次在 16 线程下约 `34.9s`，包含 T-Matrix、多组 MC、NPZ/HTML、跨线程一致性子进程测试。

主要性能瓶颈：

- Mie exact 仍是串行 kernel，避免 voxel 写竞争。
- Mie exact 的 `escape_transmittance` 对 detector cone 每个方向做步进积分。
- Julia Step2 T-Matrix 对 `nmax / Nr / Ntheta / n_radii / shape_type` 敏感。
- 浏览器渲染随 `grid_dim^3 * 字段数` 增长。

## 运行与测试

安装环境：

```powershell
pixi install
```

启动 GUI：

```powershell
pixi run -e gui python src/gui.py
```

Mie worker：

```powershell
pixi run -e mie python -u src/mie_worker.py --project_name test1 --config "{...}" --cpu_limit 4
```

Julia server：

```powershell
pixi run -e julia julia --project=src/julia src/julia/iitm_server.jl --port 2700 --root .
```

Mie 测试：

```powershell
pixi run -e mie python -m unittest discover -s tests -p "test_mie_*.py"
pixi run -e mie python -u src/mie_worker.py --test_only
```

Julia 测试：

```powershell
pixi run -e julia julia --project=src/julia tests/julia_field_contract.jl
pixi run -e julia julia --project=src/julia tests/julia_server_contract.jl
pixi run -e julia julia --threads auto --project=src/julia src/julia/iitm_physics.jl
```

语法检查：

```powershell
pixi run -e mie python -m py_compile src/mie_core.py src/mie_numba.py src/mie_worker.py
pixi run -e gui python -m py_compile src/gui.py src/iitm_http_worker.py src/iitm_renderer.py
```

## 使用建议

- 球形粒子快速扫描：优先 Mie proxy。
- 球形粒子关注探测器响应空间分布：使用 Mie exact 或 both。
- 非球形粒子：使用 Julia/IITM。
- 需要 solver 诊断：使用 Julia/IITM。
- exact 场定量分析必须同时报告：
  - photon 数
  - detector cone 半角
  - quadrature 设置
  - `event_count`
- 高 `grid_dim` 或 `field_compute_mode=both` 会显著增加内存、NPZ 体积和浏览器渲染压力。

## 仓库协作约定

- 当前 GitHub 远端仓库用于保存当前代码基线与日常备份。
- 大规模重构或高风险试验前，应先提交并推送稳定状态。
- 本地运行产物如 `outputs/`、`temp/`、`log/`、`.pixi/` 不作为备份内容。
- 后端协议、字段名、渲染产物、测试命令变化后，应同步更新 `Readme.md` 和 `todo.md`。
