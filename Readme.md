# MonteCarlo

`MonteCarlo` 是一个面向云雾/气溶胶光散射问题的异构仿真工作台。项目通过 GUI 统一调度两条后端：

- `Mie + Monte Carlo + Numba`
  - 面向球形粒子和等效粒径分布。
  - 适合快速基线、参数扫描、工程诊断。
- `IITM/T-Matrix + Monte Carlo + Julia`
  - 面向 `sphere / cylinder / spheroid`。
  - 适合非球形粒子、solver 诊断和更严格的散射物理分析。

当前日期状态：`2026-04-29`。

## 当前状态

- GUI 已支持 `mie` 和 `iitm` 两个后端切换。
- 两套后端均导出同构数据包：`density.npz + render_main.html + render_front.html + render_top.html + render_right.html`。
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
- GUI 会从后端返回的 `field_catalog` 或现有 `density.npz` 中恢复 `proxy/exact` 字段目录；切换项目后仍可选择精确场。
- 字段切换优先通过 iframe `postMessage` 完成，避免频繁销毁 Plotly/WebGL 上下文；切换视图文件时才重新加载 iframe。

## 内置仿真情景

旧样例已清理，当前主项目内置 8 个可运行配置，见 `inputs/SCENARIOS.md`。

| 后端 | 环境 | 配置 |
| --- | --- | --- |
| Mie | 雾 | `inputs/mie/pdf_fog_radiation.json` |
| Mie | 霾 | `inputs/mie/pdf_haze_dust.json` |
| Mie | 雨 | `inputs/mie/pdf_rain_moderate.json` |
| Mie | 快速展示 | `inputs/mie/quick_display.json` |
| IITM | 雾 | `inputs/iitm/pdf_fog_radiation.json` |
| IITM | 霾 | `inputs/iitm/pdf_haze_dust.json` |
| IITM | 雨 | `inputs/iitm/pdf_rain_moderate.json` |
| IITM | 快速展示 | `inputs/iitm/quick_display.json` |

共同约束：

- 完整情景按 `temp/计算流程.pdf` 的雨/雾/霾设定组织，波长为 `1550 nm`，名义探测距离为 `1-2000 m`。
- 完整情景距离门宽度为 `10 m`，接收重叠因子在 `200 m` 达到全重叠。
- Mie 情景使用点源入射，以便与 IITM/T-Matrix 的点源 Monte Carlo 结果可比。
- GUI 预览配置默认使用 `preview_max_grid=48`，请求预览时传递 `max_grid=48`，降低 WebView/Plotly 内存压力。
- 完整 Marshall-Palmer 雨滴积分仍以 `temp/lidar_1d` 为物理参考；主项目 3D 后端中的雨情景使用可运行、可预览的水滴代理配置。

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
│  ├─ test_mie_lidar_observation.py
│  ├─ test_dataset_contract.py
│  ├─ julia_field_contract.jl
│  ├─ julia_lidar_contract.jl
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
pixi run -e mie python -u src/mie_worker.py --project_name quick_display --config (Get-Content inputs/mie/quick_display.json -Raw) --cpu_limit 4
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
4. `render_to_html()` 生成四视图 HTML：主视、前视、顶视、右视。

## 文件输出协议

两套后端最终输出目录：

- `outputs/mie/<project>`
- `outputs/iitm/<project>`

标准产物：

```text
density.npz
render_main.html
render_front.html
render_top.html
render_right.html
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
| `range_bins_m` | 距离门坐标 |
| `echo_I/Q/U/V` | 距离门 Stokes 通道 |
| `echo_power` | 距离门回波强度 |
| `echo_depol` | 距离门退偏比 |
| `echo_event_count` | 每个距离门采样事件数 |
| `echo_weight_sum` | 每个距离门权重和 |
| `echo_relative_error_est` | 距离门相对误差估计 |
| `beta_back` | primary family legacy alias |
| `beta_forward` | primary family legacy alias |
| `depol_ratio` | primary family legacy alias |
| `summary` | primary family legacy alias |

## 浏览器渲染流程

两套后端当前渲染方式同构。

1. 后端写 `density.npz`。
2. 后端写四份 HTML，每份只包含轻量 JS 模板和默认 camera。
3. HTML 加载：
   - JSZip
   - Plotly
   - 内置 `.npy` parser
4. 浏览器执行 `fetch('./density.npz?t=...')`。
5. JS 从 NPZ 中读取 `density / axis / field_catalog`，并按当前 `family + field` 选择实际渲染数组。
6. 根据 `field_catalog` 生成字段按钮。
7. 使用 Plotly `volume` trace 渲染。
8. GUI iframe 通过 query 初始化，通过 `postMessage` 切换：
   - `family=proxy|exact`
   - `field=beta_back|beta_forward|depol_ratio|density|event_count`
   - `view=main|front|top|right`

字段切换不重新创建 iframe，优先复用当前 Plotly/WebGL 上下文；只有视图文件变化时才重新加载对应 HTML。GUI 在切换项目后会从 worker 返回值或已存在的 `density.npz` 恢复 `proxy/exact` 字段目录，避免 exact 按钮丢失。

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
| `preview_max_grid` | 支持 | 支持 | GUI 预览降采样上限，默认建议 `48` |
| `lidar_enabled` | 支持 | 支持 | 是否输出距离门激光雷达观测量 |
| `range_bin_width_m` | 支持 | 支持 | 距离门宽度 |
| `range_max_m` | 支持 | 支持 | 最大观测距离，`0` 时使用后端默认范围 |
| `receiver_overlap_min` | 支持 | 支持 | 近场接收重叠因子下限 |
| `receiver_overlap_full_range_m` | 支持 | 支持 | 接收重叠因子达到 1 的距离 |

### Mie 侧参数

| 参数 | 含义 |
| --- | --- |
| `mie_layer_count` | 光学层数量 |
| `mie_n_radii` | lognormal 粒径积分采样数 |
| `forward_cone_deg` | proxy 前向参考 cone |
| `source_type` | Monte Carlo 入射源，`point` 或 `planar` |
| `source_width_m` | 平面源半宽；点源时可为 `0` |
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

- proxy depol：来自 LUT 或散射参考量。若场景只有单一均匀粒径谱/形状谱，且 LUT 不随高度、混合比例或微物理状态变化，proxy depol 出现常数是合理结果，不代表探测器响应退化。
- exact depol：来自 voxel 累积 Stokes 分量：
  - `1 - sqrt(Q^2 + U^2 + V^2) / I`
- echo depol：来自距离门 Stokes 通道，是后续反演数据库优先使用的观测量。

要让 proxy 退偏比具有空间变化，需要引入分层、双峰/混合粒径谱、形状混合、折射率变化或 range-varying microphysics；否则应优先使用 exact/echo depol 解释 MC 探测响应。两类三维场来源不同，当前字段名相同但通过 family 区分。

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
pixi run -e mie python -u src/mie_worker.py --project_name quick_display --config (Get-Content inputs/mie/quick_display.json -Raw) --cpu_limit 4
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
- 需要与 IITM/T-Matrix 的空间图像做形态比较：Mie 使用 `source_type=point`，避免平面源把入射光束展宽为近似平面化结构。
- exact 场定量分析必须同时报告：
  - photon 数
  - detector cone 半角
  - quadrature 设置
  - `event_count`
- 高 `grid_dim` 或 `field_compute_mode=both` 会显著增加内存、NPZ 体积和浏览器渲染压力。
- 频繁切换 GUI 预览时保守使用 `preview_max_grid=48`，并优先切换字段而非反复打开新项目窗口。
- 三维场均提供 `main/front/top/right` 四个视图；回波曲线和反演样本协议以 `echo_*` 数组为准，不从 HTML 截图反推。

## 【新增】面向激光雷达反演数据库的下一步工作（第一优先级）

> 本节为新加入内容，目标是将当前主项目从“前向仿真与可视化平台”扩展为“可批量生成高可信反演样本的数据引擎”。
> 适用范围：`Mie + Numba` 与 `Julia/IITM` 两条后端。

### 目标声明（用于后续实现约束）

当前项目后续迭代将明确服务于：

- 建立激光雷达反演数据库（training/validation/test datasets）。
- 数据库样本输出必须与观测量一致（距离门回波强度与偏振通道），而不仅是体素场或全局散射统计量。
- 每条样本必须附带数值误差与配置元数据，保证可追溯和可复现实验。

---

### 第一优先级 P0（必须先完成）

#### P0-1. 时间门控观测算子（Time-Gated Lidar Observation Operator）【最高优先级】

当前 `exact/proxy` 体素结果不能直接等价为 PDF 所需 `P(R)` 曲线。需要新增“观测门函数+距离门累积”模块：

1. 定义距离门/时间门：
   - `R_i = i * ΔR`，`ΔR` 来自激光脉宽与采样设置。
   - 对应飞行时间 `t_i = 2R_i/c`。
2. 对每次散射贡献记录“到达探测器的总光程”，映射到 `R-bin`。
3. 在每个 `R-bin` 累积 Stokes 通道：
   - `I(R_i), Q(R_i), U(R_i), V(R_i)`。
4. 从门控 Stokes 计算观测量：
   - `P(R_i)`（功率或归一化功率）
   - `δ(R_i)`（退偏比）
   - 可选 `SNR(R_i)`。
5. 接收器模型参数化：
   - `overlap O(R)`
   - `FOV`（前/后向探测锥）
   - 噪声底与门宽。

**建议新增输出（非绘图）**：

- `arrays.range_bins_m`
- `arrays.echo_I, echo_Q, echo_U, echo_V`
- `arrays.echo_power`
- `arrays.echo_depol`
- `meta.receiver_model`（门宽/FOV/overlap/noise 等）

**当前实现状态（2026-04-29 迭代）**：

- Mie/Numba 与 Julia/IITM 已接入同构距离门观测输出。
- 输出字段保存到 `density.npz`：
  - `range_bins_m`
  - `echo_I / echo_Q / echo_U / echo_V`
  - `echo_power`
  - `echo_depol`
  - `echo_event_count`
  - `echo_weight_sum`
  - `echo_relative_error_est`
- 当前 `echo_*` 来源为 accepted scattering event 的 detector-cone backscatter response：
  - 距离门映射使用 `R = (path_length_to_event + escape_path_to_receiver) / 2`。
  - receiver overlap 当前使用简化线性模型。
- Mie 批量数据集 runner 已支持单样本协议：
  - `observation.npz`
  - `truth.json`
  - `receiver.json`
  - `quality.json`
  - `run_config.json`

**验收标准**：

- 在均匀介质下，门控 `P(R)` 与解析雷达方程趋势一致。
- 在非理想重叠设置下，近场抑制行为符合输入 `O(R)`。
- `δ(R)` 不再被迫为常数，且与场景参数变化一致。

#### P0-2. 反演数据库样本协议（Dataset Contract）

建立统一样本协议，确保后续训练与验证可复用：

- 输入标签（truth）：
  - 介质参数（粒径谱、形状谱、折射率、浓度、空间分布参数）
  - 仪器参数（波长、门宽、FOV、重叠、噪声）
- 输出观测（features）：
  - `P(R)`、`δ(R)`、可选多通道偏振信号
- 质量元数据：
  - 光子数、角分辨、积分节点数、收敛诊断、随机种子、代码版本 hash。

**验收标准**：

- 任意样本可通过元数据完整复算。
- 不同后端（Mie / IITM）可输出同构观测字段。

---

### 第二优先级 P1（完成 P0 后推进）

#### P1-1. 逐体素微物理参数兼容（高自由度场）

在保留当前 `density_grid` 的基础上，扩展到可输入：

- `voxel_mie_id_grid`（每体素散射类型索引）
- 或分块参数场（比完全逐体素更稳健）：
  - `N0(x,y,z), r_med(x,y,z), sigma_ln(x,y,z), shape(x,y,z), axis_ratio(x,y,z), m(x,y,z)`

建议采用“两级策略”：

1. 先支持分块（tile/chunk）参数场；
2. 再支持完整逐体素自由度。

**原因**：兼顾可行性（内存/计算）与物理表达能力。

#### P1-2. 粒径与形状谱求解器分层

为不同场景提供分层精度档：

- 快速档：lognormal + 低阶角分辨
- 标准档：lognormal/bimodal + 中高角分辨
- 高精档：非球形 + 高阶 T-matrix 半径求积 + 收敛检查

并将档位与误差阈值绑定（不是只给参数，不给通过标准）。

---

### 第三优先级 P2（数据库规模化与可信度增强）

#### P2-1. 收敛与不确定度体系（必须随样本输出）

每条样本输出至少三类误差指标：

1. 粒径积分误差（例如 n 与 n/2 对比）
2. 角分辨误差（角网格加密前后对比）
3. MC 统计误差（按 bin 的方差/置信区间）

#### P2-2. 场景采样设计（避免数据库偏置）

采用分层采样/LHS 方案覆盖：

- 雾/霾/雨多场景
- 能见度、粒径谱参数、非球形比例、湿增长参数
- 仪器参数扰动（FOV、overlap、噪声）

避免“只在少数参数区域高密采样”，导致反演模型泛化差。

#### P2-3. 基准与域校准

增加与解析/文献/高保真参考的交叉校准流程：

- Rayleigh / 单散射极限
- 典型雾霾雨工况的经验区间
- 不同后端同场景一致性检查（Mie vs IITM 在可比条件下）

---

### 实施顺序建议（可直接执行）

1. 先做 `P0-1` 时间门控观测算子（核心瓶颈，第一优先级）。
2. 同步落地 `P0-2` 样本协议（避免后续返工）。
3. 再做 `P1-1` 分块/逐体素微物理自由度扩展。
4. 最后推进 `P1-2`、`P2-*` 做规模化与可信度闭环。

### 风险与边界

- 若跳过 `P0-1`，则数据库“标签-观测一致性”不足，不建议直接用于反演训练。
- 若只做逐体素参数、不做误差输出，数据库可用性会受限（难以筛选低质量样本）。
- 若仅增加样本数量、不做场景采样设计，反演模型容易发生参数域偏置。

## 仓库协作约定

- 当前 GitHub 远端仓库用于保存当前代码基线与日常备份。
- 大规模重构或高风险试验前，应先提交并推送稳定状态。
- 本地运行产物如 `outputs/`、`temp/`、`log/`、`.pixi/` 不作为备份内容。
- 后端协议、字段名、渲染产物、测试命令变化后，应同步更新 `Readme.md` 和 `todo.md`。
