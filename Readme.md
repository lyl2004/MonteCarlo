# MonteCarlo

`MonteCarlo` 是一个面向光散射问题的异构仿真工作台，当前维护两条主链路：

- `Mie + Monte Carlo + Numba`
  适用于球形粒子与等效粒径分布的快速基线仿真。
- `IITM/T-Matrix + Monte Carlo + Julia`
  适用于非球形粒子的散射计算、solver 诊断和三维场展示。

项目通过 `Pixi` 管理环境，通过 GUI 统一调度 Python 和 Julia 后端。

## 当前状态

截至 `2026-04-27`，当前代码状态可以概括为：

- GUI 已可在 `mie` 和 `iitm` 两个后端之间切换。
- `Mie` 链路是球形粒子的快速基线实现。
- `Julia` 链路是非球形粒子的主计算实现，支持 `cylinder / spheroid / sphere`。
- 两条链路都支持三类展示字段：
  - `beta_back`
  - `beta_forward`
  - `depol_ratio`
- 当前场计算协议已经进入双轨阶段：
  - `proxy`：代理场，用于快速预览、参数扫描和工程诊断。
  - `exact`：detector-conditioned response field，目前仅 Julia 链路支持。
- `field_compute_mode` 可取：
  - `proxy_only`
  - `exact_only`
  - `both`
- 后端会返回：
  - `requested_field_compute_mode`
  - `effective_field_compute_mode`
  - `available_field_families`
  - `field_catalog`

## 后端能力矩阵

| 能力 | Mie 后端 | Julia/IITM 后端 |
| --- | --- | --- |
| 球形粒子散射 | 支持 | 支持 |
| 非球形粒子散射 | 不支持 | 支持 |
| 对数正态粒径聚合 | 支持 | 支持 |
| 偏振 Mueller 表 | 支持 | 支持 |
| Monte Carlo 传输 | Numba | Julia threads |
| `proxy` 三维场 | 支持 | 支持 |
| `exact` 三维场 | 未实现 | 支持 |
| `field_compute_mode=both` | 请求会回落为 `proxy_only` | 支持 |
| 离线渲染 | PyVista + Panel HTML | NPZ + 浏览器端 Plotly |
| solver 诊断 | 不适用 | 支持 |

## 核心计算链路

### Mie 链路

Mie 后端的核心文件：

- `src/mie_core.py`
- `src/mie_numba.py`
- `src/mie_worker.py`

计算流程：

1. `mie_core.py` 根据能见度、波长、粒径和折射率计算 Mie 有效散射参数。
2. `mie_worker.py` 将垂直粒径剖面离散为少量光学层，生成层级 LUT。
3. `mie_worker.py` 生成三维湍流密度场，并用 `density × LUT` 展开代理场。
4. `mie_numba.py` 运行 Numba Monte Carlo，输出传输统计。
5. `mie_worker.py` 用 PyVista + Panel 导出多字段、多视角 HTML。

主要输出：

- 核心散射参数
  - `sigma_ext`
  - `sigma_sca`
  - `omega0`
  - `g`
  - `M11/M12/M33/M34`
- 截面参考量
  - `sigma_back_ref`
  - `sigma_forward_ref`
  - `forward_back_ratio`
  - `depol_back`
  - `depol_forward`
- Monte Carlo 统计
  - `R_back`
  - `R_trans`
  - `R_abs`
  - `depol_ratio`
- 三维代理场
  - `beta_back`
  - `beta_forward`
  - `depol_ratio`

注意：Mie worker 当前只导出 `proxy` 场族。即使请求 `exact_only` 或 `both`，`effective_field_compute_mode` 也会返回 `proxy_only`。

### Julia/IITM 链路

Julia 后端的核心文件：

- `src/julia/iitm_physics.jl`
- `src/julia/iitm_server.jl`
- `src/iitm_http_worker.py`

计算流程：

1. `iitm_server.jl` 将 GUI 配置拆成散射配置和 Monte Carlo 配置。
2. `iitm_physics.jl` 使用 T-Matrix 求解单粒子散射，并对粒径分布积分。
3. `iitm_physics.jl` 运行 3D density-aware 偏振 Monte Carlo。
4. 当请求 `exact_only` 或 `both` 时，Julia 从 MC voxel observables 生成 detector-conditioned exact 场。
5. `iitm_physics.jl` 保存 `density.npz`，HTML 通过浏览器端 JSZip + Plotly 渲染。

主要输出：

- 与 Mie 对齐的核心散射量
- `proxy` 场族
  - `beta_back`
  - `beta_forward`
  - `depol_ratio`
  - `density`
- `exact` 场族
  - `beta_back`
  - `beta_forward`
  - `depol_ratio`
  - `event_count`
- solver 诊断
  - `solver_requested`
  - `solver_used`
  - `fallback_used`
  - `fallback_reason`
  - `solver_path_summary`
  - `ebcm_count / iitm_count`
  - `nmax_min / nmax_max`
  - `ebcm_loss_estimate_max / mean`

## 精度与物理语义边界

当前需要特别区分三类量：

- `sigma_*_ref`
  单粒子或粒径分布等效后的截面参考量，主要来自相函数和散射截面。
- `beta_*_ref`
  体积系数参考量，Mie Numba 侧用于层平均指标，单位和语义不同于 `sigma_*_ref`。
- 三维场
  展示和诊断用体场，不能直接和单粒子截面混用。

当前仍需明确或改进的物理语义：

- `forward_back_ratio` 仍是工程代理指标，不是严格的前后向能量比。
- Mie 的 `beta_back / beta_forward / depol_ratio` 是 LUT 展开的代理场。
- Julia 的 `proxy` 场是代理场，Julia 的 `exact` 场是 detector-conditioned response field。
- `depol_ratio` 在 proxy 和 exact 场族中来源不同，后续文档和 GUI 应明确区分。
- 当前 `safe_depol_ratio()` 只做非负钳制，极端或数值噪声下参考退偏量可能略大于 `1`，后续需要决定是否强制 clamp 到 `[0, 1]`。

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
│  └─ julia_field_contract.jl
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

## 环境

当前使用三个 Pixi 环境：

- `gui`
- `mie`
- `julia`

安装：

```powershell
pixi install
```

## 运行方式

启动 GUI：

```powershell
pixi run -e gui python src/gui.py
```

直接运行 Mie worker：

```powershell
pixi run -e mie python -u src/mie_worker.py --project_name test1 --config "{...}" --cpu_limit 4
```

直接启动 Julia 服务：

```powershell
pixi run -e julia julia --project=src/julia src/julia/iitm_server.jl --port 2700 --root .
```

## 测试

Python/Mie 轻量协议测试：

```powershell
pixi run -e mie python -m unittest discover -s tests -p "test_mie_*.py"
```

Julia proxy/exact 场族协议测试：

```powershell
pixi run -e julia julia --project=src/julia tests/julia_field_contract.jl
```

Julia 内置物理和渲染回归测试：

```powershell
pixi run -e julia julia --threads auto --project=src/julia src/julia/iitm_physics.jl
```

Python 语法检查：

```powershell
pixi run -e mie python -m py_compile src/mie_core.py src/mie_numba.py src/mie_worker.py
pixi run -e gui python -m py_compile src/gui.py src/iitm_http_worker.py src/iitm_renderer.py
```

## 当前主要剩余问题

详见 [todo.md](todo.md)。

当前最高优先级剩余项主要是：

- 为 `proxy` 和 `exact` 场族补充严格物理定义和单位说明。
- Mie exact 场尚未实现。
- Julia `proxy_only` 当前仍会收集 voxel fields，存在不必要的 exact 累积开销。
- Mie 离线渲染链路仍然偏重。
- Python/Mie 缺少足够的自动回归测试。
- README、TODO、GUI 标签和指标命名需要继续保持同步。

## 仓库协作约定

- 当前 GitHub 远端仓库用于保存当前代码基线与日常备份。
- 后续进行大规模代码修改、重构或高风险试验前，应先将本地稳定状态提交并推送到远端。
- 本地运行产物（如 `outputs/`、`temp/`、`log/`、`.pixi/`）不作为备份内容，远端只保留源码、配置、输入样例与必要依赖。
