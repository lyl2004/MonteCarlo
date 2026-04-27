# MonteCarlo

`MonteCarlo` 是一个面向光散射问题的异构仿真工作台，当前维护两条主链路：

- `Mie + Monte Carlo + Numba`
  适用于球形粒子与等效粒径分布的快速仿真
- `IITM/T-Matrix + Monte Carlo + Julia`
  适用于非球形粒子的散射计算与三维场展示

项目通过 `Pixi` 管理环境，通过 GUI 统一调度 Python 和 Julia 后端。

## 当前状态

截至 `2026-04-23`，当前代码状态可以概括为：

- GUI 已可在 `mie` 和 `iitm` 两个后端之间切换
- `Mie` 链路当前是球形粒子的主基线实现
- `Julia` 链路当前是非球形粒子的主计算实现
- 两条链路都已支持三类结果输出：
  - `beta_back`
  - `beta_forward`
  - `depol_ratio`
- `Mie` 与 `Julia` 在以下核心量上已完成一轮收敛：
  - `sigma_ext`
  - `sigma_sca`
  - `omega0`
  - `g`
  - `M11/M12/M33/M34`
  - Monte Carlo 后向统计下的 `depol`

## 本轮已完成的关键修复

### 1. Mie / Julia 计算一致性收敛

已修复并统一以下问题：

- `Mie mono` 单半径时的积分错误
- `Julia lognormal` 粒径聚合与 Python 侧不一致的问题
- `Python / Julia / Numba` 三侧 Mueller 角度查表统一为线性插值
- `depol` 在无后向散射时的回退语义统一
- `depol` 的强度权重语义统一到与 `Numba` 一致

当前结论：

- 单次散射层面的核心散射量已基本对齐到浮点精度级别
- 多次散射下的 `R_back / R_trans / depol` 已收敛，剩余差异主要来自 Monte Carlo 采样噪声

### 2. Mie 三通道输出补齐

此前 `Mie` 实际只贯通了后向场。现在已补齐：

- 核心观测量：
  - `sigma_back_ref`
  - `sigma_forward_ref`
  - `forward_back_ratio`
  - `depol_back`
  - `depol_forward`
- 3D 场与 LUT：
  - `beta_back`
  - `beta_forward`
  - `depol_ratio`
  - `lut_back`
  - `lut_forward`
  - `lut_depol`
- 离线预览文件：
  - `render_main.html` 等：后向场
  - `render_main__beta_forward.html` 等：前向场
  - `render_main__depol_ratio.html` 等：退偏场

GUI 现已支持在 `Mie` 后端下切换这三类场。

### 3. 命名口径修正

此前存在同名不同物理量的问题：

- `mie_core.run_simulation()` 中的 `beta_back_ref / beta_forward_ref`
  实际返回的是参考散射截面，而不是体散射系数

现已改为：

- `sigma_back_ref`
- `sigma_forward_ref`

`Numba` 侧保留 `beta_back_ref / beta_forward_ref`，它们表示层平均后的体系数，语义与单位不同。

## 当前两条链路的设计定位

### Mie 链路

适合做：

- 球形粒子快速基线
- 高吞吐 Monte Carlo 传输
- 参数扫描
- 与 Julia 非球形链路做一致性对比

当前输出包括：

- 核心散射参数
  - `sigma_ext`
  - `sigma_sca`
  - `omega0`
  - `g`
  - `M11/M12/M33/M34`
- 观测代理量
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
- 三维场
  - `beta_back`
  - `beta_forward`
  - `depol_ratio`

### Julia 链路

适合做：

- 非球形粒子散射
- `cylinder / spheroid / sphere`
- EBCM 优先、IITM 兜底的 T-Matrix 求解
- 更丰富的求解器诊断与场输出

当前输出包括：

- 与 Mie 对齐的核心散射量
- 三维场
  - `beta_back`
  - `beta_forward`
  - `depol_ratio`
- 求解器诊断
  - `solver_requested`
  - `solver_used`
  - `fallback_used`
  - `fallback_reason`
  - `solver_path_summary`
  - `ebcm_count / iitm_count`
  - `nmax_min / nmax_max`
  - `ebcm_loss_estimate_max / mean`

## 当前仍需明确的设计边界

下面这些点现在已经“工程上可用”，但还不是最终物理定义：

- `forward_back_ratio`
  当前是工程代理量，不是严格的前后向能量比
- `beta_back` / `beta_forward` 三维场
  当前用于展示与诊断，不应直接等价理解为严格观测体场
- `depol_ratio` 三维场
  当前是按层 LUT 生成的代理场，不是由体素级传播历史反演出的真实退偏场
- `Mie` 与 `Julia` 的三维场
  两侧现在在展示语义上已基本对齐，但都仍属于“展示/诊断场”，不能直接等价理解为严格观测场

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
├─ src/
│  ├─ gui.py
│  ├─ mie_core.py
│  ├─ mie_numba.py
│  ├─ mie_worker.py
│  ├─ iitm_http_worker.py
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

### 启动 GUI

```powershell
pixi run -e gui python src/gui.py
```

### 直接运行 Mie worker

```powershell
pixi run -e mie python -u src/mie_worker.py --project_name test1 --config "{...}" --cpu_limit 4
```

### 直接启动 Julia 服务

```powershell
pixi run -e julia julia --project=src/julia src/julia/iitm_server.jl --port 2700 --root .
```

## 当前主要剩余问题

详见 [todo.md](D:/Code/Python/MonteCarlo/todo.md)。

当前最高优先级剩余项主要是：

- `depol_ratio` 三维场的物理语义仍是代理量
- `Mie` 与 `Julia` 的体场定义还不够严格
- `Mie` 离线渲染链路仍然偏重
- `README / GUI / 指标命名` 需要持续与代码同步
