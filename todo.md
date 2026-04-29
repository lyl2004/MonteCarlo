# TODO

本文件只保留当前仍需继续推进的事项。已经完成的内容已同步到 `Readme.md`。

当前日期状态：`2026-04-27`。

## 当前完成状态

- GUI 支持 `mie` 和 `iitm` 两个后端。
- 两套后端均支持 `field_compute_mode = proxy_only | exact_only | both`。
- 两套后端均返回：
  - `field_catalog`
  - `available_field_families`
  - `requested_field_compute_mode`
  - `effective_field_compute_mode`
  - `primary_field_family`
- 两套后端均输出同构格式：
  - `density.npz`
  - `render_main.html`
  - `render_top.html`
  - `render_front.html`
- Mie exact 已按 Julia 当前 detector-conditioned response field 语义接入。
- Mie MC 已完成 z-slab local majorant、exact 原地累积、8192 batch 优化。
- Julia `proxy_only` 已避免不必要的 voxel exact 收集。
- Python 和 Julia 的退偏比均已钳制到 `[0, 1]`。
- 已有测试：
  - Python/Mie 协议测试：`tests/test_mie_contracts.py`
  - Python/Mie 距离门观测合约测试：`tests/test_mie_lidar_observation.py`
  - 数据集样本协议测试：`tests/test_dataset_contract.py`
  - Julia 场族协议测试：`tests/julia_field_contract.jl`
  - Julia 距离门观测合约测试：`tests/julia_lidar_contract.jl`
  - Julia server `field_compute_mode` 测试：`tests/julia_server_contract.jl`
  - Julia 内置物理回归：`src/julia/iitm_physics.jl`

## P0

### T0. 时间门控激光雷达观测算子与样本协议

当前状态：

- Mie/Numba exact 事件累积已新增 range-bin lidar observation。
- Julia/IITM chunk-local MC 已新增同构 range-bin lidar observation，并在主线程归并。
- 两套后端均使用同名输出字段：
  - `range_bins_m`
  - `echo_I / echo_Q / echo_U / echo_V`
  - `echo_power`
  - `echo_depol`
  - `echo_event_count`
  - `echo_weight_sum`
  - `echo_relative_error_est`
- `density.npz` 已保存 lidar observation 数组。
- Mie 与 Julia 均支持 receiver overlap 简化线性模型：
  - `receiver_overlap_min`
  - `receiver_overlap_full_range_m`
- 已新增数据集样本协议与 Mie 批量 runner：
  - `src/dataset_sampling.py`
  - `src/dataset_runner.py`
  - 单样本输出 `observation.npz / truth.json / receiver.json / quality.json / run_config.json`

仍需完成：

- 增加更严格的解析雷达方程趋势回归。
- 增加大样本统计稳定性验收。
- Julia 批量 dataset runner 仍需接入 HTTP/server 执行路径。
- GUI 目前只提供参数入口和运行状态提示，尚未增加 `P(R)` 曲线预览。

### T1. 收口 `proxy` 与 `exact` 的正式物理定义

当前状态：

- `Mie proxy`：三维 density 乘以按层 Mie LUT。
- `Julia proxy`：三维 density 乘以 scatter reference quantities。
- `Mie exact`：MC 散射事件处的 detector-conditioned response accumulation。
- `Julia exact`：MC 散射事件处的 detector-conditioned response accumulation。

仍需完成：

- 给 `proxy` 和 `exact` 写正式术语定义。
- 给每个字段写单位或无量纲说明。
- 明确定性展示、半定量诊断、定量分析的边界。
- 明确 proxy 与 exact 的积分量是否允许比较，以及比较前需要满足的条件。
- 在 GUI 中对 proxy/exact 来源差异给出更清晰标签或提示。

### T2. 收口 `depol_ratio` 语义

当前状态：

- proxy depol 来自 LUT 或 scatter reference。
- exact depol 来自 voxel 累积 Stokes：
  - `1 - sqrt(Q^2 + U^2 + V^2) / I`
- Python 和 Julia 均将结果钳制到 `[0, 1]`。

仍需完成：

- 决定 proxy depol 与 exact depol 是否共享同一解释口径。
- 决定字段名是否继续用 family 区分，或改成更显式名称。
- 在 README 和 GUI 中说明 `event_count=0` voxel 的 depol 不可定量解释。

### T3. 收口 `forward_back_ratio`

当前状态：

- `forward_back_ratio` 是工程 proxy 指标。
- 它不是严格前/后向总能量比。

仍需完成：

- 决定是否保留旧名。
- 如果保留旧名，需要文档中持续标注 `proxy`。
- 如果改名为 `forward_back_proxy`，需要设计兼容期和旧项目迁移策略。

## P1

### T4. 继续优化 Mie exact 性能

当前状态：

- Mie exact 已可用。
- exact kernel 仍为串行 Numba kernel，以避免多个 photon 同时写同一 voxel。
- 已完成：
  - z-slab local majorant。
  - exact arrays 原地累积。
  - fast/exact batch size 提高到 8192。

剩余优化方向：

- 设计线程局部 voxel buffer，然后归并。
- 评估 sparse voxel map，减少低事件数场景的大数组写入。
- 优化 `escape_transmittance`：
  - 路径积分缓存。
  - 可控步长策略。
  - 层内近似。
- 给 detector cone quadrature 默认值补收敛测试。
- 对超大 photon 数做日志节流，避免 batch 日志过密。

### T5. 统一字段单位和指标命名

当前状态：

- `sigma_back_ref / sigma_forward_ref` 已在 Mie core 和 Julia HTTP 指标中使用。
- Mie Numba 仍保留 `beta_back_ref / beta_forward_ref`，表示层平均体系数。
- NPZ 仍保留 legacy alias：
  - `beta_back`
  - `beta_forward`
  - `depol_ratio`
  - `summary`

仍需完成：

- GUI 指标区分截面量和体系数量。
- 检查旧项目 JSON、后处理脚本、日志中是否依赖旧键名。
- 给 legacy alias 设计保留周期。
- 如未来移除 alias，需要加载旧项目时给迁移提示。

### T6. 建立参数生效矩阵

需要逐项标明：

- 参数名。
- GUI 是否可见。
- Mie 是否生效。
- Julia 是否生效。
- 生效阶段：
  - 散射求解。
  - Monte Carlo。
  - 三维场生成。
  - 渲染导出。

优先覆盖：

- `shape_type`
- `r_eff`
- `axis_ratio`
- `nmax_override`
- `Nr`
- `Ntheta`
- `n_radii`
- `mie_layer_count`
- `mie_n_radii`
- `mc_use_3d_density`
- `mc_density_sampling`
- `field_compute_mode`
- detector cone 参数

## P2

### T7. 深化跨后端物理一致性测试

当前测试仍偏合约和单后端回归。

仍需补充：

- `Mie mono sphere` 与 `Julia sphere` 的单次散射对比。
- lognormal 粒径聚合一致性。
- 高光学厚度下：
  - `R_back`
  - `R_trans`
  - `R_abs`
  - `depol`
- Mie 与 Julia 的 NPZ 字段同构深度回归。
- exact 场 `event_count` 和 field sums 的统计稳定性测试。

### T8. 渲染链路去重和浏览器端测试

当前状态：

- Mie 和 Julia 都使用 `density.npz + 浏览器端 Plotly`。
- 两侧 HTML/JS 模板仍有重复。

仍需完成：

- 抽出共用 HTML 模板。
- 给浏览器渲染增加 smoke test。
- 检查大 `grid_dim` 下的浏览器内存占用。
- 评估是否按需加载字段，减少一次性读取所有 arrays。

### T9. 降低大数组内存压力

当前 Mie `generate_field()` 仍显式保留：

- `density`
- `beta_ext`
- `beta_back`
- `beta_forward`
- `depol_ratio`

Julia 场导出在 `both` 模式下也会保存多组 3D arrays。

仍需评估：

- 哪些数组可以延迟生成。
- 哪些数组只在保存 NPZ 时生成。
- 是否将中间 double arrays 更早转换为 Float32。
- 是否支持按 field family 输出子集，减少 `exact_only` 或 `proxy_only` 下不需要的数组。

## P3

### T10. 增强 Mie MC 可复现性

当前 Mie Numba MC 主要使用 Numba 随机数，测试更多验证合约和物理闭合。

仍需完成：

- 评估是否增加 seed 参数。
- 设计不同线程数下的统计容差测试。
- 检查 fast kernel reduction 在不同平台和线程数下的波动。
- 给 `record_spatial / record_back_hist` 路径补充稳定性测试。

### T11. 文档和协议维护约束

后续变更至少应满足：

- 指标字段变更必须同步 README。
- 预览输出变更必须同步 README 和 TODO。
- 后端协议变更必须至少有一个自动测试。
- `field_catalog` 字段增删必须同步 GUI 标签。
- 物理口径变化必须写明是否影响旧项目结果解释。

### T12. 技术债清理

- 将 `np.trapz` 替换为 `np.trapezoid`，消除弃用警告。
- 清理历史 mojibake 注释。
- 检查 Windows CRLF 提示是否需要 `.gitattributes` 进一步约束。
- 评估是否移除未使用的旧渲染依赖或保留为兼容路径。
