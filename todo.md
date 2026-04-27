# TODO

本文件只保留当前还没有完成的事项。已经完成的内容已同步到 `Readme.md`。

## 当前重构状态（2026-04-27）

“代理场 / 精确场”双轨重构已经进入中段：

- GUI 已有 `field_compute_mode = proxy_only | exact_only | both`。
- GUI 已有场族切换状态 `field_family = proxy | exact`。
- Mie worker 已返回统一字段：
  - `field_catalog`
  - `available_field_families`
  - `requested_field_compute_mode`
  - `effective_field_compute_mode`
- Julia HTTP 链路已返回同一套字段。
- Julia 后端已支持 `proxy` 和 `exact` 双场族。
- Mie 后端仍只支持 `proxy`，请求 `exact_only` 或 `both` 时会显式回落到 `proxy_only`。

当前重构的核心剩余目标：

- 为 `proxy` 和 `exact` 场族给出严格、稳定、可引用的物理定义。
- 接入 Mie exact 场，或明确长期不支持的边界。
- 降低 Julia `proxy_only` 模式下不必要的 exact 累积开销。
- 补齐 Python/Mie 自动回归测试。

## P0

### T1. 明确 `proxy` 与 `exact` 场族的物理定义

当前：

- `Mie proxy`
  - `beta_back`
  - `beta_forward`
  - `depol_ratio`
  - 来源：三维密度场乘以按层 Mie LUT。
- `Julia proxy`
  - `beta_back`
  - `beta_forward`
  - `depol_ratio`
  - `density`
  - 来源：三维密度场乘以 scatter 参考量和几何缩放。
- `Julia exact`
  - `beta_back`
  - `beta_forward`
  - `depol_ratio`
  - `event_count`
  - 来源：Monte Carlo 散射事件处的 detector-conditioned response accumulation。

需要补充：

- 每个场族的正式术语。
- 每个字段的单位或无量纲说明。
- 哪些字段可用于定性展示，哪些字段可做定量分析。
- `proxy` 与 `exact` 的积分量是否允许互相比较。

### T2. 明确 `depol_ratio` 的定义和数值范围

当前：

- Mie proxy 的 `depol_ratio` 是按层 LUT 生成的代理场。
- Julia proxy 的 `depol_ratio` 是后向退偏参考量展开得到的代理场。
- Julia exact 的 `depol_ratio` 来自 voxel 累积 Stokes 分量。
- Python 和 Julia 的 `safe_depol_ratio()` 当前只钳制下限，不钳制上限；数值噪声或定义边界下可能略大于 `1`。

需要决定：

- 退偏比是否在所有对外接口中强制 clamp 到 `[0, 1]`。
- 若不 clamp，GUI 和文档如何解释略大于 `1` 的参考值。
- `depol_ratio` 字段名是否需要按场族细分，例如 `depol_proxy` / `depol_exact`。

### T3. 再次核实 `forward_back_ratio` 的对外表述

当前 `forward_back_ratio` 是工程 proxy，不是严格的前后向能量比。

需要决定：

- 保持名称不变，但在 UI/README 中继续明确标注 `proxy`。
- 或改成更保守的名字，例如 `forward_back_proxy`。
- 如果保留旧名，需要定义兼容周期和旧项目迁移策略。

## P1

### T4. 接入或正式放弃 Mie exact 场

当前 Mie 后端只支持 `proxy_only`。

可选方向：

- 在 `mie_numba` 中补 voxel-level detector response accumulation，生成 `exact` 场。
- 或明确 Mie 后端长期只承担快速 proxy 基线角色，不接入 exact 场。

需要注意：

- Mie exact 场会增加 Numba 内核复杂度和内存写入压力。
- 若接入，需要同步更新 `field_catalog`、NPZ/HTML 输出策略和 GUI 标签。

### T5. 降低 Julia `proxy_only` 模式的性能开销

当前 `src/julia/iitm_server.jl` 的 `step3_mc()` 会无条件设置：

```julia
mc_cfg_local["collect_voxel_fields"] = true
```

这意味着即使用户请求 `field_compute_mode = proxy_only`，Julia 仍会收集 exact 场所需的 voxel observables。

需要改为：

- `proxy_only`：默认不收集 voxel fields。
- `exact_only` / `both`：收集 voxel fields。
- 如果未来某些指标需要 voxel fields，应显式列出原因。

### T6. 进一步统一 `sigma_*_ref` 与 `beta_*_ref` 的对外协议

当前已完成：

- `mie_core` 输出 `sigma_back_ref / sigma_forward_ref`。
- Julia HTTP 指标输出 `sigma_back_ref / sigma_forward_ref`。
- Mie Numba 保留 `beta_back_ref / beta_forward_ref`，表示层平均后的体系数。

仍需继续清理：

- GUI 指标展示需要显式区分“截面量”和“体系数量”。
- 日志、旧项目文件和后处理脚本需要检查旧键名依赖。
- 为 `sigma_*_ref` 和 `beta_*_ref` 补充单位说明。

### T7. 给参数建立“生效矩阵”

当前很多参数虽然可在 GUI 中看到，但并非在两条链路中同样生效。

建议建立一张参数矩阵，逐项标明：

- 参数名
- GUI 是否可见
- Mie 是否生效
- Julia 是否生效
- 生效于哪一层：
  - 散射求解
  - Monte Carlo
  - 三维场生成
  - 预览导出

### T8. 审查旧键名兼容策略

当前仍存在“新命名 + legacy alias”并存状态，例如 Julia NPZ 中保留主场族的 `beta_back / beta_forward / depol_ratio` alias。

需要决定：

- 是否保留兼容层。
- 保留多久。
- 是否在加载旧项目时做自动迁移。
- 是否为旧键名打印迁移提示。

## P2

### T9. 优化 Mie 离线渲染链路

当前 Mie 预览已经支持三通道，但离线导出仍然偏重：

- 多视角。
- 多字段。
- 每次都走 `PyVista + Panel` 嵌入式 HTML。

可考虑的方向：

- 只导出当前所需视图。
- 改用更轻量的数据包 + 前端切换。
- 让 Mie 侧逐步靠近 Julia 的 `NPZ + 浏览器端渲染` 方案。

### T10. 评估 Mie `generate_field()` 中的大数组开销

当前 Mie 侧会显式生成：

- `X/Y/Z`
- `density`
- `beta_ext`
- `beta_back`
- `beta_forward`
- `depol_ratio`

在高 `grid_dim` 下，这部分会迅速成为内存瓶颈。

需要评估：

- 哪些数组可以延迟生成。
- 哪些可以只在渲染时切片生成。
- 哪些可以只保存标量场，不保留中间网格。

### T11. 继续复查 Numba 并行统计的稳定性

当前静态检查显示：

- fast kernel 主要使用 Numba reduction。
- 带空间/角度记录的 kernel 未开启 `parallel=True`。

仍建议继续审查：

- 不同线程数下结果波动是否合理。
- 回波角分布和空间统计在并行路径下是否存在竞争风险。
- 是否需要为 Python/Mie MC 增加 seed 参数以便回归测试。

## P3

### T12. 增加自动回归测试覆盖

已新增轻量测试：

- Python/Mie 协议测试：`tests/test_mie_contracts.py`
- Julia 场族协议测试：`tests/julia_field_contract.jl`

仍需补充：

- `Mie mono` 与 `Julia sphere` 的单次散射一致性。
- `lognormal` 聚合一致性。
- 高光学厚度下 `R_back / R_trans / depol` 的收敛性。
- `Mie` 三通道 HTML 是否都能生成。
- `sigma_*_ref` 与 `beta_*_ref` 命名口径是否被误改。

### T13. 为文档和协议建立更新约束

后续至少应满足：

- 指标字段变更必须同步 README。
- 预览输出变更必须同步 TODO/说明。
- 后端协议变更必须至少有一个 smoke test。
- `field_catalog` 的字段增删必须同步 GUI 标签。
