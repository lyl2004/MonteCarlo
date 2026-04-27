# TODO

本文件只保留当前还没有完成的事项。已经完成的内容已从 TODO 中移出，并同步到 `Readme.md`。

---

## 当前重构决策（2026-04-23）

### R0. 启动“代理场 / 精确场”双轨重构

目标：

- 保留当前代理场链路，继续用于快速预览、参数扫描和工程诊断
- 新增精确场链路，逐步统一到 detector-conditioned response field
- 在 GUI 左上角增加场族切换，在左侧参数区增加计算模式选择

执行约束：

- 先改前端与中间层协议，再接入后端精确场算法
- 前端、worker、HTTP 协议、README、TODO 使用同一套字段命名
- 过渡期允许后端只返回单一场族，但必须显式声明 `requested` 与 `effective` 模式

分阶段计划：

1. 前端与中间层先行
   - 增加 `field_compute_mode = proxy_only | exact_only | both`
   - 增加预览状态 `field_family = proxy | exact`
   - worker / Julia HTTP 返回统一的 `field_catalog`、`available_field_families`、
     `requested_field_compute_mode`、`effective_field_compute_mode`
2. Julia 双轨输出
   - 保留当前 detector-conditioned 场作为 `exact`
   - 补充 `proxy` 场输出，并统一到前端场族切换
3. Mie 精确场接入
   - 保留当前 LUT 展开场作为 `proxy`
   - 在 `mie_numba` 中补 voxel-level detector response accumulation，生成 `exact`
4. 指标与场统一
   - 标量指标按 `reference / transport / field-response` 分层
   - 展示字段与对应积分指标使用同一来源，避免混义

### R1. 当前阶段任务：只修改前端和中间层

本阶段不做：

- `Mie` 精确场算法实现
- 精确场数值定义最终收口
- 大规模渲染和性能优化

本阶段要完成：

- GUI 配置项、场族切换状态、预览切换逻辑
- Mie / Julia worker 与 HTTP 结果协议扩展
- 为后续算法接入预留稳定字段与元数据接口

## P0

### T1. 明确 `depol_ratio` 三维场的物理定义

当前 `Mie` 和 `Julia` 的 `depol_ratio` 三维场都已经打通：

- 计算
- 保存
- 预览

但它们目前仍然是“按层 LUT 生成的代理场”，不是由真实三维传播历史得到的退偏场。

需要后续明确：

- 是否继续保留它作为工程诊断场
- 是否改名以避免与严格物理退偏场混淆
- 是否要引入更严格的体传播统计定义

### T2. 统一三维场的物理语义说明

当前：

- `beta_back`
- `beta_forward`
- `depol_ratio`

在 `Mie` 和 `Julia` 两条链路里已基本同构，但都不应被直接理解为严格观测场。

需要补充：

- 文档层面的正式术语
- GUI 中的说明文字
- 后续数据分析中的字段解释

### T3. 再次核实 `forward_back_ratio` 的对外表述

当前 `forward_back_ratio` 是工程 proxy，不是严格的前后向能量比。

需要决定：

- 保持名称不变，但在 UI/README 中明确说明
- 或改成更保守的名字，例如 `forward_back_proxy`

---

## P1

### T4. 进一步统一 `sigma_*_ref` 与 `beta_*_ref` 的对外协议

本轮已完成：

- `mie_core` 输出改为 `sigma_back_ref / sigma_forward_ref`
- `Julia HTTP` 指标也同步改为 `sigma_*_ref`

当前仍需继续清理：

- 检查 GUI 展示层是否需要显式区分“截面量”和“体系数量”
- 检查日志、后处理脚本、旧项目文件是否仍依赖旧键名
- 为这些字段补上单位说明

### T5. 给参数建立“生效矩阵”

当前很多参数虽然可在 GUI 中看到，但并非在两条链路中同样生效。

建议建立一张参数矩阵，逐项标明：

- 参数名
- GUI 是否可见
- `Mie` 是否生效
- `Julia` 是否生效
- 生效于哪一层
  - 散射求解
  - Monte Carlo
  - 三维场生成
  - 预览导出

### T6. 审查旧键名兼容策略

本轮修改后，存在一部分“新命名 + 旧命名并存”的历史过渡状态风险。

需要决定：

- 是否保留兼容层
- 保留多久
- 是否在加载旧项目时做自动迁移

---

## P2

### T7. 优化 `Mie` 离线渲染链路

当前 `Mie` 预览已经支持三通道，但离线导出仍然偏重：

- 多视角
- 多字段
- 每次都走 `PyVista + Panel` 嵌入式 HTML

可考虑的方向：

- 只导出当前所需视图
- 改用更轻量的数据包 + 前端切换
- 让 `Mie` 侧也逐步靠近 Julia 的 `NPZ + 浏览器端渲染` 方案

### T8. 评估 `generate_field()` 中的大数组开销

`Mie` 侧仍然会显式生成：

- `X/Y/Z`
- 多个 `N^3` 场

在更高分辨率下，这部分会迅速成为瓶颈。

需要评估：

- 哪些数组可以延迟生成
- 哪些可以只在渲染时切片生成
- 哪些可以只保存标量场，不保留中间网格

### T9. 继续复查 Numba 并行统计的稳定性

虽然当前主要一致性问题已修复，但仍建议继续审查：

- 并行累加统计是否稳定
- 不同线程数下结果波动是否合理
- 回波角分布和空间统计在并行路径下是否存在竞争带来的偏差

---

## P3

### T10. 增加自动回归测试

建议把这轮已经验证过的关键收敛基准固化为测试：

- `Mie mono` 与 `Julia sphere` 的单次散射一致性
- `lognormal` 聚合一致性
- 高光学厚度下 `R_back / R_trans / depol` 的收敛性
- `Mie` 三通道 HTML 是否都能生成
- `sigma_*_ref` 与 `beta_*_ref` 命名口径是否被误改

### T11. 为文档和协议建立更新约束

这轮问题里，很多偏差本质上是：

- 代码已经变了
- 文档和字段定义还停在旧状态

建议后续建立最小约束：

- 指标字段变更必须同步 README
- 预览输出变更必须同步 TODO/说明
- 后端协议变更必须至少有一个 smoke test
