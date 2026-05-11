# TODO

当前日期状态：`2026-05-12`。

本文件只保留偏振字段重构后仍需推进的事项。当前主线已完成 Mie / IITM 距离门偏振通道同构输出：`echo_parallel_power`、`echo_perpendicular_power`、`linear_depol_ratio`，旧 `echo_depol` 保留为 legacy 偏振损失诊断字段。

## 已完成里程碑

### ✓ 阶段一：物理闭合测试（2026-05-12）

**范围**：WF1 能量守恒、WF2 P(R) 解析回归、WF3 光子收敛、WF4 跨后端一致性  
**状态**：全部通过（9 tests OK）  
**详情**：见 `Readme.md` § 当前验证数据 § 阶段一：物理闭合测试

**交付物**：
- `tests/test_physics_closure.py`：WF1 + WF2 Python 端（6 tests）
- `tests/test_convergence.py`：WF3 Python 端（2 tests，slow）
- `tests/test_cross_backend.py`：WF4 L1/L2/L3 Python 端（7 tests，L2/L3 slow）
- `tests/julia_physics_closure.jl`：WF1 + WF2 Julia 端
- `tests/generate_iitm_reference.jl` + `tests/fixtures/iitm_sphere_reference.json`：WF4-L1 参考值

**关键发现**：
- 两后端能量守恒偏差 < 1e-10（浮点精度级）
- τ≈0.4 时多散射导致 P(R) 斜率偏离单散射理论 ~54%，已在容差中体现
- 球形粒子 τ≈0.4 时 median(linear_depol_ratio)≈0.15（多散射偏振混合）
- Mie 与 IITM 散射参量一致性：σ_ext/σ_sca < 1%，ω₀ < 0.002，g < 0.01，M11 相关 > 0.9999
- 跨后端 R_back 相对偏差 ~10%（不同密度场实例），echo_power 斜率偏差 ~2.5%

---

## 当前优先级

### 阶段二：数据质量与协议稳定性（进行中）

**目标**：证明”可稳定产数、可长期复用”，为后续训练数据、跨后端对比与历史样本兼容奠定基础。

**核心任务**：
1. **T1 Dataset 大样本验收**（P0，必须）
2. **T2 字段命名与协议收口**（P0，必须）
3. **T0 偏振字段验收加固**（P0，可选，已部分完成）

**预期交付**：
- Dataset runner 端到端验收报告（Julia/IITM 优先）
- 字段协议版本规范（`proxy/exact/echo` 定义、单位说明、legacy alias 保留周期）
- 质量分级建议（低事件数 bin、零平行通道、极低回波功率）

**影响范围**：
- 数据质量与复现性
- 前后端/后处理接口稳定性
- 后续性能优化和功能迭代成本

---

## P0：必须完成（阶段二核心）

### T1. Dataset 大样本验收

**动机**：阶段一证明了”物理正确”，但未验证”批量产数稳定性”。需确保 dataset runner 在大样本下不会产生异常值、NaN、或质量退化。

**任务清单**：
- [ ] Julia/IITM dataset runner 做端到端大样本验收（建议 ≥100 样本，覆盖 sphere/cylinder/spheroid）
- [ ] 检查 `quality.json` 中 `median_linear_depol_ratio`、`max_linear_depol_ratio` 对低事件数 bin 的稳定性
- [ ] 给 dataset manifest 增加字段协议版本（`schema_version: “2.0”`），便于区分旧 `echo_depol` 主字段样本和新 `linear_depol_ratio` 主字段样本
- [ ] Mie Numba seed 当前仍只承诺统计复现，需后续评估严格复现方案（可推迟至 P1）
- [ ] 补充 dataset 质量分级逻辑：标记低事件数 bin、零平行通道、极低回波功率样本

**验收标准**：
- 100 样本无 NaN、无 inf、无负功率
- `quality.json` 中 `median_linear_depol_ratio` 在合理范围内（sphere < 0.3，cylinder/spheroid 视 aspect ratio）
- manifest 包含 `schema_version` 字段
- 质量分级逻辑可区分”可用于训练”与”仅供诊断”样本

**预计工作量**：3-5 天（含 Julia runner 调试、质量指标设计、文档更新）

---

### T2. 字段命名与协议收口

**动机**：当前 `proxy/exact/echo` 三类结果定义散落在代码注释中，字段单位未统一标注，legacy alias 无明确保留周期。需正式化协议，避免后续接口变更引发兼容性问题。

**任务清单**：
- [ ] 在 `Readme.md` 新增”字段协议规范”章节，明确：
  - `proxy`：快速代理场，基于 β_back/β_forward/depol_ratio 解析公式，无 Monte Carlo 事件
  - `exact`：detector-conditioned response field，由 MC 散射事件处的探测锥响应累积得到
  - `echo`：range-gated lidar observation，距离门统计结果
- [ ] 给所有导出字段补单位或无量纲说明（建议在 NPZ 文件 metadata 或 README 表格中维护）
- [ ] 给 legacy alias 设计保留周期（建议：`echo_depol` 保留至 2026-12-31，之后仅保留 `linear_depol_ratio`）
- [ ] 检查旧项目、后处理脚本、测试和 GUI 是否依赖旧键名（grep `echo_depol`、`depol_ratio` 等）
- [ ] 字段变更必须同步 README、GUI 标签和自动测试

**验收标准**：
- `Readme.md` 包含”字段协议规范”章节，定义清晰
- 所有导出字段有单位说明（表格或 metadata）
- legacy alias 保留周期明确，有弃用警告机制
- 旧键名依赖已清理或标记为 deprecated

**预计工作量**：2-3 天（含文档编写、代码审查、GUI 标签更新）

---

### T0. 偏振字段验收加固（可选，部分已完成）

**状态**：阶段一已完成球形粒子低退偏 sanity check（WF3）和跨后端趋势一致性（WF4-L3），剩余任务可推迟至 P1。

**剩余任务**：
- [ ] 增加更大 photon 数下的 `linear_depol_ratio` 统计稳定性验收（建议 ≥5M 光子，slow test）
- [ ] 明确 Stokes `Q` 正方向与”平行偏振通道”的坐标约定，并在 README 中持续维护

**预计工作量**：1-2 天（若优先级降低可推迟）

---

## P1：重要但可推迟

### T3. 三维结果质量评估

**动机**：当前 `exact` 三维场和 `echo` 距离门结果缺少系统性质量评估，需补充方差、置信区间、相对误差和 usable bin 摘要。

**任务清单**：
- [ ] 增加大样本统计稳定性验收（与 T1 部分重叠）
- [ ] 补充 `echo_power`、`linear_depol_ratio` 随 photon 数收敛测试（扩展 WF3）
- [ ] 继续完善方差、置信区间、相对误差和 usable bin 摘要
- [ ] 对低事件数、零平行通道、极低回波功率 bin 给出质量分级建议（与 T1 部分重叠）

**预计工作量**：3-4 天

---

### T4. Mie exact 性能优化

**动机**：当前 Mie exact 内核为串行，`escape_transmittance` 对每个 detector cone 方向做步进积分，性能瓶颈明显。

**任务清单**：
- [ ] 设计线程局部 voxel buffer 并归并
- [ ] 评估 sparse voxel map，减少低事件数场景的大数组写入
- [ ] 优化 `escape_transmittance`：路径积分缓存、可控步长策略、层内近似
- [ ] 给 detector cone quadrature 默认值补收敛测试
- [ ] 对超大 photon 数做日志节流

**预计工作量**：5-7 天（需性能 profiling 和并行化设计）

---

## P2：长期维护

### T5. 参数生效矩阵

建立矩阵，逐项标明参数名、GUI 可见性、Mie/IITM 生效性、生效阶段（散射求解、Monte Carlo、三维场生成、渲染导出）。

优先覆盖：`shape_type`、`r_eff`、`axis_ratio`、`Nr`、`Ntheta`、`n_radii`、`mie_layer_count`、`mie_n_radii`、`mc_use_3d_density`、`mc_density_sampling`、`field_compute_mode`、detector cone 参数。

**预计工作量**：2-3 天

---

### T6. 渲染与内存

- [ ] 抽出共用 HTML/JS 渲染模板
- [ ] 增加浏览器渲染 smoke test
- [ ] 检查大 `grid_dim` 下的浏览器内存占用
- [ ] 评估按需加载字段，减少一次性读取全部数组
- [ ] 评估按 field family 输出子集，减少不必要的 3D arrays

**预计工作量**：3-4 天

---

### T7. 技术债

- [ ] 将 `np.trapz` 替换为 `np.trapezoid`
- [ ] 清理历史 mojibake 注释
- [ ] 检查 Windows CRLF 与 `.gitattributes` 约束
- [ ] 清理未使用的旧渲染依赖或明确保留原因

**预计工作量**：1-2 天

---

## 下一步工作计划（2周内）

### 第 1 周：T1 Dataset 大样本验收

**目标**：验证 Julia/IITM dataset runner 批量产数稳定性，建立质量分级逻辑。

**具体任务**：
1. 运行 Julia dataset runner 生成 ≥100 样本（sphere/cylinder/spheroid 各 30+）
2. 检查 `quality.json` 中 `median_linear_depol_ratio`、`max_linear_depol_ratio` 分布
3. 给 dataset manifest 增加 `schema_version: “2.0”` 字段
4. 补充质量分级逻辑：标记低事件数 bin（< 30）、零平行通道、极低回波功率（< 1e-10）
5. 编写验收报告，记录异常样本和质量分布

**交付物**：
- 100+ 样本 dataset（`outputs/dataset_phase2/`）
- 质量分级逻辑代码（`src/dataset_quality.py` 或 Julia 等价）
- 验收报告（`temp/dataset_validation_report.md`）

---

### 第 2 周：T2 字段命名与协议收口

**目标**：正式化字段协议，清理 legacy alias 依赖，更新文档和 GUI。

**具体任务**：
1. 在 `Readme.md` 新增”字段协议规范”章节
2. 给所有导出字段补单位说明（表格或 NPZ metadata）
3. 设计 legacy alias 保留周期（`echo_depol` 至 2026-12-31）
4. grep 检查旧键名依赖，清理或标记 deprecated
5. 更新 GUI 标签和自动测试

**交付物**：
- `Readme.md` § 字段协议规范
- 字段单位表格（Markdown 或 NPZ metadata）
- legacy alias 弃用警告机制（代码 + 文档）
- GUI 标签更新 PR

---

## 完整工作规划（3个月）

### 月度 1（当前）：阶段二核心任务

- **Week 1-2**：T1 Dataset 大样本验收 + T2 字段协议收口
- **Week 3**：T0 偏振字段验收加固（可选）
- **Week 4**：阶段二总结，编写验收报告

**里程碑**：阶段二完成，数据质量与协议稳定性达标。

---

### 月度 2：性能优化与质量评估

- **Week 5-6**：T4 Mie exact 性能优化（线程局部 buffer、sparse voxel map）
- **Week 7**：T3 三维结果质量评估（方差、置信区间、usable bin 摘要）
- **Week 8**：性能与质量验收，编写优化报告

**里程碑**：Mie exact 性能提升 2-3x，质量评估体系完善。

---

### 月度 3：长期维护与文档完善

- **Week 9**：T5 参数生效矩阵
- **Week 10**：T6 渲染与内存优化
- **Week 11**：T7 技术债清理
- **Week 12**：项目文档全面审查，准备阶段三规划

**里程碑**：项目进入稳定维护期，文档完善，技术债清零。

---

## 风险与依赖

### 风险

1. **Dataset runner 稳定性**：Julia/IITM runner 可能在大样本下暴露未知 bug，需预留调试时间。
2. **Legacy alias 清理**：旧键名依赖可能散落在未追踪的后处理脚本中，需与团队确认。
3. **性能优化复杂度**：Mie exact 并行化可能引入竞态条件，需仔细设计和测试。

### 依赖

1. **Julia 环境稳定性**：T1 依赖 Julia dataset runner 正常运行。
2. **团队协作**：T2 需与 GUI 开发者、后处理脚本维护者协调。
3. **计算资源**：T1 和 T4 需要足够的计算资源（CPU/内存）。

---

## 总结

**当前状态**：阶段一物理闭合测试全部完成，项目已证明”物理正确”。

**下一步重点**：阶段二数据质量与协议稳定性，确保”可稳定产数、可长期复用”。

**核心任务**：T1 Dataset 大样本验收（Week 1）+ T2 字段协议收口（Week 2）。

**长期规划**：3 个月内完成性能优化、质量评估、文档完善，项目进入稳定维护期。
