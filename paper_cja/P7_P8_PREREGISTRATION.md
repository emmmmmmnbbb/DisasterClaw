# E4 主动重观测：新系统预注册（P6）

- `schema`: ARS Material Passport 9
- `artifact_type`: pre-registration (frozen before P8 holdout; to be signed by author)
- `paper`: `paper_cja`
- `created_at`: 2026-08-24
- `status`: **草案，待作者签字。P8（一次性 holdout）不得在签字前运行。**

---

## 0. 一句话

本预注册冻结「新系统」下 E4 主动重观测实验的配置、指标、判据与 holdout 事件，
使 P8 的结果不受 P7 pilot 结果的影响。旧 D6（旧观测模型 + 旧检测器）是另一套
系统，其 holdout 消费不适用于本系统。

## 1. 新系统定义（相对旧系统的三处变更）

| 维度 | 旧系统 | 新系统（本预注册对象） |
|---|---|---|
| 观测模型 | 单瓦片裁块 + 合成高斯模糊阶梯 | **视场收缩式降高**（fov_ladder：1330m/1.5m·px⁻¹ → 443m/0.5m·px⁻¹，= 一整瓦片） |
| 感知底座 | 自训练 ResNet34 双塔（事件不相交） | **xView2 冠军预训练权重（leaky，见 §6）** |
| 题目作用域 | 当前视场 | **ROI-scoped**（锚定一张有标注 post 瓦片） |

## 2. 冻结配置集合

沿用 D6 冻结的 V0/V1/A0/A2/A3，不新增 A1/A4/A5：

| 配置 | evidence | search | reobs | 角色 |
|---|---|---|---|---|
| V0_RAW | raw | 0 | 0 | 原始图像基线 |
| V1_STRUCT | struct | 0 | 0 | 结构化证据基线 |
| A0_HOLD | state | 6 | 0 | 搜索但**不**重观测 |
| A2_ALWAYS | state | 6 | 2 | 重观测上限对照 |
| A3_ENTROPY | state | 6 | 2 | 熵触发主动策略 |

## 3. 指标与判据（预先写死）

**主指标**（最终答案）：
- 准确率（弃答算错）、answer-only 准确率、弃答率。
- `n_correctable` / `n_harmful` / `n_both_correct` / `n_neither_correct`（配对口径）。
- 每单位额外观测的纠正数（`corrected / triggered`）。

**可识别性前置判据**（计划 §11.6，P4 已在 leaky 底座上测得）：
- P4 逐建筑 `n_correctable=375`（48 ROI / 7695 建筑），题目层面 `n_correctable=13`，
  判定 **MARGINAL**。据此预写：若 pilot 复现题目层面 `n_correctable` 仍为个位数，
  E4 按**分支 C（不可识别）**报告，不得写成「主动重观测改善答案」。

**分支口径**（计划 §3.4）：
- 分支 A（正向）：配对 CI 不含 0 且多事件方向一致——**当前 P4 证据不支持预期会到 A**。
- 分支 C（不可识别）：`n_correctable` 过少。当前最可能落点。
- 分支 B（概率质量改善）：答案不变但 Brier/NLL/ECE 改善——需补报概率指标。

**诚实下限**：无论哪条分支，摘要与结论不得出现「复检提高了任务成功率」；最多写
「在已披露感知底座下，重观测机制可测（触发通道接通），但效应小且有得有失」。

## 4. 事件划分（P7 pilot vs P8 holdout）

| 用途 | 事件 | ROI 候选（覆盖≥0.8） |
|---|---|---|
| P7 pilot | hurricane-michael + palu-tsunami | 89 + 8 |
| P8 holdout（**建议**） | **moore-tornado + nepal-flooding** | 160 + 58 |

**理由**：
- holdout 与 pilot 事件完全不相交（同一智能体、不同事件，检验跨事件稳健性）。
- pinery-bushfire 在覆盖≥0.8 下仅 4 个 ROI 候选，统计上不成立，**剔除**。
- 旧 D6 的 holdout 是 moore/nepal/pinery，本系统复用 moore/nepal 属于「新系统=
  新实验」下的重选，**必须在论文局限中披露**（holdout 事件与旧系统重叠）。

**待作者确认**：是否接受「holdout = moore + nepal，剔除 pinery」。

## 5. 一次性消费规则

- P8 holdout 只跑一次；其结果不得反过来改 prompt、阈值、动作预算、题库或事件划分。
- pilot（P7）是开发集，允许据此诊断与修复实现缺陷，但不得据此改判据。

## 6. leaky 披露（route 1 决策）

- 感知底座（xView2 冠军权重）在 xBD 官方划分上训练，**见过全部评测事件**。
- 因此本文**不主张**跨事件泛化，检测器降级为「公开披露的感知组件」，
  其绝对精度（overall 0.697）不是贡献。
- 「重观测是否值得」的结论只在「模型已学会这些事件」的前提下成立，摘要/局限须写明。

## 7. 复现与数据完整性

- 题库：`backend/data/benchmarks/agent_vqa_testset_v2*.json`（schema agent-vqa/2.0）。
- 运行环境快照：`bench_agent_vqa.py` 的 manifest.json（env + git + hash）。
- `DETECTOR_BACKEND=xview2_first`、`MOSAIC_VIEW=1`、`DEFAULT_HOVER_ALTITUDE_M=1330`。

## 8. 签字

- [ ] 作者确认 holdout 事件（moore + nepal，剔除 pinery）
- [ ] 作者确认配置集合（V0/V1/A0/A2/A3）
- [ ] 作者确认 leaky 披露口径（route 1）

签字后 P8 方可运行。
