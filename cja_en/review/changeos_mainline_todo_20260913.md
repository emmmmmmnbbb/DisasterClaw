# ChangeOS 二分类闭环实验与论文主线完成计划

日期：2026-09-13。依据：当前 `cja_en` 稿件、`revision_plan_20260912.md` 的 M1--M10，以及 `runs/benchmarks/changeos_binary/full_fixed_20260913_report`。本文区分已经得到的证据、仍属诊断的结果和最终投稿前必须完成的工作。

## 1. 结论

当前实验**可以支撑论文主线，但还不能直接作为最终主实验**。

它已经验证了 DisasterClaw 的闭环评测链能够运行：同一任务经过地理观测、ChangeOS 二分类感知、动作选择、实际复观测和最终回答，并保留逐题记录。四个分片共得到 1,280 条 episode 记录，八个配置各覆盖同一组 160 题，执行错误和 malformed JSON 均为零。

它也给出了与论文主问题直接相关的机制证据。A0_HOLD 的准确率为 71.88%，A3_ENTROPY 为 73.13%，只多答对 2/160 题；A1_RANDOM 同样达到 73.13%，但只执行 84 次复观测，而 A3_ENTROPY 执行 162 次。现有熵规则因此没有显示出统计稳定或成本上的优势。与此同时，A2_ALWAYS 相比 A0_HOLD 纠正 19 题、破坏 17 题；有限候选 hindsight 诊断 O_REF 达到 83.75%，比 A0_HOLD 高 11.88 个百分点。这说明较近视角确实包含可用信息，主要未解问题是何时复观测、怎样保持覆盖以及如何融合新旧证据。

题型结果进一步支持“任务相关主动观测”这条主线：damage 从 A0 的 74.42% 提高到 A3 的 81.40%，count 却从 74.42% 降至 69.77%，spatial 仍只有 45.16%。同一个降高动作对不同问题的价值不同，不能再用通用分类熵充当所有任务的动作价值。

题集还存在明显的答案分布基线：若只知道题型并始终输出该题型的多数答案，整体即可达到 63.75%（presence 74.42%、damage 74.42%、count 72.09%、spatial 22.58%）。A0_HOLD 比这个低成本基线只高 8.13 个百分点。最终实验必须正式加入 majority 和 question-only 对照，并改善或至少完整披露答案分布。

因此，这批结果适合定位为**开发集/诊断实验**，回答 RQ3 和 RQ4 的一部分，并为 RQ5 的新策略设计提供依据。它目前不能证明“所提出的熵复观测机制有效”，也不足以支撑方法型标题中的稳定算法收益。

## 2. 与当前论文和审稿意见的对应关系

| 项目 | 当前状态 | 判断 |
|---|---|---|
| 地理配准环境与闭环执行 | 已实现并完成 160 题运行 | 支撑平台贡献和动作—证据—答案追踪 |
| ChangeOS 替换旧分类器 | 已完成二分类运行，所有策略共享固定底座 | 符合“外部专业感知工具”定位；不得把其分类能力写成本文贡献 |
| RQ1 观测改变带来的纠正与破坏 | 在线已有 19/17 的任务级证据 | 已有重要证据；仍需用 ChangeOS 重建三档感知层分析 |
| RQ2 同预算观测规则比较 | 有单次在线比较 | 随机与熵未严格形成完整、多 seed、共同预算曲线 |
| RQ3 感知变化是否转化为任务收益 | 已得到否定性单次结果 | 可以报告为诊断，不能写成稳定总体结论 |
| RQ4 瓶颈分解 | 有 A0/A2/O_REF 和题型分解 | 仍缺 GT 感知、确定性回答器、完整任务 GT 等分层 oracle |
| RQ5 任务条件化策略 | 尚未实现 | M3 的核心缺口 |
| M4 感知底座曝光与证据一致性 | 换成固定 ChangeOS 后策略比较公平 | 仍须披露官方权重的训练来源；旧 `xview2_first` 四分类结果不能与新主实验拼成一条证据链 |
| M5 生成重复 | 当前只有一次完整策略运行 | 未满足；`temperature>0` 时 Qwen 仍采样，现有 `--seed` 不是逐调用 generation seed |
| M6 任务成本 | 已核对实际复观测位移，并给出增量估算航时 | 只能称 incremental reobservation cost；仍缺完整任务运动、模块实测时延和共同外生预算 |
| M8 题集有效性 | 自动审核 160/160 通过 | 人工审核 0/160，74 题需要作者核对，`eval_role` 为空，不能视为冻结 final |
| M9 主线聚焦 | 当前在线结果能替换“未来工作”叙述 | 应将历史 VLN 结果压到附录 |
| M10 可复现性 | 有题集 SHA、source fingerprint、episode 和报告 | 仍缺运行源码快照、Qwen revision/模型 hash、prompt hash、生成 seed 和完整命令环境 |

## 3. 推荐的论文主线

建议将主问题固定为：

> 在地理配准灾害影像构成的 UAV 主动观测任务中，智能体何时应改变观测范围，才能在有限成本下取得对当前问题有用的证据？

证据链采用同一个冻结的 ChangeOS 二分类底座：

1. 地理配准环境允许智能体连续移动并改变 footprint/effective GSD；
2. 改变视角既能纠正也能破坏感知和任务答案；
3. 通用损伤熵不能代表 presence、count、spatial 等任务的证据需求；
4. 任务条件化策略根据问题类型、目标覆盖、新旧证据和成本选择动作；
5. 用最终任务准确率、纠正/破坏、成本曲线和分层 oracle 判断收益在哪里产生、在哪里丢失。

ChangeOS 只提供建筑定位和损伤/无损伤证据，权重始终冻结。论文贡献应落在环境、闭环测量协议、任务相关动作和证据管理上。

## 4. 必须采用新的确认性测试集

当前 160 题已经被用于查看结果、识别题型弱点并设计下一版策略，因此以后只能作为 development/diagnostic set。即使现在完成人工审核，也不能再用它调好新策略后把同一结果称为未见 final 评估。

确认性实验应从未用于本轮诊断的 tile/ROI 生成新题集，并在运行任何策略之前完成：

- 固定题型、事件、答案分布和模板族；
- 校验起点覆盖、目标引用、ROI 边缘截断、空间坐标系和计数分箱；
- 两位审核者独立检查可回答性与答案，记录一致性和裁决；
- 写入 `eval_role=final`、consumption registry、题集 SHA 和不可变 manifest；
- 在 final 上只执行预先冻结的策略、阈值、预算和统计比较，不根据结果返调。

## 5. 剩余 TODO 与验收门槛

### P0：冻结当前诊断实验

- [x] 将四个 corrected shard、聚合报告、启动命令、环境信息和当前 dirty source 打包为校验快照。归档：`runs/benchmarks/changeos_binary/archive/diagnostic_20260913`。
- [x] 补 ChangeOS checkpoint 路径、权重 hash、官方训练数据/许可核验记录。官方论文确认使用 xBD 训练；仓库 `setup.py` 声明 Apache classifier，但根目录未找到独立 LICENSE，归档不复制权重。
- [x] 补 Qwen checkpoint revision、模型文件 hash、prompt hash、temperature、top-p、repetition penalty。
- [x] 在失效记录和 registry 中把当前题集标为 development/diagnostic；正文统一修改留到 P8，避免在 final 结果前反复改稿。

**验收：**任一表中数字都可回溯到题集、代码、模型、配置和 episode；当前运行不会被后续覆盖。

### P1：消除生成随机性混杂

- [x] 将 generation seed 明确传到实际 Qwen `generate` 调用，按 `(qid, repeat, step, call_role)` 派生；action seed 独立记录。
- [ ] 主实验可采用 greedy generation；若保留采样，完整策略至少 3 个 generation seeds，推荐 5 个。
- [x] 修复 unchanged-view/no-op 重问，记录图像、证据和原始输出 hash；完整 final 重复留到 P6。
- [x] 在分片、恢复和策略调用次数变化后验证同一调用键生成结果一致；真实 Qwen 同键双调用也已逐字一致。

**验收：**不同策略的初始回答可严格配对；结果变化不再混入未控制的 Qwen 采样序列。

### P2：完善任务条件化动作与证据融合

- [x] 定义统一效用：预期任务证据增益减去动作成本和覆盖损失，而不是只看损伤分类熵。
- [x] damage：使用目标建筑的二分类不确定性、目标可见性和预期像素增益决定是否居中/下降。
- [x] presence/count：优先保持完整查询 ROI；视场收窄会截断待计数范围时禁止或惩罚下降。
- [x] spatial：保留目标和参照物的共同覆盖及地理坐标，不用局部 crop 直接回答全局方位。
- [x] 跨视角 memory 按地理 ID 去重并累计证据；新视角不应无条件覆盖旧视角的正确证据。
- [x] 停止规则显式依赖答案是否已具备所需证据、预计增益和剩余预算。

**验收：**在当前 development set 上完成动作可解释性和失败审计；阈值、预算、融合规则随后冻结。

### P3：补齐动作与回答消融

- [x] no-op、仅居中、仅下降、居中+下降、保持宽视场/完整覆盖。
- [x] `D0_RULE`：预测结构化证据加确定性回答器。
- [x] `V2_STATE_VLM` 与 `A0_VLM`：纯 VLM 回答，对照当前 hybrid 回答路径。
- [x] question-only 和各题型多数答案基线，检查模板与答案分布偏置。

开发集四分片已完成并汇总至
`runs/benchmarks/changeos_binary/p3_dev_v1_20260914_report`；每个配置覆盖同一组
160 题，动作消融均执行 86 次复观测并通过共同预算审计。重复启动污染的 shard2/3
已保留原始文件，并分别生成带来源 hash 的 360 行 clean 版本。语言基线产物位于
`runs/benchmarks/changeos_binary/p3_language_baselines_dev_20260913`，其中同题型多数与
question-only 在该开发集上均为 63.75%；该结果明确标记为 same-data diagnostic，不能
外推到 final。动作审计把生成重试次数差异单独披露，不把它误报成动作预算不一致。

**验收：**能够分别判断收益来自图像改变、几何对准、证据聚合、规则回答还是 VLM 生成。

### P4：用 ChangeOS 重建一致的感知证据链

- [x] 在相同建筑/ROI 上重跑 cruise、intermediate、floor 三档 ChangeOS 二分类输出。
- [x] 报告定位、binary accuracy/F1、概率质量、未匹配、correction/harm 和 matched-only 结果。
- [ ] 重建 no-reobservation、random、always、raw entropy、calibrated/conformal 和任务条件化策略的预算曲线。
- [x] 将现有 `xview2_first` 四分类表移到历史/附录，或明确删除；不得复用其温度、APS 阈值和熵收益表。

三档开发集证据位于
`runs/benchmarks/changeos_binary/p4_fov_ladder_dev_20260914`，覆盖题库中的 43 个
唯一 ROI 和 7,238 栋 GT 建筑。cruise/intermediate/floor 的定位召回分别为
22.09%/34.06%/56.52%，全 GT 二分类准确率分别为 20.01%/31.06%/53.65%；
matched-only 准确率分别为 90.56%/91.20%/94.92%。cruise→floor 产生 2,696 次
纠正和 261 次破坏，净纠正 2,435。概率质量仅在 matched GT 上计算，未匹配 GT
在全 GT 准确率中按错误计入，二者没有混用。
旧四分类表已移动到 `cja_en/tables/historical/xview2_first_*`，正文现存引用明确标为
historical diagnostic；它们不再占用 ChangeOS 主证据链的表名，也不得向新二分类实验
传递 temperature、APS qhat 或 expected-entropy 表。

2026-09-15 P4 七策略开发集四片已验证并汇总到
`runs/benchmarks/changeos_binary/p4_policy_dev_v1_20260915_report`。新鲜重跑的 shard3
为 280/280 行、0 执行错误；四片共 1,120 行、每策略 160 题，题库 SHA、源码指纹、
generation/action seed 设置一致，无重复 `(config, qid)`。A0_HOLD、A1_RANDOM、
A2_ALWAYS、A3U_RAW_ENTROPY、A3_ENTROPY、A4_CONFORMAL、T1_TASK 的准确率依次为
67.50%、69.37%、68.13%、68.75%、68.75%、69.37%、70.63%；对应复观测执行数为
0、101、298、199、197、199、86。T1_TASK 相对 A0_HOLD 的配对差为 +3.13 个
百分点，题目 bootstrap 95% CI 为 [-1.87, +8.75] 个百分点，不能判为稳定收益。
`curves.json` 按已发生的 `n_steps` 分组，不是预设预算水平的随机化扫描；七策略
只有预算/成本 operating points，故上方「预算曲线」验收仍未勾选。`O_REF` 是离线
GT 事后选择的结果参照，不能列为在线策略或把它的差值解释为动作因果效应。
这些数值只用于 same-data development/diagnostic，不进入 final 主表。

**验收：**RQ1--RQ5 的主要证据均来自同一个冻结 ChangeOS 版本和二分类语义。ChangeOS 若曾在相关 xBD 数据上预训练，应如实披露，并把论断限定为固定工具条件下的策略比较，不声称未见灾害感知泛化。

### P5：完成分层 oracle 和瓶颈诊断

- [ ] 当前可见范围 GT 感知 + Qwen。
- [x] 预测结构化证据 + 确定性回答器（开发集 D0_RULE 诊断）。
- [ ] 完整任务范围 GT 结构化证据 + Qwen。
- [x] GT 结构化证据 + 确定性回答器的题库流程自检。
- [ ] always-floor/完整覆盖参照和预算受限 hindsight selection。

**验收：**每个诊断只替换一个清楚定义的环节，按题型报告，不把 oracle 差值简单相加成错误比例。

完整 ROI xBD GT + 二分类规则回答器的离线题库自检位于
`runs/benchmarks/changeos_binary/p5_gt_rule_dev_20260915`：43 个 ROI、160 道题中
160 道答案与 GT 规则推导一致，四类题型分别为 presence 43/43、damage 43/43、
count 43/43、spatial 31/31。输入题库与 xBD manifest 的 SHA-256 已记录；这仅核查
题库答案链路，不是独立标注审核，也不代表 Qwen 或在线策略的准确率。
预测 ChangeOS 结构化证据 + 确定性回答器已有 P3 的 `D0_RULE` 独立开发集诊断，
位于 `runs/benchmarks/changeos_binary/p3_dev_v1_20260914_report`：160 题、总体
accuracy 66.87%，presence/damage/count/spatial 分别为
88.37%/58.14%/74.42%/38.71%。四片 D0_RULE 所属运行均为零执行错误、同一题库
SHA 与源码指纹；但 P3 与本轮 P4 的可执行源码指纹不同，不能把二者分数差写成
单一环节的严格因果分解。`D0_RULE` 不调用 Qwen，也没有搜索/复观测预算。
有限预算 hindsight 选择器已实现，三片有效开发集的临时报告位于
`runs/benchmarks/changeos_binary/p5_hindsight_budget_dev_3shards_20260915`（120 题）；
它只在成对 A0/A2 已观察结果中事后挑选正成本纠正，排除零动作但答案不同的题，
不能部署，也不是动作因果效应。完整 160 题报告须等待 P4 新 shard3 验证通过。
四片验证后，完整报告位于
`runs/benchmarks/changeos_binary/p5_hindsight_budget_dev_4shards_20260915`：160 对题、
A0 基线 67.50%、A2 共 298 次已执行复观测、21 个正成本可纠正、19 个正成本
有害、1 个零动作答案差异被排除。在 10%/25% 的 A2 总动作预算下，事后选择
分别达到 76.25%/80.625%，实际只花 28/42 次动作；这些是观察结果中的
GT-informed 上界，不是可部署策略，也不说明真实有限预算策略一定能达到该数值。
「always-floor/完整覆盖参照」和其余分层 oracle 尚未完成，故对应整项仍未勾选。

### P6：冻结 final 并运行确认性实验

主表最低配置：

1. A0_HOLD；
2. A1_RANDOM，使用独立 action seeds；
3. A2_ALWAYS；
4. A3U_RAW_ENTROPY；
5. 任务条件化策略；
6. D0_RULE 和必要的纯 VLM 诊断单独成表。

calibrated entropy 和 conformal 可作为补充比较，不应挤掉 raw entropy、random 或动作消融。所有在线策略共享任务、起点、ChangeOS、Qwen、generation seeds 和外生预算。至少报告 3 个完整 generation seeds；资源允许时固定为 5 个。本轮预定七个在线配置，每 repeat 为 160×7=1,120 episodes；三个完整 repeat 共 3,360 episodes，另加独立 oracle/规则诊断与消融，不能把 repeat 当作新增独立题。

主要终点预先设为全题 task accuracy；次要终点包括各题型准确率、abstention、correction/harm、执行动作数、实际水平/垂直距离、增量估算航时和各模块实测时延。统计保留同题、ROI、事件和 seed 的层级，不把重复 seed 当成新增独立题。

**验收：**给出逐 seed 结果、跨 seed 效应、配对区间、逐事件/题型结果和 accuracy--cost 曲线；无论结果正负都完整报告。

2026-09-15 最初生成**未冻结候选**
`backend/data/benchmarks/agent_vqa_final_candidate_20260915_binary.json`：160 题、44 个
未消费 ROI，与当前开发集/消费登记表均无重叠；100% 自动审核零错误、GT 规则答案
160/160 一致。生成时全部 160 题的 human_review 为 pending，其中 88 题带歧义标志；
后续作者审核状态见下方更新。
目前未消费且满足覆盖阈值的评估事件只有 hurricane-michael、moore-tornado、
nepal-flooding 三个；palu-tsunami 与 pinery-bushfire 在当前登记表和阈值下已无
未用合格 ROI。逐题审核材料位于
`runs/benchmarks/changeos_binary/final_candidate_20260915_review_sheets_v2`，其中
`author_decisions_template.csv` 的 decision/reviewer/note 留空，供作者逐题填写。
候选文件当前 SHA-256 为
`c355270f6ece475beca730647430badf8a2917e1feeae90a0fd262c556d63607`；
该文件与运行报告位于 git 忽略目录，最终冻结时须另行做不可变归档和校验。
原候选不直接用于确认性策略；必须使用下面的审核后哈希绑定版本。

2026-09-15 作者 lc 已在
`runs/benchmarks/changeos_binary/final_candidate_20260915_review_sheets_v2/author_decisions_filled.csv`
逐题人工审核并将 160/160 题判为 approve；作者随后明确接受由
hurricane-michael、moore-tornado、nepal-flooding 构成的三事件 final 设计。
机械导入后的独立题库副本为
`backend/data/benchmarks/agent_vqa_final_candidate_20260915_binary_human_reviewed_validated.json`
（SHA-256 `7179c42d4a7cb0bc8fed642a235bec3c89fca2ff642d2cf38c7b5dc5283890ab`），
审核报告为
`runs/benchmarks/changeos_binary/final_candidate_20260915_human_review_validated.json`
（SHA-256 `3916b50bd13c9eb0d93a9bc6ceb837ef9e88ba559e9112ad5b84755ba486fd45`）。
报告显示自动与人工均 160/160 approved、0 rejected、0 pending；原 88 项歧义标记
照实保留。CSV 原件 SHA-256 为
`2ac2acc156112a6d94f33984b76e8f22d71d4beaa8ba7ad231cabce51551968f`。
ChangeOS 二分类阈值、题库和审核报告已绑定在新冻结配置
`runs/benchmarks/changeos_binary/final_three_event_20260915_frozen_manifest.json`，
不再复用指向开发集的旧 `frozen_manifest.json`。尚需冻结在线策略源码、最终
generation repeats 与统计/预算协议，然后才能启动 P6 确认性运行。
按本待办的最低三完整 generation seeds，拟使用同一 base seed 42000 和
`--generation-repeat 0/1/2`；`--seed 42` 仅用于独立 action RNG，不能把三个 repeat
当作新增的独立题或事件。P6 前瞻性协议已固定在
`cja_en/review/changeos_p6_final_protocol_20260915.json`：五个主配置
A0_HOLD/A1_RANDOM/A2_ALWAYS/A3U_RAW_ENTROPY/T1_TASK，A3_ENTROPY 和
A4_CONFORMAL 作为补充，三 repeat、四片、共同动作上限、四个预设 T1 配对比较
与 ROI-cluster bootstrap（四主比较采用 Bonferroni 98.75% 区间）。
`scripts/benchmarks/run_changeos_p6_final_shard.sh` 默认 `DRY_RUN=1`，单片单
repeat 启动，失败不自动续跑/重试；`preflight_changeos_p6_final.py` 在模型加载前
验证题库/审核/阈值/权重/策略源码/分析源码/提示词 SHA，
`report_changeos_p6_final.py` 在运行后按 `(repeat, qid)` 配对并严格审计每片，
不得用旧 `report_agent_vqa.py` 直接跨 repeat 聚合。2026-09-15 准备检查：
12/12 片干跑预检查通过、shell 语法通过、相关测试 36 passed；未启动任何 final
GPU episode。运行后还需核验全部 12 片科学身份与零执行错误，再发布确认性结果。

2026-09-17 P6 已完成：严格汇总
`runs/benchmarks/changeos_binary/p6_final_three_event_v1_20260915_report.json`
确认 12/12 片有效、3,360/3,360 条、零执行错误，160 题 × 3 个 generation repeats。
T1_TASK 75.00%，A0_HOLD 72.29%；预设配对差 +2.71 个百分点，ROI-cluster
Bonferroni 98.75% 区间为 [−2.04, +7.50] 个百分点，不能认定稳定胜过 HOLD。
T1 相对 A1_RANDOM 的校正区间下界恰为零，相对 A2_ALWAYS 为正，
相对 A3U_RAW_ENTROPY 跨零。完整逐 repeat、成本及边界见
`cja_en/review/changeos_p6_p7_postrun_audit_20260917.md`。按第 6 节既定规则，
论文转为平台与机制测量路线，不能把 T1 或 entropy 写成已证实优越的方法。
验收中的逐题型/事件和配对区间已完成；共同预算的在线七策略是实际 operating
points，**不是**随机化预算曲线，因此 accuracy--cost 曲线仍未完成。

### P7：成本与航空定位

- [ ] 记录 requested/allocated/executed 动作和失败原因。
- [ ] 从执行前后状态计算全部任务运动，不只累计复观测增量。
- [ ] 实测 render、ChangeOS、controller、Qwen 和端到端时延，注明硬件和串并行方式。
- [ ] 若保留 UAV/flight simulation 的强表述，增加有限的 PX4/Gazebo 动作映射验证；否则采用 imagery-grounded active observation 的边界表述。

**验收：**成本名称和实际测量范围一致；不把 `d/v` 估算写成实测航时，也不把图像裁剪直接等同于真实低空飞行。

现有三片有效开发集的轨迹状态成本可观测性审计位于
`runs/benchmarks/changeos_binary/p7_cost_observability_dev_3shards_20260915`：840 个 episode
中 839 个有完整终止状态，可分别计算整趟任务搜索与复观测的几何水平/垂直位移；
1 个终止于 fly_relative 的记录缺少后续状态，标记为不完整。复观测申请、执行和
预算扣减可追踪，但全部搜索动作的申请/分配参数及 render、ChangeOS、controller、
Qwen 的分模块时延尚未记录；这些几何距离不是实测飞行航时。
四片完整审计更新为
`runs/benchmarks/changeos_binary/p7_cost_observability_dev_4shards_20260915`：1,120 个
episode 中 1,119 个运动轨迹完整，缺口仍是一个 A0_HOLD 的终止 fly_relative；
按记录的前后状态可以区分整趟搜索与复观测的水平/垂直几何距离，并有每 episode
端到端 `wall_s`。这是已有日志的离线重算，不等于分模块实测时延、真实航时或
完整的 requested/allocated/executed 动作审计，所以 P7 四项仍未勾选。

2026-09-17 已对 P6 final 三个 repeat 分别运行现有 P7 成本审计，产物为
`runs/benchmarks/changeos_binary/p7_cost_observability_final_repeat{0,1,2}_20260917`。
合计 3,360 条中 3,318 条轨迹状态完整；42 条末步 `fly_relative` 缺后续状态。
复观测申请、预算扣减推断的分配、执行与 episode 总数逐条一致，但搜索动作
请求/分配、失败原因和分模块时延仍缺，几何距离及 episode 墙钟时间也不是
实测 UAV 航时。P7 仍不能全勾选；论文应采用保守的 imagery-grounded
active-observation 边界表述，除非另有航空动作映射验证。

### P8：重写论文

- [x] `main.tex` 摘要删除“闭环 QA 仍是未来工作”，换成经过 final 验证的结果。
- [x] 引言按环境、测量协议、任务条件化机制三项贡献组织；将 T1 的数值提升与尚未确定的稳定优势分开陈述。
- [x] 方法中的 `xview2_first`、四分类概率和 `log 4` 改为固定 ChangeOS 二分类及 `log 2`，并定义动作效用和证据融合。
- [x] 实验设置加入题集审核、模型/seed、共同上限、尚缺的 oracle 与统计协议；不把不同的实际动作数称为等成本。
- [x] 结果依次回答 RQ1--RQ5，同时报告纠正和破坏、成本与失败边界。
- [x] 将历史四分类与 VLN 正文结果压至附录，避免分散主动观测主线。
- [x] 局限中披露 ChangeOS 预训练暴露、正射影像/简化运动、事件数量和外部有效性边界。

**验收：**摘要、贡献、方法、表格和结论只陈述同一冻结证据链真实支持的内容。

P8 正文改写前的证据与旧术语冲突清单见
`cja_en/review/changeos_p8_prewrite_audit_20260915.md`。目前只完成审计，未把
开发集或待审核 final 的数值写成确认性论文结论。

P6 完成后的新证据与 P7 可观测性边界已记在
`cja_en/review/changeos_p6_p7_postrun_audit_20260917.md`。旧 prewrite 清单的
“final 尚未运行”描述现已过时。2026-09-17 已按该新报告重写主文摘要、引言、
方法、实验设置、结果、局限和结论；旧四分类与 VLN 表格在附录保留为历史诊断。
`cja_en/main.pdf` 可编译。P8 文本验收不等于投稿就绪：P4 随机化预算曲线、P5
分层 oracle、P7 分模块时延与物理动作映射仍是独立缺口；引用完整性、作者信息、
许可和发布包也尚需最终核验。

## 6. 决策规则

- 若任务条件化策略在预先冻结的 final、多 seed、共同预算下表现出稳定的 accuracy--cost 改善，论文可走“Task-Conditioned Active Observation”方法路线。
- 若没有稳定改善，保留完整负结果，走平台与机制测量路线：DisasterClaw 揭示视角改变为何既纠正也破坏，以及感知收益为何不能自动转化为任务收益。此时不得继续把 entropy gate 称为有效核心方法。
- 当前单次实验无论哪条路线都值得保留为开发诊断，但不进入最终主表的确认性排名。

## 7. 建议执行顺序

`P0 冻结现状 -> P1 生成可复现 -> 当前 160 题作为开发集 -> P2/P3 策略与消融 -> P4 ChangeOS 三档证据 -> 新 final 题集先审核冻结 -> P5/P6 确认性实验 -> P7 成本/动作验证 -> P8 论文重写`

最先动手的代码工作应是 P1，因为未冻结的生成随机性会污染之后所有策略比较；随后实施 P2 的任务条件化证据与覆盖规则。人工审核和新 final 题集可以与代码开发并行准备，但 final 运行必须放在规则全部冻结之后。
