# DisasterClaw 审稿意见与实验计划

审阅日期:2026-08-18
审阅材料:`main.pdf`(9 页编译产物)、`sections/*.tex`、`appendix/*.tex`、`CLAIMS.md`、
`EXPERIMENT_PROTOCOL.md`、`generated/*.json`、`generated/*.tex`、
`generated/vln_recheck_testset.json`

---

## 0. 总评

**模拟评分:2 / Strong Reject,置信度高。**

论文的中心机制(calibrated uncertainty → active reinspection)在全部 720 个
evidence-rich episode 中**从未触发过一次**:SR = 0、ΔU = 0、evidence 观测数 = 0。
摘要自述为 "an executable, auditable hypothesis rather than a demonstrated
improvement",审稿人读到此处即会停止评估。CVPR 不接受机制未被执行的方法论文。

当前唯一站得住的正面结果是 temperature scaling 将 in-domain ECE 从 0.095 降至
0.037,这是 Guo et al. 2017 的教科书复现,不构成会议级贡献。

**根本判断:问题不在"结果不够好",而在于**

1. 核心假设在当前仿真环境中**原理上无法被验证**(见 §1.1);
2. 当前的全零结果**几乎确定是工程缺陷**,不是科学负结果(见 §1.2);
3. 实验设计把机制假设绑在**最长最脆的因果链末端**,N = 0 是结构性必然(见 §2)。

值得保留的优点:`CLAIMS.md` 的 claim contract、`provenance.json` 的 SHA-256 溯源、
预注册假设、strict SR 与 semSR 的严格区分、把 localization 明确标注为 scaffolding。
这套科研卫生习惯优于多数投稿。问题是这套严谨性目前用在了一个测不出信号的设计上。

---

## 1. 致命问题(按严重程度排序)

### 1.1 【最致命】环境无法提供真实的"下降增益",核心假设不可验证

`sections/limitations.tex:20` 自述:"approximates altitude changes by changing crop
footprint"。xBD 仅有单一分辨率卫星影像,因此仿真中的"下降"等价于在同一批像素上裁
小窗口并放大,**不产生任何新信息**。

后果:方法核心量

```
IG(a) = U_t − E[U_{t+1} | a]
```

的期望值在信息论意义上约为 0。任何观测到的 ΔU 都是插值、aliasing 或边界裁剪的伪影。
熟悉遥感的审稿人只需问一句"下降之后有效 GSD 变了吗",论文即无法回答。

**这个缺陷的优先级高于所有 bug**:即使工程问题全部修复、机制成功触发,该设计缺陷
依然独立构成拒稿理由。

修复方向见 **实验 X1**。

### 1.2 全零结果是工程缺陷,不是负结果 —— 六条独立证据

| # | 证据 | 来源 | 含义 |
|---|------|------|------|
| 1 | 首题 `start_goal_distance_m = 48.19`,`success_radius_m = 25` | `generated/vln_recheck_testset.json:51,68` | 起点距目标约 48 m |
| 2 | E11 mean NE = **587.6 m** | `generated/reinspection_table.tex` | 从 48 m 出发终止于 588 m 外,agent 朝反方向飞行 |
| 3 | Steps 恒 = **12.00**(所有配置) | `generated/navigation_table.tex` | 从不主动停止,全部撞步数上限截断 |
| 4 | 6 策略 × 3 种子 → SR/semSR/NE/SPL **完全同值** | `generated/reinspection_table.tex` | 策略与随机种子对轨迹零影响,分支未被执行 |
| 5 | E1 (B1–B3) 与 E11(6 策略)的 mean NE **同为 587.6**、semNE 同为 528.3,而二者是**不同的 40 题任务集** | 两份 `generated/*_table.tex` | 概率上近乎不可能,指向配置串台或任务集加载错误 |
| 6 | 换用更强 localizer(Dice 0.75)后 SR 由 YOLO 时代 ~5% 降至 **0**,触发数由 3/120 降至 0 | `main.pdf` Tab.6 vs 当前源码 | 上游变强、下游变差,典型集成缺陷 |
| 7 | "This run also lacked a reachable VLM/LLM endpoint" | `sections/limitations.tex:12` | hybrid grounding 退化为 detector-only,**一个组件处于宕机状态** |

证据 7 单独即足以作废这批实验:**不能在模块不可用的前提下报告"机制无效"**。审稿人
会将其判定为 infrastructure failure 而非 finding。

排查步骤见 **实验 X0**。

### 1.3 旗舰 benchmark 自身存在事件泄漏

论文最大卖点是 leakage-safe split(`sections/task_benchmark.tex:29-31`:"No tile,
building, temporal pair, or disaster identity may cross training, calibration,
validation, and test partitions")。

但比对 `generated/vln_recheck_testset.json` 的 `by_disaster` 与
`generated/loc_metrics.json` 的 `event_split`:

| 分区 | 事件 | evidence-rich 题数 |
|------|------|-----|
| **train** | guatemala-volcano(2)、hurricane-florence(3)、hurricane-matthew(6)、midwest-flooding(3)、santa-rosa-wildfire(3)、socal-fire(3) | **20** |
| **val**(温度拟合) | hurricane-harvey(5)、mexico-earthquake(1) | **6** |
| test | hurricane-michael(8)、palu-tsunami(6) | 14 |
| holdout | — | **0** |

即 **evidence-rich 主实验集 50% 的题目来自感知模型的训练事件**,另 15% 来自温度拟合
事件,holdout 事件占比为零。这与论文声明直接矛盾。一旦机制产出正结果,该条即可用来
推翻它。必须在正式实验前重新生成子集(**实验 X4**)。

### 1.4 当前任务不构成 VLN,而是模板化的类别条件搜索

`generated/vln_recheck_testset.json:22-28`:

- `by_difficulty: {easy: 40, medium: 0, hard: 0}`
- `multi_landmark: 0`
- 指令形如 `"寻找完全损毁建筑"`

40 题的指令空间实质上是 4 个损伤类别的模板字符串。以此对标 R2R / AerialVLN 属于
framing 过度。二选一:

- 提升语言复杂度(空间关系、多地标、指代、否定、距离与方位约束);或
- 更名为 language-conditioned target search,不使用 VLN 一词。

**附带的表述准确性问题:**`review.reviewer` 字段为
`"GPT-5.6 Sol visual review"`(`generated/vln_recheck_testset.json:61`),而论文写
"visually reviewed"、`CLAIMS.md` 写 "manually reviewed task subset"。把模型审核表述为
人工审核是硬伤,必须改为 "model-assisted review with author spot-checks" 之类的准确
措辞,或补做真实人工核验。

### 1.5 统计功效在设计层面即不足

40 题 × 3 种子,效应量处于 2–5 个百分点量级,还需通过 6 策略的 Holm 校正。最小可检测
效应远大于可能观测到的效应。这不是"本次不显著",而是"该设计永不显著"。

需在方法或附录补 power analysis:给定 baseline SR 与目标效应,所需题数约为**数百量级**。
该结论直接支持 §2 的路线改造。

### 1.6 校准分析存在 base-rate 混淆

`generated/loc_metrics.json` 与校准表显示:

- holdout `mean_gt_boxes = 17.7` vs test `67.4` —— holdout 分区显著更稀疏
- holdout accuracy 0.803 **高于** test accuracy 0.765

即 holdout 是**更简单**的分区。因此 "test ECE 0.037 → holdout ECE 0.084" 的退化叙事被
分区难度差异污染,不能干净地归因于 distribution shift。需做难度匹配,或显式报告类别
分布并改用 balanced / per-class 指标。

四类损伤严重不平衡,仅报 accuracy 与 ECE 不足,须补 macro-F1 与 per-class reliability。

---

## 2. 技术路线诊断

现状:**把一个精细的机制假设,绑在一条最长最脆的因果链末端去验证。**

```
导航 → 建筑定位 → 裁图 → 双时相分类 → 校准 → 熵阈值 → IG 代理 → 触发 → 二次观测 → 判断改变
```

链上任一环返回空,可用样本量即为 0。而目前第一环就在朝反方向飞。**联合成功率决定样本量,
这是结构性必然,不是运气问题。**

### 改造原则:把机制研究与系统研究拆开,自下而上倒着做

| Layer | 内容 | 可用 N | 对应实验 |
|-------|------|--------|----------|
| **0** | 离线机制研究,无导航,直接在 building crop 上做预算受限的观测分配 | ~7 万 | **X2** |
| **1** | Oracle 阶梯,定位瓶颈,并首次让 controller 真正被执行 | 40–数百 | **X3** |
| **2** | 端到端,修完缺陷、恢复 VLM、benchmark 扩容与重新隔离后再跑 | 数百 | **X6** |

Layer 0 是唯一能在数周内产出**非零且有统计功效的正面结果**的路径,应作为论文主体。

### 建议的投稿定位

| 方案 | 内容 | 评估 |
|------|------|------|
| **A. 机制论文**(推荐) | Layer 0 为主体 + Layer 1 诊断 + 端到端入附录;标题改为"校准不确定性驱动的预算受限主动重观测用于双时相损伤评估" | 范围窄但结论硬,3–4 周可得真结果 |
| B. Benchmark + 分析论文 | 主打任务定义、泄漏安全协议、oracle 阶梯诊断"为何现有 stack 失败" | 需 benchmark 扩至数百题 + 语言复杂度 + baseline 真正跑通;工作量大,赛道竞争激烈 |
| C. 降级投稿 | 现状 + 缺陷修复,投 workshop / IROS / RA-L | 保底;系统类会议对"中等结果 + 诚实分析"容忍度高 |

**推荐走 A,B 的材料留作附录。** 当前最缺的不是工程量,而是一个非零、有功效的正面结果。

---

## 3. 需要做的实验

### X0 —— 端到端缺陷排查(阻断级,最高优先)

**目的:** 判定全零结果是缺陷还是真实负结果。在此完成前,所有实验结论无效。

| 步骤 | 操作 | 判定标准 |
|------|------|----------|
| X0.1 | 取 `vln_recheck_testset.json` 首题,逐步打印 `(target_lat, target_lon, agent_lat, agent_lon, geodesic_dist)` | 距离**单调增大** → 航向符号错误或 lat/lon 顺序颠倒(588 m 量级高度符合 lat/lon swap 特征);**跳变** → 坐标系/投影转换错误 |
| X0.2 | 核对 E1 与 E11 实际加载的 benchmark 文件路径与 item id 列表 | 两者 item id 集合必须不同;若相同 → 配置串台,证据 5 得解释 |
| X0.3 | 在 evidence 计数处加 assert / 日志,统计 720 episode 中 proposal 数、crop 数、classifier 调用数、evidence 数 | 定位链条在哪一步返回空 |
| X0.4 | 检查 U-Net box → 地理坐标的转换,与 YOLO 路径逐行对齐 | 上游 Dice 提升而下游 SR 归零,必有转换或类别映射差异 |
| X0.5 | 恢复可达的 VLM / LLM endpoint,重跑 1 个 seed 的 E11 | hybrid grounding 必须真正为 hybrid |

**验收标准:** 单题 trace 上 NE 从 588 m 降至 < 48 m(即至少不比起点更差);至少出现
一次非零 evidence 观测与一次真实 reinspection 触发。

### X1 —— 环境改造:让"高度"对应真实的分辨率阶梯(解锁一切的前提)

**目的:** 修复 §1.1 —— 使下降动作真正带来新信息,使 IG 成为可测的物理量。

**做法:** 重新定义高度—分辨率映射,不需要新数据。

- 巡航高度视图 = xBD 原生分辨率的 **4× 降采样** + 高度相关的模糊与噪声
- `descend_center` 逐级恢复分辨率,最低高度对应 **原生分辨率**
- 记录每个高度档的**有效 GSD**,写入 episode 日志

**产出:** 曲线图 **有效 GSD → 损伤分类 macro-F1 / ECE**。该图独立成立,可直接作为论文
的 headline figure,并从物理上证明 reinspection 有信息论意义上的收益空间。

**验收标准:** 原生分辨率档的 macro-F1 显著高于巡航档(paired bootstrap CI 不含 0)。
若不显著,说明该数据源上"下降"确实无增益,必须立即转向真实低空数据(见 X1b)。

### X1b —— (条件触发)引入真实低空 UAV 数据

**触发条件:** X1 验收不通过,或希望把故事从"仿真近似"升级为"真实多尺度证据"。

**候选数据集**(需自行核实许可与标注粒度):

- **RescueNet** —— Hurricane Michael 无人机影像,含建筑损伤标注
- **FloodNet** —— Hurricane Harvey 无人机影像

**价值:** 支撑"卫星 → 低空"的真实跨域下降叙事,同时为 §1.6 的 domain shift 讨论提供
真实的 shift 而非分区难度差异。

### X2 —— 【论文主体】离线预算受限观测分配实验

**目的:** 回答真正的科学问题 —— **校准质量能否转化为决策质量?** 现有论文中 RQ1 与
RQ2–RQ4 是断开的,本实验是唯一能把二者连起来的设计。

**设置:**

- 数据:strict test 分区的 71,293 个 bitemporal building crop(N 为万级,功效充足)
- 视图:X1 定义的巡航档 / 原生档
- 预算 B:每样本平均允许 0, 0.1, 0.25, 0.5, 1.0 次下降观测(扫描)
- 无导航、无地图、无 memory

**对比策略:**

| 策略 | 说明 |
|------|------|
| None | 永不下降,直接判断 |
| Random | 随机花预算 |
| Entropy-threshold(uncalibrated) | 用**未校准**熵 + 阈值 |
| Entropy-threshold(calibrated) | 用**校准后**熵 + 阈值 |
| Calibrated-IG | 完整方法 |
| **Oracle** | 已知哪些样本二次观测后标签会翻转,给出上界 |

**主图:** x 轴 = 平均每样本额外观测次数,y 轴 = damage macro-F1;六条曲线 + oracle 上界。

**次要指标:** ECE、Brier、NLL、每次观测的边际 F1 增益、决策 regret(相对 oracle)、
翻转样本的 precision / recall。

**统计:** item-level 配对;bootstrap 95% CI;paired permutation test;Holm 校正策略族。

**验收标准(论文能否成立的分水岭):**

1. calibrated 策略曲线在相同预算下**显著高于** uncalibrated 与 Random(CI 不重叠);
2. Calibrated-IG ≥ Entropy-threshold(说明 IG 代理有价值);
3. 与 oracle 的 gap 被量化并报告。

若 (1) 不成立,则"校准 → 决策"这条主张在数据上不成立,必须放弃方案 A,改走 B 或 C。

### X3 —— Oracle 阶梯:定位瓶颈

**目的:** 论文现断言 "grounding bottleneck" 但无证据(`sections/experiments.tex:82-85`
仅为叙述)。本实验提供证据,并让 controller 首次在带导航的设定下被真正执行。

| 配置 | oracle nav | oracle grounding | 用途 |
|------|-----------|------------------|------|
| L3 | ✓ | ✓ | 上界 |
| L2 | ✓ | ✗ | 隔离感知 |
| L1 | ✗ | ✓ | 隔离导航 |
| L0 | ✗ | ✗ | 现状(当前 SR = 0) |

**关键副产品:** L3 / L2 会直接产生非零的 evidence-positive episode,使 6 策略比较第一次
具备可执行样本。

**验收标准:** 四行数字齐全,且能明确指出瓶颈层;即使绝对数值低,该表本身即为有价值的
诊断性贡献。

### X4 —— 重新生成事件隔离的 evidence-rich 子集

**目的:** 修复 §1.3 的泄漏。

**要求:**

- 题目**仅**取自 test 与 holdout 事件(hurricane-michael、palu-tsunami、moore-tornado、
  nepal-flooding、pinery-bushfire),完全排除 12 个 train 事件与 2 个 val 事件
- 规模按 X5 的 power analysis 结果确定(预期数百题)
- 难度分层不再全为 easy:补 medium / hard 与 multi-landmark
- 语言复杂度按 §1.4 提升(空间关系、方位、指代、否定)
- 审核字段准确记录审核者身份(模型 / 人工 / 混合),不得表述为纯人工

**验收标准:** 生成脚本内置断言:任一 item 的 disaster 落入 train ∪ val 即报错退出
(与 `--require-event-disjoint` 同等强度)。

### X5 —— Power analysis 与统计设计

**目的:** 修复 §1.5,并为 X4 的规模决策提供依据。

**内容:**

- 给定 baseline SR、目标效应(绝对与相对)、α = 0.05、power = 0.8、6 策略 Holm 校正,
  计算所需 item 数与 seed 数
- 报告当前 40 × 3 设计的**最小可检测效应**,写入论文正文
- 明确"零触发 run 视为 conditional ΔU 缺失"的处理规则(`EXPERIMENT_PROTOCOL.md:54-55`
  已有,保留并在正文引用)

### X6 —— 端到端重跑(在 X0/X1/X4 全部完成后)

**内容:** 修复缺陷、恢复 VLM endpoint、启用新分辨率映射、使用 X4 的新 benchmark,
重跑 E1(B0–B3)与 E11(6 策略),以及 `EXPERIMENT_PROTOCOL.md:57-60` 中挂起的
GPS 扰动(0/2/5/10 m)与 forced-degraded 两套 robustness суite。

**定位:** 论文后半段的"现实检验",允许结果不漂亮,但必须是**在所有模块可用前提下**
得到的结果。

### X7 —— 校准方法升级

**目的:** 使 RQ1 具备会议级分量。

- 加入 vector scaling / Dirichlet scaling,与单一标量温度对比
- 加入 deep ensemble 或 MC-dropout 作为不确定性来源对比项
- **重点:引入 conformal prediction** —— 输出集合预测 + 覆盖保证,与"证据是否足够"
  这一决策天然对接;robotics 方向已有先例(如 KnowNo 用 conformal prediction 决定
  LLM planner 何时求助),须补入 related work
- 报告 macro-F1、per-class reliability、balanced ECE,修复 §1.6 的 base-rate 混淆
- 对 test / holdout 做难度匹配(按 gt_boxes 密度或类别分布分层)后再比较 ECE

---

## 4. 论文修订清单

### 4.1 阻断级(必须修)

- [ ] **PDF 已过期**:`main.pdf` 仍是 YOLO 时代数字(Tab.1 SR 7.5/2.5、Tab.6 SR 4.2–6.7),
      而 `sections/` 与 `generated/` 已全部改为 0。重新编译,避免组会材料与文件不一致。
- [ ] **`\ProjectName` 后缺空格**:PDF 中出现 "DisasterClawcloses"、"DisasterClawinstantiates"、
      "DisasterClawprovides"、"DisasterClawis"。改 `macros.tex:1` 为
      `\newcommand{\ProjectName}{DisasterClaw\xspace}` 并在 `preamble.tex` 加
      `\usepackage{xspace}`。
- [ ] **Figure 4 contact sheet 中文标签渲染为豆腐块**(□□□),改英文标签重新生成。
- [ ] **匿名性泄漏**:附录含绝对路径 `/home/lc/disasterclaw`、`runs/benchmarks/paper_strict_unet_v1/`
      (`appendix/implementation.tex`、`main.pdf` L585)。CVPR 双盲下改为仓库相对路径。
- [ ] **审核表述失准**:`review.reviewer = "GPT-5.6 Sol visual review"` 与论文
      "visually reviewed" / `CLAIMS.md` "manually reviewed" 冲突,见 §1.4。

### 4.2 内容级(必须补)

- [ ] **Eq. 5 的 IG 代理完全未定义**。`sections/method.tex:68` 仅写 "a geometry-conditioned
      proxy predicts the entropy after the next observation" —— 是学习出的回归器还是手调
      启发式?这是方法核心,当前为黑箱。若为启发式,"information gain" 属过度声明,应改称
      geometry-based gain heuristic。
- [ ] τ、gain margin、per-location / per-episode budget 三组超参**无敏感性分析**。补 sweep,
      这本身就是主图之一。
- [ ] **Related work 仅 12 篇**,CVPR 需 40+。明显缺失:遥感变化检测(BIT / ChangeFormer /
      SNUNet 等)、xBD 损伤评估后续工作、学习化 next-best-view、embodied uncertainty /
      introspection、分布偏移下的校准(Ovadia et al. 2019)、conformal prediction in robotics、
      更多 aerial VLN。
- [ ] 章节组织:不要以 "RQ6: Navigation Success Remains Zero" 开场
      (`sections/experiments.tex:17`)。把 X2 的机制曲线提到实验节最前。
- [ ] 补 power analysis 段落(X5 产出)。
- [ ] 补 balanced / per-class 感知指标(X7 产出)。

### 4.3 保留(勿改)

- `CLAIMS.md` 的 prohibited claims 清单
- `generated/provenance.json` 的 SHA-256 溯源机制
- 附录 A 的预注册假设 H1–H4 与 failure accounting
- strict SR 与 semSR 的严格区分
- 将 localization / HSPM / memory 明确标注为 scaffolding 而非贡献

---

## 5. 建议时间线

| 周次 | 任务 | 验收标志 |
|------|------|----------|
| 第 1 周 | **X0** 缺陷排查 + 恢复 VLM endpoint | 单题 NE < 48 m;出现非零 evidence 与真实触发 |
| 第 2 周 | **X1** 高度—分辨率映射改造 | 原生分辨率档 macro-F1 显著优于巡航档 |
| 第 3–4 周 | **X2** 离线机制实验(论文主体) | "额外观测次数 vs macro-F1 + oracle 上界" 曲线,calibrated 显著优于 uncalibrated |
| 第 5 周 | **X4** 重新生成隔离 benchmark + **X3** oracle 阶梯 | 四行 oracle 表齐全;新 benchmark 通过 disjoint 断言 |
| 第 6 周 | **X5** power analysis + **X7** 校准升级 | 正文可写入最小可检测效应与 conformal 结果 |
| 第 7 周起 | **X6** 端到端重跑 + 按方案 A 重写论文骨架 | — |

**关键依赖:** X2 是分水岭。若 X2 验收标准 (1) 不成立,立即停止方案 A,改走 B(benchmark +
分析)或 C(降级投稿),不要在无信号的设计上继续投入工程量。

---

## 6. 一句话总结

诚实性不是问题,`CLAIMS.md` 那套 claim contract 应当保留并继续执行。问题是这套严谨性
目前正在诚实地报告**一个测不出信号的实验设计**。把高度做成真实的分辨率阶梯(X1)、把
机制验证从端到端链条中解耦出来做成离线预算实验(X2),这两步就能把论文从 reject 拉回
到有据可辩的位置。
