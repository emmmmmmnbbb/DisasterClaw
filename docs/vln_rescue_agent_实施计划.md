# DisasterClaw 改造实施计划（一步步做、一步步测）

本计划完全基于 `vln_rescue_agent_方案.md`，把"会看灾情的语言导航无人机"拆成 5 个阶段。
原则：**每个阶段都小、可独立验收、能回退**；不通过验收不进下一阶段。

> 视觉路线（已定）：先把"俯视地图 + 语言"这条路做出论文级结果，第一视角列为未来工作。
> 实现风格：新增模块为主，尽量不破坏现有 `ai_planner` / `vln_navigator` / `run_vln_episode` 既有链路，
> 新功能用环境变量开关挂载，旧行为保留作 fallback 与对照基线。

---

## 范围

- **In**：2D 地理语义地图 + 记忆；STMR 文字矩阵；HSPM 三层规划；灾情不确定性驱动复核；
  记忆拓扑图 + 图搜索；混合 grounding；VLN 标准评测 + 救灾鲁棒性评测。
- **Out**：第一视角/3D 点云/深度相机；真实飞行动力学；多机协同；前端大改（仅按需加最小可视化）。

---

## 阶段总览

| 阶段 | 名称 | 对应方案 | 性价比/优先级 |
|---|---|---|---|
| P0 | 地基：2D 地理语义地图 + 累积记忆 | 第 6 步·第 0 步 | 必做前置 |
| P1 | STMR 文字矩阵 + HSPM 三层规划 | 第 6 步·第 1 步 | **最高，先做** |
| P2 | 灾情不确定性驱动主动复核（C2 主线） | 第 6 步·第 2 步 | 创新核心 |
| P3 | 记忆拓扑图 + LM-Nav 图搜索 | 第 6 步·第 2 步 | 长程稳定性 |
| P4 | 混合 grounding + 评测成绩单 | 第 4、7 节 | 论文支撑 |

---

## P0 — 地基：2D 地理语义地图 + 累积记忆

**目标**：建一张"会一直累积"的地图，记下飞过哪、每格看到啥。后续所有阶段都建在它上面。

**改动**
- [x] 新增 `backend/semantic_map.py`：以 episode 起点为原点的栅格地图，按 CityNav GSM 分 5 层
  （当前视场 / 已探索区 / 地标 / 周围物体 / 候选目的地），单架 UAV、单 episode 内累积，可序列化。
- [x] 复用 vln_navigator 几何约定（归一化中心 + patch 半径）+ `geo.meters_to_latlon`
  把检测框中心投回 lat/lon 写入地图（免改 perception 暴露裁剪偏移）。
- [x] 在 `app.py` 的 `_vln_perceive`（以及 `execute_action` 的 `detect_disaster` 分支）每次感知后更新地图。
- [x] 通过 socket `semantic_map` 事件发精简地图状态；新增 `GET /api/semantic_map(?full=1)` 供调试/落盘。

**怎么测（验收点）**
- [x] 单元测试 `backend/tests/test_semantic_map.py`：投影往返误差 0.000m（≤1 格）、explored 随移动增长、
  degraded 只记 explored、序列化字段完整、局部矩阵语义正确 —— **5/5 通过**。
- [ ] 端到端（需起后端 + 模型）：跑一条会移动的指令，`GET /api/semantic_map` 看探索区随轨迹增长、
  检测落点与瓦片标注目视对齐（留待联调）。

---

## P1 — STMR 文字矩阵 + HSPM 三层规划（先做，最划算）

**目标**：把"关键词硬匹配 + 朝质心贪心"升级为"读懂长指令 → 拆地标 → 常识推理 → 出航点"。

**改动**
- [x] 新增 `backend/stmr_matrix.py`：从 P0 地图取 UAV 周围 `window_m×window_m`（默认 200m）→ `grid_n×grid_n`
  （默认 20×20）文字矩阵（语义 max-pooling、数字码 + 图例 + UAV 标 'U'），并附"目标方位摘要"
  （每个目标相对 UAV 的八方位 + 距离），输出 LLM 友好文本。
- [x] 新增 `backend/hspm_planner.py`，三层：
  - **landmark-level**：`plan_landmarks` 用 planner LLM 把指令拆成有序地标序列（失败回退开放词汇短语）；
  - **object-level（OROI）**：`reason_oroi` 喂 STMR 文本 + 当前子目标，LLM 选"下一步朝哪个方位"（八方位，失败回退方向先验）；
  - **motion-level**：看得到子目标→朝质心 `fly_relative` 步进、到达→推进/完成；看不到→按 OROI 方位探索一步。
- [x] grounding 修复（并入本阶段）：① `parse_instruction` 增 `target_phrase`（`extract_target_phrase` 保留"蓝色"等开放词汇修饰，不被 YOLO 类别词典改写）；
  ② VLM grounder 改"**坐标-or-没有**"范式（实测小 VLM 输出 `present` 布尔会无视自身描述默认 false），`parse_ground_xy` 鲁棒解析。
- [x] 在 `run_vln_episode` 用 `VLN_PLANNER=hspm` 开关挂载 `HspmNavigator`（`_make_hspm_navigator`）；
  `VlnNavigator`（关键词+贪心）保留为 `legacy`（默认）fallback 与对照基线。

**怎么测（验收点）**
- [x] 单元测试 `backend/tests/test_grounding_parse.py`（5/5）：开放词汇短语保留"蓝色"、坐标/否定解析鲁棒。
- [x] 单元测试 `backend/tests/test_stmr_matrix.py`（3/3）：图例/网格/U 标记/维度、目标码进矩阵、方位摘要正确。
- [x] 单元测试 `backend/tests/test_hspm_planner.py`（7/7，mock LLM/grounder/STMR）：多地标拆解+回退、OROI 方位、
  看得到朝质心/看不到朝方位、多地标推进（hover→stop arrived）、degraded 不崩。
- [x] grounding 实测：对原"找不到的蓝色建筑"patch，旧 `present` 布尔提示→false（自相矛盾）；新"坐标-or-没有"提示→返回坐标命中。
- [ ] 端到端对照（需起后端 + 模型，`VLN_PLANNER=hspm`）：同一批指令下 **HSPM 的到达率/步数 vs legacy**（留待联调，作 P4 消融素材）。

---

## P2 — 灾情不确定性驱动主动复核（创新主线 C2）

**目标**：让"灾情判断的把握程度"指挥飞行——没把握就飞近/换高度再确认。

**改动**
- [ ] 在导航决策层（HSPM motion 之后）加一个"复核触发器"：当 grounding 置信度低、
  或 `perception.risk_level` 可疑（如 moderate 且证据弱）时，插入"下降高度 + 飞近 + 重新感知"动作。
- [ ] 复核前后对比 `risk_level`/置信度，写回 P0 地图（候选目的地层升/降级），并在报告里说明复核结论。
- [ ] 用 `VLN_RECHECK=1` 开关；设最大复核次数与高度下限，防止死循环。

**怎么测（验收点）**
- 构造一个低置信/边界场景（远距离小目标），验证**触发复核**，且复核后置信度或判定**发生更新**。
- 防回归：高置信场景**不触发**复核（不浪费步数）。
- 指标：记录"复核带来的不确定性下降"，作为 C2 的核心实验数据。

---

## P3 — 记忆拓扑图 + LM-Nav 图搜索

**目标**：跨步骤/跨任务复用走过的路，支持大范围长程巡查，少做无用搜索。

**改动**
- [ ] 新增 `backend/memory_graph.py`：把成功轨迹的航点存成 **2D 拓扑图**（节点含 lat/lon + 观测摘要，
  边按距离加权；阈值合并邻近节点）。
- [ ] grounding 打分 `P(节点|地标)`（用 VLM/CLIP 或现有 grounder），用 **LM-Nav 式图搜索**选最优 walk。
- [ ] motion-level 优先查记忆图走熟路，查不到再回退 P1 的探索式规划。

**怎么测（验收点）**
- 同一区域跑第二条相似指令：**命中记忆图**、到达步数较首次**明显下降**。
- 防回归：全新区域无记忆时，行为与 P1 一致（不退化）。

---

## P4 — 混合 grounding + 评测成绩单（论文支撑）

**目标**：补齐"看得准"和"有成绩单"两块短板。

**改动**
- [ ] 混合 grounding：YOLO（域内、快）优先，认不出/出现新词再用开放词汇模型兜底
  （接 `locate_ground_fn` → LocateAnything，子进程/HTTP worker，独立 conda 环境）。
  注意实测零样本偏弱（F1@0.5≈0.14、损伤分级差），加置信度过滤 / NMS / 单实例约束。
- [ ] 新增 `scripts/benchmarks/bench_vln_navigation.py`：算 **SR / SPL / NE** + 不确定性下降 / 覆盖率。
- [ ] 准备小规模"**指令 → 正确轨迹**"测试题库（先 20~50 条，覆盖单/多地标、有/无方向、各灾种）。
- [ ] 救灾鲁棒性评测：洪水/地震瓦片 + 模拟 GPS 噪声，复用 CityNav 协议。
- [ ] 消融表：legacy 贪心 / +HSPM / +复核 / +记忆图，逐项看指标变化。

**怎么测（验收点）**
- 跑出**第一版成绩单**（baseline + 各消融），数值可复现（脚本一键出 `results.json`）。
- 鲁棒性测试在 GPS 噪声下成功率下降幅度**可量化**。

---

## 里程碑与依赖

```
P0 ──> P1 ──> P2 ──┐
        └──> P3 ───┴──> P4(评测贯穿,P1 起即可开始记录)
```

- **M1（P0+P1）**：能听长指令、拆地标、按文字地图找到目标 —— 可演示。
- **M2（+P2）**：灾情驱动复核闭环成立 —— 创新点可写。
- **M3（+P3+P4）**：长程稳定 + 完整成绩单 —— 论文实验齐备。

---

## 风险与对策

- **LLM 时延**：HSPM 多次调用大模型可能慢 → 缓存子目标、object-level 限频、必要时小模型跑 OROI。
- **开放词汇 grounding 弱**：已实测偏弱 → 以混合策略为主，别把成败押在它上面。
- **跨瓦片**：长程导航依赖 `fly_to_geo` 自动对齐 → P3 记忆图节点显式记录瓦片归属，规避丢覆盖。
- **每阶段开关化**：所有新功能用环境变量挂载，随时回退旧链路，保证 demo 可用。

---

## 待确认（开始 P0 前）

1. 地图与记忆先做**单 episode 内累积**，还是一上来就做**跨任务持久化**（落盘）？（建议先前者，P3 再持久化）
2. HSPM 的 landmark/OROI 推理用哪个模型？复用 `.env` 里 planner LLM，还是单独配一个更强/更快的？
3. 评测题库（指令→轨迹）由我**基于现有瓦片标注自动生成草稿**，还是你手工标一小批更可靠？
