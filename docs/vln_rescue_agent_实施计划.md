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
- [x] 新增 `backend/recheck.py`：`RecheckController` 按位置去重、带预算的复核状态机。
  不确定性 = 0.5×risk暧昧度 + 0.5×(1−证据conf)；`risk_level='low'`（"轻度受灾或可疑"）最不确定、`'high'` 最笃定。
  触发后产出"**降高 + 飞近居中**"机动——利用 `patch半径 = alt×factor`，降高=视场变小=GSD 更细=看得更清。
- [x] 复核闭环：到"把握足够 / 预算耗尽 / 到高度下限"后定论 confirmed/dismissed/inconclusive，
  连同**不确定性下降量**写回 P0 地图 `candidate_goals` 层（待复核→定论），并进 `summary`/报告（`recheck`/`recheck_log`）。
- [x] 在 `run_vln_episode` 用 `VLN_RECHECK=1` 开关（默认关）；`VLN_RECHECK_MAX`（单点次数）+
  `VLN_RECHECK_MAX_TOTAL`（episode 总上限）+ `VLN_RECHECK_ALT_MIN_M`（高度下限）三重防死循环；复核机动同样走 POST 防越界。

**怎么测（验收点）**
- [x] 单元测试 `backend/tests/test_recheck.py`（8/8）：不确定性评分、best_evidence、低置信触发复核（降高+居中）、
  高置信不触发、预算耗尽/高度下限定论、改善路径 confirmed（不确定性下降 0.65）、degraded 仅降高。
- [ ] 端到端（需起后端 + 模型，`VLN_RECHECK=1`）：低置信小目标场景触发复核并更新判定；高置信不触发（留待联调）。
- 指标：`report.recheck.avg_uncertainty_reduction` 作为 C2 的核心实验数据。

---

## P3 — 记忆拓扑图 + LM-Nav 图搜索

**目标**：跨步骤/跨任务复用走过的路，支持大范围长程巡查，少做无用搜索。

**改动**
- [x] 新增 `backend/memory_graph.py`：`MemoryGraph` 把成功轨迹存成 **2D 拓扑图**（节点 lat/lon +
  观测摘要：目标类别/risk/到过的指令&地标；边按地表距离加权；`merge_radius_m` 阈值合并邻近点），
  **跨任务 JSON 持久化**（save/load，"越用越熟"）。
- [x] grounding 打分 `P(节点|地标)`：默认 `text_match_scorer`（字符 bigram 相似，可换 VLM/CLIP）；
  `plan()` 用 **LM-Nav 式**为最终地标匹配目标节点 + Dijkstra 串出 walk（起点近→走熟路 `graph_walk`，远→`direct` 直飞）。
- [x] 在 `run_vln_episode` 用 `VLN_MEMORY=1` 开关：episode 开始 `_vln_memory_prefly` 命中则沿熟路预飞到目标附近
  （带 POST 防越界、`VLN_MEMORY_MAX_HOPS` 上限、跳过过近航点），再进入 P1 grounding 精定位；
  成功到达后把轨迹沉淀进图并落盘。查不到熟路 → 行为与 P1 一致（不退化）。新增 `GET /api/memory_graph(?full=1)`。

**怎么测（验收点）**
- [x] 单元测试 `backend/tests/test_memory_graph.py`（6/6）：阈值合并/新建、Dijkstra 最短路、地标匹配、
  plan 命中给 walk / 无关指令返回 None、起点远→direct、序列化往返后仍可规划。
- [x] 端到端 E4（`bench_e4_memory.py`）：14 瓦片对 × cold/warm/seed，seed SR=43% vs cold 0%，机制有效；warm 受首趟到达率限制。
  全新区域无记忆**不退化**（留待联调）。

---

## P4 — 混合 grounding + 评测成绩单（论文支撑）

**目标**：补齐"看得准"和"有成绩单"两块短板，跑出可复现的消融与鲁棒性成绩单。

### P4.0 实验假设（成绩单要回答的 4 个问题）

| 编号 | 假设 | 验证手段 |
|---|---|---|
| H1 | HSPM 三层规划 > 关键词+贪心 | B1 vs B0：SR↑ / NE↓ / SPL↑ |
| H2 | 不确定性驱动复核能提升灾情判定质量 | B2 vs B1：ΔU>0、灾情判定准确率↑（代价是步数↑） |
| H3 | 记忆拓扑图能降低重复区域的导航开销 | B3 同区第二趟：步数 / 路径长↓，SR 不降 |
| H4 | 系统对 GPS 噪声 / 跨灾种 / 退化视场有一定鲁棒性 | 鲁棒性曲线：扰动幅度 vs SR 衰减可量化 |

### P4.1 评测指标

- **标准 VLN**：NE（终点到 GT 的米数）、SR（NE≤`success_radius_m` 视为成功）、
  SPL（`SR · 最短路径 / max(实际路径, 最短路径)`，用 report 的 `path_len_m` / `shortest_path_m`）、Steps（消耗步数）。
- **救灾专属**：ΔU（`report.recheck.avg_uncertainty_reduction`）、灾情判定准确率（终点判定 vs GT 损伤等级）、
  覆盖率（探索栅格 / 题目相关区域）、记忆增益（同区第二趟相对首趟的步数/路径下降）。

### P4.2 题库（指令 → GT）  ✅ 已完成（P4-1）

- [x] 新增 `scripts/benchmarks/gen_vln_testset.py`：从 xBD label（`lng_lat` 多边形 + `subtype` 损伤等级）
  自动生成"指令→GT"草稿。每条含 `instruction / start(lat,lon,alt) / landmarks / goals(质心+损伤等级) /
  success_radius_m / shortest_path_m / difficulty / disaster`。
- [x] **起点按难度桶（easy/medium/hard）从目标反推并 clamp 进瓦片**，保证难度分层均衡 + 起点必在 POST 覆盖内。
- [x] 灾种轮转 + 单/多地标 + 有/无方向先验混合；首版 40 条落 `backend/data/benchmarks/vln_testset.json`
  （实测 12/12/16 难度均衡、4 灾种、23 带方向、14 多地标）。
- [ ] **人工校验**：剔除歧义题（目标过近、同类多实例混淆、方向先验与真实方位冲突）后定稿正式题库。

### P4.3 无头评测入口  ✅ 已完成（P4-2）

- [x] `run_vln_episode` 重构：**返回 enriched report**（新增 `final_pos / path_len_m / landmarks /
  target_classes / planner / grounder / trajectory / config / steps_executed`），异常/忙时也返回带 error 的 dict；
  全程复用真实感知/规划/复核/记忆链路，socket 广播在无客户端时为 no-op。
- [x] 新增 `run_vln_episode_headless(instruction, start, source="bench")`：按起点对齐 POST 瓦片 + 定位 UAV，
  同步阻塞跑完返回 report，起点无覆盖返回 `start_not_covered`。供评测脚本逐题调用、零前端依赖。

### P4.4 评测脚本  ✅ 已完成（P4-3，bench_report 待做）

- [x] 新增 `scripts/benchmarks/bench_vln_navigation.py`：读题库 → 每个配置在 import 后**直接改 `app.VLN_*` 全局**切换
  （B0~B3）→ 逐题 `run_vln_episode_headless`（真实感知 + 本地 Qwen2.5-VL + 规划/复核/记忆）
  → 由 report + GT 算 NE/SR/SPL/Steps/ΔU/判定准确率 + 难度分桶 → 落 `runs/benchmarks/<run_id>/`
  （`results.json` / `summary.md` / `episodes.jsonl`，每条 episode 即时 flush）。固定随机种子、`env_snapshot` 自描述。
- [x] **修复 P3 记忆跨区域 bug**：`memory_graph.plan` 加 `max_dist_m` 地理门控 + `_vln_memory_prefly` 加
  `VLN_MEMORY_MAX_DIST_M`（默认 1500m）航点距离上限。此前 B3 因"完全损毁建筑"按纯文本跨灾种匹配，
  印尼起点命中佛州节点 → 预飞 `fly_to_geo` 飞了 ~2 万公里（NE 爆表）；修复后 B3 NE 回到正常量级。
- [x] 新增 `scripts/benchmarks/bench_report.py`：聚合 run → 主消融表 / E3 复核价值 / E8 难度分桶 / E6 跨灾种 / E2 grounder 对比。

**第三版成绩单（E1 全量：40 题 × B0~B3，grounder=hybrid，repeat=1）**
`runs/benchmarks/20260623_202108_e1full/`（聚合 `report.md`）：

| 配置 | SR | semSR | NE(m) | semNE(m) | SPL | Steps | ΔU |
|---|---|---|---|---|---|---|---|
| B0 baseline | 0.075 | 0.125 | 150 | 136 | 0.072 | 3.3 | — |
| B1 +HSPM | 0.075 | 0.125 | 226 | 193 | 0.068 | 4.8 | — |
| B2 +复核 | 0.025 | **0.15** | 230 | **190** | 0.025 | 5.6 | 0.0 |
| B3 +记忆 | 0.025 | 0.075 | 301 | 261 | 0.025 | 5.5 | 0.0 |

> 相对 8 题冒烟：方差收敛后 **B0 NE 最低**（150m），HSPM/记忆/复核均未带来 strict SR 提升。
> H1：B1 步数↑、NE↑（hard 题绕远）。H2：B2 semSR 略升但 ΔU≈0（复核在 episode 内几乎未定论）。
> H3：B3 NE 最差（301m）。E8：hard 桶 B1 NE≈291m vs B0≈143m。E6：palu-tsunami/mexico-earthquake 相对更好。
> E2 grounder 对比：B1×yolo/vlm 补跑中（hybrid=本表 B1 行）。

**首版成绩单（E1 子集：前 8 题 × B0~B3，grounder=vlm，RTX 6000D）**
`runs/benchmarks/20260622_181546_e1fix/`：

| 配置 | SR | NE(m) | SPL | Steps |
|---|---|---|---|---|
| B0 baseline | 0.125 | 86.5 | 0.092 | 2.5 |
| B1 +HSPM | 0.125 | 269.4 | 0.092 | 6.1 |
| B2 +复核 | 0.0 | 352.1 | 0.0 | 4.75 |
| B3 +记忆 | 0.0 | 245.8 | 0.0 | 3.6 |

> 说明：这是**冒烟级首版**（仅 8 道偏难题、repeat=1）。整体 SR 偏低，主因 grounding 弱——
> 多数 patch 下 YOLO 检出 0 目标、小 VLM 易早停/漂移；HSPM 在多地标 hard 题上会绕远（NE↑、步数↑）。
> 结论：**先扩题量 + 修 grounding/早停，再下消融定论**；当前数据用于打通管线、暴露问题，不作最终结论。
> 跑全量：`--limit 0 --repeat 3`（约 40 题 ×4 配置 ×3 ≈ 480 episode，需数小时）。

#### 早停问题深挖与结论（诊断 + 两次实验）

用 `scripts/benchmarks/diag_grounding.py` 在"目标处 vs 起点处"各感知一次，得到关键证据：
- **YOLO 在 xBD 瓦片上全程 0 检出**（权重 `mars/RescueNet` 域，与 xBD 有域差）→ YOLO 门控对 xBD 无效。
- **起点（目标不在视场）VLM 正确回"没有"**，并不会凭空幻觉 → 不存在"起点瞬间假到达"。
- **目标处 VLM 召回仅 ~50%**，且**空间定位偏中心**：目标即便偏在视场边缘，VLM 也常报"≈正中央"。

由此定位"早停"真实机制：agent 探索 1 步后，VLM 把"其实还差 50~160m / 偏在边缘"的目标判读成居中，
偏移按 patch 半径(alt30→60m)换算出 <35m → **假到达**，停在离目标 50~160m 处。这是 **VLM 俯视空间精度**问题，
不是停车逻辑 bug。

- [x] **实验 A：降高确认到达**（高空抓到候选不立刻停，朝目标居中+降高到细分辨率再核验）→ **回退**。
  实测对远距离 hard 题有效（NE 1331→158），但**缩小视场会丢失近处目标**，把原本能成的近距离例毁掉
  （NE 23.8→968），整体 SR 反降，故回退（保留 `vln_navigator._clamp_step` 等，移除降高门控）。
- [x] **保留：多地标 target_phrase 修复**（停用词补 先/再/然后/接着 等连接词，避免 legacy 短语被抽成乱码）。
- [x] **保留：评测加语义判定 semSR/semNE**（到达瓦片内任一**同类**受损建筑即算成功；class 级指令更公平）。
  实测 semSR 仍≈0、semNE≈NE——因为这些瓦片目标类建筑稀疏、GT 往往就是最近的同类，语义判定救不动，
  进一步印证瓶颈在 grounding 精度/召回而非目标歧义。

**第二版成绩单（回退降高确认 + 语义判定）** `runs/benchmarks/20260623_163157_e1sem/`：

| 配置 | SR | semSR | NE(m) | semNE(m) | Steps |
|---|---|---|---|---|---|
| B0 | 0.0 | 0.0 | 218 | 202 | 3.0 |
| B1 | 0.0 | 0.125 | 242 | 207 | 3.5 |
| B2 | 0.0 | 0.0 | 267 | 220 | 3.6 |
| B3 | 0.0 | 0.0 | 194 | 180 | 3.4 |

> 注：n=8、repeat=1，VLM 采样随机，**run 间方差极大**（同 B0 两次 NE 86 vs 218），小样本不可作结论。
>
> **下一步真正的杠杆（按性价比）**：① grounding 精度——低空裁剪复核 / 多尺度搜索 / 在 xBD 上微调检测器；
> ② 搜索视场——抬高搜索高度扩 FOV 先粗定位再逼近（需做 alt 扫描实验）；③ 扩样本到全量降方差，再下消融定论。
> 单纯调"停车阈值"头部空间有限（VLM 常报死中心，阈值再紧也拦不住），故不再盲调。

#### 实验：裁剪复核（杠杆①轻量版）— 实测**无效，已默认关**

- 做法：`_vln_vlm_ground` 粗命中后，以命中点为中心裁原 patch 的 0.34 边长小窗，放大到 512² 让 VLM 再精定位一次，
  窗内坐标线性映射回原 patch 归一化坐标（`_vlm_refine_xy`）。不动 UAV、不缩真实视场，仅多一次 VLM 调用。开关 `VLN_VLM_REFINE`。
- A/B（B1×8 题，grounder=vlm，仅切 `VLN_VLM_REFINE` 0/1）：

| 复核 | SR | semSR | NE(m) | semNE(m) | 中位NE | run |
|---|---|---|---|---|---|---|
| OFF | 0.125 | 0.125 | 175 | 149 | 99 | `runs/benchmarks/20260623_170348_refoff/` |
| ON  | 0.0   | 0.0   | 290 | 244 | 151 | `runs/benchmarks/20260623_170808_refon/` |

- 逐题：**5 题更差 / 1 题更好 / 2 题持平**（典型恶化 24m→978m、27m→156m；唯一改善 814m→103m）。
- 原因：高空 patch 本就低分辨率，"数字放大"只放大模糊；二次 VLM 常改选裁剪窗内**相邻另一栋**建筑，引入新误差，得不偿失。
- 结论：轻量裁剪复核**此路不通**，已将 `VLN_VLM_REFINE` 默认置 0（代码保留可复现实验）。真正要降 NE 需走
  **多尺度/抬高搜索粗定位 + 在 xBD 上微调检测器**，而非对同一张低分辨率图二次定位。

#### 杠杆②（主攻）：在 xBD 上微调域内检测器 — 给 grounding 一个像素级精确来源

**动机**：现用 YOLO 是 RescueNet（低空斜拍无人机）权重，在 xBD（卫星正射 1024²）上几乎**检出为 0**（域不匹配）。
于是 grounding 只能死靠开放词汇 VLM，而 VLM 俯视空间精度差（常报死中心）→ 假到达、NE 大。
若有一个在 xBD 上训练的检测器，`ground_with_yolo` 已具备的"类别过滤 + 面积/置信/居中打分 + 框中心→归一化坐标"链路即可直接产出**像素级精确**的目标点。

**数据**：`/home/lc/datasets/xbd`，仅用 **post_disaster** 标签（pre 无损伤分级）。
- 训练池：`train`(2799) + `tier3`(6369) = **9168 张**；独立测试：`test`(933 张)。
- 标注：每栋楼 `features.xy` 的 POLYGON → 外接 bbox（clamp 到 [0,W]/[0,H]），`subtype` → 类别。
- 4 类（对齐 perception 中文标签）：`no-damage/minor-damage/major-damage/destroyed`；`un-classified` 跳过。

**类别对齐（关键、纯增量）**：训练时类别名用 xBD 英文 subtype；在 `perception.YOLO_LABEL_MAP` 加 4 条
`no-damage→无损伤建筑 / minor-damage→轻微损伤建筑 / major-damage→严重损伤建筑 / destroyed→完全损毁建筑`。
perception 已有 `.get(raw, raw)` 兜底，grounding 按中文类匹配，**无需改 grounding 逻辑**。

**步骤**：
1. `scripts/training/gen_xbd_yolo_dataset.py`：xBD post 标注 → YOLO 检测集（图软链 + txt 标签 + `data.yaml`），
   train 池按瓦片 95/5 切 train/val，xBD test 作 test，输出到 `/home/lc/datasets/xbd_yolo/`。
2. 抽检：随机几张画框可视化，确认 bbox/类别正确。
3. 训练：`yolov8s`，`imgsz=1024`（卫星小目标，需大分辨率），GPU3 后台，权重落 `runs/train/`。
4. 集成：`YOLO_LABEL_MAP` 加英文→中文；perception `YOLOTool` 支持 `YOLO_IMGSZ`（默认 1024）；`YOLO_WEIGHTS` 指向新 `best.pt`。
5. 评测：`diag_grounding.py` 确认 YOLO 在目标处检出>0；bench 跑 `grounder=yolo` 与 `hybrid`，与 `vlm` 比 NE/SR。

**验收**：xBD test mAP50 有意义（>0.3 量级即可用）；diag 在目标处 YOLO 检出>0；bench `grounder=yolo/hybrid` 的 NE 显著低于 `vlm`。

**结果（已完成，2026-06-23）** — 验收**通过**，是目前最有效的一招：

- 训练：`yolov8s` imgsz1024，60 epoch，权重 `runs/train/xbd_yolov8s_1024/weights/best.pt`。
- 检测精度（xBD test，739 图）：mAP50=**0.342**，逐类 no-damage 0.445 / minor 0.216 / **major 0.454** / destroyed 0.253。
  正射视角下损伤分级本就难（destroyed/minor 偏低），但相比旧 RescueNet 权重"检出≈0"已是质变。
- diag：新 YOLO 在 xBD 上**真的检出建筑**（旧权重全 0）；但受损类召回不足，目标多被判 no-damage。
- VLN 评测（B1×8 题，只切 grounder）：

| grounder | SR | semSR | NE(m) | semNE(m) | Steps | run |
|---|---|---|---|---|---|---|
| vlm（旧） | 0.125 | 0.125 | 175.4 | 149.4 | 3.0 | `20260623_170348_refoff/` |
| **yolo(xBD)** | 0.125 | **0.25** | **110.4** | **81.1** | 2.1 | `20260623_195917_yolo/` |
| hybrid | 0.125 | 0.25 | 110.4 | 81.1 | 2.1 | `20260623_200453_hybrid/` |

- **semNE 几乎砍半（149→81m）、NE 降 37%、步数更少**。核心增益：YOLO 给像素级框中心，
  消除了 VLM 的灾难性远距离误命中（单题 814m→117m、semNE 721m→117m）。
- 逐题 better/worse=3/3（±5m），但赢在量级——回归集中在 palu-tsunami（海啸影像，训练池占比少）。
- strict SR 仍 0.125：受损分级召回是新瓶颈（destroyed mAP 0.25）；yolo 与 hybrid 同分，说明 YOLO 命中主导结果。
- **下一步**：①降 `YOLO_CONF_THRESHOLD` 提受损类召回；②按灾型补训/重采样（海啸欠拟合）；③扩题量降方差后再下定论。

**第三版成绩单（E1 全量：40 题 × B0~B3，grounder=hybrid，repeat=1）** `runs/benchmarks/20260623_202108_e1full/`：

| 配置 | SR | semSR | NE(m) | semNE(m) | SPL | Steps | ΔU | judge_acc |
|---|---|---|---|---|---|---|---|---|
| B0 | 0.075 | 0.125 | 150 | 136 | 0.072 | 3.3 | — | 0.034 |
| B1 | 0.075 | 0.125 | 226 | 193 | 0.068 | 4.8 | — | 0.036 |
| B2 | 0.025 | **0.15** | 230 | **190** | 0.025 | 5.6 | 0.0 | 0.033 |
| B3 | 0.025 | 0.075 | 301 | 261 | 0.025 | 5.5 | 0.0 | 0.077 |

> 注：40 题、repeat=1；grounder=hybrid（xBD 域内 YOLO + VLM 兜底）。完整聚合报告见同目录 `report.md`。
>
> **E8 难度分桶**：medium 题 B1 semSR=0.25 最好；hard 题全线 SR≈0（NE 200~360m），瓶颈在远距离 grounding。
> **E6 跨灾种**：palu-tsunami / mexico-earthquake NE 最低（72~84m）；hurricane-michael 最难（NE 180~396m）。
> **E3 复核（B1 vs B2）**：semNE 193→190 略降，Steps 4.8→5.6（+0.8），ΔU≈0——复核在导航 episode 里几乎未触发有效定论（受损类检出不足 → 无证据）。
>
> **阶段性结论**：B0 baseline NE 最低（150m），HSPM hard 题绕远；B3 记忆本 run 无收益；SR 仍低但 semSR/semNE 更有区分度。
> E2 grounder 三选一进行中；E5~E7 鲁棒性待做。

### P4.5 消融矩阵

| 配置 | 开关 | 看什么 |
|---|---|---|
| B0 baseline | `VLN_PLANNER=legacy` | 关键词+贪心对照 |
| B1 +HSPM | `VLN_PLANNER=hspm` | H1 |
| B2 +复核 | B1 + `VLN_RECHECK=1` | H2 |
| B3 full | B2 + `VLN_MEMORY=1` | H3 |
| B1 + OROI-Score | B1 + `VLN_OROI_SCORE=1` | C3 工程改进消融（非 headline，见 E12） |
| B1 + FBE（纯几何基线） | B1 + `VLN_OROI_SCORE=1` 三路权重设为 `(0,0,1)` | 经典 Frontier-Based Exploration 对照（见 E12） |

- grounding 维度：`VLN_GROUNDER ∈ {yolo, vlm, hybrid}` 交叉对比（H 系列默认固定一种，单列 grounding 对比表）。
- OROI 打分融合（借鉴 Say-REAPEx 的打分式动作选择）：HSPM 的 `reason_oroi` 原本让 LLM 在八方位里"自由选一个"，改为对 8 个方位分别用「LLM affordance + 方向先验一致度 + 未探索区域增益」三路信号加权打分再取最大，`VLN_OROI_SCORE=1` 开启。这是 C3（HSPM 运动层）的工程改进，**不作为独立 contribution**，只作为消融条目验证是否缓解 hard/多地标题绕远的问题。

### P4.6 鲁棒性 / 混合 grounding（待做）

- [ ] 混合 grounding：YOLO（域内、快）优先，认不出/出现新词再用开放词汇模型兜底
  （接 `locate_ground_fn` → LocateAnything，子进程/HTTP worker，独立 conda 环境）。
  实测零样本偏弱（F1@0.5≈0.14、损伤分级差）→ 加置信度过滤 / NMS / 单实例约束。
- [ ] GPS 噪声：起点/观测注入高斯噪声（σ 扫描），画 σ vs SR 衰减曲线。
- [ ] 跨灾种泛化：tsunami/earthquake/flood 分别建子题库，交叉评测。
- [ ] 退化视场：复用 perception `degraded` 路径，量化对 SR/ΔU 的影响。

### P4.7 实验清单（逐条说明，大白话）

> 先把几个指标翻译成人话：
> - **SR（成功率）**：100 道题里，无人机最后停在目标 25 米内的比例。**越高越好**，最重要。
> - **NE（导航误差）**：最后停的位置离正确目标差几米。**越小越好**。
> - **SPL（带路径的成功率）**：成功的同时有没有"走直线、不绕路"。绕远路会扣分。**越高越好**。
> - **Steps（步数）**：飞了几步才结束，代表花了多少力气。**越少越好**（但不能为了省步数而失败）。
> - **ΔU（不确定性下降）**：复核前"拿不准"，复核后"更有把握"，把握提升了多少。**越大说明复核越值**。
> - **判定准确率**：无人机对目标"受灾等级"的判断，和真实标注对不对得上。**越高越好**。
>
> 下面把上面的方案拆成一条条**能直接开跑**的实验。统一做法：固定随机种子、每题重复 3 次取平均，
> 结果落 `runs/benchmarks/<run_id>/results.json`，最后用 `bench_report.py` 汇总成表和图。

**E1 — 主消融：四个版本同台比武（核心成绩单，对应 H1+H2+H3）**
- 想搞清楚：HSPM、复核、记忆这三样东西，**各自到底加了多少分**。
- 怎么做：同一套 40 道题，分别用 4 个配置各跑一遍：
  - B0 = 老办法（关键词+贪心），B1 = +HSPM，B2 = B1+复核，B3 = B1+复核+记忆。
  - 一步步往上加功能，看指标怎么变（这叫"消融"）。
- 看哪些数：SR / NE / SPL / Steps 排成一张主表，每加一个功能涨了多少。
- 期望结论：B1>B0 说明会读长指令更强；B2 的 ΔU 和判定准确率更高（步数会变多，正常）；B3 在重复区域步数更省。

**E2 — 目标识别器三选一（grounding 对比）**
- 想搞清楚：找目标到底该用哪种"眼睛"。
- 怎么做：固定一个配置（一般用 B1），把 `VLN_GROUNDER` 换成 `yolo` / `vlm` / `hybrid` 各跑一遍。
  - yolo = 域内训练过、快但只认固定类别；vlm = 能认开放词汇（如"蓝色屋顶"）但慢、易保守；hybrid = 先 yolo 再 vlm 兜底。
- 看哪些数：三者的 SR / NE，外加平均耗时。
- 期望结论：hybrid 在"成功率"和"速度"之间最均衡；vlm 在开放词汇题上救回 yolo 找不到的目标。

**E3 — 复核到底值不值（对应 H2）**
- 想搞清楚：让无人机"没把握就飞近降高再看一眼"，是不是真能看得更准。
- 怎么做：B1（不复核） vs B2（复核），只对比"看得准"相关的指标。
- 看哪些数：ΔU（把握提升多少）、判定准确率、以及为此多花的步数。
- 期望结论：B2 的 ΔU>0、判定准确率明显高于 B1；代价是步数上升——用"多花的步数换来的准确率提升"来论证值得。

**E4 — 记忆越用越熟（对应 H3）**
- 想搞清楚：走过一次的区域，第二次是不是能少绕路。
- 怎么做：开 `VLN_MEMORY=1`，在**同一片区域**先跑一批指令（让它记住路），再跑一批相似指令。
  - 对照：关掉记忆再跑同样的第二批。
- 看哪些数：第二趟的 Steps / 路径长，相对首趟（或相对关记忆）的下降比例；同时确认 SR 没掉。
- 期望结论：第二趟更省力（步数/路径下降），且成功率不降——"记忆是净赚"。

**E5 — GPS 会飘，抗不抗造（对应 H4）**
- 想搞清楚：定位有误差时，系统会不会就崩了。
- 怎么做：给起点/观测位置注入高斯噪声，噪声强度 σ 从小到大扫一遍（如 0/2/5/10 米）。
- 看哪些数：σ 越大，SR 掉多少——画一条"σ vs 成功率"的曲线。
- 期望结论：SR 随噪声平缓下降而不是断崖式崩，说明系统有容错。

**E6 — 换个灾种还行不行（跨灾种泛化，对应 H4）**
- 想搞清楚：在海啸场景调好的东西，搬到地震/洪水还灵不灵。
- 怎么做：按灾种拆成 tsunami / earthquake / flood 三个子题库，分别评测（必要时交叉）。
- 看哪些数：每个灾种各自的 SR / NE，看差距大不大。
- 期望结论：各灾种 SR 都在可接受范围、没有某个灾种特别拉胯，说明方法不挑场景。

**E7 — 看不清的时候（退化视场，对应 H4）**
- 想搞清楚：图像模糊/视场退化（perception 的 `degraded` 情况）时表现如何。
- 怎么做：在退化视场下重跑题库（或注入退化），对比正常视场。
- 看哪些数：SR / ΔU 掉多少；复核机制是否在这种情况下更频繁触发、能不能补救。
- 期望结论：退化下指标下降但复核能部分兜底，体现"没把握就再看一眼"的价值。

**E8 — 难、中、易分开看（瓶颈分析）**
- 想搞清楚：失败主要发生在哪种题上（题库已自带 `difficulty` 标签）。
- 怎么做：把 E1 的结果按 easy / medium / hard 分桶统计。
- 看哪些数：每个难度桶的 SR / NE。
- 期望结论：定位瓶颈（比如远距离 hard 题是主要失败来源），指导后续改进方向。

**E10 — 校准前后不确定性质量对比（对应 P5）** ✅ 已完成（2026-07-11）

- 想搞清楚：换成温度校准的熵之后，`U_t` 是不是真的比查表启发式更"诚实"。
- 怎么做：在 xBD test 集上跑 `calibration_bench.py`，对比 `heuristic` 与 `entropy`（标定前/后）三版的 ECE / Brier Score / NLL，画 reliability diagram。
- 期望结论：标定后的熵版本 ECE 明显低于启发式版本，且低于未标定的熵版本（证明温度标定本身有效）。
- **实测结果**（`change_perception.py` 训练 6 epoch，best val_acc=0.731，学到的温度 T=1.668）：

| 子集 | n | 版本 | ECE | Brier | NLL | Acc |
|---|---|---|---|---|---|---|
| test（同源） | 21717 | 未标定 T=1.0 | 0.2762 | 0.7276 | 1.7057 | 0.524 |
| test（同源） | 21717 | 标定 T=1.668 | **0.1674** | **0.6534** | **1.2864** | 0.524 |
| holdout（跨灾害，moore-tornado/nepal-flooding/pinery-bushfire） | 69307 | 未标定 | 0.2777 | 0.7377 | 2.2233 | 0.516 |
| holdout（跨灾害） | 69307 | 标定 | **0.1820** | **0.6729** | **1.5891** | 0.516 |

结论符合预期：温度标定把 ECE 降了 ~39%（test）/ ~34%（holdout），Brier/NLL 同步下降，Acc 不变（标定只重塑分布形状，不改 argmax，符合定义）。跨灾害 holdout 上标定收益同样成立，说明温度标定不是过拟合到训练时见过的灾害类型。**注意**：这里的模型训练时用的子采样训练集未做 tier3 灾种排除（与 E13 的 YOLO 检测器同一限制），holdout 数值是"标定流程在跨灾害数据上的演示"而非严格 unseen 泛化实验；严格版本需要用排除 holdout 灾种后的训练集重训 `change_perception`，留作后续工作。产物：`runs/benchmarks/calibration/{test,holdout}_{summary.json,reliability.png}`。

**E10b — 温度标定 vs MC-Dropout vs Deep Ensemble（对应"用算法改动超越" baseline 调研）** ✅ 已完成（2026-07-25）

- 想搞清楚：E10 只验证了"温度标定比不标定好"，但文献（Ovadia et al. 2019《Can You Trust Your Model's Uncertainty?》、Lakshminarayanan et al. 2017 Deep Ensembles）指出温度标定只是最轻量的后处理手段，MC-Dropout / Deep Ensemble 通常校准质量更好。调研后把这两个方法接入同一套 `calibration_bench.py` 评测流程，在完全相同的 test/holdout 子集上直接对比。
- 怎么做：① **MC-Dropout**（Gal & Ghahramani 2016）：给 `ChangeMultiTaskNet` 两个头加 `dropout_p=0.3`（`--dropout` 开关），训练方式不变，推理时保持 Dropout 随机采样、其余层仍用 eval 统计量，跑 T=30 次前向，在概率空间取平均；② **Deep Ensemble**（Lakshminarayanan et al. 2017）：额外训练 2 个不同 `--seed` 的独立模型（连同 baseline_seed0 共 K=3 个成员），每个成员用自己学到的温度做 softmax 后在概率空间取平均（Ovadia et al. 推荐的 pool-then-average 做法，而不是共享一个温度）。
- **实测结果**（同一 baseline_seed0 作为单模型对照，E15 用的同一批 test=21717/holdout=69307 子集）：

| 方法 | test Acc | test ECE | holdout Acc | holdout ECE |
|---|---|---|---|---|
| 单模型 + 温度标定（E10 做法） | 0.548 | 0.1129 | 0.521 | 0.1494 |
| MC-Dropout（T=30 次前向）+ 温度标定 | 0.508 | 0.1327 | 0.527 | 0.1409 |
| **Deep Ensemble（K=3，各自标定后取平均）** | **0.553** | **0.0909** | **0.534** | **0.0961** |

- **结论**：三者里 **Deep Ensemble 全面最优**——Acc 和 ECE 在 test/holdout 上都优于单模型温度标定（test ECE 降 19%、holdout ECE 降 36%），和文献结论完全一致，代价是要独立训练/保存/推理 K 个模型（本次 K=3，训练成本 3 倍）。**MC-Dropout 没有体现出预期优势**：test 上 Acc 反而下降（0.548→0.508，dropout 在小模型上对同域拟合有一定损伤），ECE 也没有比单纯温度标定更好（0.1327 > 0.1129）；只在 holdout 上 ECE 略有改善（0.1409 vs 0.1494）。如实报告这个"MC-Dropout 不如预期"的中性结果：dropout_p=0.3、T=30 次前向是本次固定的超参组合，MC-Dropout 对 dropout 率/次数比较敏感，不能排除调参后表现更好的可能，但至少在当前设置下，**温度标定仍是"性价比"最高的选项，Deep Ensemble 是"效果优先"时的更优替代**，而不是无脑三选一都要上。产物：`backend/outputs/change_perception/{ensemble_seed1,ensemble_seed2,mc_dropout}.pt`、`runs/benchmarks/calibration_e10b/`。

**E11 — 复核策略六选一（对应 P5，深化 E3）** ✅ 已完成（2026-07-11，n=40/档，`vln_testset.json`，grounder=vlm）

- 想搞清楚：到底哪种复核策略最值。
- 怎么做：固定 B1+复核开启，`VLN_RECHECK_TRIGGER`/`VLN_UNCERTAINTY_MODE` 组合出六档：不复核 / 随机复核 / 固定降高 / 现有启发式阈值 / 校准熵阈值 / 信息增益驱动（`recheck.py` 新增 `trigger_mode="fixed"/"random"` 支持这两档对照基线）。
- 看哪些数：复核前后判定准确率变化、每多花一步换来的准确率增益、错误停止率、风险-覆盖率曲线——不只看 ΔU。
- 期望结论：信息增益驱动版本在"准确率增益/步数"上优于阈值版本，阈值版本优于随机/固定档。
- **第一版实测结果（2026-07-11，ΔU 恒为 0，事后查出是 bug）**：

| 配置 | SR | semSR | NE(m) | SPL | Steps | ΔU | judge_acc |
|---|---|---|---|---|---|---|---|
| 不复核 | 0.050 | 0.150 | 170.4 | 0.043 | 4.88 | — | 0.034 |
| 随机复核 p=0.5 | 0.025 | 0.075 | 248.1 | 0.025 | 5.95 | 0 | 0.037 |
| 固定降高（有证据必复核） | 0.025 | 0.100 | 224.3 | 0.025 | 5.43 | 0 | 0.071 |
| 现有启发式阈值 | 0.075 | 0.100 | 262.5 | 0.075 | 5.43 | 0 | 0 |
| 校准熵阈值 | 0.025 | 0.075 | 268.8 | 0.025 | 6.20 | 0 | 0.036 |
| 校准熵+信息增益 | 0.025 | 0.100 | 271.0 | 0.025 | 6.23 | 0 | 0.036 |

bootstrap 95% CI + 配对置换检验（`bench_report.py`）：六档两两 SR 差异全部**不显著**（p 全部 ≥ 0.51，n=40 时置信区间宽达 [0,0.18]）——**没有支持"信息增益驱动更优"这个预期结论**，n=40 下六档几乎不可分。更值得记录的反而是一个**方法论发现**：六档 ΔU（`avg_uncertainty_reduction`）全部恒为 **0.0**，不是"下降很小"而是字面上的零。产物：`runs/benchmarks/20260711_185109_e11a/`（前四档）+ `runs/benchmarks/20260711_194030_e11b/`（entropy/infogain 两档）。

- **第二版实测结果（2026-07-12，修复两个 ΔU bug 后重跑，n=40/档不变）**：

| 配置 | SR | semSR | NE(m) | SPL | Steps | ΔU(聚合) | judge_acc |
|---|---|---|---|---|---|---|---|
| E11_NONE 不复核 | 0.025 | 0.075 | 268.8 | 0.025 | 6.20 | — | 0.036 |
| E11_RANDOM 随机复核 p=0.5 | 0.075 | 0.200 | 209.5 | 0.044 | 5.20 | -0.0 | 0 |
| E11_FIXED 固定降高 | 0.050 | 0.150 | 227.6 | 0.043 | 5.60 | 0.0 | 0.034 |
| E11_HEURISTIC 现有启发式 | 0.075 | 0.150 | 226.1 | 0.044 | 5.50 | 0.0 | 0 |
| E11_ENTROPY 校准熵阈值 | 0.025 | 0.075 | 270.4 | 0.025 | 6.475 | 0.0 | 0.036 |
| E11_INFOGAIN 校准熵+信息增益 | 0.075 | 0.125 | 184.7 | 0.068 | 4.825 | 0.0 | 0.033 |

bootstrap 95% CI + 配对置换检验：六档两两 SR 差异仍全部**不显著**（p 全部 ≥ 0.49），SR/NE 数字与修复前几乎没变（符合预期，两个 bug 影响的只是 ΔU 统计，不影响导航决策本身）。**聚合 ΔU 看上去仍然约等于 0，但这次不是 bug，是被稀释了**：逐题查 `episodes.jsonl` 发现，40 题里**只有 1 题**（`midwest-flooding_00000400_post_disaster__4790`）在探索过程中真的检测到过受灾证据、触发了复核，其余 39 题全程 `risk_level="none"`，对该题贡献的 ΔU 恒为 0——40 个数取平均，1 个非零值自然被抹平成"0.0"。**只看这唯一命中证据的题，四种"有真复核"的档位显示出清晰的梯度**：

| 配置 | 该题 ΔU（唯一命中证据的样本） |
|---|---|
| E11_RANDOM（随机） | -0.007（复核后反而更不确定） |
| E11_FIXED（固定降高，无信号） | 0.001 |
| E11_HEURISTIC（heuristic 查表） | 0.001 |
| E11_ENTROPY（校准熵阈值） | 0.011 |
| E11_INFOGAIN（校准熵+信息增益） | 0.013 |

这个梯度**方向符合最初的假设**：校准概率（entropy/infogain）比 heuristic 查表对"降高看得更清楚"这件事更敏感，随机复核（不看任何信号）甚至可能变差。但样本量是 **n=1**，只能当作"修复生效、方向正确"的定性证据，不能作为统计意义上的结论——真正验证需要一个"受灾证据出现率更高"的题库（当前 40 题库里 97.5% 的题全程无证据，复核这条支线几乎没有被真正考验过）。产物：`runs/benchmarks/20260712_002632_e11c_*/`（NONE/RANDOM/FIXED/HEURISTIC）+ `runs/benchmarks/20260712_011133_e11c_*/`（ENTROPY/INFOGAIN），合并报告 `runs/benchmarks/e11c_六档汇总报告.md`。

**ΔU=0 根因排查（2026-07-11 复盘，两层问题，均已修复代码，未重跑六档实验）：**

最初怀疑是纯统计口径问题（"未 resolve 的复核不计入平均值"），已修复（见下），但补丁后拿真实 episode 重新抽样验证时发现 ΔU 依然是 0——说明**真正的根因在更上游**：

1. **根因（主要）：巡航高度=复核高度下限，复核从触发起就"没有降的空间"。** `app.py` 的 `DEFAULT_HOVER_ALTITUDE_M=30.0` 恰好等于旧默认的 `VLN_RECHECK_ALT_MIN_M=30.0`。`recheck.py` 里判断"到高度下限"的条件是 `alt <= alt_min_m`，episode 全程巡航高度都是 30m，所以这个条件从第一次 `assess()` 起就恒为真——复核证据一旦出现，直接走"预算耗尽/到高度下限"分支**立即 resolve**，根本拿不到 `kind="recheck"` 的真实"降高+居中"机动；而这次 resolve 的 before/after 又是同一次观测（还没来得及降高看清楚），reduction 自然恒为 0。**修复**：把 `VLN_RECHECK_DESCEND_M` / `VLN_RECHECK_ALT_MIN_M` 默认值从 30/20 改成 10/10（`backend/app.py`、`backend/recheck.py` 的 `RecheckConfig` 默认值同步改），30m 巡航高度下留出 20m 真实可降空间，够两步各降 10m 后再收尾。
2. **次要（统计口径，已一并修复，防御性加固）：episode 常在复核循环走到正式 resolve 之前就结束**（到达终点/步数耗尽打断），导致 `resolved_log` 漏记这部分样本。`RecheckController` 新增 `finalize()`：episode 收尾时把仍"复核中"但没等到定论的位置，按最新一次观测补记一笔账（status="episode_end"，不计入 confirmed/dismissed/inconclusive，但计入 `avg_uncertainty_reduction`），`app.py` 在写 `summary`/`report` 前调用一次。单测见 `test_finalize_flushes_pending_reduction` / `test_finalize_uses_latest_observation`。

**修复后的实测验证**（`trigger_mode="fixed"` + 修复后默认值，抽样 16 题里恰好命中 1 题有真实受灾证据 `midwest-flooding_00000400_post_disaster__4790`）：
```
{'resolved': 8, 'confirmed': 2, 'dismissed': 0, 'inconclusive': 1,
 'episode_end_pending': 5, 'avg_uncertainty_reduction': 0.001, 'pending': 0}
```
两点确认：① `resolved=8` 里有 `episode_end_pending=5`——如果没有 `finalize()` 补丁，这 5 条会直接从统计里消失；② `avg_uncertainty_reduction` 从恒为 0.0 变成 **0.001**（非零但很小）——说明高度下限修好后复核机制真的能跑出"降高→再观测→resolve"的完整链路，但 heuristic 不确定性公式（risk 暧昧度 + 1-conf）对"降到 10m 后看得更清楚"这件事本身不敏感，降高带来的置信度提升幅度有限。**这不再是"统计口径 bug"或"配置死锁"，而是一个更真实的、值得在 entropy/校准概率模式下重新检验的效果量问题**——加上 `VLN_CHANGE_PERCEPTION` 的校准熵/信息增益模式，理论上应该比 heuristic 公式对分辨率变化更敏感，值得优先重跑 E11_ENTROPY/E11_INFOGAIN 两档验证。

**尚未做的事（已更新为第二版结果之后的后续项）**：两个 bug 已修复、六档已重跑完并写进上面的第二版结果表；下一步不是再重跑同一份 40 题库（几乎必然还是只命中 1 题），而是①**构造一个"受灾证据出现率更高"的小题库**（比如直接挑 xBD 里 major/destroyed 密度高的瓦片、缩短起点-目标距离，让探索过程更容易撞见受损建筑），专门用来给 ΔU/复核策略做统计意义上的对比，和现有面向 SR/NE 的题库分开维护；②在这个新题库上重跑六档，届时才适合再做一次 bootstrap CI/配对检验来判断"entropy/infogain 显著优于 heuristic/random"这个方向性发现是否成立；③把 `stats()` 增加"按是否命中过 `EVIDENCE_CLASSES` 分层"的口径，避免以后又被"全程无证据"的样本稀释成 0。

**E12 — OROI 打分 vs 自由选择 A/B（对应 P4.5 新增行，C3 工程改进）** ✅ 已完成（2026-07-11，n=40/档）

- 想搞清楚：把 LLM 的方位选择从"自由拍板"换成"多信号打分"，会不会缓解 B1 在 hard/多地标题上绕远的问题。
- 怎么做：固定 B1，只切 `VLN_OROI_SCORE`（0/1），其余配置不变，跑同一题库。
- 看哪些数：重点看 E8 难度分桶里 hard 桶的 NE/Steps 有没有收窄（当前 B1 hard NE≈291m vs B0≈143m）。
- 期望结论：打分版本 hard 桶 NE 下降，但这是 C3 的工程改进证据，不写进 contribution 列表。
- **实测结果**：整体 SR 0.050→0.075、SPL 0.043→0.071 有小幅提升，但 semSR 反而下降（0.150→0.100），NE 变差（170→203m）；bootstrap 配对检验 ΔSR 不显著（p=1）。难度分桶：**hard 桶并未如预期收窄**（OFF 181.7m vs ON 189.1m，几乎没变化甚至略差），反倒是 medium 桶 SR 从 0.167 升到 0.25。跨灾种拆解显示效果方向在不同灾害间也不一致（`mexico-earthquake` 上 SR 0→0.33，`hurricane-michael` 上几乎不变）。**结论**：n=40 下看不出 OROI 打分对 hard/多地标题有稳定收益，原始假设（收窄 hard NE）未被证实；维持"非 headline、写成消融行"的定位是对的，且应在文字里如实报告这个中性/混合结果，而不是只挑 SR/SPL 的好看数字。产物：`runs/benchmarks/20260711_185120_e12/`。

- **新增第三档 E12_FBE（对标主流探索文献的标准基线）** ✅ 已完成（2026-07-25，n=40，`vln_testset.json`，grounder=vlm）：调研 ObjectNav/主动探索文献后发现，之前的 E12 只对比了"LLM 自由选择"与"打分融合"两档，缺一个来自经典机器人学的标准基线——**Frontier-Based Exploration**（Yamauchi 1997：永远朝"未探索覆盖增益"最大的方向走，完全不依赖任何语言/语义信号）。做法：把 `score_oroi` 的三路权重设成 `(llm=0, prior=0, frontier=1)`，复用已有的 `SemanticMap.frontier_score` 探针，且权重为 0 时直接跳过 LLM 调用（不是"权重恰好乘 0"，是让基线在算法定义上就不依赖 LLM）。

| 配置 | 说明 | n | SR | semSR | NE(m) | semNE(m) | SPL | Steps |
|---|---|---|---|---|---|---|---|---|
| E12_OFF | OROI 自由选择（=B1） | 40 | 0.050 | 0.150 | 170.351 | 145.054 | 0.043 | 4.875 |
| E12_ON | OROI 打分融合 | 40 | 0.075 | 0.100 | 203.041 | 183.083 | 0.071 | 5.425 |
| E12_FBE | 纯 Frontier-Based Exploration | 40 | 0.025 | 0.050 | 389.14 | 341.978 | 0.022 | 6.475 |

**结论**：纯 FBE 是三档里最差的——SR/semSR 最低，NE/semNE 几乎是另外两档的 2 倍，Steps 最多（说明走了更多弯路却更难命中目标）。这个结果符合直觉但仍值得写进论文：**"最大化未探索覆盖"本身不是一个面向目标的信号**——它只会让无人机均匀地探索全图，而不会像 LLM 自由选择或 OROI 打分那样，哪怕只是模糊地"朝已发现的受损建筑方向"或"朝语义先验方向"走。这为本项目在 exploration 阶段引入语言/语义信号（而不是纯几何覆盖）提供了一个来自标准基线的负向对照证据：不加语义引导的经典探索算法在"找特定语义目标"这个任务上明显更差，从而间接支撑 C3（HSPM 三层规划 + OROI 语义打分）存在的必要性。产物：`runs/benchmarks/20260725_153648_e12_fbe/`。

**E13 — 跨灾种泛化 + 规模统计检验（对应 P6）** ✅ 已完成（2026-07-11）

- 想搞清楚：① 检测器在真正没见过的灾害类型上表现如何（P6 tier3 留出集）；② 40 题的消融结论在更大样本下是否稳健。
- 怎么做：用留出的 3~4 种 tier3 灾害跑一版独立 mAP；把题库扩到 200~500 题后重跑 E1 主消融，`bench_report.py` 出带 bootstrap 95% CI 的表，并做配对检验。
- 期望结论：跨灾害 mAP 低于同源 test（说明之前的数字有一定虚高，属正常现象，报告出来即可）；扩样本后消融结论的方向不变但置信区间收窄。
- **实测结果 ①（检测器 xbd_yolo_v2）**：test 集 mAP50=0.368 / mAP50-95=0.176（739 图，53850 实例）；跨灾害 holdout 集（moore-tornado / nepal-flooding / pinery-bushfire）mAP50=0.381 / mAP50-95=0.206（1223 图，69391 实例）——**跨灾害 mAP 没有降低，反而略高于同源 test**，与预期结论相反。这不代表模型真的泛化到未见灾种：`best.pt` 训练时用的是旧版数据集切分，这些 tier3 灾害当时**已经被并入训练池**（详见 P6 的 tier3 切分修复说明），所以本次 holdout 评测只是新版 `gen_xbd_yolo_dataset.py` 切分流程的正确性演示，不是严格的 unseen-disaster 泛化实验；holdout 略高的 mAP 更可能是这几种灾害本身建筑边界更规整/损伤更明显。严格版本需要用 `xbd_yolo_v2/train`（已排除这三种灾害）重新训练 `best.pt` 再评测，受时间限制未执行，留作后续工作，已在代码注释与本节如实说明。
- **实测结果 ②（题库扩至 240 题，10 种灾害，B0~B3 各跑 30 题 `--limit 30`）**：

| 配置 | SR (95% CI) | n |
|---|---|---|
| B0 | 0.067 (0.000, 0.167) | 30 |
| B1 | 0.033 (0.000, 0.100) | 30 |
| B2 | 0.000 (0.000, 0.000) | 30 |
| B3 | 0.033 (0.000, 0.100) | 30 |

六对两两配对检验全部不显著（p≥0.5）。**结论方向与原 40 题 E1 一致**（B1/B2/B3 相对 B0 都没有稳定的 SR 提升，B2 这次甚至是 0），进一步坐实了"不能把 HSPM/复核/记忆当作稳定提升 SR 的 headline 结论"这一判断——这恰恰是本轮把核心贡献收窄到 C2（且 C2 的证据应看 judge_acc/效率而不是裸 SR）的依据。受时间/共享 GPU 资源限制，本次只跑了 240 题里的 30×4=120 题（而不是全部 240×4），置信区间比原计划窄化的目标（200~500 题全跑）弱一些，跑满全量题库留作后续工作。产物：`runs/benchmarks/20260711_185454_e13scale/`。

**E14 — 跨数据集泛化：xBD 训练的检测器搬到真实无人机影像上还灵不灵（对应"增强学术性"调研）** ✅ 已完成（2026-07-25）

- 想搞清楚：E13 的"跨灾害"其实是 xBD 内部换灾种（同一卫星正射视角、同一套标注规范），学术上更硬的证据是换一个**完全独立的第三方数据集**——不同传感器/视角（低空无人机 vs 卫星）、不同灾害事件、不同标注团队。调研后选定两个类别体系与 xBD 4 类损伤直接可对齐/部分可对齐的公开数据集：
  - **RescueNet**（Rahnemoonfar et al., *Scientific Data* 2023）：Hurricane Michael 低空无人机斜拍，10 类像素级分割，含与 xBD 完全一致的 4 档建筑损伤（No/Minor/Major-Damage + Total-Destruction）。
  - **FloodNet**（Rahnemoonfar et al., *IEEE Access* 2021）：Hurricane Harvey 低空无人机斜拍，仅 2 档建筑标签（Flooded/Non-Flooded），语义比 xBD 粗（"被水淹"≠"结构性损毁"）。
  - **BRIGHT**（Chen et al., *ESSD* 2025，14 个全球灾害事件、光学+SAR 双模态）调研后判定**不适合**直接接入现有 pipeline：它是"灾前光学+灾后 SAR"配对，而不是 xBD/本项目用的光学-光学双时相，SAR 图像不能直接喂给 RGB 检测器/变化感知模型，接入需要额外训练 SAR 域模型，超出"跑一次跨数据集验证"的范畴——已归入 related work 引用（第六节相关工作定位），未落地实验。
- 怎么做（两个子实验，脚本均新增在 `scripts/training/` + `scripts/benchmarks/`）：
  - **E14a（RescueNet，标准 mAP）**：`gen_rescuenet_yolo_testset.py` 把 RescueNet 测试集 450 张 mask（像素值 2~5 对应 4 档损伤）按连通域转成 YOLO 检测框（**注**：RescueNet 无实例级标注，"每个连通域一个框"是该数据集标注粒度本身的限制，非本脚本近似误差），零样本（不重新训练）跑 `xbd_yolov8s_1024` 的 `model.val()`。
  - **E14b（FloodNet，证据敏感度）**：因 FloodNet 标签体系更粗，不硬凑 4 类跑 mAP（会引入语义不一致的假结论），改用 `gen_floodnet_yolo_testset.py` + `eval_floodnet_evidence.py` 检验一个更具体、更贴合 C2 主线的问题——检测器对 IoU 匹配到的 flooded 建筑，判成"有损伤类"（minor/major/destroyed 任一）的比例，是否明显高于 non-flooded 建筑；这直接对应 `backend/recheck.py` 的 `EVIDENCE_CLASSES` 假设（"水/受损同属风险证据"）能否在训练域外的真实无人机影像上站得住。
- 期望结论：mAP 会低于 xBD 同源 test（跨域是常识），但如果保持一定水平/证据敏感度方向正确，仍是有价值的泛化证据；如果崩到几乎 0，则如实报告"域差主导，尚不具备跨视角泛化能力"。
- **实测结果 E14a（RescueNet，330/450 张含建筑损伤框，1115 个框）**：

| 子集 | n(图) | 类别数 | mAP50 | mAP50-95 |
|---|---|---|---|---|
| xBD test（同源，E13①） | 739 | 4 | 0.368 | 0.176 |
| xBD holdout（跨灾害，同数据集，E13①） | 1223 | 4 | 0.381 | 0.206 |
| **RescueNet test（跨数据集+跨视角，零样本）** | 330 | 4 | **0.021** | **0.014** |

逐类 mAP50：no-damage 0.074 / minor-damage 0.003 / major-damage 0.007 / destroyed 0.00001——**几乎完全崩溃**，相对 xBD 同源 test 暴跌 94%（0.368→0.021）。这与文档已记录的"旧 RescueNet 权重在 xBD 上检出≈0"（第 251~254 行）方向完全对称：**卫星正射视角（xBD）与低空斜拍视角（RescueNet）之间存在双向、几乎不可逾越的域差**，用其中一个视角训练的检测器不能指望零样本迁移到另一视角。这个负面结果本身有学术价值——它是"为什么本项目要做 Bird's-eye VLN 而不是直接复用第一视角/低空无人机检测器"这一问题定义（C1）的一条直接实证支撑，应写进 related work 或 limitation。

- **实测结果 E14b（FloodNet，199 张、604 个 flooded + 657 个 non-flooded 建筑框）**：

| 建筑类别 | GT 总数 | 定位召回（IoU≥0.3） | 命中框中判成"有损伤类"的比例 |
|---|---|---|---|
| flooded（真实被淹） | 604 | 67.2% | **0.49%** |
| non-flooded（真实未淹） | 657 | 37.7% | 4.03% |

两个反直觉的发现，如实记录：① 定位召回本身不算差（flooded 67%），说明检测器的"建筑候选框定位"能力有一定跨视角保留，问题主要在**损伤分类头**；② 损伤分类完全没有跨域迁移——不仅 flooded 建筑几乎全被判成 no-damage（仅 0.49% 判成有损伤类），non-flooded 建筑判成"有损伤"的比例反而更高（4.03% vs 0.49%），方向与假设**相反**。说明检测器在域外数据上的分类头基本退化成"默认输出 no-damage"，而不是真的学到了可迁移的损伤视觉特征。这进一步印证 E10/E11 的判断：**当前 C2（不确定性驱动复核）的收益完全依赖同域检测器的分类质量**，一旦换到训练时没见过的视角/传感器，"受灾证据"这个信号源本身先失效，复核机制无从谈起——这是比"样本量不足看不出差异"更根本的一层局限，建议作为 limitation 明确写出，而不是留给审稿人自己发现。
- **结论**：不把这个负面结果藏起来，而是正面写成"Bird's-eye VLN 问题定义（C1）"的论据之一——现有第一视角/低空无人机检测器不能直接套用到俯视灾害场景，反之亦然，这是本项目选择在 xBD 卫星正射域内训练/评测的合理性依据，而不是偷懒。后续如果要真正解决跨视角泛化，需要域适应（domain adaptation）或多视角联合训练，超出当前范围，留作后续工作。

**E15 — 双时相变化感知：差分注意力 vs 简单拼接（对应"用算法改动超越 baseline" baseline 调研）** ✅ 已完成（2026-07-25）

- 想搞清楚：`change_perception.py` 现有的融合方式是"直接拼接 `[f_pre,f_post,f_post-f_pre]`"，学术界的双时相变化检测 baseline 梯队（ChangeFormer/BIT/SNUNet/D2ANet）普遍认为"不加注意力的简单差分"不是最优融合方式。调研后选择 **D2ANet**（Difference-aware Attention Network，与本项目同用 xBD 建筑损伤数据）作为对标对象——同数据集、同任务，指标可直接比较，且原论文的 DTA（双时态聚合门控）+ DA（差分注意力）两个模块思路可以适配到本项目"编码器输出全局特征向量、不是卷积特征图"的架构（详见 `backend/change_perception.py` 的 `DifferenceAttention` 类注释）。
- 怎么做：新增 `DifferenceAttention` 模块（SE-style 通道门控，先对 `f_pre`/`f_post` 联合门控再相减，再对差分结果二次门控），`--diff-attention` 开关控制，其余训练超参（6 epoch、batch=64、lr=1e-3、xBD change train=25356）与 baseline 完全一致，且**固定同一个 seed=0** 消除初始化/数据顺序的随机性干扰，只让融合方式这一个变量不同。评测走 E10 同一套流程：`xbd_change/test.jsonl`（同源，n=21717）+ `xbd_change_full/holdout.jsonl`（跨 3 种未见灾害，n=69307）。
- **实测结果**：

| 模型 | val_acc(best) | test Acc | test ECE（标定后） | holdout Acc | holdout ECE（标定后） |
|---|---|---|---|---|---|
| baseline（简单拼接，seed=0） | 0.756 | 0.548 | 0.1129 | 0.521 | 0.1494 |
| **+ DifferenceAttention（D2ANet 思路，seed=0）** | 0.745 | 0.532 | 0.1382 | **0.590** | **0.0917** |

- **结论（一个有取舍但方向清晰的正面结果）**：差分注意力在**同源 val/test 上略输**（val_acc -1.1pp，test Acc -1.6pp）——两个门控层增加了参数量和优化难度，同域拟合能力略有下降；但在**跨灾害 holdout 上明显更好**：Acc **+6.9pp**（0.521→0.590），标定后 ECE **降 39%**（0.1494→0.0917，比 baseline 的标定收益更大）。这与 D2ANet 论文的设计动机一致：DTA+DA 两级门控学的是"哪些特征通道对捕捉变化本身敏感"，而不是"哪些通道对同源数据的纹理/光照分布敏感"，因此更不容易过拟合到训练时见过的灾害类型的表面特征，在真正没见过的灾害上保留更多可迁移的判别信号。**这是本轮调研里唯一一个"改算法真的在关键维度上超过 baseline"的结果**（E12_FBE 是负向对照，E14 是负向发现），且超过的维度（跨灾害泛化）恰好是 E14 揭示的当前系统最大短板，建议作为论文里"算法贡献"部分的正面证据，同时如实报告同源精度的小幅取舍，不只挑好看的数字。产物：`backend/outputs/change_perception/{baseline_seed0,diff_attention_seed0}.pt`、`runs/benchmarks/calibration_e15/`。
- **产物**：转换脚本 `scripts/training/gen_rescuenet_yolo_testset.py` / `gen_floodnet_yolo_testset.py`；评测脚本 `scripts/benchmarks/eval_floodnet_evidence.py`；结果 `runs/detect/rescuenet_zero_shot/`、`runs/benchmarks/e14_floodnet_evidence.json`；原始数据 `/home/lc/datasets/rescuenet/`、`/home/lc/datasets/floodnet/extracted/`（均已删除下载用的原始 zip，仅保留解压后的图片+标注）。

**E9 — 挑几条画出来看（定性案例）** ✅ `runs/benchmarks/e9_figures/`（脚本 `plot_vln_e9.py`）

- 想搞清楚：成功和失败的典型长什么样，方便写报告/论文配图。
- 怎么做：从结果里各挑 2~3 条成功 / 失败案例，把轨迹 + 语义地图 + 复核点画在瓦片底图上。
- 看什么：直观展示"读指令→拆地标→找目标→复核→到达"的全过程，以及失败时卡在哪。

**已生成（2026-06-24）**：

| 类别 | 文件 | 用途 |
|---|---|---|
| 汇总 14 张 | `01~14_*.png` | E1 消融 / E2 grounder / E8 难度 / E4 记忆 / E6 灾种 / E3 复核 / grounding 故事 / 管线示意 / 热力图 / 配对对照 / 散点 |
| 案例轨迹 | `cases/case_*.png` | E4 seed 成功、cold 对照、E1 成功/失败、easy 近失（叠 xBD 瓦片） |
| 索引 | `INDEX.md` | 汇报 slide 选用建议 |
| 副本 | `/home/lc/ppt_figures/vln_e9/` | 方便直接拖进 PPT |

一键重跑：
```bash
cd backend && set -a && source ../.env && set +a && \
  python ../scripts/benchmarks/plot_vln_e9.py \
    --e1-run ../runs/benchmarks/20260623_202108_e1full \
    --e4-run ../runs/benchmarks/20260624_222318_e4 \
    --out ../runs/benchmarks/e9_figures
# 只要汇总图：加 --summary-only
```

**怎么测（验收点）**
- 跑出**第一版成绩单**（E1 主表 + E2 grounding 对比），数值可复现（脚本一键出 `results.json`）。
- 鲁棒性（E5）在 GPS 噪声下成功率下降幅度**可量化**（σ-SR 曲线）。

**工作量拆分**：P4-1 题库生成 ✅ → P4-2 无头入口 ✅ → P4-3 评测脚本 ✅ → P4-4 E1 全量消融 ✅ →
P4-5 E2 grounding 对比 ✅ → **P4-5b E4 记忆二趟 ✅** → P4-6 鲁棒性扫描（E5~E7 待做）→ **P4-7 聚合出图 + 定性案例（E9 ✅ 汇总图完成，案例轨迹生成中）**。

**E4 记忆二趟（已完成）** `runs/benchmarks/20260624_222318_e4/`（脚本 `bench_e4_memory.py`）

设计：14 个瓦片各取 pass1（单目标）+ pass2（多目标，最终目标常相同）；在**同一 pass2** 上对比三种记忆状态（B3 配置 + hybrid）：
- **cold**：空记忆（对照）
- **warm**：先真实跑 pass1 积累（仅到达才沉淀）
- **seed**：用 pass2 GT 目标播种记忆（oracle，验证 prefly 机制上界）

| 模式 | n | SR | semSR | NE(m) | semNE(m) | Steps | 路径(m) |
|---|---|---|---|---|---|---|---|
| cold | 14 | 0.0 | 0.0 | 292 | 261 | 5.14 | 470 |
| warm | 14 | 0.0 | **0.143** | 335 | 256 | **4.14** | 529 |
| **seed** | 14 | **0.429** | **0.429** | **224** | **215** | **4.07** | 574 |

> **结论（H3 部分验证）**：
> - **seed（oracle）证明记忆机制本身有效**：相对 cold，SR 0→43%、semNE 261→215m（−18%）、Steps 5.1→4.1（−20%）；10/14 题 NE 更优。
> - **warm（端到端真实积累）收益有限但存在**：pass1 到达率极低，多数瓦片记忆稀疏；semSR 0→14%、Steps −19%，但 NE 受 outlier 拉高（pair12 NE=1305m）。
> - **瓶颈**：记忆沉淀依赖 pass1 **成功到达**（`run_vln_episode` 仅 arrived 才写盘）；当前 grounding 弱 → warm 几乎攒不满图 → E1 里 B3 也无收益。
> - **论文写法**：seed 作"机制上界"消融；warm 作"真实部署"预期；主张"记忆净赚"需先提升首趟到达率或允许部分成功沉淀。

**E2 grounder 三选一（已完成）** `runs/benchmarks/e2_grounder_report.md`：

| grounder | SR | semSR | NE(m) | semNE(m) | Steps | wall_s |
|---|---|---|---|---|---|---|
| yolo | 0.025 | 0.075 | 269 | 242 | 6.2 | 75 |
| vlm | 0.025 | 0.075 | 269 | 242 | 6.2 | 66 |
| **hybrid** | **0.075** | **0.125** | **226** | **193** | **4.8** | 62 |

> yolo 与 vlm 40/40 题轨迹完全相同（grounding 均弱→同一探索路径）；hybrid 在 13/40 题 semNE 优 >5m。**默认 grounder=hybrid**。

---

## P5 — 校准的双时相变化感知（C2 正式实现，第六节"升级接口"落地）

**背景**：第六节"升级接口"预留了两处升级点：① \(U_t\) 从启发式标量换成分布熵；② 复核触发从阈值判断换成信息增益动作选择。P5 就是把这两处从公式变成代码。**这不是新增的独立贡献，而是 C2 的正式实现方案**——C2 仍然是全文唯一的"刀尖"，P5 只是让它的数学定义有一个真正校准过的实现，而不是查表拍脑袋。

**目标**：让 \(U_t\) 真正是温度校准过的熵，让复核触发是信息增益驱动，而不是阈值判断。同时利用 xBD 天然的 pre/post 配对（本地已有完整数据，train 2799 对 + tier3 6369 对 + test 933 对，无需额外采购），让检测器输出的置信度更可信。

**改动**
- [ ] 新增 `scripts/training/gen_xbd_change_dataset.py`：读 xBD pre+post 配对标注，构造 `changed = post.subtype != 'no-damage'` 的二值变化标签 + 4 类损伤标签，输出配对训练集（图片仍软链，不复制）。
- [ ] 新增 `backend/change_perception.py`：pre+post 配对 patch 输入 → 共享编码器多任务头，输出 4 类损伤的 **softmax 概率向量**（而非当前 YOLO 的单一 top-1 conf）；训练后做温度标定（temperature scaling，用验证集拟合标量 \(T\)，`p_calibrated = softmax(logits / T)`）。
- [ ] 改 `backend/perception.py`：`_detect` 目前只暴露每个检测框的 `conf`（top-1 置信度），扩展检测结果结构，增加 `class_probs`（4 类概率分布）字段；通过 `VLN_CHANGE_PERCEPTION=1` 开关接入 `change_perception.py`，关闭时保留现有 YOLO-only 路径不受影响（新旧对照基线并存）。
- [ ] 改 `backend/recheck.py`：
  - `uncertainty_score` 增加熵模式：\(U_t=-\sum_i p_i\log p_i\)（归一化到 [0,1]），`p` 取自 `class_probs`；用 `VLN_UNCERTAINTY_MODE ∈ {heuristic, entropy}` 开关切换，`heuristic` 为现有查表公式（默认，保证向后兼容）。
  - `assess()` 的触发/收尾逻辑增加信息增益模式：定义复核动作候选集合 \(A=\{\text{descend\_center}, \text{hold}\}\)，用 GSD-置信度校准曲线估计每个动作执行后的期望熵 \(\mathbb E H(P_{t+1}^a)\)（简化确定性代理，非蒙特卡洛），取 \(a_t^\star=\arg\max_a[H(P_t)-\mathbb E H(P_{t+1}^a)]\)；用 `VLN_RECHECK_TRIGGER ∈ {threshold, info_gain}` 开关切换。
- [ ] 新增 `scripts/benchmarks/calibration_bench.py`：在 xBD test 集上计算 ECE / Brier Score / NLL，画 reliability diagram，验证温度标定确实降低了校准误差（不是自称"校准"）。

**怎么测（验收点）**
- [ ] 单元测试扩展 `backend/tests/test_recheck.py`：熵模式 `uncertainty_score` 在已知概率分布下数值正确；`heuristic`/`entropy` 两种模式在同一输入下都能跑通、互不干扰。
- [ ] `calibration_bench.py` 跑出的 ECE 相对未标定版本明显下降。
- [ ] `bench_vln_navigation.py` 在 `VLN_CHANGE_PERCEPTION=1` 下跑出的 ΔU 不再恒为 0（因为 `has_evidence` 判定基于更可信的概率分布，而不是稀疏的 top-1 conf）。

---

## P6 — 评测严谨性补丁

**目标**：修掉两个会被审稿人挑出来的漏洞——训练/测试的跨灾害泄漏、样本量不足以支撑消融结论。

**改动**
- [ ] `scripts/training/gen_xbd_yolo_dataset.py`：tier3 目前把全部 9 种独立灾害事件（joplin-tornado / lower-puna-volcano / moore-tornado / nepal-flooding / pinery-bushfire / portugal-wildfire / sunda-tsunami / tuscaloosa-tornado / woolsey-fire）都并入训练池，与 train/test 共享的 10 种灾害事件不同，天然是一个未被利用的跨灾害留出集。改为默认排出 3~4 种（如 nepal-flooding / moore-tornado / pinery-bushfire）不参与任何训练，作为 `--holdout-disasters` 参数控制的 unseen-disaster 测试集。
- [ ] `scripts/benchmarks/gen_vln_testset.py`：题库从 40 题扩到 200~500 题（自动生成，复用现有生成逻辑，不需要新标注）。
- [ ] `scripts/benchmarks/bench_report.py`：增加 bootstrap 95% CI 与配对检验（paired test），用于消融对比的显著性判断。

**怎么测（验收点）**
- [ ] 用新脚本重新生成的检测器训练/测试划分下，unseen-disaster 子集单独跑出一版 mAP，与原 test（同源灾害）对比，验证"域内检测器"的跨灾害泛化程度是否被之前的划分方式高估。
- [ ] `bench_report.py` 输出的消融表带 CI 区间。

---

## 相关工作定位（C1，写作素材，无代码改动）

DisasterClaw 的问题定义（C1，Bird's-eye VLN under uncertain disaster observations）需要在论文里明确同几类邻近工作的差异，避免被误读为"缝合已有模块":

- **xBD/xView2**（Gupta et al., CVPRW 2019）：定义了 pre/post 建筑损伤评估任务与 Joint Damage Scale，但是纯静态图像分类，不涉及"谁去拍这张图"。
- **Change-Agent**（Liu et al., TGRS 2024）与 **ISPRS 2026 多任务变化论文**（Wang et al.）：把变化理解做成了 agent（检测+描述+计数+归因），但处理的是已经摆在桌面上的两张图，agent 不会自己决定飞近再看。
- **BayeSiamMTL**（JAG 2025）：把校准不确定性带进了 xBD 数据集本身，产出置信度分层的损伤图，但停留在单张图的后处理，不驱动任何导航动作。
- **AeroVerse**（TPAMI 2026）：定义了完整的 UAV agent 能力体系（感知/推理/导航/规划/决策），但场景是第一视角城市，没有灾害语义、没有双时相变化。
- **ESARBench**（2026）：第一个具身搜救基准，但基于 UE5/AirSim 仿真环境与 GIS 建图，不是真实卫星影像；其最优基线 SR 仅 13.89%，可作为"这类任务本身就难"的外部佐证，避免本文低 SR 被误读为系统缺陷。
- **AirNav / OpenFly / HUGE-Bench**（2025~2026，大规模航拍 VLN）：规模远超本项目（10 万级轨迹），但场景是通用城市导航/巡检，不涉及灾害语义、不涉及双时相变化检测，说明"通用航拍 VLN 数据集"覆盖不了本项目的任务需求，不是简单缝合已有 benchmark 就能得到。
- **DisasterBench**（2026）：UAV 灾害多模态推理 benchmark，14 种灾害场景类型、9 类响应任务，但是多选 VQA 形式（问答），不驱动导航动作——停留在"看图回答"层面，没有"要不要飞近再看一眼"这个决策环节。
- **BRIGHT**（Chen et al., *ESSD* 2025，IEEE GRSS DFC 2025 Track II 官方数据集）：目前最新、最权威的多灾害多模态（光学+SAR）建筑损伤评测集，14 个全球灾害事件，专门设计了跨事件迁移评测协议。调研后判定其"灾前光学+灾后 SAR"的配对方式与本项目"光学-光学双时相"不兼容，SAR 图像不能直接喂给现有 RGB 检测器/变化感知模型，故未接入定量实验（见 E14 说明），但其"跨事件迁移"协议设计思路值得本项目后续扩展跨传感器泛化时参考。
- **Uncertainty-Informed Active Perception for Open-Vocabulary ObjectNav**（2025）：**方法论上与本项目 C2 最接近的工作**，同样是"感知不确定性 → 主动感知动作选择"的闭环（用 VLM 语义相似度的不确定性驱动 frontier exploration）。**必须在 related work 中正面区分**，核心差异两点：① 场景与几何约束不同——本项目的不确定性降低有明确的物理机制（降高→GSD 变细→证据置信提升，第 0 节 \(R_t,g_t\) 的单调关系），室内 ObjectNav 没有这种"分辨率随动作确定性变化"的耦合；② 不确定性来源不同——本项目用经过温度校准的双时相变化检测概率分布（第 2 节 \(U_t\)），而非 VLM 开放词汇语义相似度，前者有 E10 的 ECE/Brier/NLL 校准质量验证，后者停留在启发式相似度分数。

DisasterClaw 用**真实地理配准的双时相卫星影像**替代仿真环境，是差异化卖点而非弱点，应在 related work 中明确写出这条对比。

**理论脉络补充**（不确定性驱动主动感知的数学传统）：本项目 P5"升级接口"里的信息增益触发算子（\(a_t^\star=\arg\max_a[H(P_t)-\mathbb E H(P_{t+1}^a)]\)）并非凭空设计，而是 **Bajcsy（1988，*Active Perception*，主动感知奠基性工作）** 定义的感知-动作闭环，在 **POMDP-IR / AP²-POMDP**（信息增益驱动的感知动作选择框架，IJCAI 2019 等）框架下针对"俯视灾害场景"的一个确定性代理实现——写作时补一句理论溯源，能让 P5 的公式化显得有学术脉络支撑，而不是像临时发明的启发式。

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



## 我的期望
我认为不要试图同时强调 HSPM、STMR、记忆图、Grounding 等所有模块，否则创新点会被稀释。比较合理的核心创新可以浓缩为下面三个：

### ① 面向灾害响应的 Bird's-eye VLN 框架（主要问题定义）

提出一种基于卫星影像和 UAV 俯视观测的 Vision-Language Navigation 框架，将传统第一视角 VLN 扩展到灾害场景中的空中智能体。

一句话：

> 首次将 VLN 从室内第一视角扩展到灾害俯视视角。

---

### ② 不确定性感知驱动的主动复核机制（主要算法贡献）

当视觉模型对灾害目标置信度不足时，智能体主动调整观察位置（例如降低高度、重新采样），通过多次观测提高判断可靠性。

一句话：

> 将视觉不确定性直接转化为导航行为，实现 uncertainty-aware active reinspection。

这是我认为你最强、最值得重点突出的一点。

---

### ③ 基于语义地图的长程层次规划（支撑模块）

利用文本化语义地图（STMR）和层次规划（HSPM），实现大尺度灾害区域中的长程导航。

这里建议写成：

> We adopt a semantic spatial memory representation and hierarchical planning strategy.

不要强调这是全新的。

---

如果只允许写三条 contribution，我会写：

> • We formulate Bird's-eye Vision-Language Navigation for aerial disaster response.
>
> • We propose an uncertainty-aware active reinspection mechanism that couples perception confidence with navigation decisions.
>
> • We build a semantic map based long-horizon navigation framework over xBD disaster imagery.

其中：

* **C1 = 新问题；**
* **C2 = 真正的创新；**
* **C3 = 工程系统支撑。**

我个人认为：

### 真正能支撑 RA-L / IROS 的“刀尖”只有一个：

> **Uncertainty-aware Active Reinspection for Bird's-eye VLN.**

其它模块都应该服务于它，而不是与它竞争贡献点。这样整篇论文的故事会非常清晰。


能，而且我觉得比你自己想象的更有学术性。

但前提是：**不要把创新点写成“工程模块”**，而要把它们提升成一个明确的数学问题。

这是很多机器人论文的关键。

---

# 一、现在的三个 contribution 学术性够不够？

我的评价：

### C1 Bird's-eye VLN（★★★★☆）

属于 Problem Formulation。

这种贡献本身不需要复杂公式。

类似：

* R2R 定义 Indoor VLN；
* VLN-CE 定义 Continuous VLN；
* RxR 定义多语言 VLN；

你的：

> Aerial Bird's-eye VLN。

这一条主要靠：

* 新场景；
* 新 benchmark；
* 新任务定义；

支撑。

---

### C2 Uncertainty-aware Reinspection（★★★★★）

这是最强的一条。

因为：

### 可以数学化。

这是 reviewer 最喜欢的。

---

### C3 Semantic Memory + HSPM（★★★☆☆）

建议不要硬说创新。

写：

> We adopt a semantic spatial memory representation.

即可。

否则 reviewer 很容易：

> “这不是别人做过了吗？”

---

# 二、你的真正创新其实可以重新定义

我甚至觉得：

不要叫：

### uncertainty-aware reinspection

而应该叫：

## Confidence-guided Active Observation

或者：

## Uncertainty-aware Information-seeking Navigation

因为：

从数学上看：

你其实是在：

### perception uncertainty → action selection

这是一个非常标准的：

### Active Perception

问题。

---

# 三、完全可以写公式

### 1. Bird's-eye VLN 定义

定义：

状态：

```math
s_t = (M_t, p_t, I_t)
```

其中：

* (M_t)：当前语义地图；
* (p_t)：UAV位置；
* (I_t)：当前观测；

语言指令：

```math
L
```

策略：

```math
\pi(a_t|s_t,L)
```

目标：

```math
\max_\pi P(goal|L)
```

这一部分已经符合标准 VLN 写法。

---

# 2. 不确定性定义

假设 VLM 输出：

```math
P(y|I_t)
```

例如：

```text
destroyed : 0.55
major : 0.30
minor : 0.15
```

则：

### Shannon Entropy：

```math
U_t = -\sum_i p_i\log p_i
```

或者：

最大概率：

```math
U_t = 1-\max_i P(y_i|I_t)
```

这两个都可以。

---

# 3. 复核触发

定义：

```math
a_t^{reinspect}
=
\begin{cases}
1,& U_t>\tau\\
0,& otherwise
\end{cases}
```

其中：

(\tau) 为阈值。

这是最简单版本。

---

# 4. 更高级版本（我更推荐）

不要：

```math
if uncertainty > threshold
```

而是：

### 选择最大信息增益动作

定义：

动作集合：

```math
A=\{down,left,right,forward\}
```

选择：

```math
a_t^*
=
\arg\max_a IG(a)
```

其中：

```math
IG(a)
=
H(P(y|I_t))
-
H(P(y|I_{t+1}^a))
```

也就是：

> 当前熵 − 执行动作后的熵。

这个瞬间就从：

### 工程 heuristic

变成：

### Information-theoretic Active Perception。

学术味会暴涨。

---

# 5. 长程规划目标

你的 HSPM：

实际上可以：

### coarse → fine

定义：

高层：

```math
g_k=\arg\max_i Score(tile_i,L)
```

低层：

```math
a_t=\pi(s_t,g_k)
```

整个：

```math
\pi
=
\pi_{local}
\circ
\pi_{global}
```

这就是：

Hierarchical Policy。

---

# 四、我觉得还能再提升一层

其实你现在非常接近：

### POMDP

定义：

隐藏状态：

```math
x_t
```

观测：

```math
o_t
```

belief：

```math
b_t=P(x_t|o_{1:t})
```

更新：

```math
b_{t+1}
=
\eta P(o_{t+1}|x_{t+1})
\sum_x P(x_{t+1}|x_t)b_t
```

行动：

```math
a_t
=
\arg\max_a
E[R(b_t,a)]
```

然后：

你的复核：

其实就是：

### belief refinement。

这一套放出来，RA-L reviewer 会非常熟悉。

---

# 五、我认为最漂亮的最终故事应该是：

### Problem

Bird's-eye VLN under uncertain disaster observations.

↓

### Method

Hierarchical Semantic Navigation Framework.

↓

### Core Innovation

Confidence-guided Active Observation：

```math
U_t
\rightarrow
a_t^{reinspect}
```

↓

### Goal

Reduce belief uncertainty.

---

# 我甚至觉得题目都可以直接围绕这一点：

### Uncertainty-aware Bird's-eye Vision-Language Navigation for Disaster Response

或者：

### Active Bird's-eye VLN via Confidence-guided Reinspection

或者：

### Information-seeking Vision-Language Navigation for Aerial Disaster Assessment

这几个题目已经是标准 IROS / RA-L 风格了。

说实话，我觉得你的工作现在已经不是“有没有学术性”的问题。

而是：

> **怎样把一个工程系统重新表述成一个数学问题。**

这是从「做项目」进入「写论文」的关键一步。

---

# 形式化（与当前代码严格一致版 · 可直接进 Method）

> 说明：本节**不是上面的理想公式**，而是把现在 `recheck.py / perception.py / vln_navigator.py / hspm_planner.py`
> **正在做的事**如实写成数学符号。每个量都对得上代码，便于直接写进论文方法节；待要升级到熵 / 信息增益 / POMDP
> 时，只需替换其中 \(U_t\) 与触发算子两处（见末尾"升级接口"）。

## 0. 观测几何（perception）

无人机高度 \(h_t\)。俯视 patch 的地面半径与地面采样距离（GSD）：

```math
R_t = \mathrm{clip}\big(R_{\min},\; \alpha\, h_t,\; R_{\max}\big),\qquad
g_t = \frac{2R_t}{W}
```

其中 \(\alpha=2,\ R_{\min}=20\text{m},\ R_{\max}=300\text{m}\)，\(W\) 为 patch 像素宽。
**关键单调性**：\(h_t\!\downarrow\,\Rightarrow R_t\!\downarrow\,\Rightarrow g_t\!\downarrow\)（分辨率变细）——这是"降高复核能看得更清"的几何依据。

patch 内归一化坐标 \((n_x,n_y)\in[0,1]^2\)（中心 0.5）映射到机体相对位移（北/东，米）：

```math
e = (n_x-0.5)\cdot 2R_t,\qquad n = -(n_y-0.5)\cdot 2R_t
```

## 1. Bird's-eye VLN（问题定义，C1）

状态 \(s_t=(M_t,\,p_t,\,I_t)\)：语义地图 \(M_t\)、位姿 \(p_t=(\mathrm{lat},\mathrm{lon},h_t)\)、当前观测 \(I_t\)（patch、检测集 \(\mathcal D_t\)、分割、风险等级 \(\rho_t\)）。
指令 \(L\) 解析为目标类别集合 \(C\)、开放词汇短语、方向先验。策略 \(\pi(a_t\mid s_t,L)\)，目标 \(\max_\pi P(\text{goal}\mid L)\)。

**Grounding 算子**（YOLO 后端，`ground_with_yolo`）：在检测集中按下式打分并取最优框，输出其归一化中心：

```math
d_t^\star=\arg\max_{d\in\mathcal D_t,\ \mathrm{cls}(d)\in C}
\Big[\underbrace{\tfrac{\mathrm{area}(d)}{A}}_{\text{显著}}
+0.3\,\mathrm{conf}(d)
-0.2\,\big\lVert c(d)-c_0\big\rVert\Big]
\quad\text{s.t. } \tfrac{\mathrm{area}(d)}{A}\ge 8\times10^{-4}
```

\(A\) 为 patch 面积，\(c(d)\) 为框中心、\(c_0\) 为 patch 中心。命中后经第 0 节映射为相对位移，按 \(\lVert\cdot\rVert\le r_{\text{arr}}\) 判定到达（hybrid：YOLO 空则回退 VLM）。

## 2. 不确定性度量（C2，当前实现）

受灾证据类别 \(\mathcal E=\{\text{minor, major, destroyed, water}\}\)。定义：

```math
c_t=\max_{d\in\mathcal D_t,\ \mathrm{cls}(d)\in\mathcal E}\mathrm{conf}(d),\qquad
b_t=\mathbb 1\!\left[\exists\,d:\mathrm{cls}(d)\in\mathcal E\right]
```

```math
e_t=b_t\ \lor\ \mathbb 1[\rho_t\neq\text{none}],\qquad
\hat c_t=\begin{cases}c_t,&b_t=1\\ 0.3,&b_t=0\end{cases}
```

风险暧昧度查表 \(\varrho(\rho)\)：\(\text{none}\!\to\!0.2,\ \text{low}\!\to\!0.9,\ \text{moderate}\!\to\!0.6,\ \text{high}\!\to\!0.15\)。
**不确定性**（证据暧昧度与低置信度各半）：

```math
U_t=\begin{cases}
0,& e_t=0\\[2pt]
\tfrac12\,\varrho(\rho_t)+\tfrac12\big(1-\mathrm{clip}(\hat c_t,0,1)\big),& e_t=1
\end{cases}
```

## 3. 主动复核：触发—机动—定论（当前实现）

按位置量化分桶 \(\kappa(p)=\big(\lfloor \mathrm{lat}/\Delta\rceil,\lfloor \mathrm{lon}/\Delta\rceil\big)\)（\(\Delta\) 由 `cell_m` 定），每桶维护已复核次数 \(n_\kappa\) 与首次不确定性 \(U^{0}_\kappa\)。给定阈值 \(\tau\)、预算 \(K\)、高度下限 \(h_{\min}\)、降幅 \(\delta\)、居中限幅 \(d_{\max}\)：

**(a) 把握足够或无证据**（\(e_t=0\) 或 \(U_t<\tau\)）：

```math
\text{若 }\kappa\text{ 在复核中}\Rightarrow \textbf{resolve},\quad
\text{status}=\begin{cases}\text{confirmed},&\rho_t\in\{\text{high,moderate}\}\\ \text{dismissed},&\text{otherwise}\end{cases}
```

否则 **skip**。

**(b) 可疑且没把握**（\(e_t=1\land U_t\ge\tau\)）：

```math
n_\kappa\ge K\ \lor\ h_t\le h_{\min}\ \Rightarrow\ \textbf{resolve},\quad
\text{status}=\begin{cases}\text{confirmed},&\rho_t=\text{high}\ \lor\ c_t\ge c_{\text{thr}}\\ \text{inconclusive},&\text{otherwise}\end{cases}
```

否则发出**复核机动** \(a_t\)（降高 + 朝证据框居中），\(n_\kappa\!\mathrel{+}=\!1\)：

```math
\Delta h=-\min\big(\delta,\ h_t-h_{\min}\big),\qquad
(\Delta n,\Delta e)=\Pi_{d_{\max}}\!\big(\text{offset}(c(\text{bbox}),R_t)\big)
```

\(\Pi_{d_{\max}}\) 为模长不超过 \(d_{\max}\) 的截断（一步不飞太远，多步逼近）。

**机制（为何 \(U\) 会降）**：\(\Delta h<0\Rightarrow R_{t+1}\le R_t\Rightarrow g_{t+1}\le g_t\)（更细）\(\Rightarrow\) 证据置信 \(\hat c\!\uparrow\Rightarrow U\!\downarrow\)；居中又减小边缘畸变。

## 4. 评测量（已落地）

每个被定论候选记录**不确定性下降量**，并在成绩单聚合平均：

```math
\Delta U_\kappa=U^{0}_\kappa-U^{\text{final}}_\kappa,\qquad
\overline{\Delta U}=\frac{1}{|\mathcal R|}\sum_{\kappa\in\mathcal R}\Delta U_\kappa
```

\(\mathcal R\) 为已定论候选集合（`RecheckController.stats` 的 `avg_uncertainty_reduction`）。这正是"复核降低 belief 不确定性"目标的经验实现。

## 5. 长程层次规划（C3，coarse→fine）

```math
g_k=\arg\max_i \mathrm{Score}(\text{landmark/tile}_i,\,L)\ \ (\text{global}),\qquad
a_t=\pi_{\text{local}}(s_t,g_k),\qquad
\pi=\pi_{\text{local}}\circ\pi_{\text{global}}
```

## 6. 升级接口（如要更"信息论"，只动两处）

- \(U_t\)：把第 2 节的启发式标量替换为分布熵 \(U_t=-\sum_i p_i\log p_i\)，其中 \(p=\mathrm{softmax}\) 检测器 4 类损伤打分（需温度校准）。其余符号与流程不变。
- 触发算子：把第 3 节阈值 \(U_t\ge\tau\) 替换为信息增益动作选择 \(a_t^\star=\arg\max_a\big[H(P_t)-\mathbb E\,H(P_{t+1}^a)\big]\)。
