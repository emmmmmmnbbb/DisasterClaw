# UAV 自主导航 Baseline 调研

面向 DisasterClaw 灾害遥感智能体的"主动式具身导航"（论文创新点 C2）前置调研。
目标：梳理 UAV 自主导航/探索的主要方法族与代表工作，给出可在 DisasterClaw
现有 2D 地理瓦片仿真中落地的 baseline 短名单与推荐起点。

---

## 1. 现状：DisasterClaw 当前的"导航"是什么

通过阅读 `backend/world.py` 与 `backend/ai_planner.py`，当前导航能力可概括为
**"LLM 直接吐航点"**，并非自主探索：

- 世界模型 `WorldModel`：维护锚点 anchor、激活的 xBD 瓦片 `active_tile`、
  `latlon_bounds`、单架 `UAV_1` 的位姿（lat/lon/alt + NED 偏移）、targets、reports。
- 规划器 `TaskPlanner.plan()`：调用 LLM 生成 JSON 动作序列，动作空间仅 6 个：
  `fly_to_geo` / `fly_relative` / `hover` / `detect_disaster` / `mark_target` /
  `report_observation`；LLM 不可用时回退到关键词规则规划 `_fallback_plan`。
- 没有：占据/不确定性地图、frontier 检测、视点候选生成、信息增益评估、
  覆盖率统计、自主"下一步飞哪"的决策闭环。

因此"自主导航"在本项目语境下，本质是要在这套世界模型 + 动作集之上，
增加一个**根据态势自己决定下一观测点**的决策层。

---

## 2. 方法族梳理（taxonomy + 代表工作）

### 2.1 覆盖路径规划（Coverage Path Planning, CPP）
目标：用规则化路径扫遍整个已知区域，不依赖在线决策。
- Boustrophedon / lawnmower（牛耕式往返）：最经典的全覆盖基线。
- 生成树覆盖 STC（Gabriely & Rimon, 2001）。
- 综述：Choset (2001)；Galceran & Carreras, *A survey on coverage path planning
  for robotics*, RAS 2013。
- 特点：实现极简、可复现、可作"上限覆盖/下限智能"的对照；缺点是与目标无关、
  无重点、效率低（在大灾区做无差别扫描代价高）。

### 2.2 前沿探索（Frontier-based Exploration）
目标：在"已知/未知边界"（frontier）中选下一目标，逐步探索未知区域。
- 开山之作：Yamauchi, *A frontier-based approach for autonomous exploration*,
  CIRA 1997。
- 代价函数化：把路径长度、传感范围、信息增益组合为 frontier cost，转化为
  最小比率 TSP 求解（见 PMC11985980, 2025）。
- 学习式前沿：FrontierNet（2025）从 2D 视觉线索预测 frontier 及其信息增益，
  减少对完整 3D 地图操作的依赖。
- 特点：比随机采样更高效、2D 场景友好；需要可维护的"已探索/未探索"地图。

### 2.3 采样式探索（Sampling-based）
- RRT/RRT* 驱动的探索（随机扩展树选目标）。
- 特点：实现简单，但目标选择有随机性，效率不稳定。

### 2.4 下一最佳视点 / 信息论主动感知（NBV / Active Perception）
目标：迭代选择"信息量最大"的下一视点，是与本项目最契合的范式。
- 思想源头：Connolly, *The determination of next best views*, ICRA 1985；
  Bajcsy, *Active perception*, 1988。
- 经典在线规划：Bircher et al., *Receding Horizon NBV planner (NBVP)*, ICRA 2016。
- 信息增益度量：熵 / 互信息 / 占据栅格不确定性（entropy-based vs frontier-based，
  见 arXiv:2511.20353 对两类的对比）。
- 近期代表：
  - 多 UAV 融合 frontier + NBV，Drones 2024, 8(11): 630。
  - VIN-NBV（arXiv:2505.06219）：用 View Introspection Network 预测某视点的
    "相对重建提升"，贪心选点，胜过基于覆盖的 RL 策略。
  - PB-NBV（arXiv:2501.10663）：投影代替 ray-casting，大幅降算力。
  - 质量引导 NBV（arXiv:2511.20353）：以重建质量目标驱动视点生成与选择。
- 特点：天然支持"哪里不确定就去哪看"，可由感知置信度驱动；多数面向 3D 重建，
  需迁移到"灾情判读不确定性"语义。

### 2.5 强化学习导航/探索（RL-based）
- 端到端 RL 学习探索/导航策略（覆盖率或信息增益作奖励）。
- 特点：潜力大但训练成本高、样本效率低、sim-to-real 难；VIN-NBV 等工作指出
  其常被更简单的贪心 NBV 策略超过。

### 2.6 空中视觉语言导航（Aerial VLN, AVLN）
目标：UAV 理解自然语言指令 + 视觉观测，在连续空间自主导航到目标。
- 数据/基准：
  - AerialVLN（Liu et al., ICCV 2023）：25 个城市场景 >8400 条人类轨迹。
  - CityNav（Lee et al., ICCV 2025, arXiv:2406.14240）：真实城市三维扫描，
    32k 指令/轨迹，含地理语义地图与洪水灾害鲁棒性测试；baseline 为
    Seq2Seq / CMA / AerialVLN。
  - OpenFly（Gao et al., 2025）、OpenUAV、AVDN（对话式）、HaL-13k 等。
- 代表方法：CityNavAgent（Tsinghua, 2025）——开放词汇感知 + LLM 分层语义规划
  + 拓扑图全局记忆，零样本城市 AVLN。
- 综述：AeroVerse-Review, *Comprehensive survey on aerial embodied VLN*,
  The Innovation 2025。
- 特点：与"语言可查询感知 + LLM 规划"的主线高度契合；但多为城市导航/找物，
  需迁移到灾后巡查 + 不确定性复核。

### 2.7 LLM/VLM 驱动的无人机智能体
- AerialClaw（XDEI-Group, 2026，论文[46]）：DisasterClaw 的直接基座，通用自主
  UAV 的个性化 AI Agent。
- RescueADI（Liu et al., TGRS 2025，论文[33]）：遥感自主灾害判读 agent。
- 特点：用 LLM 做高层规划/工具调度，导航多为"吐航点"，缺少信息增益闭环——
  这正是本项目可补的空白。

---

## 3. 与本项目相关的评测指标

- 任务级：成功率 SR、SPL（成功且路径效率）、到达可信结论所需步数。
- 探索级：区域覆盖率 %、探索时间、路径长度、累计信息增益。
- 感知-导航耦合：单位飞行代价下的"目标发现/复核数"、不确定性下降速率。
- 工程级（呼应论文 Bench-A/C）：规划延迟、长程连续任务成功率与失败模式。

---

## 4. 可在 DisasterClaw 落地的 Baseline 短名单

落地约束：单架 UAV、2D 地理瓦片、动作集为 `fly_to_geo/fly_relative/hover/
detect_disaster/mark_target/report_observation`，感知由 `detect_disaster`
（YOLO+SegFormer，未来可换 LocateAnything）提供。

### B1. Lawnmower 覆盖（最简对照基线，必做）
- 原理：在 `latlon_bounds` 内按固定行距生成牛耕式 `fly_to_geo` 航点序列，
  每点 `detect_disaster`，扫完输出报告。
- 落地：纯几何，新增一个 `coverage_planner` 生成航点；几乎不改 `world.py`。
- 成本：低（1 个脚本/函数）。
- 优劣：可复现、与目标无关、效率低；用作"覆盖上限 / 智能下限"的对照。

### B2. 贪心 Frontier / 不确定性驱动（中间对照）
- 原理：在瓦片上维护一张粗粒度"已观测/不确定"栅格；每步选取信息价值最高的
  frontier 单元飞过去观测，更新栅格，直到覆盖或预算耗尽。
- 落地：在 `WorldModel` 增加观测栅格状态；新增 frontier 选择器；复用
  `fly_to_geo + detect_disaster`。
- 成本：中（栅格状态 + 选择器 + 终止条件）。
- 优劣：比覆盖更高效、实现仍可控；信息价值此处偏几何，未用语义置信度。

### B3. 信息增益 NBV（LocateAnything 置信度驱动）——推荐主线，呼应 C2
- 原理：把 Track A 的开放词汇 grounding 置信度/损伤等级不确定性作为信息增益，
  候选视点价值 = 语义不确定性下降期望 + 覆盖增益 − 飞行代价；贪心选 next-best-view，
  对低置信目标主动飞近复核。形成"感知不确定性 → 主动探索"的闭环。
- 落地：B2 的栅格框架 + 用 grounding 置信度替换几何信息增益；LLM 负责高层语义
  目标（"复核疑似倒塌建筑"），轻量 NBV 控制器负责选点执行（亦可缓解论文
  Bench-A 揭示的 L3 规划延迟瓶颈）。
- 成本：中高（依赖 Track A 产出 + NBV 评分函数 + 与规划器解耦）。
- 优劣：与论文主线"主动式开放词汇具身智能体"完全一致，新颖性最强；
  需先完成 Track A 的感知置信度产出。

### 推荐路线
以 **B1 作对照基线**、**B3 作创新主线**，**B2 作中间消融档**，三者共用同一
`fly_to_geo + detect_disaster` 闭环与同一套指标（覆盖率/步数/不确定性下降/成功率），
即可在仿真内做"覆盖 vs 前沿 vs 信息增益"三档对照，直接支撑 C2 的实验设计。

---

## 5. 可引用清单（精选）

- Yamauchi B. A Frontier-Based Approach for Autonomous Exploration. IEEE CIRA, 1997.
- Connolly C. The Determination of Next Best Views. IEEE ICRA, 1985.
- Bajcsy R. Active Perception. Proceedings of the IEEE, 1988.
- Gabriely Y., Rimon E. Spanning-tree based coverage of continuous areas by a
  mobile robot. Annals of Math. and AI, 2001.
- Galceran E., Carreras M. A survey on coverage path planning for robotics. RAS, 2013.
- Bircher A., et al. Receding Horizon "Next-Best-View" Planner for 3D Exploration.
  IEEE ICRA, 2016.
- Distributed Multi-UAV Frontier + NBV Exploration. Drones, 2024, 8(11): 630.
- VIN-NBV: A View Introspection Network for Next-Best-View Selection. arXiv:2505.06219, 2025.
- PB-NBV: Efficient Projection-Based Next-Best-View Planning. arXiv:2501.10663, 2025.
- Quality-guided UAV Surface Exploration for 3D Reconstruction. arXiv:2511.20353, 2025.
- Liu S., et al. AerialVLN: Vision-and-Language Navigation for UAVs. ICCV, 2023.
- Lee J., et al. CityNav: Language-Goal Aerial Navigation Dataset with Geographic
  Information. ICCV 2025, arXiv:2406.14240.
- CityNavAgent: Aerial VLN with Hierarchical Semantic Planning and Global Memory. 2025.
- AeroVerse-Review: Comprehensive Survey on Aerial Embodied VLN. The Innovation, 2025.
- Liu Z., et al. RescueADI: Adaptive Disaster Interpretation in Remote Sensing Images
  with Autonomous Agents. IEEE TGRS, 2025.
- XDEI-Group. AerialClaw: Personalized AI Agent for General Autonomous UAV Systems. 2026.

> 说明：部分 arXiv 编号以调研当时检索为准，正式引用前需逐条复核年份/卷期/页码。
