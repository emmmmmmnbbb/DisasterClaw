# paper_cja 智能体系统 + Agent-VQA 完整修改计划

> 文档状态：修订提案，待作者按阶段确认后执行。  
> 当前作者已确认：论文以智能体系统为主题，并引入 VQA 实验。  
> 默认范围：以 xBD 可验证的封闭式灾情 VQA 为主，不开展正式人机实验；自由文本报告仅作案例展示。  
> 目标稿件：`paper_cja`。  
> 关联账本：`REVISION_EXPERIMENTS.md` 继续记录原审稿意见对应的补实验；本文档负责整篇论文转向 Agent-VQA 主线后的完整修改。  
> 审稿标准说明：当前没有作者确认的正式期刊审稿标准绑定，故按现有审稿意见与通用航空、机器人和遥感论文规范规划，不声称已严格符合某一具体期刊标准（`criteria_binding_unavailable`）。

---

## 1. 总体修改目标

### 1.1 新的论文主线

将论文从“校准不确定性驱动的主动重观测方法”改为：

> 面向航空应急侦察的可审计语言条件智能体。智能体接收操作员的自然语言问题或任务，在地理配准的俯视环境中搜索目标、收集证据、回答灾情问题，并在证据不足时以有限预算执行继续搜索、目标居中或下降重观测。

新的完整闭环为：

```text
操作员问题或任务
→ 问题解析与子目标分解
→ 当前视场感知与目标指认
→ 语义地图和历史状态更新
→ 生成当前答案、置信度与证据定位
→ 判断证据是否充分
→ 回答 / 继续搜索 / 居中下降 / 弃答
→ 新观测与最终答案
→ 结构化决策轨迹和任务报告
```

### 1.2 建议题名

中文首选：

> 面向航空应急侦察的可审计语言条件智能体：灾情视觉问答与预算受限主动重观测

英文首选：

> An Auditable Language-Conditioned Agent for Aerial Emergency Reconnaissance: Disaster VQA and Budget-Constrained Re-observation

备选中文题名：

> 面向航空灾害侦察的多模态智能体：目标搜索、灾情问答与主动重观测

### 1.3 论文的核心科学问题

1. **RQ1：单次观测问答能力。** 给定当前俯视观测，智能体能否正确回答灾情存在、损伤等级、数量和空间位置问题？
2. **RQ2：结构化证据价值。** 检测结果、双时相损伤概率、STMR 语义矩阵和历史观测是否能改善 VQA 的正确性、定位和校准？
3. **RQ3：主动问答价值。** 当第一次观测不足时，智能体能否在预算约束下通过搜索或重观测改善最终答案？
4. **RQ4：系统瓶颈归因。** 端到端失败主要来自目标指认、损伤识别、规划、动作执行还是回答生成？
5. **RQ5：退化与恢复。** 当 LLM/VLM 不可用、目标越界、指认错误或证据缺失时，规则回退和恢复机制能否维持合法执行并给出可审计结果？

### 1.4 贡献层级

论文贡献按以下顺序组织，避免把所有模块并列包装成创新：

1. **智能体系统贡献。** 给出一个真实实现的航空灾害语言智能体，将任务规划、目标搜索、视觉问答、重观测、报告和前端监控闭合在有限动作集内。
2. **Agent-VQA 任务与评测贡献。** 将灾害 VQA 从“给定图片后回答”扩展为“智能体决定去哪里看、是否再看以及何时回答”。
3. **主动重观测机制贡献。** 使用当前观测的不确定性决定是否执行下降和居中，但仅在实验支持时声称有效。
4. **可识别性优先评估贡献。** 在报告策略收益前先检查答案翻转、信息上限、事件方差和 oracle 间隙，区分机制无效、机制不可测和上游感知失败。
5. **分层瓶颈诊断贡献。** 使用目标指认、初始定位、答案和动作层的 oracle 参照定位系统瓶颈，不把 oracle 结果误写成可部署性能。

---

## 2. 修改边界与默认决策

### 2.1 纳入正文的内容

- 自然语言任务和问题输入。
- 有限动作与技能规划。
- HSPM 地标层、目标层和运动层。
- STMR 文本语义矩阵与地理语义地图。
- 混合目标指认。
- 灾情 VQA 与结构化回答。
- 证据不足检测与预算受限重观测。
- 结构化决策轨迹、规则回退和最终报告。
- 无头 Agent-VQA 基准与前后端演示链路。
- 事件隔离、配对统计和 oracle 瓶颈诊断。

### 2.2 降为附录或补充材料的内容

- RescueNet 单时相复制实验。
- 历史三处实现缺陷的完整排查日志。
- 大段内部调试输出。
- 记忆图的详细实现，除非新实验得到可解释的重复任务收益。
- 全部 19 折 LOEO 明细表。
- 全部 VQA 模板、审查表和逐题输出。
- 完整前端截图集；正文只保留系统全貌图与单条 Agent-VQA 轨迹图。

### 2.3 暂不纳入的内容

- 真实六自由度飞行控制、风扰、避障、电池模型。
- 多无人机协同。
- 第一视角三维 VLN。
- 无真值支持的道路可通行性、人员被困、伤亡数量等问答。
- 正式人机交互用户研究。
- 对真实救援部署有效性、安全性或认证状态的声明。
- “首次提出”类新颖性主张，除非完成独立、可核验的文献检索。

### 2.4 默认作者决策

除非作者另行修改，执行时采用以下默认选择：

- 主文 VQA 使用封闭式、可自动评分问题。
- 主数据源为 xBD 双时相与建筑多边形标注。
- 当前 200 个 evidence-rich 场景作为开发和流水线验证集，不直接作为最终充分功效的主结果。
- 正式 Agent-VQA 测试集覆盖尽可能多的灾害事件，并按事件隔离。
- Qwen2.5-VL-7B 是系统默认 VLM；不声称 VQA 模型层面的 SOTA。
- 前端只作为系统演示和案例证据，不作为定量实验数据来源。
- 记忆图作为可选能力或负结果报告，不放入主贡献标题。

---

## 3. 研究主张重建

### 3.1 允许在实现完成后陈述的存在性主张

- 系统支持自然语言任务与灾情问题输入。
- 系统使用显式、有限的动作与技能集合。
- 系统能将当前图像、结构化感知、语义地图和历史状态组织为回答证据。
- 系统能输出回答、置信度、证据定位、动作选择和结构化动作理由。
- 系统能在 VLM/LLM 不可用时执行规则回退。
- 前端能实时显示当前问题、证据、动作、重观测和最终报告。

这些是工程实现声明，不等价于效果声明。

### 3.2 只有实验满足条件后才能陈述的效果主张

- 结构化证据提高 VQA 正确率或校准性。
- HSPM 提高端到端问答成功率或路径效率。
- 主动重观测提高最终答案正确率、降低风险或改善校准。
- 记忆图降低重复任务的路径或步骤成本。
- 某个模块是系统瓶颈或必要组成。
- 系统具有跨事件泛化能力。

### 3.3 明确禁止的表述

- 不得将当前 VQA 上传接口直接称为“智能体问答贡献”。
- 不得将前端 `thinking` 字段称为 chain-of-thought。
- 不得把合成有效 GSD 阶梯解释为真实低空厘米级信息获取。
- 不得把 oracle 参照称为可部署方法或性能上界。
- 不得用不显著结果证明“没有差异”。
- 不得用 2 个测试事件证明一般性跨事件泛化。
- 不得把训练规模或架构差距写成已证实的因果解释，除非有控制实验。
- 不得把定位 Dice、提议召回、VQA Accuracy 直接写成导航或真实救援成功率。

### 3.4 结果分支与写作口径

#### 分支 A：主动 VQA 得到正向结果

只有同时满足以下条件，才可写“主动重观测改善最终回答”：

- 与不重观测和预算匹配随机策略相比，配对置信区间不含 0；
- 效应在多个灾害事件上方向基本一致；
- 改善不仅来自调用次数增加，且报告单位额外观测收益；
- 没有明显增加正确答案被破坏的比例；
- 在线策略没有读取未来测试观测。

#### 分支 B：答案准确率不变，但概率质量改善

写为：

> 主动策略未改变离散答案，但改善了概率质量、选择性风险或弃答行为；作用发生在置信分布而非 argmax 标签。

必须同时报告 Brier、NLL、ECE 和风险覆盖指标，不能只挑 ECE。

#### 分支 C：策略不可识别

写为：

> 当前感知底座下，巡航与近距观测产生的答案翻转或风险差异过少，主动策略在该评测构造中不可识别；这不是对真实重观测无效性的证明。

#### 分支 D：结构化证据或 HSPM 无收益

如实作为负结果报告，并将对应模块从主要效果贡献降为系统能力。不得通过更换指标或只选个别题型制造正向结论。

---

## 4. 论文目录重构

### 4.1 推荐目录

```text
1 引言
2 相关工作
3 问题定义与可审计智能体系统
  3.1 航空灾害 Agent-VQA 任务
  3.2 双模式编排与有限动作集
  3.3 感知、目标指认与证据状态
  3.4 HSPM、STMR 与可选记忆
  3.5 人机协同控制台与结构化决策轨迹
4 方法与评测协议
  4.1 问题解析与结构化回答
  4.2 证据充分性与预算受限主动重观测
  4.3 双时相损伤分类与校准
  4.4 事件隔离、泄漏约束与可识别性判据
  4.5 指标、统计检验与功效
5 实验
  5.1 感知底座与跨事件核查
  5.2 静态灾情 VQA
  5.3 结构化证据与目标指认消融
  5.4 主动 VQA 与观测预算
  5.5 端到端 Agent-VQA 任务
  5.6 Oracle 阶梯、退化与失败恢复
6 局限、伦理与适用边界
7 结论
附录 A 数据、问题模板与切分
附录 B 实现缺陷排查与复现记录
附录 C RescueNet 域外检查
附录 D 完整实验表与逐事件结果
```

### 4.2 现有章节迁移

| 当前文件 | 修改方向 |
|---|---|
| `sections/abstract.tex` | 完全重写为 Agent-VQA 主线，移除审稿过程叙事 |
| `sections/introduction.tex` | 以“回答问题需要主动获得证据”为动机，重写贡献列表 |
| `sections/related_work.tex` | 增加灾害 VQA、主动 VQA、具身问答和航空智能体文献族 |
| `sections/architecture.tex` | 保留真实系统，增加 Agent-VQA 控制器和证据状态；压缩重复模块描述 |
| `sections/method.tex` | 从“重观测单一方法”扩展为问题解析、回答、证据充分性、动作选择与协议 |
| `sections/experiments.tex` | 按六个实验块重排；调试历史和 RescueNet 下移附录 |
| `sections/limitations.tex` | 增加 VQA 标签边界、合成 GSD、模型依赖、事件规模和真实飞行边界 |
| `sections/conclusion.tex` | 以系统、Agent-VQA 任务和分层诊断收束，不承诺未验证增益 |
| `sections/appendix.tex` | 接收完整模板、LOEO、失败轨迹、RescueNet 和复现信息 |
| `main.tex` | 更新题名、前置信息、章节顺序和英文信息 |
| `macros.tex` | 增加 VQA/Agent 指标宏，清除不再使用的旧命名 |
| `refs.bib` | 补充并核验相关工作与目标期刊文献 |

### 4.3 摘要重写结构

摘要按五句组组织：

1. 航空灾害侦察中“回答问题”与“主动获取证据”的任务缺口。
2. 系统：语言条件智能体、有限动作、地理配准环境和可审计轨迹。
3. 方法：结构化灾情 VQA、证据充分性和预算受限重观测。
4. 实验：事件隔离 Agent-VQA、静态/主动/端到端/分层 oracle。
5. 结果：只填最终正式实验数字；负结果按适用边界表述。

摘要中删除：

- “外部审稿指出”。
- “第四处未发现的实现缺陷”等审稿过程语言。
- 未经 LOEO 支撑的“一般性跨事件不泛化”。
- “多数差距应归因于架构/训练规模”。
- 所有来源账本不一致的数字拼接。

---

## 5. Agent-VQA 任务定义

### 5.1 输入

每个 Agent-VQA 回合包含：

- 自然语言问题 `q`；
- UAV 当前位姿 `x_t=(lat, lon, alt)`；
- 当前观测图像 `I_t`；
- 可选灾前参考图像 `I_pre`；
- 结构化感知 `z_t`，包括检测、分割、四类损伤概率和场景摘要；
- 语义地图局部状态 `M_t`；
- 历史观测、动作和回答状态 `h_t`；
- 剩余动作与观测预算 `b_t`。

### 5.2 输出

统一结构化输出：

```json
{
  "question_id": "...",
  "question_type": "presence|damage|count|spatial",
  "answer": "...",
  "confidence": 0.0,
  "abstain": false,
  "evidence": {
    "source": "image|detector|change_classifier|semantic_map|history",
    "target_label": "...",
    "norm_xy": [0.0, 0.0],
    "observation_id": "..."
  },
  "decision": "answer|continue_search|reobserve|abstain",
  "reason_code": "sufficient_evidence|target_missing|low_confidence|budget_exhausted"
}
```

约束：

- `confidence` 必须在 `[0,1]`。
- `decision` 必须来自封闭集合。
- `reason_code` 是结构化动作理由，不是思维链。
- VLM 输出解析失败时返回显式 `invalid_output`，不得静默猜测。
- `evidence.norm_xy` 只有在确实定位到目标时才能填写。

### 5.3 动作集合

Agent-VQA 使用已有动作，不新增无法执行的抽象工具：

- `fly_to_geo`
- `fly_relative`
- `hover`
- `detect_disaster`
- `mark_target`
- `report_observation`
- `stop`

高层问答决策映射：

| 高层决策 | 底层动作 |
|---|---|
| `answer` | `report_observation` + `stop` |
| `continue_search` | HSPM 输出的 `fly_relative` 或 `fly_to_geo` |
| `reobserve` | 居中 `fly_relative` + 下降 `fly_relative` + `detect_disaster` |
| `abstain` | `report_observation` 明示证据不足 + `stop` |

### 5.4 单回合终止条件

- 已生成合法答案且证据充分。
- 明确弃答。
- 步数预算耗尽。
- 重观测预算耗尽。
- 到达高度下限。
- 连续离开 POST 影像覆盖。
- 用户中止。
- 不可恢复的执行错误。

所有终止条件进入结果日志，避免把超时、弃答和错误统一记作“回答错误”。

---

## 6. VQA 题库设计

### 6.1 主问题类型

#### Q1：存在判断

示例：

- 当前视场是否存在完全损毁建筑？
- 当前视场是否存在严重损伤建筑？

答案：`是/否`。

生成约束：

- 正例目标多边形必须达到规定的可见面积比例。
- 负例必须确认当前视场及边界缓冲区内没有对应类别。
- 正负样本按事件和类别近似平衡。

#### Q2：损伤等级

示例：

- 标记建筑的损伤等级是什么？
- 当前候选建筑属于无损伤、轻微、严重还是完全损毁？

答案：四分类。

生成约束：

- 必须有唯一目标指代。
- 使用目标编号、标记点或明确空间描述，避免“画面里的建筑”歧义。
- 目标裁块像素占比和边界截断率写入难度字段。

#### Q3：数量判断

示例：

- 当前视场有多少栋严重或完全损毁建筑？

答案采用分桶：`0 / 1 / 2 / 3+`。

不以精确计数作为唯一主指标，原因是视场边缘截断和密集建筑实例边界可能存在标注歧义。

#### Q4：空间定位

示例：

- 最近的完全损毁建筑位于无人机哪个方向？

答案：八方向 `北/东北/东/东南/南/西南/西/西北`，并可附归一化点坐标。

生成约束：

- 方向由地理坐标计算，不由图像文字模板猜测。
- 距离相近的多个同类目标须过滤或标记为歧义。

### 6.2 可选问题类型

以下题型放在 P2 扩展，不阻断主稿：

- 灾前/灾后变化比较。
- 多地标顺序问答。
- 重复区域记忆问答。
- 自由文本灾情摘要。
- 反事实问题，如“如果不下降，是否应弃答”。

### 6.3 不生成的问题

- 道路是否安全可通行。
- 是否有人被困或需要医疗救援。
- 火势、洪水深度或结构稳定性精确等级。
- xBD 标注无法直接验证的任意开放式建议。

这些内容可出现在 VLM 演示文本中，但不得进入自动评分主表。

### 6.4 题库记录格式

建议新建 `backend/data/benchmarks/agent_vqa_testset.json`：

```json
{
  "schema_version": "agent-vqa/1.0",
  "dataset_manifest_sha256": "...",
  "split_policy": "event-disjoint",
  "items": [
    {
      "id": "...",
      "scene_id": "...",
      "tile_id": "...",
      "disaster": "...",
      "split": "test",
      "question_type": "damage",
      "question": "...",
      "choices": ["无损伤", "轻微损伤", "严重损伤", "完全损毁"],
      "answer": "完全损毁",
      "start": {"lat": 0.0, "lon": 0.0, "alt": 30.0},
      "target": {"lat": 0.0, "lon": 0.0, "subtype": "destroyed"},
      "observation_profile": "cruise",
      "difficulty": {
        "distance": "medium",
        "target_pixels": 0,
        "clutter": 0,
        "edge_truncation": 0.0
      },
      "review": {
        "status": "pending",
        "ambiguity_flags": [],
        "author_checked": false
      }
    }
  ]
}
```

### 6.5 数据生成程序

新增：

- `scripts/benchmarks/gen_agent_vqa_testset.py`
- `scripts/benchmarks/review_agent_vqa_testset.py`
- `scripts/benchmarks/render_agent_vqa_review_sheet.py`

复用：

- `scripts/benchmarks/gen_vln_testset.py` 的瓦片、建筑、多边形、起点和方位逻辑。
- `xbd_map` 和 `xbd_store` 的地理投影与瓦片查找。

生成器必须执行：

1. 事件白名单和黑名单检查。
2. 目标多边形可见性检查。
3. 负例边界缓冲检查。
4. 唯一目标与空间歧义检查。
5. 问题类型和答案分布统计。
6. 题库和数据清单 SHA-256 记录。
7. train/val/test 事件交集断言。
8. 重复题、重复目标和近重复问题检查。

### 6.6 审核协议

- 100% 自动几何与答案一致性检查。
- 所有带歧义标志的题由作者检查。
- 每个事件、问题类型和难度层至少抽查一定比例。
- 审核记录必须写“模型辅助生成 + 作者抽查”，不得写成纯人工审核。
- 审核只检查问题是否由标注支持，不使用测试模型输出来决定保留哪些题。

### 6.7 数据规模策略

1. **开发集：**复用当前 200 个 evidence-rich 场景，每场景生成 3 至 4 个事实问题，用于代码和协议调试。
2. **正式集：**根据功效分析决定场景数；若以检测 5 个百分点差异为目标，至少按当前功效表规划约 444 至 500 个独立场景，而不是把同一场景的多个问题当作独立场景。
3. **事件单位：**尽可能覆盖全部可用事件；结果按事件聚类并报告事件间方差。
4. **问题单位：**同一场景派生的多个问题可用于题型分析，但主统计必须处理场景内相关性。

---

## 7. 后端修改计划

### 7.1 新增 `backend/agent_vqa.py`

职责：

- 问题类型识别。
- 目标短语和空间约束提取。
- VLM 结构化提示构造。
- JSON 响应解析与 schema 校验。
- 结构化感知证据注入。
- 回答充分性判断。
- 最终答案与证据记录。

建议核心对象：

- `QuestionSpec`
- `EvidenceBundle`
- `VqaAnswer`
- `AgentVqaConfig`
- `AgentVqaController`

避免把全部逻辑继续堆进 `backend/app.py`。

### 7.2 修改 `backend/vlm_analyzer.py`

1. 保留现有自由文本 `analyze_image_bytes()`，兼容上传演示。
2. 增加结构化问答接口，例如 `answer_image_question()`。
3. 支持自定义严格 system prompt。
4. 将温度默认设为确定性或接近确定性的配置，并记录实际 decoding 参数。
5. 增加 JSON 格式错误、非法 confidence、非法 choice 的显式错误类型。
6. 可选增加多图输入，为灾前/灾后成对 VQA 服务；如暂不实现，则在论文中明确主 VQA 只直接读取当前视图，双时相信息通过结构化分类器提供。

### 7.3 修改 `backend/app.py`

新增控制器与入口：

- `run_agent_vqa_episode(question, source="ai")`
- `run_agent_vqa_episode_headless(question, start, item=None, source="bench")`
- REST：`POST /api/agent/query`
- Socket：`agent_query`
- Socket 输出：`agent_query_started`、`agent_query_update`、`agent_query_result`

执行流程：

1. 解析问题和目标。
2. 初始化语义地图、问答状态和预算。
3. 获取当前观测。
4. 生成候选答案和证据。
5. 若目标缺失，调用 HSPM 继续搜索。
6. 若目标存在但证据不足，调用重观测控制器。
7. 生成最终回答或弃答。
8. 返回完整轨迹、问答历史、预算使用和错误状态。

必须避免：

- 从测试条目的 `answer` 或未来图像读取在线决策信息。
- 将前端可见的 GT 建筑足迹传入智能体观测。
- 通过 `item` 参数向非 oracle 配置泄漏目标坐标。

### 7.4 复用 `backend/recheck.py`

保留已有重观测状态机，但增加 Agent-VQA 所需字段：

- `answer_before`
- `confidence_before`
- `answer_after`
- `confidence_after`
- `answer_changed`
- `answer_corrected`，仅离线评测阶段计算，在线控制器不可读取。
- `answer_harmed`，仅离线评测阶段计算。

修正条件期望策略：

- 当前错误实现不能使用测试样本未来原生视图熵。
- 新查找表只能在验证事件上拟合。
- 在线特征只允许包含当前 GSD、当前预测类、当前熵分桶、当前目标大小或其他当前可见特征。
- 推荐形式：

```text
E[U_next | current_gsd, current_predicted_class, current_entropy_bin, action]
```

- 未来测试视图只用于回放评价和 oracle 参照。

### 7.5 修改 `backend/perception.py`

- 为每个候选目标生成稳定 `evidence_id`。
- 保留 `class_probs`、检测来源、目标框、图像位置和观测 ID。
- 记录目标是否来自检测器、VLM 还是 oracle。
- 不把 `scene_text` 中未经验证的自由文本当作事实标签。
- 为问答控制器提供精简、固定字段的 `EvidenceBundle`。

### 7.6 修改 `backend/vln_navigator.py` 与 `backend/hspm_planner.py`

- 允许从问题中提取目标短语，而不要求输入是命令句。
- 支持“是否存在”“损伤等级是什么”“位于哪里”等问句。
- HSPM 只负责搜索和运动，不直接伪造事实答案。
- 目标不可见时允许 `continue_search`；目标可见后把回答权交给 Agent-VQA 控制器。
- 对 LLM 输出使用固定 JSON schema 和封闭方位集合。
- 所有 fallback 进入 `reason_code` 和轨迹日志。

### 7.7 状态与日志

每步记录：

- `question_id`
- `observation_id`
- `position`
- `question_type`
- `candidate_answer`
- `confidence`
- `evidence_ids`
- `decision`
- `reason_code`
- `action`
- `budget_before/after`
- `fallback_used`
- `degraded_reason`

对外展示字段不得命名为 `chain_of_thought`。

### 7.8 后端单元测试

新增：

- `backend/tests/test_agent_vqa.py`
- `backend/tests/test_agent_vqa_schema.py`

至少覆盖：

1. 四种问题类型解析。
2. 中文和英文问句。
3. 合法 JSON 回答解析。
4. 非法答案、非法 confidence 和缺字段拒绝。
5. 目标不存在时的继续搜索。
6. 低置信时的重观测。
7. 预算耗尽时的弃答。
8. LLM/VLM 不可用时 fallback。
9. 非 oracle 配置无法读取 GT。
10. 当前观测不变时结果结构稳定。
11. 条件期望策略不访问未来观测。
12. 日志区分错误、弃答和普通错误答案。

---

## 8. 前端修改计划

### 8.1 任务入口

修改 `frontend/src/components/TaskPanel.jsx`：

- 增加“Ask Agent”问题输入。
- 与现有“AI Task”和“VLN”保持清晰分工。
- 提交事件调用 `onSubmitAgentQuery(question)`。
- 提供停止按钮，复用现有 `stop_execution`。

建议交互：

- AI Task：一次性复合动作规划。
- VLN：只导航到语言目标。
- Ask Agent：搜索证据并回答灾情问题，是论文主演示入口。

### 8.2 Socket 状态

修改 `frontend/src/hooks/useSocket.js`：

- 增加 `agentQueryState`。
- 监听 `agent_query_started`。
- 监听 `agent_query_update`。
- 监听 `agent_query_result`。
- 新增 `submitAgentQuery`。
- 新任务开始时清理旧回答和旧证据，防止状态残留。

### 8.3 问答结果展示

修改 `frontend/src/components/PerceptionPanel.jsx` 或新增 `AgentQueryPanel.jsx`，显示：

- 当前问题。
- 当前答案与置信度。
- 证据来源和目标位置。
- 当前动作：搜索、居中、下降、回答或弃答。
- 剩余预算。
- 重观测前后答案对比。
- 结构化动作理由。
- 最终状态与失败原因。

不显示模型私有思维过程，不使用“思维链”字样。

### 8.4 上传图片 VQA

保留 `frontend/src/components/VisionPanel.jsx`，但重新定位为：

- “静态 VQA 基线/独立图像分析”。
- 不与完整 Agent-VQA 混为同一模式。
- 结构化问答与自由文本分析使用不同接口和显示区域。

### 8.5 前端验收

- 问题文本在桌面和移动宽度下不溢出。
- 问答状态不会与旧 VLN 状态串台。
- 重观测前后答案可以明确区分。
- 停止后状态进入 `cancelled`，不显示为成功。
- 后端错误、弃答和无证据分别显示。
- 截图中不出现未填写占位符、内部实验编号或开发路径。

---

## 9. Benchmark 与结果生成程序

### 9.1 新增主评测脚本

新增 `scripts/benchmarks/bench_agent_vqa.py`。

职责：

- 读取 Agent-VQA 题库。
- 按配置运行同一问题的配对实验。
- 调用 `run_agent_vqa_episode_headless()`。
- 每题即时保存，支持中断续跑。
- 保存环境、模型、prompt、阈值、git commit 和数据 hash。
- 区分在线字段与离线评分字段。

建议配置：

| 配置 | 图像/状态 | 可移动 | 可重观测 | 目的 |
|---|---|---:|---:|---|
| `V0_RAW` | 当前图像 | 否 | 否 | 静态 VLM 基线 |
| `V1_STRUCT` | 图像 + 结构化感知 | 否 | 否 | 结构化证据价值 |
| `V2_STATE` | V1 + STMR + 历史 | 否 | 否 | 状态与历史价值 |
| `A0_HOLD` | 完整当前状态 | 是 | 否 | Agent 单观测基线 |
| `A1_RANDOM` | 完整状态 | 是 | 随机 | 预算匹配对照 |
| `A2_ALWAYS` | 完整状态 | 是 | 总是 | 额外观测上限对照 |
| `A3_ENTROPY` | 完整状态 | 是 | 当前熵 | 主动策略 |
| `A4_CONFORMAL` | 完整状态 | 是 | 共形集合 | 不确定性对照 |
| `A5_EXPECTED` | 完整状态 | 是 | 验证集期望收益 | 泄漏安全条件策略 |
| `O_REF` | 使用未来结果 | 诊断 | 诊断 | oracle 参照，禁止部署 |

正文不必展示全部配置。正式表根据结果压缩为主要对照，完整结果放附录。

### 9.2 新增汇总脚本

新增 `scripts/benchmarks/report_agent_vqa.py`。

输出：

- 总体与分题型指标。
- 分事件指标。
- 分难度指标。
- 配对差值和置信区间。
- 回答翻转、纠错和损害矩阵。
- 风险覆盖曲线。
- 预算效用曲线。
- fallback 与错误类型统计。

### 9.3 新增论文资产生成

扩展或新增：

- `scripts/benchmarks/export_agent_vqa_assets.py`
- 或将逻辑接入 `export_cja_assets.py`，但必须保持 Agent-VQA 数据源独立且可追溯。

禁止在生成器中硬编码论文结果。所有 LaTeX 表必须由正式结果 JSON/CSV 生成。

### 9.4 结果目录

建议：

```text
runs/benchmarks/cja_agent_vqa/
  manifest.json
  env_snapshot.json
  prompt_manifest.json
  configs/
    V0_RAW/
    V1_STRUCT/
    V2_STATE/
    A0_HOLD/
    A1_RANDOM/
    A2_ALWAYS/
    A3_ENTROPY/
    A4_CONFORMAL/
    A5_EXPECTED/
    O_REF/
  reports/
    aggregate.json
    paired_tests.json
    event_breakdown.csv
    failure_taxonomy.csv
  figures/
  tables/
```

---

## 10. 完整实验计划

### E0：协议与泄漏单元测试

**目的：**证明实验实现没有读取未来观测、GT 答案或前端标注层。

**操作：**

1. 对未来观测对象加访问哨兵，在线策略访问即测试失败。
2. 对题库 `answer` 字段在运行前进行遮蔽，评测完成后才恢复评分。
3. 验证 agent 输入中不含 xBD GT 多边形和类别。
4. 验证 oracle 配置与普通配置使用不同明确开关。
5. 检查事件切分交集为空。

**产物：**测试报告与数据流图。

**阻断条件：**任一泄漏测试失败，不得运行或报告正式结果。

### E1：静态灾情 VQA

**对应 RQ：**RQ1。

**比较：**

- 多数类/规则基线。
- `V0_RAW`。
- 可选目标裁块 oracle 诊断。

**问题：**存在、损伤、数量、空间四类。

**指标：**

- Accuracy、Macro-F1。
- 分题型 F1。
- Count MAE。
- 八方向准确率、点定位误差。
- ECE、Brier、NLL。
- 无效输出率和弃答率。

**解释边界：**静态 VQA 只测“看见后能否回答”，不构成智能体效果。

### E2：结构化证据消融

**对应 RQ：**RQ2。

**比较：**

- `V0_RAW`：图像。
- `V1_STRUCT`：图像 + detector/change classifier/scene fields。
- `V2_STATE`：V1 + STMR + 历史。
- 可选 `V2_NO_IMAGE`：仅结构化状态，诊断答案是否被检测器完全决定。

**控制变量：**

- 同一 VLM。
- 同一问题。
- 同一 decoding。
- 同一图像。
- 同一输出 schema。

**主指标：**配对答案正确率差、Brier 差、无效输出率差。

**必须检查：**结构化文本是否直接泄漏 GT 标签；只能使用模型预测和在线状态。

### E3：目标指认与 grounded VQA

**对应 RQ：**RQ2、RQ4。

**比较：**

- YOLO/change classifier 指认。
- VLM 指认。
- Hybrid 指认。
- Oracle target crop 参照。

**指标：**

- Pointing accuracy。
- 到 GT 多边形距离。
- Grounded Answer Success：答案正确且证据点命中目标。
- 误指认率。
- 正确答案但错误证据率。

**重要性：**防止 VLM 猜对类别却指错建筑，或在错误目标上生成看似正确的答案。

### E4：主动 VQA 与预算分配

**对应 RQ：**RQ3。

**比较：**`A0_HOLD`、`A1_RANDOM`、`A2_ALWAYS`、`A3_ENTROPY`、`A4_CONFORMAL`、`A5_EXPECTED`、`O_REF`。

**预算：**沿用 `B∈{0,0.1,0.25,0.5,1.0}`，含义为每题平均额外观测次数。

**主指标：**

- 最终答案 Macro-F1。
- 最终 Brier/NLL。
- AURC。
- 每次额外观测纠正的答案数。
- `corrected / triggered`。
- `harmed / triggered`。
- 单位水平和垂直运动成本收益。
- 触发率、完成率和预算耗尽率。

**前置检查：**

- `n_answer_flip`：巡航与近距答案翻转数。
- `n_correctable`：巡航错误且近距正确的题数。
- `n_harmful`：巡航正确且近距错误的题数。
- oracle 参照与 hold 的最大可用间隙。

若 `n_correctable` 过少，E4 只能报告不可识别性，不能比较策略优劣。

### E5：端到端 Agent-VQA

**对应 RQ：**RQ1 至 RQ4。

**场景：**目标初始可能不在视场内，智能体必须搜索、指认、接近、判断并回答。

**配置：**

- 关键词贪心 + 静态回答。
- HSPM + 静态回答。
- HSPM + 结构化回答。
- HSPM + 结构化回答 + 主动重观测。
- 记忆版仅在专门重复任务实验中报告。

**指标：**

- Answer Success：最终答案正确。
- Grounded Answer Success：答案正确且证据目标正确。
- Strict SR、NE、SPL。
- Answer-conditioned SPL：只有答案与证据均正确时计路径效率。
- Steps、路径长度、额外观测数。
- 合法动作率、越界次数、fallback 率。
- 弃答率和错误终止率。

**解释：**回答正确、到达成功和证据正确必须分别报告，不能压成一个总分掩盖失效层。

### E6：规划与有限技能合法性

**对应 RQ：**RQ5。

**任务族：**

1. 明确坐标巡检。
2. 方位目标搜索。
3. 多地标顺序任务。
4. 否定和空间关系指令。
5. 需要灾情检测和报告的复合任务。

**指标：**

- JSON/schema 合法率。
- 动作名称合法率。
- 参数字段合法率。
- 到达后执行 `detect_disaster` 的证据观察率。
- 计划执行完成率。
- 越界动作率。
- LLM 失败后的规则回退成功率。

此实验提供智能体系统的正面工程证据，不依赖损伤分类器必须取得高 F1。

### E7：重复任务与记忆，可选

**对应 RQ：**只在作者决定保留记忆为正文因素时运行。

**设计：**同一或相邻区域的冷启动、首次任务和重复任务配对。

**指标：**

- 重复任务步骤数和路径长度变化。
- 首答案延迟。
- 记忆命中率。
- 错误记忆率。
- SR/Grounded Answer Success 是否退化。

若记忆提高命中率但使 NE 或答案正确率变差，应作为安全风险或负结果，不得只报路径缩短。

### E8：退化与失败恢复

**对应 RQ：**RQ5。

**退化条件：**

- LLM 不可用。
- VLM 不可用。
- 检测器零提议。
- 目标指认误报。
- 当前视场无 POST 覆盖。
- 强制几何投影退化。
- GPS 起点噪声。
- 预算耗尽。

**指标：**

- 未处理异常率。
- fallback 使用率。
- 合法终止率。
- 恢复后 Answer Success。
- 额外步骤和路径。
- 错误状态是否被前端正确展示。

几何退化与正常条件使用同一批题时，采用配对检验。

### E9：感知底座补强与跨事件实验

这一实验块继承 `REVISION_EXPERIMENTS.md`，仍是 Agent-VQA 正式结果的前置，而不是被 VQA 取代。

#### E9.1 19 事件 LOEO

- 19 个事件逐个留出。
- 每折至少 3 个 seed。
- 每折重新拟合温度。
- 记录类别先验、逐类召回和事件间方差。
- 输出 forest plot。

#### E9.2 四个主要训练配置多种子

- 报告均值、标准差和配对差。
- 若标准切分与事件切分区间重叠，删除确定性架构归因。

#### E9.3 224 px 对照

- 明确这是输入上下文/裁块尺寸实验。
- 不得称为产生了新的物理分辨率。
- 与 96 px 使用同一事件切分和训练配方。

#### E9.4 ResNet50 + focal loss，可选

- 只有要保留“架构/训练规模是主要解释”时才需要。
- 否则将架构差距改写为候选解释。

### E10：X2 概率质量重分析

- Brier。
- NLL。
- 风险覆盖与 AURC。
- 代价敏感风险，至少报告 3:1、5:1、10:1 敏感性。
- 当前条件期望策略修复后全部重跑。
- 与 Agent-VQA E4 使用一致的在线/离线信息边界。

---

## 11. 统计分析计划

### 11.1 分析单位

- 单题指标：VQA 答案和定位。
- 场景单位：同一图像派生多个问题时的聚类单位。
- 事件单位：跨灾害泛化的主要独立单位。
- seed 单位：训练与随机生成波动。

不得将同一场景的多个模板问句当作完全独立样本。

### 11.2 主检验

- 二元正确/错误：McNemar。
- 多分类差异：配对 bootstrap 的 Macro-F1 差。
- 连续指标：事件或场景聚类 paired bootstrap。
- 路径和成本：配对置换或 bootstrap。
- 多策略比较：Holm 校正。
- LOEO：报告事件均值、事件间标准差/方差和 seed 内方差。

### 11.3 置信区间

- 单一比例：Wilson 95% CI。
- 指标差：配对 bootstrap 95% CI。
- 事件汇总：同时报告宏平均和事件分布，不只给合并样本 CI。

### 11.4 校准分析

- ECE 只作一个描述指标，不单独证明校准更好。
- 同时报告 Brier 和 NLL。
- 报告 reliability diagram。
- 报告风险覆盖曲线和 AURC。
- 共形方法报告目标覆盖率、经验覆盖率和集合大小。

### 11.5 主动观测效用

分别报告，不急于压成单一分数：

- 答案收益。
- 概率收益。
- 运动成本。
- 额外观测成本。
- 弃答/覆盖变化。

如需要总代价效用，必须对多个代价权重做敏感性分析，不选择一个最有利于方法的权重作为唯一结论。

### 11.6 预先写入的可识别性判据

正式跑策略比较前检查：

1. 近距观测是否改变答案或概率。
2. 可纠正样本是否足够。
3. oracle 参照是否与 hold 拉开可测间隙。
4. 样本量是否能检测预期效应。
5. 事件间方差是否大于策略差。

任一项不满足时，在结果中标记 `UNIDENTIFIABLE` 或 `UNDERPOWERED`，不写成策略失败。

---

## 12. 复现与数据完整性计划

### 12.1 每个正式 run 必须记录

- git commit 和工作区 dirty 状态。
- Python、PyTorch、CUDA 和模型版本。
- VLM 模型路径、权重 hash 和 decoding 参数。
- 数据 manifest 与题库 hash。
- 所有环境开关。
- prompt 文本和 prompt hash。
- 随机种子。
- 事件切分。
- 温度和共形阈值拟合来源。
- 启动命令、退出码和输出路径。

### 12.2 预测缓存

- 静态 VQA 原始响应逐题保存。
- 解析后的结构化响应单独保存。
- 不覆盖原始响应。
- 聚合脚本只读缓存，不重新请求模型。
- 题库或 prompt hash 变化后缓存自动失效。

### 12.3 错误状态

至少区分：

- `invalid_question`
- `invalid_model_output`
- `target_not_found`
- `abstained`
- `budget_exhausted`
- `out_of_coverage`
- `planner_unavailable`
- `vlm_unavailable`
- `execution_error`
- `cancelled`

### 12.4 测试门槛

正式实验前必须通过：

- 后端单元测试。
- benchmark 工具测试。
- 小规模 8 至 20 场景 smoke test。
- 题库泄漏和事件交集检查。
- 同一 seed 重放检查。
- 结果表生成检查。
- LaTeX 生成资产缺失检查。

---

## 13. 论文图表计划

### 13.1 正文图

1. **图 1：Agent-VQA 系统架构。** 操作员问题、编排器、感知、HSPM/STMR、问答控制器、重观测、有限动作和前端。
2. **图 2：完整问题轨迹。** 问题、当前观测、候选答案、低证据、下降居中、新观测、最终答案。
3. **图 3：静态与结构化 VQA 分题型表现。** 带置信区间。
4. **图 4：预算-答案质量/风险覆盖曲线。** 不只画 Macro-F1。
5. **图 5：分层 oracle 瓶颈图。** 明确每个 oracle 替换的层。
6. **图 6：事件级结果或 LOEO forest plot。** 若版面不足移附录。

### 13.2 正文表

1. 任务、动作与证据字段定义。
2. Agent-VQA 数据集分布。
3. 静态 VQA 与结构化证据消融。
4. 主动 VQA 主要策略。
5. 端到端 Agent-VQA 与路径指标。
6. 退化恢复与合法动作指标。

完整策略表、逐事件表、逐类表和全部题型放附录。

### 13.3 前端截图

正文最多保留两张：

- 系统控制台全貌。
- 单次 Agent-VQA 重观测前后状态。

补齐：

- `paper_cja/figures/ui_overview.png`
- `paper_cja/figures/agent_vqa_trace.png`

截图必须来自真实运行，不得保留“待补截图”占位框。

---

## 14. 逐文件写作修改清单

### 14.1 `paper_cja/main.tex`

- [ ] 替换中文题名。
- [ ] 增加英文题名。
- [ ] 增加中英文作者与单位占位。
- [ ] 增加中英文关键词。
- [ ] 增加中图分类号、文献标识码、基金和作者简介占位。
- [ ] 确认所有章节文件都被 `\input`。
- [ ] 更新目录顺序。
- [ ] 检查匿名投稿信息。

### 14.2 `paper_cja/sections/abstract.tex`

- [ ] 重写中文摘要。
- [ ] 新增英文摘要。
- [ ] 用最终 Agent-VQA 结果替换旧数字。
- [ ] 删除审稿过程、bug 发现过程和过强归因。
- [ ] 明确有效 GSD 是合成阶梯。
- [ ] 明确系统存在性与效果结论分开。

### 14.3 `paper_cja/sections/introduction.tex`

- [ ] 第一段改为“操作员问题需要证据获取”。
- [ ] 对比静态灾害 VQA 与可行动 Agent-VQA。
- [ ] 给出五个 RQ 或压缩为四个主 RQ。
- [ ] 重写贡献列表。
- [ ] 删除“所有模块都是创新”的写法。
- [ ] 删除未经验证的“首次”。

### 14.4 `paper_cja/sections/related_work.tex`

- [ ] 航空/俯视 VLN。
- [ ] 具身问答与交互式 VQA。
- [ ] 主动视觉问答和主动感知。
- [ ] 灾害、遥感和航空 VQA。
- [ ] 双时相变化理解与损伤评估。
- [ ] 不确定性校准、选择性预测和共形预测。
- [ ] 每个文献族最后明确本文差异。
- [ ] 所有新增引用查 DOI 或官方元数据，禁止依据现有笔记直接造引用。

### 14.5 `paper_cja/sections/architecture.tex`

- [ ] 将独立图片上传 VQA 改为静态基线/辅助功能。
- [ ] 增加 Agent-VQA 控制器。
- [ ] 增加 EvidenceBundle 与结构化回答数据流。
- [ ] 描述 Socket.IO 问答事件。
- [ ] 将 `thinking` 改称结构化决策轨迹或动作理由。
- [ ] 压缩前端功能清单，正文使用一条完整任务 trace 讲系统。
- [ ] 将系统存在性声明与性能主张分开。

### 14.6 `paper_cja/sections/method.tex`

- [ ] 增加问题、状态、动作、回答和预算形式化。
- [ ] 定义问题类型和结构化答案。
- [ ] 定义回答、弃答、继续搜索和重观测动作。
- [ ] 修正条件期望信息增益的在线信息边界。
- [ ] 定义 Answer Success、Grounded Answer Success 和 Answer-conditioned SPL。
- [ ] 增加 VQA 校准与选择性风险指标。
- [ ] 保留并缩短双时相分类器说明。
- [ ] 将功效分析和可识别性判据放在结果之前。

### 14.7 `paper_cja/sections/experiments.tex`

- [ ] 删除所有“审稿意见 C1/C2”等内部标记。
- [ ] 按 E1 至 E8 重新组织。
- [ ] 分类器核查作为感知底座实验，不再占据摘要主体。
- [ ] 加入静态 VQA 表。
- [ ] 加入结构化证据消融。
- [ ] 加入主动 VQA 预算实验。
- [ ] 加入端到端 Agent-VQA。
- [ ] 重做 oracle 阶梯命名。
- [ ] 所有相同题库比较改用配对统计。
- [ ] 将 RescueNet 移附录。
- [ ] 将缺陷排查移附录。
- [ ] 压缩恒等性结果和零触发表。

### 14.8 `paper_cja/sections/limitations.tex`

- [ ] xBD 问题类型受建筑损伤标注限制。
- [ ] VQA 主要是封闭式自动评分，不代表开放式应急推理。
- [ ] 有效 GSD 阶梯不产生真实低空新信息。
- [ ] 二维匀速模型不代表真实飞控。
- [ ] VLM 结果依赖单一主模型。
- [ ] 事件数和样本量限制。
- [ ] 记忆、HSPM 和主动复观测的负结果或未识别状态。
- [ ] 前端不是正式人因验证。
- [ ] 不涉及真实部署安全认证。

### 14.9 `paper_cja/sections/conclusion.tex`

- [ ] 回答 RQ，而不是重复模块列表。
- [ ] 分开系统实现、实证结果和负结果。
- [ ] 删除跨账本数字混用。
- [ ] 不用 oracle 结果证明真实系统已解决导航。
- [ ] 给出下一步真实低空数据、更多事件和真实飞控验证。

### 14.10 `paper_cja/sections/appendix.tex`

- [ ] 题库 schema 与模板。
- [ ] 事件切分和数据统计。
- [ ] LOEO 逐事件结果。
- [ ] 完整策略结果。
- [ ] 失败类型与典型轨迹。
- [ ] 缺陷排查记录。
- [ ] RescueNet 检查。
- [ ] 运行命令和配置清单。
- [ ] 删除绝对路径和内部审稿文件名。

### 14.11 `paper_cja/refs.bib`

- [ ] 引用数量扩充到能覆盖所有相关工作族。
- [ ] 补充中文航空、遥感和应急智能体相关文献。
- [ ] 补充目标期刊近期相关论文。
- [ ] 补充 active VQA、embodied QA、selective prediction、conformal prediction。
- [ ] 核验每条作者、题名、年份、期刊/会议、DOI。
- [ ] 检查正文引用与参考文献一一对应。

### 14.12 资产生成与格式

- [ ] `export_cja_assets.py` 删除“Oracle 上界”和内部 `M7`。
- [ ] 新表统一由正式结果生成。
- [ ] 修复最大约 233 pt 的 overfull box。
- [ ] 表格使用 `tabularx`、缩写或拆表，避免横向出栏。
- [ ] 运行 `latexmk -xelatex` 并确保无缺图、缺表、未定义引用。
- [ ] 清理 `n/a`、TODO、“待补截图”和作者内部备注。

---

## 15. 文献与定位计划

### 15.1 文献检索主题

1. Aerial Vision-Language Navigation。
2. Embodied Question Answering。
3. Interactive/Active Visual Question Answering。
4. Remote Sensing VQA。
5. Disaster VQA and disaster multimodal benchmarks。
6. Active Perception and information-gain planning。
7. Selective classification and risk-coverage。
8. Conformal prediction for vision/robotics。
9. xBD/xView2 building damage assessment。
10. Agent-based change understanding。

### 15.2 检索输出

- 文献矩阵：任务、输入、动作能力、是否可移动、是否有灾害语义、是否有问答、是否有不确定性。
- 一张相关工作对比表。
- 每条核心新颖性主张对应至少一组检索记录。

### 15.3 新颖性边界

推荐表述：

> 本文研究静态灾害 VQA 与航空语言导航之间的交叉问题：智能体不仅回答当前图像，还可在地理配准环境中采取搜索和重观测动作获取回答证据。

在未完成系统检索前，不使用“首个”“首次”“填补空白”等绝对表述。

---

## 16. 执行阶段与依赖关系

以下顺序由科学与代码依赖决定，不表示审稿意见重要性评分。

### 阶段 D0：冻结论文边界

- [x] 作者确认以智能体系统为主题。
- [x] 作者确认加入 VQA 实验。
- [x] 确认封闭式 xBD VQA 为主实验。
- [x] 确认不做正式人机实验。
- [x] 确认记忆图默认移附录。
- [ ] 确认目标期刊与格式要求。

**出口：**题名、RQ、贡献和禁止主张冻结。

### 阶段 D1：修复现有科学有效性阻断项

- [x] 修复 X2 未来观测泄漏。
- [x] 改用配对统计。
- [x] 完成事件切分断言。
- [ ] 规划并启动 LOEO + 多 seed。
- [ ] 明确 224 px 实验含义。
- [ ] 清理内部审稿标记和 oracle 命名。

**出口：**旧实验不再给新 Agent-VQA 提供错误方法或错误统计基础。

### 阶段 D2：Agent-VQA schema 与题库

- [x] 冻结四种主问题类型。
- [x] 实现题库生成、审核和渲染。
- [x] 生成开发集。
- [x] 完成泄漏测试和分布检查。
- [ ] 作者抽查。

**出口：**固定 hash 的开发题库与协议说明。

### 阶段 D3：后端闭环

- [x] 实现结构化问答控制器。
- [x] 接入 HSPM 搜索和 recheck。
- [x] 增加 REST、Socket 和 headless 入口。
- [x] 完成单元测试。
- [x] 记录结构化轨迹。

**出口：**单题可从问题运行到答案/弃答，且无 GT 泄漏。

### 阶段 D4：前端问答界面

- [x] 增加 Ask Agent 输入。
- [x] 展示当前答案、证据、动作和预算。
- [x] 展示复观测前后对比。
- [x] 错误和弃答状态可辨。
- [ ] 拍摄正式截图。

**出口：**系统演示链路完整，但不将截图作为实验结果。

#### 2026-08-20 核验记录

- D1 为部分完成：X2 泄漏边界、配对统计和事件切分断言已落实；LOEO、多 seed、
  224 px 对照和内部审稿标记清理仍是阻断项，因此不得把 D1 整体写成完成。
- D2 自动审核为 200/200 通过，题库 hash 为 `21edf4939ca4e389`；96 道带歧义标志
  的题仍需作者抽查，不得声称“人工审核完成”。
- D3 相关控制器、schema、app、recheck 和 benchmark 测试共 75 项通过；另修复了
  Qwen/感知并发加载造成的 BF16/FP32 dtype 竞态，并记录 VLM 原始输出、schema 错误、
  在线 evidence、重观测原因和动作前观测位姿。
- D4 前端生产构建通过；正式截图尚未拍摄。
- 有效 GPU smoke 目录为 `runs/benchmarks/cja_agent_vqa/d5_smoke_gpu1_after_schema/`，
  仅含 8 题且运行于重观测审计增强之前。其 4 个配置的配对 95% CI 均包含 0，不能
  支撑性能提升结论。`d5_smoke_reobs_audit/` 与 `d5_smoke_reobs_gpu1/` 各含 40 个
  执行错误，已由报告入口拒绝；CPU 结果 fallback rate 为 1.0，只能证明动作通道，
  不进入 VQA 准确率主分析。

### 阶段 D5：小规模 pilot

- [x] 8 至 20 场景 smoke test（作者指示跳过；由 100 题 GPU test 替代）。
- [x] 100 题 test 开发评测（holdout 未消费；原“200 场景”会提前用掉 holdout）。
- [x] 检查无效输出、答案分布和运行成本。
- [x] 检查 `n_correctable` 和 oracle 间隙。
- [x] 根据 pilot 做功效分析并冻结正式样本量：holdout 维持 n=100，配置冻结为
      V0/V1/A0/A2/A3（不含 A1/A4/A5，理由见
      `AGENT_VQA_EXPERIMENT_STATUS.md`“冻结决定”）。spatial
      `decision_reason_mismatch` 已修 prompt，尚待 GPU 重跑验证后再提交并启动 holdout。

**出口：**确认正式实验可识别，或提前记录不可识别风险。

### 阶段 D6：正式实验

- [ ] E1 静态 VQA。
- [ ] E2 结构化证据。
- [ ] E3 grounded VQA。
- [ ] E4 主动 VQA（配置冻结为 V0/V1/A0/A2/A3；`n_correctable=4` 下 A1/A4/A5
      大概率不可识别，不纳入本轮，预写“可识别但样本不足以排序”作为候选结论分支）。
- [ ] E5 端到端 Agent-VQA。
- [ ] E6 规划合法性。
- [ ] E8 退化恢复。
- [ ] E9 感知底座实验。
- [ ] E10 概率质量重分析。
- [ ] E7 仅在正文保留记忆时运行。

**出口：**冻结的正式结果目录与聚合报告。

### 阶段 D7：论文重写

- [ ] 先写方法和协议。
- [ ] 再按冻结结果写实验。
- [ ] 重写引言、摘要和结论。
- [ ] 更新图表和附录。
- [ ] 补文献与体例。

**出口：**不含占位符、内部标记和未支撑主张的完整稿。

### 阶段 D8：完整性与复审

- [ ] 数字与结果文件逐项对照。
- [ ] 引用存在性与支持关系检查。
- [ ] 主张强度检查。
- [ ] LaTeX 编译与视觉检查。
- [ ] 再做一次独立审稿模拟。
- [ ] 对照本计划逐项关闭。

**出口：**可投稿版本与修订记录。

---

## 17. 优先级与投稿阻断项

### P0：不完成则不能投稿

- [ ] X2 未来观测泄漏修复并重跑。
- [ ] Agent-VQA 题库无 GT/事件泄漏。
- [ ] 相同题目比较使用配对统计。
- [ ] 端到端回答区分答案、证据、到达和错误状态。
- [ ] 正式结果有功效和事件层分析。
- [ ] 论文不再用 2 个事件支撑一般性跨事件结论。
- [ ] 所有数字可追溯到冻结结果。
- [ ] 英文题名、摘要、关键词和投稿体例齐全。
- [ ] PDF 无占位符、缺图、出栏和内部审稿标记。

### P1：Agent-VQA 主线成立所必需

- [ ] 四类可评分问题。
- [ ] 结构化问答 schema。
- [ ] 静态、结构化和主动三层实验。
- [ ] Headless Agent-VQA benchmark。
- [ ] 前后端完整演示轨迹。
- [ ] Grounded Answer Success。
- [ ] 风险覆盖和成本指标。
- [ ] 退化恢复实验。

### P2：增强项

- [ ] 多图 VQA。
- [ ] 第二 VLM 骨干敏感性。
- [ ] 记忆重复问答。
- [ ] 自由文本报告人工评分。
- [ ] 真实 UAV 或厘米级低空数据。
- [ ] 正式人因研究。

---

## 18. 最终验收清单

### 18.1 科学问题

- [ ] 摘要、引言、方法、实验和结论围绕同一组 RQ。
- [ ] VQA 是智能体证据闭环的一部分，不是孤立上传功能。
- [ ] 重观测是动作机制，不再独占整篇论文主题。
- [ ] 负结果与不可识别结果被正确区分。

### 18.2 数据与实验

- [ ] 每道主问题都能由数据标注客观评分。
- [ ] train/val/test 事件无交集。
- [ ] 在线策略不读取未来观测和 GT。
- [ ] 同题策略比较配对。
- [ ] 事件内相关性被处理。
- [ ] 正式样本量与功效声明一致。
- [ ] 所有实验配置有完整 manifest。

### 18.3 系统实现

- [ ] 问题可触发搜索、感知、回答和重观测。
- [ ] 输出为合法结构化 schema。
- [ ] 前端展示结构化动作理由而非思维链。
- [ ] fallback 和失败状态可审计。
- [ ] headless 与前端调用共享同一核心逻辑。

### 18.4 统计与结果

- [ ] Accuracy/F1 之外有校准和风险覆盖。
- [ ] 有成本和有害重观测指标。
- [ ] 有配对 CI 和多重比较处理。
- [ ] oracle 只作参照。
- [ ] 不显著不被写成无效。

### 18.5 写作与格式

- [ ] 无审稿编号和内部文件名。
- [ ] 无绝对本机路径。
- [ ] 无 `n/a`、TODO 和待补截图。
- [ ] 中英文信息完整。
- [ ] 图表不出栏。
- [ ] 引用真实且支持对应主张。
- [ ] 结论数字与表格完全一致。

---

## 19. 预期新增文件与产物

### 19.1 代码

```text
backend/agent_vqa.py
backend/tests/test_agent_vqa.py
backend/tests/test_agent_vqa_schema.py
scripts/benchmarks/gen_agent_vqa_testset.py
scripts/benchmarks/review_agent_vqa_testset.py
scripts/benchmarks/render_agent_vqa_review_sheet.py
scripts/benchmarks/bench_agent_vqa.py
scripts/benchmarks/report_agent_vqa.py
scripts/benchmarks/export_agent_vqa_assets.py
```

### 19.2 数据与结果

```text
backend/data/benchmarks/agent_vqa_testset.json
runs/benchmarks/cja_agent_vqa/manifest.json
runs/benchmarks/cja_agent_vqa/reports/aggregate.json
runs/benchmarks/cja_agent_vqa/reports/paired_tests.json
runs/benchmarks/cja_agent_vqa/reports/event_breakdown.csv
runs/benchmarks/cja_agent_vqa/reports/failure_taxonomy.csv
```

### 19.3 论文资产

```text
paper_cja/generated/agent_vqa_dataset_table.tex
paper_cja/generated/agent_vqa_static_table.tex
paper_cja/generated/agent_vqa_active_table.tex
paper_cja/generated/agent_vqa_e2e_table.tex
paper_cja/generated/agent_vqa_risk_coverage.pdf
paper_cja/generated/agent_vqa_budget_curve.pdf
paper_cja/generated/agent_vqa_oracle_ladder.pdf
paper_cja/figures/ui_overview.png
paper_cja/figures/agent_vqa_trace.png
```

---

## 20. 与 `REVISION_EXPERIMENTS.md` 的关系

`REVISION_EXPERIMENTS.md` 中以下任务继续有效，并被本计划吸收为前置：

- 19 事件 LOEO。
- 多 seed。
- X2 Brier/NLL/风险覆盖/代价敏感重分析。
- 224 px 裁块对照。
- 可选 ResNet50 + focal loss。
- 清理审稿标记、oracle 命名、投稿体例和参考文献。

新增 Agent-VQA 不得被用来跳过这些问题。正确关系是：

```text
感知底座和统计协议可信
→ 静态 VQA 可解释
→ 主动 VQA 可识别
→ 端到端智能体结论才成立
```

执行时，两个文档采用同一结果账本和同一论文数字来源，避免旧实验与 Agent-VQA 实验分别维护相互冲突的数字。
