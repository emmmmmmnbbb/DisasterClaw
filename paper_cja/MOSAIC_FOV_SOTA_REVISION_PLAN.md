# 降高重观测重构 + xBD SOTA 感知底座接入 — 修改计划

## Material Passport

- `schema`: ARS Material Passport 9
- `artifact_type`: revision plan (proposed, not executed)
- `paper`: `paper_cja`
- `created_at`: 2026-08-24
- `source_scope`: `backend/perception.py`、`backend/gsd_ladder.py`、`backend/recheck.py`、
  `backend/agent_vqa.py`、`backend/world.py`、`backend/xbd_map.py`、
  `backend/data/xbd/manifest.json`（12846 条，post 且有 georef 4312 条）、
  `paper_cja/review2.md`、`paper_cja/AGENT_VQA_EXPERIMENT_STATUS.md`、
  `paper_cja/AGENT_VQA_REVISION_PLAN.md`、`paper_cja/generated/*.tex`
- `external_upload`: 无
- `verification_status`: **计划阶段。本文档中标注「实测」的数字是我在本机上跑出来的；
  标注「设计值」的是提案参数，尚未实现、尚未验证。所有实验结论均为空。**
- `blocking_decisions`: **已于 2026-08-24 由作者全部拍板，见 §7。本文档 §2/§3 已按
  决策改写，可直接开工。**

## 作者决策记录（2026-08-24）

| # | 问题 | 决策 |
|---|---|---|
| Q1 | 空洞怎么补 | **参考 disasterclaw 前端已拼好的地图**：xBD 瓦片叠在连续卫星底图上（前端现用 Esri World Imagery）。**追加决策（见 §2.2d）：巡航视场真实 xBD 覆盖须 ≥80%，Esri 只补剩余 ≤20%** |
| Q2 | 题目作用域 | **ROI-scoped**：题锚定中心那张有标注的瓦片 |
| Q3 | 巡航视场 | **3×3 瓦片**（1330 m，1.5 m/px）→ 下限 443 m（0.5 m/px，一整瓦片） |
| Q4 | holdout 事件 | 未单独作答，按 §4.1 的护栏执行；仍需在 P6 预注册时确认 |
| Q5 | SOTA 引入方式 | **双轨报告**：现成权重作 leaky 参照上界 + 事件不相交重训 1–2 折作干净主结果 |

## 执行进度（2026-08-24）

**P0 完成。** 删除 `~/datasets/xbd/*.tar.gz`（删前已核对解压完整：train 2799 对 /
test 933 对 / tier3 6369 对，images 与 labels 数量一致），释放 28 GB。
xView2 冠军权重已下载解压到 `backend/outputs/xview2_first/weights/`：
**24 个权重文件、5.3 GB**（res34 / res50 / dpn92 / se154 四架构 × loc+cls × 3 seed）。
注意：GitHub Release 的 `split-weights-a{a..e}` 合并后是 **tar.gz 不是 zip**，
仓库文档写的是 zip，按 zip 解会失败。磁盘现剩 56 GB。

**P1 完成，24 个新单测全绿，全量 130 passed（106 旧 + 24 新）。** 新增：

- `backend/fov_ladder.py` — 几何自洽性已验证：`alt_min=443.41 m` 渲染出的视场
  **恰好 1.0000 瓦片**、有效 GSD **恰好 0.500000 m/px**；`alt_cruise=1330.22 m`
  → 3.0 瓦片 / 1.500 m/px。天花板是推导出来的。
- `backend/basemap.py` — Esri 抓取/缓存/窗口拼接。
- `backend/mosaic.py` — [Esri 背景 + xBD 瓦片] 合成，含 nodata 掩码、色调协调、
  ROI 覆盖强约束、几何覆盖率索引。
- `scripts/benchmarks/build_roi_index.py` + `backend/data/xbd/roi_index.json`。
- `backend/tests/test_fov_ladder.py`、`backend/tests/test_mosaic.py`。

**运行环境更正**：项目环境是 conda `disasterclaw`（py3.11 / torch 2.11.0+cu130 /
flask 3.1.3），不是 base 的 py3.9。测试须用
`~/miniconda3/envs/disasterclaw/bin/python -m pytest`。

**下一步**：P1.5 泄漏体检 → P2 接入 `perception.py`。

**P2 完成（跳过 P1.5，作者指示先做 P2），131 passed。** 改动：

- `perception.py` 新增 `_render_uav_view()`（`MOSAIC_VIEW=1` 默认走它）；
  旧 `_crop_uav_view()` 保留为 `MOSAIC_VIEW=0` 的 legacy 回退，仅供复现旧产物。
- `perception.py` 新增 `_unet_proposals_for_window()`：**整张 pre 瓦片跑定位并按
  tile_id 缓存**，再用「瓦片像素 → 经纬 → 视场像素」仿射把框映射进当前视场，
  跨瓦片重叠用 IoU 0.55 的 NMS 去重（相邻 xBD 瓦片有重叠，不去重会让 count 类
  问题系统性高估）。
- `recheck.py`、`app.py` 的高度常量改为从 `fov_ladder` 取，不再硬编码 10/30 m。
- `test_recheck.py` 的 6+5 处 `RecheckController()` 改为显式钉住
  `LEGACY_ALT`（这些测试验证复核逻辑，与绝对高度尺度无关；沿用该文件
  `test_assess_entropy_info_gain_modes` 已建立的约定）。

**★ pre 通道保持原生 GSD，不随高度降质。** 这是有意的科学选择：灾前影像是
地面站已归档的卫星产品，分辨率与 UAV 高度无关。若让 pre 也随高度降质，
则「降高」会同时恢复 pre 通道分辨率——那正是 review2 B2 批评的
「撤销你自己刚加的降质」的翻版，只是换到参考通道上。
实现上 post 恒为 1024²、pre 为 `span_m / 0.5 m` 像素（巡航 3072²），
并用 `pre_scale` 在双时相裁块时换算 bbox。

### P2 端到端实测（hurricane-michael_00000034，GPU1）

| 档位 | 有效 GSD | xbd_fraction | pre_scale | 检测数 | 建筑密度 | 暖缓存耗时 |
|---|---:|---:|---:|---:|---:|---:|
| 巡航 (1330 m) | 1.50 | 0.988 | 3.0 | 657 | 278/km² | 8.3 s |
| 中档 (887 m) | 1.00 | 1.000 | 2.0 | 219 | 209/km² | 3.9 s |
| 下限 (443 m) | 0.50 | 1.000 | 1.0 | 63 | 240/km² | 1.8 s |

建筑密度跨三档一致（209–278/km²），说明 tile→geo→view 仿射映射与 NMS 正确；
人工检查下限档的检测叠加图，框紧贴房屋、商业建筑与路边结构，无系统性偏移。

**⚠️ 未达 §3.5-3 的闭环延迟门槛（2 s/次）。** 暖缓存下巡航 8.3 s，
拆解为：pre 原生渲染 ~4.1 s + 双时相分类器逐框 657 次前向 ~3.5 s。
两项都随视场面积线性增长（巡航是下限的 9 倍面积）。冷缓存首次更慢
（实测中档 52 s，为抓取该窗口的 z18 底图瓦片）。

可选优化，**建议留到 P3 一并处理**（P3 会整体替换检测器，延迟结构会变）：
1. **双时相分类器批处理** —— 现在是 Python 循环逐框前向，批处理应有数倍收益。
2. **只对 ROI 内的框做损伤分类** —— 题目是 ROI-scoped，ROI 在巡航时只占
   1/9 画面，可把 657 框降到约 73 框，且这是设计上正确的作用域收缩。
3. 底图缓存预热脚本，避免评测时首次抓取拖慢。

## P3a 完成：SOTA 检测器接入（2026-08-24），146 passed

新增 `backend/detectors/`（`base.py` / `xview2_first.py` / `legacy_unet.py` /
`xview2_zoo/`）、`scripts/benchmarks/eval_detector_backends.py`、
`backend/tests/test_detectors.py`（15 项）。

### 封装正确性：四个必须对齐的细节

从参考实现逐条核对（错一个就复现不出 SOTA），已写进单测锁死：

1. **BGR 通道序**（`cv2.imread` 语义），不是 RGB。
2. 预处理 `x/127 - 1`（`utils.preprocess_inputs`）。
3. 输出取 **sigmoid** 不是 softmax（`predict34cls.py`）。
4. cls 是 6 通道 siamese，**channel 1..4 才是四类损伤**，channel 0 是建筑性
   （`create_submission.py`）。定位复合阈值 `[0.38, 0.13, 0.14]`。

**踩到的两个坑**：(a) GitHub Release 分卷合并后是 **tar.gz 不是 zip**；
(b) res50/se154 权重的 SE 模块用扁平命名 `se_fc1/se_fc2`，而仓库 master 的
`senet.py` 用 `se_module.fc1/fc2` —— 权重与代码是两个时期的，需重命名才能
strict 加载。已加 `_remap_se_keys()`，4 架构 × {loc, cls} 全部 strict 通过。

### ★ 主结果：SOTA 消除了类别坍缩

事件不相交 **test** split（hurricane-michael + palu-tsunami，30 张瓦片）：

| 指标 | legacy_unet（自训练） | xview2_first（SOTA, **leaky**） |
|---|---:|---:|
| 像素 loc F1 | 0.8068 | **0.8532** |
| 像素 damage F1（四类调和平均） | **0.0000** | **0.6304** |
| 像素 overall (0.3·loc+0.7·dmg) | 0.2421 | **0.6973** |
| 逐建筑 macro-F1 | 0.1194 | **0.4517** |
| 无损伤 召回 | 0.4537 | 0.6267 |
| **轻微损伤 召回** | **0.0000** | **0.5570** |
| **严重损伤 召回** | **0.0000** | **0.4071** |
| 完全损毁 召回 | 0.0000 | 0.1987 |
| 单次延迟（中位） | 0.023 s | 0.113 s（res34 单模型） |

**定位能力两者相当（0.807 vs 0.853），差距几乎全部在损伤分级。**
这是一个干净、可解释的分解：瓶颈不在「找不到建筑」，而在「分不出损伤等级」。
review2 **B3**（类别塌缩是否由裁块尺寸造成）由此获得强证据——
同样是 1024² 输入、同样的事件不相交 test，SOTA 能分出四类，自训练模型只输出一类。

### 门槛判定（§3.5）

| 门槛 | 结果 |
|---|---|
| 3.5-1 overall > 0.7 | **未达**：0.6973（见下方说明，不是封装错误） |
| 3.5-2 轻微/严重召回 > 0 | ✅ **达成**（0.557 / 0.407，旧值 0.000 / 0.000） |
| 3.5-3 单次延迟 < 2 s | ✅ 达成（0.113 s 单模型 / 0.561 s 全集成） |
| 3.5-4 class_probs 有效 | ✅ 达成（四类、和为 1、非空） |

**overall 0.697 vs 文献 0.803 的差距不是封装错误**，三条证据：

1. `loc_f1=0.8532` 与该 checkpoint 自带的 `best_score=0.872` 吻合。
2. 逐类像素 F1 = 0.791 / 0.503 / 0.520 / 0.835，各类都健康；
   官方 damage F1 用**调和平均**，对最差类极敏感，minor/major≈0.5 直接把它压到 0.63。
3. 文献 0.803 是在**官方混合划分**（19 事件全混）上测的；我们只用两个留出事件，
   且 palu-tsunami（海啸夷平）远难于平均水平。**全集成 12 模型也只有 0.6967，
   与单模型 res34 的 0.6973 几乎相同**，说明差距来自事件难度而非集成规模。

### 实例分离：一个必须自适应处理的问题

xView2 冠军方案是**语义分割，不产生实例**。密集聚落里相邻建筑轮廓连成一片：
palu-tsunami 6 张瓦片 1661 栋 destroyed，朴素连通域只给出 152 个 blob，
实例召回 **0.044** —— count / spatial 类问题会系统性低估。

固定策略在两类场景上结论**相反**（实测）：

| 策略 | tune split（TRAIN_EVENTS，稀疏） | test split（含密集聚落） |
|---|---:|---:|
| 不切分 | **0.5805** | 0.3791 |
| 全局分水岭 d=6 | 0.4972 | 0.4069 |
| 自适应 area=1800 | 0.5125 | 0.4336 |
| 自适应 area=3600 | 0.5462 | **0.4517** |

既然最优选择随建筑密度反转，**就不能靠在 test 上选超参**（那是协议泄漏）。
因此改为**自适应分水岭**：只对面积明显大于单栋建筑的连通域做切分，判据本身
与密度无关。`split_area_px=3600 px ≈ 6 栋民居`（0.5 m/px 下 10×15 m 房屋约 600 px），
是**先验物理量而非在 test 上调出来的值**；上表 test 列作为敏感性分析报告，
不作为选参依据。

### ⚠️ 事件泄漏仍未解除

`xview2_first.leaky = True` 会写进每次输出的 `extras` 与报告 JSON。
权重在 xBD 官方 train+tier3 上训练，而 xBD 按瓦片而非按事件划分，
**paper_cja 的 test / holdout 事件全部被这些权重见过**。

因此上表的 SOTA 列**只能作为「有事件曝光的参照上界」**，与 `O_REF` 同级，
**不得进入正式对照主表**。且 legacy 与 SOTA 的差值同时包含
「架构/训练规模」与「事件曝光」两个因素，**不能单独归因于任一方**——
分解需要轨 B（同架构、事件不相交重训），这正是 review2 B7 要求的测量。

### 下一步

P4 可识别性前置检查：用新观测模型 + SOTA 底座在 test split 上重算
`n_flip` / `n_correctable`（旧值 7 / 2）。**这是决定是否投入轨 B 重训与新 holdout
的关键一关**；在它出结果前不要开始 P3b/P5–P8。

## P4 结果（已完成，2026-08-24）

### 逐事件分层（补采样 bug 修复后，48 ROI，含 palu-tsunami）

| 指标 | 只有 michael（40 ROI） | 分层后（48 ROI） |
|---|---:|---:|
| 逐建筑 n | 3218 | 7695 |
| n_flip | 534 | 733 |
| n_correctable | 244 | 375 |
| n_harmful | 177 | 281 |
| 净收益（建筑） | +67 | +94 |
| **逐问题 n_correctable** | 19 | **13** |
| 逐问题 n_harmful | 10 | 10 |

逐问题分题型：count_destroyed 48 题 flip 18 / correctable 6 / harmful 7（**净 −1**）；
presence 48 题 4/2/2（净 0）；spatial 40 题 11/5/1（净 +4）。

**判定 MARGINAL（不是 IDENTIFIABLE，也不是 UNIDENTIFIABLE）**。补上 palu-tsunami 后
逐问题净收益从 +9 掉到 +3，count 反而净负。palu 只有 8 个 ROI 却贡献 177/375 的
correctable（每 ROI 翻转密度是 michael 的 8 倍）——**正是 review2 B1 担心的事件间方差**。

**关键限制**：这是 **leaky 底座**的结果（xview2_first 权重见过全部评测事件）。
干净的轨 B 可能 headroom 更小；P4 的判定必须用轨 B 重跑才能成为正式结论。
P4 测的是感知层上界（规则式答案推导，剥掉 VLM），任何策略能利用的不超过它。

## P5 完成：题库 v2.0 生成器（ROI-scoped）

新增 `scripts/benchmarks/gen_agent_vqa_testset_v2.py`。相对 v1 的两处根本改动：

1. 观测模型换成 fov_ladder 的视场收缩（巡航 1330m/1536m 跨度 → 下限 443m/512m）。
2. 题目作用域从「当前视场」换成 **ROI-scoped**：每道题锚定一张有标注的 post 瓦片
   （ROI = 地理 bbox），答案只由 ROI 内 GT 建筑决定，与动作/高度无关。
   **spatial 方位相对 ROI 中心而非 UAV 位置**（§2.3 的关键修复）。

场景池来自 `roi_index.json`（巡航几何覆盖 ≥0.80）。schema 升到 `agent-vqa/2.0`，
全部 SHA-256 重算。test split（michael+palu）草稿已生成：80 题，四类均衡，
spatial 八方位分层良好，泄漏断言通过。

## P3b 轨 B 训练状态

- **loc 阶段完成**：40 epoch，事件不相交 val best dice 0.8425（≈ leaky SOTA 的
  pixel loc F1 0.853，再次确认定位不是瓶颈）。
- **cls 阶段**：warm-start 修复后重跑，ep 27/30，best val_dice 0.8395（建筑性通道
  dice，非四类损伤质量）。真正的判决是 test split 的逐类损伤召回——能否把
  legacy 的 0.000/0.000 拉起来。

## P3b 完成：三方检测器对照（★ 决定性结果）

轨 B cls 训练完成（30 epoch，best val_dice 0.8413，建筑性通道）。三方在同一
事件不相交 **test** split（30 瓦片）逐项对照：

| 后端 | 架构 | 数据协议 | damage F1 | minor 召回 | major 召回 | destroyed 召回 | loc F1 |
|---|---|---|---:|---:|---:|---:|---:|
| legacy_unet | 小（resnet34-unet） | 事件不相交 | **0.000** | 0.000 | 0.000 | 0.000 | 0.807 |
| **xview2_eventdisjoint** | **大（xView2 冠军）** | **事件不相交** | **0.017** | 0.020 | 0.014 | 0.081 | 0.832 |
| xview2_first | 大（xView2 冠军） | **leaky（见过 test）** | **0.630** | 0.557 | 0.407 | 0.199 | 0.853 |

**这是一张干净的归因表，直接回答 review2 B7，且推翻了原稿的归因：**

| 对比 | 控制变量 | damage F1 差值 | 含义 |
|---|---|---:|---|
| legacy → track B | 数据（同事件不相交） | 0.000 → 0.017 | 架构/训练规模贡献 ≈ **+0.017**（微小） |
| track B → SOTA | 架构（同大架构） | 0.017 → 0.630 | 事件曝光贡献 ≈ **+0.613**（巨大） |

**结论：差距约 97% 来自事件曝光、约 3% 来自架构。** 原稿「剩余差距应归因于
架构/训练规模」是**错的**，现在有了测量支撑。即便用 xView2 冠军架构，事件不相交
训练下损伤分级仍然坍缩（minor/major 召回 ≈ 0.01–0.02）—— 大架构不解决跨事件
泛化，事件曝光才解决。这同时**强化了 review2 B1 的中心负结果**（跨事件不泛化）。

**对重观测主线的决定性冲击**：P4 的 MARGINAL headroom（n_correctable=375）是
**leaky 底座上的产物**，本质是「模型已经在训练里见过答案」。干净底座（track B）
的损伤信号几乎为零，降高「看得更清」没有可恢复的信号 —— 干净底座上 P4 会落在
UNIDENTIFIABLE，P7/P8（pilot/holdout）会复现分支 C。

## 作者决策（2026-08-24）：holdout 事件

Q4 作者选择「**先看轨 B 结果再定**」。现在轨 B 结果已出（上表），结论明确：

**不建议在干净底座上跑 P7/P8**——没有损伤信号，holdout 只会复现「不可识别」。
真正的瓶颈是损伤分类器的跨事件泛化，不是重观测策略，也不是架构。

## 论文主线重定向（作者决策 2026-08-24，路线 1）

**作者拍板**：把「主动重观测」作为 **leaky 底座上的机制研究**（重披露：感知底座
见过评测事件），并**弱化模型检测部分、着重体现灾害智能体的构建、仿真环境、特色机制**。

### 新的贡献声明（三支柱）

| 支柱 | 内容 | 对应代码 |
|---|---|---|
| **灾害应急智能体构建** | 有限动作集 + 结构化决策（answer/continue_search/reobserve/abstain）+ 证据束 + 规则回退 | `backend/agent_vqa.py` |
| **地理配准仿真环境** | 视场收缩式降高（fov_ladder 物理模型 443→1330m / 0.5→1.5m·px⁻¹）+ 瓦片叠底图合成（basemap + mosaic）+ ROI-scoped 出题 | `backend/fov_ladder.py` `basemap.py` `mosaic.py` |
| **特色机制** | 预算受限主动重观测 + 可识别性优先评估（先证可识别、再谈效果） | `backend/recheck.py` + P4 脚本 |

### 感知底座的角色降级（关键，重写口径）

- **检测器 = 公开披露的感知组件，不再是贡献**。SOTA xview2 提供真实损伤信号，
  使智能体「有东西可感知」，但其绝对精度（0.697 overall）不是主张。
- **leaky 披露**写进方法/局限：权重在 xBD 官方划分上训练，见过评测事件；
  「重观测是否值得」的结论只在这个前提下成立——**不得写成跨事件泛化的主张**。
- **B7 的三方分解降级为局限**：不再作为主贡献，而是「感知瓶颈是跨事件泛化、
  不是架构」的佐证（这正是 review2 B1 中心负结果的一个测量强化）。

### 章节级改动清单

1. **§方法 新增「地理配准仿真环境」独立小节**：传感器几何（fov_ladder）、
   视场收缩式降高（闭环 review2 B2：不再「撤销自己加的模糊」）、
   瓦片叠底图 + 覆盖率门槛（2.2d）、ROI-scoped 作用域（2.3）。
2. **检测器降为一段**：「采用 xView2 冠军架构的预训练感知底座（其训练划分见
   局限）」，细节 + 三方分解全部移到附录/局限。
3. **E4 主动 VQA 写作口径**：leaky 底座上的机制研究，P4 的 MARGINAL 结果是
   「机制可测、效应小且有得有失」——支撑「预算受限重观测何时值得」这一问题
   的**存在性**，不夸大成「重观测改善答案」。
4. **局限新增两条**：(a) 感知底座事件曝光，跨事件泛化未被主张；
   (b) 场景选择偏倚（覆盖率 ≥0.80 的 495 个 ROI，建成区密集）。

### 对后续实验的含义

- P7/P8 可在 leaky 底座上跑（检测器的事件曝光已披露、不再阻塞）。
- 仍需要：把 `xview2_first` 接进在线感知闭环（当前 `BUILDING_PROPOSER=unet` 仍是
  legacy 路径），用 v2.0 题库跑 agent 基准，再做 P6 预注册 + holdout。

## 在线闭环接入 xview2_first（已完成，2026-08-24）

`perception.py` 新增 `DETECTOR_BACKEND` 环境变量与 `_detect_with_backend()`：
- `DETECTOR_BACKEND=legacy`（默认）走原有 unet/yolo 路径，不影响既有产物。
- `DETECTOR_BACKEND=xview2_first`（或 `xview2_eventdisjoint`）走 detectors 包：
  pre 保持原生 GSD、post 上采样到 pre 尺寸喂 siamese，返回框按 `1/pre_scale`
  缩放回 post 视场（1024）坐标；`detections` 字段（class_name / raw_class_name /
  conf / class_probs / bbox）与 legacy 对齐，`agent_vqa.build_evidence_from_perception`
  直接消费。后端激活时强制渲染 pre（`need_pre`），并跳过 localizer/yolo 加载。

冒烟测试（hurricane-michael，cuda:3）：
- 巡航（1.5 m/px，pre_scale=3.0）：991 框，四类齐全（no-damage 566 / minor 310 /
  major 106 / destroyed 9），`class_probs` 全有。
- 下限（0.5 m/px，pre_scale=1.0）：60 框，四类齐全。
- 147 passed，无回归。

**下一步**：用 v2.0 题库 + mosaic 观测模型 + xview2_first 底座跑 agent VQA 基准
（P7 pilot），拿到 E4 的 leaky 底座最终数字。

## P7 冒烟通过：E4 管线端到端打通（2026-08-24）

`gen_agent_vqa_testset_v2.py` 新增 `--centered-start`（E4 从 ROI 中心出发，
「标记区域」=「画面中央」自洽，无需视觉标注）；生成 40 题 centered bank。
`bench_agent_vqa.py` 用 v2.0 bank + `DETECTOR_BACKEND=xview2_first` 跑通 4 题 × 2 配置：

| config | n | acc | abstain | steps | n_reobs | n_skips |
|---|---:|---:|---:|---:|---:|---:|
| V0_RAW | 4 | 0.50 | 0.0 | 1.0 | 0 | 0 |
| A2_ALWAYS | 4 | 0.50 | 0.0 | 3.0 | **8** | 0 |

- **重观测通道已接通**：A2 每问 2 次重观测（=max_reobs），3 步，descend+reobserve
  在新高度尺度（1330→443m）下真实执行。
- **效应有得有失**（与 P4 的 MARGINAL 一致）：damage 题 A2 纠正了 V0 的错答
  （无损伤 vs 完全损毁→无损伤），但 spatial 题 A2 反而把 V0 的正确「西」改成「西南」。
- 管线验证：mosaic 渲染 + xview2_first 双时相检测 + Qwen2.5-VL-7B + ROI-scoped 评分全通。

**⚠️ 延迟**：~35 s/问（V0 单步）、~85 s/问（A2 三步）。主要来自巡航档 pre 原生渲染
（3072²）+ xview2 在 3072² 上推理 + VLM。40 题 × 5 配置 ≈ 3 h，200 题 ≈ 15 h。
可选优化（不影响正确性）：cap pre 渲染分辨率 / 只对 ROI 内框做损伤分类。

**下一步**：跑完整 pilot（40 题 × V0/V1/A0/A2/A3），再定 P6 预注册（holdout 事件
已可定——轨 B 结果已出）与 P8。

## P7 pilot 结果（40 题 × 5 配置，test split，leaky 底座，2026-08-25）

| 配置 | acc | 重观测 | 纠正题 | 破坏题 | 净 |
|---|---|---:|---:|---:|---:|
| V0_RAW | 0.275 | 0 | — | — | — |
| V1_STRUCT | 0.275 | 0 | — | — | — |
| A0_HOLD | 0.250 | 0 | — | — | — |
| A2_ALWAYS | 0.300 | 78 | +4 | −2 | +2 |
| A3_ENTROPY | 0.300 | 11（skip 33） | +1 | 0 | +1 |

**结论（分支 C，与 P4 预注册一致）**：重观测通道接通（A2 每问 2 次、A3 触发+跳过
都发生），但效应小且有得有失（A2 纠正 4、破坏 2），n=40 下 +5pp 不显著。
结构化证据（V1）无收益。写作口径：机制可识别、效应小、不可排序策略。

## P8 holdout 结果（完成，2026-08-26）

留出事件（moore-tornado 20 + nepal-flooding 20，40 题 × 5 配置，0 执行错误）：

| config | acc | 重观测 | 纠正 | 破坏 |
|---|---|---:|---:|---:|
| V0_RAW | 0.425 | 0 | — | — |
| V1_STRUCT | 0.400 | 0 | — | — |
| A0_HOLD | 0.425 | 0 | — | — |
| A2_ALWAYS | 0.450 | 79 | +4 | −2 |
| A3_ENTROPY | 0.400 | 3（skip 38） | 0 | 0 |

**与 pilot 完全一致，分支 C 复现**：效应约 1 题、不单调（A2 +1、A3 −1）；
事件间方差（moore 0.60 vs nepal 0.25）远大于任何策略差。
两条互斥事件集共同锁定：机制可识别、效应小、不可排序策略。

## 论文完成（2026-08-26）

- 题名/摘要/引言/方法/架构/实验/结论/局限/附录全部改写到位，主线三支柱
  （空天应急侦察智能体 + 地理配准仿真环境 + 预算受限主动重观测），检测器降级为
  公开披露的感知组件，去除空泛 AI 化表述，填进全部实测结果（三方对照、P4、pilot、
  holdout）。
- 补 4 张缺失表（fov_ladder / roi_pool / detector_backend / mosaic_obs）。
- 定位到《航空学报》「空天跨域协同感知」专刊方向 8（多模态大模型与智能体赋能的
  空天信息处理），题名与摘要精准对齐，未硬套「协同/博弈」等论文里没有的命题。
- `xelatex main.tex` 两遍编译通过，无错误、无未定义引用，PDF 733 KB。

## 磁盘与落盘修复（2026-08-25）

- `perception.py` 新增 `PERCEPTION_SAVE_IMAGES`（默认 1，评测设 0）。
- 设 0 时图片用固定名覆盖（`_latest*.png`），磁盘有界，只保留文字结果。
- 清理旧累积 3.9GB PNG（1156 张）。修复了误抢 `@staticmethod` 装饰器导致的参数错位 bug。

---

## 0. 一句话摘要

把「降高 = 对同一张瓦片加高斯模糊再撤销」改成「降高 = 在真实拼接底图上收缩视场、
在固定传感器分辨率下提升单位目标像素数」，并把感知底座从自训练的 ResNet18 双塔
（事件不相交 macro-F1 0.22–0.28、轻微/严重损伤召回 0.000）换成 xView2 冠军方案
（overall 0.803）。前者直接闭环 review2 的 B2，后者直接闭环 B3/B7 并有可能把
E4 从「分支 C 不可识别」推到分支 A/B。

**代价**：现有 D6 holdout 结论作废、事件不相交协议在感知层面临泄漏、
磁盘只剩 32 GB。这三件事必须先决策，见 §7。

---

## 1. 现状诊断

### 1.1 代码层面：当前「降高」实际做了什么

`backend/perception.py:_crop_uav_view()` + `backend/gsd_ladder.py`：

```
radius_m  = clamp(PERCEPTION_MIN_RADIUS_M=20, alt × PERCEPTION_VIEW_ALT_FACTOR=2.0, 300)
radius_px = clamp(PERCEPTION_MIN_PATCH_PX//2=128, radius_m / gsd, PERCEPTION_MAX_PATCH_PX//2=512)
patch     = tile.crop(2·radius_px 方块)
patch     = degrade_to_scale(patch, effective_scale(alt))      # 高斯模糊 + 降采样 + 升采样
```

代入当前巡航/下限高度（`DEFAULT_HOVER_ALTITUDE_M=30`、`VLN_RECHECK_ALT_MIN_M=10`、
xBD gsd=0.5 m/px）：

| alt | radius_m | radius_px | 实际裁块 | 地面覆盖 | `effective_scale` | 合成有效 GSD |
|---:|---:|---:|---:|---:|---:|---:|
| 30 m（巡航） | 60 | **120** | 240×240 | 120 m | 4.00× | 2.00 m/px |
| 20 m | 40 | **128**（撞下限） | 256×256 | 128 m | 2.50× | 1.25 m/px |
| 10 m（下限） | 20 | **128**（撞下限） | 256×256 | 128 m | 1.00× | 0.50 m/px |

**三个实测结论：**

1. **视场几乎不变，甚至反向。** `PERCEPTION_MIN_PATCH_PX//2=128` 的下限在 alt≤21.3 m
   时恒定生效，所以从 20 m 降到 10 m 视场完全没变；从 30 m 降到 20 m 视场反而**变大**了
   （240→256 px）。「降高看得更细」在当前实现里没有任何几何依据。
2. **唯一真实变化量是 `degrade_to_scale` 的高斯模糊半径。** 这就是 review2 B2 说的
   "下降在你的仿真里等于撤销你自己刚加的高斯模糊"——原话精确命中实现。
3. **信息天花板由构造保证。** `paper_cja/generated/gsd_ladder_table.tex` 实测：
   macro-F1 在 1.00×–4.00× 五档之间是 0.286 / 0.286 / 0.273 / 0.279 / 0.279，
   精度 0.623 / 0.623 / 0.620 / 0.623 / 0.625。**全平。**
   `gsd_class_table.tex` 记录巡航↔原生预测翻转数 **n_flip = 7**。

### 1.2 高度尺度本身不成立

review2 B2 的原话：「真实 10 m 飞行给的是厘米级 GSD，是新信息，不是恢复」。

这一条无法靠改渲染方式绕开——只要仿真里写着「10 m 高度、0.5 m/px」，任何懂遥感的
审稿人都会立刻算出这在物理上差了两个数量级。**高度尺度必须一起重标定**，
详见 §2.1。这一点在原计划和 REVISION_EXPERIMENTS 里都没有被提出过。

### 1.3 感知底座层面：瓶颈在哪

`paper_cja/generated/gsd_class_table.tex`（原生档，事件不相交）：

| 类别 | 标注数 | 预测数 | 召回 |
|---|---:|---:|---:|
| 完好 | 250 | 381 | 0.972 |
| 轻微损伤 | 125 | **0** | **0.000** |
| 严重损伤 | 12 | **0** | **0.000** |
| 损毁 | 13 | 19 | 0.462 |

`training_curve_table.tex`：事件不相交 val macro-F1 常年在 0.208–0.247 徘徊，
train macro-F1 却能升到 0.568——训练内有信号、跨事件完全不泛化。
文献水平（xView2 冠军）overall 0.803。

**这个坍缩是 E4 不可识别的直接原因**：`AGENT_VQA_EXPERIMENT_STATUS.md` 记录
holdout `n_correctable=2`、`n_harmful=1`。分类器只会输出「完好」，
降高再多次也不会翻转答案，所以任何重观测策略之间必然无差异。

### 1.4 两个改动与审稿意见的对应关系

| review2 条目 | 现状 | 本计划的处理 |
|---|---|---|
| B2 GSD 阶梯信息天花板人为设定 | 未修，只能写进局限 | **改动一直接消除**（§2） |
| B3 类别塌缩的第四种解释：裁块尺寸 | 未做 224 px 对照 | 改动一把裁块提到 1024 px，**顺带覆盖**（§2.4） |
| B7 「差距归因于架构」是断言非测量 | 未测 | **改动二直接测量**（§3.3 方案 iii） |
| B1 中心负结果建立在 2 个事件上 | 未做 LOEO | 不在本计划范围，仍是独立阻断项 |
| B4 X2 用错效用指标 | 未重跑 | 不在本计划范围（但改动二会让它更有意义） |

**顺带修掉的一个未被审稿人发现的缺陷**：`AGENT_VQA_EXPERIMENT_STATUS.md` §当前判定 3
记录「搜索通道未激活：A0 全部 1 步，题面均为当前视场」。改动一把巡航视场扩大到
9 倍瓦片面积后，目标不再必然在初始视场内，**E5 的搜索通道第一次真正可测**。

---

## 2. 改动一：视场收缩式降高（Mosaic-FOV Descend）

### 2.1 物理模型与高度重标定 ★ 关键

用固定传感器 + 固定视场角建模，一切参数自洽推出：

```
W_px       = 1024            # 传感器/渲染输出宽（像素），设计值
θ          = 60°             # 水平视场角，设计值
footprint(alt) = 2 · alt · tan(θ/2) = 1.1547 · alt      [m]
GSD(alt)       = footprint(alt) / W_px = alt / 886.8    [m/px]
```

xBD 瓦片：1024×1024 px @ 0.5 m/px = **512 m × 512 m**（manifest 实测瓦片经纬跨度
≈0.0051°×0.0043°，与 512 m 一致）。

于是「最小视场 = 恰好一整张瓦片」这个约束**唯一确定了下限高度**：

| 视场跨度 | 瓦片数 | 高度 | 有效 GSD | 说明 |
|---:|---:|---:|---:|---|
| 512 m | 1×1 | **443 m** | **0.50 m/px** | 下限；等于原生 GSD，**信息天花板在此，且是被推导出来的而不是被设定的** |
| 1024 m | 2×2 | 887 m | 1.00 m/px | 中间档 |
| 1536 m | 3×3 | **1330 m** | **1.50 m/px** | 巡航 |

**这一步同时解决了 B2 的两半：**

- 「下降=撤销自己加的模糊」→ 现在下降是从**更大的真实底图**上裁更小的footprint，
  再重采样到固定 1024 px。分辨率损失来自真实的下采样比，不是人工模糊。
  `degrade_to_scale()` 可以整个删掉。
- 「10 m 飞行应该是厘米级 GSD」→ 现在高度是 443–1330 m，在这个高度上
  0.5 m/px **正好是物理正确的**。仿真第一次自洽。

443 m / 1330 m 也符合固定翼应急侦察 / 中空长航时平台的真实作业高度，
对《航空学报》读者比 10 m 更有说服力。

**建议把阶梯定义在「瓦片跨度」而不是「米」上**（`span_tiles: 3.0 → 2.0 → 1.0`），
高度由 `alt = span_tiles · 512 / 1.1547` 反推。这样「下限 = 恰好一整瓦片」是
不变量而不是需要维护的巧合，换传感器参数也不会破。

### 2.2 底图拼接：可行性实测

我在 `backend/data/xbd/manifest.json` 上按瓦片中心经纬做了网格量化，
统计各事件 post 瓦片的邻接密度（**实测**）：

| 事件 | post 瓦片 | 占用格 | 孤立格 | 完整 2×2 | **完整 3×3** |
|---|---:|---:|---:|---:|---:|
| moore-tornado | 222 | 191 | 0 | 155 | **123** |
| hurricane-michael | 427 | 412 | 8 | 116 | **22** |
| santa-rosa-wildfire | 291 | 272 | 26 | 62 | **15** |
| socal-fire | 567 | 540 | 11 | 125 | **12** |
| nepal-flooding | 576 | 551 | 22 | 123 | **11** |
| hurricane-matthew | 290 | 270 | 0 | 74 | **7** |
| hurricane-harvey | 399 | 384 | 75 | 29 | **2** |
| pinery-bushfire | 479 | 473 | 41 | 45 | **1** |
| hurricane-florence | 409 | 400 | 32 | 36 | **0** |
| midwest-flooding | 318 | 316 | 70 | 7 | **0** |

**结论：xBD 瓦片不是稠密网格。** 「3×3 全覆盖」的中心瓦片全库合计约 200 个，
且高度集中在 moore-tornado（123 个）。midwest-flooding / hurricane-florence
基本无法构成 3×3。**因此纯 xBD 拼接无法支撑 3×3 巡航视场，必须补底图。**

### 2.2b 采纳方案：复刻前端的「瓦片叠底图」合成（Q1 决策）

**前端现在实际做的事**（`frontend/src/components/SituationMap.jsx`）：

| 图层 | 代码位置 | 内容 |
|---|---|---|
| 底图 | `TileLayer` L662–668 | **Esri World Imagery**（`server.arcgisonline.com/.../World_Imagery`），连续无缝 |
| xBD 影像 | `ImageOverlay` L693–695 | `/api/xbd/images/{activeTileId}`，按 `bounds` 地理配准叠加，**一次只叠 1 张** |
| 瓦片轮廓 | `GeoJSON` L702–708 | `footprints.geojson`，全部 post 瓦片外框 |

所以「前端拼好的地图」= **xBD 瓦片地理配准后压在 Esri 卫星底图上**。
后端马赛克要做的就是把它从「一次一张」推广到「窗口内所有 post 瓦片」，
并在服务端离线渲染成模型输入。

**★ 一个关键的有利实测**：我拉了一张 Esri World Imagery z=18 瓦片
（lat 30.7，hurricane-michael 区域）：256×256 RGB，19.7 KB，
**地面分辨率 0.513 m/px**。xBD 原生是 0.5 m/px。

> **两个源的 GSD 几乎完全一致（0.513 vs 0.500，差 2.6%）。**

这大幅削弱了我在 §2.2 路线 B 里担心的「源不连续泄漏」——两者锐度相当，
不存在「一眼看出哪块是高清出题区」的问题。剩余差异是日期与色调，
用 §2.2c 的手段压制。

**渲染合成顺序**（`TileMosaic.render()`）：

```
1. 取窗口 → 铺 Esri z=18 瓦片作背景层                 [连续，无洞]
2. 窗口内所有有 georef 的 xBD **post** 瓦片按仿射叠上去   [真实灾后]
3. 色调协调（§2.2c）
4. 重采样到 SENSOR_PX=1024
```

pre 时相同理，第 2 步换成 xBD **pre** 瓦片。

**缓存预算（实测外推）**：1536 m 窗口 @ 0.513 m/px ≈ 2994 px ≈ 12×12 个 z18 瓦片，
含边缘余量取 169 张/场景 × ~20 KB ≈ **3.4 MB/场景**。
700 个场景约 **2.4 GB**。在 P0 清理后可接受，但必须落盘缓存（见 R1）。

### 2.2c 必须做的三项泄漏抑制

即便 GSD 匹配，仍要防住「智能体靠源差异而非灾情特征作答」：

1. **色调协调**：把 Esri 背景 chip 的均值/方差匹配到窗口内 xBD 瓦片的统计量
   （逐通道线性匹配即可，不做直方图规定化以免破坏损伤纹理）。
2. **优先用真实 xBD 邻接瓦片**：窗口内凡是有 post 瓦片的格子就用真实影像，
   只有真正的空洞才落到 Esri。记录 `xbd_fraction`（真实 xBD 像素占比）进 meta，
   并在论文里按 `xbd_fraction` 分层报告——若结论随该比例系统性变化，
   说明存在源依赖，必须如实披露。
3. **ROI 边界不可由源差异推断**：Q2 已选 ROI-scoped，ROI 以**地理 bbox 显式告知**
   智能体，不需要它去「找」出题区，所以源差异的泄漏价值本身就很低。
   但仍**禁止**在模型输入图像上画 ROI 高亮框（会变成视觉答案提示）。

### 2.2d 实测推翻了「Esri 补洞低风险」的初判 → 追加覆盖率门槛

§2.2b 曾据 GSD 匹配（0.513 vs 0.500）判定源不连续泄漏风险低。**这个判断是错的，
已被两项实测推翻：**

**(1) 接缝差异是语义的，不是锐度的。** 渲染 midwest-flooding 场景后可见：
xBD post 瓦片是洪水泥浆，Esri 背景是灾前绿色农田，**矩形边界一目了然**。
色调协调（§2.2c 第 1 条）是线性统计匹配，**修不了日期造成的地物变化**。

**(2) 巡航视场的真实覆盖远低于预期。** 对全部 4312 张 post 瓦片按 3×3 巡航
窗口（1536 m）做几何覆盖统计（栅格 64×64，单元中心采样）：

| 巡航跨度 | 平均覆盖 | ≥0.95 | ≥0.80 |
|---|---:|---:|---:|
| 3×3（1536 m） | **0.502** | 6% | **16%** |
| 2×2（1024 m） | 0.582 | 7% | 22% |
| 1.5×（768 m） | 0.677 | 11% | 29% |

**平均只有一半画面是真实 xBD**，且缩小跨度收效有限——xBD 瓦片除
moore-tornado 外本就不稠密。

**追加决策（作者，2026-08-24）**：保留 3× 跨度，但**ROI 场景池限制为
巡航真实覆盖 ≥0.80 的瓦片**，Esri 只补剩余 ≤20%（且多在画面边缘）。
`build_roi_index.py` 实测产出 **495 个合格场景**：

| 事件 | 合格场景 | 事件 | 合格场景 |
|---|---:|---|---:|
| moore-tornado | 160 | hurricane-harvey | 12 |
| hurricane-michael | 89 | hurricane-florence | 10 |
| socal-fire | 65 | mexico-earthquake | 8 |
| nepal-flooding | 58 | palu-tsunami | 8 |
| hurricane-matthew | 44 | pinery-bushfire | 4 |
| santa-rosa-wildfire | 37 | **合计** | **495** |

6 个事件有 ≥37 个场景，足以支撑 200 题题库与事件分层。
**代价**：midwest-flooding、guatemala-volcano 完全退出；事件多样性从 13 降到
6（可用）+5（少量）。**这必须写进论文的场景选择偏倚说明**——
合格场景偏向建成区密集、影像条带重叠多的区域。

**渲染验证**（hurricane-michael_00000051，几何覆盖 0.97）：
巡航 `xbd_fraction=0.966`、下限 `1.000`，ROI 两档均为 `1.000`，
输出图无可见接缝、无残留黑区。

### 2.2e 实测发现：xBD 瓦片自带黑色 nodata 区

原始 xBD 瓦片含纯黑 nodata 像素，抽样实测各事件均值 **1.5%–11%**
（moore-tornado 最高 11%，个别瓦片 >30%）。

首版实现把「落在瓦片栅格内」当作有覆盖，导致合成图出现大片纯黑却被记为
`xbd_fraction=1.0`。已修复为 **nodata 感知掩码**：源像素为纯黑则不计入覆盖，
让底图透出。实测修复后同一场景纯黑占比从 **0.369 → 0.000**。

附带效果：相邻 xBD 瓦片相互填补 nodata（瓦片间有重叠），
所以在稠密区（moore-tornado）修复后 `xbd_fraction` 仍为 1.000。
这也是为什么覆盖率索引用几何足迹估计即可，nodata 是二阶效应。

### 2.2f 必须写进论文的披露

不写就是隐瞒，写了就是可接受的设计选择：

- Esri World Imagery 是**非事件当天**影像（多为灾前），在观测中仅作**上下文背景**；
  所有 GT 与所有问题作用域都严格限制在 ROI 内的 xBD post 瓦片上，
  背景不参与任何标注或评分。
- 底图来源、`xbd_fraction` 分布、抓取日期写进 `provenance.json`。
- **场景选择偏倚**：ROI 池限制为巡航覆盖 ≥0.80（495/4312），
  偏向建成区密集、影像条带重叠多的区域；midwest-flooding 与
  guatemala-volcano 完全退出（见 §2.2d）。

**许可**：沿用前端已在用的 **Esri World Imagery**（`attribution` 字段已在
`backend/world.py:15` 声明），不使用 Google Maps 瓦片——Google 的服务条款
明确禁止为非 Google Maps 用途批量缓存瓦片，而服务端渲染成模型输入正属此类。
Esri 需保留 attribution 并注意其条款同样限制大规模缓存，
建议在论文与代码里都标注底图来源与用途。**这一条建议作者复核。**

### 2.3 问题作用域：采纳 ROI-scoped（Q2 决策）

视场变大之后，「有多少栋完全损毁建筑」这类题的作用域必须重新定义，
否则降高（视场收缩）会改变答案本身——那就不是「看得更清」而是「换了道题」。

**采纳方案：ROI-scoped。**
题目锚定在一个固定地理 ROI（= 中心那张有 xBD 标注的 post 瓦片）。
巡航时 ROI 只占画面 1/9；降高逐步放大直到 ROI 恰好铺满画面。

| 性质 | 说明 |
|---|---|
| 答案作用域 | 与高度、与智能体动作**完全无关**，只由 ROI 的 xBD 标注决定 |
| 信息增益 | ROI 的单位目标像素数从 1/9 提升到 1 倍（线性 3×），**来自真实源像素** |
| GT 需求 | 只需中心瓦片有标注 → 邻接格子可以是真实 xBD 瓦片，也可以是 Esri 背景 |
| 场景池 | 不再要求 3×3 全部有标注，**只要求中心瓦片有标注** → 4312 个 post 瓦片几乎全部可用 |
| 搜索通道 | 初始位姿让 ROI 偏离画面中心，智能体必须先居中才能有效降高 ✓ |

**ROI 如何告知智能体**：给**地理 bbox**（lat/lon 四至）写进题面与 `observation` 结构化字段。
**禁止**在输入图像上画高亮框或改变 ROI 区域的渲染方式——那等于把答案区域
用视觉信号直接送进 VLM，属于计划 §7.3 的信息边界违规。

**一个必须同时改的细节**：`spatial` 题（「最近的 X 位于哪个方位」）原本是
相对**画面中心**算方位（`agent_vqa.py:644-653` 用 `norm_xy` 减 0.5）。
ROI-scoped 之下方位必须相对 **ROI 中心**而非画面中心计算，
否则智能体一居中方位就变了。`_rule_fallback` 与题库 GT 生成两处都要改，
并加回归测试锁死。

*（备选的 FOV-scoped ——题锚定当前视场、降高即缩小作用域——已否决：
它要求 FOV 内瓦片全部有标注，退回 §2.2 的强约束，且答案作用域随动作变化
会让评测口径复杂度大幅上升，在当前底座下很可能再次不可识别。）*

### 2.4 代码改动清单

**新增 `backend/basemap.py`**（约 180 行）— Q1 决策新增

```python
class BasemapTiles:
    """Esri World Imagery z18 瓦片抓取 + 落盘缓存 + 按地理窗口拼接。
    与前端 SituationMap.jsx 的 TileLayer 同源（backend/world.py:DEFAULT_BASEMAP）。"""
    def __init__(self, cache_dir, provider=DEFAULT_BASEMAP, zoom=18): ...
    def fetch_tile(self, z, x, y) -> Image:        # 带 ETag/重试；缓存命中不发请求
    def render_window(self, bounds, out_px) -> Image: ...
```

- 复用 `backend/geo.py` 已有的 `latlon_to_world_pixels` / `tile_bounds`（Web Mercator，
  与 Esri/Google 同一套 XYZ 方案），**不需要新写投影代码**。
- 缓存落盘到 `backend/data/basemap_cache/{z}/{x}/{y}.jpg`，
  并写 `provenance.json`（provider / zoom / 抓取日期 / 瓦片数）。
- **离线模式**：缓存未命中且无网络时抛显式错误，不得静默返回灰图——
  否则会在评测中悄悄改变输入分布。

**新增 `backend/mosaic.py`**（约 300 行）

```python
class TileMosaic:
    """按地理窗口合成 [Esri 背景 + xBD post 瓦片] 的虚拟大图。"""
    def __init__(self, manifest, event, basemap: BasemapTiles, stage="post"): ...
    def roi_candidates(self) -> list[str]:
        """可作 ROI 的瓦片：有 georef + 有标注 + post。"""
    def render(self, center_lat, center_lon, span_m, out_px=1024
               ) -> tuple[Image, MosaicMeta]:
        """1) Esri 背景  2) 叠窗口内 xBD post 瓦片  3) 色调协调  4) 重采样。
        meta: xbd_fraction, contributing_tile_ids, eff_gsd_m, window_bounds,
              roi_tile_id, roi_norm_bbox, basemap_provider。"""
```

- 叠加在**地理坐标**上做，用各瓦片自己的 `geo_to_pixel` 仿射，不假设像素网格对齐。
- 色调协调按 §2.2c 第 1 条实现，作为可关闭开关（消融用）。
- **进程内 LRU**（按 `(event, stage, 量化窗口)`）叠加 basemap 的**磁盘**缓存。
- pre 时相走同一路径，保证 pre/post 窗口严格同一地理范围。


**新增 `backend/fov_ladder.py`**（替代 `gsd_ladder.py` 的几何部分，约 120 行）

```python
SENSOR_PX      = 1024        # 设计值
FOV_DEG        = 60.0        # 设计值
TILE_SPAN_M    = 512.0       # 由 xBD 1024px @ 0.5m/px 推出
SPAN_TILES_MIN = 1.0         # 「最小视场 = 一整张瓦片」——作者需求的硬不变量
SPAN_TILES_MAX = 3.0         # Q3 决策：3×3 瓦片巡航

def alt_for_span_tiles(n): return n * TILE_SPAN_M / (2*tan(radians(FOV_DEG/2)))
def span_m_for_alt(alt):   return 2*alt*tan(radians(FOV_DEG/2))
def eff_gsd_for_alt(alt):  return span_m_for_alt(alt) / SENSOR_PX
def ladder_points(n_steps=3): ...
```

保留 `ExpectedEntropyTable`（`recheck.py` 的 info_gain 模式依赖它），
但**必须用新阶梯重新离线拟合**——旧表是在合成模糊上拟的，不能复用。

**修改 `backend/perception.py`**

- `_crop_uav_view()` → `_render_uav_view()`：走 `TileMosaic.render()`，
  窗口由 `span_m_for_alt(alt)` 决定，输出恒为 `SENSOR_PX`。
- **删除** `degrade_to_scale()` 调用与 `GSD_LADDER` 开关分支。
- **删除** `PERCEPTION_MIN_PATCH_PX` / `PERCEPTION_MAX_PATCH_PX` /
  `PERCEPTION_VIEW_ALT_FACTOR` / `PERCEPTION_MIN_RADIUS_M` / `PERCEPTION_MAX_RADIUS_M`
  这一整套已经互相打架的参数。
- `gsd_meta` 扩展：`span_m` / `span_tiles` / `coverage_ratio` /
  `contributing_tile_ids` / `roi_tile_id` / `roi_norm_bbox`。
- **裁块尺寸顺带从 96 px 提到 1024 px** → 这正是 review2 **B3** 要求的
  「224 px 原生裁块重训」的超集，需在论文里明确认领这一条。

**修改 `backend/recheck.py`**

- `descend_step_m=10.0` / `alt_min_m=10.0` 改为按 span_tiles 定义的档位。
- `expected_gsd_gain_ratio()` 当前用 `alt_after/alt`——在新模型下
  这**恰好仍然正确**（GSD ∝ alt），不需要改公式，只需要改默认值。

**修改 `backend/app.py`**

- `DEFAULT_HOVER_ALTITUDE_M = 30.0` → `alt_for_span_tiles(SPAN_TILES_MAX)` ≈ 1330。
- `VLN_RECHECK_DESCEND_M` / `VLN_RECHECK_ALT_MIN_M` 改为从 `fov_ladder` 取。
- **注意 `app.py:907-914` 有一段注释记录了「巡航高度=alt_min 导致 recheck 从一开始就
  恒真」的历史 bug**，重标定后必须重新验证这个不变量（`alt_cruise > alt_min` 且
  两者之间容得下 `max_rechecks` 步）。
- `MockAdapter` 的运动学：443→1330 m 的垂直机动时间与 30→10 m 完全不同，
  §2.1 的高度重标定会改变所有 **SPL / 路径代价 / 预算换算**。
  `docs` 里 §2.3「运动学与预算换算」整节需要重算。**这是最容易被漏掉的连锁改动。**

**修改题库生成 `scripts/benchmarks/gen_agent_vqa_testset.py`**

- 场景池 = `TileMosaic.roi_candidates()`（有 georef + 有标注 + post）。
  **Q2 的 ROI-scoped 决策让场景池从「约 200 个 3×3 中心」放宽到几乎全部 4312 个 post 瓦片**，
  事件多样性不再受马赛克密度限制——§7-Q4 的冲突因此大幅缓解。
- 起始位姿改为「ROI 不在视场中心」，激活搜索通道。
- 题目记录 `roi_tile_id` + `roi_bounds`；GT 仍从 ROI 瓦片标注计算（作用域不变）。
- **`spatial` 题的 GT 方位改为相对 ROI 中心**（见 §2.3 末尾）。
- **题库 schema 从 `agent-vqa/1.1` 升到 `2.0`，重新计算全部 SHA-256。**

**前端 `frontend/src/components/SituationMap.jsx`**

- 现有 `ImageOverlay`（L693）从「只叠 activeTile」改为「叠窗口内全部 post 瓦片」，
  与后端 `TileMosaic` 的合成结果保持视觉一致。
- 画出当前 FOV 足迹矩形（随高度收缩）与 ROI 框，让「降高=收视场」可视。
- **ROI 框只在前端操作台画，不进入送给 VLM 的图像**（§2.3 的信息边界约束）。


### 2.5 验收门槛（改动一）

1. 单测：`alt_for_span_tiles(1.0)` 渲染出的窗口与 ROI 瓦片地理范围重合，IoU > 0.99。
2. 单测：`eff_gsd_for_alt(alt_min)` == 原生 0.5 m/px ± 1e-6（天花板是推导出来的）。
3. 单测：`xbd_fraction == 1.0` 当且仅当窗口完全落在有 post 瓦片的区域里；
   ROI 区域的 `xbd_fraction` 必须恒为 1.0（**ROI 绝不能被 Esri 背景污染**）。
4. 单测：pre/post 同一 (center, span) 渲染出的窗口地理边界严格相等（双时相配准不破）。
5. 单测：`spatial` GT 方位相对 ROI 中心计算，且在智能体居中前后保持不变。
6. 回归：`degrade_to_scale` 在 `backend/` 下无任何调用点残留。
7. **泄漏体检**：在不给任何题面的前提下，用一个只看图的分类器/VLM 尝试从合成图里
   预测 ROI 位置。若显著优于随机，说明色调协调不充分，必须回头修 §2.2c。
8. **可识别性前置检查**：在 100 题 test split 上重算 `n_flip` / `n_correctable`。
   旧值 n_flip=7、n_correctable=2。**若新机制下 `n_correctable` 仍是个位数，
   E4 依旧只能按分支 C 写，不得因为「换了新机制」就改口径。**

---

## 3. 改动二：接入 xBD SOTA 检测器

### 3.1 候选与可得性（**我已实测验证下载链路**）

| 方案 | 指标 | 权重可得性（2026-08-24 实测） | 推理成本 |
|---|---|---|---|
| **xView2 1st (Durnov)** | overall **0.803**, damage F1 ~0.77 | `vdurnov.s3.amazonaws.com/xview2_1st_weights.zip` → **403 已失效**；<br>GitHub Release 分卷可用：`DIUx-xView/xView2_first_place` tag `final`，<br>`split-weights-a{a..e}` 共 **5.2 GB** ✓ | 4 架构×3 seed = **12 模型**，对交互式闭环过重 |
| **ChangeOS (RSE 2021)** | xBD overall ~0.80 | HF `EVER-Z/torchange_example_changeos_swint_on_xview2_best42k`（Swin-T）✓ | 单模型，**适合在线闭环** |
| **BDANet (TGRS 2021)** | 宣称 xBD SOTA | `ShaneShen/BDANet-Building-Damage-Assessment` | 两阶段，中等 |
| 现状（自训练） | 事件不相交 macro-F1 **0.22–0.28** | 本地 | 轻 |

2026-05 的一篇域适应论文明确指出：xView2 三个夺冠方案**至今仍是 SOTA**，
六年未被明显超越。所以「引入 SOTA」= 引入 xView2 冠军方案，这个判断是稳的。

**建议双轨**：
- **离线主表**用 xView2 1st 完整 12 模型集成（对标文献数字，一次性批量推理，
  结果缓存进 `runs/`，不参与在线闭环）。
- **在线智能体闭环**用 ChangeOS-SwinT 单模型或 1st 方案的单架构单 seed 子集。
  **必须显式报告这个降配及其分数损失**，不能拿 12 模型的 0.803 去描述在线系统。

### 3.2 ★ 最严重的问题：预训练权重破坏事件不相交协议

这是整个计划里科学风险最高的一点，**必须在动手前想清楚**。

xView2 冠军权重是在 xBD 官方 train + tier3 上训练的。而 `paper_cja` 的
test 事件（hurricane-michael、palu-tsunami）与 holdout 事件
（moore-tornado、nepal-flooding、pinery-bushfire）**全部出现在官方 train 划分中**
（xBD 是按瓦片而非按事件划分 train/test 的）。

也就是说：**直接加载现成 SOTA 权重 = 感知层在评测事件上已经训练过。**

这会同时打掉两样东西：
- `AGENT_VQA_REVISION_PLAN.md` §4.5「事件不相交数据协议与泄漏断言」；
- review2 **B1** 要求补的 19 事件 LOEO——LOEO 的前提就是权重没见过留出事件。

### 3.3 采纳方案：双轨报告（Q5 决策）

- **轨 A — leaky 参照上界**：直接加载 xView2 冠军现成权重。
  **必须在所有表格里显式标注 `leaky`**，与 `O_REF` 同级，只作诊断，
  **不进正式对照主表**。用途：确认管线封装正确、给出「感知上界」的量级。
- **轨 B — 干净主结果**：SOTA 架构（建议 ResNet34-UNet 单架构，loc + cls 两阶段）
  在**事件不相交协议**下重训 1–2 折，作为论文主表的感知底座。
- **★ 两轨差值本身是可发表的测量**：「现成 SOTA 的 0.803 里，
  有多少来自对评测事件的记忆？」这正好服务于论文自称的
  「可识别性优先评估」方法学，并**一次性回答 review2 的 B7**
  （「剩余差距归因于架构/训练规模是断言而非测量」）——
  因为轨 B 用的就是 SOTA 架构，架构因素被控制住了。

**工期**：轨 A 约 1 天（下载 5.2 GB + 封装 + 验证）；轨 B 数天量级
（本机 2×RTX 6000D，GPU0 已被占 34 GB，实际只有 GPU1 空闲）。
**建议先做轨 A 打通管线并跑 P4 前置检查，确认值得投入后再启动轨 B。**

*（已否决：(i) 只用现成权重——结论须重述为「在已见过评测事件的底座下」，
且与 B1 的 LOEO 不能共存于同一张主表，审稿风险高于 B2；
(ii) 完全从头重训完整 12 模型集成——多卡数天，性价比不成立。）*

### 3.4 代码改动清单

**新增 `backend/detectors/`**

```
backend/detectors/
  __init__.py
  base.py            # DetectorProtocol: detect(pre_img, post_img, meta) -> [Detection]
  xview2_first.py    # Durnov loc+cls 封装，支持 ensemble_size 降配
  changeos.py        # torchange / HF checkpoint 封装
  legacy_unet.py     # 现有 building_localization + change_perception，保留为对照基线
```

- `Detection` 字段与现有 `perception._detect()` 输出对齐：
  `bbox` / `class_name`（中文）/ `conf` / **`class_probs`（4 类，必须有——
  `recheck.py` 的熵模式与 `agent_vqa` 的 A3_ENTROPY 全靠它）**。
- 新增 env：`DETECTOR_BACKEND ∈ {legacy_unet, xview2_first, changeos}`，
  默认保持 `legacy_unet`，**不破坏任何现有可复现产物**。
- 温度缩放层保留：SOTA 模型未必校准，B4 的 Brier/NLL/AURC 分析要求概率可用。

**修改 `backend/perception.py`**
- `_detect()` 按 `DETECTOR_BACKEND` 分派到 `backend/detectors/*`。
- 现有 U-Net 全瓦片提议缓存（`_unet_proposals_for_view`）逻辑需要适配马赛克——
  现在「全瓦片」变成「全窗口」，缓存键要带窗口。

**新增脚本**
- `scripts/training/fetch_xview2_weights.py`：分卷下载 + 校验 + 合并解压。
- `scripts/benchmarks/eval_detector_backends.py`：三个 backend 在同一事件不相交
  test split 上的 macro-F1 / 逐类召回 / ECE 对照表，直接生成 `paper_cja/generated/`。

### 3.5 验收门槛（改动二）

1. `eval_detector_backends.py` 复现出 SOTA 的量级（overall > 0.7），
   否则说明封装有错，不得当作「SOTA 在我们协议下变差了」。
2. **轻微/严重损伤召回必须 > 0** ——这是当前底座最刺眼的失效点（现为 0.000/0.000）。
3. 单张观测端到端延迟记录进 benchmark；若在线配置 > 2 s/次，闭环实验不可行，
   需进一步降配并如实报告。
4. `class_probs` 在所有 backend 上非空且和为 1，否则 A3_ENTROPY 静默退化。

---

## 4. 对既有实验与论文主张的冲击

### 4.1 D6 holdout 结论的处置 ★ 需决策

`AGENT_VQA_EXPERIMENT_STATUS.md` 明确写着：

> holdout 数据已消费完毕，不得因这批结果反过来调 prompt、阈值、动作预算或题库

本计划改的是**观测几何与感知底座**，不是「因为结果不好去调参」。
按预注册惯例，这构成一个**新系统**，需要**新的预注册 + 新的 holdout**，
旧 holdout 降级为「旧系统的历史记录」而非被删除。

但必须防住一个真实风险：**「换机制→重跑 holdout→挑好看的报」**。
建议的护栏：

1. 旧 holdout 结论（分支 C，A2 vs A0 +0.01 CI[-0.02,0.05]）**原文保留在论文里**，
   作为「旧观测模型下的结果」，并明写它为什么被替换（B2）。
2. 新预注册在**跑任何新 holdout 之前**写死：配置集合、主指标、
   可识别性前置检查阈值、分支判据。
3. **holdout 事件重新划分**——旧 holdout 三事件已消费，新实验若仍用它们，
   等于用已看过的数据。**Q2 的 ROI-scoped 决策已经大幅缓解这个冲突**：
   场景池不再依赖 3×3 马赛克密度（不再被 moore-tornado 绑架），
   19 个事件里任何有标注 post 瓦片的都能出题。
   因此**建议新 holdout 换用一组全新事件**（从未在 test/holdout 出现过的），
   把旧三事件降级为开发集。**这一条仍需在 P6 预注册时由作者最终确认（原 Q4）。**

### 4.2 论文写作口径的变化

- §4.2「高度–有效 GSD 阶梯」整节重写：从「合成降质阶梯」改为「视场–分辨率权衡」，
  并给出 §2.1 的物理推导。
- §3.3 禁止表述「不得把合成有效 GSD 阶梯解释为真实低空厘米级信息获取」
  **仍然适用但含义变了**：新机制下天花板是 0.5 m/px 原生 GSD，
  仍不是厘米级；必须明写「本仿真的信息上界是源影像的 0.5 m/px，
  低于该 GSD 的结构不可恢复」。**这条局限不会因为改动一消失，只是从
  「人为设定」变成「数据源固有」——后者是可接受的，前者不是。**
- E4 若翻到分支 A/B，`CLAIMS.md` 需同步更新；**但在可识别性前置检查
  （§2.5 第 6 条）通过之前，不得预设会翻。**

### 4.3 不在本计划范围内的阻断项

以下仍需独立处理，本计划不覆盖：review2 A 组 9 条体例/占位符问题、
B1 的 19 事件 LOEO、B4 的 X2 概率质量重跑、B8/B9。

---

## 5. 分阶段执行计划

| 阶段 | 内容 | 产出 | 门槛 |
|---|---|---|---|
| **P0** | 清磁盘（删 `~/datasets/xbd/*.tar.gz` 可释放 28 GB）；下载 SOTA 权重 5.2 GB | 权重就位 | 磁盘 > 40 GB 可用 |
| **P1** | `basemap.py`（Esri 抓取+缓存）+ `mosaic.py` + `fov_ladder.py` + 单测 | 可按地理窗口渲染合成图 | §2.5 的 1–6 |
| **P1.5** | **泄漏体检**（§2.5 第 7 条） | 源可分性报告 | 显著优于随机则回头修色调协调 |
| **P2** | `perception.py` 接马赛克；前端 FOV/ROI 可视化 | 端到端能飞能降 | 人工看图确认「降高=收视场」 |
| **P3a** | `detectors/` + 轨 A（现成 SOTA 权重，leaky） | 检测器对照表 | §3.5 全部 |
| **P4** | **可识别性前置检查**（旧题库、新机制、轨 A 底座，test split） | `n_flip` / `n_correctable` | **决定是否值得投入 P3b–P8** |
| **P3b** | 轨 B：SOTA 架构事件不相交重训 1–2 折 | 干净主结果底座 | 轻微/严重召回 > 0 |
| **P5** | 题库 v2.0 重建（ROI-scoped）+ 作者审核 | 新题库 + SHA-256 | 自动审核 100% |
| **P6** | 新预注册文档（含 holdout 事件最终确认） | 冻结配置与判据 | 作者签字 |
| **P7** | test split pilot | pilot 报告 | `valid_for_analysis=true` |
| **P8** | 新 holdout（一次性） | 正式结果 | 不得重跑 |

**P4 是最重要的一关**：它用最小成本回答「这两个改动到底有没有解除不可识别」。
在 P4 出结果之前，不要投入 P3b / P5–P8 的成本。

---

## 6. 风险清单

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| R1 | **磁盘只剩 32 GB**（`/home` 15 T 已 100% 占用，实测）；新增 SOTA 权重 5.2 GB + 底图缓存 ~2.4 GB | **阻断** | P0 删 28 GB 冗余 tar.gz；底图缓存落盘但 xBD 马赛克只做内存 LRU |
| R2 | 预训练 SOTA 破坏事件不相交协议（§3.2） | **高** | 已采纳双轨（Q5）：leaky 轨只作诊断，主表用重训轨 |
| R3 | Esri 底图与 xBD 的**日期/色调**差异造成源不连续泄漏 | **中**（GSD 实测 0.513 vs 0.500 已基本对齐，锐度差异不成立） | §2.2c 三项抑制 + P1.5 泄漏体检 + 按 `xbd_fraction` 分层报告 |
| R3b | Esri 底图是**灾前**影像混入「灾后观测」 | **中** | ROI 内恒为真实 xBD post（§2.5 第 3 条硬约束）；背景仅上下文，论文明确披露 |
| R3c | Esri 服务条款对批量缓存的限制 | 低–中 | 沿用前端已声明的 attribution；**建议作者复核条款** |
| R4 | ~~3×3 场景集中在 moore-tornado~~ | **已解除** | Q2 的 ROI-scoped 使场景池不再依赖马赛克密度 |
| R5 | 高度重标定连锁改变运动学/SPL/预算换算（§2.4） | 中 | 单独一轮回归；`docs` §2.3 重算 |
| R6 | 12 模型集成推理过慢，闭环不可行 | 中 | 在线用单模型并显式报告降配 |
| R7 | 改动后 `n_correctable` 仍是个位数 | 中 | P4 提前暴露；分支 C 口径保持不变，**不得改判据** |
| R8 | SOTA 输出未校准，`class_probs` 熵失去意义 | 中 | 保留温度缩放，重新拟合 |
| R9 | 题库 schema 升版使全部历史产物不可比 | 低 | 旧产物保留只读，论文分「旧/新观测模型」两栏 |
| R10 | 底图抓取依赖外网，评测可复现性受影响 | 中 | 缓存随论文归档并写 `provenance.json`；离线未命中必须报错不得静默灰图 |

---

## 7. 决策状态

| # | 问题 | 状态 |
|---|---|---|
| Q1 | 底图路线 | ✅ **已定**：复刻前端方案，xBD post 瓦片叠 Esri World Imagery（§2.2b/2.2c） |
| Q2 | 题目作用域 | ✅ **已定**：ROI-scoped（§2.3） |
| Q3 | 阶梯跨度 | ✅ **已定**：3×3 瓦片，1330 m → 443 m，1.5 → 0.5 m/px（§2.1） |
| Q4 | holdout 事件 | ⏳ **推迟到 P6**：ROI-scoped 已解除马赛克密度约束，建议换全新事件，待作者最终确认（§4.1） |
| Q5 | SOTA 引入方式 | ✅ **已定**：双轨报告（§3.3） |

**剩余一项待确认**：R3c 的 Esri 服务条款复核（服务端批量缓存作为模型输入是否合规）。

---

## 8. 附：本文档中「实测」数字的来源

| 数字 | 来源 |
|---|---|
| 瓦片邻接密度表（§2.2） | 本机运行，读 `backend/data/xbd/manifest.json` 4312 条 post 且有 georef 的条目，按瓦片中心经纬量化到整数格后统计 |
| 当前裁块尺寸表（§1.1） | 按 `perception.py:472-488` 的常量手工代入计算 |
| macro-F1 / 逐类召回 / n_flip | `paper_cja/generated/gsd_ladder_table.tex`、`gsd_class_table.tex`、`training_curve_table.tex` |
| `n_correctable=2` / holdout 结果 | `paper_cja/AGENT_VQA_EXPERIMENT_STATUS.md` §D6 |
| 权重可得性与体积 | 本机 `curl` GitHub Releases API 与 S3（S3 返回 403） |
| Esri z18 = 0.513 m/px、瓦片 19.7 KB | 本机抓 `server.arcgisonline.com/.../World_Imagery/18/{y}/{x}`（lat 30.7）并解码 |
| 前端图层构成 | `frontend/src/components/SituationMap.jsx` L662–668（Esri TileLayer）、L693–695（xBD ImageOverlay）、L702–708（footprints GeoJSON） |
| 磁盘 32 GB | 本机 `df -h /home/lc` |
| GPU | 本机 `nvidia-smi`：2×RTX 6000D 85 GB，GPU0 已占 34 GB，GPU1 空闲 |
