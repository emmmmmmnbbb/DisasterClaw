# MOSAIC-FOV + xBD SOTA 最新进度

更新时间：2026-08-24

本文档依据 [`MOSAIC_FOV_SOTA_REVISION_PLAN.md`](MOSAIC_FOV_SOTA_REVISION_PLAN.md)、当前工作区代码、测试结果和 `runs/` 下的实验产物整理。只记录方案实施、验证和实验，不记录数据下载、清理磁盘等前置准备。

## Material Passport

- `schema`: ARS Material Passport 9
- `artifact_type`: implementation and experiment progress audit
- `paper`: `paper_cja`
- `source_scope`: 当前工作区代码、`backend/tests/`、`backend/outputs/uav_view/p2b_*`、`runs/benchmarks/detector_backends/*.json`
- `verification_status`: **ANALYZED**；本次重跑了全量后端单测，但没有重跑 P2/P3a GPU benchmark
- `external_upload`: 无
- `revision_state`: MOSAIC-FOV/SOTA 相关代码目前仍包含未提交修改和未跟踪文件；本文反映的是当前工作区，不是已冻结的 Git revision

## 总结

当前完成度（不计 P0 等前置准备）：**10 个实施阶段中，1 个完成、3 个进行中、6 个未开始。** P1 核心完成；P2、P3a 和 P4 处于进行中，其中 P4 只有脚本、还没有结果；P1.5、P3b、P5-P8 未开始。

目前最重要的可复核结论是：xView2 冠军权重在事件不相交 test 上消除了 legacy 模型的轻微/严重损伤类别坍缩，但这些权重见过评测事件，只能作为 leaky 诊断上界，不能作为论文主结果。新观测模型是否提高了主动复核的可识别性尚未有实验结果，因为 P4 尚未运行。

## 阶段状态

| 阶段 | 状态 | 当前判断 | 证据 |
|---|---|---|---|
| P1 观测几何、底图、马赛克 | **已完成（核心）** | `fov_ladder`、`basemap`、`mosaic`、ROI 覆盖索引和单测已实现 | [`backend/fov_ladder.py`](../backend/fov_ladder.py)、[`backend/basemap.py`](../backend/basemap.py)、[`backend/mosaic.py`](../backend/mosaic.py)、[`backend/tests/test_fov_ladder.py`](../backend/tests/test_fov_ladder.py)、[`backend/tests/test_mosaic.py`](../backend/tests/test_mosaic.py) |
| P1.5 泄漏体检 | **未完成** | 方案明确跳过；没有只看图预测 ROI/来源的正式实验或报告 | 未发现对应脚本、报告或结果 JSON |
| P2 接入马赛克观测 | **进行中** | 后端默认已切到 `mosaic_fov`，端到端样例已跑通；前端仍只显示 active tile，新熵表未重拟合，论文/运动学文档仍保留旧 30/10 m 口径 | [`backend/perception.py`](../backend/perception.py)、[`backend/app.py`](../backend/app.py)、[`frontend/src/components/SituationMap.jsx`](../frontend/src/components/SituationMap.jsx) |
| P3a SOTA 检测器轨 A | **进行中** | 独立封装、权重加载、对照 benchmark 和门槛检查已有结果；但 `perception.py` 尚未按 `DETECTOR_BACKEND` 分派，在线闭环仍走 legacy 路径 | [`backend/detectors/`](../backend/detectors/)、[`scripts/benchmarks/eval_detector_backends.py`](../scripts/benchmarks/eval_detector_backends.py)、[`runs/benchmarks/detector_backends/`](../runs/benchmarks/detector_backends/) |
| P4 可识别性前置检查 | **进行中（仅脚本）** | 评估脚本已存在，但没有 `precheck_*.json` 结果，尚不能作阶段判定 | [`scripts/benchmarks/eval_identifiability.py`](../scripts/benchmarks/eval_identifiability.py) |
| P3b 事件不相交 SOTA 重训 | **未完成** | 没有干净重训权重、训练日志或评测结果 | 未发现对应训练产物 |
| P5 ROI-scoped 题库 v2.0 | **未完成** | 现有生成器仍输出 `agent-vqa/1.1`，仍使用旧起始位姿/半径逻辑 | [`scripts/benchmarks/gen_agent_vqa_testset.py`](../scripts/benchmarks/gen_agent_vqa_testset.py) |
| P6 新预注册 | **未完成** | 未发现针对新观测模型的冻结配置和 holdout 事件确认文档 | 未发现新 preregistration artifact |
| P7 test pilot | **未完成（新系统）** | 旧 pilot 不能替代新观测模型/新题库 pilot | `runs/benchmarks/cja_agent_vqa/` 下仅有旧系统产物 |
| P8 新 holdout | **未完成** | 旧 D6 holdout 已消费，未运行新系统 holdout | `runs/benchmarks/cja_agent_vqa/d6_holdout100_*` 为旧系统记录 |

## 已完成的实现与验证

### P1：视场几何与马赛克合成

已实现并通过单元测试的核心不变量：

- 下限高度 `443.405 m` 对应恰好 `1.0` 张瓦片和 `0.5 m/px`。
- 巡航高度 `1330.215 m` 对应 `3.0` 张瓦片和 `1.5 m/px`。
- 降高时 ROI 在输出视场中的像素占比从约 `1/3` 提升到 `1/2`，再到 `1.0`。
- pre/post 使用严格相同的地理窗口。
- ROI 覆盖不足时显式抛出 `RoiCoverageError`；nodata 不计入 xBD 覆盖率。
- 当前 ROI 索引覆盖 `4312` 个 post 且有 georef 的瓦片，其中巡航几何覆盖率达到 `>=0.80` 的瓦片为 `495` 个。

与原计划的偏差：`degrade_to_scale()` 仍保留在 `MOSAIC_VIEW=0` 的 legacy 复现路径中，默认 `mosaic_fov` 路径不会调用它；因此满足“新实验不再使用合成模糊”，但没有字面满足“`backend/` 下完全无调用点”。

### P2：马赛克观测端到端样例

`hurricane-michael_00000034` 的三档观测已生成对应图像、pre 图和检测 JSON（`backend/outputs/uav_view/p2b_*`）：

| 档位 | 有效 GSD | xBD 覆盖率 | pre_scale | 检测数 | 方案记录的暖缓存耗时 |
|---|---:|---:|---:|---:|---:|
| 巡航 | 1.50 m/px | 0.988 | 3.0 | 657 | 8.3 s |
| 中档 | 1.00 m/px | 1.000 | 2.0 | 219 | 3.9 s |
| 下限 | 0.50 m/px | 1.000 | 1.0 | 63 | 1.8 s |

检测数可由现有 `p2b_*_det.json` 复核；覆盖率、`pre_scale` 和耗时来自方案文档中的执行记录，目前尚无一份汇总 P2 benchmark JSON。

结果支持“降高收缩视场、提高 ROI 单位像素数”的几何行为；但巡航档仍超过计划中的 `2 s/次` 闭环门槛。该延迟主要来自原生 pre 渲染和逐框双时相分类，尚未做批处理或 ROI 内筛选优化。

### P3a：检测器契约与 SOTA 轨 A

独立检测器包已支持 `legacy_unet` 与 `xview2_first`，并锁定 BGR、`x/127-1`、sigmoid、6 通道 siamese 和四类概率输出等参考实现细节。当前 `perception.py` 尚未调用 `detectors.get_detector()`，因此这些 SOTA 结果是离线轨 A benchmark，不代表在线智能体已经切换到 SOTA。全量后端测试结果：

```text
146 passed, 6 warnings
```

以下数字来自结果 JSON，评测集为事件不相交的 `hurricane-michael + palu-tsunami`，共 30 张瓦片。

| 指标 | legacy_unet | xview2_first res34/seed0 | xview2_first 全集成 |
|---|---:|---:|---:|
| 像素 loc F1 | 0.8068 | 0.8532 | 0.8574 |
| 像素 damage F1（调和平均） | 0.0000 | 0.6304 | 0.6278 |
| 像素 overall | 0.2421 | 0.6973 | 0.6967 |
| 逐建筑 macro-F1 | 0.1194 | 0.4069 | 0.3738 |
| 轻微损伤召回 | 0.0000 | 0.5722 | 0.4734 |
| 严重损伤召回 | 0.0000 | 0.4286 | 0.2571 |
| 完全损毁召回 | 0.0000 | 0.2177 | 0.0551 |
| 单次延迟中位数 | 0.023 s | 0.220 s | 0.561 s |

门槛判定：

- overall `> 0.7`：**未达**（res34 `0.6973`，全集成 `0.6967`）。
- 轻微/严重损伤召回 `> 0`：**达成**。
- 单次延迟 `< 2 s`：**达成**（检测器 benchmark 口径）。
- `class_probs` 有效：**达成**。

事件泄漏状态：`xview2_first` 的权重在 xBD 官方 train+tier3 上训练，而 xBD 按瓦片而非事件划分；因此 paper_cja 的 test/holdout 事件已被权重见过。上述 SOTA 数字只能作为 `leaky` 参照，不能与 legacy 的差值直接归因于架构改进，也不能写入正式主对照表。

结果文件：

- [`legacy_unet_test_resnet34unet+siamese.json`](../runs/benchmarks/detector_backends/legacy_unet_test_resnet34unet+siamese.json)
- [`xview2_first_test_res34_seeds0.json`](../runs/benchmarks/detector_backends/xview2_first_test_res34_seeds0.json)
- [`xview2_first_test_res34+res50+dpn92+se154_seeds012.json`](../runs/benchmarks/detector_backends/xview2_first_test_res34+res50+dpn92+se154_seeds012.json)

> 说明：方案正文另记录了 `逐建筑 macro-F1=0.4517`、`单模型延迟=0.113 s` 等后续数字，但当前结果目录中没有与这两个数字对应的独立 JSON；本进度只把上表中可直接追溯的结果作为正式结果，待补原始产物后再更新。

P3a 尚未闭环的实现项还包括：计划中的 `changeos.py` 未实现、在线 `_detect()` 未按 `DETECTOR_BACKEND` 分派、结果尚未导出到 `paper_cja/generated/`。

## 尚未完成的关键实验

### P1.5 泄漏体检

没有正式测试“只看合成图能否预测 ROI/影像来源”。`basemap.py` 中有色调协调实现，缓存目录也存在，但没有可审计的 source-separability 结果；因此不能声称 Esri 与 xBD 的来源泄漏已被实验排除。

### P4 可识别性前置检查

[`eval_identifiability.py`](../scripts/benchmarks/eval_identifiability.py) 已实现巡航/下限双观测、逐建筑匹配和 ROI-scoped 问题级 `n_flip` / `n_correctable` 统计，但当前不存在 `runs/benchmarks/identifiability/precheck_test_*.json` 或 `precheck_holdout_*.json`。

因此目前不能更新旧系统的 `n_flip=7`、`n_correctable=2`，也不能判断 E4 是否从“不可识别”分支 C 转为可识别。P4 是下一步的首要阻断点。

### P3b：干净 SOTA 主结果

没有事件不相交重训的 ResNet34-UNet loc/cls 权重、训练曲线、验证集结果或 test 结果。当前所有 xView2 权重均属于 leaky 轨 A，不能满足方案确定的论文主结果协议。

### P5-P8：新题库、预注册、pilot、holdout

- 生成器仍为 `agent-vqa/1.1`，尚未升到计划要求的 `2.0`。
- 生成器仍以旧 `CRUISE_RADIUS_M` 和 start-relative spatial GT 为主，未完成 ROI 中心相对方位改造。
- 前端 `SituationMap.jsx` 未实现窗口内多 post 瓦片、FOV 足迹和 ROI 框展示。
- `backend/data/gsd_entropy_table.json` 仍是旧阶梯产物，没有 `fov-ladder-entropy/1.0` schema；当前会被新加载器拒绝并回退启发式，尚未完成计划要求的新阶梯离线拟合。
- `paper_cja/sections/method.tex`、`architecture.tex` 和相关说明仍写 30/10 m 与合成模糊阶梯，尚未同步新物理口径和运动学/预算换算。
- 没有新预注册、作者审核后的 SHA-256 题库、`valid_for_analysis=true` 的新 pilot 或新 holdout。
- 旧 D6 holdout 只能作为旧系统历史记录，不能充当新系统验证集。

## 当前结论与下一步

当前工作已经证明两件事：

1. 视场收缩式观测模型在几何和代码层面已落地，并能生成真实的三档观测；
2. xView2 轨 A 能显著缓解 legacy 的损伤类别坍缩，但其事件泄漏使它只能作为诊断上界。

下一步应按以下顺序推进：

1. 先运行 P4（优先 `test` split，明确记录 leaky 标记），得到新的 `n_flip` / `n_correctable`；
2. 若 P4 达到预先设定的可识别性门槛，再投入 P3b 事件不相交重训；
3. 同步完成 P1.5 泄漏体检和 P2 前端可视化；
4. 之后才重建 v2.0 题库、冻结预注册并运行新 pilot/holdout。
