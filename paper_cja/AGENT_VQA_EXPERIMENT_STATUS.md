# Agent-VQA 实验状态与执行门槛

## Material Passport

- `schema`: ARS Material Passport 9
- `artifact_type`: experiment status and reproducibility gate
- `paper`: `paper_cja`
- `updated_at`: 2026-08-21
- `source_scope`: 本地代码、测试、题库与 `runs/benchmarks/cja_agent_vqa/` 产物
- `external_upload`: 无
- `verification_status`: D5 100 题 test GPU 评测已完成且可分析；D6 配置集合已冻结
  （V0/V1/A0/A2/A3，不含 A1/A4/A5）；spatial `decision_reason_mismatch` prompt
  修复已落地但未经 GPU 重跑验证；holdout 未启动
- `git_commit_at_audit`: `626f54745287bc7ec4fa76db271a674bc6d8d31b`，工作区仍有未提交修改
  （新增 prompt 修复与回归测试）

## 当前判定

1. D3 后端和 D4 前端的主要功能已实现；D1/D2 仍不能整体标为完成（LOEO/多 seed/作者抽查未做）。
2. 2026-08-21 已完成 100 题 test、五配置、本地 Qwen2.5-VL-7B + GPU 感知评测；四片 `valid_for_analysis=true`，执行错误 0，`fallback_rate=0`。
3. 重观测通道可识别：A2 191 次下降，A3 60 次重观测 / 65 次 skip。搜索通道未激活：A0 全部 1 步，题面均为当前视场。
4. 不得把本轮数字写入论文正式表。D6 配置集合与 spatial schema 修复已按下方“冻结决定”落定；
   holdout、题库、动作预算仍不得因为本轮或未来 holdout 结果反过来调整。

## 题库冻结信息

- 文件：`backend/data/benchmarks/agent_vqa_testset.json`
- schema：`agent-vqa/1.1`
- 题目数：200
- 题库文件 SHA-256：`21edf4939ca4e3895d11412b2560c8f179228805d5a347e83eba3ad855ab9e21`
- items 规范化 JSON SHA-256：`6ed3ce4bd321875f5e78c2edefbc3db6e1df33487a6f21eadf0b48cd21512f3a`
- 数据 manifest SHA-256：`50b827975c8cb159f71d1b04b3f2ffa3ddb88fb3dcbe85a6239d7d47c4a1b35e`
- 自动审核：200/200 通过
- 作者审核：未完成；96 道带歧义标志题仍待抽查

### 事件隔离

| 分区 | 事件 | 题数 | 当前用途 |
| --- | --- | ---: | --- |
| test | hurricane-michael, palu-tsunami | 100 | D5 pilot、协议诊断与功效估计 |
| holdout | moore-tornado, nepal-flooding, pinery-bushfire | 100 | 配置冻结后一次性正式评测 |

原计划中的“200 场景开发评测”会提前消费 holdout，现修正为 100 道 test pilot。
不得根据 holdout 输出修改 prompt、阈值、动作预算或题库。

## 已验证产物

### 有效 GPU schema smoke

目录：`runs/benchmarks/cja_agent_vqa/d5_smoke_gpu1_after_schema/`

- 8 道 test 题，覆盖 presence、damage、count、spatial。
- 配置：V0_RAW、V1_STRUCT、A0_HOLD、A3_ENTROPY。
- 执行错误：0。
- V0/V1 的非法 schema 输出：0；A0/A3 各有 1 条 `decision_reason_mismatch`。
- A3 未触发重观测；该运行早于当前轨迹与 recheck 审计增强，不能验证当前主动链路。
- V0_RAW accuracy 0.500；V1_STRUCT、A0_HOLD、A3_ENTROPY 均为 0.375。
- 所有配对 bootstrap 95% CI 均包含 0，不能宣称配置优于基线。
- 报告：`runs/benchmarks/cja_agent_vqa/d5_smoke_gpu1_after_schema/reports/`。

### 动作链 CPU 诊断

目录：`runs/benchmarks/cja_agent_vqa/d5_cpu_reobs_min2/` 与
`runs/benchmarks/cja_agent_vqa/d5_cpu_smoke8_reobs/`。

- A2_ALWAYS 能执行两次下降，轨迹出现实际 reobserve。
- VLM 连接失败，`fallback_rate=1.0`，因此只能证明动作通道，不得用于 VQA 主表。
- 诊断发现并修复：动作前图像被错误关联到动作后高度的轨迹位姿问题。

### 明确无效的目录

- `d5_smoke_reobs_audit/`：40 个执行错误。
- `d5_smoke_reobs_gpu1/`：40 个执行错误，CUDA 初始化失败。
- `d5_schema_verify_gpu1/`：2 个执行错误，属于 dtype 竞态修复前产物。

`report_agent_vqa.py` 现会拒绝 `valid_for_analysis=false` 的运行，禁止将上述目录汇入统计。

## 已修复的阻断项

1. VLM 将自然语言证据描述写入 `evidence.source`，导致合法 JSON 被严格 schema 拒绝。
2. Qwen 与 building localizer 并发加载时，PyTorch 默认 dtype 窗口使自研模型偶发变成 BF16。
3. recheck 仅在当前 argmax 已匹配题面子类时才被询问，阻断高熵但类别暂时错误的样本。
4. benchmark 未持久化原始输出、schema 错误、在线 evidence、完整轨迹和 recheck 审计字段。
5. A2_ALWAYS 在无可疑损伤框时无法作为“额外观测上限”对照执行。
6. `--resume` 跳过旧 episode 后未把旧行纳入聚合，可能覆盖出错误的 `n=0` 结果。
7. 多 GPU 脚本的设备重映射触发 CUDA 初始化错误，并会在 shard 失败后继续尝试合并。
8. 报告把 `invalid_output` 混入普通弃答，且允许聚合已标记无效的运行。

## 当前验证

- Python 相关测试：75 passed。
- 前端：`npm run build` 通过。
- `bash -n scripts/benchmarks/run_agent_vqa_parallel.sh`：通过。
- `git diff --check`：通过。

## D5 100 题 test（2026-08-21）

目录：`runs/benchmarks/cja_agent_vqa/d5_pilot100_shard{0-3}of4/`，合并报告
`d5_pilot100_reports/`。耗时约 10 分钟。git `626f547` dirty。题库 hash `21edf4939ca4e389`。

| config | n | acc | abstain | n_reobservations | n_reobserve_skips | n_steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V0_RAW | 100 | 0.35 | 0.02 | 0 | 0 | 1.00 |
| V1_STRUCT | 100 | 0.35 | 0.00 | 0 | 0 | 1.00 |
| A0_HOLD | 100 | 0.35 | 0.06 | 0 | 0 | 1.00 |
| A2_ALWAYS | 100 | 0.39 | 0.06 | 191 | 0 | 2.91 |
| A3_ENTROPY | 100 | 0.38 | 0.06 | 60 | 65 | 1.60 |
| O_REF（离线） | 100 | 0.39 | 0.04 | 24 | 0 | 1.24 |

配对 bootstrap 95% CI：

- A2 vs A0：+0.04，CI [0.01, 0.08]，显著。
- A3 vs A0：+0.03，CI [0.00, 0.07]，不显著。
- V1 vs V0：0.00，CI [-0.04, 0.04]，不显著。
- O_REF vs A0：+0.04，与 A2 重合；`n_correctable=4`，`n_harmful=0`，`n_neither_correct=61`。

可识别性：A0 零重观测、A2 有重观测、A3 有 skip+触发，满足主动通道判据。题库为当前视场问题，A0 未搜索，不能用来主张“搜索改善答案”。

答案坍缩（V0）：presence 37/39 预测“否”；damage 22/22“无损伤”；count 22/22“0”；spatial 集中在西/西南。主失败是错答，不是弃答。

无效输出：agent 三配置各 6 条，几乎全是 spatial `decision_reason_mismatch`（`decision=answer` 且 `reason_code=target_missing`）。V1 无非法输出。不得静默改写成合法答案。

功效建议（待作者确认后再冻结）：

- holdout 维持 n=100，只用于一次性确认 A2 vs A0 的 +4pp，不调参。
- A3 vs A0、A1/A4/A5 策略排序在 `n_correctable=4` 下很可能不可识别，D6 E4 应预写成上限/不可识别，而不是再堆配置。
- D6 前先修 spatial schema 冲突，并提交当前 dirty 工作区；否则 holdout 与 pilot 不在同一 commit。

## 冻结决定（2026-08-21，pilot100 之后）

对 pilot100 提出的三项待拍板逐条落定：

1. **D6 配置集合：冻结为 V0/V1/A0/A2/A3（O_REF 仍为离线诊断，不进正式对照表）。**
   不加 A1_RANDOM/A4_CONFORMAL/A5_EXPECTED。理由：`n_correctable=4`（O_REF vs A0
   headroom）意味着策略之间可比较的样本极少，再堆策略只会把同一批极小
   headroom 切得更碎，产生看似不同实为噪声的排序；这与计划 11.6 预先写入的
   可识别性判据一致——headroom 不够时应报告“不可识别/上限”，不应扩大比较空间。
   若 holdout 复现 `n_correctable` 仍然是个位数，D6 E4 正文写法固定为“主动通道
   可识别（A2 触发、A3 触发+skip），但可纠正样本过少，无法排序候选策略”，不追加新配置来源。
2. **先修 `decision_reason_mismatch` 再冻结，已修复（本次改动，未提交）。** 见下方
   “spatial 非法输出修复”。
3. **D1 的 LOEO / 多 seed 与 Agent-VQA holdout 并行、互不替代，维持原判定不变。**

### spatial 非法输出修复

根因定位（读取 `d5_pilot100_shard*/episodes.jsonl` 中全部 6+6+6 条
`decision_reason_mismatch` 原始输出核实）：不是校验器逻辑错误——
`validate_answer_dict` 里 `decision=answer` 只允许 `reason_code=sufficient_evidence`
是计划本身要求的约束（`test_abstain_and_evidence_are_required_and_consistent`
已锁定这条），把它松绑等于允许模型一边承认证据不足一边给出答案，属于计划
3.3 明确禁止的“不得用不显著结果证明没有差异”一类静默放行。真正的根因是
`AGENT_VQA_SYSTEM_PROMPT`（`backend/vlm_analyzer.py`）只分别列出了 `decision`
和 `reason_code` 各自的封闭集合，从未告诉模型二者必须配对；6 条样本的共同
模式是模型确认“图像中没有显示任何严重损伤/完全损毁建筑”，却仍以
confidence=0.8 猜一个方位（西/西南居多，呼应 pilot100 记录的 spatial
坍缩方向），reason_code 诚实地写成 `target_missing`，但 decision 留在
`answer`。

修复：在 prompt 中加入显式配对规则（`decision=answer` 只配
`sufficient_evidence`；`continue_search` 只配 `target_missing`；`reobserve`
只配 `low_confidence`；`abstain` 可配其余），并直接点破这个失败模式，禁止
"一边 answer 一边报 target_missing/low_confidence"。新增
`test_vlm_prompt_states_decision_reason_pairing`
（`backend/tests/test_agent_vqa_schema.py`）把这句话锁进回归测试，防止未来
改动悄悄删掉。这是 prompt 侧修复，不改校验器：非法输出仍会被拒绝并计入
`invalid_output`，不会被这次修改静默判为合法。

验证：`pytest backend/tests`（106 passed）、
`pytest scripts/benchmarks/test_agent_vqa_benchmark.py`（9 passed）、
`bash -n scripts/benchmarks/run_agent_vqa_parallel.sh`、`git diff --check` 均通过。

### spatial 子集 GPU 复核（同一 test split，17 题，2026-08-21）

目录：`runs/benchmarks/cja_agent_vqa/d5_promptfix_spatial_verify_shard0of1/`，
`--split test --qtype spatial`，与 pilot100 同一题库、同一 17 道 spatial 题，
仅换了本次 prompt。GPU1，未动 holdout。

`invalid_schema_errors`（决定性对照，pilot100 全量 vs 本次 17 题子集）：

| config | pilot100 `decision_reason_mismatch`（100 题内） | 本次（17 题内） | abstain_rate 17题内：修前→修后 |
| --- | ---: | ---: | --- |
| A0_HOLD | 3+1+1=5 | 0 | 0.353 → 0.000 |
| A2_ALWAYS | 4+2=6 | 1（另 1 条 `abstain_decision_mismatch`，修前已存在，非本次引入） | 0.353 → 0.118 |
| A3_ENTROPY | 3+1+1=5 | 0（1 条 `abstain_decision_mismatch`，同上，非本次引入） | 0.353 → 0.059 |
| V0_RAW | 1+1=2 | 0 | — |
| V1_STRUCT | 0 | 0 | — |

结论：`decision_reason_mismatch` 是 pilot100 无效输出的主因，prompt 补上配对
规则后基本消失（17 题子集里仅剩 1 条，且是新样本非重复失败）；`abstain_decision_mismatch`
是修复前就存在的另一个更小的独立失败模式（pilot100 全量里也只有 2 条），
不是这次改动引入的回归，保留作为已知的、更低优先级的开放问题。spatial
accuracy 在 17 题内于 0.12–0.18 波动，属小样本噪声，未见系统性变化——这次
只是无效输出/弃答率修复，不是答案质量的改进，spatial 坍缩为主失败模式的
判断（见上文“答案坍缩”）不受影响。

**下一步（未执行，待确认）：**
1. 提交当前 dirty 工作区（`backend/agent_vqa.py` 未改，只有
   `backend/vlm_analyzer.py`、两个测试文件、以及此前已完成的重观测审计相关改动）。
2. 提交后启动正式 n=100 holdout（一次性）。

## 下一道强制门槛

1. ~~D6 配置集合~~ ——已冻结为 V0/V1/A0/A2/A3，不含 A1/A4/A5（见上）。
2. ~~是否先修 `decision_reason_mismatch`~~ ——已修复 prompt，并在 test split 17
   道 spatial 题上完成 GPU 复核（见上）：`decision_reason_mismatch` 基本消失，
   accuracy 无系统性变化。
3. ~~在 test split 上重跑验证~~ ——已完成，结论见上。
4. 提交当前 dirty 工作区（含本次 prompt 修复与回归测试），再启动正式 n=100
   holdout（一次性，只用于确认 A2 vs A0 的 +4pp，不因 holdout 结果反过来调参、
   改 prompt 或改题库）。**本状态文档写作时尚未提交、尚未启动 holdout，等待作者/
   协作者最终拍板。**
5. D1 的 LOEO / 多 seed 仍是投稿阻断，与 Agent-VQA holdout 并行，不互相替代。
