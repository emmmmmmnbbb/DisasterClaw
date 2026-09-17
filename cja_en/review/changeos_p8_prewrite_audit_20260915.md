# P8 前置证据审计（2026-09-15，非正式论文完整性认证）

> 状态更新（2026-09-17）：本文保留为运行前历史清单，其中“P4/P6 未完成、final
> 待人工审核”的时间状态已经过时。当前确认性结果与 P7 成本边界请以
> `changeos_p6_p7_postrun_audit_20260917.md` 为准；原清单不得作为当前进度证明。

> 2026-09-17 P8 正文修订已实施于 `main.tex`、`sections/` 与
> `appendix/reproducibility.tex`，并已编译出 `main.pdf`。这仍是 author-review
> draft；下列原冲突清单作为修订前审计保留，不代表当前正文仍有这些错误。

此文档只记录正文重写的已知输入与依赖，不替代 P6 final 确认性实验、作者审核、
引用核验或论文最终完整性检查。

## 当前正文与新证据链的冲突

1. `main.tex` 摘要仍以历史 `xview2_first` 的 3,753 栋建筑、44 区域和四分类
   offline macro-F1 为主结果，并写“corrected closed-loop question-answering evaluation
   remains future work”。新开发集已经完成闭环 QA 诊断，但确认性 final 尚未冻结；
   摘要不能继续把旧 F1 当 ChangeOS 主结果，也不能提前写 final 收益。
2. `sections/01_introduction.tex` 仍说“corrected task-level evaluation is deferred”；
   应在 final 完成后改为按 RQ1--RQ5 组织的平台、测量协议和机制/瓶颈贡献。
3. `sections/04_methodology.tex` 把 `xview2_first` 四分类概率写作 evaluated
   backend，熵式以 `log 4` 归一化；ChangeOS 二分类主链必须重新定义损伤语义、
   `log 2` 熵、任务条件化效用、证据融合和外生动作预算。旧方法应明确标为历史诊断。
4. `sections/05_experimental_setup.tex` 仍固定旧四分类温度
   `T=3.5635948726`、APS `qhat=0.8064657026`、四分类 macro-F1 与多分类 Brier。
   这些值不能传到新的二分类 ChangeOS 实验；新的模型、标定、题集审核、seed、
   oracle 和统计方案须从冻结产物重新填写。
5. `sections/06_results_discussion.tex` 已把旧表路径迁至 `tables/historical/`，
   但整个正文结果顺序仍由历史四分类 FOV 与 allocation 诊断主导。P6 完成后
   应先回答 RQ1--RQ5，历史内容压至附录，并并列呈现 correction 与 harm。
6. `sections/07_limitations.tex` 对旧撤回实验“不做任务准确率主张”仍成立；
   但新正文须区分 same-data development 诊断与审核冻结的 final，并披露
   ChangeOS 对相关 xBD 数据的预训练暴露、正射影像、简化运动和有限事件数。

## 已有可引用的本地证据与不可越过的门槛

- P3：四片 clean 开发集动作消融，可作为机制诊断，不可当 final 主表。
- P4：43 ROI 的 cruise/intermediate/floor 二分类三档证据已完成；七策略
  预算曲线的新 shard3 尚在全片重跑，四片合并未获有效性验收。
- P5：完整 ROI GT + 规则回答器 160/160 的题库流程自检已完成；
  当前可见 GT + Qwen、完整任务 GT + Qwen 与完整覆盖参照尚未完成。
- P6：新 final **候选**为 160 题、44 个未消费 ROI，仅覆盖 3 个评估事件；
  自动检查零错误，但全部 human_review=pending（其中 88 题有歧义提示）。
  作者逐题审核、冻结 manifest 与 generation seeds 之前，不得运行确认性策略。
- P7：轨迹状态可计算几何总运动；分模块时延、搜索动作申请/分配细节和
  真实航空动作验证仍缺。论文不得把几何距离或 `d/v` 称为实测飞行航时。

只有在 P4 有效汇总、P5 分层 oracle、P6 审核冻结且多 seed 结果、P7 成本边界
齐全后，才能把 P8 正文改成最终结果链。路线选择必须依照预先规定的决策规则：
任务条件化策略若无稳定改善，保留负结果并改写为平台与机制测量贡献；
不能把熵规则或任务条件化策略继续称为已证实有效的核心方法。
