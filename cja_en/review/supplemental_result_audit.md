# 补实验结果纳入审计

日期：2026-09-07。输入为 `matched_budget_result.md` 与 `no_move_reask_result.md`。本地尚无新实验的逐题 `episodes.jsonl`、`paired_tests.json` 或完整 `summary.json`，因此本记录核对摘要内部算术和论断边界，不代替逐记录复现。

## 可直接纳入的结果

- 匹配预算重跑使用同一 160 题。正确题数可还原为：hold 41、A5 42、random 45、fixed 42、center-only 40、descend-only 44、combined 42。
- A5 和仅居中执行 98 次动作；其余活动策略 97 次。分配预算逐题匹配，执行预算相差至多 1 次，因此正文使用 `near-matched executed cost`。
- A5 相对 hold 的 6 次纠正与 5 次破坏等于净增加 1 题，与正确题数一致。
- 报告声明所有两两 McNemar 检验经 Holm 校正后均不显著。论文据此表述“未检出策略优势”，不表述等效。
- 不动重问中 11/160 题答案翻转；按题型为 damage 0/43、presence 0/44、count 4/43、spatial 7/30；decision 翻转 1/160。

## 发现并处理的冲突

- 匹配预算报告中随机、固定、下降和组合配置的部分 gain/loss 差与各自正确题数相对 hold 的差不一致。正文删除这些行的 gain/loss，只保留自洽的正确题数与动作数；A5 的成对计数单独报告。
- 不动重问原报告写“158/160 完全一致”，与 11/160 翻转及四类合计冲突。应为 149/160，已在结果文档中修正。

## VLM 身份

作者确认全程使用 Qwen。实验计划和不动重问记录给出完整模型名 `Qwen/Qwen2.5-VL-7B-Instruct`，正文据此补齐模型家族、版本和规模。当前源码显示结构化问答接口使用 temperature 0.1、max tokens 300；本地 backend 的 top-p 0.9 和 repetition penalty 1.1 是默认值，因旧 manifest 字段为空，正文没有把这两个默认值冒充冻结运行事实。

## 尚待逐记录核对

取得新实验原始目录后，应核对每个配置均有同一 160 个 qid、无重复记录、correct 与题库真值一致、动作总数和预算来源一致，并重算 McNemar 与 Holm 校正。还应保存模型 revision、有效环境变量、prompt hash 和源码 commit/dirty diff。
