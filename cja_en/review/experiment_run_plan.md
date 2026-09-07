# 补实验执行记录（匹配预算 + 不动重问）

日期：2026-09-06。GPU 空闲后按此执行。**只新增子目录，绝不覆盖任何原始结果；不改 backend/ 与 scripts/ 源码。** 两个实验均需本地 Qwen2.5-VL-7B-Instruct（已在 `~/.cache/huggingface`），需空闲 GPU。

前置环境：`/home/lc/miniconda3/envs/disasterclaw/bin/python`；`.env` 已配 `BASE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct`（本地加载）。

## 实验 1：匹配预算重跑（修正旧续跑缓存导致的预算不匹配）

旧 `paper_cja_mech_final` 因 `--resume --allow-crash-resume` 复用了基于旧参考预算的消融记录，导致 14/15/11 次动作与参考不一致。本轮在**全新 tag、全新 out-dir** 上一次性重跑，参考 `A5_EXPECTED` 先跑、随后 matched 配置沿用同一参考预算。

```bash
cd /home/lc/disasterclaw
# 2 卡分片并行；fresh tag 保证全新 out-dir，--allow-crash-resume 仅用于崩溃续跑（首次无旧记录，不会串）
# ⚠️ 必须显式传 --testset 与 --frozen-manifest：final 题集 eval_role=final，缺一则会静默回退到 80 题旧题集
bash scripts/benchmarks/run_agent_vqa_parallel.sh \
  "A5_EXPECTED,A0_HOLD,A1_RANDOM_MATCHED,A2_FIXED_MATCHED,AB_CENTER,AB_DESCEND,AB_FULL" \
  all 2 paper_cja_mech_final_rerun \
  --testset /home/lc/disasterclaw/backend/data/benchmarks/agent_vqa_final_v2.json \
  --frozen-manifest /home/lc/disasterclaw/runs/benchmarks/paper_cja_mech_v1/frozen_manifest.json \
  --allow-crash-resume
```

- 输出：`runs/benchmarks/cja_agent_vqa/paper_cja_mech_final_rerun_shard{0,1}of2/`
- 7 配置 × 160 题 = 1120 episode；参考 A5_EXPECTED 先跑，`reference_budget_by_qid` 由其逐题 `n_reobservations` 得到。
- 主终点：任务正确率、纠正/破坏数（`paired_tests.json` 的 discordant counts + McNemar）。

合并分片并出报告：

```bash
python scripts/benchmarks/report_agent_vqa.py \
  --runs runs/benchmarks/cja_agent_vqa/paper_cja_mech_final_rerun_shard0of2 \
          runs/benchmarks/cja_agent_vqa/paper_cja_mech_final_rerun_shard1of2 \
  --out runs/benchmarks/cja_agent_vqa/paper_cja_mech_final_rerun_reports
```

## 实验 2：不动重问（位置不动、同一观测重复问 VLM）

复用论文实验同一 `vlm_answer_fn → VLMAnalyzer.answer_image_question`（temperature=0.1，evidence_level=state），对每题 step 0 观测连续问 K 次，统计答案翻转率，隔离 VLM 采样随机性。

```bash
cd /home/lc/disasterclaw/backend && set -a && source ../.env && set +a && \
  PERCEPTION_DEVICE=cuda:0 VLM_LOCAL_DEVICE=cuda:0 HF_HUB_OFFLINE=1 \
  python ../cja_en/scripts/no_move_reask.py --k 5 --evidence-level state
```

- 输出：`cja_en/runs/no_move_reask/episodes.jsonl` 与 `summary.json`（翻转率、按题型、决策翻转率）。
- 建议先 `--limit 4` 冒烟确认流水线，再跑全 160 题。

## 执行顺序建议

1. GPU 空闲后先跑两个 `--limit 4` 冒烟，确认模型加载与路径无误、估单题耗时。
2. 实验 1（匹配预算）与实验 2（不动重问）可在不同卡上并行，或先后执行。
3. 结果落盘后，用 `report_agent_vqa.py` 与 `summary.json` 分析，据结论决定论文中策略贡献的留/删，再改 `cja_en/` 正文。
