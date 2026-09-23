# Exploratory token-entropy MESSI replay

This variant was designed after seeing that the first MESSI adapter requested
reobservation on all 107 descend and all 95 path evaluation units. It is an
exploratory follow-up, not a replacement for that frozen run. The dataset,
reference answers, 100-to-30-m candidate frames, action limit, controller and
T1 utility formula are the same as in the exhaustive public-sample protocols.

Qwen2.5-VL-7B-Instruct is prompted to answer with one token, `YES` or `NO`.
At the next-token position, its logits for exactly those two token IDs are
normalized with a two-way softmax. The resulting `p_yes` determines the
answer and normalized binary entropy. This conditional two-token score is a
decision signal, **not a calibrated probability** over all possible text.
The first prompt uses only the marked high image. Following an allowed action,
the second prompt sees the high image, first answer, and newly released low
image. The paired hold and agent arms use an identical cached first score.
Full model scores, action trajectories and image-read audits are logged.

The uncertainty trigger is 0.90. It was chosen using only the four excluded
`100_0001` pilot items: their token entropies were 0.726, 0.997, 0.840 and
0.933, yielding two rechecks and two stops. The cost weight remains 0.05,
cost scale 60 s, minimum utility 0.05 and maximum rechecks one. The adapter
still maps the vegetation presence task onto the gate's marked-target damage
branch; ChangeOS and the damage-specific assessor are absent. This tests an
adapted decision rule on real imagery, not the unmodified disaster policy or
flight dynamics.

Separate `exhaustive_descend` and `exhaustive_paths` runs use the fixed
manifests and preserve every item. Each run has its own configuration and
`episodes.jsonl` under `runs/agent_replay_logits_{split}/`; scoring writes
`results/agent_replay_logits_{split}_summary.json`. The no-new-observation
arm is the paired control for this variant. Its accuracy should not be
compared directly with the prior self-reported-confidence arm as an isolated
policy effect, because the first-answer prompt and scoring method also
changed.

```sh
python experiments/messi_case/scripts/run_agent_replay_logits.py --split exhaustive_descend --device cuda:3
python experiments/messi_case/scripts/run_agent_replay_logits.py --split exhaustive_paths --device cuda:3
python experiments/messi_case/scripts/score_agent_replay.py --split exhaustive_descend --variant logits
python experiments/messi_case/scripts/score_agent_replay.py --split exhaustive_paths --variant logits
```
