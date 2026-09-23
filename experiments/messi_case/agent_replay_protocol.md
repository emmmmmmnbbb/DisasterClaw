# MESSI offline agent replay

This follow-up is exploratory and uses the twelve regions and high/low frame
pairs already selected for the fixed-observation case. It was designed after
seeing those case results. The four `100_0001` items were used to check the
prompt and implementation; they are excluded from replay scoring. The formal
replay configuration was written before formal replay inference.

## Runtime boundary and intervention

`scripts/run_agent_replay.py` executes the repository's `AgentVqaController`
and `task_conditioned_recheck_decision`. The offline environment first returns
only the high marked image and its height relative to takeoff. The low crop is
opened and hashed by `perceive()` only after `reobserve()` returns an allowed
action and switches the environment state. The control arm sets the controller's
reobservation budget to zero; the agent arm permits one reobservation. Both
arms share the exact first Qwen response, and all other settings match.
Neither the online controller nor the Qwen prompt reads the separate ground
truth file. The only possible second observation is the preselected paired
low frame; this tests a decision to acquire it, not route or frame selection.

The task asks whether a marked region contains visible tree/shrub canopy.
The controller parses the Chinese presence question and uses its normal answer,
budget, trajectory and stop paths with `answer_mode=vlm` and no search.
The MESSI perception adapter supplies image bytes and no damage detections.
Qwen returns YES/NO and a self-reported probability of its answer. The adapter
computes normalized binary entropy and passes it to the existing T1 utility
gate. Because the original presence branch always skips local descent, the
adapter invokes the gate's damage branch for this marked binary target, with
target visibility stipulated by the red ROI and zero horizontal recentering.
The gate retains its uncertainty trigger 0.5, flight-cost weight 0.05,
cost scale 60 s and minimum utility 0.05. Its proposed descent spans the
preselected high/low relative heights (~100/~30 m). The damage-specific
`RecheckController.assess` and ChangeOS evidence are not used. These are
substantive task adaptations, so the case does not validate the unmodified
presence or disaster-damage policy, actual flight feasibility, or viewpoint
selection. Self-reported confidence is uncalibrated.

The high prompt asks for a JSON `answer` (YES/NO) and `confidence` (0.50--1.00).
After an action, the low prompt asks the same question with the initial image
and response in history. Qwen2.5-VL-7B-Instruct uses the cached revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, greedy decoding, 64-token
cap and seed 0. Invalid JSON triggers the controller's invalid-output stop.
`runs/agent_replay_formal/config.json` records the full configuration;
`episodes.jsonl` contains the full controller trajectories, model outputs,
image-read audit and stop reasons. Scoring reads the separate ground truth
only after both arms finish.

## Formal result

The hold arm answered 8/12 correctly and acquired no new observations. The
agent arm answered 10/12 correctly, correcting two answers without harms.
It acquired the low image on all 12 items; both arms ended all items with
`sufficient_evidence`. This shows the adapted controller can authorize and use
a real lower-altitude observation under a fixed offline action limit. Since it
chose reobservation for every item, these data do **not** show selective
suppression of unnecessary actions. The two-correct difference is descriptive
for this small, correlated set. The fixed-observation A/B/C results answer a
different question about information value and are preserved separately.

Reproduce with the local environment and prepared images:

```sh
/home/lc/miniconda3/envs/disasterclaw/bin/python experiments/messi_case/scripts/run_agent_replay.py --split formal
/home/lc/miniconda3/envs/disasterclaw/bin/python experiments/messi_case/scripts/score_agent_replay.py
```

The run script refuses to overwrite an existing episode log.
