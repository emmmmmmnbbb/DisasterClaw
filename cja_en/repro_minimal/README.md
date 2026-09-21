# Minimal frozen DisasterClaw episode

This self-contained package contains one **real frozen T1_TASK damage episode**
(`hurricane-michael_00000435_post_disaster__damage_0001_9430`, repeat 0), its
paired 1024×1024 xBD source images, the post-disaster building annotations,
the georeferenced scene definition, and a compact action–observation trace.

## Environment and files

The deterministic replay uses **Python 3.11 standard library only**. The
repository's full inference environment is described by `environment.yml` and
`backend/requirements.txt`; it additionally needs the frozen ChangeOS and
Qwen2.5-VL-7B-Instruct weights and a CUDA GPU. Those large weights are not in
this small package.

- `assets/pre_disaster.png`, `assets/post_disaster.png`: original paired xBD tile.
- `assets/post_disaster_labels.json`: original post-disaster xBD polygons and UIDs.
- `scene.json`: tile, ROI, affine placement, initial pose, observation model,
  asset paths and SHA-256 digests.
- `episode.json`: frozen question, target UID, observations, decisions, actions,
  candidate answers and final answer, reduced from the archived P6 episode.
- `replay.py`: validates assets, constructs the three geographic observation
  windows from recorded poses, finds the target building in the annotation,
  replays the recorded trace, and scores its terminal answer.
- `expected_output.json`: exact expected replay output.

## Run and score

From the repository root:

```bash
python3 cja_en/repro_minimal/replay.py > /tmp/disasterclaw_episode.json
diff -u cja_en/repro_minimal/expected_output.json /tmp/disasterclaw_episode.json
```

The expected score is `{"reference":"损伤","answer":"损伤","correct":true}`.
The trace has three observations at approximately 1330, 887 and 443 m, with
candidate answers `无损伤 → 无损伤 → 损伤` and actions `Fly → Fly → Report`.

**Scope:** This is a deterministic replay of a frozen platform episode and
annotation-derived scoring. It validates scene linkage, geometry, action
trace, and scoring; it does **not** rerun ChangeOS or Qwen inference. The
original frozen online run is
`runs/benchmarks/changeos_binary/p6_final_three_event_v1_20260915_repeat0_shard1of4/episodes.jsonl`.
For a fresh model inference run of the first frozen damage question (the same
question ID), activate the full `disasterclaw` environment and run from the
repository root after obtaining the frozen ChangeOS and Qwen weights:

```bash
export DATASET_MODE=xbd DETECTOR_BACKEND=changeos DAMAGE_LABEL_MODE=binary
export CHANGEOS_WEIGHTS="$PWD/backend/outputs/changeos/changeos_r34.pt"
export VLM_PROVIDER=qwen_vl_local VLM_LOCAL_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export PERCEPTION_DEVICE=cuda:0 VLM_LOCAL_DEVICE=cuda:0
python3 scripts/benchmarks/bench_agent_vqa.py \
  --testset backend/data/benchmarks/agent_vqa_final_candidate_20260915_binary_human_reviewed_validated.json \
  --review-report runs/benchmarks/changeos_binary/final_candidate_20260915_human_review_validated.json \
  --frozen-manifest runs/benchmarks/changeos_binary/final_three_event_20260915_frozen_manifest.json \
  --configs T1_TASK --qtype damage --limit 1 \
  --seed 42 --generation-seed 42000 --generation-repeat 0 \
  --out-dir /tmp/disasterclaw_live_damage_smoke
```

The full final task file is required because the frozen manifest checks its
hash. `--qtype damage --limit 1` selects the packaged question as the first
damage item. The live result is written to the specified output directory and
can be compared with `episode.json`; fresh inference may differ from the
archived trace even with the same generation repeat.
