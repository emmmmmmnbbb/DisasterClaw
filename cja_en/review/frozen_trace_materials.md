# Frozen matched-view trace materials

This file supplies the source records for the proposed two-row main figure. It does not claim that a new experiment was run, and it does not replace the renderer or create synthetic imagery. Each row is an actual T1 matched-view fork from the frozen final-set report.

The shared frozen inputs are `backend/data/benchmarks/agent_vqa_final_candidate_20260915_binary_human_reviewed_validated.json` (testset hash `7179c42d4a7cb0bc8fed642a235bec3c89fca2ff642d2cf38c7b5dc5283890ab`) and `cja_en/review/changeos_p6_final_protocol_20260915.json`. The matched-view manifest also records the renderer/backend fingerprint and the strict validity gate.

## Row A: damage help

**Narrative:** a wide/intermediate observation keeps the marked building on the no-damage side; the changed floor observation switches the candidate to damaged and matches the reference answer. The fixed-view branch uses the old image and remains incorrect.

| field | value |
|---|---|
| question | `hurricane-michael_00000435_post_disaster__damage_0001_9430` |
| event / repeat / recheck index | Hurricane Michael / 0 / 1 |
| reference answer | damaged (`损伤`) |
| pre / fixed / changed answer | no damage / no damage / damaged |
| pre / fixed / changed observation ID | `vln-1285598-1789865455379567149` / `vln-1285598-1789865463665426733` / `vln-1285598-1789865476874815561` |
| fixed image gate | byte-identical to pre-recheck image: `true` |
| planned move | north 27.8 m, east 28.8 m, up -443.4 m |
| target damaged probability | pre 0.1583; intermediate/fixed 0.4219; changed floor 0.5875 |
| observation GSD | pre 1.5 m; fixed 1.0 m; changed 0.5 m |
| target normalized position | pre/fixed `[0.717285, 0.289551]`; changed `[0.860840, 0.155273]` |
| target bounding box | fixed `[713,274,756,319]`; changed `[856,140,907,178]` |
| ROI coverage | 1.0 at all logged states |
| source record | `runs/benchmarks/changeos_binary/revision_matched_view_repeat0_shard1of4_full_20260920/episodes.jsonl` |

**Suggested panels:** pre-recheck wide view; fixed-view diagnostic (reuse the pre image, annotate the fixed candidate); changed floor view; a small evidence strip showing damaged probability `0.158 → 0.422 → 0.588` and answer `no damage → no damage → damaged`.

## Row B: spatial harm

**Narrative:** the fixed branch preserves the correct north answer, while the changed observation selects a different predicted damaged target and returns southwest. This is a harm transition despite full logged ROI coverage.

| field | value |
|---|---|
| question | `hurricane-michael_00000306_post_disaster__spatial_0092_5452` |
| event / repeat / recheck index | Hurricane Michael / 0 / 0 |
| reference answer | north (`北`) |
| pre / fixed / changed answer | north / north / southwest |
| pre / fixed / changed observation ID | `vln-1259362-1789865002197335637` / `vln-1259362-1789865007950442518` / `vln-1259362-1789865018724754002` |
| fixed image gate | byte-identical to pre-recheck image: `true` |
| planned move | north 0 m, east 0 m, up -443.4 m |
| selected damaged target confidence | pre/fixed 0.680; changed 0.667 |
| selected target coordinate | fixed `[30.15523619, -85.62757144]`; changed `[30.15474313, -85.62771168]` |
| target normalized position | fixed `[0.587891, 0.661133]`; changed `[0.618653, 0.794922]` |
| observation GSD | pre 1.5 m; fixed 1.5 m; changed 1.0 m |
| ROI coverage | 1.0 at both logged T1 states |
| source record | `runs/benchmarks/changeos_binary/revision_matched_view_repeat0_shard0of4_full_20260920/episodes.jsonl` |

**Suggested panels:** pre-recheck ROI with the selected target and north bearing; fixed-view diagnostic retaining the same target; changed intermediate view with the shifted target and southwest bearing; a small answer strip showing `north → north → southwest`.

## Figure construction constraints

1. Label the figure as a frozen matched-view diagnostic, not a new terminal-episode experiment.
2. Show the fixed branch as a diagnostic duplicate of the old view or mark it with a `fixed-view rerun` badge; do not imply that it is a second physical image.
3. Keep the observed image IDs, GSD, target coordinates, and candidate answers in the source/alt text or supplementary caption.
4. Use the same panel order for both rows: pre-recheck, fixed-view diagnostic, changed view, answer/evidence strip.
5. The source JSONL/log directories contain observation IDs and detector/evidence metadata rather than colocated raster files. The corresponding rendered PNGs and detection overlays were recovered from `backend/outputs/uav_view_revision_matched_repeat0_shard{0,1}_full_20260920/` and copied into `cja_en/review/frozen_trace_raw/`; pre/fixed images are byte-identical in each matched control.
