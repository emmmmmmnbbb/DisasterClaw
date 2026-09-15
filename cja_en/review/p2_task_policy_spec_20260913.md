# P2 task-conditioned policy specification (development freeze candidate)

Status: **candidate parameters; freeze only after the complete 160-question development audit passes.**

## Information boundary

The online policy may use the operator question, its declared ROI, the marked damage-target reference/coordinate, current UAV pose, predicted ChangeOS boxes/probabilities, and earlier online observations from the same episode. It must not read the answer, damage subtype annotation, building polygon annotation, or a future view.

## Utility and action rules

For a proposed descent,

`utility = task_uncertainty × expected_detail_gain - 0.05 × flight_time/60 - 1.0 × coverage_loss`.

The candidate gate requires uncertainty at least `0.5`, utility at least `0.05`, and—on count/spatial tasks—predicted ROI coverage at least `0.98`. One descent step is `443.4 m`; the altitude floor is `443.4 m`; the episode budget is two reobservations.

- **damage:** entropy is computed from the marked building's binary probability. A visible marked point with no localization match receives uncertainty 1.0. The vehicle descends and moves toward the question marker, with horizontal displacement capped at 40 m per step. A detector box may be associated by containment or within 15 m of the marker; the old 8%-of-image threshold is forbidden.
- **presence:** an observed positive or validated current-view negative is answered at the wide view. Missing-target navigation remains the search controller's responsibility; damage entropy must not shrink the query view.
- **count:** uncertainty is normalized entropy of the Poisson-binomial answer distribution over `0/1/2/3+`, not the maximum entropy of one building. The vehicle descends vertically and only when the complete ROI remains covered.
- **spatial:** uncertainty comes from the selected target's binary probability. The vehicle descends vertically and only while target plus reference ROI remain jointly covered. The answer is computed against the registered ROI centre rather than the local crop centre.

Every decision logs task uncertainty and its source, detail gain, predicted ROI coverage/loss, estimated incremental flight time, utility, motion mode, and a stop/trigger reason.

## Cross-view evidence

- Damage sightings are keyed by the public target `ref_id`.
- Other predicted buildings receive episode-local geographic track IDs. Association uses predicted geographic bbox IoU (`>=0.2`) or a predicted-centre radius of `6 m`; these are not xBD ground-truth IDs.
- All sightings and observation IDs are retained. Binary probabilities are averaged with weight proportional to inverse effective GSD, so a later view does not unconditionally replace an earlier view.
- Count evidence is deduplicated geographically; spatial target selection uses registered coordinates relative to the ROI centre.

The 6 m association radius must be inspected on the development run for dense-building merges and cross-scale fragmentation before it is frozen.

## Validation record

- Unit/integration regression suite: 95 passed on 2026-09-13.
- Count smoke: valid, correct `3+`, zero reobservations after task-bucket entropy replaced maximum object entropy.
- Damage smoke before the metric association fix exposed a 33.8 px (~51 m) neighbour match; that run is diagnostic only.
- Damage rerun after the fix was invalidated by external GPU OOM before perception and provides no scientific result.
- Complete development run: waiting for a GPU with at least 22 GiB free and utilization at most 20%.


### 2026-09-13 spatial boundary and stale-track correction

A real spatial smoke case (`nepal-flooding_00000225_post_disaster__spatial_0032_7638`) exposed two distinct failure modes. First, class entropy alone cannot represent answer uncertainty near an eight-way bearing boundary, so spatial decisions now use the maximum of binary class entropy and a GSD-dependent bearing uncertainty. Second, a coarse-view false positive persisted in geographic memory after disappearing from the finer full-ROI observation. Historical tracks remain logged and their probabilities may be fused when re-detected, but count/spatial/presence answers only use tracks visible in the current observation. This prevents stale detections from permanently dominating nearest-object or count queries.

The pre-fix v3 smoke used both reobservations but remained Southeast; the v4 smoke returned South and passed the task-policy audit with zero violations. The item carries `ambiguous_neighbour`, so this smoke validates controller mechanics only and is not evidence of final-set quality.
