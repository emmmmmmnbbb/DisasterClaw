# Supplementary random-budget evaluation

The selection manifest was written before supplementary inference. This is a
post hoc supplement to the published-in-draft controller result, not an
independent preregistered test. Seeds 0–19 each sample exactly 20 of the sorted
409 eligible (split, sample_id) pairs uniformly without replacement, using
Python's `random.Random(seed).sample`. All seeds are reported. Selection reads
only sample IDs and split membership, not annotations or lower-view predictions.
Selection SHA256: 357f1558b40ec8c998064da4e190707be5eb0370b1751c5d346ea95904f133bd.

The original high predictions are shared. Existing YOLO/SegFormer weights,
vegetation mapping, AgentVqaController evidence/answer/stop paths, registered
lower frame, and one-action allowance are retained. Only the selection rule is
replaced by the frozen random membership. Unselected episodes retain the archived
high answer; selected episodes run the controller and release the lower crop
only after its action callback. Each policy therefore has 409 initial
observations plus 20 additional observations and deterministic answers. No Qwen
calls are made. This matches observation counts, not flight energy or time.

A preliminary CPU run was interrupted at the user's request; its partial logs
are retained separately and are not pooled into the GPU comparison. The supplied
GPU launcher writes to `supplement_random_budget_gpu`. The original 20
entropy-triggered regions are re-inferred on the same selected GPU as the random
comparator. Any disagreement with archived outcomes is reported. A single
per-region deterministic controller outcome is reused across seeds; these are
simulated seed-level policies, not independent repeated model inferences. No
cross-region evidence is shared. Raw traces, model hashes, and scores are saved.
The supplied launcher fails if CUDA is unavailable and does not fall back to CPU.

Primary summary: final correct answers out of 409 per seed, mean, sample standard
deviation, range, paired corrections/harms, and comparison with the original
entropy-selected set rerun on CPU. Seed spread measures selection variability
on this fixed cohort, not generalization confidence. No thresholds are tuned.

Cost audit separately summarizes archived benchmark observation/answer-call
records, dedicated and search actions, geometric motion between logged states,
and software wall time. Missing terminal motion is flagged, not imputed. Software
latency is not flight time. MESSI cached first predictions must not be counted as
fresh inference or used to claim end-to-end speed advantage.
