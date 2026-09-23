# MESSI case protocol, frozen before formal model outputs

## Material Passport

- Origin: `shuanglan/MESSI航空应用补充案例执行计划.md`, implemented in this worktree
- Version: `messi_vegetation_v1`
- Freeze point: after four `100_0001` pilot samples and before any formal Qwen call
- Verification status at freeze: sample/GT images visually inspected; formal inference pending

## Task and sample rule

The question is whether a fixed ground region contains **visible tree or shrub
canopy**. The official MESSI `vegetation` class (ID 12) is supporting evidence,
and both RGB observations were checked by eye before model outputs. This replaces
the vehicle question during pilot preparation because the three other downloaded
sequences did not offer stable positive vehicle regions at the selected height
pair. MESSI `soft terrain` or a shadow alone does not make a positive answer.

`100_0001` is pilot only. The formal set is twelve nonoverlapping 760 × 760
low-frame regions, four each from `100_0002`, `100_0003`, and `100_0042`.
Each sequence has two positive and two negative regions. They were selected
by paired annotation fractions and visual review of high/low crops, before
formal inference. No adjacent frames or repeated generations inflate N.
The selected positions and answers are recorded in `manifests/formal_selection.json`.
Nearby regions within a sequence remain correlated; three sequences are not
three independent disaster environments.

The frame in each sequence nearest 100 m relative to takeoff is the high
observation, and the frame nearest 30 m is the low observation. These images
were acquired at different heights; frame order is not interpreted as a
recorded descent decision. SIFT matching and a RANSAC homography map the
low-frame square to the high-frame quadrilateral. The high crop is its
axis-aligned bounding rectangle; perspective and planar approximations leave
some coverage mismatch. The high bounding rectangles enclose 2.8--15.9% more
image area than their mapped quadrilaterals in the twelve selected samples.
The correspondence was reviewed on paired RGB crops.
The CSV `rel_alt` value is reported as relative height, not exact height above
the local ground.

## Model inputs and branches

All images come from original RGB JPEGs. The first image is the high frame,
resized within 1200 × 800 pixels and outlined in red around the fixed ROI.
The second A image repeats those exact pixels. B is the mapped rectangle
cropped from the **original high JPEG** and resized to 800 × 800. C is the
corresponding square from the original low JPEG, resized to 800 × 800. B and C
therefore share a similar ground footprint and output size. A retains the wide
field of view and serves as the repeat-call control.

The first Qwen output is copied into the history of all three branches; no
branch sees another branch's second output. The second prompt and invocation
budget are identical. The model is the local cached
`Qwen/Qwen2.5-VL-7B-Instruct` revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, greedy decoding,
16 output tokens, seed 0, no extra retry. The exact prompts and image
mapping are constants in `scripts/run_case.py`. The model sees no GT mask,
reference answer, label-bearing path, or filename. An answer parses only if
it is exactly YES, NO, or UNCERTAIN (optional terminal punctuation); all
other completed outputs count as invalid and incorrect. UNCERTAIN counts
incorrect and is reported separately. A runtime failure is logged and is
excluded from the three-branch complete-case contrast, with its frequency
reported. The planned budget is one first call plus A/B/C second calls per
sample, 48 calls total. This is an offline observation review, not an
autonomous action-selection or flight test.

## Frozen file digests (SHA-256)

| File | SHA-256 |
|---|---|
| `manifests/formal_selection.json` | `28b9281dd7f7f586b76d39a7fbbdcd1660d202004885e92a9c1c6302` |
| `manifests/formal_samples.jsonl` | `a44ae8fdd0469a94b6603c50fe734f0fad4ca68d8815f4fab8c1d4b682ee1889` |
| `manifests/formal_ground_truth.jsonl` | `27c1be0aa88b240f905a51b13c692833e334aa3b23e2243dc715bb2bb49f4efc` |
| `scripts/run_case.py` | `a573a590d9c9c29f3169e87c9f79a0f8837626d956bf7c97a2e8d31757b8b7fb` |

The sample manifest also carries individual input PNG and original ZIP hashes.
Formal ground truth is stored separately from the inference manifest.
