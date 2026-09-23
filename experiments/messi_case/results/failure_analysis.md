# MESSI case result audit

## Material Passport

- Origin: frozen `messi_vegetation_v1` protocol and `runs/formal/responses.jsonl`
- Verification status: SCORED (all 48 requests completed; descriptive case only)
- Denominator: 12 fixed regions, 3 sequences, one first call plus A/B/C per region

The first high-image answer was correct on 7/12 regions. A repeated the high
image and ended correct on 5/12, B (original high JPEG crop) on 10/12, and C
(real low image) on 11/12. B and C differ on three paired regions:

| Region | Reference | B | C | Observed outcome |
|---|---|---|---|---|
| `100_0002_r20` | NO | UNCERTAIN | NO | C corrects a B abstention on paving |
| `100_0042_r16` | NO | UNCERTAIN | NO | C corrects a B abstention on paving |
| `100_0002_r02` | YES | YES | NO | C loses a correct canopy answer despite visible vegetation at crop edge |

The two C-versus-B corrections and one harm give a net difference of one
correct answer, or +8.33 percentage points on this selected set. Relative to
the common first answer, C has five corrections and one harm. A has two
`UNCERTAIN` outputs and B has two; C has none. There are no parse failures or
runtime failures. The C error in `100_0002_r02` is directly visible in the
paired crop; whether it arose from the altered canopy scale, the wider grass
area, image perspective, or generation behavior cannot be isolated here.

Per-sequence B/C correct counts are 3/4 vs 3/4 for `100_0002`, 4/4 vs 4/4
for `100_0003`, and 3/4 vs 4/4 for `100_0042`. One sequence accounts for the
net aggregate advantage. The figure pairs `100_0042_r16` (help) and
`100_0002_r02` (harm); no case was removed after inference.
