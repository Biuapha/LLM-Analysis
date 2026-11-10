# Page 6: MTP5 full-depth ITL calibration

The page-six EP8 curves now pass through the verified full-depth B28 profile
cycle measurements: 850 = 100.08025 ms, 850E = 62.041625 ms. The previous
reduced-depth projections at B28 were 123.532428 and 91.778531 ms.

For each hardware, every original EP8 point is multiplied by the measured
B28 cycle duration divided by its original B28 prediction. This preserves
the batch trend; the factors are 0.8101536705222848 and 0.6759927852743857.
The stars mark the two measured inputs. Other batches remain predictions.

Each GMM contribution is independently calibrated with its full-depth B28
mean kernel-duration sum (8 ranks, 20 cycles, 47 calls per operator/cycle):

| Hardware | GroupedMatmulSwigluQuantV2 ms/cycle | GroupedMatmul ms/cycle |
| --- | ---: | ---: |
| 850 | 18.3941316375 | 9.21206706875 |
| 850E | 9.1163062375 | 4.36945876875 |

GMM(B) is its original batch-dependent estimate multiplied by measured
full-depth GMM(28) / original GMM(28), separately for each type and hardware.
EP16 = calibrated EP8 - 0.5 * calibrated GMM sum.
EP32 = calibrated EP8 - 0.75 * calibrated GMM sum.
The residual EP8 minus both GMM contributions is unchanged between EP levels,
as requested. This is an additive sensitivity assumption; overlapping kernels
are not generally additive wall time, and EP16/EP32 were not rerun.

The full-depth anchors were captured on 2026-11-06, with B28 per DP,
DP8/TP1/EP8/MTP5, 131072 input tokens and 4096 requested output tokens.
The metric is same-layer KDA start-to-start spacing: every 34 KDA kernels,
20 cycles / 19 intervals per rank, median of eight rank medians. It includes
profiling impact, and is not a client arrival-timestamp measurement. Raw
kernel CSVs for all sixteen ranks were rechecked against these summaries.

The independent unprofiled streaming-derived cycles were 97.995742 and
61.163304 ms. These are references only and are not mixed with the profile
anchors. Neither the chart nor its EP extrapolation divides by a token step.

The reduced sweep is from 2026-11-10 with 1024 output tokens and different
hosts/configuration. Single-point calibration cannot establish the true
full-depth B1-100 client ITL or memory feasibility. Page-six labels state this.

Published numerical evidence:

- `data/glm53-mtp5-itl-b28-calibrated.csv`: all 200 hardware/batch rows.
- `data/glm53-mtp5-itl-b28-calibration.json`: anchors, factors, formulas and limits.

Top-10 operator data and pages 1-5 are unchanged. The standalone six-curve
asset and earlier combined Top-10/EP8 asset both contain the corrected ITL.
