# Page 5: GLM-5.3-Flash 850 / 850E comparison

Open `glm53_prefill_decode_operator_block_diagram.html#comparison` or the fifth tab.
The page presents the six v13 curves from capacity- and bandwidth-corrected engine commit `facb291`, using a fixed case:
128K input, 1K output, P prefix hit 95%, D physical prefix sharing 95%, MTP5,
ITL/TPOT 3.5, TP1, DP=EP 8/16/32, multistream enabled, memory utilization 80%,
4 GiB runtime reserve, batch per DP 0–90. B0 has no defined TPOT or QPS.

The figure separates memory-feasible predictions from faded over-budget
predictions. The six QPS labels and summary show each configuration's maximum
per-card QPS satisfying TPOT <=30ms, ITL <=105ms and the model's memory budget.
850E EP8 stops at B41 (17.308ms); B42 is over budget. It must not be presented as
reaching the 30ms line with an executable batch.

The legend toggles curves. The range input or chart click inspects a batch.
Downloads include CSV/JSON for all series and SVG for the visible curves.
This fixed GLM result is independent of page-four edits. The fifth tab is
disabled for the DeepSeek model; switching to DeepSeek while on page five opens
its performance page so GLM results are never labeled as DeepSeek predictions.

## Rebuilding

The original report is saved in `data/decode-comparison-v13.json`; it includes
the source engine, commit, original HTML digest, configurations and predictions.
These identify the original curve-generation engine, not the later page layout.
Editable assets are `src/decode-comparison.{html,css,js}`. Run:

```sh
python scripts/embed-decode-comparison.py
node tests/decode-comparison.test.cjs
```

The embedder is idempotent and retains the standalone HTML with no external
chart library or data fetch. The data regression test recomputes all 540 nonzero
points through the production calibrated stage entry and checks all six optima.
Any future model revision needs an intentional snapshot/version update.

850 has no native timing calibration. 850E EP8 uses sparse measured correction;
EP16/32 transfer that correction to modeled target topologies. Complete curves
are predictions. Their automatic KV dtypes differ: 850 INT8, 850E FP8. Cache-hit
rate alone does not establish physical sharing, and decode attends full context.

## Atlas 850 capacity correction (2026-11-06)

Atlas 850 now uses the user-confirmed 112 GiB capacity. Its legacy ID
`a5-128` and msModeling reference-profile name remain stable for saved-config
compatibility; `memoryGiB=112` is authoritative. Native requests with automatic
capacity explicitly send `usableGiB=112` rather than inheriting the old profile.
At that capacity-only revision, bandwidth was still 1.6 TiB/s; the subsequent
bandwidth correction is described below. Atlas 850E remains 96 GiB / 91.75 GiB visible.

The chart uses an 89.6 GiB budget (112 x 80%), including the 4 GiB runtime
reserve. 850 EP8 is feasible through B72; B73 and above are shown as
over-budget predictions. For that capacity-only correction, all three 850 optima under 30ms were unchanged
(B43/B40/B38, 1.411/1.312/1.252 QPS per card), since their peak memory already
fits the corrected budget. All 540 points were recomputed from the corrected
engine. The saved report retains its generating commit and HTML digest.

## Atlas 850 bandwidth correction (2026-11-06)

Interpret the user-provided 1.4T as decimal **1.4 TB/s = 1400 GB/s =
1,400,000,000,000 bytes/s**. The old profile used 1.6 TiB/s, approximately
1.759 TB/s, and overpredicted throughput. The default memory-efficiency
factor remains 0.8, and the physical capacity remains 112 GiB. The browser
GLM and DeepSeek models share this corrected hardware profile.

All six curves were regenerated from the current production engine. 850E
results and 850 memory feasibility are unchanged. For 850 under 30ms:

| DP = EP | Best batch / DP | TPOT ms | QPS / card |
| --- | ---: | ---: | ---: |
| 8 | 28 | 29.760 | 0.919701 |
| 16 | 32 | 29.968 | 1.043784 |
| 32 | 32 | 29.776 | 1.050539 |
