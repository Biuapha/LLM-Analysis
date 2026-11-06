# Page 5: GLM-5.3-Flash 850 / 850E comparison

Open `glm53_prefill_decode_operator_block_diagram.html#comparison` or the fifth tab.
The page presents the frozen v13 curves with a per-hardware multiplicative
correction to the user-confirmed B28 / DP8 / EP8 ITL anchors: 850 = 100 ms,
850E = 62.4 ms. The factors are 0.9600549953756905 and 1.2536630749579762.
Apply the same hardware factor to EP8, EP16 and EP32, preserving each curve's
shape, hardware settings and capacity model. The page-four simulator is unchanged.

The figure retains all six QPS labels, recalculated constrained optima and two
B28 anchor markers. The legend toggles curves and corresponding labels; the
slider/click inspector and CSV, JSON and SVG exports use the corrected data.
TPOT <= 30 ms corresponds to ITL <= 105 ms under the assumed ratio of 3.5.
850E EP8 stops at the inherited B41 memory boundary, now at 21.698 ms.

The anchors came from the 4K-output profiling workload. The chart retains the
128K-input / 1K-output QPS convention and TPOT = ITL / 3.5. This TPOT is derived,
not streaming-measured TPOT. Only the two B28 EP8 anchors are measured inputs;
other batches and cross-node EP16/32 are extrapolated. In particular, a whole-
curve factor also scales modeled communication; it does not calibrate the network.

Capacity assumptions remain 80% memory utilization, 4 GiB reserve, 95% physical
prefix sharing, 850 INT8 KV and 850E FP8 KV. The B28 run used FP8 KV on both
hardware, 92.96875% actual cache hits and different memory settings. No capacity
endpoints were revalidated. DeepSeek and the page-four engine are unchanged.

## Rebuilding

Keep the frozen baseline in `data/decode-comparison-v13.json`. The published
corrected report is `data/decode-comparison-b28.json`. Editable chart assets are
`src/decode-comparison.{html,css,js}`. Run:

```sh
python scripts/embed-decode-comparison.py
node tests/decode-comparison.test.cjs
```

The embedder is idempotent. Cache reference data and comparison data are loaded
through synchronous local JS assets so every file uploaded is under 1 MB. Each
loader inserts the same application/json script element before its consumer;
no fetch race or external chart library is introduced. Cache reference JSON
values are unchanged. Publish the HTML and `data/` assets together.

The test independently reproduces the baseline stage results, verifies all 540
corrected points and six constrained optima, and checks the data loaders and
interactive renderer without reading any images.

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
