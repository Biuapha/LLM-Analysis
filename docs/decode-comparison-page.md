# Page 5: GLM-5.3-Flash 850 / 850E comparison

Open `glm53_prefill_decode_operator_block_diagram.html#comparison` or the fifth tab.
The page presents the six v13 curves from commit `580da4d`, using a fixed case:
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
