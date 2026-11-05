# GLM Decode Indexer cost correction (2026-11-05)

The reported 128K/1K comparison was reproduced on `248dfd8`: A3 560T
DP8 TP2 EP16 selected B9 at 3.58472 QPS, while A5 850E DP8 TP1 EP8
selected B32 at 15.44850 QPS. Both used MTP5, explicit accepted output
3.5 tokens/cycle, a shared prefix, 80% memory utilization, and the mean
ITL/TPOT limits 70/20 ms. Neither was memory limited.

These were cross-hardware estimates, not observations on those devices.
The source observations were collected on A3 800T DP4 TP4 EP16. The
transfer uses source observed time multiplied by the ratio of target
and source-proxy operator costs. An incorrect source operator cost thus
also changes the target's transfer multiplier.

## Corrected operator costs

1. **BF16 matrix inputs.** `pool_key_indexer` receives BF16 queries and
   pooled keys. Its matrix cost must use the BF16 rate, while FP32
   accumulation and vector reduction do not make the matrix inputs FP32.
   The old model used the FP32 matrix rate. Head reduction and top-k
   remain on the FP32 vector budget.
2. **Query-head matrix shape.** The indexer has 32 replicated query heads
   of width 128. For each request the score product has
   `M = local_query_tokens * index_n_heads`, `K = index_head_dim`, and
   `N = pooled_context`. The old occupancy heuristic omitted the heads.
   The heuristic now uses that matrix M dimension. It remains an
   estimate, not a measured kernel utilization curve. Indexer heads are
   still replicated across TP; they are not attention heads divided by TP.
3. **Incremental KeyPool writes.** KeyPool compression reads new keys
   and bounded FP32 K/gate tail state. It does not rescan the entire
   historical pooled cache at each decode update. Historical pooled-key
   reads now belong to index scoring, with two bytes per BF16 element.
   A bounded tail and query/weight input traffic are included separately.

The audited upstream implementation is
[vllm-ascend indexer_kpool.py, revision 0f2eee9](https://github.com/vllm-project/vllm-ascend/blob/0f2eee9dfc5b4567c83752abb6b81ae302f83a05/vllm_ascend/attention/indexer_kpool.py):
the cache dtype checks, `_key_pool` inputs, and `query =
q_values.to(indexer_cache.dtype)` / `weights.to(torch.bfloat16)` call
establish the input precision and incremental cache path. The
[upstream reference test](https://github.com/vllm-project/vllm-ascend/blob/0f2eee9dfc5b4567c83752abb6b81ae302f83a05/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_pool_key_indexer_triton.py)
uses 32 heads and forms per-head query/key products before head reduction.

## Result and limits

| Configuration | Previous batch/DP | Previous QPS | Corrected batch/DP | Corrected mean ITL | Corrected mean TPOT | Corrected QPS |
|---|---:|---:|---:|---:|---:|---:|
| A3 560T DP8 TP2 EP16 | 9 | 3.58472 | 11 | 69.42496 ms | 19.83570 ms | 4.33670 |
| A5 850E DP8 TP1 EP8 | 32 | 15.44850 | 24 | 68.86559 ms | 19.67588 ms | 9.53875 |

Both now stop at a latency constraint: the next batch exceeds 20 ms
mean TPOT and 70 ms mean ITL. The UI reports this actual limiting
condition rather than labeling every candidate search with a calibration
cap as coverage limited. It also shows `DP * batch` concurrent requests
and the scanned batch range.

The old fixed-B32 A5 estimate got faster from 16K to 128K input
(20.91516 to 16.19862 ms TPOT). The corrected model no longer has that
inversion. Tests cover multiple fixed batches and intermediate lengths.

No hardware peaks, acceptance rates, source training/validation anchors,
or transfer formula were changed to target a desired A3/A5 ratio. The
remaining difference combines the existing hardware profiles with their
different feasible batches under the same latency limit. It is not a
validated hardware speedup. Source residuals, occupancy, fused-kernel
efficiency and communication still require matching target-device traces.

The existing MTP QPS convention is unchanged: the output includes the
first prefill-produced token, so decode duration uses `(output - 1) *
mean_TPOT` for this 1024-token case. The visible preset estimates and
their regression baselines were refreshed. Previously exported Excel
files were not modified by this repository fix.

Validation:

```text
node tests/decode-indexer.test.cjs
node tests/calibrated-hardware.test.cjs
node tests/prefill-options.test.cjs
node tests/weight-shapes.test.cjs
```

The checks cover the reported cases, BF16 sensitivity, query-head
occupancy, bounded append traffic, context growth, source-anchor
isolation, scan boundaries, all 14 form presets, Prefix Cache/PCP,
weight dimensions and embedded script syntax. No image or NPU access
is required.
