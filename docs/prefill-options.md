# GLM 纯 P：Prefix Cache 与 PCP

第四页的 GLM 纯 P 配置支持命中率 0–100%，PCP 1/2/4/8/16/32/64。默认 0%、PCP1 保留原计算结果。理论容量、算子估计和实测估计均可设置；没有同配置实测的新场景使用解析估计。

## 计算口径

- 命中率表示每个请求可复用的前缀 token 比例，不是命中请求占比。`cached = min(input - 1, floor(input * hitRate / 100))`，`computed = input - cached`。100% 命中仍重算最后一个 token，并执行一次 LM head。
- 从已命中位置开始分块计算后缀。Attention / Indexer 仍读取完整历史，KDA 读取已有状态。假设 KV、KDA 状态已在命中边界驻留；不模拟缓存预热、淘汰、远程拉取及混合模型的实际保留边界。输入命中率应是这些条件约束后的有效复用比例。
- 原输入 tok/s 保留，另显示实际计算 tok/s。QPS 始终按完整请求数计数。
- `world = nodes * ranksPerNode = DP * TP * PCP`，EP 为 1 或 world。固定卡数增大 PCP 会减少 DP，因此不保证提高 QPS。
- 普通投影按 PCP 本地 query 估算；SFA / Indexer 采用均衡 query 分配与双 chunk padding 的工作量近似，保留全局 KV 读取。MoE 路由按全局逻辑 token 计算一次。
- GLM KDA 汇聚完整序列。TP=1 时使用 PCP 分头，否则仍按 TP 分头，不假设 TP×PCP 联合分头。计入 KDA 输入汇聚、输出归约和 SFA KV / 索引交换成本。通信沿用原有带宽与拓扑代理模型，不能作为实测加速比。
- DCP 固定为 1，PCP 不直接将 KV 容量除以 PCP；命中缓存仍占显存。不根据命中率推断不同请求共享同一份物理缓存。TP=1 的 KDA 权重和状态按 PCP 分头，会影响物理池的 padding 大小。
- 命中率非零或 PCP>1 时禁用纯 P 冷启动实测残差与同配置对比，纯 D 保持独立。实测 case / 默认预设会恢复命中率 0%、PCP1。导出 JSON 包含输入值、有效命中 token、实算 token、布局和假设。

当前独立 HTML 内的 msmodeling 适配器没有这两项扩展的能力协议。非默认配置会明确提示切换本地解析口径，不会静默提交不支持的参数。DeepSeek 的独立性能引擎不在本次 GLM 改动范围内。

## 依据与验证

- [vLLM Ascend Context Parallel 设计](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/context_parallel.html)：rank 布局、query 划分、DCP1 的完整 KV。
- [GLM KDA 实现](https://github.com/vllm-project/vllm-ascend/blob/0f2eee9dfc5b4567c83752abb6b81ae302f83a05/vllm_ascend/models/glm5next/kda.py)：`get_kda_parallel_size` 与完整序列汇聚。

运行 `node tests/prefill-options.test.cjs` 与 `node tests/weight-shapes.test.cjs`。前者覆盖命中边界、分块逐次求和、并行资源守恒、KDA 分头、KV 驻留、D 隔离、校准范围、原生接口保护和导出。可追加旧 HTML 路径，验证默认参数的数值兼容。
