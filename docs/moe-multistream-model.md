# v10：共享专家多流与统一理论口径

当前默认已更新为 [v13 A3/A5 校正](a3-a5-correction-v13.md)。本文保留历史版本说明，默认模式与数值以 v13 为准。

默认入口、总表预设及六条 850/850E 曲线采用同一硬件算子模型，开启共享专家多流，不叠加跨硬件迁移系数或本机实测时延系数。这样比较 EP、硬件或 batch 时，不会在 EP8 与 EP16 之间切换校准口径。实测校正仍是用户可单独选择的模式。

## 调度与资源约束

实现位于 `src/moe-multistream.js`，嵌入独立 HTML。依据测试镜像的 vLLM-Ascend 提交 `e2e591b166b73a85ec3f9b7c5de229dde84432d8`：

- 共享专家输入量化与 Router 重叠；Gate-Up 等待 Router 输出。
- 非融合 MC2/All2All 的共享激活等待 Dispatch 完成；AllGather 路径等待路由 GMM2 边界。
- 共享 Down 等待 Combine 开始，最后两条分支汇合。
- 同一微批的 Dispatch、路由专家 GMM、Combine 保持依赖。TP/PCP 外围通信保留，未假设 DBO 或融合 MC2。

关键路径还不得低于参与算子的总计算预算、总 HBM 访问预算；多流不能复制硬件资源。详细共享资源争用、事件与主机调度额外开销未实测，结果仍是乐观调度估计。重叠收益归入共享专家算子组的隐藏部分，分组是关键路径预算而非实测 kernel 时长。额外通信隐藏比例不能与显式多流同时开启，避免重复扣减。

P、D、MTP 目标验证与每次草稿使用同一个调度函数；支持 TP/PCP 布局及 EP 开关。并行存活的共享专家和路由专家缓冲同时计入 FFN workspace。所有算子组保持非负，严格保留上下文积分与分块的算子成本。

源码：[共享专家事件等待](https://github.com/vllm-project/vllm-ascend/blob/e2e591b166b73a85ec3f9b7c5de229dde84432d8/vllm_ascend/ops/fused_moe/shared_experts.py#L664-L686)、[通信后端里程碑](https://github.com/vllm-project/vllm-ascend/blob/e2e591b166b73a85ec3f9b7c5de229dde84432d8/vllm_ascend/ops/fused_moe/moe_comm_method.py#L154-L191)。

## 校准、精度与容量

850E r71 训练已开启共享专家多流。因此校正模型始终以多流开启的训练配置重新拟合，再对用户选择的开关计算候选；关闭多流不会重新拟合掉额外耗时。训练与验证数据没有修改。D 新比例约 0.884724，原串行模型的 0.844454 不再直接套用。独立验证点误差：P 16K B3 +3.60%、B6 +4.72%、128K B3 +0.45%；D 16K B3 −5.41%、128K B3 −8.02%。这些小 batch 误差不构成未测区域置信区间。

800T 历史迁移未记录可核实的相同多流状态，多流开启时禁用这条迁移路径；关闭多流时保留历史对照。没有同画像实测的场景明确返回理论估计，不暗示已经校准。800T/9392 本机绝对实测参考仍保留。

自动 KV 精度由硬件画像决定，与性能模式和 EP 数无关：850E 为 FP8 KV/indexer，其他画像为 INT8 KV/BF16 indexer；可以显式选择 BF16、INT8 或 FP8。850E 统一使用已观测的 91.75 GiB 可见容量。其 EP8 同画像权重与物理页成本仍来自观测，其他拓扑按解析容量规划。

D 物理共享前缀比例与 P Prefix Cache 命中率是两个独立参数。共享比例只影响 D 的物理页数量，不缩短 D 的注意力上下文：

`共享块 = floor(floor(input × sharedRate) / blockSize)`

`总块 = 共享块 + batch × (每请求总块 − 共享块)`

每请求总块仍包含输出与 MTP 保护块。表格的 D case 采用原来明确的完整物理共享输入假设；六条曲线采用 95% 物理共享输入。命中率本身不能证明跨请求物理共享。

P chunk 在网页和总表统一解释为每 DP 总 token 预算。旧 API 的显式 `per_sequence` 只为历史对照保留。最优整数 batch 由容量、TTFT/ITL/TPOT、用户搜索范围筛选，再按 QPS 排序；实测覆盖不限制硬件模型的搜索。

## 验证

`node tests/moe-multistream.test.cjs` 检查 528 组 P/D、target/draft、TP/PCP/EP、资源下界、六条曲线最优点、精度与共享前缀独立性。其他测试覆盖精确 token 积分、短输出、模型显存边界、真实表单预设及实测数据隔离。`python scripts/embed-850e-calibration.py --check` 检查源代码同步。

本次没有重新启动机器测试，也没有新增未经验证的硬件带宽、算力或网络测量值。
