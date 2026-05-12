## GreenContext

术语解释：
GreenContext 是 NVIDIA 提供的 intra-process GPU spatial multiplexing 机制，论文用它把不同 CUDA streams 绑定到指定 SM 集合，从而在同一进程内为 prefill 和 decode 划分计算资源。论文未展开 GreenContext 的驱动内部实现，只说明其重配置成本约为一次 stream synchronization，属于微秒级。

术语关联术语的使用例子：
MuxWise 在一个进程中创建 prefill stream 和 decode stream，并通过 GreenContext 将它们绑定到不同 SM partition。这样 prefill layers 与 decode iteration 可以并发执行，同时共享同一显存地址空间和 KV cache pool。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

## SM Spatial Partitioning

术语解释：
SM spatial partitioning 指在同一 GPU 内把 Streaming Multiprocessors 按空间方式分给不同任务或阶段运行。论文中该机制是 PD multiplexing 的核心 runtime 能力：decode 获得满足 TBT SLO 的 best-fit SM 数，prefill 使用剩余 SM。

术语关联术语的使用例子：
在 MuxWise 的 dispatching policy 中，若 estimator 预测 decode batch 使用 60% SM 即可满足 SLO，则系统把剩余 40% SM 用于 prefill layers。后续随着 decode batch size 或 prefill 请求长度变化，partition configuration 会动态调整。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

## Layer-wise Prefill Execution

术语解释：
Layer-wise prefill execution 是把完整 prefill phase 拆成 transformer layer 粒度发射和调度的 runtime 机制。论文认为 LLM 天然由多层 transformer blocks 组成，因此这种拆分开销很小，却显著缩短了 prefill 的可调度单元，便于与 decode iteration 对齐。

术语关联术语的使用例子：
MuxWise 根据估计的 decode latency 和 prefill latency 计算本轮要发射的 prefill layer 数，使 prefill layers 的运行时间尽量覆盖 decode iteration。该机制还支持短请求抢占长 prefill 请求，降低 skewed workload 下的 TTFT 尾延迟。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

## Query-based Synchronization

术语解释：
Query-based synchronization 是 MuxWise 用 CUDA event polling 进行的异步同步方式，用来发现 prefill 最后一层是否完成。它避免 prefill 完成时阻塞下一轮 decode iteration 太久，从而减少 inflight batching 合并请求造成的小 GPU bubbles。

术语关联术语的使用例子：
MuxWise 持续异步发射 decode batches 和 prefill layers；当 event polling 发现某个 prefill request 完成后，系统立即把该请求合并进当前 decode batch，而不是让 decode 长时间等待 prefill 结束。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

## CUDA Graph / Graph-level Scheduling

术语解释：
CUDA Graph 是一种降低 GPU kernel launch overhead 的执行机制。论文中 decode phase 适合 graph-level scheduling，因为 decode iteration 的变化主要来自 batch size，配置空间相对有限；prefill phase 的 batch size 和 input length 变化更大，完整 CUDA graph 捕获会带来较高内存开销。

术语关联术语的使用例子：
MuxWise 优先发射 decode iteration，因为 decode graph launch latency 低于 0.5ms；prefill 则采用 layer-wise execution，而不是为大量 prefill 配置捕获完整 graph。论文还提到 piecewise CUDA graph 可用于 prefill，但 Llama-70B 在 8 张 A100 上仍约有 10ms launch overhead。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

## Contention Guard

术语解释：
Contention guard 是 MuxWise estimator 中为 decode latency 叠加的 worst-case slowdown factor，用于抵御空间 multiplexing 下不可精确控制的共享资源竞争。论文强调 GreenContext 能精确划分 SM，但不能直接管理 memory 或 network bandwidth，因此需要保守的 runtime 保护项。

术语关联术语的使用例子：
MuxWise 通过 grid-sampling offline profiling 构建 contention guard，采样 prefill 新 token / reused token、decode batch size、decode reused token 和 partition configuration。在线调度时，decode solo-run predictor 的结果会乘上该 guard，作为 SLO-aware dispatcher 的安全估计。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

## Token Budget

术语解释：
Token budget 是 chunked-prefill 中控制一次 fused execution 的 token 数上限，通常等于 prefill chunk 的 new tokens 与 decode batch tokens 之和。它是 chunked-prefill 调度的核心旋钮，但论文指出它同时影响 GPU 饱和度和 decode TBT，难以同时满足两者。

术语关联术语的使用例子：
SGLang chunked-prefill baseline 会针对 workload 和 TBT SLO offline tuning token budget。MuxWise 不再用 token budget 把 prefill chunk 与 decode iteration 绑在一起，而是通过 SM spatial partitioning 让二者独立并发。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

## FlashInfer

术语解释：
FlashInfer 是论文 baseline 中使用的高性能 inference kernel library。SGLang 的 chunked-prefill baseline 使用 FlashInfer 融合 prefill / decode attention kernel，论文将其作为强 baseline 的一部分，而 MuxWise 的贡献不在于提出新的 attention kernel。

术语关联术语的使用例子：
在 baseline 设置中，SGLang chunked-prefill 采用 SARATHI-Serve 风格 token budget，同时通过 FlashInfer 将 prefill 和 decode attention kernel 融合。MuxWise 则主要改变 runtime scheduling 和 SM partition，而不是替换 FlashInfer 的数学算子。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing
