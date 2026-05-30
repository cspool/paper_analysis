论文标题：KUNSERVE: Parameter-centric Memory Management for Efficient Memory Overloading Handling in LLM Serving

LLM Serving中的KVCache的缓存策略优化，丢弃跨实例重复的KVCache。

开源仓库确认：

- 状态：已找到
- 链接：https://github.com/SJTU-IPADS/kunserve
- 说明：论文正文明确写出 KUNSERVE open-sourced 于该仓库；仓库 README 标注为该 EuroSys 2026 论文的代码仓库，包含 KunServe 的 Python orchestration、Ray、调度实验，以及 FlashTransformer C++/CUDA dense Transformer inference backend。EuroSys 2026 accepted papers 页面也列出该论文与作者信息。仓库当前更偏研究原型与可复现实验，不是生产级托管推理服务。

1、论文工作：

- 论文要解决的核心问题：LLM serving 在真实 burst workload 下会因为 GPU HBM 中 KVCache 累积而进入 memory overloading，导致新请求必须等待已有请求释放 KVCache，TTFT 出现数量级级别的尾延迟尖峰。论文报告 BurstGPT 场景中 vLLM 会出现 overloading event，即使整体 KVCache 平均需求低于容量很多；队列等待时间可能持续到已有 decode 请求生成 EOS，而长输出请求可以占用 KVCache 很久。
- 论文的主要贡献：提出 parameter-centric memory management，把传统“围绕 KVCache 做迁移、交换、丢弃”的思路改成“在过载时临时丢弃跨实例重复的 model parameters”，快速释放 HBM 给 KVCache；设计在线 drop plan 生成、基于 CUDA virtual memory management 的本地统一内存管理、协调式 KVCache exchange、lookahead batch formulation 和动态参数恢复；实现 KUNSERVE，并在 BurstGPT、ShareGPT、LongBench 等 workload 上相对 vLLM、InferCept、Llumnix 显著降低 tail TTFT。
- 论文所处背景：现代 LLM serving 通常以多个 serving instance 复制模型参数来提高吞吐，每个 instance 可能是一张 GPU，也可能是多张 GPU 通过 TP/PP 组成的最小完整模型副本。参数和 KVCache 同时占用 HBM，论文给出的典型模型参数占比为 34.4% 到 74.8%，例如 Qwen-2.5-14B 在 80 GB GPU 上参数约 28 GB，Qwen-2.5-72B 在 4 张 80 GB GPU 上参数约 136 GB。因此，在 burst 时被复制的参数本身是一块可临时借给请求状态的内存资源。

2、相对 Baseline 解决的问题与设计方法：

- Baseline 的具体问题：vLLM 默认采用 KVCache 丢弃 / recomputation 来应对内存不足，会让被丢 KVCache 的请求重新排队并重算；InferCept 这类 swap 方法把 KVCache 换到 CPU DRAM 等存储层，但新请求仍然要等 ongoing requests 释放或换出足够空间，同时 swap-out 请求的 TPOT 会变差；Llumnix 这类迁移方法把请求和 KVCache 迁到较空闲 instance，但当集群整体被 burst 压满或迁移占用目标端内存时，队列仍会阻塞。共同缺陷是它们只是在有限 HBM 内重新组织 KVCache，不能快速为所有 queued requests 创造足够额外空间，因此仍会把一部分请求留在队列里。
- 论文的设计方法：KUNSERVE 在监控到或预测到 memory overloading 时，由 global memory manager 根据 queued requests 的内存需求在线生成 drop plan，选择若干 instance group 合并，并丢弃 group 内重复的 layer 参数副本，保证 group 合起来仍有一份完整模型参数。被释放的参数物理内存通过 CUDA VMM API 重新映射到 KVCache 虚拟地址范围，使未修改的 attention kernels 仍能按连续 KVCache 地址访问。随后 global scheduler 将 queued 和 ongoing requests 重新调度到拥有完整参数覆盖的 group，用 pipeline parallelism 跨 instance 执行。
- 方法如何对冲 Baseline 缺陷：Baseline 等待 KVCache 自然释放、迁移或交换，瓶颈仍是请求状态本身；KUNSERVE 利用参数与请求无关、且跨 instance 复制这一性质，把“短期不必每个 instance 都完整持有”的参数转成 KVCache 空间，从源头减少排队。为了保证正确性，它只丢弃 group 内冗余参数，使 pipeline group 层级上仍有完整模型；为了降低性能损失，它贪心地尽量用少量 instance group 释放足够内存，并在执行阶段用 lookahead microbatch 降低 pipeline bubbles。
- 关键 trade-off：KUNSERVE 用更复杂的全局调度、跨实例 pipeline execution、KVCache exchange 和参数恢复，换取 burst 时的 TTFT 稳定性；它会提高部分 TPOT，因为请求在更大的 batch 和 pipeline group 中执行，论文在 LongBench-14B 中报告 KUNSERVE 的 P50 TPOT 比部分 baseline 高 15.8% 到 22.7%，但仍在目标 SLO 内。它的可释放内存上限受模型参数大小限制，极端且持续的 burst 仍需要 autoscaling；lookahead batch formulation 依赖过载时有足够 queued requests，正常低负载下直接等待更多请求会引入额外延迟。

3、论文实现：

- Baseline 如何实现：vLLM 使用 release v0.6.3，比较 default 配置和 PP 配置；default 每个 instance 保存完整参数，内存不足时用 recomputation，PP 配置每个 instance 释放一半参数并跨两个 instance pipeline 执行，但会引入 pipeline overhead。论文调优了 vLLM block size，最终选择 64，以平衡 fragmentation 和性能。InferCept 原始开源实现基于较老 vLLM v0.2.0，缺少 FlashAttention / FlashInfer kernels 和 chunked prefill 等优化，因此作者把自己的 scheduler 和 attention backend 集成到 InferCept 中做公平比较。Llumnix 使用 release v0.1.0，采用 load balancing 与 KVCache migration 来处理 instance 内存过载。
- 新设计如何实现：KUNSERVE 是 cluster-serving system，包含 global dispatcher、global monitor、global memory manager、distributed execution scheduler、local memory manager、network coordinator 和 GPU executor。global monitor 收集 instance load，触发 drop / restore；drop plan 生成算法用 priority queue 贪心合并 group，复杂度为 O(N log N)，目标是在释放 queued requests 所需内存的同时最小化 pipeline stage 数。local memory manager 用 cuMemCreate / cuMemMap / cuMemUnmap / cuMemSetAccess 等 CUDA VMM API 统一管理参数和 KVCache 物理内存，把被丢弃参数的物理页映射到 KVCache 尾部虚拟地址。network coordinator 把 KVCache exchange 切成细粒度 chunk，并优先让 pipeline activation transfer 通过，避免大块 KVCache 传输阻塞前向激活。
- 实验 / 实现平台：实验使用两个集群。Cluster A 为 8 台服务器、每台 1 张 A800 80 GB GPU，scale-out GPU-GPU 网络为 200 Gbps RDMA，用于较小模型。Cluster B 为 2 台服务器、每台 8 张 H800 80 GB GPU，服务器内 GPU-GPU 为 300 GB/s NVLink，服务器间为 400 Gbps RDMA，用于 Qwen-2.5-72B 等多 GPU instance。模型选择 Qwen-2.5-14B 和 Qwen-2.5-72B，二者都采用 GQA。评测 workload 包括 BurstGPT trace 的请求到达模式，并结合 BurstGPT、ShareGPT、LongBench 的输入输出长度分布。
- 关键实验设置与指标：核心指标是客户端视角 TTFT、TPOT、throughput 和 SLO violation。BurstGPT 原 trace 用 TraceUpscaler 调整 RPS 以匹配 testbed serving capacity，同时保持时间模式，并保证全程平均内存需求低于总内存 60%。端到端结果显示 KUNSERVE 的 P99 TTFT 相比 vLLM、InferCept、Llumnix 快 12.7 到 72.2 倍，并带来 7.2% 到 12.8% 的平均 SLO violation 降低；多 GPU Qwen-2.5-72B 在 LongBench 上趋势类似，P99 latency 降低 8.4 到 11.9 倍。Ablation 显示 dynamic parameter drop 是最大贡献，coordinated exchange 继续降低 P99 / P999 TTFT，lookahead batch formulation 将 pipeline bubble time 从 21.9% 降到 8.3%，并提升约 20% throughput。cost model 相对真实执行时间偏差小于 5%，而忽略 attention cost 的模型在有 / 无 prefix attention 情况下可出现最高 48% / 74% 偏差。

4、pipeline/kernel解析：

- 新pipeline/kernel是什么：论文没有提出一个新的 GPU kernel，而是提出一条 memory-overload handling pipeline：overload detection -> drop plan generation -> parameter physical memory remapping -> request rescheduling with pipeline parallelism -> coordinated KVCache exchange -> lookahead microbatch execution -> dynamic parameter restoration。最关键的 runtime path 是“参数丢弃后跨 instance 的 pipeline serving path”，它把原来每个 instance 独立完整执行的请求，转换成多个参数互补 instance 组成的 group 共同执行。kernel 层面，KUNSERVE 的要点是通过 CUDA VMM 保持 PagedAttention 等已有 kernel 的 KVCache 虚拟地址布局不变，而不是重写 attention kernel。
- 新pipeline/kernel的执行流例子：假设两个单 GPU instance A 和 B 都部署同一个 8 层模型，并且 burst 导致两边 KVCache 都接近 HBM 上限。KUNSERVE 触发 drop plan 后，在 A 上丢弃后 4 层参数，在 B 上丢弃前 4 层参数；local memory manager 将这些被丢参数的物理页映射为额外 KVCache 空间，于是 queued requests 可以被接纳。此后一个新请求先在 A 上执行层 0 到 3，activation 通过 RDMA 发给 B，再在 B 上执行层 4 到 7，形成 pipeline parallelism。对于原本已经在 A 上 decode 的 ongoing request，B 执行后 4 层时还需要对应 layer 的 KVCache，因此 network coordinator 会把相关 KVCache 从 A 传给 B；传输被切成 chunk，每次检查是否有 activation transfer 等待，若有则暂停 KVCache chunk，让 activation 先过，减少 pipeline stall。调度器在 queued requests 足够多时不按简单 token count 组 batch，而是用包含 prefix-attn、自注意力、FFN 和其它开销的 cost model 递归拆分 microbatch，使各 microbatch 执行时间接近平衡，降低 GPU bubble。等 global monitor 发现 KVCache 使用低于阈值，例如论文实现中使用低于未 drop 状态 GPU 容量 50% 的简单阈值，就从其它 instance、host DRAM 或 SSD 恢复缺失参数，并回到常规低延迟执行路径。
