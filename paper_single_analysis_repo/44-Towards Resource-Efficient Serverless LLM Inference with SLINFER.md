论文标题：Towards Resource-Efficient Serverless LLM Inference with SLINFER

CPU-GPU系统的模型调度，从模型实例-GPU一对一绑定，到多模型实例共享Node（CPU/GPU），动态分时支持动态分区。

本地条目说明：
    - 本地编号：paper_2026 第 44 篇
    - 本地 PDF：paper_2026/44-Towards Resource-Efficient Serverless LLM Inference with SLINFER.pdf
    - 本地文本抽取：paper_2026/44-Towards Resource-Efficient Serverless LLM Inference with SLINFER.txt
    - 发表信息：HPCA 2026，DOI：10.1109/HPCA68181.2026.11408548

开源仓库确认：
    - 状态：已找到
    - 链接：https://github.com/BarrinXu/SLINFER
    - Artifact：论文附录声明 artifact 已公开，并给出 Zenodo DOI：10.5281/zenodo.17846442
    - 说明：论文 Artifact Appendix 明确写明 SLINFER 的 prototype、modified ServerlessLLM、modified vLLM 和实验 workflow 已包含在 artifact 中，并给出 GitHub 仓库 BarrinXu/SLINFER。附录还列出 Ubuntu 22.04、CUDA 12.4、4 张 A100-80GB GPU、4 台 32-core Intel 4th Gen Xeon CPU、AzureFunctionsDataset2019、AzureLLMInferenceDataset2023 以及 Llama-3.2-3B-Instruct、Llama-2-7b-chat-hf、Llama-2-13b-chat-hf 等复现实验依赖。因此这里将该 GitHub/Zenodo 视为官方 artifact / 源码入口。

1、论文工作：
    - 论文要解决的核心问题：
      SLINFER 面向私有化、serverless 风格的小到中等规模 LLM 部署。论文观察到这类部署有两个关键特征：一是 8B 以下模型在 HuggingFace 下载中占绝大多数，二是许多私有模型的请求频率低且波动大。现有 serverless LLM 系统通常在请求到达时为每个模型独占分配 GPU，这种方式能降低 cold-start，但在“模型很多、每个模型请求稀疏”的场景下会造成 GPU 数量成为瓶颈。论文以 ServerlessLLM 服务 64 个 3B-13B 模型、4 张 A100-80GB GPU 为例，指出 33% 请求因排队无法满足 SLO，而单 GPU 平均显存利用率只有 23%。
    - 瓶颈来源：
      主要瓶颈来自资源调度和显存管理，而不是单个 attention kernel 的计算效率。GPU 稀缺导致每个模型独占 GPU 时出现严重 over-provisioning；同时 CPU 在 GPU serving 中基本空闲，但 4th Gen Intel Xeon 这类带 AMX 的 CPU 已能在部分 7B/13B、短输入、中等 SLO 场景中独立服务 LLM。另一方面，LLM 请求在 prefill/decode 间、token 间、batch size 间资源需求波动很大，KV-cache 随并发和输出长度动态膨胀，简单静态切分 GPU/CPU 资源会导致 SLO 违约、OOM 风险或碎片化实例。
    - 论文的主要贡献：
      SLINFER 提出一种 heterogeneous CPU/GPU serverless LLM serving 系统，将异构硬件抽象成 CPU/GPU nodes，并通过三个机制实现弹性共享：headroom-driven compute subsystem 在 token 粒度调度实例；hazard-aware memory subsystem 以 watermark 和全局预算机制管理 KV-cache 扩缩容；efficiency-oriented consolidator 通过 proactive preemption 和 reactive bin-packing 减少同一模型的碎片化实例。系统目标不是提升单请求峰值速度，而是在满足 TTFT/TPOT SLO 的前提下提高可服务模型数量和 SLO-met requests。
    - 论文所处背景：
      该工作属于 LLM serving、serverless inference、异构资源调度和在线 SLO 管理方向。它不同于 vLLM、DistServe、LoongServe 这类主要面向高负载单模型或 prefill/decode 阶段优化的系统，也不同于 ServerlessLLM/Medusa/ParaServe 主要关注 cold-start 的做法；SLINFER 关注的是多模型、低频但突发、私有部署场景中的 CPU/GPU 透明共享。

2、相对 Baseline 解决的问题与设计方法：
    - Baseline 的具体问题：
      论文的核心 baseline 是 ServerlessLLM，记为 sllm。sllm 只支持 GPU，每个模型实例独占 GPU，并依赖快速模型加载来降低 cold-start。这个策略在模型数较少时可行，但当小模型数量增加时，GPU 被模型独占导致请求排队，即使 GPU 显存和计算资源大多空闲也无法被其他模型利用。论文还构造了两个增强 baseline：sllm+c 加入 CPU 支持但仍采用实例级资源分配；sllm+c+s 进一步支持 CPU/GPU 上的 time-sharing，但采用固定资源分区。它们的问题是：CPU/GPU 能力、batch size、token length 和请求负载动态变化，固定并发阈值或静态分区不能精确适配 token-level SLO，容易出现资源浪费或 SLO 违约。
    - 论文的设计方法：
      SLINFER 采用事件驱动的多实例部署方式。新请求到达后，系统优先尝试加入已有实例，尤其优先 CPU 节点；CPU 是否可用由离线 profiling 决定，不支持 AMX 或无法满足模型/输入/SLO 的 CPU 会被排除。加入某个实例前，compute subsystem 执行 shadow validation，模拟加入请求后的 prefill/decode 过程，确认新请求和已有请求的 headroom 不会变成负数；memory subsystem 同时检查节点显存/内存是否足以容纳 KV-cache 扩容。如果无法加入现有实例，consolidator 尝试通过 preemption 或 bin-packing 避免创建碎片化实例，最后才创建新实例。
    - 方法如何对冲 Baseline 缺陷：
      对独占 GPU 带来的 over-provisioning，SLINFER 允许多个 LLM 在同一 CPU/GPU 节点上按需共享，并将 AMX CPU 当作可独立 serving 的资源，而不仅是 GPU 的辅助设备。对静态分区无法适应动态需求的问题，SLINFER 用 headroom 公式把 TTFT/TPOT SLO 转成每个请求还能等待多久的时间预算，并在每个 token iteration 周期选择最紧急的实例执行。对 KV-cache 动态扩缩容的 OOM 风险，SLINFER 不让实例各自无协调地扩缩容，而是维护 optimistic budget 和 pessimistic global tracking。对同模型多实例碎片化，SLINFER 优先让大 batch 实例继续长大，并尽早回收小 batch 碎片实例。
    - 关键 trade-off：
      SLINFER 接受更复杂的在线调度、性能 profiling、memory budget bookkeeping 和 shadow validation 开销，以换取更高的 deployment density。它的收益依赖小到中等模型、请求低频但突发、SLO 相对适中、CPU 具备矩阵加速单元等条件。对于大模型、长输入、非常紧的 TPOT SLO，CPU 不能满足要求时系统会退回 GPU；当所有节点都饱和或大模型占主导时，共享收益会下降，行为逐渐接近独占 GPU 分配。

3、论文实现：
    - Baseline 如何实现：
      sllm 使用 ServerlessLLM 作为 baseline，内部推理引擎为 vLLM，模型冷启动使用 ServerlessLLM 的 loader。sllm 只支持 GPU，模型实例按 GPU 独占方式部署。sllm+c 在此基础上加入 CPU 节点支持，并和 SLINFER 一样优先使用 CPU。sllm+c+s 进一步让 CPU/GPU 节点支持固定 time-sharing：除 13B CPU 模型外，每个实例只分配半个节点资源。为了公平，论文根据 profiling 为 baseline 调高并发限制，避免使用 ServerlessLLM 默认 scale-out 阈值 2 导致过度低效。
    - 新设计如何实现：
      SLINFER 原型包含 SLINFER_core、modified ServerlessLLM 和 modified vLLM。GPU 侧使用 vLLM 0.5.2，CPU 侧用 OpenVINO 2024.6.0 替换 vLLM GPU backend 以支持 AMX CPU 推理。系统通过 scheduler 维护 CPU/GPU pool、实例状态、请求 headroom、KV-cache 预算和扩缩容操作队列。模型被缓存到 CPU memory，cold-start 流程与 ServerlessLLM 类似；论文环境中 7B 模型加载约 1 秒，作者为发生 cold-start 的请求放宽一个等于加载时长的 TTFT grace window。
    - Headroom-driven compute 实现：
      SLINFER 离线采样每个模型在每类硬件上的 prefill 和 decode 延迟。Prefill 时间近似随输入长度线性变化，因此用一维线性插值估计；decode 时间同时受 batch size 和平均 token length 影响，因此用二维线性插值估计。采样点按 2X 间隔生成，若最大长度 Lmax、最大 batch size Bmax，则只需 O(log Lmax * log Bmax) 级别样本。论文报告随机 workload 上 TTFT/TPOT 估计平均相对误差分别为 5.9% 和 3.9%。在线 shadow validation 会把每个 iteration 额外高估 10%，以吸收运行时波动和 decode token 变长。
    - Hazard-aware memory 实现：
      一个实例的显存需求由模型权重和 KV-cache 构成。模型权重固定，KV-cache 依赖并发请求、输入长度和已生成输出长度。SLINFER 用历史平均输出长度和 lower bound 估计每个请求最终至少需要的 KV 空间，并使用 watermark 机制做 early scale-up 和 lazy scale-down。若当前 KV-cache 不足，则扩到 Mrequire * (1 + w%)；请求完成后，只有推荐容量明显低于当前容量才缩容。论文推荐 w=25%，因为关闭 watermark 会使实例 11.3% 生命周期花在频繁 scaling 上，而 25% 时 scaling overhead 已降到约 1.4%，且因低估导致的 request migration rate 仅 0-0.3%。
    - Memory orchestration 实现：
      KV-cache scaling 基于 paged attention 时需要重新分配 cache blocks 并复制旧 KV-cache，论文测得 GPU 上从 32GB 缩到 16GB 约 0.3s，扩到 64GB 约 1.9s。因此多个实例并发扩缩容可能产生 OOM hazard。SLINFER 用 optimistic budget 接收 scale-up/scale-down 请求，用 pessimistic tracking 决定何时真正执行：scale-down 可直接发出，但在 pessimistic tracking 中仍按旧容量计入，直到完成后通知 reservation station；scale-up 若存在 OOM 风险则先进入 reservation station，等待已有 scale-down 完成后再重新检查执行。
    - Consolidation 实现：
      Proactive consolidation 在某个实例需要 scale-up 但被邻居占用资源时，允许它 preempt 较小 batch size 的邻居，并将被抢占实例的请求重新调度到其他节点。为避免把已有大实例打碎，SLINFER 只允许抢占 batch size 小于自己的实例，并且用 shadow validation 确保被迁移请求仍满足 SLO。Reactive consolidation 在同一模型已经有多个实例时，把新请求优先路由到 batch size 最大的实例，使小 batch 碎片实例更快完成剩余请求并被回收。
    - 实验 / 实现平台：
      主实验使用 4 个 32-core Intel Xeon 6462C @3.3GHz CPU nodes 和 4 个 NVIDIA A100-80GB GPU nodes，逻辑上从两台各 2 GPU 的物理机器中划分。模型为 16-bit Llama-3.2-3B、Llama-2-7B、Llama-2-13B；混合部署实验还包含 CodeLlama-34B，并用 tensor parallelism 运行在 2 GPUs/instance。输入/输出长度来自 Azure LLM Conversation dataset，多模型到达模式来自 Azure Serverless Trace，作者抽取 30 分钟片段并采样 32、64、128 个函数映射为模型。TTFT SLO 设置为 min(max(0.5, input length/512), 8)s，TPOT SLO 设置为 0.25s。
    - 关键实验设置与指标：
      主要指标包括 SLO-met requests、TTFT CDF、average nodes used、decode speed、GPU/CPU 使用量、memory utilization、batch size、scheduling overhead 和 ablation。端到端实验显示，当服务 128 个模型时，SLINFER 相比 sllm 的 SLO-met requests 提升 86%-154%，相比 sllm+c 提升 47%-62%，相比 sllm+c+s 提升 18%-70%。在 32 个 3B 模型场景中，SLINFER 只用 3.0 个 CPU、0 个 GPU，而 sllm 需要约 3.2 个独占 GPU；在 32 个 7B 模型场景中，sllm+c 使用 1.5 个 GPU，sllm+c+s 使用 2.0 个 GPU，SLINFER 只用 0.9 个 GPU。消融实验显示，去掉 sharing 时 SLO compliance 降到 89%，去掉 CPU 或 consolidation 都会增加 GPU 使用。
    - 局限与假设：
      SLINFER 当前主要针对 small- to mid-sized LLM。对于长输入和紧 SLO，CPU serving 不一定可行；论文讨论中指出 32k 输入在当前 CPU 上 prefill 可能需要 84s，远超 8s TTFT SLO。对大模型，SLINFER 会回退到 ServerlessLLM 式的独占 GPU 分配。实验中的多模型 workload 是用 Azure Serverless Trace 映射 LLM 模型构造的，能反映 serverless 热点/突发特征，但仍不是某个真实生产 LLM 平台的完整 trace。系统还依赖离线 profiling 与 workload 统计，硬件、模型实现、量化格式或 SLO 变化时需要重新校准。

4、pipeline/kernel 解析：
    - 新 pipeline/kernel 是什么：
      论文没有提出新的 CUDA attention kernel 或矩阵乘 kernel。最接近“新 pipeline”的是 SLINFER 的 request lifecycle pipeline：request arrival -> instance selection with shadow validation -> token-level headroom scheduling -> hazard-aware KV-cache scaling -> proactive/reactive consolidation -> idle reclaim。它是一个 serving runtime pipeline，而不是单个算子 pipeline。核心执行粒度从“每个模型独占 GPU 实例”变成“多个 CPU/GPU 节点上多个模型实例的 token iteration、KV-cache resize operation 和 instance consolidation operation”。
    - 一个请求的执行流例子：
      假设模型 A 已在 GPU Node-1 上有一个 batch size 为 4 的实例，同时 CPU Node-1 上也有可服务 A 的 AMX CPU 实例。新请求 r 到达后，SLINFER 先查找 A 的已有实例，并优先尝试 CPU 实例。compute subsystem 根据 r 的输入长度估计 prefill 时间，根据加入后的 batch size 和平均 token length 估计后续 decode 时间。它把 r 虚拟加入候选实例，模拟未来若干 prefill/decode iteration，并计算每个请求的 headroom = ST + TTFTSLO + TPOTSLO * O - CT。若 r 的 prefill 太慢、已有请求被 r 的 prefill 阻塞太久、或加入后某轮 decode 总耗时超过 TPOT SLO，shadow validation 失败，系统尝试下一个实例或 GPU 节点。
    - Token-level scheduling 路径：
      当 r 被接纳到某个实例后，节点调度器不是让该实例一直独占节点，而是在每个 scheduling cycle 从节点上的多个实例中挑选“最短 headroom 的请求所在实例”执行一个 prefill 或 decode iteration。一次 iteration 完成后，该请求已生成 token 数 O 增加，headroom 会重新计算。例如 TPOT SLO 为 0.25s，某 decode iteration 用 0.2s，则该请求的下一 token headroom 相当于增加 0.25s 又消耗 0.2s。调度器持续重复这一过程，使节点被多实例共享，同时优先保护最接近 SLO 边界的请求。
    - KV-cache scaling 路径：
      如果 r 加入后候选实例的 KV-cache 当前容量 Mcur 小于估计需求 Mrequire，memory subsystem 会请求 scale-up 到 Mrequire * (1 + watermark)。该 scale-up 先进入节点的 optimistic budget 检查；若预算允许但 pessimistic tracking 认为实际执行会与其他未完成操作叠加导致 OOM，则该操作进入 reservation station。等某个 scale-down 或 model unload 真实完成后，reservation station 重新检查并发出 scale-up。执行 scale-up 时，底层需要新建 cache blocks、复制已用 KV pages、删除旧 blocks，因此被作为有明显时延和 OOM 风险的异步操作，而不是普通元数据更新。
    - Consolidation 路径：
      如果 r 无法加入已有实例，是因为目标实例想 scale-up 但被邻居模型占用资源，SLINFER 会尝试 proactive consolidation。例如 A 的 batch size 为 4，邻居 B 的 batch size 为 2，A 加入 r 后可形成更大的 batch 并提升资源效率，则 A 可抢占 B 的资源；B 的请求被重新放回调度流程，只有 shadow validation 确认它们迁移后仍满足 SLO 时 preemption 才执行。如果 A 已被迫在另一个节点创建了碎片实例，后续 A 的新请求会优先路由到 batch size 更大的实例，让小实例自然 drain 后回收，这就是 reactive bin-packing consolidation。
    - 与 baseline pipeline 的区别：
      ServerlessLLM 的路径更接近“请求到达 -> 若模型实例不存在则找空闲 GPU 启动实例 -> GPU 独占服务 -> keep-alive 后回收”。它优化了 model loading，但资源分配粒度仍是整 GPU / 整实例。SLINFER 的路径则在请求进入前就进行 SLO-aware admission simulation，在执行中按 token iteration 共享节点，在内存层对 KV-cache resize 做全局协调，在实例层主动减少碎片。这样它不是把 cold-start 做得更快，而是尽量减少不必要的独占实例和碎片实例。
    - pipeline 的局限：
      该 pipeline 的正确性依赖 profiling 估计足够准确、headroom 模型能覆盖实际 SLO、KV-cache 需求估计不过度偏小，并且调度器开销相对于 token iteration 足够低。论文报告 shadow validation 和 token-level scheduling overhead 都很低，但当模型数、实例数、硬件异构性和 SLO 类型继续扩展时，scheduler 状态和 profiling 维护成本会增加。它也没有解决单模型内部 kernel 加速、跨节点 KV 迁移优化、模型量化误差或更复杂 PD disaggregation 的全局路由问题。
