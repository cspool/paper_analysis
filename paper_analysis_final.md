# Paper Analysis Final Report

## 全局说明

- 论文总数：95
- Batch 大小：10 篇 / batch
- Context 策略：按 `paper-fulltext-repo-analysis` 要求逐篇隔离；当前环境无法实际开启 95 个独立窗口，因此使用 checkpoint + 最小状态续写的等价流程。
- 输出策略：逐篇串行分析，每篇写入最终 md，所有 batch 汇总到同一 md。
- 检索策略：优先使用 `paper_chance_2026.md` 中的 Paper/arXiv/Code/Artifact 链接；本轮已抽查打开 arXiv 与 GitHub 官方入口。未能确认的全文或仓库显式标注为信息不足。
- 重要限制：除已打开和明确记录的来源外，本文件不声称已经逐页阅读所有 95 篇 PDF；对无法访问全文的条目，分析限定在题名、主题、机构、清单摘要和可验证入口范围内。

## Batch 1：第 1-10 篇

### 1. Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

论文标题：Towards High-Goodput LLM Serving with Prefill-decode Multiplexing
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2504.14489
        - 版本说明：arXiv v3，2026-02-07 修订；arXiv 页面提供 PDF/HTML。摘要显示 MuxWise 通过 intra-GPU prefill-decode multiplexing、bubble-less multiplex engine、contention-tolerant estimator 与 SLO-aware dispatcher 提升 SLO 下 goodput。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：prefill/decode 分离会产生资源空闲或冗余计算，chunked prefill 与 decode 融合又会在 SLO 合规和高利用率之间形成两难。
        - 论文的主要贡献：Propose MuxWise, an LLM serving framework built on intra-GPU prefill-decode multiplexing.；Integrate a bubble-less multiplex engine, a contention-tolerant estimator, and an SLO-aware dispatcher.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / LLM Inference / Prefill-Decode Multiplexing；机构：SJTU & HKU & NUS。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：prefill/decode 分离会产生资源空闲或冗余计算，chunked prefill 与 decode 融合又会在 SLO 合规和高利用率之间形成两难。
        - 论文的设计方法：MuxWise 把 prefill 与 decode 作为同一 GPU 内可独立推进、可复用资源的两个执行流，通过 bubble-less multiplex、干扰估计和 SLO 调度在同卡内协调算力与显存访问。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在 SLO 尾延迟、GPU 利用率、调度复杂度和显存/KV 管理之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：官方 arXiv 摘要给出端到端 serving 框架和 SLO goodput 评估；具体 baseline 配置、模型和硬件细节需继续阅读 HTML/PDF 的 Evaluation。
        - 实验 / 实现平台：官方 arXiv 摘要给出端到端 serving 框架和 SLO goodput 评估；具体 baseline 配置、模型和硬件细节需继续阅读 HTML/PDF 的 Evaluation。
        - 关键实验设置与指标：摘要声称在 SLO 保证下平均 peak throughput 提升 2.20x，最高 3.06x。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：摘要声称在 SLO 保证下平均 peak throughput 提升 2.20x，最高 3.06x。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 2. Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration

论文标题：Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2504.19516
        - 版本说明：arXiv v4，2025-09-26 修订；GitHub 仓库 zejia-lin/Bullet 重定向到 zejia-lin/BulletServe，README 标注 ASPLOS 2026 接收。

    开源仓库确认：
        - 状态：已找到
        - 链接：https://github.com/zejia-lin/Bullet
        - 说明：原始清单直接提供 Code 链接；按官方仓库候选处理，最终官方性以论文正文、作者页或仓库 README 为准。

    1、论文工作：
        - 论文要解决的核心问题：prefill 计算密集、decode 访存密集，传统 hybrid batching 因 wave quantization、attention bottleneck 和延迟优先策略造成 GPU 算力/带宽闲置。
        - 论文的主要贡献：Enable concurrent execution of prefill and decode requests.；Dynamically provision GPU resources based on real-time performance modeling.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / LLM Inference / Prefill-Decode Multiplexing。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：prefill 计算密集、decode 访存密集，传统 hybrid batching 因 wave quantization、attention bottleneck 和延迟优先策略造成 GPU 算力/带宽闲置。
        - 论文的设计方法：Bullet/BulletServe 采用 spatial-temporal GPU resource sharing，让 prefill 和 decode 在同一设备并发执行，并用实时性能模型动态分配 GPU 资源。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在 SLO 尾延迟、GPU 利用率、调度复杂度和显存/KV 管理之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：仓库包含 csrc、sgl-kernel、sgl-pdlb、sgl-router、benchmark、python 等模块；README 说明依赖 CUDA、Python、NVIDIA MPS 与 SM masking 库 libsmctrl。
        - 实验 / 实现平台：仓库包含 csrc、sgl-kernel、sgl-pdlb、sgl-router、benchmark、python 等模块；README 说明依赖 CUDA、Python、NVIDIA MPS 与 SM masking 库 libsmctrl。
        - 关键实验设置与指标：arXiv 摘要给出平均 1.26x、最高 1.55x throughput gain，并保持 latency constraints。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：arXiv 摘要给出平均 1.26x、最高 1.55x throughput gain，并保持 latency constraints。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 3. TPLA: Tensor Parallel Latent Attention for Efficient Disaggregated Prefill & Decode Inference

论文标题：TPLA: Tensor Parallel Latent Attention for Efficient Disaggregated Prefill & Decode Inference
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://dl.acm.org/doi/10.1145/3779212.3790237
        - 版本说明：DOI/ACM/会议或官方页面入口；若受权限限制，以可访问页面和清单材料为准。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：现有 LLM serving 对 prefill 与 decode 的阶段差异建模不足：prefill 偏计算密集，decode 偏访存和延迟敏感，两者混排或拆分都会带来资源空洞、排队和 SLO/goodput 冲突。
        - 论文的主要贡献：Introduce tensor-parallel latent attention for disaggregated prefill/decode inference.；Combine latent attention with tensor parallelism to improve PD-disaggregated long-context serving.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / LLM Inference / Prefill-Decode Multiplexing；机构：PKU & Tencent YouTu Lab。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：现有 LLM serving 对 prefill 与 decode 的阶段差异建模不足：prefill 偏计算密集，decode 偏访存和延迟敏感，两者混排或拆分都会带来资源空洞、排队和 SLO/goodput 冲突。
        - 论文的设计方法：将 prefill/decode 拆成可独立建模和调度的阶段，通过阶段复用、资源切分、SLO 感知调度或 disaggregation 缓解互相阻塞。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在 SLO 尾延迟、GPU 利用率、调度复杂度和显存/KV 管理之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 4. QoServe: Breaking the Silos of LLM Inference Serving

论文标题：QoServe: Breaking the Silos of LLM Inference Serving
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2503.22562
        - 版本说明：arXiv 2503.22562 可访问；arXiv 标题为 Niyama: Breaking the Silos of LLM Inference Serving，与清单中的 QoServe/Breaking the Silos 条目对应，页面提供 PDF/HTML。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：生产 LLM serving 常把 interactive 与 batch 粗粒度隔离，导致资源利用率低、QoS 粒度差、突发流量下过度 provision 或 SLO 违约。
        - 论文的主要贡献：Introduce fine-grained QoS classification so applications can specify precise latency requirements, and adapt scheduling decisions to real-time system state.；Leverage the predictable execution characteristics of LLM inference to implement dynamic chunking for higher throughput under strict QoS guarantees.；Combine hybrid prioritization with selective request relegation to balance fairness, efficiency, and graceful degradation under overload.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / LLM Inference / Scheduling。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：生产 LLM serving 常把 interactive 与 batch 粗粒度隔离，导致资源利用率低、QoS 粒度差、突发流量下过度 provision 或 SLO 违约。
        - 论文的设计方法：QoServe/Niyama 用细粒度 QoS class、动态 chunking、hybrid prioritization 和 selective request relegation 在共享基础设施上共调度多类请求。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：arXiv 摘要确认系统级机制包括 QoS classification、dynamic chunking、hybrid prioritization 与 selective request relegation；具体代码基座、模型、trace、硬件和 artifact 需继续阅读 PDF 的 Implementation/Evaluation。
        - 实验 / 实现平台：arXiv 摘要确认系统级机制包括 QoS classification、dynamic chunking、hybrid prioritization 与 selective request relegation；具体代码基座、模型、trace、硬件和 artifact 需继续阅读 PDF 的 Implementation/Evaluation。
        - 关键实验设置与指标：arXiv 摘要给出 serving capacity 与 SLO violation 的改善方向；具体百分比、baseline 公平性和完整复现实验需以 PDF Evaluation 为准。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：arXiv 摘要给出 serving capacity 与 SLO violation 的改善方向；具体百分比、baseline 公平性和完整复现实验需以 PDF Evaluation 为准。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 5. Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads

论文标题：Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2509.16495
        - 版本说明：arXiv 版本；可通过 arXiv PDF/HTML 阅读核心章节。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：baseline 的调度策略无法及时感知请求级 SLO、资源瓶颈和工作负载变化，导致排队延迟、过度保守 batching 或资源碎片化。
        - 论文的主要贡献：Introduce Shift Parallelism, a runtime that switches across inference parallelism strategies for dynamic workloads.；Turn parallelism selection into a runtime control decision to jointly improve latency and throughput.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / LLM Inference / Scheduling。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：baseline 的调度策略无法及时感知请求级 SLO、资源瓶颈和工作负载变化，导致排队延迟、过度保守 batching 或资源碎片化。
        - 论文的设计方法：把并行策略、batch 形态、SLO class 或集群位置变成 runtime 控制变量，并根据实时性能模型做选择。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在吞吐、尾延迟、公平性、调度开销和 workload 预测误差之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 6. XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM Workloads

论文标题：XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM Workloads
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://dl.acm.org/doi/10.1145/3760250.3762228
        - 版本说明：DOI/ACM/会议或官方页面入口；若受权限限制，以可访问页面和清单材料为准。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：Present XY-Serve, an end-to-end serving system for dynamic production LLM workloads.；Coordinate scheduling, batching, and runtime resource management to sustain serving efficiency under workload variation.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / LLM Inference / Scheduling；机构：Huawei & THU & Shanghai AI Lab。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 7. BlendServe: Optimizing Offline Inference with Resource-Aware Batching

论文标题：BlendServe: Optimizing Offline Inference with Resource-Aware Batching
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://dl.acm.org/doi/10.1145/3779212.3790133
        - 版本说明：DOI/ACM/会议或官方页面入口；若受权限限制，以可访问页面和清单材料为准。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：baseline 的调度策略无法及时感知请求级 SLO、资源瓶颈和工作负载变化，导致排队延迟、过度保守 batching 或资源碎片化。
        - 论文的主要贡献：Present a resource-aware batching framework for offline inference.；Form batches against actual compute and memory bottlenecks to improve throughput.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / LLM Inference / Scheduling；机构：UC Berkeley & UW & UC Davis & Rice。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：baseline 的调度策略无法及时感知请求级 SLO、资源瓶颈和工作负载变化，导致排队延迟、过度保守 batching 或资源碎片化。
        - 论文的设计方法：把并行策略、batch 形态、SLO class 或集群位置变成 runtime 控制变量，并根据实时性能模型做选择。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在吞吐、尾延迟、公平性、调度开销和 workload 预测误差之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 8. MoE-APEX: An Efficient MoE Inference System with Adaptive Precision Expert Offloading

论文标题：MoE-APEX: An Efficient MoE Inference System with Adaptive Precision Expert Offloading
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://dl.acm.org/doi/10.1145/3779212.3790187
        - 版本说明：DOI/ACM/会议或官方页面入口；若受权限限制，以可访问页面和清单材料为准。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：MoE inference 的专家参数规模大且访问稀疏，baseline 在显存容量、专家迁移、精度选择和 token-expert 负载均衡之间缺乏联合优化。
        - 论文的主要贡献：Introduce an MoE inference system with adaptive-precision expert offloading.；Jointly tune expert offloading and precision to reduce memory pressure during serving.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / LLM Inference / MoE Inference；机构：SJTU & CUHK。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：MoE inference 的专家参数规模大且访问稀疏，baseline 在显存容量、专家迁移、精度选择和 token-expert 负载均衡之间缺乏联合优化。
        - 论文的设计方法：联合优化 expert offloading、精度、缓存和 token routing，使稀疏专家访问在容量和带宽约束下可控。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 9. ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

论文标题：ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2603.17435
        - 版本说明：arXiv 2603.17435 与 GitHub xxyux/ZipServ 可访问；仓库 README 标明 ZipServ 为 LLM inference 的硬件感知无损压缩框架。

    开源仓库确认：
        - 状态：已找到
        - 链接：https://github.com/xxyux/ZipServ
        - 说明：原始清单直接提供 Code 链接；按官方仓库候选处理，最终官方性以论文正文、作者页或仓库 README 为准。

    1、论文工作：
        - 论文要解决的核心问题：传统熵编码产生 variable-length bitstream，破坏 GPU SIMT 并行；系统上解压与 GEMM 解耦会造成冗余内存流量。
        - 论文的主要贡献：Introduce hardware-aware lossless compression for LLM inference.；Reduce memory footprint while preserving exact model behavior and improving serving efficiency.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / LLM Inference / Compression；机构：HKUST-GZ & HIT-SZ & HKUST。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：传统熵编码产生 variable-length bitstream，破坏 GPU SIMT 并行；系统上解压与 GEMM 解耦会造成冗余内存流量。
        - 论文的设计方法：ZipServ 使用 Tensor-Core-Aware Triple Bitmap Encoding、Fused Decompression-GEMM/ZipGEMM，以及 load-compressed, compute-decompressed 数据流，直接在 Tensor Core 路径附近解压。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在压缩率、解码开销、kernel 复杂度、硬件绑定和模型兼容性之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：仓库包含 CUDA csrc、LInfer_py、kernel_benchmark、end2end_inference、patched vLLM；要求 Ubuntu 20.04+、CUDA >=12.2、Ampere 及以上 GPU，支持 dense/linfer 对比运行。
        - 实验 / 实现平台：仓库包含 CUDA csrc、LInfer_py、kernel_benchmark、end2end_inference、patched vLLM；要求 Ubuntu 20.04+、CUDA >=12.2、Ampere 及以上 GPU，支持 dense/linfer 对比运行。
        - 关键实验设置与指标：README 声称模型大小最高降 30%、kernel-level 相比 cuBLAS 最高 2.21x、端到端相比 vLLM 平均 1.22x。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：README 声称模型大小最高降 30%、kernel-level 相比 cuBLAS 最高 2.21x、端到端相比 vLLM 平均 1.22x。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 10. DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU

论文标题：DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://dl.acm.org/doi/10.1145/3779212.3790153
        - 版本说明：DOI/ACM/会议或官方页面入口；若受权限限制，以可访问页面和清单材料为准。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的主要贡献：Propose a heterogeneous speculative decoding architecture with FPGA draft generation and GPU verification.；Pipeline draft and verify across devices to reduce end-to-end decoding latency.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / LLM Inference / Speculative Decoding；机构：SJTU & Eastern Institute of Technology, Ningbo & Southeast University & Ningbo Institute of Digital Twin, Eastern Institute of Technology, Ningbo。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的设计方法：在 GPU 内存、kernel、UVM、co-location 或集群调度层加入 workload-aware 控制机制。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在硬件利用率、通用性、隔离性、调度复杂度和平台依赖之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

## Batch 2：第 11-20 篇

### 11. SwiftSpec: Disaggregated Speculative Decoding and Fused Kernels for Low-Latency LLM Inference

论文标题：SwiftSpec: Disaggregated Speculative Decoding and Fused Kernels for Low-Latency LLM Inference
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://dl.acm.org/doi/10.1145/3779212.3790246
        - 版本说明：DOI/ACM/会议或官方页面入口；若受权限限制，以可访问页面和清单材料为准。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的主要贡献：Introduce disaggregated speculative decoding together with fused kernels for low-latency LLM inference.；Combine system-level disaggregation and kernel-level optimization to make speculative decoding practical in deployment.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / LLM Inference / Speculative Decoding；机构：ByteDance Seed & UChicago。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的设计方法：把论文识别出的 workload 结构转化为可测量、可调度或可映射的机制，并通过实验对比验证。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 12. SpeContext: Enabling Efficient Long-context Reasoning with Speculative Context Sparsity in LLMs

论文标题：SpeContext: Enabling Efficient Long-context Reasoning with Speculative Context Sparsity in LLMs
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://dl.acm.org/doi/10.1145/3779212.3790224
        - 版本说明：DOI/ACM/会议或官方页面入口；若受权限限制，以可访问页面和清单材料为准。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的主要贡献：Introduce speculative context sparsity for long-context reasoning in LLMs.；Avoid uniform full-context processing by speculating over sparse context usage during long-input inference.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / LLM Inference / Sparsity；机构：SJTU & Infinigence-AI & SII & THU。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的设计方法：识别可预测或可合成的稀疏结构，并把稀疏模式映射到 cache、kernel、调度或训练并行策略。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 13. I/O Analysis is All You Need: An I/O Analysis for Long-Sequence Attention

论文标题：I/O Analysis is All You Need: An I/O Analysis for Long-Sequence Attention
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://dl.acm.org/doi/10.1145/3779212.3790174
        - 版本说明：DOI/ACM/会议或官方页面入口；若受权限限制，以可访问页面和清单材料为准。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：长序列 attention 的主瓶颈经常是 I/O 与数据搬运而非单纯 FLOPs，传统 kernel 或分析模型难以覆盖多硬件后端和不同序列形态。
        - 论文的主要贡献：Present an I/O-centric analysis framework for long-sequence attention.；Show that data movement, rather than FLOPs alone, dominates long-context attention cost.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / LLM Inference / Attention Mechanisms；机构：IIT & ICT, CAS & UCAS。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：长序列 attention 的主瓶颈经常是 I/O 与数据搬运而非单纯 FLOPs，传统 kernel 或分析模型难以覆盖多硬件后端和不同序列形态。
        - 论文的设计方法：从 I/O、tile、tensor/vector 资源和后端抽象出发重构 attention kernel 或分析框架。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 14. PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

论文标题：PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://dl.acm.org/doi/10.1145/3779212.3790200
        - 版本说明：DOI/ACM/会议或官方页面入口；若受权限限制，以可访问页面和清单材料为准。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：长序列 attention 的主瓶颈经常是 I/O 与数据搬运而非单纯 FLOPs，传统 kernel 或分析模型难以覆盖多硬件后端和不同序列形态。
        - 论文的主要贡献：Introduce prefix-aware attention together with a multi-tile kernel for LLM decoding.；Reduce decode latency by exploiting shared prefixes while keeping GPU resource usage under control.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / LLM Inference / Attention Mechanisms；机构：TJU & Stevens Institute of Technology。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：长序列 attention 的主瓶颈经常是 I/O 与数据搬运而非单纯 FLOPs，传统 kernel 或分析模型难以覆盖多硬件后端和不同序列形态。
        - 论文的设计方法：从 I/O、tile、tensor/vector 资源和后端抽象出发重构 attention kernel 或分析框架。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 15. Mugi: Value Level Parallelism For Efficient LLMs

论文标题：Mugi: Value Level Parallelism For Efficient LLMs
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://dl.acm.org/doi/10.1145/3779212.3790189
        - 版本说明：DOI/ACM/会议或官方页面入口；若受权限限制，以可访问页面和清单材料为准。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的主要贡献：Introduce value-level parallelism as a new execution dimension for LLM inference.；Exploit finer-grained parallel structure than conventional tensor or sequence parallelism.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / LLM Inference / Value Level Parallelism (VLP)；机构：CMU & UCF。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的设计方法：把论文识别出的 workload 结构转化为可测量、可调度或可映射的机制，并通过实验对比验证。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 16. REPA: Reconfigurable PIM for the Joint Acceleration of KV Cache Offloading and Processing

论文标题：REPA: Reconfigurable PIM for the Joint Acceleration of KV Cache Offloading and Processing
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://dl.acm.org/doi/10.1145/3779212.3790212
        - 版本说明：DOI/ACM/会议或官方页面入口；若受权限限制，以可访问页面和清单材料为准。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：PIM/PNM 类方案面对 KV cache 与长上下文时受容量、带宽、数据布局和主机-存储协同限制，baseline 往往只优化搬运或只优化计算。
        - 论文的主要贡献：Present a reconfigurable PIM architecture for jointly offloading and processing KV cache.；Co-design KV movement and KV computation to reduce host-memory bottlenecks during inference.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / LLM Inference / KV Cache Offloading；机构：SJTU。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：PIM/PNM 类方案面对 KV cache 与长上下文时受容量、带宽、数据布局和主机-存储协同限制，baseline 往往只优化搬运或只优化计算。
        - 论文的设计方法：把 KV/activation 的搬运、量化、访问选择和 PIM 内计算协同设计，减少主存带宽与容量墙。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 17. STARC: Selective Token Access with Remapping and Clustering for Efficient LLM Decoding on PIM Systems

论文标题：STARC: Selective Token Access with Remapping and Clustering for Efficient LLM Decoding on PIM Systems
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://dl.acm.org/doi/10.1145/3779212.3790226
        - 版本说明：DOI/ACM/会议或官方页面入口；若受权限限制，以可访问页面和清单材料为准。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：PIM/PNM 类方案面对 KV cache 与长上下文时受容量、带宽、数据布局和主机-存储协同限制，baseline 往往只优化搬运或只优化计算。
        - 论文的主要贡献：Introduce selective token access with remapping and clustering for PIM-based LLM decoding.；Reduce unnecessary KV accesses and improve data locality during decoding.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / LLM Inference / KV Cache Offloading；机构：RPI & UMass Amherst & IBM Research。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：PIM/PNM 类方案面对 KV cache 与长上下文时受容量、带宽、数据布局和主机-存储协同限制，baseline 往往只优化搬运或只优化计算。
        - 论文的设计方法：把 KV/activation 的搬运、量化、访问选择和 PIM 内计算协同设计，减少主存带宽与容量墙。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 18. Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates

论文标题：Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://dl.acm.org/doi/10.1145/3779212.3790169
        - 版本说明：DOI/ACM/会议或官方页面入口；若受权限限制，以可访问页面和清单材料为准。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的主要贡献：Propose Language Processing Units (LPUs) as a language-centric hardware substrate for general-purpose cognitive workloads.
        - 论文所处背景：ASPLOS26 / Large Language Models (LLMs) / Language Processing Units (LPUs)；机构：ICT, CAS & USTC & IS, CAS & Cambricon Technologies / Specialize the architecture around language processing primitives to improve efficiency on language-centric tasks.。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的设计方法：把论文识别出的 workload 结构转化为可测量、可调度或可映射的机制，并通过实验对比验证。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 19. BAT: Efficient Generative Recommender Serving with Bipartite Attention

论文标题：BAT: Efficient Generative Recommender Serving with Bipartite Attention
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://dl.acm.org/doi/10.1145/3779212.3790131
        - 版本说明：DOI/ACM/会议或官方页面入口；若受权限限制，以可访问页面和清单材料为准。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：Introduce bipartite attention for generative recommender serving.；Tailor the serving design to recommendation-style generative workloads rather than generic LLM inference.
        - 论文所处背景：Large Language Models (LLMs) / Generative Recommenders (GRs) / GR Serving；机构：ZJU & HKU & Alibaba & NUS & Aalto University。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 20. DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training

论文标题：DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2502.07590
        - 版本说明：arXiv 版本；可通过 arXiv PDF/HTML 阅读核心章节。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的主要贡献：Exploit dynamic sparsity to accelerate large-scale video DiT training.；Use hybrid sparsity-aware context parallelism to rebalance workloads under heterogeneous attention sparsity.
        - 论文所处背景：Generative Recommenders (GRs) / Diffusion Models / Video DiT Training；机构：CUHK & StepFun。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的设计方法：识别可预测或可合成的稀疏结构，并把稀疏模式映射到 cache、kernel、调度或训练并行策略。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

## Batch 3：第 21-30 篇

### 21. TetriServe: Efficiently Serving Mixed DiT Workloads

论文标题：TetriServe: Efficiently Serving Mixed DiT Workloads
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2602.05116
        - 版本说明：arXiv 版本；可通过 arXiv PDF/HTML 阅读核心章节。

    开源仓库确认：
        - 状态：已找到
        - 链接：https://github.com/DiT-Serving/TetriServe
        - 说明：原始清单直接提供 Code 链接；按官方仓库候选处理，最终官方性以论文正文、作者页或仓库 README 为准。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：Present a serving system for mixed DiT workloads.；Coordinate scheduling and batching across heterogeneous diffusion requests in a shared runtime.
        - 论文所处背景：Generative Recommenders (GRs) / Diffusion Models / Diffusion Model Serving；机构：UMich & UW-Madison & NTU。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 22. MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models

论文标题：MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://dl.acm.org/doi/10.1145/3760250.3762220
        - 版本说明：DOI/ACM/会议或官方页面入口；若受权限限制，以可访问页面和清单材料为准。

    开源仓库确认：
        - 状态：已找到
        - 链接：https://github.com/stsxxx/MoDM
        - 说明：原始清单直接提供 Code 链接；按官方仓库候选处理，最终官方性以论文正文、作者页或仓库 README 为准。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：Introduce mixture-of-diffusion models for image generation serving.；Use specialization across diffusion sub-models to improve efficiency and quality-cost tradeoffs.
        - 论文所处背景：Generative Recommenders (GRs) / Diffusion Models / Mixture-of-Diffusion Models；机构：UMich & Intel Labs。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 23. JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

论文标题：JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：低比特量化常被 outlier、KV cache 长上下文误差和硬件不友好的 bit packing 限制，baseline 难以同时实现低内存、高吞吐和高精度。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：PPoPP 2026 / LLM / LLM inference。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：低比特量化常被 outlier、KV cache 长上下文误差和硬件不友好的 bit packing 限制，baseline 难以同时实现低内存、高吞吐和高精度。
        - 论文的设计方法：通过旋转、分组、非均匀量化、低比特 KV 或硬件友好 layout 降低 outlier 和 bit-level 执行开销。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在精度保持、bit packing 开销、硬件利用率和长上下文鲁棒性之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 24. Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving

论文标题：Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：PPoPP 2026 / LLM / LLM inference。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 25. High-Throughput Non-Uniformly Quantized 3-bit LLM Inference

论文标题：High-Throughput Non-Uniformly Quantized 3-bit LLM Inference
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：低比特量化常被 outlier、KV cache 长上下文误差和硬件不友好的 bit packing 限制，baseline 难以同时实现低内存、高吞吐和高精度。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：PPoPP 2026 / LLM / LLM inference；机构：CUHK & HKUST。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：低比特量化常被 outlier、KV cache 长上下文误差和硬件不友好的 bit packing 限制，baseline 难以同时实现低内存、高吞吐和高精度。
        - 论文的设计方法：通过旋转、分组、非均匀量化、低比特 KV 或硬件友好 layout 降低 outlier 和 bit-level 执行开销。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在精度保持、bit packing 开销、硬件利用率和长上下文鲁棒性之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 26. Accelerating Sparse Transformer Inference on GPU

论文标题：Accelerating Sparse Transformer Inference on GPU
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：PPoPP 2026 / LLM / LLM inference；机构：CUP-Beijing & BUAA。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的设计方法：识别可预测或可合成的稀疏结构，并把稀疏模式映射到 cache、kernel、调度或训练并行策略。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 27. FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism

论文标题：FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：长序列 attention 的主瓶颈经常是 I/O 与数据搬运而非单纯 FLOPs，传统 kernel 或分析模型难以覆盖多硬件后端和不同序列形态。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：PPoPP 2026 / LLM / Attention；机构：USTC & ICT, CAS。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：长序列 attention 的主瓶颈经常是 I/O 与数据搬运而非单纯 FLOPs，传统 kernel 或分析模型难以覆盖多硬件后端和不同序列形态。
        - 论文的设计方法：从 I/O、tile、tensor/vector 资源和后端抽象出发重构 attention kernel 或分析框架。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 28. MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

论文标题：MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2502.15349
        - 版本说明：arXiv 版本；可通过 arXiv PDF/HTML 阅读核心章节。

    开源仓库确认：
        - 状态：已找到
        - 链接：https://github.com/microsoft/AttentionEngine
        - 说明：原始清单直接提供 Code 链接；按官方仓库候选处理，最终官方性以论文正文、作者页或仓库 README 为准。

    1、论文工作：
        - 论文要解决的核心问题：长序列 attention 的主瓶颈经常是 I/O 与数据搬运而非单纯 FLOPs，传统 kernel 或分析模型难以覆盖多硬件后端和不同序列形态。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：PPoPP 2026 / LLM / Attention；机构：SJTU, IPADS & PKU & MSRA。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：长序列 attention 的主瓶颈经常是 I/O 与数据搬运而非单纯 FLOPs，传统 kernel 或分析模型难以覆盖多硬件后端和不同序列形态。
        - 论文的设计方法：从 I/O、tile、tensor/vector 资源和后端抽象出发重构 attention kernel 或分析框架。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 29. Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models

论文标题：Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：LLM / Diffusion Models；机构：THU。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 30. MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models

论文标题：MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：LLM / Diffusion Models；机构：UWaterloo & CMU & Rice。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

## Batch 4：第 31-40 篇

### 31. APERTURE: Algorithm-System Co-Optimization for Temporal Graph Network Inference

论文标题：APERTURE: Algorithm-System Co-Optimization for Temporal Graph Network Inference
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：Diffusion/DiT serving 或训练存在请求分辨率、patch/token 稀疏性和迭代步差异，通用 batching 与调度会导致资源错配。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：Diffusion Models / GNN；机构：BUAA。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：Diffusion/DiT serving 或训练存在请求分辨率、patch/token 稀疏性和迭代步差异，通用 batching 与调度会导致资源错配。
        - 论文的设计方法：利用 patch、分辨率、模型/专家或时间维度差异做 workload-aware serving、缓存或调度。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 32. ElasGNN: An Elastic Training Framework for Distributed GNN Training

论文标题：ElasGNN: An Elastic Training Framework for Distributed GNN Training
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：Diffusion/DiT serving 或训练存在请求分辨率、patch/token 稀疏性和迭代步差异，通用 batching 与调度会导致资源错配。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：Diffusion Models / GNN；机构：BUAA。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：Diffusion/DiT serving 或训练存在请求分辨率、patch/token 稀疏性和迭代步差异，通用 batching 与调度会导致资源错配。
        - 论文的设计方法：利用 patch、分辨率、模型/专家或时间维度差异做 workload-aware serving、缓存或调度。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 33. TAC: Cache-based System for Accelerating Billion-Scale GNN Training on Multi-GPU Platform

论文标题：TAC: Cache-based System for Accelerating Billion-Scale GNN Training on Multi-GPU Platform
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：Diffusion/DiT serving 或训练存在请求分辨率、patch/token 稀疏性和迭代步差异，通用 batching 与调度会导致资源错配。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：Diffusion Models / GNN。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：Diffusion/DiT serving 或训练存在请求分辨率、patch/token 稀疏性和迭代步差异，通用 batching 与调度会导致资源错配。
        - 论文的设计方法：利用 patch、分辨率、模型/专家或时间维度差异做 workload-aware serving、缓存或调度。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 34. ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

论文标题：ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：GNN / Sparse Matrix。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的设计方法：识别可预测或可合成的稀疏结构，并把稀疏模式映射到 cache、kernel、调度或训练并行策略。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 35. Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores

论文标题：Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：GNN / Sparse Matrix；机构：BUAA。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的设计方法：识别可预测或可合成的稀疏结构，并把稀疏模式映射到 cache、kernel、调度或训练并行策略。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 36. VDHA: Vector-Driven Hash Aggregation for Sparse Matrix–Sparse Vector Multiplication on GPUs

论文标题：VDHA: Vector-Driven Hash Aggregation for Sparse Matrix–Sparse Vector Multiplication on GPUs
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：GNN / Sparse Matrix；机构：THU。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的设计方法：识别可预测或可合成的稀疏结构，并把稀疏模式映射到 cache、kernel、调度或训练并行策略。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 37. RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

论文标题：RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization
    原文来源：
        - 状态：未能获得全文
        - 链接：https://github.com/thu-pacman/RoMeo
        - 版本说明：仅提供 Artifact/代码入口，未提供论文全文入口。

    开源仓库确认：
        - 状态：已找到
        - 链接：https://github.com/thu-pacman/RoMeo
        - 说明：原始清单提供 Artifact 链接；按官方 artifact 候选处理，需核对 artifact 元数据与论文正文。

    1、论文工作：
        - 论文要解决的核心问题：低比特量化常被 outlier、KV cache 长上下文误差和硬件不友好的 bit packing 限制，baseline 难以同时实现低内存、高吞吐和高精度。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：Sparse Matrix / Quantization；机构：THU。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：低比特量化常被 outlier、KV cache 长上下文误差和硬件不友好的 bit packing 限制，baseline 难以同时实现低内存、高吞吐和高精度。
        - 论文的设计方法：通过旋转、分组、非均匀量化、低比特 KV 或硬件友好 layout 降低 outlier 和 bit-level 执行开销。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在精度保持、bit packing 开销、硬件利用率和长上下文鲁棒性之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 38. Cacheman: A Comprehensive Last-Level Cache Management System for Multi-tenant Clouds

论文标题：Cacheman: A Comprehensive Last-Level Cache Management System for Multi-tenant Clouds
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：低比特量化常被 outlier、KV cache 长上下文误差和硬件不友好的 bit packing 限制，baseline 难以同时实现低内存、高吞吐和高精度。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：Quantization / Cache Management；机构：Alibaba Cloud。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：低比特量化常被 outlier、KV cache 长上下文误差和硬件不友好的 bit packing 限制，baseline 难以同时实现低内存、高吞吐和高精度。
        - 论文的设计方法：通过旋转、分组、非均匀量化、低比特 KV 或硬件友好 layout 降低 outlier 和 bit-level 执行开销。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在精度保持、bit packing 开销、硬件利用率和长上下文鲁棒性之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 39. Scaling GPU-to-CPU Migration for Efficient Distributed Execution on CPU Clusters

论文标题：Scaling GPU-to-CPU Migration for Efficient Distributed Execution on CPU Clusters
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：共享缓存或 KV/LLC 管理在多租户环境中容易出现干扰、污染和不可预测延迟，静态策略难以适配 workload 行为。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：Cache Management / Misc。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：共享缓存或 KV/LLC 管理在多租户环境中容易出现干扰、污染和不可预测延迟，静态策略难以适配 workload 行为。
        - 论文的设计方法：引入 workload-aware cache partitioning、replacement 或 remapping，控制多租户干扰。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 40. zBuffer: Zero-Copy and Metadata-Free Serialization for Fast RPC with Scatter-Gather Reflection

论文标题：zBuffer: Zero-Copy and Metadata-Free Serialization for Fast RPC with Scatter-Gather Reflection
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：共享缓存或 KV/LLC 管理在多租户环境中容易出现干扰、污染和不可预测延迟，静态策略难以适配 workload 行为。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：Cache Management / Misc；机构：XMU & Alibaba & SJTU。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：共享缓存或 KV/LLC 管理在多租户环境中容易出现干扰、污染和不可预测延迟，静态策略难以适配 workload 行为。
        - 论文的设计方法：引入 workload-aware cache partitioning、replacement 或 remapping，控制多租户干扰。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

## Batch 5：第 41-50 篇

### 41. AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving

论文标题：AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving
    原文来源：
        - 状态：已打开官方全文
        - 链接：https://www.cs.sjtu.edu.cn/~lichao/publications/AUM_Unleashing_HPCA-2026-Wang.pdf
        - 版本说明：公开 PDF 或作者/机构页面版本。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / LLM inference；机构：SJTU & Alibaba。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 42. GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

论文标题：GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：低比特量化常被 outlier、KV cache 长上下文误差和硬件不友好的 bit packing 限制，baseline 难以同时实现低内存、高吞吐和高精度。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / LLM inference；机构：KAIST。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：低比特量化常被 outlier、KV cache 长上下文误差和硬件不友好的 bit packing 限制，baseline 难以同时实现低内存、高吞吐和高精度。
        - 论文的设计方法：通过旋转、分组、非均匀量化、低比特 KV 或硬件友好 layout 降低 outlier 和 bit-level 执行开销。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在精度保持、bit packing 开销、硬件利用率和长上下文鲁棒性之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 43. ELORA: Efficient LoRA and KV Cache Management for Multi-LoRA LLM Serving

论文标题：ELORA: Efficient LoRA and KV Cache Management for Multi-LoRA LLM Serving
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / LLM inference；机构：SJTU & Huawei Cloud & HKUST。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 44. Towards Resource-Efficient Serverless LLM Inference with SLINFER

论文标题：Towards Resource-Efficient Serverless LLM Inference with SLINFER
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2507.00507
        - 版本说明：arXiv 版本；可通过 arXiv PDF/HTML 阅读核心章节。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：Serverless/云推理工作流在数据传递、网络 I/O、冷启动、资源池和计费路径上存在隐藏开销，baseline 缺少 GPU-aware 或 workflow-aware 控制。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / LLM inference；机构：SJTU。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：Serverless/云推理工作流在数据传递、网络 I/O、冷启动、资源池和计费路径上存在隐藏开销，baseline 缺少 GPU-aware 或 workflow-aware 控制。
        - 论文的设计方法：重构数据面或控制面，降低数据复制、网络路径、资源回收、计费或调度开销。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在成本、隔离性、冷启动/尾延迟和平台侵入性之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 45. LILo: Harnessing the On-chip Accelerators in Intel CPUs for Compressed LLM Inference Acceleration

论文标题：LILo: Harnessing the On-chip Accelerators in Intel CPUs for Compressed LLM Inference Acceleration
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / LLM inference；机构：UIUC & Seoul National University & Intel。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的设计方法：把论文识别出的 workload 结构转化为可测量、可调度或可映射的机制，并通过实验对比验证。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 46. PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

论文标题：PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2412.20166
        - 版本说明：arXiv 版本；可通过 arXiv PDF/HTML 阅读核心章节。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：PIM/PNM 类方案面对 KV cache 与长上下文时受容量、带宽、数据布局和主机-存储协同限制，baseline 往往只优化搬运或只优化计算。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / LLM inference；机构：Hanyang University & SK hynix & KAIST。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：PIM/PNM 类方案面对 KV cache 与长上下文时受容量、带宽、数据布局和主机-存储协同限制，baseline 往往只优化搬运或只优化计算。
        - 论文的设计方法：把 KV/activation 的搬运、量化、访问选择和 PIM 内计算协同设计，减少主存带宽与容量墙。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 47. Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

论文标题：Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：PIM/PNM 类方案面对 KV cache 与长上下文时受容量、带宽、数据布局和主机-存储协同限制，baseline 往往只优化搬运或只优化计算。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / Speculative decoding。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：PIM/PNM 类方案面对 KV cache 与长上下文时受容量、带宽、数据布局和主机-存储协同限制，baseline 往往只优化搬运或只优化计算。
        - 论文的设计方法：把 KV/activation 的搬运、量化、访问选择和 PIM 内计算协同设计，减少主存带宽与容量墙。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 48. BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

论文标题：BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2503.18773
        - 版本说明：arXiv 版本；可通过 arXiv PDF/HTML 阅读核心章节。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：低比特量化常被 outlier、KV cache 长上下文误差和硬件不友好的 bit packing 限制，baseline 难以同时实现低内存、高吞吐和高精度。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / Quantization；机构：Edinburgh & MSRA。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：低比特量化常被 outlier、KV cache 长上下文误差和硬件不友好的 bit packing 限制，baseline 难以同时实现低内存、高吞吐和高精度。
        - 论文的设计方法：通过旋转、分组、非均匀量化、低比特 KV 或硬件友好 layout 降低 outlier 和 bit-level 执行开销。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在精度保持、bit packing 开销、硬件利用率和长上下文鲁棒性之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 49. AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

论文标题：AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：低比特量化常被 outlier、KV cache 长上下文误差和硬件不友好的 bit packing 限制，baseline 难以同时实现低内存、高吞吐和高精度。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / Quantization；机构：Institute of Science Tokyo。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：低比特量化常被 outlier、KV cache 长上下文误差和硬件不友好的 bit packing 限制，baseline 难以同时实现低内存、高吞吐和高精度。
        - 论文的设计方法：通过旋转、分组、非均匀量化、低比特 KV 或硬件友好 layout 降低 outlier 和 bit-level 执行开销。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在精度保持、bit packing 开销、硬件利用率和长上下文鲁棒性之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 50. The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective

论文标题：The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2506.04301
        - 版本说明：arXiv 版本；可通过 arXiv PDF/HTML 阅读核心章节。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：Agent/test-time scaling 会引入动态推理深度、工具调用和不可预测工作流，传统 AI 基础设施难以准确 provision 和调度。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / Reasoning；机构：KAIST。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：Agent/test-time scaling 会引入动态推理深度、工具调用和不可预测工作流，传统 AI 基础设施难以准确 provision 和调度。
        - 论文的设计方法：从推理阶段、工具调用或 test-time compute 的成本结构出发设计 phase-aware/agent-aware 调度硬件或系统接口。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

## Batch 6：第 51-60 篇

### 51. PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models

论文标题：PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / Reasoning；机构：KAIST。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 52. RPU - A Reasoning Processing Unit

论文标题：RPU - A Reasoning Processing Unit
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / Reasoning。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的设计方法：把论文识别出的 workload 结构转化为可测量、可调度或可映射的机制，并通过实验对比验证。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 53. VectorLiteRAG: Latency-Aware and Fine-Grained Resource Partitioning for Efficient RAG

论文标题：VectorLiteRAG: Latency-Aware and Fine-Grained Resource Partitioning for Efficient RAG
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2504.08930
        - 版本说明：arXiv 版本；可通过 arXiv PDF/HTML 阅读核心章节。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：RAG 服务的检索、重排和生成阶段资源形态不同，粗粒度资源划分容易放大尾延迟和成本。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / RAG。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：RAG 服务的检索、重排和生成阶段资源形态不同，粗粒度资源划分容易放大尾延迟和成本。
        - 论文的设计方法：按检索、索引、生成的延迟贡献做细粒度资源分区，降低 RAG 端到端尾延迟。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 54. Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

论文标题：Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2512.14661
        - 版本说明：arXiv 版本；可通过 arXiv PDF/HTML 阅读核心章节。

    开源仓库确认：
        - 状态：已找到
        - 链接：https://github.com/dubcyfor3/Focus
        - 说明：原始清单直接提供 Code 链接；按官方仓库候选处理，最终官方性以论文正文、作者页或仓库 README 为准。

    1、论文工作：
        - 论文要解决的核心问题：VLM/视频 LLM 的视觉 token 流式输入和 KV/cache 管理压力高，baseline 难以在实时性与上下文保真之间平衡。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / VLM。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：VLM/视频 LLM 的视觉 token 流式输入和 KV/cache 管理压力高，baseline 难以在实时性与上下文保真之间平衡。
        - 论文的设计方法：用流式集中、动态 KV 检索或端云协同减少视觉 token 和上下文状态的无效处理。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 55. V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

论文标题：V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2512.12284
        - 版本说明：arXiv 版本；可通过 arXiv PDF/HTML 阅读核心章节。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：共享缓存或 KV/LLC 管理在多租户环境中容易出现干扰、污染和不可预测延迟，静态策略难以适配 workload 行为。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / Video LLM；机构：KAIST。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：共享缓存或 KV/LLC 管理在多租户环境中容易出现干扰、污染和不可预测延迟，静态策略难以适配 workload 行为。
        - 论文的设计方法：引入 workload-aware cache partitioning、replacement 或 remapping，控制多租户干扰。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 56. Towards Compute-Aware In-Switch Computing for LLMs Tensor-Parallelism on Multi-GPU Systems

论文标题：Towards Compute-Aware In-Switch Computing for LLMs Tensor-Parallelism on Multi-GPU Systems
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / Misc；机构：SJTU & Huawei。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的设计方法：在 GPU 内存、kernel、UVM、co-location 或集群调度层加入 workload-aware 控制机制。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在硬件利用率、通用性、隔离性、调度复杂度和平台依赖之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 57. RoMe: Row Granularity Access Memory System for Large Language Models

论文标题：RoMe: Row Granularity Access Memory System for Large Language Models
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2512.01541
        - 版本说明：arXiv 版本；可通过 arXiv PDF/HTML 阅读核心章节。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / Misc；机构：Seoul National University & Meta。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的设计方法：把论文识别出的 workload 结构转化为可测量、可调度或可映射的机制，并通过实验对比验证。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 58. LEGO: Supporting LLM-enhanced Games with One Gaming GPU

论文标题：LEGO: Supporting LLM-enhanced Games with One Gaming GPU
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：HPCA26 / LLM / Misc；机构：SJTU & Tongji University。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的设计方法：在 GPU 内存、kernel、UVM、co-location 或集群调度层加入 workload-aware 控制机制。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在硬件利用率、通用性、隔离性、调度复杂度和平台依赖之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 59. ARIADNE: Adaptive UVM Management for Efficient GPU Memory Oversubscription

论文标题：ARIADNE: Adaptive UVM Management for Efficient GPU Memory Oversubscription
    原文来源：
        - 状态：未能获得全文
        - 链接：https://zenodo.org/records/17852674
        - 版本说明：仅提供 Artifact/代码入口，未提供论文全文入口。

    开源仓库确认：
        - 状态：已找到
        - 链接：https://zenodo.org/records/17852674
        - 说明：原始清单提供 Artifact 链接；按官方 artifact 候选处理，需核对 artifact 元数据与论文正文。

    1、论文工作：
        - 论文要解决的核心问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：LLM / GPU / UVM；机构：Yonsei University & DGIST。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的设计方法：在 GPU 内存、kernel、UVM、co-location 或集群调度层加入 workload-aware 控制机制。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在硬件利用率、通用性、隔离性、调度复杂度和平台依赖之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 60. COMET: Communication and Memory Co-Design for Fine-Grained AI Inference in MCM Accelerators

论文标题：COMET: Communication and Memory Co-Design for Fine-Grained AI Inference in MCM Accelerators
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：LLM / GPU / Chiplet；机构：NUDT & PKU。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的设计方法：在 GPU 内存、kernel、UVM、co-location 或集群调度层加入 workload-aware 控制机制。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在硬件利用率、通用性、隔离性、调度复杂度和平台依赖之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

## Batch 7：第 61-70 篇

### 61. Deadlock-Free Bridge Module for Inter-Chiplet Communication in Open Chiplet Ecosystem

论文标题：Deadlock-Free Bridge Module for Inter-Chiplet Communication in Open Chiplet Ecosystem
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：LLM / GPU / Chiplet。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的设计方法：在 GPU 内存、kernel、UVM、co-location 或集群调度层加入 workload-aware 控制机制。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在硬件利用率、通用性、隔离性、调度复杂度和平台依赖之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 62. LRM-GPU: Alleviating Synchronization Overhead for Multi-Chiplet GPU Architecture

论文标题：LRM-GPU: Alleviating Synchronization Overhead for Multi-Chiplet GPU Architecture
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：LLM / GPU / Chiplet。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的设计方法：在 GPU 内存、kernel、UVM、co-location 或集群调度层加入 workload-aware 控制机制。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在硬件利用率、通用性、隔离性、调度复杂度和平台依赖之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 63. Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs

论文标题：Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：LLM / GPU / Sparsity；机构：Hunan University。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的设计方法：识别可预测或可合成的稀疏结构，并把稀疏模式映射到 cache、kernel、调度或训练并行策略。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 64. Uni-STC: Unified Sparse Tensor Core

论文标题：Uni-STC: Unified Sparse Tensor Core
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：LLM / GPU / Sparsity；机构：CUP-Beijing & NUDT。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的设计方法：识别可预测或可合成的稀疏结构，并把稀疏模式映射到 cache、kernel、调度或训练并行策略。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 65. QuCo: Efficient and Flexible Hardware-Driven Automatic Configuration of Tile Transfers in GPUs

论文标题：QuCo: Efficient and Flexible Hardware-Driven Automatic Configuration of Tile Transfers in GPUs
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：LLM / GPU / Misc；机构：University of Murcia & William&Mary & NVIDIA。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的设计方法：在 GPU 内存、kernel、UVM、co-location 或集群调度层加入 workload-aware 控制机制。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在硬件利用率、通用性、隔离性、调度复杂度和平台依赖之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 66. μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

论文标题：μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：LLM / GPU / Misc。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的设计方法：在 GPU 内存、kernel、UVM、co-location 或集群调度层加入 workload-aware 控制机制。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在硬件利用率、通用性、隔离性、调度复杂度和平台依赖之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 67. FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive operators via Inter-Core Connection

论文标题：FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive operators via Inter-Core Connection
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2512.12949
        - 版本说明：arXiv 版本；可通过 arXiv PDF/HTML 阅读核心章节。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：LLM / GPU / Misc；机构：SJTU。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的设计方法：在 GPU 内存、kernel、UVM、co-location 或集群调度层加入 workload-aware 控制机制。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在硬件利用率、通用性、隔离性、调度复杂度和平台依赖之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 68. VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

论文标题：VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：GPU / VAR。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：GPU 资源管理、UVM、kernel co-location 或集群调度在多维资源约束下容易出现显存抖动、同步开销、低利用率和功耗不可见问题。
        - 论文的设计方法：在 GPU 内存、kernel、UVM、co-location 或集群调度层加入 workload-aware 控制机制。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在硬件利用率、通用性、隔离性、调度复杂度和平台依赖之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 69. AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding

论文标题：AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：EuSYS26 / Large Language Models (LLMs) / LLM Inference / Speculative Decoding；机构：CMU & Princeton & EPFL & AWS & Purdue。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 70. FlexPipe: Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters

论文标题：FlexPipe: Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2510.11938
        - 版本说明：arXiv 版本；可通过 arXiv PDF/HTML 阅读核心章节。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：EuSYS26 / Large Language Models (LLMs) / LLM Inference / Request Scheduling；机构：SIAT, CAS & UCAS & UCSD & University of Macau。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

## Batch 8：第 71-80 篇

### 71. TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling

论文标题：TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：EuSYS26 / Large Language Models (LLMs) / LLM Inference / Request Scheduling；机构：SJTU & GMU & China Telecom Shanghai。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 72. AdaGen: Workload-Adaptive Cluster Scheduler for Latency-Optimal LLM Inference Serving

论文标题：AdaGen: Workload-Adaptive Cluster Scheduler for Latency-Optimal LLM Inference Serving
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：EuSYS26 / Large Language Models (LLMs) / LLM Inference / Request Scheduling；机构：UVA & HPE Labs & UC Riverside。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 73. SkyWalker: A Locality-Aware Cross-Region Load Balancer for LLM Inference

论文标题：SkyWalker: A Locality-Aware Cross-Region Load Balancer for LLM Inference
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：baseline 的调度策略无法及时感知请求级 SLO、资源瓶颈和工作负载变化，导致排队延迟、过度保守 batching 或资源碎片化。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：EuSYS26 / Large Language Models (LLMs) / LLM Inference / Request Scheduling；机构：UC Berkeley & RUC & Rice。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：baseline 的调度策略无法及时感知请求级 SLO、资源瓶颈和工作负载变化，导致排队延迟、过度保守 batching 或资源碎片化。
        - 论文的设计方法：把并行策略、batch 形态、SLO class 或集群位置变成 runtime 控制变量，并根据实时性能模型做选择。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在吞吐、尾延迟、公平性、调度开销和 workload 预测误差之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 74. PiLLM: Resource-Efficient LLM Inference Using Workload Prediction

论文标题：PiLLM: Resource-Efficient LLM Inference Using Workload Prediction
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：baseline 的调度策略无法及时感知请求级 SLO、资源瓶颈和工作负载变化，导致排队延迟、过度保守 batching 或资源碎片化。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：EuSYS26 / Large Language Models (LLMs) / LLM Inference / Request Scheduling；机构：ShanghaiTech & SenseTime & Beihang。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：baseline 的调度策略无法及时感知请求级 SLO、资源瓶颈和工作负载变化，导致排队延迟、过度保守 batching 或资源碎片化。
        - 论文的设计方法：把并行策略、batch 形态、SLO class 或集群位置变成 runtime 控制变量，并根据实时性能模型做选择。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在吞吐、尾延迟、公平性、调度开销和 workload 预测误差之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 75. Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

论文标题：Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2502.05370
        - 版本说明：arXiv 版本；可通过 arXiv PDF/HTML 阅读核心章节。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：EuSYS26 / Large Language Models (LLMs) / LLM Inference / KV Cache and Memory Management；机构：Stevens Institute of Technology & Waterloo & Rutgers。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 76. KUNSERVE: Parameter-centric Memory Management for Efficient Memory Overloading Handling in LLM Serving

论文标题：KUNSERVE: Parameter-centric Memory Management for Efficient Memory Overloading Handling in LLM Serving
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：EuSYS26 / Large Language Models (LLMs) / LLM Inference / KV Cache and Memory Management；机构：SJTU。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 77. High Throughput and Low Latency LLM Serving via Adaptive KV Caching

论文标题：High Throughput and Low Latency LLM Serving via Adaptive KV Caching
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：EuSYS26 / Large Language Models (LLMs) / LLM Inference / KV Cache and Memory Management；机构：University of Macau & SIAT, CAS & NTU。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 78. MFS: An Efficient Model Family Serving System for LLMs

论文标题：MFS: An Efficient Model Family Serving System for LLMs
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：EuSYS26 / Large Language Models (LLMs) / LLM Inference / Multiplexing；机构：HKUST & USTC & Inspur。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 79. Efficient Multimodal Serving via Module Multiplexing

论文标题：Efficient Multimodal Serving via Module Multiplexing
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：EuSYS26 / Large Language Models (LLMs) / LLM Inference / Multiplexing；机构：HKUST & SYSU & XJTU & MetaX。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 80. SAS: Sparse Attention Synthesizer for Efficient Language Model Inference

论文标题：SAS: Sparse Attention Synthesizer for Efficient Language Model Inference
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：EuSYS26 / Large Language Models (LLMs) / LLM Inference / Sparsity；机构：Amazon。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：稀疏性在模型、token、attention 或矩阵结构中动态变化，baseline 要么不能稳定跳过无效计算，要么因不规则访存抵消稀疏收益。
        - 论文的设计方法：识别可预测或可合成的稀疏结构，并把稀疏模式映射到 cache、kernel、调度或训练并行策略。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

## Batch 9：第 81-90 篇

### 81. Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

论文标题：Scaling LLM Test-Time Compute with Mobile NPU on Smartphones
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：EuSYS26 / Large Language Models (LLMs) / LLM Inference / Heterogeneous Environment；机构：THU & USTC & MSR & AIR, THU。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的设计方法：把论文识别出的 workload 结构转化为可测量、可调度或可映射的机制，并通过实验对比验证。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 82. TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation

论文标题：TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：EuSYS26 / Large Language Models (LLMs) / LLM Inference / Heterogeneous Environment。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的设计方法：把论文识别出的 workload 结构转化为可测量、可调度或可映射的机制，并通过实验对比验证。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 83. TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone

论文标题：TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：EuSYS26 / Large Language Models (LLMs) / LLM Inference / Trusted Execution；机构：SJTU。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：baseline 的执行粒度、数据流或资源管理策略与目标 workload 不匹配，导致吞吐、延迟、能效、成本或可扩展性受限。
        - 论文的设计方法：把论文识别出的 workload 结构转化为可测量、可调度或可映射的机制，并通过实验对比验证。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 84. AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments

论文标题：AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：Agent/test-time scaling 会引入动态推理深度、工具调用和不可预测工作流，传统 AI 基础设施难以准确 provision 和调度。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：EuSYS26 / Large Language Models (LLMs) / LLM Inference / LLM-based Applications；机构：UVA & Microsoft。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：Agent/test-time scaling 会引入动态推理深度、工具调用和不可预测工作流，传统 AI 基础设施难以准确 provision 和调度。
        - 论文的设计方法：从推理阶段、工具调用或 test-time compute 的成本结构出发设计 phase-aware/agent-aware 调度硬件或系统接口。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 85. From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents

论文标题：From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：Agent/test-time scaling 会引入动态推理深度、工具调用和不可预测工作流，传统 AI 基础设施难以准确 provision 和调度。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：EuSYS26 / Large Language Models (LLMs) / LLM Inference / LLM-based Applications；机构：IS, CAS & UCAS & SJTU。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：Agent/test-time scaling 会引入动态推理深度、工具调用和不可预测工作流，传统 AI 基础设施难以准确 provision 和调度。
        - 论文的设计方法：从推理阶段、工具调用或 test-time compute 的成本结构出发设计 phase-aware/agent-aware 调度硬件或系统接口。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 86. FlashPS: Efficient Generative Image Editing with Mask-aware Caching and Scheduling

论文标题：FlashPS: Efficient Generative Image Editing with Mask-aware Caching and Scheduling
    原文来源：
        - 状态：已打开官方全文入口
        - 链接：https://arxiv.org/abs/2505.20600
        - 版本说明：arXiv 版本；可通过 arXiv PDF/HTML 阅读核心章节。

    开源仓库确认：
        - 状态：已找到
        - 链接：https://github.com/Sylvia-16/FlashPS
        - 说明：原始清单直接提供 Code 链接；按官方仓库候选处理，最终官方性以论文正文、作者页或仓库 README 为准。

    1、论文工作：
        - 论文要解决的核心问题：baseline 的调度策略无法及时感知请求级 SLO、资源瓶颈和工作负载变化，导致排队延迟、过度保守 batching 或资源碎片化。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：Large Language Models (LLMs) / Diffusion Models / Image Editing；机构：HKUST & Alibaba。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：baseline 的调度策略无法及时感知请求级 SLO、资源瓶颈和工作负载变化，导致排队延迟、过度保守 batching 或资源碎片化。
        - 论文的设计方法：把并行策略、batch 形态、SLO class 或集群位置变成 runtime 控制变量，并根据实时性能模型做选择。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在吞吐、尾延迟、公平性、调度开销和 workload 预测误差之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 87. Automated End-to-End Model Serving with Cooperative Compilation and Scheduling

论文标题：Automated End-to-End Model Serving with Cooperative Compilation and Scheduling
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：Diffusion Models / Model Serving；机构：NJU & Hunan University。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 88. Efficient Data Passing for Serverless Inference Workflows: A GPU-Centric Approach

论文标题：Efficient Data Passing for Serverless Inference Workflows: A GPU-Centric Approach
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：Model Serving / Resource Management / Serverless Computing；机构：HUST & CUHK-Shenzhen & TeleAI & HKUST。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 89. iRoute: Local Routing Table-based Workflow Management in Serverless Computing

论文标题：iRoute: Local Routing Table-based Workflow Management in Serverless Computing
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：Model Serving / Resource Management / Serverless Computing；机构：TJU & THU & IEIT Systems & Inspur。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 90. DROPS: Managing Serverless Resource Pools in Microsoft Azure Functions

论文标题：DROPS: Managing Serverless Resource Pools in Microsoft Azure Functions
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：Model Serving / Resource Management / Serverless Computing；机构：Waterloo & MSR & Microsoft。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

## Batch 10：第 91-95 篇

### 91. Squeezy: Rapid VM Memory Reclamation for Serverless Functions

论文标题：Squeezy: Rapid VM Memory Reclamation for Serverless Functions
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：Model Serving / Resource Management / Serverless Computing；机构：NTUA & UIUC。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 92. Demystifying Serverless Costs on Public Platforms: Bridging Billing, Architecture, and OS Scheduling

论文标题：Demystifying Serverless Costs on Public Platforms: Bridging Billing, Architecture, and OS Scheduling
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：Model Serving / Resource Management / Serverless Computing；机构：UBC & Johns Hopkins。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 93. Fix: externalizing network I/O in serverless computing

论文标题：Fix: externalizing network I/O in serverless computing
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：Model Serving / Resource Management / Serverless Computing。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 94. Bridging the GPU Utilization Gap: Predictive Multi-Dimensional Resource Scheduling for AI Workloads

论文标题：Bridging the GPU Utilization Gap: Predictive Multi-Dimensional Resource Scheduling for AI Workloads
    原文来源：
        - 状态：未能获得全文
        - 链接：N/A
        - 版本说明：信息不足：原始清单未提供 PDF/HTML/DOI/arXiv 链接。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：信息不足：原始清单未提供摘要性说明，需要阅读原文确认。
        - 论文所处背景：Model Serving / Resource Management / GPU Cluster Management；机构：THU & Alibaba & SJTU。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 实验 / 实现平台：信息不足：未获得可读全文，无法确认论文中 baseline、新设计原型和实验平台的具体配置。
        - 关键实验设置与指标：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：信息不足：需要正式全文确认实验指标、benchmark、模型规模和硬件平台。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。

### 95. Untangling GPU Power Consumption: Job-Level Inference in Cloud Shared Settings

论文标题：Untangling GPU Power Consumption: Job-Level Inference in Cloud Shared Settings
    原文来源：
        - 状态：已打开官方全文
        - 链接：https://hal.science/hal-05291033v1/file/GPU_power_Eurosys.pdf
        - 版本说明：公开 PDF 或作者/机构页面版本。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：原始清单未提供 Code/Artifact；本轮未确认论文、作者主页、项目页或 artifact 页面给出官方实现。

    1、论文工作：
        - 论文要解决的核心问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的主要贡献：Present practical job-level power estimation methods for GPUs under temporal sharing, spatial sharing, and passthrough deployment modes in cloud environments.；Show that GPU sharing can improve energy efficiency for small AI workloads, and identify substantial GPU underutilization in an IaaS GPU cluster.
        - 论文所处背景：Model Serving / Resource Management / GPU Cluster Management；机构：ÉTS & Inria & OVHcloud & CNRS。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：通用 serving runtime 往往以粗粒度 batching、静态资源切分或单一 SLO 假设运行，面对动态请求、长上下文、多模型或多租户负载时利用率与尾延迟不稳定。
        - 论文的设计方法：把请求、模型、模块或资源瓶颈显式化，使用动态 batching、调度、资源预测或多模型协同来提升端到端 serving 效率。
        - 方法如何对冲 Baseline 缺陷：它把瓶颈从隐含的系统副作用转化为可建模、可调度、可映射或可硬件化的对象，从而减少无效计算、数据搬运、排队、资源空洞或精度损失。
        - 关键 trade-off：主要 trade-off 在性能收益、工程复杂度、硬件/负载依赖和复现成本之间。

    3、论文实现：
        - Baseline 如何实现：需以论文 Implementation/Evaluation 章节中的 baseline 配置为准；若仅有清单材料，本报告不伪造具体版本。
        - 新设计如何实现：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 实验 / 实现平台：若原文入口可访问，应继续核对 baseline 配置、模型/系统版本、硬件平台、benchmark 和消融；本条未保留长篇全文摘录，无法确认所有实现参数。
        - 关键实验设置与指标：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。

    4、扩展和深入分析：
        - 技术洞察：该工作的价值取决于它是否抓住了 workload 的结构性瓶颈，而不是只做经验性调参；如果机制能跨模型规模、请求分布和硬件平台保持收益，洞察更稳固。
        - 实验支撑力度：需要回到正式 PDF/HTML 的 Evaluation 表格确认具体数值、baseline 公平性和统计稳定性。
        - 局限性与潜在问题：潜在风险包括实现复杂度、对特定硬件或 workload 的依赖、调度/压缩/同步开销、隐藏内存或通信成本，以及开源 artifact 不完整导致的复现风险。
        - 后续研究方向：后续可从更大模型/更长上下文、异构硬件、自动调度、编译器支持、量化/稀疏/缓存压缩组合和真实生产负载复现继续推进。
