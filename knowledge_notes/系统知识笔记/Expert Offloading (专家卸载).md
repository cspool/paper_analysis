## Expert Offloading (专家卸载)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Offloading 是将 MoE 模型中部分 expert 参数从 GPU HBM 卸载到更慢但更廉价的存储介质（CPU DRAM 或 SSD）的技术，使大模型能在内存受限的 GPU 上运行。MoE 模型的稀疏激活特性（每个 token 仅激活 k 个 expert）使 offloading 天然适用——大部分 expert 在大部分时间内不被使用，可以驻留在慢速存储中。典型方案：将全部非-expert 权重（attention、LayerNorm 等）保持在 GPU，仅 offload expert FFN 权重；使用 LRU caching 保留最近使用的 expert 在 HBM；按需从 CPU/SSD 加载被激活的 off-chip expert。

从系统架构角度拆解术语：
Expert Offloading 在 MoE serving 中的执行流程（以 Mixtral-8x7B on Tesla T4 16GB 为例，基于 dvmazur/mixtral-offloading）：
1. **初始化**：模型总参数 ~87GB。Attention + LayerNorm 等非-expert 权重保持在 GPU（~2GB）。所有 8×32 = 256 个 expert 权重压缩（quantization）后存储在 CPU DRAM。GPU HBM 中缓存 k_keep 个 expert per layer。
2. **Prefill 阶段**：prompt tokens 流经各层。每层 gating network 输出 Top-K experts → 检查 residency table → 若缺失则 CPU→GPU 传输 expert 权重（PCIe 瓶颈，图 2 显示 CPU read >> GPU read）→ expert 计算 → 更新 LRU cache（可能 evict 最久未用的 expert 回 CPU）。
3. **Decode 阶段**：逐 token 解码，每步重复上述过程。Expert swap 成为 memory-bound 解码的主要延迟来源。
4. **优化技术**：量化压缩（减少传输量）、LRU caching（保留热 expert）、prefetching（预测下一层 expert 提前加载）、residency-aware routing（ERAS，倾向选择已驻留 expert）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源实现：`dvmazur/mixtral-offloading`（https://github.com/dvmazur/mixtral-offloading）——支持 Mixtral-8x7B 在 Tesla T4 16GB 上运行，使用 quantization + LRU caching + speculative prefetching。
- 其他系统：MoE-Infinity（activation-aware prefetching + caching，支持 CPU DRAM + SSD 分层 offloading，核心创新为 Sparsity-Aware Expert Cache：利用 batch_size=1 时的请求级 expert 激活稀疏性和偏斜重用模式，通过 EAMC（Expert Activation Matrix Collection）追踪历史请求的 request-level EAM 并用余弦距离匹配，PredictEAM 预测未来 expert 激活概率，指导 GPU expert cache 的淘汰和预取）、DeepSpeed-Inference（layer-wise offloading，非 expert-aware）、FineMoE（fine-grained expert offloading with probability-aware cache management）。
- 关键 trade-off：内存节省 vs 延迟增加。Offload 越多，GPU memory 需求越低，但 CPU→GPU 传输延迟越高。

**LUT Offloading (MoLE 范式)**：与 expert offloading 互补的替代方案。MoLE 将 expert 重参数化为 Lookup Table（LUT = 所有 vocabulary token × 所有 expert 的预计算输出），LUT 整体 offload 到 CPU/disk。推理时仅加载当前 input_ids 对应的 lookup results（每 token dN 参数，如 1B MoLE-4E: d=2048, N=4 → 8KB），而非完整 expert 权重。对比：MoE expert offloading 每 step 加载 2dkD_r（~537MB for 1B MoE-10E），MoLE 的 LUT offloading 仅加载 dN（~8KB），通信量降低 ~60000×。代价：LUT 存储开销大（dN|V|，如 MoLE-16E 160M 时 LUT=7.4B 参数），但存储设备可扩展且随模型增大 LUT/Expert 比例下降。

- **Sub-Expert Offloading (MoE-Prism 范式)**：MoE-Prism 将 monolithic expert 分解为细粒度 sub-expert 后，offloading 从粗粒度整块加载变为精准按需传输。关键改进：(1) VRAM Cache Manager 将 GPU VRAM 作为 sub-expert 缓存（LRU 策略），CPU RAM 持所有 sub-expert。细粒度使内存碎片利用更高效——无法容纳一个完整 monolithic expert 的内存碎片可容纳多个子 expert，cache hit ratio 从 0.4375(28/64) 提升到 0.4453((28×4+2)/(64×4))（16GB 配置）。(2) Generation Step Orchestrator 每 token 步：router 确定 S_req(t) → 查询 cache 得 miss set S_miss(t) → 异步传输 S_miss(t) → 计算。每步延迟 L(t) = Latency_IO(S_miss) + Latency_compute(S_req)。(3) 细粒度消除粗粒度"量化误差"：原模型若 SLO 要求等效 4.2 个 expert 的计算量，必须加载 5 个完整 expert；MoE-Prism 仅加载 17 个 sub-expert (等效 4.25 expert)，减少 PCIe 传输量。

涉及论文标题：
- MoE-ERAS: Expert Residency Aware Selection
- Mixture of Lookup Experts
- MoE-Infinity Activation-Aware Expert Offloading for Efficient MoE Serving
- MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs
- MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts
- Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models
- Oracle-MoE: Locality-preserving Routing in the Oracle Space for Memory-constrained Large Language Model Inference

**模型级 Expert Offloading 适合度评估** (来自 "Not All Models Suit Expert Offloading", ICLR 2026)：论文提出并非所有 MoE 模型都同等适合 expert offloading——模型的路由行为（局部路由一致性）直接影响专家缓存的命中率。关键量化指标：(1) SRP (Segment Routing Best Performance)——expert 在 segment 内持续激活程度的固有度量；(2) SCH (Segment Cache Best Hit Rate)——在给定 cache size 下 oracle cache 的理论命中率上界，与 LRU/LFU 实际 hit rate 高度相关 (m=16 时 Pearson r > 90%)。部署建议：选择 SRP/SCH 高的模型（Group 1: LLaMA-MoE-v2, OLMoE, Qwen3, etc.）→ 简单的 LRU cache 即可获高命中率；Cache size ≈ 2× active experts 为大多数模型的最佳性价比点（SCH-ρ 曲线拐点）。论文补充发现 decoding 阶段 overhead 与局部路由一致性负相关 (r≈−0.3)，但 prefilling 阶段正相关 (r≈0.2)——因 prefilling 瓶颈在 expert-level 的 load balance。代码：https://github.com/ljcleo/moe-lrc。

**MoE-SpeQ 中对 Expert Offloading 瓶颈的量化**：在 A100-40G 上以 HuggingFace Transformers device_map offloading 运行 MoE 模型，PCIe 传输主导端到端时间——Mixtral-8x7B 中 Memory 占 98.9%（图 4），GPU compute < 15%。MoE-SpeQ 通过 speculative decoding 将 I/O latency 转化为 productive computation（draft 生成与 I/O overlap），首次将 speculative execution 与 expert offloading 协同设计。三阶段预取策略（Phase I cache priming → Phase II adaptive prefetch → Phase III cache saturation）使 cache 命中率达 96-99.85%（远超 LRU 的 29.2-76.56%）。

**Oracle-MoE 对 Expert Swapping 延迟的分析与缓解**：Oracle-MoE (ICML 2025) 从算法层面分析了 expert swapping 延迟的根因——token-level MoE routing 的 inter-token expert activation inconsistency（每 2 个连续 token 几乎都激活不同专家），导致频繁的 expert swapping 需求。在 Jetson Xavier NX (8GiB GPU) 上实测：memory-constrained inference 延迟是 full-model 的 15-30 倍，其中 50%-85% 来自 I/O swapping overhead。传统策略（FIFO、LRU、SwapMoE）采用工程化的经验性方法（优先保留高频/最近使用的 expert）无法从根本上解决激活不一致性问题。Oracle-MoE 通过 Oracle-Space-Based Routing（基于语义组的 routing）从源头减少 expert activation variation，使连续数百 token 不需切换专家，从而在 25% 内存预算下仅增加 3s 延迟（vs Switch Transformer 增加 2000%）。Oracle-MoE 同时提出 Expert Prediction Optimization：用第一层 embedding 预测深层 expert 激活（准确率 85%-95%），预加载 expert 以隐藏 I/O 延迟，进一步减少 10%-15% latency。

---
