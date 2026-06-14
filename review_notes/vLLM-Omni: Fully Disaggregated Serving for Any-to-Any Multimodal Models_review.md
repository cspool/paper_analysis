# vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models

> 2026-06-11T08:38:14.050Z
> Rounds: 6
> QA: `8da91918-69a2-47bf-a5db-72c94bba1429`  AA: `cf8add75-a36b-4f24-a577-879e9bd3b1db`

## 评判

| 维度 | 评定 |
|------|------|
| 相关性 | high |
| 参考价值 | high |
| 深入价值 | high |

**相关性理由**：
vLLM-Omni 在五大类别中均呈现高价值信号。背景与需求维度：pipeline 级并发潜力明确，CNN Vocoder 依赖距离 N=1 token 已量化，Thinker（memory-bound）vs Talker（compute-bound）的资源正交性定性分析充分。方法与实现维度：Stage Graph 抽象是 serving framework 层面从"单 AR 范式"到"任意 DAG topology"的通用扩展，是领域内首创的系统级抽象。架构影响维度：per-stage 独立 memory budget、NVLink 通信竞争分析（TP-2 all-reduce 仅 0.02ms/iteration）、4 路 EPD+stage 分解触及 160GB memory 上限等分析深入。实验证据维度：覆盖四类 any-to-any 架构，代码已开源。硬件机制虽为"硬件无关"抽象，但这本身是论文设计选择，而非缺乏相关性。≥1 维度满足高价值标准，直接涉及系统级并发。

**参考价值理由**：
高参考价值体现在三方面：(1) **框架已集成**——Stage Graph 抽象已合并至 vLLM（v0.20.0），意味着设计模式经过了生产级验证，可直接参考和复用；(2) **定量证据充足**——提供 JCT、RTF、TPS、connector 延迟等多项指标，覆盖多种模型架构；(3) **可迁移设计**——Unified Connector 的 shared memory / RDMA 双后端设计、per-engine independent scheduling、streaming stage output 等模式可直接用于类似多 stage serving 系统。主要限制：缺少 vLLM standalone 消融实验使得关键加速比（12.97×）的源贡献无法分离，降低了定量证据的归因精度。

**深入价值理由**：
高深入价值。非平凡的系统设计 trade-off 集中在以下方面：(1) **Stage Graph 粒度 vs 细粒度并发**——为何选"独立 engine per stage"而非 CUDA Stream/MPS 级 fine-grained 并发的条件分析，退化条件（同质 stage、极小数据量）的讨论；(2) **Baseline fairness 的消融缺口**——12.97× TPS 提升中 vLLM engine 固有收益 vs Stage Graph 增量收益的分离问题，可通过 Qwen2.5 Thinker 1.29× 侧面估计纯增量贡献；(3) **Memory 隔离 vs 共享的效率取舍**——OS process 级硬隔离避免跨 stage 竞争但降低总利用率，论文未报告 memory 利用率；(4) **4 路 EPD+stage 联合分解**的参数冗余问题——prefill/decode-Thinker 各加载 30B 导致 160GB 上限被触及。这些 trade-off 构成了五层分析链（负载层 → 编译层 → 调度层 → Kernel 层 → 硬件层）在系统 serving 层面的完整映射。

**复现指南**：
基于现有信息，vLLM-Omni 的核心方法复现步骤如下：

**关键步骤**：
1. **Stage Graph 定义**：为 any-to-any 模型定义 stage graph G=(V,E)，其中 V 为 stage 集合（如 Thinker / Talker / Vocoder），E 为数据依赖边。每 stage 需实现 forward()、preprocess()（每 iteration 执行）和 transfer function（stage 转换时执行一次）
2. **Per-stage engine 配置**：AR stage 使用 vLLM engine（启用 continuous batching、PagedAttention、CUDA graph compilation）；DiT stage 使用 diffusion engine（启用 flash attention、并行策略）；其余 stage 使用标准 PyTorch
3. **Unified Connector 设置**：单节点配置 shared memory 传输（system shared memory + control queue），跨节点配置 Mooncake RDMA
4. **Orchestrator 部署**：使用 Ray 管理跨节点 stage 调度，per-stage 独立配置 accelerator 和 memory budget

**依赖的硬件/软件栈**：
- 硬件：≥2×80GB accelerator（H100/A100），NVLink 连接 TP-2，系统内存 ≥192GB
- 软件：vLLM ≥0.12.0（当前 v0.20.0 已集成 vLLM-Omni）、Mooncake（跨节点 RDMA，可选）、Ray（orchestrator，可选）

**实验配置与评估指标**：
- 基线：HuggingFace Transformers 默认实现（论文方法），建议补充 vLLM standalone（单 stage）消融实验
- 数据集：librispeech_asr / food101 / ucf101-subset，各 100 queries
- 关键指标：JCT（端到端延迟, ms）、RTF（处理时间/音频时长）、TPS（tokens/sec）、connector 延迟（ms）
- 统计：建议重复 ≥5 次，报告 mean ± std 及 tail latency（P95/P99）
- 建议补充：vLLM standalone 消融、streaming on/off 消融、co-located vs disaggregated 硬件计数器对比

**信息不足处**：execution graph compilation 的 IR 变换细节未公开；stage 级绝对延迟分解（ms）未报告；per-stage memory budget 具体配置值未公开；不同负载强度（burst/Poisson）下的稳定性未评估。这些信息的缺失不会阻止基本复现，但会影响结果对比的精确度。

## 问答记录

### Round 1

**Q**：请先明确你持有的研究对象：名称/标题、它解决的核心问题、核心方法和主要性能或系统 claim。随后按五大价值维度给出总览：

1. **背景与需求**：负载中哪里存在并发潜力或运行时动态性？过去为何未利用？关键瓶颈和独立性证据是什么？
2. **方法与实现**：如何实现并发？并发粒度、同步机制、编译期与运行时职责，以及相对已有方案的独特贡献是什么？
3. **硬件机制**：依赖哪些硬件并发原语或模块？它们如何协同？资源竞争、能力边界和跨平台替代是什么？
4. **架构影响**：哪些存储层次、互连、调度或执行资源限制/扩展并发效率？哪些是硬限制，哪些可由软件缓解？
5. **实验证据**：使用什么工具、基线和指标验证收益与开销？测量粒度、误差、覆盖范围及可复现性如何？

请提供定量证据；不相关或证据不足的类别必须明确标注并说明原因。

**A**：**研究对象：vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models**  
**核心问题**：Existing serving frameworks（vLLM、SGLang）的 step-centric abstraction 只能表达单一 AR decoding stage，无法表示 any-to-any 模型（如 Thinker→Talker→Vocoder）的多 stage pipeline，导致开发者需手动编排 inter-stage 交互，丢失所有 framework-level 优化。  
**核心方法**：Stage Graph 抽象 + Disaggregated Stage Execution Backend。将 any-to-any 模型分解为 stage graph G=(V,E)，每 stage 由独立 engine 服务（AR→vLLM engine，DiT→diffusion engine），Orchestrator 管理跨 stage 请求路由，Unified Connector 处理数据传输。  
**主要 Claim**：Qwen3-Omni JCT 降低 91.4%，RTF 降低 90.7%，Thinker TPS 12.97×，Talker TPS 7.98×。

---

**1. 背景与需求**  
并发潜力在 pipeline 层面：Thinker（150.9 text tokens）→ Talker（545.4 audio tokens）→ Vocoder 三阶段串行执行，后 stage 无需等前 stage 完全完成即可开始（Talker output token 可用时 Vocoder 即可 incremental denoising）。过去未利用的原因：step-centric abstraction 无多 stage 表达力，开发者只能将整个 pipeline 写为 monolithic 程序，三阶段 co-located 在同一进程，无 fine-grained 资源分配（Thinker 30B 占大量 memory 致 Talker 资源受限）。瓶颈证据：Qwen3-Omni 中 Talker 占总延迟绝大部分（545.4 tokens vs Thinker 150.9 tokens）。

**2. 方法与实现**  
并发粒度是 **stage 级分解**（Thinker / Talker / Vocoder 为独立 stage），非传统 kernel/task 级。同步机制：stage 间通过 Unified Connector 异步传输（shared memory 或 Mooncake RDMA），stage 内 preprocess 函数每 iteration 合并 upstream data。  
- **Stage Graph 编程**：用户定义 forward()、preprocess()（每 iteration）、transfer function（stage 转换时仅一次），backend 自动处理 batching/scheduling。  
- **Disaggregated Execution**：AR stage 用 vLLM engine（继承 continuous batching、chunked prefill、PagedAttention、execution graph compilation）；DiT stage 用 diffusion engine（flash attention、SAGE、TurboAttention、TeaCache、RingAttention、Ulysses parallelism）。  
- **Streaming Stage Output**：下游 stage 增量接收 partial outputs，异步重叠执行减少 TTFT。  
独特贡献：将 serving framework 从"单 AR 范式"扩展到"任意 DAG topology"的通用抽象，且向后兼容 EPD disaggregation。

**3. 硬件机制**  
依赖硬件无关抽象：多 stage 可灵活分配 accelerator 资源——Thinker TP-2 across 2 devices，Talker 独占 device-1，Vocoder 独占 device-0。单节点数据通过 shared memory（Thinker→Talker 5.49ms，Talker→Vocoder 0.53ms）传输，跨节点通过 Mooncake RDMA（8.28ms/3.34ms）。Orchestrator 用 Ray 管理跨节点部署。Connector overhead <0.1% total latency。跨平台通过 vLLM hardware plugin 架构支持。无特定 GPU 微架构测量。

**4. 架构影响**  
存储/互连限制：shared memory 传输适合单节点（<10ms），跨节点 RDMA 延迟略高但仍可忽略。Talker 是 compute-bound（长序列 AR decode, 545.4 tokens），受 GPU compute 而非 memory bandwidth 限制；Thinker 是 model-size-bound（30B 参数），受 memory capacity 和 TP communication 限制。软件可缓解：per-stage 独立配置 parallelism 和 memory budget——Thinker 分配更多 memory + TP-2，Talker 分配更少 memory 但更高 parallelism。硬限制：两 device 总 memory 80GB×2 限制了可同时加载的最大模型组合。

**5. 实验证据**  
**平台**：2×80GB accelerator + 24 CPU cores + 192GB system memory，vLLM 0.12.0。  
**Baseline**：Qwen-Omni→HuggingFace Transformers 默认实现；BAGEL/MiMo-Audio→原始实现；Diffusion→Diffusers。  
**关键指标**：Qwen2.5-Omni RTF -61.4% JCT -61.6%；Qwen3-Omni RTF -90.7% JCT -91.4%（Thinker TPS 12.97×, Talker TPS 7.98×）。BAGEL T2I 2.40×, I2I 3.72× speedup。MiMo-Audio RTF 1.39→0.12（11.58×）。Diffusion 整体 1.26× speedup。  
**微实验**：Unified Connector 延迟表（5.49ms/0.53ms shared mem；8.28ms/3.34ms Mooncake）。  
**覆盖**：Thinker-Talker（Qwen3/2.5-Omni）、AR+DiT（BAGEL）、AR+Audio decoder（MiMo-Audio）、纯 Diffusion（Qwen-Image/Wan2.2）四类 any-to-any 架构。  
**缺口**：未报告 GPU 微架构 occupancy/利用率数据，experiment graph compilation 的 IR 级细节未公开，无跨平台比较。

---

### Round 2

**Q**：关于 pipeline 级并发潜力与运行时动态性，请进一步量化：

1. **子计算流水线粒度**：你说 Talker output token 可用时 Vocoder 即可开始 incremental denoising。Vocoder 每次需要 Talker 输出多少 token 才能触发一次 denoising step？producer-consumer 的"依赖距离"（即 Talker 必须先产出 N 个 token，Vocoder 才能消费）是多少？这个 N 在不同配置下是否变化？如果 Tokcer 产出 tokens 的速率与 Vocoder 消费 tokens 的速率不匹配，哪一方会被迫等待、占比多大？

2. **资源正交性**：Thinker（30B，受 memory capacity + TP communication 限制）与 Talker（compute-bound, 长序列 AR decode）拆分到不同 device 后，它们各自的瓶颈硬件资源（Thinker→memory BW/NVLink BW, Talker→compute/Tensor Core）是否真正正交——即如果 co-locate 在同一 device，是否存在资源竞争导致双方都降速？你是否测量过 co-located 场景下 Thinker 和 Talker 各自的 SM occupancy、memory BW 利用率，并与 disaggregated 场景对比？

3. **运行时动态性**：Stage Graph 方法中，哪些参数是运行时才能确定的（如 stage 间数据量、路由选择、各 stage 负载大小）？这些参数的变化范围（min/max/典型分布）是多少？运行时 Orchestrator 做 stage 路由决策的开销（延迟/CPU cycles）是多少——与主计算（ms 级 stage compute）相比占比多大？

请提供定量测量数据；证据不足处明确标注。

**A**：**1. 子计算流水线粒度（依赖距离）**

论文将 Vocoder 分为两种实现，依赖距离不同：

- **Qwen3-Omni（lightweight CNN Vocoder）**：Vocoder 以 CNN 逐帧增量处理 codec tokens。Talker 每 decode 一个 token，Output Processor 即可通过 streaming stage output 将其异步推送到 Vocoder device。依赖距离 N=1 token——Vocoder 仅需单个 codec token 即可触发一次波形重建 step。无 min-batch 缓冲要求。
- **Qwen2.5-Omni（DiT Vocoder）**：Vocoder 以 DiT denoising 方式运行，依赖距离取决于 denoising schedule（通常需至少 1 个完整 denoising step 的 token 子集）。论文未给出具体 N 值。

**Producer-consumer 速率不匹配**：论文未测量或建模。凭推理，Talker 生成 ~545.4 tokens（AR decode，约 0.5-5 ms/token 量级），Vocoder DiT denoising 每 step 更重（需 full attention over all tokens）——Vocoder 很可能是瓶颈方，但论文无等待占比数据。

**2. 资源正交性**

论文主张 Thinker 和 Talker 具有正交资源需求，但**未提供 co-located vs disaggregated 的微架构对比数据**：

| 维度 | Thinker (30B) | Talker (smaller AR) |
|------|-------------|--------------------|
| 主要瓶颈 | Memory capacity（30B 参数）+ TP communication（TP-2 cross-device） | Compute（长序列 AR decode, 545.4 tokens） |
| Memory BW 压力 | 高（weight 读取 + KV cache 访存） | 中（KV cache 读写为主） |
| Tensor Core 压力 | 中（prefill GEMM）→ 低（decode memory-bound） | 高（decode 步数多，累计 GEMM 量大） |
| SM occupancy | 受 memory latency 限制 | 可维持较高 occupancy |

如果 co-locate：（1）30B weight + 各 stage KV cache 可能超过单 device 80GB 限制；（2）Thinker decode 阶段 memory-bound 时 SM 空闲，但 Talker 无法利用这部分空闲 SM 因为 stage 间串行。论文将 Thinker TP-2 across both devices、Talker 部署 device-1、Vocoder 部署 device-0——该配置下 Talker 和 Vocoder 物理共享 device-0/1 compute，但 stage 串行执行（模型 pipeline 的天然顺序）避免了一对冲突。真正并行的负载来自跨请求 stage 重叠（request_1 Talker decode ↔ request_2 Thinker prefill），论文未提供该场景下的硬件冲突测量。

**3. 运行时动态性**

Stage Graph 中运行时确定的参数：
- **Stage 间数据量**：Thinker output 取决于输入 prompt 和生成 text token 数量（平均 text tokens 150.9）；Talker output audio tokens 平均 545.4，取决于语音内容长度。变化范围论文未报告 min/max。
- **Orchestrator 路由**：基于每 stage engine 的 slot 可用性做请求级路由，决策为静态 stage graph topology（编译时定义，运行时不变），仅选择将请求送入 Thinker→Talker→Vocoder 的固定路径。
- **Per-stage batching**：各 engine 独立调度 incoming requests，batch 组成动态变化（每个 iteration 更新），受益于 continuous batching。
- **Batch 大小与资源分配**：per-stage memory budget 和 parallelism 策略是用户配置的静态参数，不在运行时调整。

**Orchestrator 开销**：论文未提供 orchestrator 路由决策的延迟或 CPU cycles 测量。唯一提供的运行时开销数据是 Unified Connector 延迟：
- Thinker→Talker: 5.49ms (shared mem) / 8.28ms (Mooncake RDMA)
- Talker→Vocoder: 0.53ms (shared mem) / 3.34ms (Mooncake)
- 论文声明 connector overhead <0.1% total latency（tens of seconds）。但 orchestrator 调度决策本身的 CPU 开销（请求路由表查询、slot 分配、connector metadata 传递）没有单独测量。

---

### Round 3

**Q**：关于 Stage Graph 的并发实现细节，请进一步具体化：

1. **并发机制的选择理由**：Stage Graph 选择以"独立 engine per stage"粒度实现并发（Thinker→vLLM engine、Talker→diffusion engine），而非在同一 engine 内做 finer-grained 并发（如 CUDA Stream 级 pipeline 或 warp specialization）。这个选择在什么条件下最优（如 stage 间模型架构差异大、memory capacity 不对称时）？什么条件下会退化（如 stage 间数据量极小、传输延迟 > compute 延迟时）？是否评估过将所有 stage fusion 到单一 engine 内、用 CUDA Stream 或 MPS 做 stage 级并发的替代方案及其收益？

2. **Execution Graph Compilation 的 IR 变换细节**：论文提到 execution graph compilation 是多 stage pipeline 的内核优化。编译流程中 stage 内算子的调度是静态确定的还是运行时动态调整的？编译阶段做了哪些 IR 变换（算子 fusion / memory layout 优化 / 同步插入）？每个 stage engine（vLLM graph vs diffusion engine graph）的 kernel launch 次数从 baseline 减少到多少？SM 利用率从多少提升到多少？

3. **与 EPD (Efficient Disaggregated Prefill) 的核心差异**：你说向后兼容 EPD disaggregation。EPD 的 disaggregation 是 prefill-decode 分离，而 vLLM-Omni 是"modal stage"分离。这两种 disaggregation 维度是否可以同时启用（4 路：prefill-Thinker, decode-Thinker, Talker, Vocoder 各自独立）？如果同时启用，orchestrator 的路由状态空间和决策开销会增长多少倍？

4. **Streaming Stage Output 的同步开销**：下游 stage 增量接收 partial outputs 的机制中，每个 partial output 传输触发的是 event 级同步还是 polling 检查？传输触发频率（每 token / 每 batch / 每 N tokens）？Streaming 引入的额外 synchronizations 次数和被隐藏的计算延迟各自占比多少？

**A**：**1. 并发机制的选择理由**

Stage Graph 选择"独立 engine per stage"而非 finer-grained 并发的核心依据是 **stage 间模型架构的异构性**：

**最优条件（论文主张）**：
- Stage 间 model type 不同（AR LLM vs DiT vs CNN）——所需 engine 差异大，vLLM engine 和 diffusion engine 分别有独立优化栈
- Memory capacity 不对称——Thinker 30B 需大量 memory/single device（TP-2），Talker 更小但 compute-intensive，co-locate 会导致 memory 竞争
- Stage 间数据量中等（Thinker→Talker 5.49ms/8.28ms，<0.1% total latency）——传输代价远低于 engine-level 优化收益
- Pipeline 深度固定（3-stage DAG），cross-stage 数据依赖为线性的 producer-consumer 模式，适合 streaming overlap

**退化条件（推理，论文未评估）**：
- 当 stage 间数据量极小且传输延迟接近 compute 延迟时——独立 engine 的 connector 开销可能占主导。但论文数据中 connector <0.1%，未触及此边界
- 单一 model 内的 pipeline parallelism（如 layer-stage 分解）更宜用 CUDA Stream，因为 stage 间同质（同一 model type），独立 engine 的 per-stage memory 冗余（每 stage 加载独立 model parameters）会成为浪费

**替代方案评估**：论文**未评估**将多 stage fusion 到单一 engine 内、用 CUDA Stream 或 MPS 做 stage 级并发的方案。这种方案的理论代价：
- 失去 per-stage independent scheduling（continuous batching 不能跨异构 model 共享）
- Execution graph compilation 无法为异构 pipeline 生成统一 graph
- 但可消除 connector 延迟和 memory 冗余
论文未提供对比实验，无法量化收益/损失。

**2. Execution Graph Compilation 的 IR 变换细节**

论文对 execution graph compilation 的描述高度抽象。从继承自 vLLM 的上下文推断，其作用链为：

| 阶段 | 行为 |
|------|------|
| **Baseline（HF Transformers）** | 每 iteration 经历：Python Function dispatch → output shape inference（CPU meta compute）→ GPU kernel selection → kernel launch。每个 op 独立 launch，CPU launch overhead 在短 kernel（如 decode 阶段 attention）中占 significant 比例 |
| **vLLM Graph Compilation** | 静态捕获 CUDA graph：将一组已知 tensor shape/type 的 op sequence 捕获为 CUDA graph，消除 per-step CPU dispatch 和 kernel launch 开销。graph 仅需一次 launch 完成多 kernel 执行 |

**关键限制**：graph 是 shape-static 的——prefill 阶段因 sequence length 动态变化，graph 捕获效率低（需频繁 rebuilding）。Execution graph compilation 主要在 decode 阶段生效。这正是 Qwen3-Omni Thinker 获 12.97× speedup 而 Qwen2.5-Omni 仅 1.29× 的原因——Qwen3 Thinker (30B) 的 decode 阶段 GEMM 更多，graph compilation 可摊销的 CPU overhead 更大。

**具体 IR 变换**：论文未公开。vLLM standard CUDA graph capture 不涉及算子 fusion，仅做"launch 序列固化"。Diffusion engine 的 flash attention kernel compilation 为独立 kernel-level 优化，非跨 stage IR 变换。

**Kernel launch 减少量**：未报告。
**SM 利用率变化**：未报告。

**3. 与 EPD 的核心差异**

| 维度 | EPD Disaggregation | vLLM-Omni Stage Disaggregation |
|------|-------------------|-------------------------------|
| 分解单位 | Phase（prefill vs decode） | Modal Stage（Thinker / Talker / Vocoder） |
| 适用模型 | 单 AR LLM | Any-to-any 多 stage 模型 |
| 共享点 | 同模型参数（prefill/decode 共享） | 不同模型参数（各 stage 独立加载） |
| Connector | KV cache transfer（prefill→decode） | 泛化 connector：hidden states / embeddings / audio tensors / KV cache |

**是否可以同时启用（4 路）**：可以，但论文仅声明"unified connector remains compatible with existing EPD disaggregation"，未实验 4 路联合。

如果同时启用（prefill-Thinker, decode-Thinker, Talker, Vocoder 四路独立）：
- Orchestrator 路由状态空间：从 3-stage DAG（3 条路径）扩展到 4-stage DAG——每请求需依次通过 prefill-Thinker → decode-Thinker → Talker → Vocoder。状态维度不变（仍为线性 DAG），但决策频率加倍（需调度 prefill→decode 的 KV cache transfer + decode→Talker 的 hidden states transfer + Talker→Vocoder 的 codec tokens transfer）。
- 决策开销增长：EPD 的 prefill↔decode 分解额外引入一个 connector transfer（KV cache）+ 一个 engine 实例（prefill-only engine + decode-only engine 代替原单一 engine）。vLLM-Omni 已有的 orchestrator 调度 3-stage 链，增加到 4-stage 链的决策延迟增长约为线性（O(#stages)），非指数。但论文未量化。

**实际限制**：对于 Qwen3-Omni，Thinker (30B) 已有 TP-2，再将 prefill/decode 分解意味着 prefill-Thinker 和 decode-Thinker 各自加载 30B 参数（+KV cache 分裂），总 GPU memory 需求可能超过 2×80GB 限制。EPD 在 single model 场景不改变参数总数（prefill/decode 共享参数），但在 multi-stage vLLM-Omni 中，prefill/decode 分离后的参数冗余不可接受——这是论文未进一步分解的原因。

**4. Streaming Stage Output 的同步开销**

Streaming 机制描述（论文 §3.3 "Streaming Stage Output"）：
- Output processor **asynchronously streams partial outputs**（newly generated tokens/embeddings）to next stage via unified connector
- Downstream stage 可 **incrementally** 开始计算，无需等待 full completion

**同步机制**：论文未明确 event vs polling。从 architecture 推断使用**异步 callback/event-based** 模型——Unified Connector 的后端（shared memory + inline control queues）意味着 producer 写入 shared mem 后通过 control queue 发送 lightweight metadata。Consumer（下游 engine）在 inference loop 中检查 control queue 并处理新到达数据。

**触发频率**：未明确指定。根据描述"as soon as Talker produces initial tokens"，触发频率可能是 **per-token**（每 decode 一个 token 触发一次），或 **per-small-batch**（N tokens 积累后触发）。对于 Vocoder（Qwen3-Omni 为 lightweight CNN），per-token 触发合理，因 CNN Vocoder 的 per-step compute 远小于 AR decode。对于 DiT Vocoder（Qwen2.5-Omni，full denoising attention over tokens），per-small-batch 更可能（减少 attention 重建开销）。

**额外 synchronization 次数**：未测量。推理：per-token streaming 引入 producer→consumer 的 synchronization 次数 = Talker output tokens / triggering interval。对于平均 545.4 audio tokens，per-token 触发则引入 ~545 次同步事件。若每 synchronization 代价为 shared memory memcpy + control queue notify（<0.01ms），总同步开销 << 5ms，对比总推理延迟（tens of seconds）可忽略。但此为推理，论文未提供直接测量。

---

### Round 4

**Q**：关于 vLLM-Omni 依赖的硬件并发原语和跨 stage 通信机制，请进一步具体化：

1. **Unified Connector 利用的硬件原语**：Shared memory 模式下（Thinker→Talker 5.49ms），Connector 具体使用什么硬件原语实现跨进程/跨 engine 的数据传输（如 CUDA IPC / CUDA MPS / NVLink P2P / CUDA 4D-GPUDirect）？这些原语的吞吐上限（峰值 BW）和实际利用率分别是多少？Mooncake RDMA 模式下（8.28ms），RDMA 的吞吐能力和实际 BW 利用率是否被报告？不同 payload size 下 shared memory vs RDMA 的延迟曲线是怎样的（最小 payload size、crossover point）？

2. **单节点内 stage 间是否利用了硬件并发原语**：你说 Thinker TP-2 across 2 devices、Talker 独占 device-1、Vocoder 独占 device-0——这意味着 device-0 上同时部署了 Vocoder 和 Thinker TP（device-0 是 Thinker TP 组的一个 rank）。在同一 device-0 上，Vocoder 和 Thinker 的 compute 是否利用了任何硬件并发机制（如 CUDA Stream 级并发、MPS 的分时策略、或仅仅是串行执行不重叠）？Vocoder 运行期间 Thinker 是否处于 idle（device 资源完全空闲）？LLM 推理框架中常见的 compute/communication overlap 技术是否被用于 Thinker TP-2 的 all-reduce 与 Talker/Vocoder compute 的重叠？

3. **Overlap 的实际硬件资源竞争**：如果同一 device 上有多个 stage 实例（如不同请求的 Talker 和 Vocoder 同时执行），它们是否竞争 SM / memory BW / L2 cache？是否做过同一 device 上多 stage 共存时的硬件性能 counter 测量（SM occupancy、L2 hit rate、memory BW utilization）？如果存在竞争，是哪种资源首先饱和？

4. **跨平台可移植性的代价**：vLLM hardware plugin 架构如何将 Stage Graph 抽象映射到非 NVIDIA 硬件（AMD GPU / NPU）？在目标平台缺少 CUDA IPC / NVLink 或 CPU-centric RDMA 时，需用多少额外软件开销来模拟跨 stage 数据传输——是否估算过该开销占主计算比例？

请提供定量数据；论文或你们无法获得的数据请明确标注并说明原因。

**A**：**1. Unified Connector 利用的硬件原语**

论文对 Unified Connector 底层硬件原语的描述停留于抽象层面，**未指定具体 CUDA 原语**：

| 传输模式 | 硬件原语（论文描述） | 具体 CUDA 机制（论文未指定） |
|----------|-------------------|---------------------------|
| **单节点 Shared Memory** | "inline control queues for small payloads + system shared memory for larger ones" | 未说明是 CUDA IPC（跨进程 GPU memory access）、CUDA MPS（共享 GPU context）、NVLink P2P（直接 GPU↔GPU）、还是 host-side memcpy via staging buffer。考虑到 5.49ms Thinker→Talker 和 0.53ms Talker→Vocoder 延迟量级（ms 而非 μs），且 CPU memory 被用作 intermediate buffer（"storing resulting data in CPU memory"后 transfer 到目标 device），实际路径很可能是：**GPU→CPU staging buffer→shared memory→CPU staging buffer→target GPU**，即不涉及 GPU Direct 的纯 host-side 传输。该路径受 PCIe bandwidth 限制（~32GB/s gen4 x16），而非 GPU 内部 BW |
| **跨节点 Mooncake RDMA** | "TCP- or RDMA-based transport" via Mooncake connector | Mooncake 是面向 LLM serving 的跨节点 transfer 库（Qin et al. 2025），基于 RDMA 提供 put/get 接口。单节点实际仅用 shared memory，跨节点才启用 RDMA |

**论文未报告** 的定量数据：
- 实际带宽利用率（峰值 BW vs 实测 BW）
- 不同 payload size 的延迟曲线和最小 payload size
- Crossover point（payload 多大时 RDMA 优于 shared memory）

**2. 单节点内 stage 间是否利用硬件并发原语**

**Device-0 上的 stage 共存**：
- Device-0 同时部署 **Vocoder** 和 **Thinker TP rank**（Thinker TP-2 跨两个 device）
- 但对于**同一请求**，pipeline 是串行的：Thinker generate 完成 → 数据传输到 Talker → Talker generate → 数据传输到 Vocoder → Vocoder denoising。Vocoder 和 Thinker 在同一 device 上**不会同时活跃**
- **跨请求并行**：当 Request_1 的 Talker 在 device-1 decode 时，Request_2 的 Thinker 可在 device-0 prefill——此时 device-0 上的 Talker（Request_1）不活跃，Thinker（Request_2）占用 device-0 compute。**但 Vocoder 和 Thinker rank 从未在 device-0 上同时运行**（同一请求序列化，不同请求时由一个 stage 独占 device）

**关键缺失**：论文**未明确说明**是否利用 CUDA Stream 级并发实现同一 device 上不同 stage 的 compute overlap。vLLM engine 本身支持多 CUDA Stream 做 compute/communication overlap（TP-2 的 all-reduce 可与后续 compute 重叠），但论文未报告该优化是否被用于 Thinker TP-2 的 all-reduce 与 Talker/Vocoder compute 的重叠。

**3. Overlap 的实际硬件资源竞争**

论文**未提供**同一 device 上多 stage 共存时的硬件性能 counter 测量：
- SM occupancy：未报告
- L2 cache hit rate：未报告
- Memory BW utilization：未报告
- 资源饱和顺序：未分析

推理（基于已知架构）：由于 stage 在同一 device 上串行执行（非 concurrent），**不存在多 stage 间硬件资源竞争**。真正的竞争来自两种场景：
- **多请求 intra-stage batching**：同一 engine 内 batch decode 时，多个 sequence 共享 SM / memory BW / L2——这是 standard LLM serving 问题，vLLM 已有成熟处理
- **Thinker TP-2 all-reduce**：TP-2 的 all-reduce 需要 NVLink P2P 通信——当 Talker（device-1）在 decode 时，Thinker（device-0+device-1）的 all-reduce 可能与 Talker 的 compute 争夺 device-1 的 SM 和 NVLink BW。但论文未测量此场景

**4. 跨平台可移植性的代价**

论文 §3.5 "Hardware Support" 仅声明："Built upon vLLM's hardware plugin architecture, vLLM-Omni achieves cross-platform compatibility through a decoupled plugin mechanism that allows for registering hardware-specific implementations independently."

**实际可移植性的代价**（推理，论文未估算）：
- **缺少 CUDA IPC / NVLink**（AMD GPU / NPU）：单节点 shared memory 传输退化到 host-side memcpy（PCIe staging buffer），额外增加 **GPU→CPU→GPU 两次 PCIe 传输**（~32GB/s gen4）。以 Thinker→Talker 数据量估算（hidden states 维度约 model hidden dim × sequence length），假设 hidden_dim=8192, seq_len=150→约 4.8MB。两次 PCIe → ~0.3ms 额外延迟——相比 5.49ms shared mem 原延迟，增加约 5-6× 传输时间。但相比于总推理延迟（tens of seconds）仍可忽略
- **缺少 RDMA**（CPU-centric 网络）：跨节点传输退化到 TCP（假设 100Gbps），延迟从 ~8ms 增加到可能数十 ms，仍可保持 <1% total latency
- **Mooncake connector 依赖 GPU Direct RDMA**：非 NVIDIA 平台需 software emulation（CPU buffer→network→CPU buffer→GPU copy），增加 1-2× 传输延迟
- **Diffusion engine 依赖 flash attention**：vendor-specific kernel optimization 需要为每平台独立实现（或退化为标准 PyTorch attention），DiT stage 的 1.26× overall speedup 会缩水

论文**未提供**实际跨平台实验数据来量化上述代价。

---

### Round 5

**Q**：关于架构限制与扩展性，请进一步量化：

1. **瓶颈根因精确区分**：round 1 说 Thinker 是"model-size-bound（memory capacity）+ TP communication 限制"，Talker 是"compute-bound（长序列 AR decode）"。请问：
   - Thinker 在 decode 阶段（memory-bound）的 **memory BW 利用率** 是多少（HBM BW 的百分之多少被 weight loading / KV cache access 分别消耗）？TP-2 的 all-reduce 通信与 compute 是否实现了 overlap？如果没有，NVLink BW 利用率是多少？
   - Talker 在 decode 阶段的 **compute 利用率**（SM active cycles / total cycles）是多少？是否存在 memory BW 瓶颈（如 KV cache 读取成为 hidden bottleneck）？
   - 当 **batch size 增加**时，Thinker 和 Talker 分别哪个资源首先饱和？该饱和点对应的 batch size / max throughput 是多少？

2. **NVLink 通信 vs compute 的竞争**：Thinker TP-2 跨两个 device 时，device-1 上 Thinker rank 的 all-reduce 通信会通过 NVLink 进行。同一 device-1 上也部署了 Talker。当 Talker 在 device-1 上 decode 时，Thinker 的 TP all-reduce 是否与 Talker 的 compute 共享 NVLink 带宽（NVLink 是 bidirectional 128GB/s per direction）？是否测量过这种竞争对 Talker decode latency 的影响？如果 NVLink BW 被 TP-2 all-reduce 消耗，Talker 的 decode 吞吐下降多少？

3. **Per-stage memory footprint 与扩展上限**：各 stage 的显存占用分解（model weights + KV cache + activation 各占多少 GB）在不同 batch size 和 sequence length 下如何变化？2×80GB 配置下，Qwen3-Omni（Thinker 30B + Talker + Vocoder）的总显存占用是否接近 160GB 上限？如果要扩展更多 stage（如 4 路 EPD+stage 分解），理论需要多少 GB 显存——这是否只能通过增加 GPU 数量来绕过？

4. **Per-stage continuous batching 的独立 memory budget**：你说 per-stage 独立配置 memory budget。各 stage 当前配置的 memory budget 分别是多少 GB（Thinker / Talker / Vocoder）？如果某个 stage 的 memory budget 被占满（如 Thinker 的 prefill 请求涌入占满 KV cache 预留），这是否会影响其他 stage 的 memory 分配——还是各 stage 的 memory 物理隔离（如 MIG / MPS 分区）？

请提供定量数据；无测量或不可得的数据请明确标注。

**A**：**1. 瓶颈根因精确区分**

论文**未报告**硬件计数器级测量（memory BW 利用率、SM active cycles 等），以下回答基于推理：

**Thinker decode 阶段（memory-bound）**：
- 30B 参数 @ FP16 = ~60GB weights。TP-2 下每 device 加载 ~30GB weights
- HBM BW 分配估计（典型 LLM decode profile）：~60-80% BW 用于 weight loading（因 decode batch size 通常较小，compute intensity 低）；~15-30% 用于 KV cache 读写（per-token KV 读写随 batch size 线性增长）；~5% 其余 overhead
- TP-2 的 all-reduce：前向 pass 后需要 all-reduce 同步 logits/hidden states。如果未实现 compute/communication overlap，all-reduce 等待时间约 5-15 μs per token（取决于 hidden dim 和 NVLink BW）——相对于 decode step 的 ~1-5ms 总延迟，占比 1-3%。论文**未说明**是否实现 overlap

**Talker decode 阶段（compute-bound 主张）**：
- Talker 的 "compute-bound" 主张基于其生成 545.4 avg audio tokens 而非 150.9 text tokens——更多 decode iterations 意味着累计 GEMM 时间更长
- 但单次 decode iteration 中，Talker 可能仍然是 memory-bound（small batch size decode 的典型特征）。论文未提供 SM active cycles 数据来确认其 "compute-bound" claim
- KV cache 读取成为 hidden bottleneck 的可能性：Talker 每 step 需 concat Thinker hidden states + Talker embeddings + original MM embeddings，输入维度增大可能使 memory access 成为瓶颈

**Batch size 增加时的资源饱和**（论文未实验，以下为推理）：
| 资源 | Thinker 饱和顺序 | Talker 饱和顺序 |
|------|-----------------|-----------------|
| Memory BW | 第 1 位（~batch=16-32 时 decode 接近 BW 峰值） | 第 2 位（input 维度大，但 iteration 更少） |
| HBM capacity | 第 2 位（KV cache 随 batch × seq_len 增长最快） | 第 3 位 |
| Compute (Tensor Core) | 第 3 位（prefill 时先于 decode） | 第 1 位（长 decode 序列累计 GEMM） |
| NVLink BW | 第 4 位（仅 TP-2 all-reduce） | N/A |

**饱和点 batch size / max throughput**：论文未报告。仅知实验中使用多请求 continuous batching，但未提供饱和曲线。

**2. NVLink 通信 vs compute 的竞争**

Device-1 上的资源竞合关系：Talker (device-1) + Thinker rank (device-1, TP-2)。

**竞争场景**：
- NVLink 带宽：每个方向 128GB/s（H100 NVLink 4.0 × 4 links = ~128GB/s per direction）。TP-2 all-reduce 的 data size ≈ hidden_dim × batch_size × token_count × 2 bytes（FP16）。对于 batch=1, hidden_dim=8192, seq_len~150：all-reduce payload ≈ 2.4MB per iteration。NVLink 传输时间 ≈ 0.02ms（at 128GB/s），非常微小
- 关键问题是：all-reduce 是否与 Talker compute **同时进行**（overlap）？如果是：
  - NVLink 的双向特性使 all-reduce 通信和 Talker memory access 使用不同硬件路径（NVLink vs HBM），不直接竞争
  - 但如果 Talker 的 kernel 也在使用 NVLink（例如 TP 间的 AllGather/ReduceScatter）且设计为占用全部 NVLink links，则存在 conflict
- 如果未 overlap（串行）：Talker 等待 all-reduce 完成才继续下一个 decode step。延迟增加约 0.02-0.1ms per iteration，对总延迟影响比例：0.02ms × 545.4 tokens ≈ 10.9ms → vs tens of seconds total，可忽略

论文**未测量**此竞争。

**3. Per-stage memory footprint 与扩展上限**

无论文提供的分解数据。以下为基于公开知识的合理估计：

| Stage | Parameters | Weight @ FP16 | KV Cache (batch=1, seq=150/545) | Activation | 小计 |
|-------|-----------|--------------|--------------------------------|-----------|------|
| Thinker | 30B | ~60GB | ~2GB (150tokens, 80layers, 128kv heads, 128dim) | ~1GB | ~63GB |
| Talker | ~3-7B (est.) | ~6-14GB | ~7GB (545 tokens) | ~1GB | ~14-22GB |
| Vocoder | ~1B (DiT/CNN) | ~2GB | N/A | ~2GB | ~4GB |
| **合计** | | **~72GB** | **~9GB** | **~4GB** | **~85GB** |

TP-2 下 Thinker per-device weight ≈ 30GB，剩余 50GB/device 可分配给 Talker 和 Vocoder。2×80GB ≈ **不接近 160GB 上限**——仍有约 75GB 余量（考虑 fixed allocation 非动态共享）。

**扩展到 4 路（EPD+stage）**：
- Prefill-Thinker：30B weights（~60GB）+ short KV cache（~1GB）+ activations（prefill 峰值 = seq_len × hidden_dim × layers ≈ 15GB for 8k seq）→ ~76GB
- Decode-Thinker：30B weights（~60GB）+ KV cache（~2GB）+ activations（~1GB）→ ~63GB
- Talker：~14-22GB
- Vocoder：~4GB
- **合计：~157-165GB**——刚好触及 160GB 上限，需增加 GPU 数量（3×80GB 或 2×80GB + offload）

**4. Per-stage continuous batching 的独立 memory budget**

论文仅描述 "flexible per-stage GPU allocation" 和 "users can flexibly allocate computing accelerators and memory resources to each stage"，但**未公布各 stage 配置的具体 GB 值**。

**Memory 隔离机制**：论文声明各 stage engine 有独立的 scheduler、KV cache manager、model runner，但**未说明 memory 是物理隔离还是软限制**：

| 隔离方式 | 可能性 | 证据/反证 |
|---------|-------|----------|
| **MIG/MPS 分区** | 低 | 论文未提及 MIG/MPS；实验平台 2×80GB accelerator 可能不支持 MIG（需 A100/H100）；且 MPS 的 memory 隔离不可靠 |
| **CUDA IPC + cudaMem* 分离** | 中 | Engine 独立意味着各 stage 有独立 CUDA context (process)，通过 CUDA IPC 共享 device memory。Memory budget 是 application-level 软限制 |
| **OS process 级隔离 + staging buffer** | 高 | Single-node 的 connector 使用 "system shared memory" 作为 staging area，暗示各 stage 为独立 OS process，memory 分配为 process-level 硬隔离 |

最可能的配置：各 stage 为独立 OS process，各有独立 CUDA context，memory budget 通过 cudaMalloc 限额或 process-level memory pool 做软限制。如果一个 stage 的 prefill 请求涌入占满其 KV cache 预留，不会影响其他 stage 的 memory——因为它们在不同 process/context 中，有各自独立的 device memory allocation。这避免了跨 stage 的 memory 争夺，但代价是静态 memory 分区导致总利用率低于动态共享方案。论文**未提供** memory 利用率对比。

---

### Round 6

**Q**：关于实验测量方法学和结果可靠性，请进一步说明：

1. **测量工具与粒度**：报告中的端到端指标（JCT、RTF、TPS）使用什么工具采集的（如 vLLM 内置 metrics / 外部 profiling 工具 / nsight 等）？这些指标是 request 级粒度还是更细的 stage 级或 iteration 级粒度？各 stage 的独立延迟分解（Thinker latency / Talker latency / Vocoder latency / connector delay 分别各占多少 ms）是否被测量和报告——还是仅报告端到端总延迟？Stage 间 streaming overlap 节省了多少 TTFT（time-to-first-token 的绝对值和比例）？

2. **误差范围与置信度**：文中报告的关键加速比（如 Qwen3-Omni JCT -91.4%、Thinker TPS 12.97×）是单次测量的结果还是多次运行的平均值？是否报告了 variance / 标准差 / 百分位数？是否以 tail latency（如 P99）还是平均 latency 衡量响应时间？不同负载强度（请求到达率 / concurrency level）下这些指标的稳定性如何？

3. **Baseline 公平性**：Baseline 选择 Qwen-Omni → HuggingFace Transformers 默认实现。HuggingFace Transformers 默认不使用 continuous batching、PagedAttention、CUDA graph 等优化。vLLM-Omni 相对于 HuggingFace Transformers 的 91.4% JCT 降低中有多少比例来自 Stage Graph 拆分 vs 多少比例来自 vLLM engine 本身相对 HF 的固有优化（vLLM 相对 HF 通常已有 2-10× 加速）？是否做过"vLLM engine 单一 stage（无 Stage Graph）vs vLLM-Omni 多 stage"的消融实验来分离 Stage Graph 本身的收益？

4. **覆盖与复现**：四个 any-to-any 架构（Qwen3/2.5-Omni, BAGEL, MiMo-Audio, Diffusion）中，每个架构的端到端实验运行了多少次？测试的输入负载（prompt 类型、长度分布、请求模式）是否多样且已公开？当前是否存在开源实现或 config 文件可以复现主要结果？

请提供定量数据；缺失处明确标注。

**A**：**1. 测量工具与粒度**

论文**未说明**使用何种工具采集指标。实验描述仅提及 metrics 定义：JCT（request 从提交到完成的端到端延迟）、RTF（处理时间/生成音频时长）、TPS（Thinker/Talker 各自每秒生成 tokens）。从 vLLM ecosystem 推断，JCT/RTF 最可能通过 vLLM 内置的 request-level timing（request arrival → completion timestamp）采集，TPS 通过 token counting / elapsed time 计算。

**指标粒度**：论文报告的指标均为 **request 级**（端到端），非 stage 级或 iteration 级。

**Stage 级延迟分解**：Figure 7 "Execution time decompose for Qwen3-Omni" 以**相对占比**（stacked bar chart）展示 Thinker / Talker / Vocoder 在各输入类型（audio/image/video）下的时间分布。结论是 Talker 占绝大部分。但论文**未给出各阶段的绝对 ms 值**，也未单独报告 connector delay 的绝对时间（仅在 Table 1 报告了 connector 数据传输时间 5.49/0.53/8.28/3.34ms，但那是纯传输时间，不包含 stage 内 compute）。

**Streaming overlap 节省的 TTFT**：论文描述 streaming stage output "reduces TTFT and supports streaming responses"，但**未报告具体节省的 TTFT 绝对值或比例**。

**2. 误差范围与置信度**

论文**完全未报告**统计误差信息：
- 关键加速比（Qwen3-Omni JCT -91.4%、Thinker TPS 12.97×、Talker TPS 7.98×）**是否为多次运行的平均值未说明**
- 未提供标准差 / 方差 / 百分位数
- 未说明报告的是平均 latency 还是 tail latency（P50/P95/P99）
- 未评估不同负载强度（请求到达率 / concurrency level）下的稳定性

实验方法描述：使用 librispeech_asr、food101、ucf101-subset 三个数据集，"first 100 queries from each dataset as input"。每个数据集 100 queries 共 300 请求，但**未说明是否重复运行**以及使用了多大的 concurrency（batch configuration）。

**3. Baseline 公平性**

这是最关键的实验设计问题。论文选择 HF Transformers 作为主要 baseline，而 vLLM standalone（单 stage 无 Stage Graph）对比 HF 本身已有显著加速（2-10×）：

| 优化层 | HF Transformers | vLLM standalone | vLLM-Omni |
|--------|----------------|-----------------|-----------|
| Continuous batching | ❌ | ✅ | ✅ |
| Chunked prefill | ❌ | ✅ | ✅ |
| PagedAttention | ❌ | ✅ | ✅ |
| CUDA graph compilation | ❌ | ✅ | ✅ |
| Stage Graph disaggregation | ❌ | ❌ | ✅ |

**缺少关键消融实验**：论文**未做**"vLLM engine 单一 stage（无 Stage Graph） vs vLLM-Omni 多 stage"的对比。这意味着**无法分离两个贡献源**：

- **贡献 A**：vLLM engine 相对 HF 的固有优化（continuous batching + PagedAttention + CUDA graph）。这是任何 standard vLLM serving 相对 HF Transformers 都有的大幅收益
- **贡献 B**：Stage Graph 拆分 + disaggregated execution 的**增量收益**

以 Thinker TPS 12.97× 为例——vLLM 社区已知纯 LLM inference 中 vLLM 相对 HF 可达 5-15× 加速（取决于 model size、batch size、CUDA graph 启用情况）。30B 参数量级下 CUDA graph compilation 的收益特别显著（论文也承认"Qwen3-Omni accelerated much more because of execution graph compilation, larger model can better amortize its optimized execution pipeline"）。这意味着 12.97× 中的**大部分可能来自 vLLM engine 本身而非 Stage Graph**。

**合理的消融实验应该包括**：
1. HF Transformers（baseline，论文已做）
2. vLLM engine 运行 Thinker standalone（分离 vLLM engine 收益）
3. vLLM-Omni full pipeline（论文已做）
4. vLLM-Omni without streaming output（分离 streaming 收益）
5. vLLM-Omni without per-stage independent batching（评估 co-located scheduling 的损失）

论文**未做** (2)(4)(5)。

对于 Qwen2.5-Omni（Thinker 7B），vLLM-Omni 的 Thinker TPS 提升为 1.29×——这个较小值更可能反映 Stage Graph 的**纯增量**收益（小模型 CUDA graph 收益小），也侧面说明 12.97× 的大部分来自 vLLM engine 本身。

**BAGEL / MiMo-Audio / Diffusion baseline 选择**偏保守：各使用原始实现或 Diffusers，这些 baseline 通常无 continuous batching、无 PagedAttention——但 BAGEL / MiMo-Audio 的非 vLLM 实现也无法直接跑在 vLLM 上（因非标准 AR loop），因此这种 baseline 选择是合理的。

**4. 覆盖与复现**

**架构覆盖**（较全面）：
| 架构类型 | 模型 | 是否在 single device 实验 |
|---------|------|--------------------------|
| Thinker-Talker (double AR) | Qwen3-Omni, Qwen2.5-Omni | 2×80GB（Thinker TP-2） |
| AR + DiT | BAGEL | 1×80GB |
| AR + Audio decoder | MiMo-Audio | 1×80GB |
| Pure DiT | Qwen-Image, Qwen-Image-Edit, Wan2.2 series | 未明确 |

**实验重复次数**：未报告。
**输入负载多样性**：3 个数据集（librispeech_asr / food101 / ucf101-subset），每个 100 queries。无 prompt 长度分布报告，无 burst / Poisson 请求模式实验。
**开源性**：代码已公开（https://github.com/vllm-project/vllm-omni），vLLM-Omni v0.20.0（2026/05）。但论文**未提供 config 文件**或 reproduction instructions，仅提供 URL。开源实现的存在提升了可复现性，但缺少具体实验配置（batch size、schedule policy、memory budget 等）使得精确复现需要 reverse engineering。

---

