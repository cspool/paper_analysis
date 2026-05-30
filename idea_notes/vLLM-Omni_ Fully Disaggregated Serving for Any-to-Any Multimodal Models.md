## vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models

- baseline方法是什么？
  Baseline是HuggingFace Transformers的默认实现（Qwen-Omni系列），以及其他模型原始实现（BAGEL、MiMo-Audio）或Diffusers库（Diffusion模型）。核心问题：Existing LLM serving frameworks（vLLM、SGLang）使用**step-centric abstraction**——将模型推理抽象为单个forward function，框架内部封装iteration logic和KV cache management。这种抽象专为text-only LLM的单AR decoding设计。

  Baseline全栈执行例子（以Qwen3-Omni Thinker-Talker-Vocoder pipeline，2×80GB accelerators为例）：
  - 算法层：Thinker-Talker双AR LLM + Vocoder架构。Thinker (30B LLM)生成text tokens + hidden states → Talker (smaller LLM)生成audio codec tokens → Vocoder (DiT/CNN)生成audio waveforms。
  - 系统框架层：HuggingFace Transformers默认实现。开发者需手动实现pipeline：(1) 对Thinker实现custom generate()函数 → (2) 提取output hidden states → (3) 编排cross-stage transfer（将hidden states转为Talker input embeddings）→ (4) Talker custom generate() → (5) 提取codec tokens → (6) Vocoder generate waveforms。每个stage独立运行自己的generate loop。
  - 编译框架层：无execution graph compilation优化，kernel launch overhead大。
  - Kernel调度层：无continuous batching——每个request的Thinker和Talker分别独立运行generate loop。无chunked prefill——长prompt全量prefill。无paged attention——KV cache management低效。Thinker/Talker/Vocoder串行执行，后一stage必须等前一stage完全完成。
  - 硬件架构层：2× 80GB accelerator。所有stage co-located在单进程中，无法fine-grained resource allocation。Thinker(30B)占用大量memory和compute，Talker和Vocoder资源受限。计算资源利用不充分——Thinker compute-bound阶段GPU空闲时Talker无法开始。

  Baseline两大核心缺陷：
  1. **Step-centric抽象无法表示multi-stage pipeline**：vLLM/SGLang的step-centric interface只能封装单个AR decoding的forward+iteration loop。Qwen3-Omni的Thinker→Talker→Vocoder三层pipeline及其cross-stage dependency（Talker每步decoding需连接Thinker hidden states）超出了单forward function的表达能力。开发者被迫在serving framework外手动实现inter-stage transfer，丢失所有framework-level优化。
  2. **Monolithic execution导致资源分配低效**：所有stage co-located在同一进程中，computing resources无法按stage需求灵活分配。Thinker(30B)需大量memory，Talker compute-intensive需更多parallelism——两阶段需求相互冲突。且计算资源在stage间无法concurrent利用——pipeline串行，下一stage等待上一stage完全完成。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出vLLM-Omni，核心创新为**Stage Abstraction + Disaggregated Stage Execution**，将any-to-any pipeline从monolithic execution解耦为independently served stages：

  **(1) Stage Graph Abstraction**——解决"step-centric abstraction无法表示multi-stage pipeline"缺陷：
  将any-to-any model定义为stage graph G=(V,E)。V中每个node为独立stage，需实现：(a) forward function——step-centric batched forward（兼容vLLM的iteration loop优化）；(b) preprocess function——每iteration调用，modify stage input以incorporate upstream data。E中每条edge定义stage-transfer function——控制query states和intermediate data如何在stage间转换。Stage graph原生表达多stage pipeline（如Thinker→Talker→Vocoder），users仅实现per-stage forward/preprocess/transfer逻辑，无需手动管理batching/scheduling。

  **(2) Disaggregated Stage Execution Backend**——解决"monolithic execution资源分配低效"缺陷：
  每个stage由独立execution engine serving：
  - AR stages → vLLM engine（继承continuous batching、chunked prefill、PagedAttention KV cache management、execution graph compilation）
  - DiT stages → 专用diffusion engine（flash attention、SAGE attention、TurboAttention、TeaCache/cache-dit caching、RingAttention/Ulysses parallelism）
  - Per-stage independent scheduling、KV cache management、model execution
  - Flexible per-stage GPU allocation：Thinker(30B)分配更多memory + TP-2 across both accelerators；Talker分配更少memory但更高parallelism；Vocoder分配独立device
  - Stage间通过Unified Connector数据传输：shared memory（单节点, 5.49ms Thinker→Talker）或Mooncake RDMA（跨节点, 8.28ms）

  **(3) Streaming Stage Output**——解决pipeline串行等待问题：
  下游stage在上游未完全完成时即开始incremental处理。Talker产出initial tokens时Vocoder即可开始denoising，减少TTFT和enabling streaming responses。

  vLLM-Omni方法全栈执行例子（以Qwen3-Omni同一请求为例）：
  - 算法层：同Thinker-Talker-Vocoder架构，但执行方式完全不同。
  - 系统框架层：vLLM-Omni stage graph编程——users定义thinker_forward/talker_forward/dit_decode（forward functions）、mm_encode/process_input（preprocess functions）、Thinker2Talker/Talker2Vocoder（transfer functions）。Orchestrator管理请求routing through stage graph。每个engine独立配置parallelism policy和memory budget。
  - 编译框架层：vLLM engine复用execution graph compilation（Qwen3-Omni Thinker获12.97× speedup的主要来源之一）。Diffusion engine集成flash attention kernel compilation。
  - Kernel调度层：
    Thinker stage (TP-2, device-0+device-1):
      continuous batching + chunked prefill + paged attention
      → Thinker generate text tokens + hidden states (150.9 avg output tokens)
    Talker stage (device-1):
      每decode iteration: preprocess concatenate Thinker hidden states + Talker embeddings
      → Talker generate audio codec tokens (545.4 avg output tokens)
      → streaming to Vocoder
    Vocoder stage (device-0):
      incremental DiT denoising with TeaCache caching
      → final audio waveforms
    Stage间并行：当Request_1的Talker在device-1 decode时，Request_2的Thinker在device-0 prefill——compute和memory跨stage、跨请求自然重叠。
  - 硬件架构层：2× 80GB accelerator。Thinker TP-2 spread across both devices，Talker和Vocoder各占一个device。Per-stage independent GPU allocation最大化memory和compute利用率。Unified connector overhead negligible（<0.1% of total latency）。

  设计思路核心：
  vLLM-Omni的本质是将any-to-any model serving从"application-level manual orchestration"下沉为"framework-level automatic disaggregation"。关键洞察是**complex multimodal architectures can be decomposed into modular stages, each of which is just a standard AR or DiT component**——这些component本身可以被existing serving engines高效执行，难点在于stage graph的表达和执行。通过提供stage abstraction前端（表达任意topology）+ disaggregated execution后端（各stage独立优化），vLLM-Omni使得"支持任意any-to-any model"成为可能——从Thinker-Talker双AR（Qwen-Omni）到AR+DiT（BAGEL, GLM-Image）到纯DiT（Qwen-Image, Wan2.2），所有architecture均可用同一stage graph范式表达和执行。实验结果中Qwen3-Omni JCT降低91.4%的根本原因在于：baseline缺失的性能优化（continuous batching、chunked prefill、execution graph compilation）在vLLM-Omni中通过解耦stages自然获得，且30B Thinker相比7B Thinker(Qwen2.5-Omni)能更充分摊销优化pipeline，实现超线性加速比（91.4% vs 61.6%）。
