# Idea 库

## Using Span Queries to Optimize Cache and Attention Locality

- baseline方法是什么？
  Baseline 是标准 Chat Completion API 下的 vLLM serving——所有 inference 请求以线性 token 序列（message list）形式表达，KV cache 复用仅限于 exact prefix match（prefix caching）。具体地：
  
  **Baseline 全栈执行例子**（以 vLLM 服务 RAG 场景为例，2 个 documents，每个 N tokens）：
  
  - **算法pipeline**：Client 拼接 prompt = system + doc1 + doc2 + query → 发送线性 token 序列给 vLLM → 模型按标准 causal attention 对所有 token 计算 QK^T softmax V。Documents 之间无结构信息——模型"看到"的是一长串 token 序列，attention 在 doc1 和 doc2 的 token 之间计算跨 document 的 attention。
  
  - **系统框架/Serving**：vLLM 接收单次 Chat Completion 请求 → tokenize → scheduler 分配 KV cache pages → 每个 decode step FlashInfer dense decode kernel 从 HBM 读取完整 KV cache → 输出。**复用仅限 prefix caching**：若后续请求的 prompt 与之前请求共享 exact token-level prefix，vLLM 的 block hash matching 可复用对应 KV cache pages。对于 RAG 场景，不同 query 共享相同的 documents 但 query 在末尾 → prefix 从开头到 query 前都是共享的 → 可 prefix-cache documents 部分。但对于多轮 RAG（同一 query 对多组 documents），prefix 仅在 query 出现前共享 → documents 变化意味着 prefix 的尾端变化 → cache miss。
  
  - **编译框架**：论文未明确说明。
  
  - **kernel调度**：FlashInfer dense decode kernel，标准 paged attention。每个 decode step 读取 O(N·d) 字节 KV cache，memory-bound（受 HBM 带宽限制）。
  
  - **硬件架构**：NVIDIA GPU（型号未明确），利用 Tensor Core 做 FP16/BF16 MMA。KV cache 存储在 HBM 中。
  
  Baseline 核心痛点：
  1. **P1 — KV cache 复用受限于 linear prefix 假设**：vLLM 的 prefix caching 仅对 exact match 的 token 序列前缀生效。对于 RAG（多 documents 互相 independent）、Judge-Generator（多个 generator 输出 independent）、multi-turn agentic loops（不同 path 的 history 部分重叠）等场景中大量 logically independent 且 reusable 的 token segments 无法被复用，因为它们在 token 序列中的**位置不同**。位置绑定（position binding）使 KV cache 复用退化。
  2. **P2 — API 结构信息丢失**：Chat Completion API 将所有内容 flatten 为线性 token 序列。Documents 之间的独立性、agentic loops 的分支结构、Judge-Generator 的树状依赖——这些结构信息在 API 层面丢失，server 端无感知，无法利用。
  3. **P3 — Attention 跨无关 token 浪费**：线性 token 序列中，所有 token attend to all prior tokens（causal attention）。Document 1 的 token attend to document 2 的 token——这是不必要且有害的（导致 lost-in-the-middle 问题，attention mass 被稀释到大量无关 hay token 中）。
  4. **P4 — Inference-time scaling 的成本爆炸**：Deep reasoning 和 agentic workloads 生成大量 structurally similar 的 token 序列（如多路生成+判断、iterative refinement）。Chat API 将每个 variant 视为独立请求，KV cache 完全不共享，计算和内存开销线性增长。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 **Span Query** 将 inference API 从线性 token 序列 generalize 为带 commutativity constraints 的 expression tree，使 server 端可进行 SQL-style query plan optimization。核心设计：(1) Span（contiguous sequence of pages in a paged attention server）作为 cache 和 attention 的基本单位；(2) Commutativity hypothesis——若 span 间无依赖，其 KV cache pages 可任意排列和复用；(3) 三层优化栈——high-level tree rewriting + query tokenization + low-level block optimization；(4) 仅 492 行 vLLM 修改实现全部能力。
  
  **应对 P1（KV cache 复用受限于 linear prefix）—— Span + Commutativity**：
  - Span 定义为"contiguous sequence of pages"，引入**额外间接层**：GPU memory 中 physical pages → span 的 virtual ordering。这使得同一 physical page 可在不同 span 中以不同 virtual position 被引用。Commutativity hypothesis 保证：若 span A 和 span B commute（AB=BA），则 A 的 pages 和 B 的 pages 可被后续请求以任意交错方式复用。
  - 具体机制：RoPE on read（非 write）→ KV cache page 不带 position encoding（position 在读取时动态注入）；Hash chaining selective disable（span 边界暂停 hash accumulation）→ prefix caching 在 span 粒度工作而非 token 粒度。
  - 效果：RAG 场景中，即使 query 不同，documents 的 KV cache pages 可以 100% 复用（因为 documents 之间 commute 且与 query 无位置绑定）。
  
  **应对 P2（API 结构信息丢失）—— Expression tree + ++ operator**：
  - Span Query API 引入 `++`（commutative join）操作符和 expression tree 结构。Client 显式表达：哪些部分 commute、哪些有顺序依赖。Server 端的 parser/planner 基于结构信息进行优化——如同 SQL query planner 利用 join commutativity 重排执行计划。
  - RAG 场景的表达：`R(query, ++(doc1, doc2, ..., docN))`——"query 与每个 document 配对，documents 之间相互独立"。
  - Judge-Generator 场景的表达：`C(system, ++(G(prompt1), G(prompt2)), J(outputs))`——"两个 generator 独立运行，judge 消费两者输出"。
  
  **应对 P3（Attention 跨无关 token 浪费）—— Sparser attention via span structure**：
  - Span query 的结构信息使 attention 可限制在 span 内（token 仅 attend to same-span tokens）。RAG 场景：document tokens 不 attend to other documents' tokens → attention mass 不被稀释 → lost-in-the-middle 问题缓解。
  - 实证：attention-optimized span query on **2B model** 的 needle-in-haystack accuracy **超过 stock 8B model**（因为 stock 8B 仍受 lost-in-the-middle 影响，而 span query 2B 的 attention 聚焦在 needle 所在 document 内）。
  - Cache miss 场景下 span query 仍比 stock vLLM 更快——因为 attention 更稀疏（tokens within a document attend only within that document）。
  
  **应对 P4（Inference-time scaling 成本爆炸）—— KV cache sharing + CIDRA**：
  - 多路生成（fan-out N）→ N 个 span 的 KV cache 可复用父节点（system message + prompt）的 pages。CIDRA 算法处理并发 block repositioning（多个请求将同一 block 移到不同 virtual position）——duplicate block on demand（仅出度 >1 时复制），其余 in-place remap。
  - Fan-out 24 时 TTFT 加速 12-13×；RAG 场景 TTFT 加速 10-20×。Bulk execution（1024 queries）在 2Wiki 上 1.31× speedup，whole corpus 1.59×。
  
  **Span Query 全栈执行例子**（对比 baseline，RAG 场景：1 query + 2 documents，第 2 次请求复用 documents）：
  
  - **算法pipeline**：
    Baseline：Client 拼接 prompt = system + doc1 + doc2 + query → 线性 token 序列 → model 对所有 token full causal attention。
    Span Query：Client 发送 `R(query, ++(doc1, doc2))` → server parse expression tree → tokenize（pad to block boundary + special tokens for span boundaries）→ 前向 pass 中 attention 限制在 span 内（doc1 内 token 仅 attend doc1 token + system + query；不 attend doc2 token）→ 输出。
    第 2 次请求（不同 query，相同 docs）：Span Query 直接复用 doc1 和 doc2 的 physical KV cache pages（RoPE on read 确保 position 正确），仅需新计算 query 部分的 KV cache。Baseline 需从 doc1 开始重新计算（因为 query 不同 → token 序列不同 → prefix 在 query 位置 mismatch → cache miss）。
  
  - **系统框架/Serving**：vLLM + Span Query extension（492 行修改）。Span-aware scheduler 按 expression tree 结构调度 compute。Block pool 中 physical pages 通过 span table（physical→virtual mapping）支持多对多 mapping。
  
  - **编译框架**：论文未明确说明。无编译框架修改。
  
  - **kernel调度**：CIDRA 算法执行 block repositioning（并发请求复用 blocks 时的 remap/duplicate）。标准 FlashInfer dense decode kernel（无修改）——attention sparsity 通过 span 结构隐式实现（tokenization 阶段已确保跨 span 的 attention 不必要），而非修改 attention kernel。
  
  - **硬件架构**：论文未明确说明 GPU 型号。无硬件修改。利用现有 GPU HBM 和 Tensor Core。核心洞察：软件层（API + scheduler + KV cache manager）的简单变更（~500 行）即可获得数量级收益，而非依赖硬件或 kernel 创新。
  
  **设计对应关系总结**：
  | Baseline 缺陷 | Span Query 设计 | 效果 |
  |---|---|---|
  | KV cache 复用限于 exact prefix match | Span + commutativity：pages 可任意顺序复用；RoPE on read；hash chaining selective disable | RAG documents 跨不同 queries 100% KV cache 复用 |
  | API 丢失结构信息（documents independence, tree structure） | Expression tree + `++` operator：client 显式表达 commutativity 和依赖 | Server 可做 SQL-style query plan optimization |
  | Attention 跨无关 token 浪费（lost-in-the-middle） | Span-bounded attention：token 仅 attend same-span tokens | 2B model + span query > 8B stock model accuracy on needle-in-haystack |
  | Inference-time scaling 成本线性增长（fan-out N → N× cost） | KV cache sharing + CIDRA（concurrent block repositioning） | Fan-out 24: 12-13× TTFT speedup；RAG: 10-20× TTFT speedup |
  | vLLM 260K 行代码，大规模修改风险高 | 仅 492 行修改（7 个文件），核心是 RoPE on read + hash chaining 修改 | 低风险、易维护、易上游合并 |
  
  **创新本质**：Span Query 发现了 LLM inference 中"位置"（position）和"顺序"（order）的过度耦合——Chat Completion API 将 token 序列的顺序性硬编码为 KV cache 的物理布局，但许多应用场景中部分 token segments 实际上 order-independent（commutative）。通过将 RoPE 从 write-time 移到 read-time 和选择性暂停 prefix hash chaining，Span Query 仅需 ~500 行代码就解耦了这两个概念，使 KV cache 从"缓存 prefix"升级为"缓存任意 commutative sub-trees"。

## Sieve

- baseline方法是什么？
  Baseline 是现有的 PIM-enabled LLM serving 系统，具体分为三类静态 expert placement 策略：
  
  - **NoExp [22, 32]**：仅将 attention 操作 offload 到 PIM，所有 FFN expert 计算在 GPU 上执行。这是当前最主流的 PIM 加速策略。
  - **AllExp [21, 43]**：将所有 expert 计算（含 FFN）也 offload 到 PIM 执行。
  - **PIMoE [51]**：使用静态 threshold 决定 expert placement——experts 首先都 assign 到 PIM，然后将 token count ≥ threshold 的最 popular expert 从最繁忙的 PIM channel 迁移到 GPU，直到 GPU execution time > PIM execution time。
  
  所有 baseline 共享相同的非 MoE 优化：attention offload to PIM、KV cache 分布到 PIM channels、DRAM/PIM 命令分离路径。

  **Baseline（NoExp/PIMoE）全栈执行例子**（以 Qwen3.5-397B-A17B 在 8×B200 GPU + HBM-PIM 上 decode 一个 batch B=64 为例，512 experts，top-10 gating）：
  
  - **算法pipeline**：每个 token 通过 gating network 选出 top-10 experts → 每个 expert 执行 FFN（输入 hidden_state ∈ R^d，权重 W₁ ∈ R^{d×d_ff}, W₂ ∈ R^{d_ff×d}）→ 加权聚合 expert 输出。NoExp 将全部 10 个 expert FFN 在 GPU 上用 grouped GEMM 执行；PIMoE 根据静态 threshold 将部分 expert 留在 PIM、其余放到 GPU。
  
  - **系统框架/Serving**：静态 placement 策略，在模型加载时即确定哪些操作在 PIM 执行。NoExp 在 FFN 阶段 PIM 空闲（Figure 6b）；AllExp 在 FFN 阶段 GPU 空闲（Figure 6c）；PIMoE 使用 mini-batch interleaving 策略（N/2 split）重叠 GPU 和 PIM 计算，但忽略 attention 在 PIM 上的时间开销和 inter-GPU 通信开销（假设 global interconnect 任意 GPU 可直接访问任意 PIM device）。

  - **编译框架**：论文未明确说明。PIM 操作通过 GPU kernel 直接发出 PIM 命令（PIM_GWRITE、PIM_GEMV），不涉及编译框架修改。

  - **kernel调度**：GPU 侧使用 grouped GEMM kernel 批量执行多个 expert FFN；PIM 侧将每个 token 的 expert 计算转换为 GEMV（dot product via adder tree），通过 bank-level parallelism 并行执行。PIMoE 使用 expert parallelism 跨 PIM channels（不同 channel 执行不同 expert），但 bimodal distribution 导致某些 channel 负载极重、其他 channel 空闲（Figure 10）。

  - **硬件架构**：NVIDIA B200 GPU + Samsung HBM-PIM stack（每个 GPU 附带多个 HBM-PIM die，PIM die 仅通过 attached GPU 访问）。HBM-PIM 内部：processing unit 置于 DRAM bank 旁，利用内部高带宽（~8 TB/s per stack，远高于外部 HBM-GPU 接口带宽）做 GEMV。无硬件修改。

  Baseline 核心痛点：
  1. **P1 — 静态 placement 不适应 bimodal expert distribution**：现代 MoE 模型（如 GPT-OSS、Qwen3-Next）中，token-to-expert 分布极度不均衡——batch size 64 时 44.2% 的 experts 仅收到 1 个 token（GEMV），89.3% 收到 ≤4 个 token（skinny GEMM）。静态规则（如 PIMoE 的 threshold）无法感知每个 batch 的实际分布，导致 popular experts（高算术强度，适合 GPU）被错误地放在 PIM 上执行，或 unpopular experts（低算术强度，适合 PIM）被留在 GPU 上消耗 memory bandwidth。
  2. **P2 — 忽略 attention 在 PIM 上的时间开销**：PIMoE 在 partitioning 决策时仅比较 GPU vs PIM 的 expert 执行时间，忽略 attention 操作已经占用 PIM 资源。当 sequence 长度增长或 batch size 增大时，PIM 上的 attention 时间显著增加，使 PIM 成为瓶颈，expert placement 看似平衡实则无效。
  3. **P3 — 忽略 inter-GPU 通信开销（假设 global interconnect）**：Prior work 假设任意 GPU/NPU 可直接访问任意 PIM device（global interconnect），忽略 multi-GPU MoE serving 中 expert parallelism 导致的 all-to-all token dispatch/combine 通信开销。在真实系统中，每个 HBM-PIM stack 仅通过其 attached GPU 访问，remote expert 需要经过 NVLink 跨 GPU 传输 token。PIMoE 的 placement 决策不考虑该开销，将过多 expert 分配到 PIM，使得 PIM execution time 超过通信 time，通信成为瓶颈时 PIM 无法加速。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Sieve 提出两个核心组件——**Sieve Scheduler**（运行时动态调度器）和 **Sieve System**（运行时框架），在 multi-GPU + HBM-PIM 系统上实现自适应 expert placement。

  **应对 P1（静态 placement）—— Sieve Scheduler 动态贪心分区**：
  - Sieve Scheduler 在每次迭代中，根据运行时 token-to-expert 分布动态决定每个 expert 在 GPU 还是 PIM 执行。目标函数：
    S* = argmin_{S⊆E} max(T_Comm, T_GPU(G), T_PIM(S))
  - 贪心算法：按 token count 降序排序所有激活的 experts → 从全部 assign 到 PIM 开始 → 依次将 token count 最高的 expert 移到 GPU → 直到 max(T_Comm, T_GPU, T_PIM) 不再下降。
  - PIM 侧的 execution time 使用运行时 cost table（exponential moving average of observed GEMV times），而非 roofline 估计（roofline 可能高估 1.8–4.2×，因未捕获 DRAM timing overhead）。
  - Overhead 仅 ~20 μs（B200 GPU），不影响 critical path。
  - 效果：低算术强度的 GEMV experts（1 token）走 PIM，高算术强度的 GEMM experts（多 token）走 GPU，自适应 batch 内 expert distribution 变化。

  **应对 P2（忽略 PIM 上 attention 开销）—— 目标函数联合建模 PIM attention**：
  - Sieve 的目标函数中 T_PIM(S) 包含两部分：attention operations（固定开销，取决于 sequence length 和 batch size）+ unpopular expert GEMV 计算（取决于 S 的大小和 token count）。PIMoE 仅比较 expert execution time，Sieve 将 attention 开销显式纳入 T_PIM(S)。当 batch size 增大、attention 时间增长时，Sieve 会自动将更多 expert 推到 GPU 以避免 PIM 过载。

  **应对 P3（忽略 inter-GPU 通信）—— T_Comm 建模与 PIM tensor parallelism**：
  - Sieve 在目标函数中显式包含 T_Comm（inter-GPU dispatch/combine 通信开销 = data volume / NVLink BW），与 T_GPU 和 T_PIM 同时作为瓶颈考量。当 T_Comm 成为瓶颈时，Sieve 会倾向于将更多 expert 放在 GPU 以减少 PIM 执行时间，避免 PIM 时间超过通信时间导致的整体吞吐下降。
  - PIM 侧采用 tensor parallelism 而非 expert parallelism：每个 expert 的参数均匀 shard 到所有 PIM channels，每次 GEMV 在所有 channels 上并行执行，保证无论 expert distribution 多么不平衡，PIM channels 利用率始终均匀。PIMoE 的 expert parallelism 在 bimodal distribution 下会导致某些 channel 空闲、某些过载。
  - 不需要 global interconnect 假设：每个 PIM die 仅通过 attached GPU 访问，符合真实多 GPU 系统拓扑（GPU 间 NVLink 互联，HBM die 不直接跨 GPU 连接）。

  **Sieve 全栈执行例子**（同样 Qwen3.5-397B-A17B, 8×B200 GPU + HBM-PIM, B=64, 512 experts, top-10 gating）：
  
  - **算法pipeline**：Router 计算 token-to-expert 映射 → Sieve Scheduler 对 512 个 activated experts 按 token count 降序排序 → 贪心分区：memory-bound GEMV experts（≤2 tokens，约占 50%+）→ PIM；compute-bound GEMM experts（≥3 tokens，约占 <50%）→ GPU。PIM 侧每个 expert 的 FFN 分解为串行 GEMV（每个 token 一个 GEMV）；GPU 侧使用 grouped GEMM 批量执行多个 expert。
  
  - **系统框架/Serving**：Sieve System 构建 MoE layer 的 DAG：
    ¹ Attention output → ² Router → ³ AllGather (global token-expert mapping) → ⁴ Metadata preprocessing → ⁵ Dispatch (token routing across GPUs) ‖ ⁵ Sieve Scheduler (per-GPU partitioning) → ⁶ GPU 路径：HBM-PIM→GPU load params / PIM 路径：GPU→HBM-PIM send tokens + PIM commands → ⁷ GPU grouped GEMM ‖ ⁷ PIM GEMV → ⁸ PIM→GPU readback results → ⁹ Combine + Aggregate (weighted sum per token)。
    关键 overlap：(a) ⁵ Dispatch 和 ⁵ Sieve 并行；(b) shared experts weight loading 在 ⁴ 之后就启动（不等待 Dispatch）；(c) GPU compute (⁷ GPU) 和 PIM compute (⁷ HBM-PIM) 并行执行；(d) PIM compute 和 HBM→GPU data transfer (⁶ 下一层) 可重叠。
  
  - **编译框架**：论文未明确说明。无编译框架修改。
  
  - **kernel调度**：GPU kernel 动态发出 PIM 命令（PIM_GWRITE broadcast token, PIM_GEMV dot product），命令参数（地址、尺寸）由 Sieve Scheduler 输出在运行时计算。PIM tensor parallelism：每个 expert 参数 shard 到所有 32 pseudo-channels，每个 GEMV 在所有 channels 上并行，保证 PIM 利用率不受 expert distribution 影响。GPU grouped GEMM：variable group sizes 由各 expert 的 token count 决定。
  
  - **硬件架构**：与 baseline 相同——NVIDIA B200 + Samsung HBM-PIM，无硬件修改（"SIEVE requires no hardware modifications: no new PIM commands and no changes to the existing PIM architecture or command interface"）。

  **量化效果**：Qwen3.5 (8 GPU) 1.26× throughput & interactivity vs PIMoE @B=256；GPT-OSS (4 GPU) 1.11×–1.17× throughput vs PIMoE @B≥32，interactivity 最高 1.25×；Qwen3 (1 GPU) 1.6× vs PIMoE。Colocated PD 场景下（batch 含 prefill+decode），Qwen3 B=16 时 2.4× vs NoExp，B=32 时 2.3×。Sieve 是唯一在高 batch 下同时满足 throughput 和 interactivity SLA 的方案——NoExp 呈 L 型曲线（throughput 过早饱和），AllExp 呈水平线（throughput B≥32 后无法增长），PIMoE 和 Sieve 接近理想 inverted-L 曲线，但 Sieve 在高峰值 throughput 上超出 PIMoE 26%。

## SageBwd

- baseline方法是什么？
  Baseline 是 Full-Precision Attention（FPA），即标准的 scaled dot-product attention（FlashAttention2 实现），所有 7 个矩阵乘法均在 FP16/BF16 精度下执行，无量化、无 smoothing。
  
  **Baseline（FPA）全栈执行例子**（以 325M Llama 在 RTX 4090/B200 上预训练一个 attention 头为例，序列长度 N=4096, head dim d=128）：
  - **算法pipeline**：Q,K,V ∈ R^{N×d} 输入 → 标准 S = QKᵀ/√d（FP16 MatMul）→ P = softmax(S)（FP16 softmax）→ O = PV（FP16 MatMul）→ 反向 dP = dOVᵀ（FP16 MatMul）→ dS = P∘(dP−δ)（FP16 element-wise）→ dQ = dSK（FP16 MatMul）→ dK = dSᵀQ（FP16 MatMul）→ dV = PᵀdO（FP16 MatMul）。所有 7 个 MatMul 均为 FP16，无精度损失，计算量大（O(N²d) per MatMul）。
  - **系统框架/Serving**：论文未明确说明（研究聚焦于训练阶段的 attention 计算）。
  - **编译框架**：使用 OpenAI Triton 实现 FlashAttention2 kernel，无编译框架修改。
  - **kernel调度**：FlashAttention2 tiled kernel（Triton 实现），通过 online softmax 避免物化 S 和 P 到全局内存，使用 FP16 Tensor Core 执行所有 MatMul。
  - **硬件架构**：NVIDIA RTX 4090 / B200 GPU，利用 FP16 Tensor Core 和 on-chip SRAM tiling。
  
  Baseline 核心痛点（FPA 的训练计算瓶颈）：
  1. **P1 — 训练时注意力计算开销大**：训练需要前向+反向共 7 个 MatMul（QKᵀ, PV, dOVᵀ, PᵀdO, dSK, dSᵀQ，加上中间 softmax），每个 O(N²d)。FP16 计算量大，成为大规模预训练的瓶颈。
  2. **P2 — 低比特量化难以应用于训练**：现有低比特注意力（SageAttention 系列、FlashAttention3 FP8）仅支持推理，不支持训练。训练的反向 pass 对量化误差极其敏感，尤其是 softmax 梯度 dS = P∘(dP−δ)，其 RMS 仅约 1×10⁻⁷（序列长度 N=4096 时），INT8 的固定绝对量化噪声在此量级下产生巨大相对误差。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SageBwd 提出一种可训练的 INT8 注意力机制，将 7 个 MatMul 中的 6 个量化为 INT8（利用 GPU INT8 Tensor Core 加速），仅保留 dP = dOVᵀ 为 FP16。核心设计决策和分析如下：
  
  **应对 P1（训练计算开销大）**：
  - 前向 pass：QKᵀ 使用 per-block INT8（配合 K-smoothing），PV 使用混合 per-token/per-block INT8。两个前向 MatMul 从 FP16 降为 INT8，计算吞吐提升（INT8 Tensor Core 的理论吞吐高于 FP16）。
  - 反向 pass：dP = dOVᵀ 保持 FP16（不量化），其余 dV = PᵀdO、dS 计算、dQ = dSK、dK = dSᵀQ 使用 per-block INT8。反向共 4 个 MatMul 量化。
  - 在 RTX 4090 上达到最高 1.67× 端到端吞吐加速（vs FlashAttention2）。
  
  **应对 P2（训练时量化误差敏感）**：
  通过四个维度的分析和设计来克服训练中的量化不稳定性：
  - **(i) dP 保持 FP16（最关键的设计）**：dP = dOVᵀ 是唯一不量化的 MatMul。因为 dP 的量化误差会通过 dS = P∘(dP−δ) 传播，而 dS 的 RMS 仅 ~10⁻⁷，即使很小的绝对量化误差也会在 dS 上产生极大的相对误差，并进一步通过 dQ = dSK、dK = dSᵀQ 被 Q 和 K 的范数放大。保留 dP 为 FP16 切断了这条件最危险的误差传播链。
  - **(ii) QK-norm 控制激活动态范围**：在 Q 和 K 上应用 RMSNorm，压缩其动态范围，减小 INT8 均匀量化的步长（scale = max(|X|)/127），从而降低绝对量化误差。实验证明在大 TPS（2.1M tokens/step）下 QK-norm 是防止 loss 爆炸的必要条件。
  - **(iii) K-smoothing 消除通道异常值**：对 K 做列均值减法（K^{sm} = K - mean_row(K)），无需修改反向 pass（因为 dS 每行和为 0，dQ = dSK = dSK^{sm} 数学上等价）。Q-smoothing 在训练中并未带来持续收益——因为它需要额外的 bias 分支 dK_bias = (dSᵀ1)μ_Qᵀ 来校正梯度，这引入了新的量化噪声路径。因此 SageBwd 默认仅使用 K-smoothing。
  - **(iv) 通过降低 TPS 利用梯度噪声掩蔽量化误差**：在大 TPS（2.1M，batch_size=512）下，梯度噪声低，系统性的 INT8 量化偏差（尤其是沿 dS 路径）被优化器感知，导致收敛到次优解。在小 TPS（260K，batch_size=64）下，随机梯度噪声天然较大，量化误差被梯度噪声淹没，SageBwd 可匹配 FPA 预训练性能（loss: 2.561 vs 2.563）。
  
  **SageBwd 全栈执行例子**（对比 baseline 的变化）：
  - **算法pipeline**：Q,K 输入 → QK-norm（RMSNorm）→ K-smoothing（K=K-mean_row(K)）→ per-block INT8 量化 Q,K,V → 前向 Q̂K̂ᵀ（INT8 MatMul, dequant×s_Q×s_K）→ online softmax → P̂V̂（per-token quant P + per-block INT8 MatMul + dequant）→ O。反向 → dP=dOVᵀ（FP16 MatMul，**不量化**）→ dS=P∘(dP−δ) → INT8 quant dS → dQ=d̂SK̂（INT8 MatMul）→ dK=d̂SᵀQ̂（INT8 MatMul）→ dV=P̂ᵀd̂O（INT8 MatMul）。关键变化：(a) 6/7 MatMul 降为 INT8；(b) dP 单独保留 FP16 阻断误差放大；(c) 加入 QK-norm + K-smoothing 控制量化误差。
  - **系统框架/Serving**：论文未明确说明。
  - **编译框架**：使用 OpenAI Triton 编写自定义 kernel，无编译器修改。Triton 自动生成 CUDA PTX，调用 GPU INT8 Tensor Core 指令（如 `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`）。
  - **kernel调度**：在 FlashAttention tiled 框架内嵌入 per-block INT8 量化操作。Triton kernel 在 RTX 4090 上实现 1.67× 加速 vs FlashAttention2。
  - **硬件架构**：NVIDIA RTX 4090 / B200 GPU，利用 INT8 Tensor Core（SM 内 mma 指令）和 FP16 Tensor Core（dP 路径）。

## SLA2

- baseline方法是什么？
  Baseline 是 SLA（Sparse-Linear Attention），一种启发式路由的稀疏-线性混合注意力方法。SLA 对压缩后的 Q、K 计算 P_c = softmax(pool(Q)pool(K)ᵀ/√d)，取每行 top-k_h% 为稀疏 attention（M=1），其余走线性 attention（M=0），输出 O = O_s + Proj(O_l)，其中 Proj(O_l) 是一个可训练的线性投影，用于补偿稀疏分支的缩放失配和线性分支的近似误差。
  
  **Baseline（SLA）全栈执行例子**（以 Wan2.1-T2V-1.3B 在 RTX 5090 上 decode 阶段一个 attention 头为例，序列长度 N，head dim d）：
  - **算法pipeline**：输入 Q,K,V → pool 压缩得 Q̄,K̄ → 计算 P_c → 启发式 Top-k_h% 选 M=1 → 稀疏 attention O_s = softmax(QKᵀ/√d ⊙ M)V → 线性 attention O_l = φ(Q)(φ(K)ᵀ(1-M)V) / φ(Q)(φ(K)ᵀ(1-M)1) → O = O_s + Proj(O_l)。Proj ∈ R^{d×d} 是额外可训练参数，需要同时补偿 (α-1)⊙O_s（稀疏分支归一化导致的缩放失配）和 P_2V（线性分支的近似目标），学习负担重。
  - **系统框架/Serving**：论文未明确说明。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：在 FlashAttention 之上实现 block-wise 稀疏+线性 attention kernel。稀疏分支仅对 M=1 的 block 计算 matmul，线性分支用 Q(KᵀV) 重排降低复杂度。Top-k 路由在 GPU 上基于压缩 P_c 的每行排序完成。
  - **硬件架构**：RTX 5090，利用 Tensor Core 加速 matmul，稀疏路径仍为 memory-bound（需读取选中的 K,V block）。

  Baseline 核心痛点（SLA 的两大限制）：
  1. **L1 — 稀疏-线性分解失配（Scaling Mismatch）**：SLA 的设计动机是将 full attention 分解为 P = P₁ + P₂，其中 P₁ = P⊙M（稀疏分支处理）、P₂ = P⊙(1-M)（线性分支处理）。但实际稀疏 attention 产生的是行归一化后的 P_s = P₁/α（α = P₁·1 为每行概率和），因此 P₁V = α⊙O_s ≠ O_s。SLA 通过引入 Proj(O_l) 来吸收这个失配：Proj(O_l) ≈ P₂V + (α-1)⊙O_s，但线性分支被迫同时补偿两个误差源（自身的近似误差 + 稀疏分支的缩放误差），使学习更加困难，且与原始分解动机不一致。
  2. **L2 — 启发式路由非最优（Heuristic Top-k Routing）**：SLA 基于压缩注意力权重 P_c = softmax(pool(Q)pool(K)ᵀ) 的最大值进行 Top-k 选择，假设大的注意力权重应走稀疏分支。但这种启发式选择并非最优：例如，将某些权重从 P₁ 移到 P₂ 可能不增加 P₂ 的秩（因为 P₂ 已包含相似模式的权重），但仍能提高 P₁ 的稀疏度。缺乏一个基于明确优化目标的路由机制。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SLA2 提出三项核心改进：(I) 可学习路由器 R 替代启发式 Top-k；(II) 直接 α-组合公式 O = α⊙O_s + (1-α)⊙O_l 替代 O_s + Proj(O_l)；(III) QAT 低比特 attention 进一步加速。
  
  **应对 L1（分解失配）**：SLA2 使用 O = α⊙O_s + (1-α)⊙O_l 直接匹配分解 P ≈ α⊙P_s + (1-α)⊙P_l。α 可学习且每行独立，消除 P_s 的行归一化缩放失配——α⊙P_s 直接对应 P₁，(1-α)⊙P_l 对应 P₂，且 α⊙P_s+(1-α)⊙P_l 自动行归一化（每行和为 1）。不再需要 Proj(O_l)，线性分支仅专注于近似 P₂V，学习目标更纯粹。
  
  **应对 L2（启发式路由）**：SLA2 引入可学习路由器 R(Q,K)：M_c = Top-k(k%, proj_q(Q̄)proj_k(K̄)ᵀ)。其中 proj_q, proj_k ∈ R^{d×d} 是可训练的线性投影，使路由在投影空间中学会选择最优的稀疏/线性分派。训练时通过最小化 MSE(FullAttn(Q,K,V), SLA2(Q,K,V)) 直接优化路由质量，替代 SLA 的启发式 Top-k(P_c) 选择。
  
  **低比特加速**：在稀疏 attention + 可学习路由基础之上，引入 QAT 使 forward pass 使用 INT8/FP8 量化 attention（Q,K,P,V 全部量化），backward 保持 FP16。训练中模型参数适应量化误差，推理时获得额外约 1.3× kernel 加速。

  **SLA2 全栈执行例子**（对比 baseline 的变化）：
  - **算法pipeline**：Q,K,V 输入 → pool 压缩 → proj_q(Q̄)proj_k(K̄)ᵀ 得到可学习路由分数 → Top-k 路由 → 稀疏分支 O_s（含 QAT 量化）→ 线性分支 O_l（φ=softmax kernel，Q(KᵀV) 重排）→ O = α⊙O_s + (1-α)⊙O_l。关键变化：(a) 路由器加了可学习投影；(b) 组合方式从 O_s+Proj(O_l) 变为 α⊙O_s+(1-α)⊙O_l；(c) 训练目标增加 Stage 1 的 MSE 路由初始化。
  - **系统框架/Serving**：论文未明确说明。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：在 FlashAttention block-wise 框架上，kernel 新增：(a) 路由分支判断后选择稀疏量化路径（quant→matmul→dequant→softmax→quant×quant→dequant）或线性累加路径；(b) 反向 kernel 手动推导梯度（Algorithm 3），预计算 dH_i,dZ_i 减少冗余计算；(c) K 的列均值减法平滑（SageAttention 技术）。
  - **硬件架构**：RTX 5090，Tensor Core 支持 INT8/FP8 低比特 matmul，QAT 使模型适应量化误差，kernel 在 97% sparsity 下达到 18.7× FlashAttn2 加速比。

## Inference Time Context Sparsity

- baseline方法是什么？
  Baseline 是标准 dense scaled dot-product attention (SDPA) 在 LLM decode 阶段的全量 KV cache 访问。具体地，每生成一个 token，每个 query head 对所有历史 token 的 KV cache 计算 attention score，然后加权求和得到 output embedding。
  
  **Baseline 全栈执行例子**（以 Qwen2.5-72B 在 H100 上 decode 一个 token 为例）：
  - **算法pipeline**：标准 Transformer decoder，每层 multi-head attention + FFN，attention 使用完整 softmax(QK^T/√d)V，无稀疏化。所有历史 token 均参与 attention。
  - **系统框架/Serving**：vLLM 接收请求 → paged KV cache 管理 → 每个 decode step 从 HBM 读取完整 KV cache（O(N·d) 字节，N 为 context 长度）送入 FlashInfer dense decode kernel → 输出 logits → 采样下一个 token。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：FlashInfer 的 dense decode kernel 在 H100 上执行分块 GQA attention。以 128K context、Hq=32、Hkv=8、D=128 为例，每步需从 HBM 读取 128K × 8 × 128 × 2 bytes (FP16) ≈ 256 MB KV cache。Attention 计算受限于 HBM 带宽（H100 ~3.35 TB/s），单 batch 耗时 ~0.19ms。
  - **硬件架构**：H100 HBM3 80GB，利用 Tensor Core 做 FP16 矩阵乘，但由于 KV cache 读取的带宽瓶颈（memory-bound），Tensor Core 利用率低。

  Baseline 核心痛点：
  1. **理论瓶颈**：Theorem 1 证明当 hidden dimension d << context 长度 N 时，dense attention output V^T·a 无法在 attention simplex 上是单射——即多个不同的 attention 分布映射到相同的 hidden 表示。真正"密集"的 attention 在长上下文下理论上就不可能存在，因为 d 维度瓶颈不可避免地压缩了 O(N) 的 attention 信息。
  2. **内存带宽瓶颈**：decode 阶段每步从 HBM 读取 O(N·d) 字节的 KV cache，随 context 线性增长，成为不可持续的带宽瓶颈。
  3. **资源浪费**：大量 KV cache token 对当前 query 贡献极小的 attention weight，却被全量读取和计算。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法是在 decode 阶段对 attention 沿 context 维度施加极端但原则性的稀疏化：每个 query 仅选择 KV cache 中 top-k 个最相关的 token（k << N）参与 attention 计算。稀疏选择机制包括：(1) Oracle top-k 精确选择（作为上界）；(2) vAttention 基于采样的随机索引选择（解决小模型 top-k 退化问题）；(3) Double Sparsity 量化索引器（8 通道 16-bit，在线近似 top-k）。稀疏 mask 由三部分组成：sink tokens（前 128 个固定保留）+ local window（当前 token 前 128 个）+ top-k 稀疏选择。
  
  **论文方法全栈执行例子**（以 Qwen3.5-27B hybrid 在 H100 上 50× 稀疏 decode 为例）：
  - **算法pipeline**：hybrid 架构模型（线性注意力层 + 标准 SDPA 层交替）。在 SDPA 层，decode 时仅选取 KV cache 中 ~2% 的 token 参与 attention（50× 稀疏）。Qwen3.5-27B 在 RULER-HARD 和 AIME2025 上 50× 稀疏质量与 dense 持平；甚至仅用 16-32 个 top-k token（约 250× 稀疏）仍保持强劲性能。小标准模型（如 Qwen2.5-1.5B）的 top-k 退化可通过 vAttention 随机索引恢复。
  - **系统框架/Serving**：vLLM + sparse-attention-hub 后端。Serving 层新增稀疏索引器组件：每个 decode step 先用 Double Sparsity 索引器（8×16-bit 量化通道）从完整 KV cache 中快速选出 top-k token 位置，再调用稀疏 decode kernel。SWE-Bench Django 上，50× 稀疏 agent 在严格可比较子集上 resolution rate 与 dense 仅差 ~2pp（75.9% vs 77.6%），但平均 turn 数从 67 降到 55，总 token 从 1.34M 降到 1.08M。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：稀疏 decode kernel（基于 FlashInfer paged KV-cache 后端）。关键设计：(1) 根据稀疏索引 gather 选中的 KV cache 行——由于 token 级别不规则稀疏无块结构约束，利用 d=128 向量维度的连续内存访问摊销随机 gather 开销；(2) 仅在选中的 k 个 token 上计算分块 attention；(3) GQA 下 Hq/Hkv=4 时仍有效，因为 KV cache 向量维度提供足够的连续内存。不含索引器的 kernel-only 结果：50× 稀疏下 5.5–10.5× 加速 vs FlashInfer；含 Double Sparsity 索引器后 MHA 下 2× 稀疏即 break-even、GQA 下 10× 稀疏 break-even。
  - **硬件架构**：H100 HBM3 80GB。稀疏化后每步 HBM 读取从 O(N·d) 降至 O(k·d)，直接缓解 memory bandwidth 瓶颈（H100 ~3.35 TB/s）。论文论证当前硬件已足以从这种不规则稀疏模式中获得显著加速，不需要块结构稀疏（block sparsity）作为前提条件。

  **解决 Baseline 缺陷的映射**：
  1. 理论瓶颈 → 极端稀疏不是近似而是更优目标：Theorem 1 表明 d << N 下 dense attention 本质上无法保留所有 attention 分布差异，因此稀疏化并非损失信息，而是显式选择了真正重要的 token 关系。
  2. 内存带宽瓶颈 → 稀疏化将每步 HBM 读取从 O(N·d) 降至 O(k·d)，直接打破 context 长度与延迟的线性耦合。
  3. 资源浪费 → 稀疏选择确保仅相关 token 被读取和计算，实证表明模型（尤其是 hybrid 架构和大模型）天然容忍甚至受益于稀疏化。

## CAGE: Curvature-Aware Gradient Estimation for Quantization-Aware Training

- baseline方法是什么？
  **Baseline: QuEST (ICML 2025) + Straight-Through Estimator (STE)**
  
  QuEST 是此前 SOTA 的 QAT 方法，其核心仍是 STE：在量化前向传播时使用离散量化值 `Q(x)`，在反向传播时直接将量化器的梯度视为恒等映射（`∂Q(x)/∂x ≈ I`），即梯度直接穿过量化算子传递到 master weight。标准 QAT 优化目标为 `min_x f(Q(x))`，仅最小化量化后权重的损失函数值，但**忽略了量化操作本身引入的失真**——即 `x_t - Q(x_t)`（量化误差）在优化过程中不被显式控制和最小化。
  
  **Baseline 全栈执行例子（STE QAT with QuEST）**：
  - **算法pipeline**：前向 `f(Q(x))` + 反向 `∇f(Q(x))`（STE 直通），无量化误差反馈 → AdamW 更新 master weight `x_{t+1} = x_t - α * AdamW(∇f(Q(x_t)))`
  - **Serving调度**：论文未明确说明
  - **编译框架**：论文未明确说明
  - **kernel调度**：论文未明确说明（量化kernel由PyTorch原生支持，QuEST量化器在CUDA kernel层面执行量化/反量化）
  - **硬件架构**：论文未明确说明（运行在 NVIDIA H100 GPU 上，不涉及硬件修改）
  - **芯片设计**：论文未明确说明

  **Baseline 痛点**：STE 忽略了 `Q(x)` 在 x 处的局部曲率信息。当量化误差 `x - Q(x)` 较大时，损失函数在量化前后的梯度方向可能显著偏离，导致优化陷入 sub-optimal 的 Pareto 前沿——即损失函数最小化和量化误差最小化两个目标之间存在冲突，STE 无法协调。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **CAGE: 曲率感知梯度估计**
  
  CAGE 将 QAT 重新框架化为**多目标优化问题**：同时最小化 `f(Q(x))`（任务损失）和 `‖x - Q(x)‖`（量化误差）。在 Pareto 最优状态下满足：`∇f(x*) + λ(x* - Q(x*)) = 0`。CAGE 通过向 STE 梯度添加量化误差修正项 `λ_t * (x_t - Q(x_t))`，使优化器感知量化操作在参数空间局部引起的损失曲率变化。
  
  **CAGE 全栈执行例子（对比 baseline）**：
  - **算法pipeline**：
    1. 前向传播：同 baseline，`f(Q(x))` 量化前向
    2. 反向传播：`g_t = ∇̃f(x_t) + λ_t * (x_t - Q(x_t))`——在 STE 梯度基础上加曲率修正
    3. 沉默期机制：前 s 比例步数 λ_t=0（纯 STE），之后线性 ramp-up，避免训练初期量化误差大时误导优化方向
    4. Coupled 模式：修正加到梯度上再由优化器处理；Decoupled 模式：修正直接作用于 AdamW 的 Δ_t，避免 AdamW 二阶矩归一化削弱修正信号
    5. 理论保证：`E[‖∇_{λP} f(Q(x̂_T))‖²] = O(1/√T)`，收敛到 Pareto 最优

  - **Serving调度**：论文未明确说明
  - **编译框架**：论文未明确说明
  - **kernel调度**：论文未明确说明（CAGE 在算法层面增加一个逐元素减法+标量乘法操作，在 PyTorch 中实现，不涉及自定义 CUDA kernel 修改。开销极小：100M 模型上 per-iteration 101ms vs QuEST 101ms，几乎无额外开销）
  - **硬件架构**：论文未明确说明
  - **芯片设计**：论文未明确说明

  **设计对应关系总结**：
  | Baseline 缺陷 | CAGE 设计 | 效果 |
  |---|---|---|
  | STE 忽略量化误差，损失函数和量化目标冲突 | 多目标优化框架：同时优化 f(Q(x)) 和 ‖x-Q(x)‖ | Pareto 最优收敛保证 |
  | 梯度不感知量化引起的局部曲率变化 | λ_t*(x_t-Q(x_t)) 曲率修正项 | W3A3 loss < QuEST W4A4 loss |
  | 训练早期量化误差大，修正可能误导 | Silence period (s=0.8) 机制 | 先学好表示再施加量化约束 |
  | AdamW 二阶矩可能削弱修正信号 | Decoupled CAGE：修正直接加在 Δ_t | 更强的量化约束效果 |
  | 与特定优化器/量化器绑定 | Optimizer-agnostic + Quantizer-agnostic | 可配合任意优化器和量化器使用 |

---

## FlashAttention-4

- baseline方法是什么？
  **Baseline: FlashAttention-3 (FA3) on Hopper H100 + cuDNN/Triton on Blackwell B200**

  FA3 是此前针对 Hopper H100 的最优 attention kernel，其设计围绕 H100 的硬件特性：
  - **Warp specialization**：CTA 内 warps 分为 producer（TMA 数据搬运）和 consumer（MMA + softmax），通过异步执行重叠数据搬运和计算。
  - **Ping-pong 调度**：两个 Q tile 交替计算 softmax 和 MMA，将 softmax 延迟隐藏在 MMA 执行期间。
  - **FP8 支持**：利用 Hopper 的 FP8 tensor core。
  - **寄存器 accumulator**：Hopper MMA 输出写入寄存器（register file），需 interleaved 4 threads/row 的 warp 分配模式。

  **Baseline 全栈执行例子（FA3 on H100 → 直接移植到 B200）**：
  - **算法pipeline**：tiling + online softmax + recomputation（与 FA1/FA2 相同的算法框架），forward 2 MMA（QK^T, PV），backward 5 MMA + recompute S/P
  - **Serving调度**：论文未明确说明（FA3/FA4 均为算子级 kernel，不涉及 serving 框架调度）
  - **编译框架**：FA3 使用 C++ 模板元编程 + CUDA C++（CUTLASS 3.x），forward kernel 编译 55s，backward 45s；需预编译数百个 kernel 变体
  - **kernel调度**：FA3 on Hopper: MMA 64×128 tile (accumulator in registers) → warp specialized, ping-pong pipeline, async TMA。FA3 无法直接运行在 B200 上（Hopper MMA 指令无前向兼容），直接端口到 Blackwell 会遇到以下瓶颈：
    - 寄存器 accumulator 无法容纳 B200 的 128×128 tile（寄存器压力远超 Hopper）
    - MUFU 指数单元 16 ops/clock/SM vs tensor core 8192 ops/clock/SM → 指数成为新瓶颈
    - Shared memory 带宽 128 bytes/clock/SM vs tensor core 翻倍 → SMEM 流量相对 MMA 成为瓶颈（后向 SMEM 3328 cycles vs MMA 2560 cycles，超出 30%）
    - 无法利用 TMEM、2-CTA MMA、DSMEM 等 Blackwell 新特性
  - **硬件架构**：运行在 NVIDIA GPU 上，不涉及硬件修改
  - **芯片设计**：论文未明确说明

  **Baseline 痛点**：Blackwell B200 的非对称硬件扩展——tensor core 吞吐翻倍（Hopper 的 2×），但 MUFU 指数单元（16 ops/clock/SM）和 SMEM 带宽（128 bytes/clock/SM）保持不变——导致瓶颈从 MMA 计算转移到：(1) softmax 中的指数运算（$T_{\text{exp}} = MN/16$ cycles，在 forward 中与 MMA 同等耗时），(2) backward pass 的 shared memory 流量（超出 MMA 计算时间约 30%）。直接使用 Hopper 的 kernel 设计无法解决这些新瓶颈。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling**

  FA4 通过算法- kernel 协同设计，系统性地应对 Blackwell 的非对称瓶颈：

  **FA4 全栈执行例子（对比 baseline）**：

  - **算法pipeline**：
    1. **软件指数模拟（应对 MUFU 瓶颈）**：将 10-25% 的 softmax 指数计算从 MUFU.EX2 迁移到 FMA 单元，通过 degree-3 多项式 + IEEE 754 位操作实现 $2^x$。BF16 精度下误差与硬件无异（量化误差 3.9×10⁻³ 主导）。等效提升指数吞吐量，使指数不再成为瓶颈。
    2. **条件 softmax rescaling（减少非 MMA 操作）**：仅当 $m_j - m_{j-1} > \tau = 8.0$ 时执行 rescaling（$O = e^{\Delta m} O + e^{S-m} V$），否则跳过。大幅减少逐元素乘加操作，将非 MMA 计算从关键路径中移除。
    3. **前向 ping-pong 增强**：利用 TMEM 解耦 P 传输和 output rescaling，将 rescaling 分离到独立 "correction" warpgroup。128 threads/warpgroup 每线程一整行（消除 FA3 的 inter-warp shuffle）。
    4. **Roofline 指导的 tile size 选择**：对 M=N=d=128 分析确认 MMA+exp 为共同瓶颈 → 用大 tile (M=256) 最大化 MMA 利用率，同时用软件指数模拟和条件 rescaling 消除 exp 瓶颈。

  - **Serving调度**：论文未明确说明。

  - **编译框架**：FA4 全部用 CuTe-DSL（嵌入 Python）编写，无 CUDA C++ 组件。JIT 编译：forward 2.5s vs FA3 C++ 55s（22× 加速），backward 1.4s vs 45s（32× 加速）。CuTe-DSL 与 CUTLASS C++ 同构，保留完整底层控制能力（含 PTX escape hatch），同时避免 C++ 模板元编程的编译时间瓶颈。**FA4 不修改编译框架本身**（使用 CuTe-DSL 而非修改它），但编译速度的提升使得迭代和实验更快。

  - **kernel调度**：
    1. **后向 2-CTA MMA 模式**（应对 SMEM 瓶颈）：M=256 tile 让两个 CTA 各 stage 一半 B 操作数，SMEM 流量从 3328 降至 2688 cycles（仅超出 MMA 5%）。dQ 步骤通过 DSMEM 交换半个 dS tile，重排归约轴使每个 CTA 做 `(M/2, 2N)×(2N, d)` MMA，dQ 原子加次数减半。
    2. **5-MMA 流水线重排**：利用 TMEM 的 4 个 accumulator tile 容量，让 S/P 共享一块 TMEM、dP/dS/dQ 共享另一块，实现前一迭代 dQ/dK MMA 与当前迭代 softmax 的重叠。FA3 受限于寄存器 accumulator 无法实现此调度。
    3. **LPT CTA 调度**（应对 load imbalance）：causal 下 reverse mblock 顺序（最长 tile 最先），varlen 下预处理排序 batches → virtual→actual index mapping。MHA causal +4-8% FLOPS，MQA-8 +7-14%。
    4. **确定性后向（SPT 调度）**：semaphore lock + SPT CTA 顺序（KV descending, Q ascending from diagonal, dQ reduction descending），确定性模式达非确定性 75% 性能。

  - **硬件架构**：论文未明确说明（运行在 B200 GPU 上，利用 TMEM/2-CTA MMA/DSMEM 等 Blackwell 架构特性，但不修改硬件）。

  - **芯片设计**：论文未明确说明。

  **设计对应关系总结**：

  | Baseline 缺陷 | FA4 设计 | 效果 |
  |---|---|---|
  | MUFU 指数吞吐量（16/clock/SM）远低于 tensor core（8192/clock/SM），softmax 成为 forward 瓶颈 | 软件指数模拟（FMA polynomial + IEEE 754 bit ops），10-25% 条目用模拟 | 指数吞吐量等效提升，BF16 精度误差无差异 |
  | 每次 KV tile 迭代都执行 rescaling，大量逐元素乘加浪费非 MMA 单元 | 条件 softmax rescaling（仅当 Δm > 8.0 时触发） | 减少 90%+ 的 rescaling 操作 |
  | Backward SMEM 流量（3328 cycles）超出 MMA 计算（2560 cycles）约 30% | 2-CTA MMA 模式：每个 CTA 只 stage 一半 B 操作数 + DSMEM 跨 CTA 交换 dS | SMEM 降至 2688 cycles（仅超 MMA 5%），dQ atomic adds 减半 |
  | FA3 寄存器 accumulator 无法容纳 Blackwell 128×128 tile，且无法利用 TMEM | TMEM accumulator + 4-tile TMEM 管理（S/P 共享，dP/dS/dQ 共享） | 解耦 MMA 与 softmax 的寄存器争用，实现更深度流水 |
  | Causal/varlen 下 CTA load imbalance（短 tile 的 SM 空闲等长 tile） | LPT 调度（reverse mblock 顺序 + varlen batch 排序） | MHA causal +4-8% FLOPS, MQA-8 +7-14% |
  | 全局 atomic add 的不确定性 + 锁争用导致确定性模式性能差 | SPT CTA 调度 + semaphore lock + 2-CTA 减半 atomic adds | 确定性 backward 达非确定性 75% 性能 |
  | C++ 模板 kernel 编译缓慢（FA3 forward 55s, backward 45s），迭代低效 | CuTe-DSL (Python JIT)，forward 2.5s, backward 1.4s | 编译 20-32× 加速，大幅提升开发效率 |

---

## Flashlight: PyTorch Compiler Extensions to Accelerate Attention Variants

- baseline方法是什么？
  **Baseline: PyTorch `torch.compile` (TorchInductor) + FlexAttention**

  **TorchInductor 默认行为**：PyTorch 2.0 的 `torch.compile` 通过 TorchDynamo 捕获计算图，AOTAutograd 分解算子，TorchInductor 进行算子融合和代码生成。但 TorchInductor 对矩阵乘法（`torch.bmm`/`torch.matmul`）有特殊处理：**直接映射到 CUBLAS (cuBLAS) 库调用，形成融合边界**（fusion boundary），隔离了 GEMM 与周围的 softmax、mask、element-wise 等计算。这导致 attention 计算被拆分为多个独立 kernel：QK^T → HBM → softmax → HBM → PV → HBM。每个 kernel 结束后中间张量需要写回全局内存（HBM），下一个 kernel 再读回，产生大量冗余内存流量。

  **FlexAttention 方案**：通过手写模板（`score_mod` + `mask_mod`）让用户描述 attention 变体的计算模式，FlexAttention 将用户定义的模板映射到预编译的融合 kernel。局限性：
  - 仅支持能在其模板模型中表达的变体（基于 score modification 和 mask 的组合）
  - 不支持 data-dependent 的 attention（如 differential attention 中的 λ 系数、Evoformer 的 gated self-attention）
  - 用户需手动管理 `block_mask` 缓存（`create_block_mask`），block-mask 创建本身有显著开销
  - 模板需要预编译，不支持动态生成的 attention 模式

  **Baseline 全栈执行例子（torch.compile 默认处理 standard attention）**：
  - **算法pipeline**：用户编写标准 PyTorch attention 代码（Q@K^T → scale → softmax → @V），计算图包含 matmul + softmax + matmul
  - **系统框架**：`torch.compile(fn)` → TorchDynamo 捕获 fx.Graph → AOTAutograd 分解 → TorchInductor 算子融合
  - **编译框架**：TorchInductor 将 `torch.bmm` 映射到 CUBLAS，GEMM 成为融合边界。softmax 和 element-wise 可在 GEMM 前后各自融合，但**无法跨 GEMM 融合**。最终生成多个独立 Triton kernel（至少 3 个：QK^T kernel、softmax kernel、PV kernel），中间张量 S 和 P 在 kernel 间通过 HBM 传递
  - **kernel调度**：每个 Triton kernel 独立调度。QK^T 和 PV kernel 使用 Triton matmul tile（利用 Tensor Core MMA），softmax kernel 使用 Triton reduction tile。无跨 kernel 的 tile 级数据复用，S 和 P 的完整 HBM 读写不可避免
  - **硬件架构**：NVIDIA H100/A100 GPU，无硬件修改

  **Baseline 痛点**：
  1. **GEMM 融合边界**：TorchInductor 将 matmul 特殊化为 CUBLAS 调用，使得 attention 中 QK^T 和 PV 两个 matmul 无法与中间的 softmax 融合。这导致中间张量 S 和 P 必须物化到 HBM，引入 O(N²d) 的额外内存流量（FlashAttention 通过手写融合 kernel 解决了此问题，但仅限于 standard attention）。
  2. **缺乏模板之外的通用性**：FlexAttention 的模板方法覆盖有限。对于 differential attention（需要两组 QK pair 的差值）、Evoformer 的 row/column-wise gated self-attention（需要在 softmax 前后插入 gating 操作）、IPA（需要 3D 坐标变换）等复杂变体，模板模型无法表达。
  3. **手写 kernel 的不可持续**：每种新 attention 变体都需要专家手写融合 kernel（如 FlashAttention 团队为每种变体编写专用 CUDA kernel），开发成本高、迭代慢。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Flashlight: 编译器驱动的 attention kernel 自动生成**

  Flashlight 将 attention kernel 优化从**手动编写问题转化为编译器优化问题**，在 TorchInductor 中引入三种编译器 pass，自动从标准 PyTorch 代码生成融合的 FlashAttention 风格 kernel。

  **Flashlight 全栈执行例子（对比 baseline）**：

  - **算法pipeline**：
    1. 用户编写标准 PyTorch attention 代码（无模板、无特殊 API）
    2. 编译时，Flashlight 的编译器 pass 自动识别 attention 模式并生成融合 kernel
    3. 运行时，生成的 Triton kernel 执行 tiled matmul + online softmax + output accumulation，**中间张量 S 和 P 从不离开 SRAM/寄存器**

  - **系统框架**：
    1. `torch.compile(fn, enable_flashlight=True)` 触发 Flashlight
    2. TorchDynamo 捕获 fx.Graph → AOTAutograd 分解 → TorchInductor + Flashlight 扩展
    3. Flashlight 在 IR 层面重写计算图，然后由 TorchInductor 的 Triton 代码生成后端输出融合 kernel

  - **编译框架**（Flashlight 的核心贡献）：
    1. **统一归约 IR（解决 GEMM 融合边界）**：将 `torch.bmm` 从 "特殊 CUBLAS 映射" 降级为 "广义归约操作"，与 `torch.max`、`torch.sum` 等在 IR 中平等对待。这消除了 GEMM 特有的融合边界，使 matmul 能够参与跨 operator 的融合和分块。**代价**：放弃 CUBLAS 专用 GEMM kernel 的极致性能，换取融合自由度。
    
    2. **结构融合 + 维度降级（解决 kernel sketch 不兼容）**：当 matmul producer（并行维度 i×j×k）与 softmax consumer（归约维度 j）的循环嵌套不兼容时，将 producer 的并行维度 j 降级为融合后的归约循环。原理是并行度↔内存局部性的权衡——牺牲部分并行度（j 维度不再并行），换取 tile 内的数据局部性（S 和 P 不离开 on-chip memory）。

    3. **语义融合 + 代数变换（消除中间张器物化）**：利用指数函数同态 $e^{a-b} = e^a/e^b$，在 IR 层面将 stable softmax 的两次归约循环（先全局 max → 再 exp+sum+div）自动转换为 online softmax 的单次融合循环（同时维护 running max `m`、running normalizer `ℓ`、running output `O`）。转换的数学正确性通过环论证明保证。

    4. **Twin Matmul 分块融合（消除连续 matmul 的中间张量）**：对 `(A·B)·D` 模式，选择分块策略使 B 的列维度 tile 在片上被即时消费，不物化完整中间结果。两个 matmul 在同一个 kernel 中以 pipeline 方式执行。对 AlphaFold Evoformer 的 gated attention（`gating(x) * softmax(QK^T) @ V`），此 pass 将 gating、QK^T、softmax、PV 全部融合为一个 kernel。

    5. **逻辑网格维度**：将 tile 的并行度分配与 GPU thread block 数量限制解耦，允许选择任意 tile size 组合（不受 `grid_size ≤ 65535` 限制），提供更大的分块自由度。

  - **kernel调度**：
    - Flashlight 生成的融合 Triton kernel 自动包含：tiled QK^T matmul（Tensor Core MMA）、online softmax（running max + running sum + running exp + output accumulation）、tiled PV matmul（Tensor Core MMA），全部在一个 kernel 内完成
    - 生成的 kernel 使用共享内存（Triton SRAM）缓存 Q/K/V tile，跨 KV 序列长度维度迭代，仅在最终写回 O
    - 对于 Evoformer 的复杂变体，单个融合 kernel 包含 gating op → matmul1 → online softmax → matmul2 的完整 pipeline
    - 与 FlexAttention 的模板 kernel 不同，Flashlight 的 kernel 是**按需即时生成**的——每种 attention 变体都生成最适合其计算图的特定融合 kernel

  - **硬件架构**：论文未明确说明（运行在 NVIDIA H100/A100 GPU 上，利用 GPU 的 Tensor Core 和 SRAM/shared memory 层次结构，不涉及硬件修改）。

  - **芯片设计**：论文未明确说明。

  **设计对应关系总结**：
  | Baseline 缺陷 | Flashlight 设计 | 效果 |
  |---|---|---|
  | TorchInductor GEMM↔CUBLAS 融合边界，matmul 无法与 softmax 融合 | 统一归约 IR：matmul → 广义归约，平等参与融合 | 编译器可跨 matmul 边界融合，自动生成 attention-fused kernel |
  | Stable softmax 需要两轮归约（max 后 exp+sum），中间结果需物化 | 语义融合 + 代数变换：同态 $e^{a-b}=e^a/e^b$，自动转换为 online softmax | S 和 P 不再离开片上内存，内存流量从 O(N²d) 降至 O(Nd) |
  | Producer（matmul）和 consumer（softmax）的 kernel sketch 不兼容 | 结构融合 + 维度降级：并行维度 → 归约维度 | matmul + softmax 可在同一 kernel 中执行，仅输出最终 O |
  | FlexAttention 模板无法表达 data-dependent attention（differential attention、gated self-attention 等） | 编译器从任意 PyTorch 代码生成融合 kernel，无模板限制 | 支持 Evoformer 5× 加速、AlphaFold 端到端 6-9% 延迟改善 |
  | FlexAttention block-mask 创建开销大 | 无需 block-mask，编译器直接在生成的 kernel 中内联 mask 逻辑 | 总执行时间（含开销）Flashlight 优于 FlexAttention |
  | 连续 matmul `(A·B)·D` 的中间张量 `A·B` 需物化 | Twin Matmul 分块融合 + 维度消除 | 中间张量在 tile 级别被即时消费，消除全局内存写入 |
  | 每种新 attention 变体需手写专用 CUDA kernel | 编译器自动生成 Triton kernel，用户只需写标准 PyTorch 代码 | 零额外开发成本支持新 attention 变体 |
  | 手写 kernel 开发者需管理 GPU 线程块数量限制 | 逻辑网格维度解耦物理 grid 限制 | 任意 tile size 组合，更大的分块优化空间 |

---

## FuseFlow

- baseline方法是什么？
  **Baseline: Custard + Stardust (C+S) — 单表达式内融合的稀疏数据流编译器**

  Custard [32] 和 Stardust [31] 是此前仅有的两个面向数据流硬件的通用稀疏张量代数编译器：
  - **Custard**：将单个 Einsum 表达式编译为 SAM（Sparse Abstract Machine）数据流图，支持 Intra-Expression Iteration Fusion (IIF) — 在单个核内通过 co-iteration 融合迭代空间、跳过零值。
  - **Stardust**：将高层稀疏张量代数语言编译到 Spatial 编译器 [39] 的 parallel patterns，再映射到 Capstan [60] 数据流加速器。同样仅支持单表达式内融合。
  - **共同的融合限制**：C+S 仅支持 **IIF（Intra-Expression Iteration Fusion）**，无法跨表达式进行融合。用户必须**手动重写**多个表达式为一个超表达式才能实现跨 kernel 融合 — 当计算图包含多于一个输出张量时无法做到，且重写工作在复杂模型（如 GPT-3 BigBird attention）中不可行。
  - **迭代策略**：C+S 默认生成 **global iteration space** 的数据流图 — 先构建所有输入的 n 维联合坐标迭代子图，再连接计算子图（类似图 11 左侧）。

  **Baseline 全栈执行例子（Custard 编译 GCN 2-layer on RDA）**：
  - **算法pipeline**（GCN 2-layer 前向传播）：
    ```
    Layer 1: H1 = σ(A × X × W1)      // 3 个独立 Einsum: A@X, (AX)@W1, σ(·)
    Layer 2: H2 = σ(A × H1 × W2)     // 3 个独立 Einsum: A@H1, (AH1)@W2, σ(·)
    ```
    Custard 将每个 Einsum 独立编译为一个 SAM graph → 6 个独立 dataflow graph（或用户手动重写为更少的 graph），每个 graph 的中间张量（如 `A @ X` 的结果）必须**物化到 HBM2 内存**再被下一个 graph 读取。
  - **系统框架**：Custard → SAM graph → Comal simulator / Onyx CGRA [42]。（Onyx 缺乏非线性函数和 masking 支持，因此 GCN 的非线性激活在 CPU 上执行。）
  - **编译框架**：Custard 编译单个 Einsum → 无跨表达式融合能力。用户手动重写 `H1 = σ(A × X × W1)` 为单表达式 → Custard 可对重写后的表达式做 IIF 融合 — 但这是手工劳动，不可扩展。
  - **kernel调度**：每个独立 SAM graph 的迭代策略为 **global iteration**：先遍历所有输入张量的 n 维联合坐标空间（例如 A 的 i、k 和 X 的 k、j 联合 → 4 维 global iteration），再执行计算。在高维稀疏 ML 模型中导致 **coordinate explosion**（坐标爆炸）：每个额外维度引入的坐标组合可能远超非零值数量。
  - **硬件架构**：SAM graph 的每个 primitive（LS, Intersect/Union, Rep, ALU, Red, LW, CD）映射到 CGRA 的 PE（Processing Element），通过 streaming interconnect 连接。Onyx [42] 是目前唯一的 sparse CGRA 物理实现（12nm, 756 GOPS/W）。
  - **芯片设计**：论文未明确说明（Onyx CGRA 属于硬件架构层，芯片物理实现细节不在本论文范围）。

  **Baseline 痛点**：
  1. **缺乏跨表达式融合（无 EKF）**：C+S 无法自动融合跨多个 Einsum 的计算。中间张量必须物化到 HBM2，引入大量内存传输——在稀疏场景下，这些传输的 overhead 往往超过实际计算。
  2. **Global iteration 导致坐标爆炸**：n 维全局迭代空间中的坐标处理开销随输入张量数量指数增长，稀疏 ML 模型常含多个高阶张量和混合稀疏/稠密索引，使 global iteration 极度低效。
  3. **Manual rewrite 不可扩展**：对于含非线性操作、masking 和多输出的模型（如 GPT-3 BigBird），手动重写为单个超表达式不可行。
  4. **缺乏 fusion granularity 控制**：C+S 的 IIF 要么完全不融合，要么通过手工重写完全融合——无 partial fusion 选项，无法探索 fusion-recomputation tradeoff 空间。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FuseFlow: 面向数据流架构的融合为中心稀疏 ML 编译框架**

  FuseFlow 通过三个核心设计系统性地解决 C+S 的缺陷：

  **FuseFlow 全栈执行例子（编译 GCN 2-layer on RDA，对比 baseline）**：
  - **算法pipeline**：
    ```
    // Baseline (Custard): 6 个独立 graph，每个中间张量物化到 HBM2
    // FuseFlow: 用户标注 Fuse{Layer1} 和 Fuse{Layer2}
    
    // FuseFlow partial fusion of GCN Layer 1:
    Fuse{
      E1: H1'_{ij} = A_{il} X_{lj}         // 稀疏-稠密 matmul (A CSR, X dense)
      E2: H1_{ik}  = H1'_{ij} W1_{jk}      // 稠密-稠密 matmul
      E3: O_{ik}   = relu(H1_{ik})         // 逐元素非线性
    }
    // → FuseFlow 编译为 1 个 SAMML graph（3 个 Einsum 融合）
    //   H1' 和 H1 不再物化到 HBM2，而是通过 streaming 流式传递
    ```
    1. **Cross-Expression Fusion (EKF)**（Section 5）：自动检测 `Fuse{}` region 内的所有 Einsum，通过 index substitution + POG 生成融合表示。跨 kernel 的索引变量（如 `H1'_{ij}` 的 `i,j` → `H1_{ik}` 的 `i`）被替换连接。POG 确保所有张量的 mode order 和 dataflow order 一致性。
    2. **Partial Fusion 支持**：用户可以按层（partial fusion）或跨层（full fusion）标注 fusion region。FuseFlow 不强制全融合 — 这是对 C+S "all-or-nothing" 融合的直接改进。
    3. **Factored Iteration (IIF 增强)**（Section 6）：FuseFlow 的 fusion table lowering **始终生成 factored iteration 而非 global iteration**。对于 GCN Layer 1，分解为：
       - 子空间 1（`i → l → j`）：A 和 X 的 co-iteration → 产生 `H1'` 的 value stream
       - 子空间 2（`i → j → k`）：`H1'`（流式）和 W1 的迭代 → 产生 `H1` 的 value stream
       - 子空间 3：relu 直接消费 `H1` stream，逐元素变换
       - 坐标处理开销限于每个二元子空间（3 维 max），而非 5 维 global iteration。

  - **系统框架**：
    - FuseFlow 将 PyTorch 模型编译为完整的 fused SAMML graph（含 ML 原语：nonlinear op、masking、dense blocks）。
    - SAMML graph 可直接映射到 Onyx [42]/Opal [10] CGRA，无需 CPU fallback（因为 SAMML 扩展了原始 SAM 以支持 non-sparse 操作）。

  - **编译框架**：
    1. **PyTorch → MLIR Linalg + SparseTensor**：Torch-MLIR（dense）+ MPACT（sparse），保留稀疏格式和调度标注。
    2. **Fused Einsum + POG**（新 IR #1）：EKF 算法自动融合 `Fuse{}` region 内的所有表达式，生成融合 Einsum 表示 + 偏序图（POG）。POG 追踪每个索引变量的 outer-to-inner 约束，通过 topological sort 产生所有合法 global dataflow order。
    3. **Fusion Table**（新 IR #2）：二维表格 IR，行 = 融合迭代顺序，列 = 张量视图，单元格 = dataflow 原语或指针引用。关键能力：
       - **Deferred graph construction**：允许在节点创建前通过 named pointers 引用未来节点
       - **Malleable topology**：通过移动 cells 即可重连 dataflow graph，无需复杂的图变换
       - **Interleaved input iteration + computation**：factored iteration 是融合表结构的自然结果
    4. **SAMML Codegen**：top-down 遍历融合表，实例化 dataflow nodes（LS → Intersect/Union → Rep → ALU/Red → LW/CD），生成 factored-iteration SAMML graph。
    5. **Scheduling Language**：用户通过 `Fuse{}` regions（融合粒度）、Linalg affine maps（dataflow order）、CLI 参数（parallelization factor, sparsity block size）显式控制编译优化。

  - **kernel调度**：
    1. **Dataflow Ordering 选择**：POG 枚举所有合法 global order，用户/autotuner 选择最优。在 nested matmul 中，最优与最差 order 的性能差 ~29×。对每个 kernel 施加局部最优 order 约束可将搜索空间缩减 68.5%-99.9%。
    2. **Parallelization**：Stream Parallelizer 对指定索引变量进行坐标 partition，复制计算子图；Stream Serializer 合并并行结果。支持 nested parallelism（两变量的 factor=4×4 产生 ~15.9× speedup）。
    3. **Sparsity Blocking**：将 dense blocks 映射到最内层坐标，外层稀疏迭代 + 内层 dense block 流式送入 vectorized ALU — 兼具稀疏数据流优势（外层跳过零块）和稠密计算密度（内层 block 级并行）。
    4. **Heuristic**：在仿真前用 FLOPs/bytes 快速分析模型 pruning suboptimal 配置（avg error: FLOPs 1.8-2.8%, bytes 5.7-11.5%）。

  - **硬件架构**：
    - SAMML graph 的每个 primitive 可综合为硬件模块（VLSI 实现），通过 streaming interconnect 连接，按 dataflow 语义执行。FuseFlow 已验证 FPGA（Xilinx VU9P, Vitis HLS）上的 cycle count 与 Comal simulator 高度一致（$R^2=0.991$）。
    - **Comal Simulator**：基于 DAM framework [81]（Rust）的 cycle-accurate 仿真器，集成 HBM2（Ramulator 2.0）。模拟原理：token-passing dataflow — 每个 node 按 pipeline depth 处理 tokens，stream edges 满时产生 back-pressure stall。

  - **芯片设计**：论文未明确说明。

  **设计对应关系总结**：
  | Baseline 缺陷 | FuseFlow 设计 | 效果 |
  |---|---|---|
  | C+S 无跨表达式融合（EKF），中间张量必须物化到 HBM2 | Cross-Expression Fusion via POG：自动融合 `Fuse{}` 内所有 Einsum | 消除中间张器物化，GPT-3 full fusion ~2.7× speedup |
  | Global iteration 导致坐标爆炸（coordinate explosion） | Factored iteration via Fusion Table：将 n 维空间分解为多个 pairwise 子空间交错执行 | 坐标处理开销限于二元操作，GCN 在 99.9% 稀疏度下仍高效 |
  | 手动重写不可扩展（GPT-3 BigBird 含 17+ 操作、非线性、masking） | 用户仅需标注 `Fuse{}` region，编译器自动完成 EKF + lower | 支持 4 类模型（GCN, GraphSAGE, SAE, GPT-3）跨领域评估 |
  | All-or-nothing 融合（要么全不融合，要么手工重写后全融合） | Partial Fusion 支持：用户按需标注 fusion boundary | GCN 和 GraphSAGE 在 partial fusion 下最优（~2.6-3.9×），full fusion 因 recomputation 反而退化 |
  | 缺乏 fusion-recomputation tradeoff 探索 | Scheduling language + Fusion heuristic（FLOPs/bytes 快估 + prune） | 可探索 56 种等效 dataflow configurations，heuristic 正确预测最优配置 |
  | Onyx CGRA 缺乏 ML 原语（非线性、masking）→ CPU fallback | SAMML IR 扩展原始 SAM 以支持非线性 op、masking、dense blocks | 完整 ML pipeline 在 dataflow 硬件上执行，无 CPU fallback |
  | C+S 无 parallelization、sparsity blocking 等 ML 级优化 | Parallelization（stream parallelizer/serializer）+ sparsity blocking + dataflow order selection | nested parallelism factor=4×4 → ~15.9× speedup；block sparse speedup ∝ block size |

## MAC-Attention: Match-Amend-Complete Attention for Efficient Long-Context Inference

- baseline方法是什么？
  Baseline 是标准 dense scaled dot-product attention (SDPA) 在 LLM decode 阶段的完整 KV cache 重计算。具体地，每生成一个 token，每个 query head 对所有历史 token 的 KV cache 计算 softmax(QK^T/√d)V，从 HBM 读取完整的 O(N·d) 字节 KV cache。

  **Baseline 全栈执行例子**（以 Llama 3.1 在 H100 上 SGLang + FlashInfer decode 为例）：
  - **算法pipeline**：标准 Transformer decoder decode phase，每层 MHA/GQA attention + FFN。每个 decode step：从 HBM 读取完整 KV cache → FlashInfer dense decode kernel 执行 tiled attention（QK^T, softmax, PV）→ 输出 attention embedding。
  - **系统框架/Serving**：SGLang 接收请求 → paged KV cache 管理 → 每个 decode step 从 HBM 读取完整 KV cache（O(N·d) 字节，N 为 context 长度）送入 FlashInfer dense decode kernel → 输出 logits → 采样下一个 token。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：FlashInfer 的 dense decode kernel 在 H100 上执行分块 GQA attention。以 128K context、BF16 精度为例，每步需从 HBM 读取约 128K × Hkv × d × 2 bytes 的 KV cache。Attention 计算为 memory-bound，受限于 HBM 带宽（H100 ~3.35 TB/s）。单个 token decode 的 attention 延迟随 context 长度线性增长。
  - **硬件架构**：NVIDIA Hopper H100，利用 Tensor Core 做 BF16 MMA 矩阵乘，FlashInfer 使用 ping-pong 异步拷贝（TMA）隐藏 HBM 延迟。但受限于 KV cache 读取的带宽瓶颈（memory-bound），Tensor Core 利用率低。

  Baseline 核心痛点：
  1. **IO 瓶颈**：decode 阶段每个 token 都要从 HBM 重新读取完整的 KV cache（O(N·d) 字节），随 context 线性增长，成为不可持续的带宽瓶颈。
  2. **计算冗余**：语义相似的历史 query 产生了相似的 attention 分布，但每个新 token 都从零开始重新计算 attention，之前的计算结果被丢弃。
  3. **压缩/驱逐方案的代价**：现有加速方法（如 KV cache 压缩、token 驱逐）虽减少 IO，但会降低 recall（fidelity loss）或限制模型可访问的 token 范围（access loss），影响长上下文任务质量。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 MAC-Attention 是一种**fidelity-preserving 和 access-preserving**的 attention 复用方案：通过 Match-Amend-Complete 三阶段，将 decode 阶段的 attention 计算从 O(N) 降为常数复杂度（匹配命中时），同时保持 full-attention 质量。核心洞察：语义相似的历史 query 产生的 attention 分布也相似，因此可以缓存并复用 prior attention output，只需对少数不匹配区域进行修正。

  **论文方法全栈执行例子**（以 Llama 3.1 在 H100 上 SGLang + MAC-Attention decode 为例）：
  - **算法pipeline**：
    1. Match：在大小 κ=512 的滑动窗口中用 pre-RoPE L2 距离匹配最相似的历史 query Q_m。匹配命中率 ~99.5%（τ=0.45）。
    2. Amend：在匹配边界附近的小 band（r=256）上重新计算 attention，修正 softmax 质量集中在 decode 光标附近导致的误差。公式：A_prefix = A_m ⊖ Attn(Q_m, K_{m−r~m}, V_{m−r~m}) ⊕ Attn(Q_n, K_{m−r~n}, V_{m−r~n})。
    3. Complete：将 amended prefix 与 KV tail 上的 fresh attention 用数值稳定的 log-sum-exp merge 融合。
    4. KV skip 比例 r_skip = (N − r)/N → 对于大 N，接近 100% KV 被跳过（120K context 时 ~98.9%）。
    5. 复杂度：匹配命中时 compute 和 bandwidth 为常数（仅处理 r 长度 rectification band + m~n tail），与 context 长度无关。
  - **系统框架/Serving**：SGLang + MAC-Attention runtime hook（零源码修改 SGLang）。Hook 在 decode 阶段拦截 BF16 paged-KV decode 调用，启动 `mac_persistent_decode_bf16` fused kernel 替代 FlashInfer。同时维护 ring cache（Q cache + A cache，大小 κ=512）。兼容 chunked prefill、continuous batching、speculative decoding、PD disaggregation、MHA/GQA。
  - **编译框架**：论文未明确说明。MAC-Attention CUDA kernel 通过 `torch.utils.cpp_extension` JIT 编译。
  - **kernel调度**：`mac_persistent_decode_bf16` — fused persistent BF16 CUDA decode kernel，在单次 launch 中完成：in-kernel L2 matching → per-head hit/miss 分类 → load scheduling → partial attention computation（rectification band + tail for hits；full KV for misses）→ stable log-sum-exp merge → cache writeback。关键设计：(1) persistent kernel 避免多次 launch 和 HBM 往返；(2) in-kernel matching 在共享内存/L2 中完成；(3) FP32 partial workspace 保证数值稳定。
  - **硬件架构**：NVIDIA Hopper H100。MAC-Attention 将命中 head 的 HBM 读取从 O(N·d) 降至 O((r+tail_length)·d) ≈ constant，直接打破 context 长度与延迟的线性耦合。在 128K context 下 attention 阶段 speedup >14.3× vs FlashInfer，end-to-end 2.6× speedup。

  **设计对应关系总结**：
  | Baseline 缺陷 | MAC-Attention 设计 | 效果 |
  |---|---|---|
  | IO 瓶颈：每 token 读取完整 KV cache（O(N·d)） | Match-Amend-Complete 复用：命中 head 仅读取 rectification band + tail（O(constant)） | KV 访问减少 99%，attention 延迟降为常数级 |
  | 计算冗余：历史 attention 结果被丢弃 | Q cache + A cache ring buffer 保存最近 512 token 的 query 和 attention output | 匹配命中率 ~99.5%，~98.9% KV 被跳过 |
  | 压缩/驱逐方案损失 fidelity 或 access | Fidelity-preserving：不压缩 token；Access-preserving：不驱逐 token | LongBench v2 质量与 full attention 持平（37.0→37.0） |
  | Softmax mass 集中在 decode 光标附近，简单复用引入误差 | Amend 阶段：在小 band（r=256）上重新计算匹配边界附近的 attention | 无 Amend 时质量下降 ~8pp（Qasper 44.8→36.8），Amend r=64 恢复至 44.2 |
  | Multi-head 特性使 query 相似性难以统一判断 | per-head matching：每个 query head 独立匹配 + load balancing planning | 各 head 可匹配到不同的历史位置，最大化总 skip 率 |
  | 与现有 serving 基础设施不兼容 | Runtime hook 注入 SGLang（零源码修改），兼容 MHA/GQA/chunked prefill/continuous batching | 直接使用 `pip install -e` + 环境变量即可启用 |

  **创新本质**：MAC-Attention 将 attention 计算从"每次从零计算"转变为"缓存+增量修正"范式，类似于 CPU cache 的 temporal locality 原理——如果最近访问过的数据很可能再次被需要，为什么不缓存计算结果？关键挑战（Amend 解决）是 softmax 的非线性特性使得简单复用引入位置偏差，需要局部修正。

## PuzzleMoE

- baseline方法是什么？
  Baseline 是现有的 MoE 压缩方法，主要分两类：
  1. **Expert Dropping**（NAEE, STUN）：根据 calibration dataset 上的 expert 输出重要性，整块丢弃"不重要"的 experts，保留"重要"experts 原样。NAEE 使用穷举搜索选择保留哪些 experts（DeepSeek-MoE 64 experts 需 ~10¹⁸ forward passes，不可行），STUN 通过 latent structure 将选择复杂度降至 O(1)。缺陷：丢弃整个 expert 可能意外丢掉关键领域知识（不同 downstream task 需不同 calibration dataset），导致 >20% accuracy drop。
  2. **Expert Merging**（HC-SMoE, Sub-MoE, D2）：基于 expert 输出相似性聚类（HC-SMoE）或低秩近似（Sub-MoE, D2），将相似 experts 合并为少量 experts。通常需要多阶段 pipeline（聚类→SVD 分解→低秩补偿），虽然优于 expert dropping，但 coarse-grained 合并（整层 expert 按聚类分配到 merged expert）模糊了 expert 间的专业化差异，50% sparsity 下 MMLU 下降 >10pp。

  **Baseline 全栈执行例子**（以 HC-SMoE 压缩 Mixtral-8x7B 50% sparsity 在 A100 上推理为例）：
  - **算法pipeline**：HC-SMoE 对每个 MoE layer：(1) 在 calibration dataset 上收集所有 experts 的输出作为特征向量；(2) 基于输出余弦相似性进行层次聚类（hierarchical clustering），将 8 个 experts 合并为 4 个 clusters；(3) 每个 cluster 内对 expert 权重取加权平均得到 merged expert 权重。合并后模型结构与原 MoE 一致，仅 expert 数量减半。
  - **系统框架/Serving**：合并后模型为标准 MoE 格式（仅 expert 数量少），可直接用 vLLM/HuggingFace Transformers 加载推理，无需特殊 runtime。路由 gate 相应减少输出维度。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：标准 cuBLAS dense GEMM 计算 expert 前向。无自定义 kernel——合并后 expert 仍为 dense weight matrix，直接调 GPU 标准矩阵乘法。
  - **硬件架构**：A100 GPU，无硬件修改。运行时与其他 MoE 推理相同：根据 routing gate 的 top-k 选择激活对应 experts，各 expert 独立执行 GEMM。
  - **芯片设计**：论文未明确说明。

  Baseline 核心痛点：
  1. **共享知识与专业化知识的冲突**：MoE experts 中同时存在共享知识（如通用语言建模的权重模式，跨 experts 相似）和专业化知识（如处理特定 domain/语言模式的参数）。Expert dropping 直接丢弃含有独特能力的整个 expert；coarse-grained merging 将整个 expert 不加区分地平均，伤害 expert 间的专业化差异。
  2. **粗粒度合并的精度损失**：HC-SMoE 等合并整层 expert 权重（d×h 全部元素一起平均），无法区分"共享可合并"和"独有需保留"的权重条目。Expert 在部分维度上相似（如语言建模基础表示）但在其他维度上分化（如专业知识表示），全量平均将分化维度也抹平。
  3. **Calibration dataset 依赖性**：NAEE 等 expert dropping 对 calibration data 敏感——使用 C4（通用语料）校准在 MATH 任务上效果差（41.5 vs 48.7 用 MATH 校准），因为 C4 中不重要的 expert 可能在数学任务中至关重要。
  4. **压缩时间过长**：D2 的 SVD 分解在 Mixtral-8x7B 上需 55 分钟；HC-SMoE 聚类 + 合并约 210 分钟；NAEE 穷举搜索在 64-expert 模型上不可行。方法难以扩展到更多 expert 数量。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 PuzzleMoE 提出了两个核心创新：(1) **Sparse Expert Merging with Dual-Mask**——将合并从 coarse-grained（整层 expert）下沉到 fine-grained（逐权重元素）的稀疏合并，通过 dual-mask（similarity mask + saliency mask）同时保留共享知识和 expert 专业化；(2) **Bit-packed Encoding + Custom CUDA GEMV Kernel**——将 binary mask 和 sign bit 嵌入 Bfloat16 未使用指数位，零额外存储开销，并通过 on-the-fly 解码 kernel 实现高效推理。

  **论文方法全栈执行例子**（以 PuzzleMoE 压缩 Mixtral-8x7B 50% sparsity 在 1×A100-80GB 上推理为例）：

  - **算法pipeline**：
    1. **Calibration**（2 分钟）：在 C4 128 samples（seq_len=2048）上运行单次前向 pass，收集每个 expert 的输入激活 X，计算 Wanda saliency A_i = |W_i| ⊙ ‖X_i‖₂。无需迭代搜索或 SVD。
    2. **Pairwise Sparse Merging**（offline）：对每个 MoE layer，随机将 8 个 experts 分为 4 对。每对 (E_i, E_j)：
       - 计算 similarity mask M^{sim}：逐元素对称百分比差异 Δ = ||W_i|−|W_j||/(|W_i|+|W_j|)，Δ≤0.4 的条目标记为"相似可合并"。
       - 计算 saliency masks M_i^{sal}, M_j^{sal}：基于 Wanda 指标，互补地标识每个位置上哪个 expert 的权重更重要。
       - Dual-mask 合并：相似条目取幅度平均，不相似条目从更 salient 的 expert 选取 → W_{merged}。这保留了"共性可共享、个性可保留"的粒度控制。
       - 存储符号矩阵 S_i, S_j（每个 expert 权重的原始符号），推理时通过 (-1)^S 恢复带符号权重。
    3. **Bit-packed Storage**：观察到 Bfloat16 指数仅使用 5-bit（值域 112-128），将指数统一偏移 112 释放 3 个冗余 bit → 嵌入 mask bit 和 sign bit → W_{merged} 仍为 Bfloat16 张量，masks 和 signs 零额外存储。
    4. **Inference**（on-device）：
       - Expert routing gate 照常选择 top-k experts。
       - 对每个激活 expert：加载 packed W_{merged} → custom CUDA GEMV kernel 在数据加载路径上 on-the-fly 解码 mask/sign → 执行 Ŵ_i = (-1)^{S_i} ⊙ M_i ⊙ W_{merged} 的等效 GEMV 计算。
       - 因为 mask 和 sign 的提取/解码在 INT32 ALU 上完成（GEMV memory-load 期间空闲），不增加延迟。
    5. **关键指标**：压缩后 Mixtral-8x7B 从 2×A100-80GB 降至 1×A100-80GB（~50% memory reduction），1.28× inference speedup。MMLU 仅下降 2.2pp（67.9→65.7 vs HC-SMoE 67.9→49.0, 下降 18.9pp）。压缩时间 2 分钟（vs D2 55min, HC-SMoE ~210min）。

  - **系统框架/Serving**：压缩后模型为标准 PyTorch checkpoint（packed Bfloat16 权重），可被 PuzzleMoE 的 custom inference pipeline 加载。Custom CUDA GEMV kernel 通过 `torch.utils.cpp_extension` 加载。论文未集成到 vLLM 等 serving 框架，但 packing format 与标准 PyTorch 张量兼容。

  - **编译框架**：论文未明确说明。

  - **kernel调度**：Custom CUDA GEMV kernel with on-the-fly decoding（详见实验_kernel调度.md 条目）。关键设计：
    - Decoding 搭载在 weight loading path（register level，不写入 memory）。
    - 利用 GEMV memory-bound 特性：weight 加载时 INT32 ALU 空闲 → decoding 填充空闲 ALU cycles。
    - 无物化解码矩阵：解码结果直接送入 FMA（fused multiply-add），消除 HBM/SMEM 中间缓冲。

  - **硬件架构**：NVIDIA A100 GPU。PuzzleMoE 利用 Bfloat16 数据格式的指数域未使用 bit 嵌入 metadata，不修改硬件本身。无 RTL/模拟器修改。

  - **芯片设计**：论文未明确说明。

  **设计对应关系总结**：
  | Baseline 缺陷 | PuzzleMoE 设计 | 效果 |
  |---|---|---|
  | 共享知识+专业化知识冲突：整层合并模糊 expert 差异 | Dual-Mask Sparse Merging：逐元素区分"相似可合并"（M^{sim}）和"独有需保留"（M^{sal}） | 50% sparsity 下 Mixtral MMLU -2.2pp vs HC-SMoE -18.9pp |
  | 粗粒度合并不分权重重要性：全量平均浪费信息 | Activation-aware Saliency（Wanda）：基于 magnitude × activation norm 识别每 expert 的关键权重 | Activation-aware vs magnitude-only 在 Deepseek-MoE 上 +0.3pp average accuracy |
  | Calibration 敏感性：NAEE 的 MATH 任务需 MATH 校准 | 方法对校准集鲁棒：C4 vs MATH 校准的 Avg Accuracy 差异仅 0.1pp（72.6 vs 72.5） | Task-agnostic 设计，无需 domain-specific 校准 |
  | 压缩时间过长（55-210+ min）| 单次前向 pass（2 min）+ 闭式 mask 构造（O(d×h) pairwise merge） | 45× 更快压缩（Mixtral-8x7B: 2min vs D2 55min） |
  | Binary mask 和 sign 存储开销抵消 memory saving | Bit-packing：利用 Bfloat16 指数域冗余 bit 嵌入 mask/sign，零额外存储 | 无需额外 metadata tensor，packed weight 即标准 Bfloat16 |
  | 合并后推理需 mask 动态查找，latency overhead | Custom CUDA GEMV kernel：on-the-fly decode 搭载数据加载路径，decode 用空闲 INT32 ALU | 1.28× speedup（Mixtral-8x7B），decode overhead 可忽略 |
  | 合并≥3 expert 的组合爆炸 | Pairwise merging + 随机分组：线性复杂度 O(N/2) 合并，16 个随机种子结果稳定（std 0.2-0.4pp） | 足够好的性能（随机 vs 搜索 仅差 0.3pp），可扩展至任意 expert 数量 |
  | 与量化不可叠加 | 兼容量化：50% merging + 3-bit AWQ → 4.8× 总压缩，精度仅下降 1.7%（Mixtral-8x7B） | 两种压缩正交叠加，无相互干扰 |

  **创新本质**：PuzzleMoE 将 MoE 压缩从"哪些 expert 要保留"（expert dropping）和"哪些 expert 可以合并"（coarse-grained merging）的 discrete 选择问题，转化为"每个权重的每个条目应该被保留还是被合并"的 continuous fine-grained 决策问题。通过 dual-mask（similarity + saliency）在逐元素粒度上同时编码"共享"和"独有"信息，将 coarse-grained merging 的"all-or-nothing"困境打破为"merge what's similar, keep what's unique"的 selective merging 范式。Bit-packing 进一步消除了 fine-grained sparse merging 引入的 metadata 存储开销——利用浮点格式的未使用指数位作为"免费"存储，将 system overhead 降至零。

## MixLLM: LLM Quantization with Global Mixed-Precision between Output and Embeddings

- baseline方法是什么？
  Baseline 方法包括两类：(1) Weight-only 量化（GPTQ W4/W5、AWQ W4）——仅对权重做 4-bit 量化，激活保持 FP16，在 small-batch decoding 中效果良好但 large-batch decoding 和 prefill 下性能不足，因为 activation 的 FP16 读取成为瓶颈；(2) Weight-Activation 量化（SmoothQuant W8A8、QuaRot W4A4/W4A8、QServe W4A8）——对权重和激活均做量化，但权重使用 uniform bit-width（全部 4-bit 或全部 8-bit），在 per-layer 或 local per-channel 粒度上分配精度，缺乏跨层的全局视角。

  **Baseline 全栈执行例子**（以 QuaRot W4A4 在 A100 上推理 Llama 3.1 70B 为例）：
  - **算法pipeline**：QuaRot 通过 Hadamard 旋转消除 outlier 通道后，对权重和激活做 uniform 4-bit 量化。权重 asymmetric per-group（group=128），激活 symmetric per-token。所有输出通道统一 4-bit，无精度区分。W4A4 下 WikiText2 PPL 从 3.66（float16）退化为 ~4.20（+0.54）；MMLU-Pro 从 35.52 降至 27.60（-7.92）。
  - **系统框架/Serving**：vLLM 或 TRT-LLM，仅支持 uniform bit-width 的量化权重加载。QuaRot 需要额外 dequant 步骤将旋转后的权重转为标准格式。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：标准 int4/int8 Tensor Core GEMM（Cutlass/cuBLAS）。Uniform 4-bit 下所有通道走相同 kernel 路径。Dequant 开销集中在 scale×weight 的逐元素运算上。
  - **硬件架构**：NVIDIA A100 80GB，利用 int8 Tensor Core（mma.sync m16n8k32），但 uniform 4-bit weight 的精度损失限制了下游任务质量。

  Baseline 核心痛点：
  1. **Uniform Precision 的无效性**：不同输出通道对最终 loss 的贡献差异巨大——v_proj 和 down_proj 的通道对量化极度敏感（需 8-bit），而 gate_proj 通道几乎不受影响（4-bit 足矣）。Uniform 4-bit 对敏感通道引入不可接受的精度损失；Uniform 8-bit 对非敏感通道浪费 bit 预算。
  2. **Local Precision 决策的次优性**：per-layer 或 per-channel（local fraction）的精度分配在各自层内决定 bit-width，缺乏跨层的全局比较——"一个层的'重要'通道可能对最终 loss 的影响远小于另一个层的'不重要'通道"。
  3. **4-bit Activation 的 ROI 递减**：MatMul 计算受 weight 张量（大）约束大于 activation 张量（小）——weight 8→4 bit 提升 80% 计算强度，activation 8→4 bit 仅提升 ~5.88%，但 activation 4-bit 引入的精度损失远超收益。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 MixLLM 提出**全局混合精度量化（Global Mixed-Precision）**：在 output channel 维度上，基于每个通道对最终模型 loss 的全局贡献（salience）统一排序，将 bit 预算精确分配到最需要精度的通道上。核心设计包括：
  1. **全局 Salience 搜索**：基于 Fisher Information Matrix + 二阶 Taylor 展开的通道 salience 度量 `S_c = 1/|D| Σ|g_d^TΔ + 0.5(g_d^TΔ)²|`，单 pass 计算所有线性层所有输出通道的 loss 敏感度并全局排序。前 10% 通道分配 8-bit，其余 4-bit（W4.4A8）。
  2. **Output Channel-wise 混合**：混合精度按 output channel 而非 input channel 或 layer 维度——每个 output channel 内的所有元素同 bit-width，使 INT4 和 INT8 kernel 子问题在 GPU 上独立并行。
  3. **Algorithm-System Co-Design**：Two-step dequantization 利用 int8 Tensor Core、Fast I2F 将 int-to-float 转为 float 减法并融合进 MMA accumulator、Multi-level software pipeline 重叠内存加载/MMA/SIMT。

  **论文方法全栈执行例子**（以 MixLLM W4.4A8 在 A100 上推理 Llama 3.1 70B 为例）：
  - **算法pipeline**：
    1. 校准阶段：在 Wikitext2 128 样本上运行单 pass 全局 salience 搜索（55 分钟）→ 全局排序所有 output channel → top 10% 标记为 INT8。
    2. 量化阶段：INT8 通道 symmetric group-wise（group=128）、INT4 通道 asymmetric group-wise（group=128）、Activation 8-bit symmetric group-wise（group=128）。
    3. 推理阶段：每个 Linear 层根据 precision map 分发通道到 INT8/INT4 kernel 路径，CUDA Graph 并行执行。
    4. 精度结果：WikiText2 PPL 增量 <0.2 vs float16（vs QuaRot W4A4 +0.54）；MMLU-Pro 34.53 vs float16 35.52（vs QuaRot W4A4 27.60）。
  - **系统框架/Serving**：vLLM v0.9.0 + MixLLM patches。Serving 层新增 per-channel precision map 的加载和 GEMM dispatch 逻辑。Weight 离线 prepacked 为 INT4 packed 格式 + INT8 通道单独存储。Fused epilogue scatter 在 CUDA Graph 中合并 INT4/INT8 两路输出。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：自研混合精度 CUDA GEMM kernel。Two-step dequant 将 uint4→int8 转换融入 MMA pipeline，利用 int8 Tensor Core 而非 FP16 路径。Fast I2F：整数加 bias `0x4b400000` 映射到连续 float 区间，转为单次 float 减法并融合进 accumulator 初始化 `D = A*B + D`。Multi-level pipeline：quantization group tile（128 元素）作为 pipeline 调度单元，重叠 Global→Shared→Reg 加载、MMA、SIMT dequant。vs float16 baseline speedup：W4A8 1.90×、W8A8 2.75×、W4.8A8 1.88×。
  - **硬件架构**：NVIDIA A100 80GB，利用 int8 Tensor Core（mma.sync m16n8k32）。CUDA Graph 用于消除 INT4/INT8 两路 kernel 的 launch overhead。Prepacked weight layout 优化 global memory coalescing。

  **设计对应关系总结**：
  | Baseline 缺陷 | MixLLM 设计 | 效果 |
  |---|---|---|
  | Uniform precision：敏感通道和非敏感通道同 bit-width，要么损失精度要么浪费 bit | 全局 Salience 搜索：基于 Fisher + Taylor 二阶展开全局排序通道敏感度 | top 10% 8-bit + 90% 4-bit（W4.4A8），PPL 增量从 SOTA ~0.5 降至 <0.2 |
  | Local precision 决策：per-layer/local fraction 无法跨层比较通道重要性 | 全局统一排序：所有层的所有 output channel 在同一标准下排名 | v_proj 71.22% 8-bit、down_proj 53.82% 8-bit、gate_proj 0.73% 8-bit——精度分配高度集中在关键层 |
  | 4-bit activation 的 ROI 递减：activation 降 bit 计算收益小但精度损失大 | 固定 activation 8-bit + weight 混合精度：W4.4A8 代替 W4A4 | 计算强度提升 ~80%（weight 4-bit）vs ~5.88%（activation 4-bit），W4.4A8 精度远优于 W4A4 |
  | 量化 kernel 未针对混合精度优化：dequant 开销和 I2F 转换成为瓶颈 | Algorithm-System Co-Design：Two-step dequant + Fast I2F + Multi-level pipeline | W4A8 vs TRT-LLM 1.26× speedup，W8A8 1.78× speedup |
  | 不同 bit-width 通道的 kernel dispatch 开销 | CUDA Graph + Fused Epilogue Scatter | Scatter 开销 "basically costless" |

  **创新本质**：MixLLM 的核心洞察是 LLM 中不同 output channel 的信息重要性天然不均——某些通道（尤其 v_proj/down_proj）承载了 loss 函数的大部分梯度信号。通过将 bit 预算从"每层均分"转变为"全局集中投放"，用 10% 额外 bit 预算换取接近无损的精度。这不是简单的"多一点 bit 好一点"——而是"把 bit 花在最值得的地方，收益远超均匀分配"。

---

## Reducing GPU Memory Fragmentation via Spatio-Temporal Allocation Planning (STAlloc)

- baseline方法是什么？
  Baseline 是 PyTorch 的 CUDA Caching Allocator——一种 online best-fit 内存分配器。其工作原理：为减少 cudaMalloc/cudaFree 系统调用开销，预先分配大的 caching blocks，在 blocks 内部按 best-fit 策略（选择最合适大小的空闲 slot）切分 chunks 分配给 tensor requests。分配决策完全在线（online），对 tensor 的 lifespan 没有任何先验知识——分配器不知道一个 tensor 何时会被释放，因此无法预判当前的空闲空间是否会在后续被需要。随着 alloc/free 交替发生，空闲空间碎片化为不连续的小段，难以容纳大尺寸的新 request。

  **Baseline 全栈执行例子**（以 Llama2-7B 在 8×A800 上训练，Virtual Pipeline + recomputation 配置为例）：

  - **算法pipeline（模型层）**：Llama2-7B Transformer stack，32 decoder layers，每层包含 multi-head attention + FFN（gate_proj/up_proj/down_proj）。Forward pass 产生 activation tensors，backward pass 消费 activation tensors 计算梯度。Forward→backward 形成天然的"先分配后释放"的 LIFO 模式，但 recomputation 打破了这一模式——某些 activation 在 forward 后立即释放（不保留到 backward），然后在 backward 时重新计算并短暂分配。

  - **系统框架/Serving（训练框架层）**：Megatron-LM 管理分布式训练（TP/PP/DP/VPP）。Pipeline parallelism 中 1F1B 调度：每个 micro-batch 先执行 forward（分配 activation），进入 1F1B 稳态后交替执行 forward 和 backward。VPP 将每个 pipeline stage 进一步划分为多个 virtual stages，增加 micro-batch 交错粒度以减少 pipeline bubble。每次 forward/backward 执行过程中，PyTorch operator 向 CUDA Caching Allocator 发出 malloc/free 请求。

  - **编译框架**：论文未明确说明。

  - **kernel调度（GPU 内存分配层）**：PyTorch CUDA Caching Allocator 接收请求流（约 86,721 requests/iteration for Llama2-7B-R）。每个 request（size s）到达时：
    1. 在已缓存的 free blocks 中搜索 best-fit block（≥s 且最小浪费）。
    2. 若找到，切分出大小为 s 的 chunk，返回其地址；剩余部分保留为新的 free block。
    3. 若找不到，调用 cudaMalloc 分配新 block（通常 2MB 起始，expandable）。
    4. Free 时将 chunk 归还并尝试与相邻 free blocks 合并。
    5. 核心缺陷：best-fit 仅关注当前 snapshot 的 fit 质量，不感知 tensor lifespan。例如，一个 early-allocated 的大 tensor 如果 lifespan 跨越整个 iteration（persistent），best-fit 可能将其放置在地址空间中段，使得后续 transient tensor 在释放后留下的空洞无法聚合为足够大的连续空间。

  - **硬件架构**：NVIDIA A800-80GB GPU，HBM2e 80GB。CUDA Caching Allocator 管理 host 侧的内存分配逻辑，通过 cudaMalloc 操作 GPU 设备内存。PyTorch 的 caching allocator 在 host 侧运行，不涉及 GPU kernel 修改。

  Baseline 核心痛点：
  1. **盲目性（Blindness to Lifespan）**：Online allocator 不知道 tensor lifespan，best-fit 的局部最优决策在全局时间轴上累积为严重碎片（fragmentation ratio 可达 43%）。
  2. **优化技术加剧碎片**：Virtual Pipeline（增加 30% allocation requests，interleaved alloc/free 模式）、recomputation（transient tensors 的快速 alloc-free 产生大量微小空洞）、ZeRO（partition optimizer states 引入更多分散分配）等必要的训练优化反而使碎片更严重。
  3. **碎片限制高吞吐配置**：更高吞吐的 parallelism 配置需要更多 GPU memory，但碎片导致实际内存使用远超理论值，触发 OOM。开发者被迫降级到低效配置（e.g., disable VPP 或增加 TP，training speed 损失 up to 24.5%）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  STAlloc 提出一种**离线规划 + 在线分配**的混合内存分配范式，利用 LLM 训练中 allocation requests 的时空规律性（spatio-temporal regularity）在训练前生成近乎最优的分配计划，训练时按计划执行分配。

  核心洞察：LLM 训练的 GPU 内存分配行为具有两个维度的规律性——
  - **Spatial Regularity**：Transformer layers 的结构重复导致 allocation size 高度重复——~10⁵ requests 中仅有 ~32 种 distinct tensor sizes。
  - **Temporal Regularity**：Tensor lifespan 分为三类——persistent（贯穿训练全程的 weights/gradients/optimizer states）、scoped（forward 分配 → backward 释放的 activations）、transient（分配后立即释放的中间结果，recomputation 下尤为显著）。

  **论文方法全栈执行例子**（对比 baseline）：

  - **算法pipeline（模型层）**：与 baseline 相同（不修改模型结构或训练算法）。

  - **系统框架/Serving（训练框架层）**：通过 PyTorch PluggableAllocator 接口加载 STAlloc，替换 CUDA Caching Allocator。训练框架代码修改不超过 5 行（monkey-patch 插桩以记录 module name）。训练初始化时，Plan Synthesizer（Python standalone tool）读取 profiler 记录的 allocation traces，生成分配计划。训练框架本身（Megatron-LM/Colossal-AI）零修改。

  - **编译框架**：论文未明确说明。STAlloc 的 Plan Synthesizer 类似于 compiler 的 register allocation / memory layout optimization pass——它接收 allocation request trace 作为 IR，执行 grouping 和 fusion optimization，输出 address-assigned allocation plan。可以视为一种 domain-specific "memory layout compiler"，但并非传统 ML compiler（如 TVM/XLA）的扩展。

  - **kernel调度（GPU 内存分配层）—— STAlloc 的核心贡献**：
    
    **Phase 1: Profiling**（训练前 3 iterations）：
    - Allocation Profiler 直接使用 cudaMalloc/cudaFree（绕过 caching allocator），以精确捕获每个 request 的真实 size 和 lifespan。记录 structure：M = {(s, t_s, t_e, p_s, p_e, dyn, l_s, l_e)}。

    **Phase 2: Plan Synthesis**（训练前，offline，~2-6 分钟）：
    1. **HomoPhase Grouping**：将 static requests（dyn=False）按 computation phase (p_s, p_e) 分组。相邻 groups 的条件 fusion——当合并后 TMP (Time-Memory Product) 提升时执行 fusion，在组内将 requests 紧密排列以减少 spatio-temporal bubbles。
    2. **HomoSize Grouping + Memory-Layer Construction**：将 HomoPhase Group 输出的 plans 按 size 分组。对特定 size S 的 requests，Algorithm 1 将 non-overlapping lifespan 的 requests 堆叠到同一 memory-layer（复用同一地址空间），最小化 layer 数量。这一步骤将 O(2^N) 的 NP-hard search space 通过 size×time 解耦降至 O(N log N)。
    3. **Global Planning**：HomoSize Groups 按 size 降序处理——大 size 优先占据地址空间，小 size 尝试填入大 groups 的空闲 intervals。这确保大 tensor 不被小 tensor 碎片化。
    4. **Dynamic Reusable Space 定位**：对 dynamic requests（dyn=True，MoE experts），利用其固定的 lifespan 特性（虽 size 不确定但 lifespan 确定），在 Static Allocation Plan 中标识时间区间内空闲的地址段 A_i——这些空间在 static requests 的 gaps 中，可以安全地共享给 dynamic requests。

    **Phase 3: Runtime Allocation**（训练时）：
    - **Static Requests**：O(1) 查找——直接返回 pre-planned address a。无搜索、无 GPU API 调用。
    - **Dynamic Requests**：计算 A_c = A_a ∩ A_i（当前实际空闲 ∩ pre-identified reusable space），best-fit 选择。无法满足时 fallback 到 Caching Allocator。
    - 静态池通过 cudaMalloc 一次性预分配（size = plan 的峰值），运行时不再调用 cudaMalloc/cudaFree。

  - **硬件架构**：NVIDIA A800/H200 + AMD MI210 GPU。STAlloc 通过 PyTorch PluggableAllocator 接口与 GPU 厂商无关——在 NVIDIA（cudaMalloc/cudaFree）和 AMD（hipMalloc/hipFree）上均验证通过。预分配 contiguous memory block 后不再调用 native GPU API，消除 GPU driver 层面的不确定性和 overhead。

  **设计对应关系总结**：

  | Baseline 缺陷 | STAlloc 设计 | 效果 |
  |---|---|---|
  | Online best-fit 不知道 tensor lifespan，局部最优导致全局碎片 | 离线规划利用完整 lifespan 信息做全局优化——HomoPhase Grouping 按 lifespan 聚类 + HomoSize Grouping 让相同 size 的 requests 复用 memory-layer | Fragmentation ratio 从 43% 降至 <5%（平均 85.1% reduction） |
  | Optimizations (VPP/Recomp) 增加 alloc/free 交错和请求量，加重碎片 | Spatio-temporal regularity 对 optimization techniques 仍然成立（~32 distinct sizes 不变），planning 将复杂 alloc pattern 预先规整 | VPP 配置下 memory efficiency 从 80%→99%+ |
  | 碎片导致 OOM，强制降级配置损失 throughput | 减少碎片释放 GPU memory → 原本 OOM 的高吞吐配置可运行 | Qwen2.5-14B 16 GPUs: STAlloc 使 VPP+TP=2 运行，比次优可运行配置快 32.5% |
  | 动态模型（MoE）的 unpredictable size 破坏 offline planning | Dynamic Reusable Space: 利用动态 requests 的 lifespan 固定但 size 不固定的特性，在 Static Plan 中 pre-identify 安全复用区间 | MoE 模型 fragmentation ratio 从 17.7%→4.3%（74.9% reduction vs PyTorch） |
  | 现有 defragmentation 方案（GMLake virtual memory stitching、PyTorch ES）在高动态场景下有显著 runtime overhead | STAlloc 的预分配 contiguous block + O(1) static allocation 无 runtime 搜索和 virtual memory 操作 | Throughput overhead <0.05%，而 GMLake 在 MoE 下 threshold 64MB 时性能下降 56.4%，PyTorch ES 在 32B recomputation 下下降 15% |
  | NP-hard 的 Dynamic Storage Allocation 问题在大规模（N>10⁵）下不可直接求解 | Spatio-temporal regularity 驱动的 grouping 解耦：将 N 个 requests 的空间和时间维度分离处理 → HomoPhase Groups（时间聚类）+ HomoSize Groups（空间聚类） | O(N log N) 复杂度，实际 plan synthesis 时间 ~2 分钟（simple cases）至 ~6 分钟（complex MoE cases） |

  **创新本质**：STAlloc 的核心方法是将 GPU memory allocation 从"反应式在线决策"（online allocator 在不知道 future 的情况下做 best-effort）转变为"前瞻式离线优化+确定式在线执行"（offline planner 知道完整的请求序列后做全局优化 → runtime 按计划机械执行）。这一范式转换之所以可行，是因为发现了 LLM 训练中的 spatio-temporal regularity——并非偶然，而是 Transformer 架构的重复结构和训练 iteration 的一致性的必然产物。STAlloc 的本质可类比为 compiler 的 register allocation（将变量分配到寄存器并管理 spill），但应用在 GPU memory 层：HomoPhase Group ≈ live range analysis、HomoSize Group ≈ graph coloring、Dynamic Reusable Space ≈ spill slot sharing。

## Streaming Tensor Programs (STeP)

- baseline方法是什么？
  Baseline 是现有 SDA 编程抽象（以 Revet [38] 为代表）下可表达的最优 static schedule。在 SDA 上执行动态 ML 模型（如 MoE、attention）时，Baseline 的**全栈执行流程**如下：
  - **算法层**：MoE layer 使用 static tiling——将每个 expert 的输入 token batch 按固定 tile size（如 32）padding 后分块，执行矩阵乘法。Attention 使用 static coarse-grained 或 interleaved parallelization——固定分配 requests 到 parallel regions。
  - **系统/编程抽象层**：Revet 的 Dataflow Thread 模型限制动态原语只能操作 scalars（不能操作 vectorized/computed tiles），导致 memory-bound tensor 应用被迫使用 static primitives。Revet 无法将 scalar streams 动态组合为 dynamically-sized tiles。
  - **编译层**：没有显式的 stream shape semantics 和符号化性能分析。Opaque data rates 需要编译器从 imperative code 推断，编译优化被动而非主动。
  - **Kernel 调度层**：Static tiling 需要 sweep tile sizes 找到 Pareto-optimal 点（small tile → memory-bound from frequent reloads; large tile → waste on-chip memory from padding）。Configuration 为每个 expert 静态分配（即使仅部分 active 也占用所有资源）。Parallelization 固定 assignment（coarse-grained 或 round-robin），无法适应 KV cache length 变化导致的负载不均衡。
  - **硬件架构层**：SDA fabric（如 SN40L [33], Plasticine [35]）上 compute units 和 memory units 通过 hardware FIFOs 和 NoC 通信。但现有抽象未充分利用硬件固有的异步执行能力——无法在 runtime 动态激活/去激活 branches、无法根据 runtime data rate 调整 tile size。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  STeP 提出一种新的 streaming abstraction，通过**显式 memory hierarchy + 符号化 data rate + data-dependent control flow operators** 三个核心属性，解决 Baseline 在各层的缺陷。**论文方法的全栈执行流程**：
  - **算法层**：Dynamic tiling 用 Promote + Accum(dynamic) 替代 Reshape(padded) + Accum(static)，使 tile size 自适应 runtime token count（无 padding）；Configuration time-multiplexing 用 EagerMerge + RandomOffChipLoad 替代 fixed LinearOffChipLoad，动态时分复用 compute/memory 资源；Dynamic parallelization 用 Partition(EagerMerge selector) 替代 static assignment，work 一旦 parallel region 可用即分发。
  - **系统/编程抽象层**：STeP 的 three key properties 直接对治 Baseline 缺陷：
    1. **Explicit Memory Hierarchy**（Off-chip memory operators + On-chip memory operators）：Bufferize/Streamify 让程序员/编译器显式控制数据在 off-chip→on-chip→compute 之间的移动。解决 Revet 的 implicit memory hierarchy 导致的无法分析和表达高效 schedule 的问题。例如：可以精确知道 tile 在 on-chip memory 的驻留时间，优化 Bufferize rank 来平衡 off-chip traffic vs on-chip memory。
    2. **Symbolic Data Rate**（Stream shape semantics with static-regular/dynamic-regular/ragged dimensions）：每个 STeP operator 的 input/output stream shape 在抽象层面即已知（包括动态维度用符号表示）。解决 Ripple 的 opaque data rates——无需"lift imperative code"即可分析。例如：Partition 的输出 stream shape 是 [D_i, ...]（D_i 为符号），frontend 可直接计算 off-chip traffic = ||output_stream|| × |dtype|。
    3. **Data-dependent Control Flow Operators**（Partition, Reassemble, EagerMerge）：以 stream 粒度而非 scalar 粒度支持动态路由和合并。解决 Revet 的 Dataflow Thread 只能操作 scalars 的限制。例如：EagerMerge 可以在 [1,64] tile 粒度汇聚 expert 输出（不是单个 scalar），保持数据复用和 tiled computation 效率。
  - **编译层**：STeP 的 symbolic frontend 提供**自动符号分析**——使用 SymPy 计算每个 operator 的 off-chip traffic 和 on-chip memory requirement 表达式。Programmer/compiler 在 compile time 即可获取性能关键指标（即使维度是动态的），指导 schedule 决策。STeP 可充当 torch.compile 的 lowering target（Section 6.1），在编译期做 DSE。
  - **Kernel 调度层**：三项 optimizations 直接解决 Baseline 的调度缺陷：
    1. **Dynamic Tiling** 打破 static tiling 的 Pareto 边界：自适应 tile size → 不需要在 frequent reloads vs capacity waste 之间 trade off。对 Qwen3-30B-A3B 在 batch=64 时 PID=2.11×（即在 on-chip memory 和 cycle count 两个维度上，dynamic tiling 的点距离 static Pareto frontier 至少 2.11× 的改进）。
    2. **Configuration Time-multiplexing** 避免静态资源分配：同一 configure region 在 expert 间动态时分复用 → compute utilization 提升 2.51×~2.64×。对 modern MoE（128+ experts，top-8 activate），资源节省显著。
    3. **Dynamic Parallelization** 消除静态负载不均衡：work dispatch 基于 parallel region availability → 在 KV cache length high variance 下 1.47×~1.57× speedup。相比 static coarse-grained（batch=16 时多个 region idle），2.72× speedup。
  - **硬件架构层**：STeP 的 dynamic features 映射到 SDA 硬件的具体方式（Section 6.2）：
    1. **Stop tokens** 通过 datapath 中的 bits 标识和级别（prior SDAs [20, 38] 已验证可行）。
    2. **Control flow operators** 通过 spatially layout all branches + data-dependent runtime activation（硬件 predicate [56] 或 NoC routing [12]）。
    3. **Dynamic tensor sizes** 通过 memory virtualization：固定粒度 allocation + noncontiguous allocation + hardware-managed mapping cache（~6% overhead with 512KB local memory per unit [33]）+ spilling mechanism [11] 支持 unbounded sizes。

## Tilus

- baseline方法是什么？
  Baseline 是现有低精度 kernel 生成方法：Triton（tile-oriented compiler，编译器生成的 kernel）和 Ladder（schedule-oriented compiler，编译器生成的 kernel），以及手工 kernel QuantLLM 和 Marlin。Baseline 在模型推理的**全栈执行流程**如下：

  - **算法层**：低精度量化（如 A16W4 或 A16W6）将模型权重和/或激活量化到低位宽（int4, int6, fp6 等），以减少 DRAM 带宽使用和加速计算。量化后的矩阵乘法 A_f16 × W_lowbit 需要先将低精度权重加载到 on-chip memory，再 cast 为标准精度（如 f16），然后使用 Tensor Core 计算。

  - **系统/编程抽象层（Triton）**：Triton 提供 tile-based 编程模型，但**不暴露 GPU memory hierarchy**（registers/shared memory/global memory），且**缺乏原生低精度数据类型支持**。用户必须手动从 uint32 等标准类型中 unpack sub-byte 数据。当权重从 shared memory 加载到 registers 后，执行低精度→标准精度 layout conversion 时必须依赖 shared memory 中转，产生显著的中间数据搬运开销（图 1(a)）。

  - **编译层（Ladder）**：Ladder 扩展 TVM scheduling system 支持低精度计算，通过 type-level packing 将低精度数据打包到标准类型（如 2×int4 → int8）。但 Ladder 有两个根本限制：(1) **仅支持 2 的幂次位宽**（如 4-bit, 8-bit），无法有效处理 3/5/6/7-bit 等非标准位宽；(2) **其 primitive-style scheduling 无法表达 software pipelining**——权重加载与计算无法异步重叠，导致 memory latency 无法隐藏（图 1(b)）。

  - **Kernel 调度层**：Triton 生成的 kernel 在图 1(a) 的 Step 4（layout conversion）成为瓶颈——依赖 shared memory 做 layout conversion 产生大量 on-chip traffic。Ladder 生成的 kernel 在图 1(b) 的流程中缺乏 software pipelining——global→register→shared memory→register 的串行过程使 memory bandwidth 利用率不足。手工 kernel（QuantLLM, Marlin）针对特定量化方案高度优化但缺乏泛化性：QuantLLM 仅支持 FP 量化且不支持 sub-channel 量化粒度，Marlin 仅支持 int4 且不支持 Hopper GPU。

  - **硬件架构层**：所有方法运行在 NVIDIA GPU（A100/L40S/H100）上，使用 Tensor Core 进行矩阵乘法。但由于上述各层的缺陷，Tensor Core 的计算能力无法被充分利用——计算单元等待数据期间处于 idle 状态。特别是对于 batch size >1 的 decode 阶段（continuous batching 开启时），software pipelining 的缺失使得 memory-bound 的 decode matmul 性能严重退化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Tilus 提出一种**tile-level GPGPU DSL 和编译器**，通过代数布局系统 + thread-block-level 编程模型 + 任意低精度数据类型支持，解决 Baseline 在各层的缺陷。**论文方法的全栈执行流程**如下：

  - **算法层**：Tilus 的单一 program template 通过参数化 tile sizes 覆盖所有低精度数据类型（uint1-uint8, int2-int8, float3-float8）。与 Baseline 每种量化方案需要单独 kernel 不同，Tilus 的通用模板自动适应任意位宽和数据类型。

  - **系统/编程抽象层**：Tilus 引入三个核心设计直接对抗 Triton 的抽象缺陷：
    1. **代数布局系统（Algebraic Layout System）**：用 primitive layouts（local/spatial）的 Kronecker product（⊗）组合表示寄存器张量在 GPU 线程间的分布。布局支持除法（逆运算），使寄存器张量可在不同 dtype 和 layout 间**无代价 reinterpretation**——只要 per-thread bit 数相同即可。**直接解决 Triton 的 layout conversion 瓶颈**：图 1(c) Step 3 中，Tilus 通过 View 指令实现零开销 reinterpretation（无需 shared memory 中转），消除了 Triton 中占支配地位的 layout conversion overhead。
    2. **Thread-Block-Level 编程模型 + 显式 Memory Hierarchy**：Tilus 的指令集显式暴露 GPU 三级 memory hierarchy（registers, shared memory, global memory），提供 CopyAsync/CopyAsyncCommitGroup/CopyAsyncWaitGroup 指令支持 **software pipelining**。图 1(c) Step 1-2 中，权重通过 cp.async 异步从 global memory 加载到 shared memory，与计算重叠执行。**直接解决 Ladder 缺少 pipelining 的缺陷**。
    3. **原生任意低精度类型支持**：Tilus 内置支持 1-8 bit 有符号/无符号整数和任意 exponent/mantissa 分布的浮点类型。通过预处理 weight layout transformation（如 i6[BK,BN]→u8[BK×BN×6/8] 紧凑打包），使低精度数据用硬件友好的 LoadGlobal 加载。**直接解决 Ladder 仅支持 2 的幂次位宽的限制**。

  - **编译层**：Tilus 的编译 pipeline（Python DSL → IR → 优化 → Hidet IR → CUDA C → nvcc → 二进制）引入多项超越 Baseline 的编译优化：
    1. **自动向量化**：对 memory load/store 自动选择向量化指令（cp.async.v4, lds128, ldg128）。
    2. **指令选择**：当寄存器布局与 `spatial(8,4).repeat(1,4)` 兼容时，自动选择 ldmatrix PTX 指令替代 lds。
    3. **Memory Planning**：Shared Memory Planner 自动计算所需共享内存并映射 shared tensor；Global Memory Planner 管理 workspace 分配。
    4. **低精度 Lowering**：将低精度操作（如 i6 cast）lowering 到 PRMT + LOP3 + bitwise 指令——所有操作在寄存器内完成，零线程间通信。

  - **Kernel 调度层**：Tilus 生成的 kernel 实现了图 1(c) 的最优流水线：
    1. cp.async 异步加载 global→shared（pipelined，与计算重叠）
    2. shared→register 加载（可用 ldmatrix 优化）
    3. View 无代价 reinterpretation（代数布局保证）
    4. PRMT/LOP3 向量化 casting（全寄存器操作）
    5. Tensor Core mma 计算
    该流水线相比 Triton（无 software pipelining + shared memory layout conversion）和 Ladder（无 software pipelining + 仅 2 的幂次位宽）在每个环节都更高效。特别对于 batch size >1 的 decode 阶段，software pipelining 显著提升 memory-bound kernel 的性能。

  - **硬件架构层**：Tilus 的 kernel 在所有三代 NVIDIA GPU 架构（Ampere A100, Ada Lovelace L40S, Hopper H100）上均验证有效。Ladder 在 H100 上甚至无法生成合法 kernel（CUDA illegal instruction error），而 Tilus 在不同架构间保持一致的性能优势。

  **设计对应关系总结**：

  | Baseline 缺陷 | Tilus 设计 | 效果 |
  |---|---|---|
  | Triton 不暴露 memory hierarchy + 缺少低精度原生支持 → layout conversion 依赖 shared memory 成为瓶颈 | 代数布局系统 + 显式 hierarchical memory space → View 指令实现无代价寄存器张量 reinterpretation | 消除 shared memory layout conversion overhead（图 1(a) Step 4 → 图 1(c) Step 3） |
  | Ladder 无法表达 software pipelining → 权重加载与计算串行执行 | Thread-block-level 指令集 + CopyAsync/CopyAsyncCommitGroup/CopyAsyncWaitGroup | 支持 software pipelining，memory latency 与计算重叠（图 1(b)→图 1(c) Step 1-2） |
  | Ladder type-level packing 仅支持 2 的幂次位宽 → 3/5/6/7-bit 量化无高效 kernel | 原生任意低精度类型 (1-8 bit) + weight layout transformation (i6→u8 compact packing) + PRMT/LOP3 向量化 casting | 首次覆盖 uint1-uint8, int2-int8, float3-float8 全谱高效 kernel（图 11） |
  | 手工 kernel（QuantLLM, Marlin）针对特定量化方案 → 缺乏灵活性和可维护性 | 单一参数化 program template + auto-tuning（~200 configs/op） | 一套代码覆盖所有量化类型，仍超越手工 kernel（Mar: 1.03×, QuantLLM: 1.29×） |
  | Triton 缺少显式布局控制 → 无法在 global memory 中变换权重布局以优化加载 | 代数布局系统的除法操作 + transform program | 权重布局预变换使低精度数据以 coalesced u8 格式加载，消除非对齐内存访问 |
  | Ladder 对 batch size >1 性能退化（decode + continuous batching） | Software pipelining + k-dimension parallelization | 端到端 decode batch=16 时显著超越 Ladder（图 12 middle column） |
  | Ladder 在 Hopper (H100) 上生成非法指令 | 编译 pipeline 通过 Hidet IR + nvcc 生成，自动适配架构 | 跨 Ampere/Ada/Hopper 三代架构一致高效（图 13） |

  **创新本质**：Tilus 的核心创新是将低精度 GPU kernel 生成从"针对每种位宽和数据类型手工编写/调优"转变为"在代数布局系统框架内，通过单一 program template + auto-tuning 自动生成全谱高效 kernel"。这一范式转变的关键在于发现了寄存器张量布局的**代数结构**——布局可分解为 primitive layouts 的 Kronecker product，且这种分解支持逆运算（除法），使得无代价的 reinterpretation 成为可能。Tilus 的方法可类比为：Triton 之于 CUDA（提高抽象层级但牺牲了低精度控制）→ Tilus 在保持 tile-level 抽象的同时重新暴露了 registers/memory hierarchy 并引入布局代数作为第一性原理——既保持编程简易性，又获得手工 kernel 级别的性能。
