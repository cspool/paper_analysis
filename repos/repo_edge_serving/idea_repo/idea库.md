## TZ-LLM

- baseline方法是什么？
  - Baseline：**Strawman TEE 方案**——将 LLM 推理直接放入 Arm TrustZone TEE 中运行，使用静态预留安全内存或在每次推理时执行完整的"冷启动"流程，且仅使用 CPU 计算。
  - Baseline 全栈执行例子（以 Llama-3-8B 8-bit 量化、512-token prompt 为例，在 RK3588 平台上）：
    - **模型推理算法**：标准 decoder-only transformer（Llama-3-8B），8-bit 量化（参数 7.9 GB）。prefill + autoregressive decoding，KV cache 随生成 token 增长。无算法层面修改。论文未明确说明 baseline 是否使用 NPU——Strawman baseline 是 CPU-only。
    - **Serving 框架**：llama.cpp 作为 TA 运行在 TEE 中。冷启动流程（图 1）：(1) 框架初始化、解析模型元数据、创建 tokenizer（2.3s）；(2) TEE 通过 CMA 从 REE 分配 8 GB 连续物理内存（TZASC 要求），需迁移 CMA 区域中的可移动页面（最高 4.2s 在 8 GB 参数下）；(3) 从 NVMe SSD 加载加密的模型参数（I/O 延迟 4.2s，顺序读 ~2 GB/s）+ AES 解密（0.9s）；(4) CPU 执行 prefill computation operator by operator（164s CPU-only）。总冷启动 TTFT ≈ 2.3 + 4.2 + 4.2 + 0.9 + 164 = 175.6s。每次推理结束后安全内存全部归还 REE，下次推理重新执行完整冷启动。
    - **编译框架**：论文未明确说明。llama.cpp 使用 ggml 张量库，无额外编译优化。CPU 上使用 NEON SIMD 加速 INT8 矩阵乘法（若 llama.cpp 支持）。
    - **Kernel 调度**：CPU-only execution。llama.cpp 的 ggml 后端在 Cortex-A76（4 核 big）+ Cortex-A55（4 核 LITTLE）上调度 matrix multiply、attention、layer norm 等 kernel。传统 TEE 仅提供单线程给 TA，baseline 需额外机制支持多线程（否则仅 1 核工作）。所有 kernel 在 CPU 上串行执行。无 NPU 参与——NPU 被静态配置为非安全设备，TEE 中无法访问。论文未明确说明 baseline 内核调度细节。
    - **硬件架构**：Rockchip RK3588 SoC（4×A76@2.4GHz + 4×A55@1.8GHz + 3 核 NPU + 16 GB LPDDR4X + 1 TB NVMe SSD）。TrustZone 硬件（TZASC-400 保护 8 个连续物理内存区域为安全内存、TZPC 禁止非安全 CPU 访问安全设备 MMIO、GIC 安全中断扩展）。Baseline 中 NPU 被配置为非安全设备，TEE 内无法使用。安全内存静态预留或在每次推理时动态分配然后完全归还。
  - Baseline 痛点（对应 Challenge #1 和 #2）：
    1. **内存效率 vs 推理速度的两难（Challenge #1）**：静态预留大块安全内存（8 GB for 8B 模型）导致 REE 内存严重不足（手机 ≤24 GB 总量），而每次动态分配则导致超长 TTFT（冷启动开销 11.6s，不含 CPU 推理时间）。CMA 分配要求物理连续内存，高内存压力下 page migration 开销大（可达 4.2s）。模型文件需加密存储，加载后需 AES 解密（额外 0.9s）。
    2. **NPU 不可用于 TEE 推理（Challenge #2）**：NPU 在移动设备上对 LLM 推理加速显著（Rockchip NPU 对 Llama-3-8B prefill 加速 12.5×、decoding 加速 1.3×），但现有方案要么将 NPU 静态配置为非安全设备（TEE 无法使用），要么在 TEE 中部署完整 NPU 驱动导致 TCB 膨胀（60K+ LoC）和世界切换需驱动重初始化开销（32ms）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法（TZ-LLM 两大创新）：
    1. **Pipelined Parameter Restoration（§4.1）+ Pipeline-Aware Secure Memory Management（§4.2）**：利用 LLM 推理的 DAG 计算图具有确定性内存访问模式（按拓扑序逐 operator 执行，每个 operator 仅需当前层参数），将内存分配（CMA）、参数加载（NVMe I/O）和解密（AES）作为 restoration operators 插入计算图，与 computation operators 并发流水线执行。配合优先级抢占式调度和 partial parameter caching，将恢复延迟隐藏在计算时间下。
    2. **Co-driver NPU Time-Sharing（§4.3）**：基于 outsource-and-verify 原则，将 NPU 驱动分离为 REE 中的控制平面（复杂但不可信）和 TEE 中的微型数据平面（~1K LoC user-mode，最小 TCB）。通过 shadow job 机制实现统一调度，通过 TZPC/TZASC/GIC 动态配置实现 NPU 安全模式切换（无需驱动重初始化）。
  - 论文方法全栈执行例子（同上 Llama-3-8B 8-bit, 512-token prompt, RK3588, 20% partial cache）：
    - **模型推理算法**：同 baseline Llama-3-8B decoder-only transformer，8-bit 量化。无算法修改——创新的核心在系统层面。计算图被扩展：每个 computation operator 前插入三个 restoration operators（allocation, I/O, decryption）。MoE 模型或 early-exit transformer 的限制：论文方法会 prefetch 所有 experts 或所有层（包括未使用的），但额外开销可被未来推理摊薄。
    - **Serving 框架**：llama.cpp TA 在 TEE 中运行，配合流水线恢复调度器：(a) 框架状态从检查点恢复（跳过 2.3s 初始化）；(b) 若启用 partial caching（20% 参数已驻留在安全内存中），operator 0∼N_cached 立即开始 CPU/NPU 计算，消除初始流水线气泡；(c) 后续 operators 的参数在计算进行时并行恢复：CMA 多线程分配（3.8 GB/s @ 4 threads）→ REE 文件系统 aio 直接 DMA 加载加密参数到 extend_allocated 但尚未 extend_protected 的内存（无需 bounce buffer，减少内存和拷贝开销）→ extend_protected 通过 TZASC 保护 → CPU AES 解密 → computation 使用。优先级调度器：ready CPU computation operator > 最早 computation operator 的 restoration operator。分配和解密 operators 被划分为 micro-operators 并支持抢占（当 computation operator 变为 ready 时抢占当前 restoration micro-operator）；(d) 矩阵乘法等 computation operators 提交到 NPU（通过 co-driver 机制）；(e) 推理结束后按逆拓扑序惰性释放安全内存（shrink 从 TZASC 区域末端），保留早期 operators 参数供下次使用。内存布局：两个独立 TZASC 区域——region 1 用于 LLM 参数（first-in-last-out 分配/释放保持连续性），region 2 用于 KV cache + activations + 其他。
    - **编译框架**：论文未修改编译框架。llama.cpp 使用 ggml 原生后端 + Rockchip NPU backend（通过 rknn-llm）。论文未明确说明编译框架层面的修改。
    - **Kernel 调度**：双世界 NPU kernel 调度：(a) LLM TA 在 TEE 中初始化安全 NPU job 的执行上下文（I/O page table、register commands、I/O buffers 均在安全内存的独立 TZASC region 中）；(b) TEE NPU 驱动（user-mode, ~1K LoC）通过 smc 向 REE 驱动提交 shadow job；(c) REE 驱动统一调度安全和非安全 NPU jobs（YOLOv5/MobileNet 等与 LLM 混合）；(d) shadow job 被调度时，REE 驱动通知 TEE 驱动接管 NPU → TZPC 配置 → 等待非安全 job 完成 → TZASC 授权 NPU DMA → GIC 配置中断路由 → MMIO 启动 job → NPU 三核执行 INT8 矩阵乘法 → completion 安全中断 → TEE 驱动归还 NPU。用户态 NPU 驱动的特权约束：仅映射 NPU MMIO 区域 + TZASC 限制 NPU 仅访问 job 执行上下文区域。(e) CPU 端：多线程 TA（shadow thread 机制），TA 线程配对 CA shadow thread，通过 smc 调度，同步原语在 TEE 中管理。(f) CMA allocation：多线程并行分配提升吞吐（4 线程 3.8 GB/s），prefill 阶段的分配延迟与 I/O 和 NPU 计算重叠。
    - **硬件架构**：同 baseline RK3588 平台。关键硬件配置变化：(a) TZASC 动态重配置——每次推理时扩展/收缩安全内存区域，而非静态预留；(b) TZPC 动态切换 NPU MMIO 归属——安全 job 执行期间归属 TEE，其余时间归属 REE；(c) GIC 动态重路由 NPU 中断——安全 job 执行期间路由到 TEE，其余时间路由到 REE；(d) NPU DMA 权限动态控制——仅安全 job 执行期间授权 NPU 访问安全内存。所有硬件配置切换开销总计占 TTFT 的 1.6%∼2.7% 和 decoding 的 2.3%∼5.7%。
  - 方法如何对应解决 Baseline 缺陷：
    1. **解决 Challenge #1（内存效率 vs TTFT 两难）**：流水线恢复将 CMA allocation + I/O + 解密与推理计算重叠，而非串行执行。优先级抢占式调度最小化流水线气泡（与理论下界差距 0.01%∼9.9%）。Partial parameter caching 进一步消除初始气泡：保留早期 operators 参数在安全内存中（first-in-last-out pattern 天然保持连续），每次推理从缓存的计算阶段恢复而非完整冷启动。CMA allocation 多线程化（3.8 GB/s）使其在 I/O 延迟下可被隐藏。最终 TTFT 降低 76.1%∼90.9%（vs strawman），同时保持动态内存扩展的内存效率。
    2. **解决 Challenge #2（NPU 不可用于 TEE 推理）**：Co-driver 设计使 TEE 可安全使用 NPU 而无需部署完整驱动。数据平面与控制平面分离后：(a) TEE 驱动仅 ~1K LoC（vs 完整驱动 60K+ LoC），最小化 TCB 增量；(b) NPU 世界切换无需控制平面重初始化，切换开销仅来自 TZASC/TZPC/GIC 硬件寄存器配置（~μs 级 vs 原 32ms 驱动重初始化）；(c) 统一的 REE 调度队列支持安全/非安全 NPU jobs 公平分时复用，NN 应用额外吞吐损失仅 ≤3.8%。NPU 加速使 prefill 速度提升至 12.5×，decoding 速度提升至 1.3×。


## On-device Semantic Selection Made Low Latency and Memory Efficient with Monolithic Forwarding

- baseline方法是什么？
  - Baseline：**HuggingFace Transformers 标准推理**——每个 query-candidate pair 作为独立 batch 分别执行完整前向传播得到 relevance score，对所有 N 个候选遍历后排序选出 top-K。两个变体：
    - **HF（in-memory）**：所有权重和中间张量在 GPU VRAM 中，内存占用与模型大小成正比
    - **HF Offload（disk offloading）**：使用 HuggingFace Accelerate 将所有 transformer 层 offload 到 SSD，执行前逐一加载。权重加载与计算完全串行（先全部加载，再计算），导致显著 I/O 延迟
  - Baseline 全栈执行例子（Qwen3-Reranker-0.6B，N=20 候选选 top-10=K，Apple M2，序列长度~500）：
    - **模型推理算法**：cross-encoder decoder-only transformer。每个 query-candidate pair 拼接为 `[CLS] q [SEP] d_i [SEP]`，独立通过 28 层 transformer（causal self-attention + FFN），最后 classifier head 取 last-token hidden state 输出标量 score。N=20 → 20 次完整前向 → 20×28 = 560 次 layer 执行。总计算量 = N × L × (28 layers × (self-attention + FFN))。复杂度 O(N × L² × D) for attention + O(N × L × D²) for FFN。
    - **Serving 框架**：HuggingFace Transformers v4.52.4 PyTorch backend。`model(**inputs)` 逐个或 mini-batch 处理 query-candidate pairs。HF Offload 使用 Accelerate `cpu_offload()`——每次前向后自动将 weights 移回 CPU，下次前向再从 CPU→GPU。论文未明确说明 batch 组织策略，但指出 vanilla 系统通常将输入拆分为多个 batch 来平衡计算和内存。
    - **编译框架**：PyTorch eager mode，无额外编译优化。论文未明确说明。
    - **Kernel 调度**：标准 PyTorch CUDA/MPS backend。Self-attention 使用 PyTorch native SDPA（scaled dot-product attention），FFN 使用 cuBLAS GEMM（GPU）/ MPS GEMM（Apple）。所有 kernel 在 GPU 上串行执行（单 stream），无 compute-I/O overlap。
    - **硬件架构**：Apple M2 SoC（统一内存 16 GiB）+ PCIe 4.0 SSD。M2 GPU 核心执行 attention/FFN GEMM。SSD 仅用于存储模型权重文件（HF Offload 场景）。所有中间张量（Q/K/V/proj、attention scores、FFN activations）在统一内存中分配，峰值内存 = 模型权重 + N × 中间张量。
  - Baseline 痛点：
    1. **Latency ∝ N**：每个候选需独立完整前向，总延迟 = N × T_single_pass。0.6B 模型选 top-5 from 20 候选：5,754 ms latency，reranker 占端到端 96.3%。
    2. **绝对分数计算冗余**：top-K 选择只需相对排名，baseline 却为每个候选计算绝对分数，浪费大量计算。排名信息在中间层已稳定但 baseline 仍执行全部 28 层。
    3. **无 I/O 重叠**：HF Offload 中权重加载与计算完全串行。当前层计算完成后才开始 SSD 加载下一层（加载延迟 ~ms 级，累积 28 层）。
    4. **Intermediate tensors 内存爆炸**：batch size × 序列长度 × hidden_dim 的中间张量（Q/K/V、attention scores、FFN outputs）与 batch size 线性增长。但 baseline 为避免 OOM 被迫使用小 batch 或无 batch，丧失了批处理效率。
    5. **Embedding table 内存占比高**：优化 Transformer 层权重后，embedding table 成为内存瓶颈（Qwen3-Reranker-0.6B: 296 MB embedding vs 60 MB active transformer layers，占 83%）。Baseline 加载完整 embedding table 却只使用 <7% 的 token。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法（PRISM 四大技术）：
    1. **Progressive Cluster Pruning（§4.1）**：利用 sequence-level sparsity（候选文档在中间层即形成稳定聚类，cluster 间相对排名不再变化），通过 K-Means 聚类 + CV dispersion check 实现三路路由（selected/dropped/deferred），仅对边界聚类中的 uncertain candidates 继续计算。
    2. **Overlapped Layer Streaming（§4.2）**：双进程架构（computation + I/O），仅保持两层权重在内存，当前层计算时并发 prefetch 下一层，释放后回收 buffer 用于再下一层。Monolithic batch 提供的充足计算窗口完全隐藏 I/O。
    3. **Chunked Execution（§4.3）**：将 monolithic batch 拆分为 chunks 逐 chunk 顺序执行，仅保留一个 chunk 的中间张量 + 全部 hidden states。支持 dynamic hidden states offloading（最多 3 chunks 在内存）。chunk size 动态计算以保证 GPU 饱和 + I/O overlap。
    4. **Embedding Table Caching（§4.4）**：LRU cache（10% vocab size）缓存激活 token 的 embedding weights，cache miss 时同步 SSD 读取。利用 token 稀疏性（<7% 词汇表被使用）和 Zipf 分布维持高命中率。
  - 论文方法全栈执行例子（同上 Qwen3-Reranker-0.6B, N=20→K=10, Apple M2, ~500 tokens/seq）：
    - **模型推理算法**：同 cross-encoder 模型架构。但通过 progressive cluster pruning，假设在 layer 9 CV > threshold 触发第一次剪枝（2 selected + 2 dropped → 16 deferred），layer 11 再次触发并满足终止条件（累计 10 selected → 直接返回）。实际执行：20 候选 × 9 layers（pre-pruning）+ 16 候选 × 2 layers（post-pruning）+ 提前终止。相比 baseline 的 20 × 28 = 560 layer-passes，实际执行约 20×9 + 16×2 = 212 layer-passes（**减少 62% layer 执行量**）。
    - **Serving 框架**：PRISM monolithic forwarding。所有 20 候选一次性输入，通过 embedding table cache（仅加载 ~10,240 unique token embeddings from 151,669 vocab, 6.75%）→ chunked execution（20 候选分 10 chunks × 2/chunk）→ overlapped layer streaming（仅 2 层权重在内存，其余从 SSD 流式加载并与计算重叠）。Computation process（处理 layer forward）+ I/O process（libuv 异步 SSD 读取）并行。
    - **编译框架**：论文未修改编译框架。同 baseline PyTorch eager mode。论文未明确说明。
    - **Kernel 调度**：同 baseline PyTorch CUDA/MPS backend。差异在于：
      - Chunked execution 将整 batch GEMM 替换为多次小 batch GEMM（chunk_size × L × D），但总计算量不变，仅增加少量 kernel launch overhead
      - I/O 进程通过 libuv 事件循环调度异步 SSD 读取，利用 CUDA MPS 使 computation 进程和 I/O 进程共享 GPU 无 context switch 开销
      - K-Means 聚类在 CPU 上执行（~1ms），不占用 GPU
      - Hidden states offloading 将 CPU↔GPU 数据传输与 GPU computation 重叠
    - **硬件架构**：同 baseline Apple M2 统一内存 + PCIe 4.0 SSD。关键区别：
      - 模型权重仅 2 层常驻内存（~30 MB vs ~1.2 GB 全模型），其余从 SSD 流式加载
      - 中间张量峰值 ∝ chunk_size × L × D（~2 × 500 × D）而非 N × L × D（20 × 500 × D），减少 ~10×
      - Embedding table 在内存中仅保留 10% vocab 的 LRU cache（~30 MB vs 296 MB 全量）
      - I/O 与计算完全重叠：SSD 读取被 GPU/CPU 计算窗口隐藏，无额外延迟
  - 对应解决 Baseline 缺陷：
    1. **Latency ∝ N** → Progressive cluster pruning 将逐候选独立计算变为 shared monolithic forward + 渐进式剪枝。剪枝后仅 boundary cluster 候选继续，实验显示 layer 执行量减少约 49%（ablation 中 progressive cluster pruning 单独贡献 49.0% latency reduction）。整体 end-to-end latency 减少 up to 89.2%。
    2. **绝对分数计算冗余** → 不再为每个候选计算精确绝对分数，而是利用 CV + K-Means 识别 cluster 间相对排名稳定性。只需确定候选属于 selected/dropped/deferred 三个集合之一，无需精确分数。Cluster γ 始终接近 1.0（18 datasets 验证），证实 cluster 间排名稳定→可安全剪枝。
    3. **无 I/O 重叠** → Overlapped layer streaming 通过 monolithic batch 创建的计算窗口（prefill-only, compute-heavy per layer）完全重叠 I/O。ablation 显示该技术单独贡献 57.8% 内存减少，仅引入 81 ms 额外延迟（因剪枝后计算窗口缩小）。嵌入 caching 后总体无延迟开销。
    4. **Intermediate tensors 内存爆炸** → Chunked execution 将峰值中间张量从 N×L×D 降至 chunk_size×L×D，同时保持足够的计算窗口用于 I/O 重叠。ablation 显示该技术将 monolithic batch 引入的 44.8% 内存增加降至 7.2%。Dynamic hidden states offloading 进一步限制到最多 3 chunks 在内存。
    5. **Embedding table 内存占比高** → Embedding table cache（10% vocab LRU）利用 token 稀疏性将 embedding 内存从 296 MB 降至 ~30 MB。ablation 显示该技术消除最后的内存瓶颈，使总峰值内存从 baseline 1,184 MB 降至 271 MB（减少 78.4%），额外延迟仅 4 ms。

## Efficient, VRAM-Constrained Cross-Lingual Model Inference on Client Devices

- baseline方法是什么？
  - Baseline：llama.cpp b6097，使用 `-ngl`（num-gpu-layers）手动指定将多少层放在 GPU，其余层在 CPU 执行。用户需手动尝试不同 -ngl 值来找合适配置。VLM baseline 额外包括 vLLM（在峰值 VRAM 下运行）。
  - Baseline 全栈执行例子（nemo8b f16, cli3, 4G VRAM, batch=1, 1K context）：
    - 模型推理算法：标准 transformer decode，prefill 产生 KV cache，decode 逐 token 自回归。f16 精度。
    - Serving 框架：llama.cpp 加载 GGUF 模型。用户通过 `-ngl N` 指定前 N 层在 GPU，其余在 CPU。每次 decode step 固定按此分派，无动态调整。
    - 编译框架：llama.cpp GGML backend，无额外编译优化。
    - Kernel 调度：GPU-resident layers 调 CUDA kernel（cuBLAS GEMM + FlashAttention），CPU-resident layers 调 CPU kernel（基于 GGML 的 OpenMP 并行）。权重和 KV cache 固定分布，无 runtime 重分配。CPU/GPU 间传输 KV cache 和中间激活。
    - 硬件架构：NVIDIA RTX 5090（32GB VRAM, Blackwell SM）+ AMD EPYC 16 cores + PCIe Gen5 (50 GBps)。所有 GPU kernel 在 SM 上执行，CPU kernel 在 EPYC 核心上执行，PCIe 传输 GPU-resident 层的输出到 CPU。
  - Baseline 痛点：
    1. 手动调优 (`-ngl`) 无法适应不同 VRAM budget、context length、batch size。
    2. 固定 layer assignment 忽略 token tier 的影响：prefill (many new tokens) 时 GPU compute 优势大，decode (1 new token) 时 PCIe 传输开销可能超过 GPU compute 收益。
    3. 无 CPU/GPU overlap：CPU compute 和 GPU weight streaming 串行执行。
    4. KV cache 随 context 增长，固定分配导致更小 VRAM budget 下 OOM。
    5. VLM vision encoder 权重(~2GB+)和 attention 中间张量(O(N²))在 VRAM 中常驻，导致高分辨率下 OOM。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：
    1. **Pipelined Sharding（Profile → Plan → Infer）**：自动化 CPU-GPU 调度。
       - Profile（安装时）：benchmark CPU/GPU kernel，构建 profile database。
       - Plan（模型加载时）：为每个 token tier 生成 GPU-only / Static / Dynamic 三种 plan，roofline + profile 选最优，写入 schedule lookup table。
       - Infer（推理时）：O(1) 查表调度，无 runtime overhead。
    2. **三种 Plan 覆盖不同场景**：GPU-only（高 new-token 数时用 GPU 算力隐藏 PCIe latency）、Static（VRAM 优先放 attention > KV cache > FFN > outputs）、Dynamic（Static + CPU compute 与 GPU weight streaming overlap）。
    3. **VLMOpt**：vision tensor offload + Tiled FlashAttention + serialized vision teardown，将 VLM VRAM 从 20+ GB 降至 2 GB。
  - 论文方法全栈执行例子（同上 nemo8b f16, cli3, 4G VRAM, batch=1, 1K context）：
    - 模型推理算法：标准 transformer decode，精度不变（lossless）。对于 1K context / batch=1，token tier 为 tier_1（1 new token），planner 选 Static plan。
    - Serving 框架：Pipelined Sharding 在 llama.cpp 中实现。Plan 阶段离线完成，Infer 阶段：查 schedule table → Static plan → layer assignment = {attention layers: GPU, FFN layers: CPU, output: CPU}（仅 VRAM 放 attention weights + KV cache + 必要中间结果）。switch plan 时同步 KV/RS cache。
    - 编译框架：论文未修改编译框架（同 baseline，论文未明确说明编译层面修改）。
    - Kernel 调度：Static plan 下，GPU 执行 attention kernel（FlashAttention），CPU 执行 FFN kernel（GGML GEMM）。Attention 输出作为中间结果通过 PCIe 传回 CPU，与下一层 CPU FFN kernel 流水线执行。Async copy backend 负责 GPU weight 按需 streaming（如切换 plan 时）。
    - 硬件架构：同 baseline。但通过 schedule table 使每个 token tier 的 kernel 分派匹配硬件特性（batch=1 时 attention 在 GPU 利用 tensor core，FFN 在 CPU 利用 16 核 EPYC 的算力 + 153.6 GBps 内存带宽，避免 PCIe 传输大权重）。
  - 对应解决 Baseline 缺陷：
    1. 手动调优 → 自动 Profile+Plan，用户仅指定 VRAM budget。
    2. 固定 layer assignment → token tier 自适应（prefill 用 GPU-only 发挥算力，decode 用 Static/Dynamic 减少 PCIe 传输）。
    3. 无 overlap → Dynamic plan 实现 CPU compute ⊗ GPU weight streaming overlap。
    4. KV cache OOM → Static plan 优先保证 KV cache 在 VRAM，context 增长时自动调整其他 layer 分配。
    5. VLM OOM → VLMOpt 三件套：vision offload（权重按需 stream）、Tiled FA（限制 attention 中间张量）、serialized teardown（peak VRAM = max(vision, language) 而非 vision + language）。

## IntAttention Fully Integer Attention Pipeline for Edge LLM Inference

- baseline方法是什么？
  - Baseline：**INT8 Quant-Only Attention**——对 Q、K、V 做 INT8 量化，QK^T 和 PV 的矩阵乘法使用 INT8 GEMM（S8×S8→S32）加速，但中间的 softmax 路径**仍为浮点**：S32 累加器先 dequantize 至 FP32 → FP32 softmax（exp + row-sum + division）→ requantize 回 S8 供下游 PV INT8 GEMM。即 `S8×S8→S32 → dequant(S32→FP32) → FP32 softmax → quant(FP32→S8) → S8×S8→S32`。
  - Baseline 全栈执行例子（LLaMA-3.2-1B 单层 attention，RK3588S2 ARM CPU，序列长度 L=1024，head_dim d=128）：
    - **模型推理算法**：标准 scaled dot-product attention。QK^T GEMM 使用 ACL INT8 kernel（S8×S8，S32 累加），PV GEMM 同。Softmax 路径：ACL `NEDequantizationLayer`（S32→FP32）→ `NEExpLayer`（FP32 exp）→ `NEPoolingLayer`（FP32 row-sum + division）→ `NEQuantizationLayer`（FP32→S8）。这 4 个 kernel 之间产生多次内存往返（至少 3 次：dequant 写 FP32 buf → softmax 读写 FP32 buf → quant 读 FP32 buf 写 S8 buf）。
    - **Serving 框架**：论文在 ACL 的 standalone 模式下评估（无 Serving 框架），仅关注单层 attention kernel 的延迟与能耗。论文未明确说明 Serving 层面修改。
    - **编译框架**：ACL 通过 scons + clang++ 构建为静态库，无额外编译框架修改。论文未明确说明编译层面修改。
    - **Kernel 调度**：RK3588S2 的 ARM Cortex-A76/A55 核心上，ACL 将 INT8 GEMM 调至 NEON SIMD（ARMv8.2-A dotprod 指令 `sdot`，每周期 4×S8×S8→S32 乘加）。Softmax 路径使用标量 FP32 exp + 除法，无法利用 NEON 向量化（因各 lane 需要行内归约）。dequant/requant 为逐元素 NEON 向量化操作但引入额外数据搬移。
    - **硬件架构**：RK3588S2 的 ARM Cortex-A76 big core（4×A76@2.4GHz, NEON 128-bit SIMD）+ Cortex-A55 LITTLE core（4×A55@1.8GHz）+ 共享 L2/L3 cache。PKM（Power/Kernel/Memory）视角：softmax 的浮点 exp 在每个核心上为标量执行（~12-20 cycles/exp），dequant+requant 额外消耗 NEON 带宽（每元素 2×load + 1×store FP32 + 1×load + 1×store S8）。
  - Baseline 痛点：
    1. **Softmax 成为主要瓶颈**：INT8 GEMM 将矩阵乘加速后，softmax 路径（dequant → exp → sum → div → requant）在注意力延迟中占比 ≤65%（论文实测，RK3588S2 上）。原因：INT8 GEMM 速度大幅提升（>2× vs FP16），但 softmax 仍为浮点且无法受益于 INT8 加速。
    2. **数据类型转换开销巨大**：S32 → FP32 dequant 和 FP32 → S8 requant 产生大量转换指令，在 ARM CPU 上尤其昂贵（每条转换指令涉及精度截断 + 饱和处理）。
    3. **内存带宽浪费**：dequant（输出 FP32 buffer）→ softmax（读写 FP32 buffer）→ requant（读 FP32、写 S8）形成 3 次 L1/L2 cache round-trip，对 memory-bound 的边缘设备极为不利。
    4. **FP32 exp 计算昂贵**：ARM Cortex-A76 NEON 无原生 FP32 exp 指令，需走标量 math library（`expf`），每个 logit 约 12-20 cycles，连续 1024 个 logit 的 exp 计算成为显著延迟来源。
    5. **INT8 表示浪费**：P 矩阵（softmax 输出）的 INT8 量化仅使用 [-128, 127] 中约一半范围（概率值天然非负），精度浪费。而 UINT8 [0, 255] 恰好匹配概率域。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法（IndexSoftmax + IntAttention Pipeline）：
    1. **Sparsity-aware Clipping**：利用 softmax 中远离 max 的 logit 贡献可忽略，裁剪 logits 至 [0, c=6.6] 范围，仅需处理稀疏有效 logits。裁剪使指数域有界，为 LUT 奠定基础。
    2. **32-entry LUT Exponential**：指数域 [0, c] 有界后，使用紧凑 32-entry UINT8 查表替代 FP32 exp。LUT 总大小仅 32 bytes，可完全放入 NEON 寄存器，查表通过 `TBL` 指令实现向量化并行。
    3. **UINT8 Integer Normalization**：将 softmax 概率 P 输出为 UINT8（[0, 255]），利用全部 256 个值（vs INT8 的 128 个有效非负值），获得 4× 精度。归一化使用定点乘除（`(prob * 255 + sum/2) / sum`）而非浮点除法。
  - 论文方法全栈执行例子（同上 LLaMA-3.2-1B L=1024 d=128, RK3588S2 ARM CPU）：
    - **模型推理算法**：标准 scaled dot-product attention。QK^T S8×S8→S32（ACL INT8 GEMM）→ IndexSoftmax S32→U8（clipping→LUT→integer norm）→ PV S8×S8→S32（ACL INT8 GEMM）。全程无浮点指令、无类型转换。P 矩阵为 UINT8，PV GEMM 等同于 S8×S8（将 U8 reinterpret 为 S8 的 non-negative 子集）。
    - **Serving 框架**：同 baseline，论文未涉及 Serving 层面修改（论文未明确说明）。
    - **编译框架**：同 baseline，论文未修改编译框架（论文未明确说明）。核心改动在 ACL kernel 层。
    - **Kernel 调度**：IndexSoftmax 在 ACL 中实现为单个融合 kernel（`NEIndexSoftmax`），消除了 baseline 中 dequant→softmax→requant 的多次 kernel 调用和缓存往返。LUT 驻留在 NEON 寄存器中避免 memory load。关键 NEON 指令：
      - `sdot`：S8×S8→S32 GEMM 乘加
      - `tbl`：并行 LUT 查表（每 16 个 UINT8 lane 独立查表）
      - `umull` + `ushr`：定点归一化的乘法 + 移位除法
    - **硬件架构**：同 baseline RK3588S2 ARM CPU。但 IndexSoftmax 通过消除类型转换、减少缓存往返 2/3、替换标量 exp 为 SIMD LUT，在相同硬件上获得：
      - 3.7× attention latency speedup vs FP16
      - 2.0× vs INT8 Quant-Only（即仅替换 softmax 路径的增益）
      - 61% energy reduction vs FP16
  - 对应解决 Baseline 缺陷：
    1. **Softmax 成为瓶颈（≤65%）** → IndexSoftmax 将 softmax 路径从浮点替换为全整数，消除 dequant/requant 转换，延迟从主要瓶颈降至可忽略。实测 attention 整体加速 2.0×（vs Quant-Only）——softmax 路径本身的加速比远超 2×，但受 Amdahl 定律限制（softmax 占比 65%，加速此部分的理论上限约 2.86×）。
    2. **类型转换开销巨大** → IntAttention 管线全程保持整数格式（S32→U8→S32），无 S32↔FP32 或 FP32↔S8 转换。消除每 attention 层 ~4Ld 次类型转换指令。
    3. **内存带宽浪费（3 次缓存往返）** → IndexSoftmax 将 softmax 实现为单个融合 kernel，仅需 1 次缓存往返（读 S32 buf → NEON 寄存器内 clip+LUT+normalize → 写 U8 buf），减少 2/3 的中间 buffer 读写。
    4. **FP32 exp 昂贵（12-20 cycles/element）** → LUT 查表通过 NEON `TBL` 指令在单周期内完成 16 个元素的并行查表（每 element <1 cycle），替换标量 `expf`。代价是精度降低（LUT 量化误差），但通过 UINT8 细粒度和 c/b 参数 sweep 控制在可接受范围（PPL 退化 <0.5，语言理解平均精度退化 <0.2%）。
    5. **INT8 浪费一半表示范围** → UINT8 P 矩阵充分利用 [0, 255] 全部 256 个值，在相同 32B 表预算下分辨率 4×（CosSim 0.999081 vs INT8 0.996612）。视觉任务中精度损失更小（Top-1 退化<0.3% vs FP16）。

## Federated Fine-Tuning of Sparsely-Activated Large Language Models on Resource-Constrained Devices

- baseline方法是什么？
  - Baseline 方法有三类：
    1. **FMD (Federated MoE fine-tuning with offloading)**：将 inactive experts 从 GPU offload 到 CPU RAM，需要时动态加载回 GPU。所有 experts 参与 forward/backward，保证精度但引入大量 CPU-GPU I/O 延迟。
    2. **FMQ (Federated MoE fine-tuning with quantization)**：所有 expert 参数从 FP32 量化到 INT4，使 participant 能在本地加载完整 MoE 模型。但量化误差在 backprop 中累积导致收敛不稳定。
    3. **FMES (Federated MoE fine-tuning with expert selection)**：按 expert activation frequency 选出高频 activated experts 进行 fine-tuning，丢弃低频 experts（类似 FedMoE [50] 的做法）。
  - Baseline 全栈执行例子（以 LLaMA-MoE 6.7B 在 NVIDIA L20 48GB 上，participant 本地数据 D_i 为例）：
    - **模型推理算法**：MoE transformer decoder。每 token 经 gating network（softmax top-k）选择 activated experts。FMD 保持完整 MoE（32 layers × 16 experts），FMQ 将所有 expert 参数量化至 INT4，FMES 只保留 top-K 高频 experts 并丢弃其余。
    - **系统框架**：parameter-server-based federated learning。Server 下发全局模型 → participant 本地 fine-tuning → 上传 expert updates → FedAvg aggregation。FMD 使用 PyTorch + 自定义 offloading logic 管理 GPU↔CPU 数据传输。FMQ 使用标准 quantization library。FMES 使用 activation frequency counting + expert filtering。
    - **编译框架**：论文未明确说明（使用 PyTorch eager mode，无额外编译优化）。
    - **Kernel 调度**：标准的 PyTorch CUDA backend。MoE FFN experts 的 GEMM 由 cuBLAS 执行，gating softmax/top-k 由 PyTorch 原生 CUDA kernel 执行。FMD 中 expert offloading 触发 cudaMemcpy 在 GPU↔CPU 之间传输 expert 参数张量（32 layers × 16 experts × ~85M params/expert for LLaMA-MoE ≈ 每次传输数 GB）。论文未明确说明 offload 策略的具体 kernel 调度细节。
    - **硬件架构**：NVIDIA L20 GPU（48GB VRAM, Ada Lovelace SM）+ host CPU + PCIe Gen4 互联。FMD 的 expert offloading 受 PCIe bandwidth 瓶颈限制（每轮 fine-tuning 需多次 CPU↔GPU expert 参数传输）。
  - Baseline 痛点：
    1. **FMD**：expert offloading 的 CPU↔GPU 数据传输延迟巨大，严重拖慢 fine-tuning 速度。PCIe bandwidth 成为瓶颈。
    2. **FMQ**：INT4 量化误差在 backprop 梯度计算中累积放大，导致训练不稳定、收敛慢甚至不收敛（Figure 10/11 中 FMQ 曲线震荡）。
    3. **FMES**：丢弃"低频" non-tuning experts 严重损害模型精度（Figure 3a：discarding non-tuning experts 导致 ROUGE 显著下降）。原因是 token 路径中 non-tuning experts 的输出错误会逐层传播累积（Figure 3b, Figure 8）。此外，仅用 activation frequency 选 expert 不准确——部分低激活频率 expert 处理的 token 具有高 attention score，对模型输出影响巨大（Figure 9）。
    4. **通用痛点**：expert activation pattern 在 training 过程中变化（Figure 6a），静态 profiling 随时间失效；expert role assignment 在 participant 异构计算资源下难以优化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法（FLUX 三大模块）：
    1. **Quantization-based Local Profiling + Stale Profiling（§4）**：用 INT4 量化 MoE 模型在本地估计 expert activation frequency、data subset D_i^e 和 attention scores ā_e，替代运行全精度模型的开销。通过 stale profiling 使 profiling 与 parameter aggregation 并行，隐藏 profiling 延迟。
    2. **Adaptive Layer-Aware Expert Merging（§5）**：不丢弃 non-tuning experts，而是按层自适应分配 merge budget（浅层+高 variance 层给更多 budget），PCA+K-Means 聚类相似 expert 后按 attention×frequency 加权合并。
    3. **Dynamic Expert Role Assignment（§6）**：定义 expert utility u_i^e = |D_i^e|√(avg gradient)，用 exploration-exploitation（动态 ε）选择 tuning experts；exploration experts 用 forward-only gradient estimation 省 backprop 开销。

  - 论文方法全栈执行例子（同上 LLaMA-MoE 6.7B, L20 48GB, participant i）：
    - **模型推理算法**：MoE transformer decoder，expert 分三类：tuning（FP32 完整更新）、exploration（FP32 forward-only 梯度估计）、merged non-tuning（frozen，加权合并后的单一 expert 参与 forward）。每次 forward 仍走 gate→top-k→expert compute 流程，但 expert 数量减少为 B_i^{tune} + B_i^{non}(l)。tuning experts 的 training 使用 profiling 得到的 D_i^e（仅用流经该 expert 的数据），提升数据效率。
    - **系统框架**：parameter-server-based FL + FLUX 定制模块。Flux.moe.customized_moe(model, exps_config) 构建每层不同 expert 数的定制 MoE。Flux.moe.load_model() 从原始 checkpoint 分离加载 expert 参数和非 expert 参数。Gate re-routing 在 merging 后更新 gating mapping。Stale profiling 使 profiling 与 server aggregation 并行（Figure 7b），round time 减少 ~28.2%。FedAvg 仅聚合 tuning experts 的 updates。
    - **编译框架**：论文未明确说明（同 baseline，PyTorch eager mode）。论文提到支持集成 Adapter 和 LoRA 等 PEFT 方法。
    - **Kernel 调度**：同 baseline PyTorch CUDA backend。merged non-tuning experts 减少 expert GEMM 调用次数（每层从 16 experts 降至 B_i^{non}(l) 个 merged experts + B_i^{tune} 个 tuning experts），降低 kernel launch overhead。exploration experts 的 forward-only gradient estimation 使用小扰动（ξ ~ N(0,σ²)）加法 + 两次 forward pass 差商近似梯度，避免 backprop kernel（省去 grad GEMM 和 grad accumulation kernel）。
    - **硬件架构**：同 baseline NVIDIA L20 + PCIe。FLUX 通过减少 GPU memory 占用（合并 non-tuning experts）使 consumer-grade GPU（48GB）能 fine-tune 原本需要更大显存的 MoE 模型。round time 中 FLUX 额外开销约 5%（Figure 20）。

  - 对应解决 Baseline 缺陷：
    1. **FMD offloading 延迟巨大** → FLUX 不 offload，而是合并 non-tuning experts + 选择 tuning experts，使模型 fit 进 GPU memory，消除 CPU↔GPU 传输开销。4.75× time-to-accuracy speedup。
    2. **FMQ 量化误差导致收敛不稳定** → FLUX 仅用量化模型做 profiling（forward only），fine-tuning 本身使用 FP32，避免 backprop 中量化误差累积。profiling 估计误差约 11.01%（4-bit），不影响 fine-tuning 精度。
    3. **FMES 丢弃 non-tuning experts 损害精度** → FLUX 保留并合并 non-tuning experts（加权合并保留关键信息），output error 相比 single expert 减少 65.6%，相比 uniform layer size 减少 47.6%（GSM8K）。最终精度与 FMD（full model）接近（Table 2：FLUX ROUGE-L 0.527 vs FMD 0.528 on Dolly）。
    4. **仅用 activation frequency 选 expert 不准确** → FLUX 定义 expert utility 结合 gradient magnitude + data utilization（公式 3），merging 权重结合 attention score + activation frequency（公式 2），更准确反映 expert 重要性。Figure 17 显示 Att.+Frq. merging 比纯频率加权减少 19.2% output error。
    5. **静态 profiling 随时间失效** → stale profiling 机制：profiling 与 aggregation 并行，每轮更新 profile 但隐藏延迟（Figure 14：误差增长 <2%，round time 减少 28.2%）。
    6. **异构 participant 下 expert role assignment 难优化** → parameter server 求解全局优化问题（公式 4）+ exploration-exploitation（动态 ε，early stage 多探索，later stage 多利用），Figure 19 显示动态 ε 比固定 ε=0.3 或 0.7 更快收敛。

## OpenJarvis

- baseline方法是什么？
  - Baseline：**云端一体式个人AI栈（Monolithic Cloud Personal AI Stack）**——以 OpenClaw 和 Hermes Agent 为代表的现有个人AI框架。Agent prompts、tool descriptions、memory configuration、Engine/runtime settings 和模型特定输出期望在框架内部与目标云端模型（如 Claude Opus 4.6）紧密耦合，各组件非独立可配置。用户替换本地模型时，所有耦合组件同时失效。
  - Baseline 全栈执行例子（OpenClaw + Claude Opus 4.6 执行 PinchBench agent 任务，云端推理）：
    - **模型推理算法**：Claude Opus 4.6（前沿云端模型，架构未公开）。推理在 Anthropic API 服务器执行。Extended-thinking + tool-use 能力。
    - **系统框架（Serving/Agent）**：OpenClaw 框架内置 Agent prompt（针对 Claude 调优的 system prompt + tool descriptions）、内置 tool set（web_search, file_read, code_interpreter 等）、内置 memory（可能为 SQLite 或向量存储）、内置 runtime settings（timeout, retry, max_turns 等）。所有组件作为框架的 baked-in 配置，切换为 Qwen3.5-9B 时 prompt 格式、tool 调用约定、输出解析逻辑全部断裂。
    - **编译框架**：论文未明确说明。云端推理的编译栈由云厂商内部管理。
    - **Kernel 调度**：论文未明确说明。云端 GPU kernel 由云厂商的 serving 基础设施管理。
    - **硬件架构**：云端数据中心 GPU/TPU 集群。论文未明确说明具体硬件。
  - Baseline 痛点：
    1. **替换模型导致整栈崩溃（Substitution breaks monolithic stacks）**：在 OpenClaw 中将 Claude Opus 4.6 替换为 Qwen3.5-9B，保持其余配置固定，PinchBench 精度从 96.0% 降至 62.3%（−33.7 pp），GAIA 从 58.0% 降至 19.2%（−38.8 pp）。原因：Agent prompts、tool descriptions、memory configuration、runtime settings 均与目标云端模型共同设计，替换模型破坏所有耦合组件。
    2. **单原语优化饱和（Single-primitive optimization plateaus）**：现有框架中仅 prompts 可无源码修改调整，其余组件 baked-in。应用 SOTA prompt optimizer（GEPA, DSPy）仅闭合本地-云端差距 5 pp——优化单一原语无法实现模型/prompts/tools/runtime 之间的协调变更。
    3. **无联合评估框架**：现有评测工具（Zeus, AI Energy Score, MLCommons）测量单一原语的单一维度（精度或能耗或延迟）。没有框架在完整的五原语组合上联合评估 accuracy、energy、latency、power 和 cost。
    4. **本地模型局限于琐碎任务**：当前本地模型仅用于 tone adjustment、text completion 等简单任务，无法胜任 agentic personal AI 场景。
    5. **云端推理的隐私/成本/离线限制**：每查询需传输敏感个人数据到第三方服务器，年费数千美元 API 订阅，需要网络连接，无模型所有权。云端推理每 token 能耗比本地执行高数个数量级。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法（OPENJARVIS 两大核心）：
    1. **Spec 抽象（五原语架构）**：将个人AI系统分解为五个类型化原语——Intelligence（模型架构+权重）、Engine（推理运行时+硬件路径+batching+量化+cache）、Agents（推理循环+prompts+tool-use policy）、Tools & Memory（外部接口+检索+持久化用户状态）、Learning（优化器，从 traces 更新 spec）——每个原语有类型化接口和注册表。五个原语通过 *spec*（类型化配置对象，TOML 序列化）组合，暴露为单一可优化对象。
    2. **LLM-guided spec search**：本地-云端协作搜索算法。云端前沿模型在搜索时诊断失败痕迹→提出跨 Intelligence/Engine/Agents/Tools & Memory 的协调编辑→held-out gate 仅接受非退化编辑→推理时 spec 完全在本地执行。
  - 论文方法全栈执行例子（OPENJARVIS spec + Qwen3.5-9B, Mac Mini M4 24 GB, PinchBench agent 任务，搜索优化后推理）：
    - **模型推理算法**：Qwen3.5-9B Dense FP16。若 LLM-guided spec search 触发了 Intelligence edit（如 LoRA fine-tuning on teacher-generated SFT pairs 或 GRPO with composite reward），则使用更新后的 LoRA adapter 权重。推理在本地 Mac Mini M4 GPU + Neural Engine 上执行。prefill ~89,912 tokens/s, decode ~9.5 ms/tok。
    - **系统框架（Serving/Agent）**：OPENJARVIS spec 显式配置各原语——Engine=Ollama（batch size 1, KV-cache 默认）、Agent=ReAct loop（max 10 turns, tools=think+calc+web_search, system prompt 已由 LLM-guided spec search 针对 Qwen3.5-9B 优化）、Tools=SQLite FTS5 memory + web_search (Tavily API ~\$0.005/query) + code_interpreter（Docker sandbox）。Agent prompt、tool descriptions 和 tool set 均已由搜索阶段针对本地模型重写（例如：tool description 从 Claude 风格的冗长自然语言改写为 Qwen3.5-9B 更易理解的简洁结构化格式）。
    - **编译框架**：论文未修改编译框架。使用 PyTorch eager mode via Ollama。论文未明确说明。
    - **Kernel 调度**：论文未修改 GPU kernel。使用 Ollama 内置的 llama.cpp 后端（Metal/CUDA backend 根据平台自动选择）。论文未明确说明。
    - **硬件架构**：Apple M4（10-core CPU + 10-core GPU + 16-core Neural Engine, 24 GB unified memory）。模型权重加载至 unified memory。GPU 核心执行 attention/FFN GEMM。Neural Engine 可用于特定算子加速。
  - 对比 Baseline 如何解决缺陷：
    1. **替换模型整栈崩溃** → **Spec 显式化所有耦合组件为独立可编辑字段**：Engine、Agent、Tool 字段可独立于 Intelligence 重定向。Table 1 显示当 Intelligence 固定为 Qwen3.5-9B 时，仅重定向 Engine/Agent/Tools 即可恢复 PinchBench 上 77% 的下降（−33.7→−7.6 pp）、GAIA 上 56-57% 的下降（−38.8→−16.5 pp）。
    2. **单原语优化饱和** → **LLM-guided spec search 跨四原语联合优化**：单次 proposal 可同时改写 tool description + 调整 prompt + 切换 Engine + 触发 LoRA 训练。四原语 joint search 比最佳单原语高 5.5–16.5 pp（Figure 8 middle）。编辑分布（Table 10）显示无单一原语达到全搜索上限——有用干预取决于 failure mode，且许多 failure 需要跨多原语协调变更。
    3. **无联合评估框架** → **Spec 级 instrumented wrapper**：每查询自动记录 accuracy、energy（J）、latency（s）、power（W）、dollar cost（\$），暴露本地-云端 Pareto 前沿。本地 spec 匹配或超过云端精度于 4/8 benchmarks，同时边际 API 成本 ~800× 更低、端到端延迟 ~4× 更低。
    4. **本地模型局限于琐碎任务** → **搜索优化使手机/笔记本级模型成为 viable personal AI 基底**：Nemotron-Nano-4B 经搜索从 11.3%→42.8%（+31.5 pp），Qwen3.5-4B 从 34.8%→57.6%（+22.9 pp）。搜索优化后的 Qwen3.5-9B 达到 PinchBench 100%、LiveCodeBench 83%、LiveResearchBench 91%。
    5. **云端推理的隐私/成本/离线限制** → **搜索时云端、推理时本地的架构分工**：搜索时云端教师仅接收脱敏 traces，推理时 spec 零云端调用、零边际 API 费。教师搜索成本 \$15.6/benchmark 一次性，100 queries/day 部署 6 个月内摊销至 <\$0.001/query——比直接使用 Claude Opus 4.6 便宜 ~10×。本地推理无网络依赖、无第三方数据传输、用户拥有模型所有权。

## Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

- baseline方法是什么？
  - Baseline：（传统移动端 LLM 推理）单 token 逐次自回归解码，每次仅处理 1 个 token 的 GEMM（退化为 GEMV），NPU 的矩阵单元（HMX）利用率极低。同时，现有 NPU-based 系统（如 PowerServe、llm.npu、ExecuTorch）均依赖闭源 QNN SDK，QNN 仅支持 per-tensor/per-channel 粗粒度量化，无法适应现代 LLM 必需的细粒度分组量化（per-group quantization），导致量化模型在推理任务上精度严重下降（MATH500: 2.1% vs AutoAWQ: 15.9%）。此外，QNN 的静态固定形状计算图无法支持 test-time scaling 所需的动态 batch size 调整。
  - Baseline 全栈执行例子（Qwen2.5-1.5B FP16，OnePlus 12 Snapdragon 8 Gen 3，单 token decode）：
    - **模型推理算法**：标准 transformer decoder 自回归生成。W4A16 细粒度分组量化（AWQ），每 32 个权重共享一个 FP16 scale。推理时动态反量化 + FP16 GEMM。Test-time scaling 未启用（batch=1）。
    - **Serving 框架**：QNN（Qualcomm AI Engine Direct）。QNN 通过 ONNX 中间表示将模型编译为静态计算图 → QNN backend 将其转换为 HTP（Hexagon Tensor Processor）可执行指令。QNN 仅支持 per-tensor/per-channel 量化（不支持 per-group），且计算图固定（batch size 固定），无法适应 test-time scaling 的动态批处理需求。PowerServe [2] 等开源系统基于 QNN 实现。
    - **编译框架**：QNN 的封闭编译栈。QNN Converter 将 ONNX → QNN IR → HTP binary。开发者无法自定义底层 kernel（HMX 指令未公开），仅能通过 QNN API 执行标准算子。论文未明确说明。
    - **Kernel 调度**：HMX 矩阵单元执行 FP16/INT8 GEMM（通过 QNN 的 closed-source kernel library），HVX 向量单元执行其余算子。但 QNN 下 HVX 的反量化 + Softmax 等操作为通用实现，未经针对 LLM 工作负载优化。Decode 阶段 batch=1 → HMX 处理的 activation tile 为 [1,32]（仅 1/32 行有效），HMX 利用率极低，大量矩阵算力闲置。
    - **硬件架构**：Qualcomm Snapdragon 8 Gen 3（Hexagon V75 NPU）。HMX（1-2 个矩阵单元，~12 TFLOPS FP16）+ HVX（4-6 个向量单元，每 thread ~32.9 GFLOPS）+ 8 MiB TCM + 1 MiB L2 cache。DDR→TCM 通过 DMA（>60 GB/s），HVX 直接访存带宽 ~26 GB/s。QNN 使用 DMA+HMX 流水线执行 GEMM，HVX 仅处理非矩阵算子。

  - Baseline 痛点：
    1. **HMX 算力大量闲置**：Decode 阶段 activation shape=[1, hidden_dim]，HMX tile 为 32×32，仅 1/32 行有效。HMX ~12 TFLOPS 的算力绝大部分被浪费。
    2. **QNN 的粗粒度量化导致严重精度损失**：QNN 仅支持 per-tensor/per-channel 量化，无法适应现代 LLM 的 per-group 细粒度量化需求。实测 Llama3.2-1B QNN W4A16：MATH500 2.1%（vs AutoAWQ 15.9%）、GSM8K 3.4%（vs 32.6%）、Wiki PPL 28.99（vs 19.42）。在数学推理等 test-time scaling 的目标任务上精度完全不达标。
    3. **HVX 通用算力与 HMX 矩阵算力差距悬殊**：HVX 单 thread FP16 GEMM ~32.9 GFLOPS vs HMX ~12032 GFLOPS（差 366×）。HVX 内存带宽 ~26 GB/s vs DMA ~60 GB/s（差 2.3×）。非矩阵算子（Softmax、反量化）在 HVX 上成为瓶颈。
    4. **QNN 静态计算图无法支持 test-time scaling**：QNN 编译后 batch size 固定，test-time scaling 需要在 decode 阶段动态调整 batch size（Best-of-N/Beam Search 的 generation budget 可变），QNN 无法支持。
    5. **HVX 宽向量寄存器与细粒度量化组不匹配**：分组量化（group_size=32）的 INT4 量化数据仅 16 bytes，远小于 HVX 128-byte 向量寄存器。逐组反量化需多次内存操作和寄存器合并指令，造成内存带宽浪费和计算开销。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法（三大核心技术）：
    1. **Test-Time Scaling on NPU**：利用 NPU 在常规 decode 中浪费的 HMX 算力，通过并行采样增大 batch size 填充 HMX activation tile，在几乎不增加 decode 延迟的情况下提升模型生成质量（Best-of-N 和 Beam Search）。
    2. **Hardware-aware Tile Quantization Scheme**：
       - **Tile-Group Quantization**：量化前将权重预先排列为 HMX tile 内存布局（32×32 tile 级 column-major + 每两行 shuffle），再在 tile 内做 group quantization（group_size=32 → 2×16 tiles）。反量化后权重连续写入 TCM，消除 scatter 操作。
       - **Group Coalesce for Wide Vector**：将 8 个量化组合并为 super-group（256 个 INT4 = 128 bytes = 1 HVX 寄存器），重新排列 INT4 值和 scale 值。
    3. **LUT-Based Computations**：
       - **LUT Softmax**：利用 Safe Softmax（exp 输入 ≤0）→ 仅需 32768-entry FP16 LUT（64 KiB TCM），用 HVX vgather 指令（64 elements/instruction）替换显式 exp。
       - **LUT Dequantization**：HVX vlut16 指令通过查表直接将 INT4→FP16（含 qfloat 内部格式），scale 广播也用 vlut16。

  - 论文方法全栈执行例子（同上 Qwen2.5-1.5B IQ4_NL+Q8_0，OnePlus 12，Best-of-N batch=8）：
    - **模型推理算法**：标准 transformer decoder。W4A16 细粒度分组量化（IQ4_NL 4.5 BPW for attention/FFN up/gate，Q8_0 8.5 BPW for FFN down）。权重已在离线阶段排列为 HMX tile 布局 + super-group coalesce。Decode 阶段 batch=8（8 条并行采样路径），HMX 处理的 activation tile 有效行数从 1/32 → 8/32。
    - **Serving 框架**：llama.cpp NPU backend（不依赖 QNN）。CPU 侧 llama.cpp 负责 tokenizer、embedding、lm_head、sampler、PRM/ORM scorer；NPU 侧 htp-ops-lib 负责 transformer layers 的 GEMM + Attention。CPU-NPU 间通过 rpcmem shared memory 零拷贝通信。NPU 侧轮询接收计算请求。Batch size 可动态调整（启动时 `--batch-size` 参数）。
    - **编译框架**：Hexagon SDK v6.0.0.2 LLVM toolchain。llama.cpp NPU backend + htp-ops-lib 为 C/C++ + inline assembly。不经过 QNN 编译栈，完全自定义 NPU 算子。论文未明确说明 LLVM 编译优化细节。
    - **Kernel 调度**：
      - **Dequantization GEMM**：HVX vlut16 查表反量化（INT4→FP16） + HMX FP16 GEMM。反量化后 weights 连续写入 TCM（HMX tile 布局），无 scatter。HVX 128-byte 寄存器满载（super-group coalesce）。
      - **FP16 FlashAttention**：HMX 执行 QK^T 和 PV 矩阵乘（FP16，FP32 accum），HVX 执行 LUT Softmax（vgather 查表）、rowmax、rowsum。Softmax 延迟降至非瓶颈。
      - **NPU 线程管理**：NPU 侧含计算线程池，支持异步执行。CPU 侧提交 request 后不阻塞，继续处理上一轮 sampler 结果。
    - **硬件架构**：同 baseline Snapdragon 8 Gen 3 Hexagon V75。关键区别：
      - HMX 利用率提升（batch=8 填充 tile 行），decode 吞吐显著增加
      - HVX 被高效利用：vlut16 反量化（替代 unmask+unpack+convert）、vgather LUT Softmax（替代标量多项式 exp）
      - TCM 中 64 KiB LUT 常驻（exp table），占 TCM 仅 0.8%
      - DMA 用于大块正则数据传输（权重、KV cache），HVX scatter/gather 仅用于必要的小块不规则访问
      - 功耗仍控制在 <5W 整机（1.5B）

  - 对应解决 Baseline 缺陷：
    1. **HMX 算力大量闲置** → Test-time scaling 通过增大 batch size 填充 HMX activation tile，将闲置算力转化为生成质量提升。Batch=8 时 decode 吞吐相比 batch=1 显著提升（见图 11），而理论延迟基本不变（HMX 计算时间不随 batch 增长）。实测与更大模型对比：Qwen2.5-1.5B + Best-of-N（batch=8）精度超越 3B baseline；2.5-3B + Best-of-N 精度超越 7B baseline。
    2. **QNN 粗粒度量化精度损失** → 不依赖 QNN，自实现 W4A16 per-group 量化（IQ4_NL + Q8_0 hybrid），通过 tile-group quantization 和 LUT 反量化实现高效运行时。精度与 AutoAWQ 可比（Tile group vs Common group: WinoGrande 62.6 vs 63.3, MMLU 35.5 vs 35.3, Wiki PPL 10.2 vs 10.2），远优于 QNN per-channel。
    3. **HVX 通用算力与 HMX 差距悬殊** → LUT-based 计算大幅减轻 HVX 负担：vlut16 反量化 1 指令完成 INT4→FP16 + scale 广播（vs 传统 3-4 指令序列），vgather LUT Softmax 替换多项式 exp（减少指令数 + 消除序列依赖）。Ablation 显示 Dequantization GEMM 加速 9.65-19.04× vs baseline scatter 方案，Softmax 加速 1.26-2.19× vs FP32 exp。
    4. **QNN 静态计算图无法动态调整 batch size** → llama.cpp NPU backend 完全绕过 QNN，在 CPU 侧控制 batch size，动态传递给 NPU。NPU kernel（GEMM、FlashAttention）天然支持可变 batch size（通过 tile 分块策略）。
    5. **HVX 宽向量与细粒度组不匹配** → Super-group coalesce 将 8 个量化组合并，256 INT4 值精确填满 128-byte HVX 寄存器，消除逐组处理的内存和指令开销。Ablation 中 group-coalesce + HMX layout vs 仅 HMX layout 加速 1.82-3.45×（图 15）。


- baseline方法是什么？
  - Baseline：**默认移动端 DVFS Governor 独立调度**——Android/Pixel 7 系统内置的 CPU governor（sched_pelt/EAS，基于利用率按需调频）、GPU governor（quickstep，基于 GPU 负载调频）、Memory governor（interactive，基于内存带宽使用调频）各自独立运行，互不感知对方状态。每个 governor 仅根据本地指标（如 CPU utilization、GPU busy percentage）做出频率决策，无跨资源协调。
  - Baseline 全栈执行例子（Pixel 7 Pro + Tensor G2，TinyLlama 1.1B Q4，一条 ShareGPT 请求 prefill=232 tokens, decode=70 tokens）：
    - **模型推理算法**：标准 transformer decoder。TinyLlama 1.1B（22 layers, 2048 hidden dim, 32 heads），4-bit 量化（Q4_K_M GGUF 格式）。prefill 阶段：并行处理 232 tokens 的 self-attention + FFN forward pass，计算 O(L²×d) attention + O(L×d²) FFN。decode 阶段：逐 token 自回归，每步仅 1 token self-attention（memory-bound O(L×d)）。
    - **Serving 框架**：llama.cpp (tag b2202) + OpenCL backend。加载 GGUF 量化模型权重到 unified memory (LPDDR5)，推理循环分为 prefill（llama_decode 批量处理 prompt tokens）和 decode（llama_decode 逐 token）。无频率控制——完全依赖 Android 内核的默认 DVFS governor 自动调节 CPU/GPU/内存频率。
    - **编译框架**：论文未修改编译框架。llama.cpp 使用 GGML 的 JIT 编译（OpenCL kernel 字符串运行时编译）。论文未明确说明编译层面修改。
    - **Kernel 调度**：OpenCL kernel 在 ARM Mali-G710 GPU 上执行 INT4 matmul（attention QK^T 和 FFN GEMM）。CPU kernel 执行 tokenization、sampling（argmax/top-p）、KV cache 管理。OpenCL 命令队列由 CPU 提交，GPU 消费。默认 governor 下：GPU quickstep 检测 GPU 利用率低（decode 阶段 GPU 等待 CPU 喂数据）→ 降低 GPU 频率；CPU EAS 检测 CPU 利用率低（等待 GPU 完成 kernel）→ 降低 CPU 频率。两者无协调，形成级联降频。
    - **硬件架构**：Google Tensor G2 SoC（Samsung 5nm）。2× Cortex-X1 (2.85 GHz) + 2× A78 (2.35 GHz) + 4× A55 (1.80 GHz) 共 8 核；ARM Mali-G710 MP7 GPU (848 MHz max)；LPDDR5 unified memory (51.2 GB/s, 3172 MHz max)。CPU/GPU/内存共享 unified memory 无显式数据传输，但频率独立决策导致各组件在错误频率点运行。
  - Baseline 痛点：
    1. **独立 Governor 远非最优（23.0–40.4% 延迟更长）**：各 governor 独立选择频率，无法达到全局最优。例如 decode 阶段最优组合为 GPU 506 MHz + CPU 1400 MHz，但默认 governor 因互相影响降至 GPU ~300 MHz + CPU ~800 MHz。
    2. **"向下螺旋"（Downward Spiral）效应**：Decode 阶段 GPU 等待 CPU 喂数据 → GPU 利用率低 → GPU governor 降 GPU 频 → GPU 变慢 → CPU 等待 GPU 更久 → CPU 利用率更低 → CPU governor 降 CPU 频 → CPU 更慢喂数据 → GPU 更慢 → 继续降频。如此级联直至 CPU/GPU 双双降至最低有效频率。
    3. **低利用率的根本矛盾**：Decode 阶段每步仅 1 token，GPU compute 极少（memory-bound kernel），GPU 利用率天然低（<20%）。但利用率导向的 governor（如 quickstep、EAS）将低利用解读为"可降频"，而实际上 decode 的 memory-bound 特性决定了这些频率不应过低（过低频率降低内存带宽有效利用率）。
    4. **Prefill/Decode 阶段的最优频率截然不同**：Prefill（compute-bound 大矩阵乘）需要高 GPU/CPU 频率；Decode（memory-bound 单 token）仅需中等频率。默认 governor 无法区分两个阶段，使用同一调度策略，导致两阶段都不在最优点。
    5. **Memory Governor 的伪问题**：实验发现 memory governor 在默认设置下已接近最优——真正的问题是 CPU/GPU 频率的协调缺失。三个 governor 同时独立运行造成不必要的复杂度。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法（CORE 统一能量感知 DVFS 调度器）：
    1. **离线 Profiling + 频率搜索**：安装时为每个模型运行一次 profiling，搜索最优频率组合。通过将 prefill 长度划分为 5 个 range + 1 个 decode setting，将搜索空间从穷举的 2808 种组合缩减至 ~45 次推理/model。每个 (prefill_range, decode_setting) 存储最优 (GPU_freq, CPU_freq)。
    2. **"GPU First, CPU Next" 两阶段搜索启发式**：利用 GPU 频率是主导因素的发现，先搜索最优 GPU 频率（从高到低），再在 GPU 固定下微调 CPU 频率。Memory governor 保留默认（已接近最优）。相比穷举减少 374× 搜索量。
    3. **运行时阶段感知频率钉扎**：推理时区分 prefill 和 decode 阶段，使用各自的最优频率组合。Prefill 用高 GPU/CPU 频率（compute-bound）；Decode 用中等频率（memory-bound，更高频率不提升吞吐但浪费能量）。频率通过 Android sysfs 接口在阶段切换时无缝切换。
  - 论文方法全栈执行例子（同上 TinyLlama 1.1B Q4 on Pixel 7 Pro, prefill=232, decode=70）：
    - **模型推理算法**：与 baseline 完全相同——标准 transformer decoder + 4-bit 量化。推理精度无损。CORE 不修改模型结构或量化方案。
    - **Serving 框架**：llama.cpp + CORE Python daemon（~2K 行）。Profiling 阶段预计算频率 lookup table；推理阶段 daemon 监听 prefill/decode 信号→查表→sysfs 钉扎频率。Prefill：GPU 806 MHz + CPU 2100 MHz → TTFT 减少 8.5-17.7%。Decode：GPU 506 MHz + CPU 1400 MHz → TPOT 减少 27.8-39.6%（decode 阶段收益更大，因为这是"向下螺旋"最严重的阶段）。阶段切换时切换频率（微秒级 sysfs 写入开销）。
    - **编译框架**：论文未修改编译框架（同 baseline，论文未明确说明编译层面修改）。
    - **Kernel 调度**：与 baseline 相同的 OpenCL/CPU kernel。但频率被 CORE 主动钉扎在最优点：prefill 高频率使 GPU matmul kernel 全速运行（848→806 MHz 的选择说明 848 并非必要，806 已接近 compute 饱和），CPU 高频率保证 GPU 命令队列快速填充；decode 中频率使 memory bandwidth 被有效利用（506 MHz GPU 足够榨取 LPDDR5 的有效带宽），CPU 中频率足以执行 sampling + KV cache 管理而不浪费能量。关键差异：kernel 执行效率由频率决定，CORE 确保 kernel 在最优频率而非默认 governor 的过低频率下运行。
    - **硬件架构**：同 baseline Google Tensor G2。CORE 通过协调 CPU/GPU 频率打破"向下螺旋"——主动钉扎两组件在最优频率，消除互相等待导致的利用率虚假偏低。Monsoon Power Monitor 验证：CORE 的 decode 能耗与默认 governor 持平（中频率避免浪费），prefill 能耗因 TTFT 缩短而人均降低。
  - 对应解决 Baseline 缺陷：
    1. **独立 Governor 远非最优** → CORE 将 CPU/GPU 频率作为联合优化变量，通过 offline profiling 找到全局最优 (GPU_freq, CPU_freq) 组合。实验显示 CORE 的 TTFT/TPOT 显著优于默认 governor，且接近穷举搜索的理论最优（pinned optimal combination）。
    2. **"向下螺旋"效应** → CORE 主动钉扎频率而非依赖 governor 反应式调频，打破 CPU⇄GPU 互相拖累的恶性循环。decode 阶段 GPU 钉扎在 506 MHz（而非 governor 自动降到的 ~300 MHz），CPU 钉扎在 1400 MHz（而非 ~800 MHz），两组件均在各自最优点独立运行。
    3. **低利用率的根本矛盾** → CORE 理解 decode 阶段的 memory-bound 特性：即使 GPU 利用率低（1 token/batch），仍需要足够的 GPU 频率来有效利用内存带宽。CORE 通过 profiling 发现了 decode 的"甜点频率"——刚好满足内存带宽需求的最低 GPU 频率，而非 governor 根据利用率推导的更低频率。
    4. **Prefill/Decode 阶段频率需求不同** → CORE 为 prefill 和 decode 分别搜索和使用不同频率组合。Prefill 偏好高 GPU（compute-bound 大矩阵乘），decode 偏好中等 GPU/CPU（memory-bound）。阶段切换时无缝切换频率（sysfs 写入延迟 < 1 ms）。
    5. **Memory Governor 无需修改** → CORE 的 profiling 实验发现 memory governor 在默认策略下已接近最优——简化了设计（仅需协调 CPU+GPU 两维），降低搜索空间（自 2808 → 18×12 = 216，再经启发式进一步减少）。
