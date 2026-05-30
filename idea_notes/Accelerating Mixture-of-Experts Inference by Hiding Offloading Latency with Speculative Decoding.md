## Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding

- baseline方法是什么？
  Baseline 为两类 SOTA MoE offloading 系统：

  1. **DeepSpeed-ZeRO-Inference [3]**（batch=1 类代表）：利用 MoE 的稀疏激活特性，每次只加载激活的 expert weights 到 GPU HBM，通过 expert prediction 和 hot expert caching 降低传输开销。适用于小 batch size，在 batch=1 时 GPU 利用率仅 0.76%（Mixtral 8x7B on A30）。

  2. **MoE-Lightning [6]**（throughput-oriented 代表）：通过增大 batch size 对抗 MoE 稀疏性，专家权重加载后 GPU 可并行处理多个 input。KV cache 维护在 CPU DRAM 中，attention 计算在 CPU 执行。但因 CPU DRAM 容量限制，batch size 仍有限，理论 GPU 利用率仅 3.13%。

  **Baseline 全栈执行例子（以 MoE-Lightning on Mixtral-8x7B, A30 GPU 为例）**：

  - **算法层**：输入 tokens → Router 计算 top-2 experts → 加载 expert weights (CPU DRAM→GPU HBM) → Expert FFN (3×矩阵乘法: W_gate, W_up, W_down) → CPU Attention (GEMV 对 KV cache in CPU DRAM) → 逐 token 自回归生成。每次 forward 仅处理 1 个 token per request。无 speculative decoding。

  - **系统框架层**：MoE-Lightning 系统 → batch 拆分为 micro-batches → GPU Other1→CPU Attention→GPU Other2→GPU MoE 流水线 → expert weights 异步预取 → KV cache 在 CPU DRAM, attention 在 CPU 执行。

  - **编译框架层**：论文未明确说明（标准 PyTorch + CUDA 执行）。

  - **Kernel/运行时调度层**：CPU Attention kernel (GEMV, 单 token) → GPU MoE kernel (GEMM, expert FFN) → HtoD Transfer (expert weights 加载)。瓶颈分析 (Figure 3 Roofline): MoE layer 主要受 CPU-GPU memory transfer for expert weights 限制（memory-bound），Attention layer 受 CPU memory access for KV cache 限制（memory-bound）。

  - **硬件架构层**：A30 GPU (165 TFLOPS, 933 GB/s) + Intel Xeon Gold 6426Y CPU + 250 GB CPU DRAM。GPU 和 CPU 通过 PCIe 连接 (25 GB/s)。Mixtral-8x7B FP16 占 87GB，CPU DRAM 中约 160GB 留给 KV cache。

  **Baseline 的核心缺陷**：
  1. **GPU 利用率极低**：batch=1 方案 GPU 利用率仅 0.76%，MoE-Lightning 仅 3.13%。根本原因是 expert loading 的 I/O 瓶颈和 MoE 稀疏激活导致每次 forward 处理 token 数太少，GPU compute 能力远未被充分利用。
  2. **CPU-GPU transfer 瓶颈**：MoE layer 的 CPU→GPU expert weight 传输几乎占满 PCIe 带宽，成为 memory-bound bottleneck。
  3. **CPU memory access 瓶颈**：Attention layer 的 CPU KV cache 访问受限于 CPU memory bandwidth，CPU computational power 也未被充分利用。
  4. **固定 hyperparameter 配置**：MoE-Lightning 使用固定 batch size/micro-batch size/缓存策略，无法在不同硬件和 workload 下自适应调节。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：SpecMoEOff = speculative decoding (EAGLE) + CPU chunked attention kernel + memory-conscious draft execution + hyperparameter optimizer。核心洞察：speculative decoding 增大每次 forward 的 workload（从 1 token 变为 k+1 tokens），天然对齐 MoE offloading 场景下 GPU 资源闲置的问题。

  **Defect→Design 映射**：

  | Baseline 缺陷 | SpecMoEOff 设计选择 | 解决机制 |
  |---|---|---|
  | GPU 利用率仅 0.76%-3.13% | Speculative decoding: draft model 生成 k tokens, target model 一次性验证 | 每次 forward 处理 k+1 tokens 而非 1 token → 增大 operational intensity → GPU MoE 层从 memory-bound 转为更接近 compute-bound |
  | CPU-GPU transfer 瓶颈 (memory-bound) | 增大每次 forward 的 batch size (k 倍 token 数) | 相同 transfer 量下计算更多 token → 每 token 的 transfer cost 分摊 |
  | CPU KV cache memory access 瓶颈 | Chunked attention: 合并 k 个 draft tokens 的 KV cache access 为一次读取 | d× 次 KV cache 读取 → 1× 次，充分利用 CPU memory bandwidth |
  | Attention 在 CPU 上低效 (per-token GEMV) | CPU chunked attention kernel (Intel MKL GEMM) | GEMV→GEMM (Q@K^T), 利用 CPU SIMD/MIMD 高效执行 |
  | Draft model KV cache 溢出 GPU HBM | CPU/GPU separation: batch 维度分片, GPU Part (全 GPU 执行) + CPU Part (CPU attention + GPU FFN) | 动态调整分离比例，优先级: draft model > target model (draft 多次调用，target 仅一次) |
  | 固定 hyperparameter 无法自适应 | Hyperparameter optimizer: 凸优化预决定 + profiling estimator DAG 模拟 | 自动搜索最优 k, 适应不同硬件/模型/workload |
  | Naive mask 存储浪费内存 | Mask 压缩: 仅存储 draft-to-draft 子区域 (n×n) | O(n·(l+n)) → O(n²) mask 内存 |

  **SpecMoEOff 全栈执行例子（以 Mixtral-8x7B + EAGLE draft, k=5 draft tokens, 单 iteration 为例）**：

  - **算法层**：
    1. Draft model (EAGLE, <2GB in GPU HBM): 输入当前 hidden state → iterative generate k=5 draft tokens (GPU Part + CPU Part 并行 attention → GPU FFN)
    2. Target model verify: 将 original tokens + 5 draft tokens 拼接 → CPU Chunked Attention (Intel MKL GEMM, Q[5,4096]@K^T[4096,517]→[5,517]) → GPU MoE (expert weights CPU→GPU, FFN x(5+1)tokens) → accepted = a(5) ≈ 3-4 tokens
    3. Next iteration: update KV cache with accepted tokens

  - **系统框架层**：SGLang + SpecMoEOff:
    - Target model pipeline: microbatch 1 的 GPU Other1 → CPU Attention (concurrent with microbatch 2 的 GPU Other1) → microbatch 1 的 GPU Other2 → GPU MoE (concurrent with microbatch 2 的 CPU Attention)
    - Next layer expert weights: async HtoD transfer (separate CUDA Stream, concurrent with current layer compute)
    - Draft model: GPU Part + CPU Part 并行 → 生成 k draft tokens → verification
    - Memory Manager: expert cache (GPU HBM) + KV cache (CPU DRAM target, GPU/CPU split draft)

  - **编译框架层**：论文未明确说明。

  - **Kernel/运行时调度层**：CPU Chunked Attention (Intel MKL GEMM → softmax → MKL GEMM, mask 压缩仅 5×5) → GPU MoE GEMM (expert FFN, 3× matrix per expert, fused MoE implementation) → HtoD Transfer (separate CUDA Stream, CUDA Event synchronization) → GPU Attention (draft model GPU Part, FlashAttention-style)。

  - **硬件架构层**：A30/4090D + Intel Xeon CPU + CPU DRAM → GPU 和 CPU 通过 PCIe (25/23 GB/s) 连接 → 所有 expert weights 均在 CPU DRAM → hot expert cache 在 GPU HBM (如 5.25 GB) → KV cache 在 CPU DRAM (target) + GPU/CPU split (draft) → speculation 减少 per-token PCIe transfer cost。

  **对比 Baseline 的核心改进路径**：
  ```
  Baseline (MoE-Lightning):
  iteration: [1 token forward] x until generation done
  per iteration: CPU Attention(GEMV,1 token) → CPU→GPU load experts → GPU MoE(1 token)
  GPU utilization: 3.13%, bottleneck: PCIe transfer + CPU mem bandwidth

  SpecMoEOff:
  iteration: [draft k tokens → verify (k+1) tokens] x until generation done
  per iteration: Draft: GPU/CPU parallel attn + GPU FFN (k times, tiny model)
              → Verify: CPU Chunked Attention(GEMM, k+1 tokens) → 
                CPU→GPU load experts → GPU MoE(k+1 tokens)
  GPU utilization: improved (more compute per load), 
  speedup: 2.5× decode throughput over MoE-Lightning
  ```

  **关键设计对应关系**：

  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | Speculative decoding (EAGLE) | GPU utilization from 3.13% higher | 1.45×-1.53× speedup over w/o-sd baseline |
  | CPU chunked attention (Intel MKL) | CPU attention 成为 bottleneck (Figure 13) | CPU Attention: 4.29s actual, 10.6% estimation error |
  | CPU/GPU draft separation | Draft KV cache > GPU HBM (A30 24GB) | 动态分离比例, 初始更多GPU, 随seqlen增长迁移CPU |
  | Hyperparameter optimizer | 固定 k 无法最优 (Figure 10 shows non-monotonic) | Dynamic k 比 fixed-k best 额外 +2% throughput |
  | Mask 压缩 | 减少 CPU DRAM 占用 | O(n×(l+n))→O(n²) |

  **Roofline 分析的核心发现**：
  - MoE layer (GPU compute + CPU-GPU transfer): arithmetic intensity 仅 3.13% of GPU peak → speculative decoding 增大 b×k 倍 token 数 → 直接提升 operational intensity
  - Attention layer (CPU memory access): KV cache access 是瓶颈 → chunked attention 将 n 次读取合并为 1 次
  - Speculative decoding 恰好同时缓解两个瓶颈
