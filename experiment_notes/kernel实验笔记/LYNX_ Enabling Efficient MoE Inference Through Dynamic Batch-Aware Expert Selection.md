## LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - LYNX 实现了 4 个 fused Triton CUDA kernels，将 MoE router 输出的后处理（confidence analysis, expert scoring, expert pruning, remapping）融合为高效 kernel，替代原本超过 700 个 PyTorch 小算子。具体 kernel 实现：
    1. **Kernel 1 — Token-wise Binning（逐 token 离散化）**：拦截 router logits，计算每 token 对 top-k 专家的 log-ratio（logit[e] - logit[top1]），做 AffinityBinning 离散化，同时计算 top-k 权重和。融合了原本需数百个 PyTorch element-wise ops（subtract, division, floor, clamp）。
    2. **Kernel 2-3 — Batch-wise Scoring & Expert Pruning（批次级评分与剪枝）**：以 batch_size 为底数的指数加权计算每个 expert 的 batch 级分数，基于分数分布和 bin width 动态确定 active expert set。融合了 reduce、scatter、top-k 选择等操作。
    3. **Kernel 4 — Expert Remapping & Compaction（专家重映射与压缩）**：将 low-confidence tokens 的 expert assignment 重映射到 active expert set，compaction 重排 token-to-expert 映射表，renormalize 权重并重新计算 top-k。融合了 gather、scatter、sort、softmax 等操作。
    4. 所有 kernel 保持静态控制流，支持 CUDA Graph capture。
  - 实验比较：
    - Latency breakdown：Baseline (vLLM default) vs LYNX 的端到端延迟分解为 expert computation 和 non-expert components
    - Kernel overhead：LYNX 的 4 个 fused kernel 开销 <4% 总体延迟
    - 不同 batch size（1-64）和 sequence length（512/4096）下的 TPOT
    - 与不同并行策略（TP, EP）的叠加效果

- 后端平台是什么，配置是什么。
  - **GPU**：NVIDIA H200 (141 GB HBM)，SXM NVLink
  - **CPU**：2x AMD EPYC 9554 64-Core，1.5 TB DRAM
  - **软件栈**：Ubuntu 22.04.4 LTS，CUDA 12.6，NVIDIA driver 560.35.05
  - **Kernel 框架**：Triton（4 个 fused kernels），PyTorch profiler 用于 kernel-level latency capture
  - **Offloading 实验**：NVIDIA A100，PCIe CPU-GPU 链路

- 评估性能的软件/脚本是什么。修改了什么。
  - vLLM v0.10.1 框架，PYTORCH CUDA profiler 捕获 kernel-level latency
  - 4 个 fused Triton kernel 是全新实现的，替代了 vLLM 默认 MoE router 后的 PyTorch dispatch pipeline
  - 修改内容：
    1. **新增 Confidence Analyzer Kernel**：Triton 实现，输入 router logits (B x N)，输出 per-token bin assignments 和 top-k weight sums。key operation：log_ratio discretization with α/β binning params。
    2. **新增 Adaptive Expert Scorer Kernels (x2)**：Triton 实现，输入 per-token bin assignments (B x k)，输出 batch-level expert scores (N) 和 active expert mask。key operation：exponential weighting with batch_size base, score-based thresholding。
    3. **新增 Expert Remapper Kernel**：Triton 实现，输入 active expert mask + per-token bin assignments，输出 compacted token-to-expert mapping + renormalized weights。key operation：gather-scatter remapping, softmax renormalization。
    4. **Phase-aware Optimizer in Batch Scheduler**：在 vLLM scheduler 中新增 memory-bound detection 逻辑（非 kernel，为 CPU 端调度逻辑）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：论文未提供开源代码链接。实现基于 vLLM v0.10.1 + Triton。
  - **Kernel 执行全流程**（以 Qwen2-57B 单层 MoE layer, decode iteration, batch B=16, N=64, k=8 为例）：
    ```
    [输入] Router logits: B x N = 16 x 64 float tensor (GPU global memory)
    
    [Kernel 1 — Token-wise Binning]
    - Grid: (B, ) 即 16 thread blocks
    - 每个 block: 加载 1 个 token 的 router logits (64 elements) → registers
    - 计算: top-1 logit → log_ratio[e] = logit[e] - top1_logit for each e in top-k
    - 离散化: bin[e] = clamp(floor(log_ratio[e] * α), -β, 0)
    - 输出: bin assignments (B x k int) + top-k weight sums (B float) → global memory
    - 融合的 PyTorch ops: logit sort, subtraction, softmax (deferred), floor, clamp
    
    [Kernel 2-3 — Batch-wise Scoring & Pruning]
    - 输入: bin assignments (B x k), batch_size B
    - Grid: (N, ) 即 64 thread blocks (每个 expert 一个)
    - 每个 block: 遍历 batch 中所有 token，若 expert 在该 token 的 top-k 中则累加 B^{bin[token][expert]}
    - 计算: score[expert] = Σ_t B^{bin[t][expert]}
    - 阈值确定: 基于 score distribution + bin_width + max_bins 动态计算
    - 输出: active expert mask (N bool) → global memory
    
    [Kernel 4 — Expert Remapping & Compaction]
    - 输入: active expert mask, per-token bin assignments, top-k per token
    - 操作: 对每个 low-confidence token，将其 lower-ranked expert 重映射到 active expert set 中的替代专家
    - Compaction: 重排 token-to-expert 映射表为连续索引
    - Renormalize: 对 remapped assignment 重新计算 softmax → 最终 dispatch weights
    - 输出: compact mapping (B x k int), renormalized weights (B x k float)
    
    [后续] Expert GEMM kernel launch with reduced expert count
    - 原本需加载 ~25 个 experts 的权重 → 现在仅需加载 ~15-18 个
    - 从 HBM 读取量: (active_count / original_count) * expert_size 字节
    ```
  - **评估原理**：LYNX 的 kernel 在 expert computation 前执行，通过减少 active expert 数量来降低 HBM 带宽消耗。4 个 fused Triton kernel 的开销 (<4% 总体延迟) 远小于因减少 expert 加载而节省的内存带宽时间。用 PyTorch profiler 在每个 iteration 内按 kernel 分解延迟，测量 expert computation latency 的减少量与 LYNX kernel overhead 的差值作为 net gain。
