## Fused MoE（融合 MoE Kernel）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused MoE 是一种针对 Mixture of Experts（MoE）推理优化的 GPU kernel 融合技术，核心思想是将 expert 选择（routing）、token 分发（dispatch）和 expert FFN 计算（GEMM + activation + GEMM）融合为单个 GPU kernel，消除中间结果的 HBM 往返和 kernel launch 开销。在未融合的 naive MoE 实现中，推理流程包含多个串行 kernel launch：(1) router 计算 top-k expert 分配 → (2) token-to-expert dispatch（scatter/gather 操作）→ (3) 逐 expert GEMM（w1 @ x → silu → w2 @ x）→ (4) weighted sum reduction。每个步骤之间的中间 tensor 需要写入 HBM 再读出，产生显著的显存带宽开销。Fused MoE 将上述步骤合并为单次 kernel launch：Triton/CUDA kernel 直接使用 sorted_token_ids（按 expert 索引排序后的 token 索引表）间接寻址 token，每个 thread block 根据 expert_ids 加载对应 expert 权重矩阵并直接计算，top-k routing weight 在 kernel 内直接乘回输出。vLLM 中 Fused MoE 的核心实现位于 `vllm/model_executor/layers/fused_moe/fused_moe.py`，使用 Triton JIT kernel 实现，支持 FP16/BF16/FP8 精度，并可与 GPTQ/AWQ 等量化方法结合。H100 GPU 上 Fused MoE 可提供 15-20% 吞吐量提升（MoE-Inference-Bench Section 7.2），大 batch 时优势更明显。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fused MoE kernel 的简化伪代码（以 vLLM Triton fused_moe_kernel 为蓝本，Mixtral-8x7B 配置：8 experts, top-k=2, hidden_dim=4096, ffn_dim=14336）：

```
// Fused MoE Kernel (Triton pseudocode)
// Grid: num_experts × ceil(num_tokens / BLOCK_M) blocks

@triton.jit
def fused_moe_kernel(
    A,                  // input tokens [total_tokens, hidden_dim]
    B,                  // stacked expert weights [E, hidden_dim, ffn_dim*2 + ffn_dim]
    C,                  // output [total_tokens, hidden_dim]
    sorted_token_ids,   // token indices sorted by expert assignment
    expert_ids,         // expert index per block
    topk_weights,       // routing weights (FP32)
    num_tokens_post_padded,
):
    pid = tl.program_id(0)
    expert_id = tl.load(expert_ids + pid)
    block_start = pid * BLOCK_M

    // Step 1: Indirect token gather — only load tokens for this expert
    token_indices = sorted_token_ids[block_start : block_start+BLOCK_M]
    a_block = tl.load(A + token_indices[:, None] * H + range(H))  // [BLOCK_M, H]

    // Step 2: Load expert weights once (w1_gate, w1_up, w2_down packed)
    w1 = tl.load(B + expert_id * stride_E)  // [H, ffn_dim * 2]

    // Step 3: FC1 — fused gate+up projection (single GEMM, no HBM write)
    gate = silu(tl.dot(a_block, w1_gate))   // [BLOCK_M, ffn_dim]
    up   = tl.dot(a_block, w1_up)            // [BLOCK_M, ffn_dim]
    hidden = gate * up                       // element-wise fused gating

    // Step 4: FC2 — down projection
    w2 = tl.load(B + expert_id * stride_E + offset_w2)  // [ffn_dim, H]
    expert_out = tl.dot(hidden, w2)          // [BLOCK_M, H]

    // Step 5: Fused routing weight application
    routing_w = tl.load(topk_weights + block_start + range(BLOCK_M))
    expert_out = expert_out * routing_w[:, None]

    // Step 6: Atomic scatter-add to output (combines top-k experts)
    tl.atomic_add(C + token_indices[:, None] * H + range(H), expert_out)
```

MoE-Inference-Bench 的关键性能发现（Section 7.2，以 Mixtral-8x7B on 4×H100 + vLLM）：Fused MoE 在大 batch 时提供 15-20% 吞吐量提升，不同 sequence length 下保持 12-18% 优势。收益机制：(a) 消除中间 tensor 的 HBM 往返（每层节省约 3-4 次 HBM read/write）；(b) 减少 kernel launch 开销（从约 6-8 次 kernel launch 降至 1-2 次）；(c) 利用 sorted token 的连续性提升 L2 cache 命中率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Fused MoE 的跨框架实现：
- **vLLM**：`vllm/model_executor/layers/fused_moe/fused_moe.py`，Triton JIT kernel。通过配置文件或环境变量启用。MoE-Inference-Bench 在 vLLM 中直接使用其内置 Fused MoE kernel。
- **Megatron-Core (NVIDIA)**：`core.inference.moe.vllm_fused_moe`，CUDA graph 兼容版本，token alignment 和 indirection table 构建均在 GPU 端完成。
- **DeepGemm**：高度优化的 FP8 MoE kernel，专门针对 Hopper (SM90+)。
- **FlashInfer**：CUTLASS-based MoE for SM90/SM100。
- 前置步骤 **moe_align_block_size**：将 topk_ids 展平、按 expert 排序、padding 到 BLOCK_SIZE 对齐，是 Fused MoE 的必要预处理。
- 限制：(a) 小 batch 下 padding overhead 可能抵消融合收益；(b) 需要 Triton 或 CUDA 支持；(c) 对 expert 数量多但每 expert token 极少的场景（如大量 expert 中仅少数激活），indirect addressing 的 cache 局部性退化。

涉及论文标题：
- MoE-Inference-Bench: Performance Evaluation of Mixture of Expert Large Language and Vision Models
- MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts

**MoE-SpeQ 的 fuseMoE Kernel 扩展**：针对量化 MoE draft 阶段的细粒度场景（如 Qwen2-MoE: K=1408, N=2048），标准 Marlin 后端慢于 PyTorch FP16（图 11），因每个 expert 矩阵太小无法占满 GPU SM。fuseMoE 将 per-layer 所有 expert 的 gate_proj + up_proj + SiLU + down_proj 融合为单次 CUDA kernel launch，batch 多 expert 的 token hidden states 增大有效矩阵维度，提升 GPU occupancy。消融显示 fused kernel 贡献 31.8% 速度提升（从 8.88 tok/s 到 13.02 tok/s）。与 vLLM Fused MoE 不同：fuseMoE 专为量化（INT4, Marlin）和 draft 阶段设计，且与 async prefetching 独立叠加以达到 additive speedup。

**MoEBlaze 的 Epilogue Fusion for Training**：MoEBlaze 将 fused kernel 概念从推理扩展到训练，针对 SwiGLU MoE 训练设计 "epilogue fusion" kernel——将两个第一层投影 (W1, W2) 的 GEMM 与 SwiGLU epilogue（SiLU + element-wise multiply）融合为单 kernel。关键差异于推理融合：(1) 融合两个独立 GEMM（W1 和 W2 投影），输入 x 仅加载一次（vs 分别两次），两个 GEMM 流式并行执行；(2) SiLU(a) 计算在 register/shared memory 中完成，不写回 HBM——forward 仅保存 a, b, y_swi；(3) backward 中 recompute SiLU(a)（activation checkpoint），利用 SiLU 的 element-wise 特性（memory bandwidth bound，recompute 开销 ≈ 直接从 HBM 读取的成本）；(4) backward 中两个分支的 activation derivatives (∇a, ∇b) 通过 tiled reduction 做 in-place 聚合。在 SwiGLU 下实现最高 4× 激活内存减少和 2×–6.2× 训练加速。
