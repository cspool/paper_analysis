## FusedMoE (vLLM Fused MoE Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FusedMoE 是 vLLM 推理框架中针对 MoE 模型的高性能融合 kernel，将 MoE 层的多个操作（token routing、expert dispatch、grouped GEMM、activation、weighted combine）融合为少量 GPU kernel 调用，减少 kernel launch overhead 和 HBM 访存次数。核心实现包括 Triton-based grouped GEMM（TritonExperts）和 CUTLASS/DeepGemm 等多种 backend。FusedMoE 是 vLLM 高效支持 Mixtral、DeepSeek-V2、Qwen-MoE 等 MoE 模型推理的基础组件。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// FusedMoE Kernel 调度流程
// 输入: hidden_states [B, L, H], 输出: expert_output [B, L, H]

// Step 1: Gate + Top-K Routing (GPU kernel, lightweight)
gate_logits = matmul(hidden_states, W_gate)        // [B, L, N_experts]
topk_vals, topk_idx = topk(gate_logits, k)          // 选 top-k experts
topk_weights = softmax(topk_vals)                   // 归一化

// Step 2: Token-to-Expert Sorting (Triton kernel)
// moe_align_block_size: 按 expert ID 排序 tokens
// 将同一 expert 的 tokens 分组到连续的 BLOCK_SIZE_M 块中
sorted_tokens, expert_offsets = sort_by_expert(
    hidden_states, topk_idx, block_size=64
)

// Step 3: Grouped GEMM - W1 (Triton Grouped GEMM kernel)
// 每个 expert 独立执行一次 GEMM，但 batch 在一起减少 kernel launch
// expert_i: sorted_tokens[offset_i:offset_{i+1}] @ W1_i
for expert_i in active_experts:
    h_i = sorted_tokens[offset_i:offset_{i+1}]  // [N_i, H]
    inter_i = h_i @ W1_i.T                       // [N_i, 4H] (gate+up projection)
    
// Step 4: SiLU Activation (fused in same kernel)
    gate, up = split(inter_i, 2)                  // gate + up projection
    act_i = up * silu(gate)                       // SwiGLU activation

// Step 5: Grouped GEMM - W2 + Reduce (Triton Grouped GEMM kernel)
    out_i = act_i @ W2_i.T                        // [N_i, H]
    // Scatter back to original token positions
    output[expert_i_tokens] += topk_weights[expert_i_tokens] * out_i
```

LExI 在 FusedMoE 上的修改最小：仅改变每层的 top-k 参数。减少 k 值直接减少 Step 2-5 的处理 token 数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

vLLM FusedMoE 位于 `vllm/model_executor/layers/fused_moe/`，核心类 `FusedMoE` 支持多种 kernel backend（Triton/CUTLASS/FlashInfer/Marlin/DeepGemm），通过 `FusedMoE.select_experts_implementation()` 自动选择。支持 FP16/BF16/FP8/INT4/INT8 量化。vLLM v0.12 引入 MoE chunking：将长序列 tokens 分块，允许 expert 计算与 all-to-all 通信重叠执行。对于 DeepSeek-V2/V3 的 shared expert，`SharedFusedMoE` 支持 shared expert 与 routed expert dispatch 的 overlap 执行。LExI 在此基础上的修改：加载模型后，修改每个 MoE layer 的 `self.top_k` 参数为进化搜索得到的 k_j 值。

涉及论文标题：
- LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference
