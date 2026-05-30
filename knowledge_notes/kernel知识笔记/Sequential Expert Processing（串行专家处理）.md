## Sequential Expert Processing（串行专家处理）

术语是什么？
Sequential Expert Processing 是 ES-MoE 提出的替代 batched GEMM 的 MoE 计算执行方式。传统 MoE 框架所有 experts 同时在 GPU 上用 batched GEMM + dispatch mask 处理；sequential approach 逐 expert 串行执行：上传 expert → 处理其 tokens → 释放 GPU 内存 → 下一个。

从kernel调度角度拆解术语：
```
for expert_j in assigned_experts:
    mask_j = (gating_result == expert_j)       # boolean mask [N_tokens]
    tokens_j = input[mask_j]                    # [T_j, H] actual tokens
    hidden = silu(gate_proj(tokens_j)) * up_proj(tokens_j)
    output[mask_j] = down_proj(hidden)          # scatter back
```
无 dispatch mask、无 zero-padding。内存仅需 1 expert weight + activations。

术语一般如何实现？如何使用？
ES-MoE 修改 Fairseq MoE layer forward：替换 batched_gemm 为 per-expert loop。当 microbatch 足够大时 per-expert GEMM 可高效利用 Tensor Core。劣势：per-expert tokens 极小时 GPU 利用率下降（如 64 experts, batch 32 → ~16 tokens/expert）。

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training
