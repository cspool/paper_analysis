## Sequential GeMM for MoE Expert Computation (MoE专家计算的顺序通用矩阵乘)

术语是什么？

Sequential GeMM 是 X-MoE 中用于替代传统 batched matmul 的 expert 计算方式。在 padding-free MoE pipeline 中，dispatch_out 是动态大小的 uneven token buffer（每个 expert 的 token 数量不同，且无 zero-padding）。Sequential GeMM 按 tokens_per_expert 数组将 dispatch_out 切片，依次为每个 expert 独立 launch 一个标准 GeMM。

从kernel调度角度拆解：

```
# dispatch_out: [Bexp, H], tokens_per_expert: [Elocal]
# w1[Elocal]: 每expert的第一层权重 [H, HFFN]
# w2[Elocal]: 每expert的第二层权重 [HFFN, H]

offset = 0
for i in range(Elocal):
    n_tokens = tokens_per_expert[i]  # expert i 的 token 数
    if n_tokens == 0:
        continue
    
    # 切片获取expert i的token
    expert_input = dispatch_out[offset : offset + n_tokens]  # [n_tokens, H]
    
    # 第一层FFN (可选activation)
    inter = matmul(expert_input, w1[i])  # [n_tokens, HFFN]
    inter = activation(inter)
    
    # 第二层FFN
    expert_output = matmul(inter, w2[i])  # [n_tokens, H]
    
    mlp_out[offset : offset + n_tokens] = expert_output
    offset += n_tokens
```

与 Grouped GEMM（Megablocks 方式）的对比：
- **Grouped GEMM**：单 kernel launch 并行计算所有 expert，但要求 padded equal-size blocks → zero-padding 开销
- **Sequential GeMM**：多次 kernel launch（Elocal 次），每次无 padding，expert 间无同步开销但 launch overhead 存在
- X-MoE 在 Small 模型上 expert computation 时间略增（因 sequential launch + data transform overhead），但总体 layer time 减少 62.3%（因消除了 zero-padding 的通信和内存收益）

术语一般如何实现？

在 X-MoE 中，Sequential GeMM 使用 Python for-loop 驱动 rocBLAS（AMD）或 cuBLAS（NVIDIA）的 GEMM 调用。每次 launch 处理一个 expert 的 tokens，GEMM 维度为 [n_tokens, H] × [H, HFFN] 或 [n_tokens, HFFN] × [HFFN, H]，其中 n_tokens 在各 expert 间通常不同。

涉及论文标题：
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms
