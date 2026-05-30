## FlashMLA

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FlashMLA 是 DeepSeek 开源的高效 MLA (Multi-head Latent Attention) decode kernel，专为 Hopper 架构（H100/H800/H200）GPU 优化。GitHub: https://github.com/deepseek-ai/FlashMLA。FlashMLA 实现了 MLA 的 Layer Reordering 优化，在 decode 阶段的 Score 和 Context 计算中通过复用压缩 KV Cache ($\mathbf{C}_{\text{KV}}$) 将 ArI 从 ~100 Op/B 提升至 ~200 Op/B（翻倍）。核心优化：Score 层从 HBM 加载 $\mathbf{C}_{\text{KV}}$ 后，Context 层立即复用共享内存中的 $\mathbf{C}_{\text{KV}}$ 而不重新从 HBM 读取，将两次内存访问合并为一次。这与 FlashAttention 的 tiling + recomputation 思路类似但适配 MLA 的计算模式。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === FlashMLA Decode Kernel (简化的计算流程) ===
# 利用 Hopper GPU 的 TMA (Tensor Memory Accelerator) 和 WGMMA 指令

# 输入在 HBM: C_Q (B, 1536), C_KV (B, L, 512), W_DQ_i (1536,128), etc.
# 输出在 HBM: O (B, 16384)

# 分块策略: 将 L 维度分块, 顺序处理
for block_l in range(0, L, BLOCK_L):
    # 1. 异步加载 C_KV[:, block_l, :] 到共享内存 (TMA)
    #    C_KV_block: (B, BLOCK_L, 512)
    load_C_KV_async(C_KV_block)

    # 2. 计算 Score (使用 WGMMA)
    #    S_block = QW_i @ C_KV_block^T → (B, BLOCK_L)
    #    QW_i = Q_i @ W_DK_i^T → (B, 512) 在 kernel 开始时计算
    compute_score_block()

    # 3. Online Softmax (类似 FlashAttention)
    #    更新 running max 和 sum
    softmax_block = online_softmax_update(S_block)

    # 4. 复用 C_KV_block (已在共享内存) 计算 Context
    #    PV_block += softmax_block @ C_KV_block → (B, 512)
    #    *** 关键: C_KV 不需要再加载, 已经在 SMEM ***
    compute_context_block_with_reuse(C_KV_block)

# 5. 最终 Context: O_i = PV @ W_DV_i → (B, 128)
# 6. 合并所有 heads 的输出
```

核心性能增益：传统实现中 Score 和 Context 两层各需加载一次 $\mathbf{C}_{\text{KV}}$，FlashMLA 通过 tiling + fused kernel 将两次 HBM 访问合并为一次，ArI 翻倍。在 DeepSeek-R1 (d_KVco=512, n_hd=128) 上，ArI 从 ~100 提升至 ~200 Op/B，初步逼近 B200 的 Ridge Point (281.25)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlashMLA 要求 Hopper 架构 GPU（SM90+），利用 CUDA 12.3+ 和 CUTLASS 库。安装使用：`pip install flash-mla` 或从源码编译。API 类似 FlashAttention：`flash_mla_fwd(q, c_kv, ...)`。已集成到 DeepSeek 官方推理代码和部分 vLLM/SGLang 版本中。仅适用于 decode 阶段（prefill 使用 reordering 无益，反而增加延迟）。论文中通过 FlashMLA 的优化效果验证了 reordered MLA 使 attention 不再需要专用 PIM 硬件的结论。

涉及论文标题：
- Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts
