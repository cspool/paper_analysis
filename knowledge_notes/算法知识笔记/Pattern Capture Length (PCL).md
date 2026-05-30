## Pattern Capture Length (PCL)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Pattern Capture Length (PCL)，记作 L，是 DAM 中提取 attention pattern 的最大序列长度。$L = \min(S, L_{\max})$，$L_{\max}$ 为 full attention 不 OOM 的最长序列。DAM 在 A100 40GB + LLaMA 3.2 3B 上设定 PCL=512，平衡了捕捉 attention pattern 的完整性和计算可行性。关键理念：从硬件最大支持长度开始，按需向下调整（"top-down" 调参），而非从短到长试探。论文图 7 证实短序列中观察到的结构模式（对角线、垂直条带）可外推至更长序列。

从算法pipeline角度拆解术语：

```
L = min(S, L_max)  // L_max from hardware constraint

// Stage 1: extract attention map only within PCL
if S <= L:
    masks = full_attention_masks(frozen_model, seq[:S])
else:
    partial_attn = full_attention_masks(frozen_model, seq[:L])  // L×L only
    true_mask = threshold(partial_attn, τ)                      // L×L binary
    matched_patterns = structural_match(true_mask, μ)
    extended_mask = extrapolate(matched_patterns, target_len=S)  // S×S
```

术语一般如何实现？如何使用？

PCL 仅在 Stage 1（离线）使用。值由硬件决定——原 LLaMA 3.2 3B 在 A100 40GB 上 >4K tokens OOM，PCL=512 提供安全裕量。更大的 GPU 允许更大的 PCL。选值策略简单：从最大可支持长度开始，仅在需要时下调。

涉及论文标题：
- DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration
