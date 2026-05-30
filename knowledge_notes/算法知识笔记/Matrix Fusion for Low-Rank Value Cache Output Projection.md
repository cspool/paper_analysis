## Matrix Fusion for Low-Rank Value Cache Output Projection

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Matrix Fusion 是将低秩 Value 压缩的右因子 R_v 预融合进 output projection W_o 的技术。标准 attention: output = Attention(Q, K, X·W_v) · W_o。低秩压缩后: output = Attention(Q, K, X·L_v·R_v) · W_o。Fusion: W_o_fused = R_v·W_o，推理时 output = Attention(Q, K, X·L_v) · W_o_fused，消除在线 Value 重建步骤。

从算法pipeline角度拆解术语：

```
// Offline: W_o_fused = R_v @ W_o  [r_v × d_model]
// 推理: V_latent = X @ L_v [seq_len, r_v], 存入 KV cache
// O = softmax(QK^T/√d) @ V_latent  [seq_len, r_v]
// output = O @ W_o_fused  [seq_len, d_model]
// 节省: 无 Value 重建, O 矩阵缩小, KV cache 从 h·d_k 降至 r_v
```

LLaMA-2-7B, r_v=2048 (50% 压缩)，O 从 [1,4096] 缩至 [1,2048]。

术语一般如何实现？如何使用？

PyTorch 一行: `W_o_fused = torch.mm(R_v, W_o)`，完全 offline。融合后数学等价于先重建再投影，无精度损失。替换原 attention 层的 output projection 权重即可。

涉及论文标题：
- ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration

---
