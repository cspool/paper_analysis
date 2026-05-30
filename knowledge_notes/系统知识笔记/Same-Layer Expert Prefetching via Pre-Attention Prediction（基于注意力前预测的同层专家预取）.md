## Same-Layer Expert Prefetching via Pre-Attention Prediction（基于注意力前预测的同层专家预取）

术语是什么？
Same-Layer Expert Prefetching 是 MoE 推理中在同一层内使用 pre-attention hidden state 预测该层 experts 并异步预取的技术。与 cross-layer prediction（用前一层预测下一层）不同：(1) pre-attention 信息"更新鲜"，temporal proximity 更高；(2) 所有层（含第一层）均可预测——无 bootstrap problem；(3) 预测 (0.15ms CPU) 与 self-attention (0.74-1.13ms GPU) 天然并行。

从系统架构角度拆解术语：
```
Timeline per MoE layer (DeepSeek-V2-Lite, A100-80GB, ~7.8ms total):
|pre-attn norm|------self-attention------|-post-attn norm-|-gate-|--expert compute--|
|   0.075ms   |       0.739ms            |    0.080ms     |0.102ms|    6.811ms       |
              |--CPU predictor 0.15ms----|                |       |                   |
              |--expert prefetch from mem (0.7-1.6ms/exp)------->|                   |

Best-case (93.03%): prefetched experts ready → zero loading latency
Worse-case (6.97%): emergency disk load (5.6-8.3ms/exp), overlapped with compute
```

术语一般如何实现？如何使用？
Per-layer predictor PyTorch 训练（30 epochs, 10M MMLU samples），CPU-only 推理。Expert 三级存储（GPU mem / CPU mem / disk）。三种模式：cloud over-provisioning (98.65%)、standard (93.03%)、edge top-1 (98.85%)。

涉及论文标题：
- Pre-Attention Expert Prediction and Prefetching for Mixture-of-Experts Large Language Models
