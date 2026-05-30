## Asymmetric KV Cache Quantization (非对称KV Cache量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Asymmetric KV Cache Quantization 是 KIVI 提出的核心设计理念：对 key cache 和 value cache 使用不同维度的量化策略——key per-channel（沿特征维度）、value per-token（沿序列维度）。这里的 "asymmetric"（非对称）不是指通常量化中 scale_pos≠scale_neg 的 asymmetric quantization，而是指 key 和 value 两个 caches 使用非对称的量化维度选择。

设计理由：(1) Key cache 少数固定 channel 存在极大 outlier（与 SmoothQuant/AWQ 观察一致），per-channel 量化将 error 隔离在 outlier channel 内；(2) Value cache 虽无 outlier 模式，但因 attention output = sparse-weighted sum of value tokens，per-token 量化保护重要 token 不受干扰。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KIVI asymmetric quantization 与 baseline uniform quantization 的对比：

```
# Baseline (FlexGen): 统一 per-token 量化
Q(X_K) = GroupQuant(X_K, dim=token)  # key per-token
Q(X_V) = GroupQuant(X_V, dim=token)  # value per-token
# 问题: INT4 OK, INT2精度崩塌 (Llama-2-13B CoQA: 66.37→52.93)

# KIVI: 非对称量化
Q(X_K) = GroupQuant(X_K, dim=channel)  # key per-channel
Q(X_V) = GroupQuant(X_V, dim=token)    # value per-token
# 2bit效果: Llama-2-13B CoQA: 66.37→66.23 (几乎无损)
```

KIVI 实验证实的其他配置均不可行：
- 2bit (K per-token, V per-channel): CoQA 2.80（全坏）
- 2bit (K per-channel, V per-channel): CoQA 2.88（全坏）
- 2bit (K per-token, V per-token): CoQA 52.93（差但能用）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现关键：(1) 对 key 使用 `GroupQuant(X_K, dim=channel, numGroup=l//G)`；(2) 对 value 使用 `GroupQuant(X_V, dim=token, numGroup=d//G)`。KIVI 依赖 CUDA/Triton kernel 实现这两种量化方向的 fused dequant+matmul。代码开源：https://github.com/jy-yuan/KIVI。该设计被后续 KV Cache 量化工作（KVQuant、GEAR、PM-KVQ）广泛引用和扩展。PM-KVQ 在 KIVI 的非对称 per-channel Key + per-token Value 量化基础上，进一步叠加渐进量化和块级内存分配，并将首 token INT16 + 128 token 滑动窗口作为默认保留策略。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

---
