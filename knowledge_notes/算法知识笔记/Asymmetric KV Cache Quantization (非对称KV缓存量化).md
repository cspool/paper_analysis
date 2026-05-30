## Asymmetric KV Cache Quantization (非对称KV缓存量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Asymmetric KV Cache Quantization 是指对 KV Cache 中不同位置的 token 应用不同精度量化的策略。在 LogQuant 和 KiVi 等 training-free 方法中，核心思想是：(1) 部分"重要"token 保留为原始精度（BF16/FP16），(2) 其余 token 被量化到低精度（如 INT2）。这种"非对称"体现在时间/位置维度——不是所有 token 被同等对待——不同于传统对称量化（所有值统一量化到同一精度）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**非对称量化 vs 对称量化 vs 逐出**：
```
对称量化（如 GEAR, QAQ）：
  for each token t in KV_cache:
    K_quant[t] = quantize(K[t], bits=2)   // 所有token统一2-bit
    V_quant[t] = quantize(V[t], bits=2)

非对称量化（KiVi, LogQuant, JanusQuant）：
  for each token t in KV_cache:
    if t in selected_tokens:              // 选择性保留
      K_store[t] = K[t]                   // BF16全精度
    else:
      K_store[t] = quantize(K[t], bits=2) // INT2量化

逐出（H2O, StreamingLLM）：
  for each token t in KV_cache:
    if t in selected_tokens:
      K_store[t] = K[t]                   // 保留
    else:
      delete K[t]                         // 彻底删除
```

**LogQuant 的非对称设计**：
- 全精度 token 数量：2W~3W（W=⌊R/3⌋），例如 R=128 时 W=42，保留 84~126 个全精度 token
- 量化 token：其余所有 token 量化为 INT2（group_size=64, key-per-channel）
- 压缩率: 16L / (2(L-2W) + 16×2W)  ≈ 16L / (2L + 28W)

**为什么非对称优于逐出**（LogQuant Section 2.3）：
在相同的对数选择方案下，量化（降低精度）比逐出（移除 token）保留更多信息。L1 Attention 误差：
- LogQuant (2-bit quantization): 432.50
- KiVi (2-bit quantization): 556.10
- LogQuant (Eviction): 1076.70
- KiVi (Eviction): 1612.56

量化误差比逐出误差小 2-3×。原因：softmax 归一化下逐出 token 会重新分配 attention 权重，导致更大的分布偏差。

术语一般如何实现？如何使用？

在 HuggingFace transformers 中，非对称量化通过继承 Cache 类实现：(1) 维护 selected_indices 标记哪些 token 是全精度；(2) 全精度 token 直接存储在 self.key_cache[layer] 和 self.value_cache[layer]；(3) 量化 token 存储为 INT2 packed format；(4) 每次 attention 前，量化 token 经 dequantize() 恢复为 BF16 后与全精度 token 拼接。

非对称量化的适用场景：(1) 长上下文推理内存受限场景；(2) batch inference where KV cache dominates memory；(3) 与 GQA/MQA 结合——非对称量化进一步减少已缩减的 KV cache 内存。

涉及论文标题：
- LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation
- KiVi: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

---
