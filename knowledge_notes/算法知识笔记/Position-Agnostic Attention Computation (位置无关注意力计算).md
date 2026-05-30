## Position-Agnostic Attention Computation (位置无关注意力计算)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Position-Agnostic Attention Computation 是 LogQuant 证明并利用的一个 Attention 属性：在解码阶段的 Scaled Dot-Product Attention 中，Key 和 Value 矩阵中 token 的排列顺序不影响最终输出结果。数学上：对于任意置换 P（{1,…,N} 的重排），有 A·V = A_P·V_P。其中 A = softmax(QK^T)，A_P 是 K 经置换 P 后的注意力分布，V_P 是 V 经置换 P 后的 Value 矩阵。

这一属性源于：softmax 对每个 token 独立计算后归一化，而最终的 A·V 是对所有 token 的 Value 加权求和。加权求和的交换律保证顺序无关。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**置换不变性证明**：
```
原始 Attention:  O = A · V
                = Σ_i a_i · v_i        // a_i = softmax(q·k_i^T/√d)_i
                                        // 对 i 的求和，交换律保证顺序无关

经置换 P 后:
  K_P = K[P]                           // 按 P 重排 Key
  V_P = V[P]                           // 按 P 重排 Value
  A_P = softmax(Q · K_P^T)             // 注意力分布对应变化
  O_P = A_P · V_P
      = Σ_i a_{P(i)} · v_{P(i)}        // 由于求和遍历全部 token
      = Σ_j a_j · v_j                  // 每个 token 恰好贡献一次
      = O                              // 与原始输出相同
```

**在 LogQuant 中的应用**：
```
// 传统方式（KiVi）：KV Cache 按原始位置存储
// [BF16_token_1, INT2_token_2, ..., BF16_token_R, ..., INT2_token_N]
// → 全精度和量化 token 交错存储，内存碎片化

// LogQuant 方式：利用置换不变性重排
// [INT2_token_1, INT2_token_2, ..., INT2_token_{N-R}, BF16_token_1, ..., BF16_token_R]
// → 全精度和量化 token 分别连续存储 → 更好的内存局部性
```

术语一般如何实现？如何使用？

在 LogQuant 实现中，position-agnostic 属性通过 concat 操作体现：在 Cache 类中，全精度 token 被连续存储在 cache 的一端，量化 token 被连续存储在另一端（或反之），中间无交错。这通过继承 HuggingFace Cache 类时修改 K/V 的存储布局实现——无需修改 attention 计算本身。

使用方式：(1) 任何需要将 KV Cache 按不同精度/格式分组存储的场景均可利用此属性；(2) 可与 fused dequantization-attention kernel 结合——连续的全精度 K/V 段避免了 gather/scatter 操作；(3) 论文注明"未来的 operator fusion 优化将在此属性基础上直接在量化 cache 上计算 attention"。

注意：此属性仅适用于解码阶段（每次仅 1 个 query token）。预填阶段（prefill）因 softmax 的 causal mask 依赖 token 顺序，不适用此属性。

涉及论文标题：
- LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation

---
