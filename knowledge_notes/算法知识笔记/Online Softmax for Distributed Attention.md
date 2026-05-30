## Online Softmax for Distributed Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Online Softmax 是一种数值稳定的 Softmax 计算方法，在分布式注意力中用于将各设备上独立计算的部分注意力结果合并为全局精确结果。传统 Softmax 需要两次遍历（一次求 max 用于数值稳定，一次求和归一化），Online Softmax 通过维护 running max 和 running sum 实现单次遍历，使各设备可独立计算部分结果后通过通信合并。

在 RINGATTN 中，online softmax 用于 ring-style 通信：每个 host 计算当前块的 max 和 sum，传递给下一个 host，后者用接收的 (m, l) 修正自己的计算。在 STARATTN/APB 的 decoding 阶段（stage-2），online softmax 用于 Gather+MergeScore：各 host 独立计算 attention，然后通过 Gather 收集各 host 的 (A_h, lse_h)，用 lse（log-sum-exp）合并为全局 attention。

从算法pipeline角度拆解术语。

**APB Decoding 中的 Online Softmax MergeScore**：

```
// 每 host 独立计算（Algorithm 3）
Q, K, V = qkv_proj(H)

if h < H:
    A_h, lse_h = Attention(Q, K_cache[h], V_cache[h])   // 对本地 KV cache
else:  // 最后一个 host
    A_h, lse_h = Attention(Q, [K_cache[H], K], [V_cache[H], V])

// Gather 所有 host 的部分结果
A_1..A_H, lse_1..lse_H = Gather(A_h, lse_h)

// MergeScore: 利用 online softmax 合并
// A_global = Σ_h A_h * exp(lse_h - lse_max) / Σ_h exp(lse_h - lse_max)
lse_max = max(lse_1, ..., lse_H)
weights = [exp(lse - lse_max) for lse in lse_1..lse_H]
A_global = sum(w_h * A_h for w_h, A_h in zip(weights, A_1..A_H)) / sum(weights)

H_out = FFN(A_global)
```

术语一般如何实现？如何使用？

Online Softmax 以 kernel 形式实现在 FLASHATTN 和各类分布式注意力框架中。APB 复用 STARATTN stage-2 的 online softmax 解码方案（Algorithm 3）。在 HuggingFace Transformers 中，通过自定义 attention forward 函数集成。FLASHATTN（Dao, 2024）是 PyTorch 中最广泛使用的 online softmax attention 实现。

涉及论文标题：
- APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs

---
