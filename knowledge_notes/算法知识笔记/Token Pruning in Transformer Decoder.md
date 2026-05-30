## Token Pruning in Transformer Decoder

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Token Pruning 是一种在 Transformer Decoder 推理过程中，基于某种重要性度量从 KV Cache 中移除不重要 token 的 KV 项，以减少显存占用和 Attention 计算量的技术。不同于 Encoder 模型的 token pruning（仅减少计算量），Decoder 中的剪枝同时减少显存和计算。剪枝粒度可以是 layer-wise（同一层所有 head 使用相同剪枝 mask）、head-wise（H2O 提出，每 head 独立维护分数和选择）或 token-wise。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Decoder Token Pruning 通用框架**：

```
// 超参数
cache_ratio = 0.2  // 保留 token 比例
K = int(N * cache_ratio)

// 每层每头独立维护
for layer in 1..L:
    for head in 1..H:
        // Step 1: 计算 Attention（可能加速）
        scores = flash_attn(Q, K_cache, V_cache)

        // Step 2: 更新重要性分数（不同方法的核心差异）
        importance = update_importance(scores, history, method)
        // method ∈ {"A2S" (H2O), "A2SF" (本文), "Gumbel-Softmax" (Keyformer), ...}

        // Step 3: 选择保留 token 并逐出
        keep_idx = top_k(importance, K)  // 强制保留 attention sink
        K_cache = K_cache[keep_idx]
        V_cache = V_cache[keep_idx]
        importance = importance[keep_idx]
```

**主要方法对比**：
| 方法 | 重要性度量 | 分配策略 |
|------|----------|---------|
| Local Attention | 位置（最近 token） | 全量 W 个最近 |
| H2O | A2S | 50% local + 50% A2S selection |
| A2SF | A2S + 遗忘因子 α | 100% selective |
| Keyformer | Gumbel-Softmax A2S | selective |
| StreamingLLM | 位置锚定 | 4 sink + W recent |

术语一般如何实现？如何使用？

H2O 是首个将 A2S-based token pruning 应用于 Decoder 的工作（NeurIPS 2024），代码开源。A2SF 基于 H2O 框架改进，也开源。在实际集成中，需修改 HuggingFace Transformers 的 Attention 层，在 `forward()` 后追加缓存管理逻辑。所有方法都是训练无关的即插即用方案，不增加 latency 开销。

涉及论文标题：
- A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder

---
