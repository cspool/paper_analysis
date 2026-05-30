## Memory Interference in Linear Sequence Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Memory Interference（记忆干扰）是线性序列模型的核心瓶颈：当序列中所有信息被压缩到单一固定大小的 memory state M ∈ R^{d×d} 时，新 token 的 K/V 写入会覆盖或衰减之前存储的信息表示。这导致模型在 recall-intensive 任务上的表现远不如为每个 token 维护独立 KV cache 的 Transformer。

数学模型：在线性注意力中，M_t = M_{t-1} + k_t^T v_t。当新的 k_t^T v_t 包含与已存储信息正交或冲突的内容时，无法将两者同时保留——它们被"搅拌"到同一个 memory state 中。即使有 forget gate a_t，也只是整体衰减，无法选择性保留特定信息。

MoM 通过多 memory 分离解决此问题：token 被路由到不同 memory，非激活 memory 保持 M_t^m = M_{t-1}^m 不变。这实现了"信息隔离"——不同类型的信息存储在不同的 memory 中，互不干扰。MoM 论文实验（Table 5）验证了各 memory 确实形成了专业化：Memory-1 偏好基础词/动词，Memory-2 偏好专有名词/科技术语，Memory-3 偏好技术术语/形容词，Memory-4 偏好疑问词/不完整名词。

从算法pipeline角度拆解术语。

```
# 单一 memory (有记忆干扰):
M_t = a_t * M_{t-1} + k_t^T v_t
# 问题: k_t^T v_t 会与 M_{t-1} 中任意行产生交互，
# M_{t-1} 中与 k_t 相似的信息被加强，相异的被稀释

# MoM 多 memory (无干扰):
if token t routed to memories {2, 4}:
  M_t^2 = GatedDeltaNet(M_{t-1}^2, k_t^2, v_t^2)   # 仅更新 Memory-2
  M_t^4 = GatedDeltaNet(M_{t-1}^4, k_t^4, v_t^4)   # 仅更新 Memory-4
  M_t^1 = M_{t-1}^1  # 未激活，保持不变——无干扰！
  M_t^3 = M_{t-1}^3  # 未激活
```

术语一般如何实现？如何使用？

Memory Interference 的缓解方法分两类：(1) Gating-based: 通过 forget gate / input gate 控制信息衰减（GLA, HGRN2, G-DeltaNet 等）；(2) Separation-based: 通过多 memory 分离不同信息（MoM）。两种方法互补——MoM 在 Gated DeltaNet 基础上叠加多 memory 分离。Router 的 auxiliary loss 确保负载均衡，防止某些 memory 成为 bottleneck。

涉及论文标题：
- MoM: Linear Sequence Modeling with Mixture-of-Memories

---
