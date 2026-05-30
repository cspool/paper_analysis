## Token-Dropless MoE Training（无 Token 丢弃 MoE 训练）

术语是什么？
Token-dropless MoE Training 是一种 MoE 训练范式，指不通过 capacity_factor 限制丢弃 token 来达到负载均衡，而是处理 Router 分配给每个 expert 的**全部** token（包括极端不均衡的情况）。由于不丢弃 token，dropless 训练的信息保留更完整，模型质量更高（MegaBlocks 实验：dropless 的 validation loss 改善 0.26 比 capacity_factor=1 的 0.15，1.73× 改善）。MoE Parallel Folding 框架通过灵活的 token dispatcher 同时支持 dropless 和 token-dropping 两种训练范式。

从算法pipeline角度拆解术语：
Token-dropless 训练的流程（以 Mixtral 8x22B, EP=8, ETP=1 为例）：

```
1. Router: probs, indices = Router(local_input)  # 每个 token 分配 top-k expert
2. 无 capacity 检查：不丢弃任何 token
3. Permutation: 将同一 expert 的 token 紧凑排列（变长 batch per expert）
4. All-to-All-V: 跨 EP 组发送变长 token batch 到对应 expert 所在 rank
5. Expert GEMM: 各 expert 独立计算（batch size 可变）
6. All-to-All-V: token 返回原 rank
7. Unpermutation + weighted sum

注意：因 batch size 可变，无法使用统一 batched GEMM。
Megatron-Core/MegaBlocks 使用 GroupedGEMM 或 block-sparse kernel 处理变长 expert batch。
```

术语一般如何实现？如何使用？
- Megatron-Core 支持 dropless 训练作为默认范式（benchmark 时可用 token-dropping 避免负载不均的性能抖动）
- 实现需变长 token 处理能力：All-to-All-V（可变消息长度）、GroupedGEMM（不等大小 batch）、block-sparse kernel
- Token-dropless 训练可能因负载不均导致某些 expert 处理极大量 token，造成 straggler 问题。此时可结合 MoE Parallel Folding 优化通信，缓解瓶颈

涉及论文标题：
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core
