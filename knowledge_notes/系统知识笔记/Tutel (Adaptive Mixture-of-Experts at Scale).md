## Tutel (Adaptive Mixture-of-Experts at Scale)

术语解释
Tutel 是 Microsoft 开发的高性能 MoE 系统框架，专门优化 All-to-All 通信和 expert 调度。BigMac 论文使用 Tutel 验证 BigMac 结构在已优化 All-to-All 的系统上仍有显著加速。

术语是什么？
Tutel 的核心优化技术：

1. **2DH All-to-All**：二维层次化 All-to-All——将 intra-node（NVLink，高带宽）和 inter-node（IB/RoCE，低带宽）通信分层处理，减少跨节点数据量。
2. **Overlap**：将 All-to-All 通信与 expert FFN 计算重叠（通过将输入 token 拆分为多个 chunks，chunk 发送完即开始计算，不等待全部 token 传输完成）。BigMac 实验中设置 overlap degree=4。
3. **Adaptive Parallelism**：根据 load 动态调整 expert parallelism 和 data parallelism 的混合配置。
4. **Dynamic Capacity Factor Adaption (f=∞)**：自动调整 expert capacity，避免 token dropping 的同时减少 token padding。
5. **Fast Encode/Decode**：优化 gating 和 token dispatch/combine 的 kernel 效率。

BigMac 在 Tutel 上的实验结果：即使 Tutel 已优化 All-to-All（2DH + overlap），BigMac 仍获得 1.71-3.09× 训练加速。这说明 BigMac 的算法级通信缩减（DCCA）与 Tutel 的系统级优化是正交叠加的。

从系统架构角度拆解术语：
Tutel 在 BigMac 训练中的运转流程：

```
Tutel MoE Layer (with BigMac DCCA):

# Tutel 配置: 2DH All-to-All + overlap degree=4 + dynamic capacity

# 1. Token partition into 4 chunks (overlap degree)
chunks = partition(tokens, 4)

# 2. Pipeline: communication overlapped with computation
for chunk in chunks:
    # Descend (BigMac)
    chunk_low = chunk @ W'_down
    # All-to-All dispatch (2DH: intra-node then inter-node)
    async alltoall_scatter(chunk_low)  # 2DH hierarchical
    # While waiting for chunk_i dispatch, compute chunk_{i-1} expert FFN
    compute_expert_ffn(previous_chunk)
    sync()
```

术语一般如何实现？如何使用？
- 开源：https://github.com/microsoft/tutel
- 与 Megatron-LM、DeepSpeed 等框架集成
- 安装：`pip install tutel`，替换 MoE 层的前向实现
- 通过 2DH 通信、overlap degree、capacity factor 等参数调优

涉及论文标题：
- BigMac A Communication-Efficient Mixture-of-Experts Model Structure for Fast Training and Inference

---
