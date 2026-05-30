## Fully Differentiable Sparse Attention (完全可微稀疏注意力 / 端到端可训练稀疏注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fully Differentiable Sparse Attention 是指稀疏模式可以端到端通过梯度下降学习的注意力机制。DMA 首次在 content-position 双感知的稀疏注意力中证明了：(1) 动态 mask 和稀疏权重不阻塞梯度；(2) 保留路径的梯度与 full attention 严格一致；(3) 即使 mask 生成涉及 top-w 离散操作，梯度也能完整流向所有参数（Δ, A, Q, K, V）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**梯度流分析**（单 head, 单 time step）：

设 I_h 为选中的 w 个位置索引集。前向：
```
s_{h,j} = (q_h · k_{h,j})/√d_h + m_{h,j}
p_{h,j} = exp(s_{h,j}) / Σ_{j'∈I_h}exp(s_{h,j'})  for j∈I_h, else 0
o_h = Σ_{j∈I_h} p_{h,j} · v_{h,j}
```

反向梯度流：
```
1. dv_{h,j} = p_{h,j} · do_h          (j∈I_h), 0 (j∉I_h)
2. dp_{h,j} = v_{h,j} · do_h
3. ds_{h,j} = p_{h,j} · (dp_{h,j} - Σ_{j'∈I_h} p_{h,j'} · dp_{h,j'})
   — 对 mask 位置 p_{h,j}=0 → ds_{h,j}=0 (自然为零，非近似)
4. dm_{h,j} = ds_{h,j}                — 梯度直接流向 mask 参数
5. dq_h = Σ_{j∈I_h} ds_{h,j} · k_{h,j}/√d_h
6. dk_{h,j} = ds_{h,j} · q_h/√d_h    (j∈I_h), 0 (j∉I_h)
```

**关键洞察**：
- 对选中位置 j∈I_h，梯度计算与 full attention 完全一致——DMA 仅裁剪了对可忽略贡献位置的算子链。
- top-w 操作在 backward pass 中不需要梯度：未选中位置 p_{h,j}=0 → ds_{h,j}=0，跳过计算和梯度传播是数学上正确的结果。
- dM = dS 的等价关系使 kernel 只需局部重算 S 而不需额外存储中间 mask 梯度张量。
- 门控参数 A 和权重 Δ 直接接收 attention weights 的梯度，快速 shaping head 特化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DMA 的 CUDA kernel（flash-sparse-attention）实现了完整的 forward+backward 可微 pipeline。Backward pass（Algorithm 2 in paper）与 forward 共享统一 skip logic——相同 mask block judge 决定是否加载 K/V tile。gradient chain 包含 fused bias gradients。与 MagicPIG（使用离散 LSH 采样→不可微）、ClusterKV（k-means 聚类→梯度阻断）等方法对比，DMA 的每个操作都是可微的。

涉及论文标题：
- Trainable_Dynamic_Mask_Sparse_Attention
