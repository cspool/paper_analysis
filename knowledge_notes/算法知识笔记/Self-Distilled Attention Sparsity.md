## Self-Distilled Attention Sparsity

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Self-Distilled Attention Sparsity（自蒸馏注意力稀疏）是 SeerAttention 系列提出的 post-training 训练范式：让一个轻量级 AttnGate 模块通过蒸馏学习原始预训练模型自身的注意力稀疏模式，无需修改原始模型参数。与 knowledge distillation（大模型教小模型）不同，这里"教师"和"学生"是同一个模型——AttnGate 学习的是原始模型注意力分布中的稀疏结构。

训练流程：
1. 用原始模型对训练数据做完整 attention forward
2. 对完整 attention scores 做 block-level maxpooling（prefill 阶段 2D maxpool，decode 阶段 1D column-wise maxpool）
3. 对 GQA group 内 query heads 再做一次 maxpool，得到 KV-head 级别的 ground truth
4. 归一化 ground truth 使和为 1
5. AttnGate 通过 KL divergence loss 学习预测与 ground truth 一致的块激活分布
6. 仅更新 AttnGate 参数（通常 <1% 模型参数），原始模型权重冻结

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Self-Distillation Training Loop
for batch in dataloader:  # e.g. OpenR1-MATH-220K, packed to 32K seq
    # === Teacher: 完整 attention forward（修改版 FA2 kernel）===
    # modified_fa2_kernel 同时输出 attention output 和 block-level ground truth
    O_dense, ground_truth = modified_fa2_forward(Q, K, V, block_size)
    # ground_truth 生成:
    #   1. 计算 full attention scores A = QK^T/sqrt(d)
    #   2. Column-wise 1D maxpool: A_pooled[t] = max(A[t, b*s : (b+1)*s])
    #   3. GQA group 内 maxpool: gt = max over query heads in each group
    #   4. Normalize: gt = gt / sum(gt)
    
    # === Student: AttnGate 预测 ===
    S_pred = attngate_forward(Q, K, block_size)  # [num_kv_heads, num_blocks]
    
    # === Loss ===
    loss = KL_divergence(S_pred, ground_truth)
    loss.backward()  # 仅 AttnGate 参数有梯度
    
# 训练配置: 0.4B tokens, batch_size=16, 800 steps, lr=1e-3, cosine decay
# 硬件: AMD MI300x GPU, DeepSpeed ZeRO-2
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现上需要：(1) 修改 FlashAttention kernel 以在计算 attention 的同时生成 block-level ground truth（复用 FlashAttention 的 block-level rowmax 等中间结果，几乎零额外开销）；(2) 将 AttnGate 模块插入每层 attention layer。与从头预训练稀疏注意力（如 NSA、MoBA）相比，自蒸馏方法可以将稀疏注意力以 plug-in 方式添加到任意预训练模型中，训练开销极小（8B 模型仅需 12 GPU hours on MI300x）。

涉及论文标题：
- SeerAttention-R: Sparse Attention Adaptation for Long Reasoning
- SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs (NeurIPS 2025)

---
