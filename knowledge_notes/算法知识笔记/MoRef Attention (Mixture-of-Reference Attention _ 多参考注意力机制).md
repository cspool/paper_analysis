## MoRef Attention (Mixture-of-Reference Attention / 多参考注意力机制)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MoRef Attention 是一种 training-free 的并行多参考注意力机制，受 MoE（Mixture-of-Experts）范式启发，用于在单次推理中让 Video-MLLM 的 LLM 并行 query 多个 vision reference chunks 并聚合为统一的 question token 激活。其核心流程为：(1) 对 N 个 parallel inference chunks 分别执行标准 causal FlashAttention，得到 O_i = [O_i^sys, O_i^vis, O_i^ques]；(2) 由于 causal attention 的单向性，各 chunk 的 O_i^sys 完全相同；(3) 保持各 chunk 的 O_i^vis 差异（保留各 reference 的视觉特征）；(4) 对 O_i^ques 执行跨 chunk 加权聚合：O_fusion = (Σ ω_i · O_i^ques).repeat(N)，ω_i = max(A[i]) / Σ max(A[j])，其中 A = softmax(Q^ques × K^vis^T) 为 query-vision 跨模态注意力图；(5) 组装最终输出 O^MoRef = [O^sys, O^vis, O_fusion]。加权系数 ω_i 表示 query 与各 reference 的相关性，使模型能自适应地从不同 reference 提取相关线索。计算复杂度约为 full attention 的 1/N（N 为 chunk 数），因为将长序列分解为 N 个短序列并行处理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MoRef Attention 替换 LLM 中每一层 shallow decoder 的标准 self-attention。伪代码如下：

```
for i in range(N):
    Q_i, K_i, V_i = W_Q(chunk_i), W_K(chunk_i), W_V(chunk_i)
    O_i = FlashAttention(Q_i, K_i, V_i, causal=True)

for i in range(N):
    A_i = softmax(Q_i^ques @ K_i^vis^T)  # R^{l_ques x l_vis_i}
    w_i = max(A_i) / sum(max(A_j) for j in range(N))

O_fusion = sum(w_i * O_i^ques for i in range(N))
O_fusion = O_fusion.repeat(N)

for i in range(N):
    O_i^MoRef = concat([O_i^sys, O_i^vis, O_fusion])
    chunk_i = chunk_i + O_i^MoRef
    chunk_i = chunk_i + FFN(LayerNorm(chunk_i))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Free-MoRef 论文中 MoRef Attention 直接替换 LLaVA-Video-7B 中 Qwen2-7B 的 self-attention layers（shallow layers 0 到 L-1）。兼容标准 FlashAttention（causal 模式），额外计算仅需一次 A = softmax(Q^ques × K^vis^T)，可忽略不计。gating weights ω_i 体现 query-aware 选择。FLOPs：128 frames (N=2) 时 110.4% baseline (vs full attention 400%)，256 frames (N=4) 时 163.2% (vs 1600%)。超参数：最佳 N = input_frame_num/64, M=64。

涉及论文标题：
- Free-MoRef__Instantly_Multiplexing_Context_Perception_Capabilities_of_Video-MLLMs_within_Single_Inference
