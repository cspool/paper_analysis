## Token Compression in Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Token Compression in Sparse Attention 是一种将连续 key/value token block 压缩为紧凑表示以减少注意力计算量的技术。不同于 token pruning（直接丢弃低分 token）或 KV cache eviction（推理时选择性淘汰），token compression 通过**可学习的参数化映射**将多个 token 的信息聚合为单一压缩表示，保留高层语义信息的同时大幅降低计算复杂度。

NSA 论文中的具体实现：给定 key 序列 $\mathbf{k}_{:t}$，以 block length l 和 stride d 滑动窗口（d < l 以缓解信息碎片化），每个长度为 l 的 block 经过含 intra-block position encoding 的 MLP φ 映射为单个压缩 key：
$$\tilde{K}_t^{\text{cmp}} = \{\varphi(\mathbf{k}_{id+1:id+l}) \mid 0 \le i \le \lfloor\frac{t-l}{d}\rfloor\}$$
同理生成压缩 value $\tilde{V}_t^{\text{cmp}}$。压缩后的 KV 长度从 t 降至 ~t/d（d=16 时约 t/16）。φ 在训练中与 backbone 联合学习最优压缩策略，不同于固定 pooling（mean/max）或哈希方法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Token Compression 模块
// 输入: K_cache ∈ R^{t×d_k}, 超参 l=32, d=16

num_comp_blocks = (t - l) // d + 1
K_cmp = zeros(num_comp_blocks, d_k)
for i in range(num_comp_blocks):
    // 取连续 block: K_cache[i*d : i*d+l] ∈ R^{l×d_k}
    // Step 1: 添加 intra-block position encoding
    K_block_pe = K_cache[i*d:i*d+l] + PE_intra_block  // PE ∈ R^{l×d_k}
    // Step 2: MLP 压缩 (l×d_k → 1×d_k)
    // φ 为 2 层 MLP: Linear(d_k→4d_k) → ReLU → Linear(4d_k→d_k)
    // 先 reshape: [l, d_k] → [l*d_k]
    K_flat = K_block_pe.reshape(l * d_k)
    K_cmp[i] = phi(K_flat)  // [d_k]
// 输出: K_cmp ∈ R^{num_comp_blocks × d_k}

// 压缩 attention 计算:
scores = q_t @ K_cmp^T / sqrt(d_k)  // [1, num_comp_blocks]
attn = softmax(scores)
output = attn @ V_cmp  // V_cmp 同理生成
```

术语一般如何实现？如何使用？

Token compression 通常以两种方式实现：(a) 可学习 MLP-based（NSA 的方法），训练中学习压缩映射，表达能力最强但需额外参数和训练成本；(b) 固定 pooling-based（如 mean pooling over block），零额外参数但压缩质量受限于 pooling 策略。

实际使用中，压缩分支通常与 fine-grained selection 分支配合——压缩负责低成本的全局扫描（粗召回），selection 负责高精度的局部检索（精排序）。这种「粗召回+精排序」的二阶段设计在信息检索领域有广泛先例，NSA 将其内化到单个 attention 层中。

涉及论文标题：
- Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

---
