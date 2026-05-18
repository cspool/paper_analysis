## Multilevel Concentration

术语是什么？通过联网搜索让回答具体和精准。
Multilevel Concentration是Focus论文提出的硬件导向的冗余消除范式，在三个粒度层次上层次化地压缩VLM的视觉-语言输入：(1) Semantic Level (token级)：基于cross-modal attention的语义引导token pruning，保留文本prompt相关的视觉区域；(2) Block Level (spatiotemporal block级)：在2×2×2时空窗口内做localized similarity comparison，限制匹配范围以保持streaming和高局部性；(3) Vector Level (sub-token vector级)：将token embedding切分为32维vectors做细粒度cosine similarity matching，捕获motion引起的partial alignment。三层concentration使Focus达到平均80.19% computation sparsity，仅1.20%平均accuracy degradation。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Multilevel Concentration在VLM推理pipeline中的执行流程：
```
# Level 1: Semantic Concentration (in attention layers)
for each attention layer at positions [3,6,9,18,26]:
    attn_scores = SoftMax(Q @ K^T)  # shape: (M+T) × (M+T)
    I = attn_scores[T:, :M]  # text-to-image block, T×M
    for each image token j in 1..M:
        s[j] = max_{i=1..T, k=1..n_heads} I[i,j]^{(k)}  # cross-modal importance
    top_k_indices = StreamingBubbleSort(s, k=M*retain_ratio)  # retain 40%→10%
    prune tokens not in top_k_indices

# Level 2 & 3: Block-level + Vector-level (in FC layers)
for each FC/PV/O_proj GEMM tile output (m=1024, n=32):
    vectors = output.reshape(m, 32)  # each row is a 32-dim vector
    for each 2×2×2 spatiotemporal block (8 vectors):
        key = vectors[block[-1]]  # highest-index vector as key
        for other_vec in block[:7]:
            cosine_sim = dot(key, other_vec) / (norm(key) * norm(other_vec))
            if cosine_sim > 0.9: mark as redundant
    store deduplicated vectors + similarity_map (1×m)
```
SEC pruning ratio逐层递减：layer 3保留40%、layer 6保留30%、layer 9保留20%、layer 18保留15%、layer 26保留10% image tokens。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Multilevel Concentration的算法实现在PyTorch中，硬件实现在SystemVerilog RTL中。算法层面使用HuggingFace Transformers集成，在attention和FC层之间插入concentration操作。硬件层面通过Focus Unit（含SEC和SIC两个子模块）嵌入systolic-array accelerator的memory interface，在GEMM tile产生后立即on-chip执行concentration。算法和硬件co-design是关键：算法sparsity（~80%）被转化为硬件友好的structured tile-local稀疏，而非留给GPU处理的irregular稀疏。开源：https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

