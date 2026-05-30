## Expert Trimming (MoE 专家修剪)

术语解释
Expert Trimming 是对 MoE 模型进行结构化压缩的一大类方法，通过识别和移除冗余的结构化模块（experts、MoE layers、transformer blocks）来减少参数数量、内存占用和计算开销。由 He et al. (2025) 在统一 MoE 压缩框架中提出，与 Expert Slimming（压缩单个 expert 内部权重）互补。

术语是什么？
Expert Trimming 的通用形式：T ← T'，其中 T 为原始专家/层/块集合，T' 为保留的子集。根据移除粒度分为三个层次：
1. **Expert Drop**（fine-grained）：移除部分不重要 expert，保留 router 对剩余 expert 的选择
2. **Layer Drop**（medium-grained）：移除整个 MoE 层（含对应 Norm 模块），消除该层的所有 expert 计算和通信
3. **Block Drop**（coarse-grained）：移除整个 Transformer block（Attention + MoE layer + Norms），同时消除 Attention 计算和 KV-Cache

Expert Drop 虽减少参数量，但保留的 MoE 层内仍有 costly computation 和 complex communication（分布式 All-to-All），speedup 微乎其微（<1% at 12.5% experts dropped）。Layer Drop 和 Block Drop 通过粗粒度移除彻底消除对应计算和通信，speedup 显著提升。

从算法pipeline角度拆解术语：
```
=== Expert Drop ===
# 重要性评分 S(E_i) = (1/|X|) * Σ G_i(x)
# Layer-wise: T'(l) = {E_t^(l)} where S(E_t) ∈ TopK({S(E_i)}, n')
# Global: T'(l) = {E_t^(l)} where S(E_t) ∈ TopK(∪_{j}{S(E_i^(j))}, n'*L)

=== Layer Drop ===
# 移除完整 MoE layer + Norm
S^(NM)_l = mean(cos_sim(x', x' + MoE_l(Norm_l(x'))))  # 层冗余度
# 按 S^(NM) 降序排列，移除 Top-K 层

=== Block Drop ===
# 移除完整 Transformer block (Attention + MoE + Norms)
S^(NM)_l = mean(cos_sim(x^l, y^l))  # block 级冗余度
# 按 S^(NM) 降序排列，移除 Top-K blocks
```

关键发现：MoE 层比 Dense 层更冗余——同深度 Mixtral-8×7B (MoE) vs Mistral-7B (Dense)，Drop 8 layers 时 MoE -7.0 vs Dense -24.3 MMLU。

术语一般如何实现？如何使用？
- Expert Drop：基于路由分选择保留 expert，不需要任何训练；可选 post-finetuning 恢复性能
- Layer Drop / Block Drop：用 calibration data（如 128 samples from C4, seq_len=2048）计算每层/块的 cosine similarity，相似度越高 → 冗余越大 → 优先 drop
- Drop 模式：深层 layers/blocks 优先被 drop（与 Xu et al. 2024 / Men et al. 2024 一致），因为深层更冗余
- 鲁棒性：相似度对 calibration 数据选择鲁棒——不同样本量（128 vs 更多）和不同数据集（C4/Lima/MetaMathQA）的相似度模式一致
- 集成策略：先 Expert Slimming（quantization）后 Expert Trimming（Layer/Block Drop），即"S+T" order，性能稍优于 "T+S"
- 效果：AWQ + Block Drop B5/32 on Mixtral-8×7B: 5.94× speedup, 21.9GB memory, Avg=68.0 (95.1% of baseline)

涉及论文标题：
- Demystifying the Compression of Mixture-of-Experts Through a Unified Framework
