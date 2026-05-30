## Adaptive Sparsity (Transformer 自适应稀疏注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Adaptive Sparsity 指注意力权重的稀疏模式由输入数据动态决定（data-dependent），区别于预定义固定模式的静态稀疏方法（sliding window、block-sparse）。α-entmax 是实现自适应稀疏性的典型方法——threshold τ 根据 logits 分布动态计算，低于 τ/(α-1) 的注意力分数被精确置零。

对比：Windowed/BigBird（固定窗口，低灵活性）；Top-k（固定预算 k，不可微）；α-entmax（数据驱动动态稀疏，精确可微）。在 ModernBERT 实验中 α=1.5 产生 ~95% 整体稀疏度，α=2.0 产生 ~99%。

从算法pipeline角度拆解术语。

α-entmax 的自适应性：对每个 attention row，τ 依赖于该行所有 logits 的分布。高方差 logits → 高 τ（高稀疏），低方差 → 低 τ（低稀疏）。训练策略：α 从 1.0 线性退火到目标值确保 smooth transition。GPU 加速：动态 block mask M 检测非零 P block，pointer-increment lookup tables 跳过 null blocks。

术语一般如何实现？如何使用？

训练：α 退火 + continuous pretraining。推理：利用动态稀疏性跳过不相关 KV block，节省 HBM 带宽和 GEMM 计算。适用于长文档分类、检索和需要选择性忽略无关上下文的场景。

涉及论文标题：
- AdaSplash: Adaptive Sparse Flash Attention
