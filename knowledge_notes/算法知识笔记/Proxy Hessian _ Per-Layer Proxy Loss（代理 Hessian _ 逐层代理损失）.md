## Proxy Hessian / Per-Layer Proxy Loss（代理 Hessian / 逐层代理损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Per-layer proxy loss 由 Nagel et al. (2020) 提出，是 LLM 后训练量化中广泛使用的优化目标。定义为 ℓ(Ŵ) = E_x[||(Ŵ - W)x||²] = tr((Ŵ - W)H(Ŵ - W)^T)，其中 H = E_x[xx^T] 称为 proxy Hessian 矩阵。动机：直接最小化逐层输出 MSE 而非权重的 MSE，因为不同权重对输出的影响由输入激活的二阶统计量 (H) 加权。损失是逐层定义的（per-layer），对大模型可处理。H 的估计：从校准数据集采样输入激活 x，计算外积 xx^T 的经验均值。复杂因素：最小化该损失因量化的不可微性而困难，已有方法包括 Hessian-based adaptive rounding (GPTQ)、alternating optimization (AQLM)、coordinate descent (QuIP#)等。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 QTIP 中，proxy loss 的优化由 BlockLDLQ 框架完成（见 BlockLDLQ 条目）。QTIP 的贡献不在此损失函数本身，而在用 TCQ 作为 BlockLDLQ 中的量化器（替代 VQ）——即"用什么量化"而非"如何量化"。具体伪代码见 Algorithm 5（QTIP with BlockLDLQ），核心是：对 H 做 T_y-block LDL 分解 → 逐列处理 → 每列重组为 T_x×T_y 高维序列 → Viterbi TCQ 量化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Hessian 生成：对于 Llama 1/2 使用 6144 sequences × 2048 tokens（RedPajama 数据集），Llama 3 使用 4096 sequences × 8192 tokens。H 的对角近似（基于激活方差的简单缩放）在一些方法（如 AWQ）中也有效，但 QuIP# 和 QTIP 需要完整 H 以执行 LDL 分解和自适应 rounding。每层的 H 独立估计，无需跨层传播。

涉及论文标题：
- QTIP: Quantization with Trellises and Incoherence Processing
