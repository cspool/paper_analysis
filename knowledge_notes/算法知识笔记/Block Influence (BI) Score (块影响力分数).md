## Block Influence (BI) Score (块影响力分数)

术语是什么？
Block Influence (BI) Score 是衡量 LLM 中每层（或块）对最终输出影响力的一种度量，用于指导层间非均匀剪枝率的分配。BI score 定义为层输入 $x_l$ 和输出 $y_l$ 之间的余弦相似度偏离期望值：
$$s_l = 1 - \mathbb{E}\left[\frac{x_l^{\top} y_l}{\|x_l\|_2 \|y_l\|_2}\right]$$
其中 $x_l$ 和 $y_l$ 分别是第 $l$ 层的输入和输出激活向量。

直觉：如果一层的输入和输出高度相似（余弦相似度接近 1 → $s_l$ 接近 0），说明该层对信息的变换很小，可以被大幅剪枝。反之，若 $s_l$ 大，说明该层做了大量信息变换，需保留更多通道。基于 BI scores，层间剪枝率通过 softmax 归一化分配：
$$r_l = L \cdot P_{\text{avg}} \cdot \operatorname{softmax}(-s / \varepsilon)_l$$
其中 $P_{\text{avg}}$ 是目标全局剪枝率，$L$ 是总层数，$\varepsilon=0.1$ 控制分配的温度。

从算法pipeline角度拆解：
在 UniQL 中，BI scores 的使用流程：
1. 用 WikiText-2 校准集（128 samples, seq_len=2048）前向传播一次全精度模型。
2. 对每层/块（Transformer block 或 Mamba block）记录 $x_l$ 和 $y_l$，计算 $s_l = 1 - \mathbb{E}[\cos\text{sim}(x_l, y_l)]$。
3. 对每个目标全局剪枝率 P ∈ {15%, 20%, 25%, 35%}，用 softmax 公式计算各层的剪枝率分配。
4. 混合模型（Nemotron-H, Bamba-v2）中，self-attention 层的 BI score 显著高于 SSM 层 → 分配更低的剪枝率，验证了 BI score 能准确反映不同层类型的敏感性差异。

术语一般如何实现？如何使用？
BI score 最初由 Men et al. (2024) 在 ShortGPT 中提出用于识别冗余层。MoDeGPT (Lin et al., 2025) 和 UniQL 均采用该方法进行层间剪枝率分配。计算开销低（仅需一次前向传播），不依赖梯度信息，适合 post-training 场景。注意：BI score 对校准数据集敏感——UniQL 使用 WikiText-2 而非 Alpaca 来计算 BI scores（保证与 MoDeGPT 的可比性），而权重排序和微调使用 Alpaca。

涉及论文标题：
- UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs
