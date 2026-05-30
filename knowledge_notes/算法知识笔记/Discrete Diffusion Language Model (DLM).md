## Discrete Diffusion Language Model (DLM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

离散扩散语言模型（Discrete Diffusion Language Model, DLM）是一类将文本生成建模为离散token空间上迭代去噪过程的生成模型。与自回归模型逐token从左到右生成不同，DLM初始化整个序列为[MASK] token（absorbing state），然后通过多步反向扩散过程逐步预测和填充token。核心数学框架：(1) 前向过程：$q(x_t|x_0) = \alpha_t x_0 + (1-\alpha_t) \mathbf{m}$，其中$\alpha_t = \prod_{i=1}^t(1-\beta_i)$，$\mathbf{m}$为[MASK]的one-hot表示；(2) 反向过程：$p_{\theta}(x_{t-1}|x_t)$通过神经网络学习近似反向transition，使用bidirectional attention同时建模所有位置；(3) 训练损失：reweighted cross-entropy $\mathcal{L}_D = \mathbb{E}_t[\frac{1}{t}\mathbb{E}_{q(x_t|x_0)}[-\sum_n \delta_{x_t^n,\mathbf{m}}(x_0^n)^\top \log f_{\theta}(x_t)^n]]$，仅对masked位置计算loss。代表模型：MDLM (NeurIPS 2024)、Dream 7B（从Qwen2.5-7B微调，580B tokens训练）、LLaDA、Mercury、Gemini Diffusion。Dimple使用Dream作为DLM backbone。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

DLM的训练和推理pipeline（以absorbing state DLM为例）：

```
# === 训练阶段 ===
Input: 文本序列 x_0, 词汇表 V (含[MASK] token)
Forward pass:
  1. 采样时间步 t ~ Uniform(0, 1]
  2. 计算mask概率: p_mask = 1 - α_t
  3. 对每个token独立以概率p_mask替换为[MASK]: x_t = mask(x_0, p_mask)
  4. 将x_t输入bidirectional Transformer (full attention over all positions)
  5. 对所有被masked的位置输出logits预测原始token: f_θ(x_t) ∈ R^{L×V}
  6. 计算损失（仅masked位置）: L = -Σ_{n: x_t^n=[MASK]} log softmax(f_θ(x_t)^n)[x_0^n] / t

# === 推理阶段（以MaskGIT为例）===
Input: 目标序列长度 L, 解码步数 T
初始化: x_T = [[MASK], ..., [MASK]]  (L个[MASK])
For t = T down to 1:
  1. z_t = f_θ(x_t)  # bidirectional forward
  2. p_t = softmax(z_t)
  3. confidence c^(i) = max(p_t^(i)) for i in masked positions
  4. 选择K = ceil(L * t/T) 个最高置信度位置
  5. 对选中位置采样: x_{t-1}^(i) ~ Categorical(p_t^(i))
  6. 其余位置保持[MASK]
Output: x_0（所有位置已去mask的token序列）
```

Annotations: $x_0$: 无噪声token序列; $x_t$: 时间步$t$的噪声序列; $\alpha_t$: 信号保留率; $\beta_i$: 每步噪声率; $f_\theta$: bidirectional Transformer; $T$: 总解码步数（8-64步）; $L$: 目标序列长度; 关键区别：DLM使用bidirectional attention（vs AR的causal），所有位置同时预测（vs AR逐位置）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DLM通常基于预训练AR LLM初始化：从AR LLM checkpoint加载权重，将causal attention替换为bidirectional，使用masked language modeling loss在大规模数据上训练。Dream (https://github.com/DreamLM/Dream) 从Qwen2.5-7B初始化，使用580B tokens训练。推理使用迭代解码策略（MaskGIT或变体），通过confidence-based selection逐步去mask。优势：并行解码（每步可同时预测多个token）、bidirectional context利于planning/infilling、可控性（精确输出长度和结构）。

**Fast-dLLM的加速贡献**：Fast-dLLM针对DLM推理提出了训练无关的加速方法：(1) Block-wise近似KV Cache利用双向注意力相邻步KV激活高余弦相似度的特性，在分块解码中缓存和复用prefix/suffix的K/V矩阵，减少重复的全注意力计算；(2) Confidence-Aware Parallel Decoding通过理论保证的安全并行解码减少总解码步数。两者结合在LLaDA 8-shot gen_len=1024上实现27.6×端到端加速。

涉及论文标题：
- Dimple Discrete Diffusion Multimodal Large Language Model with Parallel Decoding
- Fast-dLLM Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding
