## MaskGIT Decoding Algorithm

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MaskGIT（Masked Generative Image Transformer, CVPR 2022）是一种基于置信度的迭代并行解码算法，最初设计用于图像生成，后被广泛应用于离散扩散语言模型。核心思想：将所有位置初始化为[MASK]，每步通过bidirectional Transformer预测所有masked位置的token分布，按置信度排序选择最高置信度的K个位置解码，其余保持[MASK]，迭代直到完成。与自回归解码相比：每步可并行解码多个token，总步数远小于序列长度。MaskGIT使用cosine mask schedule $\gamma(t/T)$ 决定每步解码的token数量：$\gamma(r) = \cos(\pi r/2)$，从$\gamma(0)=1$（全mask）到$\gamma(1)=0$（全解码）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Input: 序列长度 L, 总步数 T, 温度 τ
初始化: x_T = [[MASK], ..., [MASK]]

For step t = T, T-1, ..., 1:
  z_t = f_θ(x_t)  # bidirectional parallel predict
  p_t = softmax(z_t / τ)
  For each masked i: c^(i) = max(p_t^(i))  # confidence
  n_masked = count([MASK])
  K = ceil(n_masked * cos(π * (t/T) / 2))  # cosine schedule
  或固定策略: K = ceil(L * t/T)
  I_t = TopK({c^(i)}, K)
  For i in I_t: x_{t-1}^(i) ~ Categorical(p_t^(i))
  其余保持[MASK]

Return: x_0
```

Annotations: T典型值8-12（图像）或等于L（文本）；τ=0时退化为greedy；Dimple评估时设τ=0、每步1 token以保证确定性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MaskGIT最初在google-research/maskgit (JAX)实现，后被移植到PyTorch。在DLM中，Dream使用MaskGIT作为默认解码算法。实现要点：(1) confidence使用pre-revision概率（在temperature/top-p调整前），避免revision后概率退化；(2) temperature调度——初始高temperature增加多样性，最终低temperature确保质量。优势：并行解码2-64x speedup vs AR。局限性：需预定义序列长度；对mask schedule敏感。

涉及论文标题：
- Dimple Discrete Diffusion Multimodal Large Language Model with Parallel Decoding
