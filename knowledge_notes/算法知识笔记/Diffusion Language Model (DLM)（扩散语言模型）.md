## Diffusion Language Model (DLM)（扩散语言模型）

术语是什么？通过联网搜索让回答具体和精准。
Diffusion Language Model (DLM) 是将扩散模型（Diffusion Models）或流模型（Flow-based Models）应用于语言建模的一类生成模型，与自回归（AR）语言模型形成互补范式。DLMs通过迭代去噪（denoising）而非逐token自回归生成文本，支持并行生成、双向上下文和迭代优化。根据Denoising和离散化的空间不同，DLMs分为两大类：(1) Continuous DLMs：将离散token映射至连续表示（embedding/simplex）后执行去噪，如Diffusion-LM、CDCD、ELF；(2) Discrete DLMs：直接在离散token space定义扩散过程，使用masked/uniform transition matrices，如MDLM (absorbing-state masking)、Duo (uniform diffusion)。2024-2025年DLMs快速发展：LLaDA (8B)是首个与AR模型竞争力相当的离散扩散LLM，Dream 7B从AR权重初始化。DLMs的核心优势包括：并行生成（达~1000 tokens/sec）、双向上下文（更丰富的表示）、迭代优化能力、可控生成长度和格式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

DLM的通用算法pipeline（以Continuous DLM如ELF为例）：
```
// 训练阶段
1: tokens s = [s_1,...,s_L] ∈ V^L
2: x = Embed(s)                        // 离散token → 连续embedding
3: t ~ schedule, ε ~ N(0,I)
4: z_t = noisify(x, ε, t)              // 加噪（取决于DLM类型）
5: x̂ = model(z_t, t)                   // 去噪预测
6: loss = L(x̂, s)                      // 不同的DLM在此处分岔：
   // Continuous DLM: L = MSE(x̂, x) 在连续空间
   // 或 L = CrossEntropy(unembed(x̂), s) 每步离散化
   // Discrete DLM: L = -log p(s|z_t) via transition matrix

// 推理阶段
1: z_0 ~ P_noise                      // 初始噪声分布
2: for step = 0 to T-1:
3:     ŝ 或 x̂ = model(z_step, t_step) // 逐步去噪
4:     z_{step+1} = sampler_step(...)  // ODE/SDE/masking step
5: tokens = argmax(unembed(z_T))       // 最终离散化
```

DLM的分类体系（基于ELF论文Tab.2的survey）：
| 类别 | 状态空间 | 训练per-step离散化 | 推理per-step离散化 | 解码器 | 代表方法 |
|------|---------|-------------------|-------------------|--------|---------|
| Embedding-space Diffusion LMs | learn/fix emb | Yes | Yes | No | Diffusion-LM, CDCD, SeqDiffuSeq |
| Simplex Diffusion LMs | simplex | Yes | Yes | No | SSD-LM, TESS, TESS 2 |
| Latent Diffusion LMs | fix enc | No | No | Yes (单独) | LD4LG, PLANNER, Cosmos |
| Flow-based LMs | simplex/one-hot/emb | Yes (部分) | Yes (部分) | Yes/No | FLM, LangFlow, DFM |
| **ELF** | fix enc | **No** (仅最后步) | **No** (仅最后步) | **No** (共享权重) | ELF |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DLMs的实现涉及多个设计选择：(1) 状态空间：Continuous DLMs使用预训练编码器（如T5）或可学习embedding将token映射到连续空间；Discrete DLMs直接在vocabulary空间定义转移矩阵；(2) 噪声调度：Continuous DLMs通常使用Gaussian noise with logit-normal/log-normal schedule，Discrete DLMs使用masking probability schedule；(3) 离散化策略：Continuous DLMs需将连续状态映射回离散token——常用rounding（最近邻embedding）、unembedding layer（可学习投影矩阵）、或argmax over simplex；(4) 采样器：Continuous DLMs使用ODE/SDE solver，Discrete DLMs使用ancestral sampling/predict-and-noise等。ELF展示了Continuous DLMs的minimalist设计——仅在最末步离散化、无需单独decoder、充分发挥连续空间灵活性——取得了与Discrete DLMs竞争力相当甚至更优的质量。开源实现：ELF (https://github.com/lillian039/ELF)、MDLM (https://github.com/kuleshov-group/mdlm)、Duo (https://github.com/s-sahoo/duo)、E2D2 (https://github.com/kuleshov-group/e2d2)。

涉及论文标题：
- ELF: Embedded Language Flows

---

