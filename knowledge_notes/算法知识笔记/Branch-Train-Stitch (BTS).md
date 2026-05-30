## Branch-Train-Stitch (BTS)

术语解释
BTS 是由 Qizhen Zhang et al. (Meta, 2025) 提出的三阶段训练算法，将多个独立训练的领域专用 LLM Expert 合并为一个通用（generalist）模型。BTS 在保持 Expert 参数完全冻结的前提下，通过插入并训练轻量 Stitch Layer（264M 参数 vs 总 11B 参数）建立 Expert 之间的可学习连接，实现比 Expert Upcycling 和 Expert Merging baseline 更好的平均下游任务性能。

术语是什么？
BTS 算法包含三个阶段：
1. **Branch**：复制预训练的 Seed 模型 $m_0$ 为 $n$ 份副本 $m_1, ..., m_n$。
2. **Train**：每个 Expert $m_i$ 在领域专用数据 $\mathcal{D}_i$（如 Code、Math、Multilingual）上独立继续预训练。完成后冻结所有参数。
3. **Stitch**：在 Seed（Hub）和 Expert（Spoke）模型之间每 $\lfloor L/K \rfloor$ 层插入 Stitch Layer（共 $K$ 个，本文 $K=4$），仅训练 stitch 层参数 15B tokens。

BTS 的推理流程：输入 → Hub+Expert Layer 1-4 → Stitch 1 (Hub-into-Experts) → Layer 5-9 → Stitch 2 (Experts-into-Hub) → Layer 10-14 → Stitch 3 (Hub-into-Experts) → Layer 15-19 → Stitch 4 (Experts-into-Hub) → Hub 输出 → LM head。

从算法pipeline角度拆解术语。
BTS 论文使用的配置：2.7B Seed 模型（20 层，dim 3072，FFN dim 12288，24 heads，GQA=1 KV head，SwiGLU，RoPE θ=500000）。3 个 Expert（Code / Math / Multilingual），各继续训练 200B tokens。4 个 Stitch Layer（交替 Experts-into-Hub 和 Hub-into-Experts）。Stitch 训练 15B tokens（batch 2M，7000 steps，LR warmup 0→5e-6 cosine decay）。

关键性质：
- **模块性**：Expert 完全冻结 → Expert 可随时增删，仅需重训 stitch 层
- **Token 级路由**：每个 token 重新计算 gate → 支持 context-switching（同一 prompt 内不同任务自动切换 Expert）
- **Cross-capability**：交替双向 stitch 使 Expert 间信息流动 → 产生超越任何单个 Expert 的跨领域能力（如 Russian Math）

术语一般如何实现？如何使用？
- 实现方式：基于标准 PyTorch Transformer + 插入自定义 StitchLayer 模块（见 Stitch Layer 条目）
- 适用场景：需要将多个领域专用 LLM 合并为通用模型，同时保持模块性和可解释性
- 局限：总参数较多（11B = 4 × 2.7B），推理时需要前向传播所有 Expert（无稀疏激活）
- 论文未开源，来自 Meta FAIR 研究

涉及论文标题：
- BTS Harmonizing Specialized Experts into a Generalist LLM

---
