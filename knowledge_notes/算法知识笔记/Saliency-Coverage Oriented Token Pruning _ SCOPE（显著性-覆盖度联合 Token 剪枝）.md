## Saliency-Coverage Oriented Token Pruning / SCOPE（显著性-覆盖度联合 Token 剪枝）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SCOPE 是一种免训练的视觉 token 剪枝算法，用于加速多模态大语言模型（MLLM）推理。其核心思想是在保留的 visual token 子集中，同时最大化**显著性（saliency）**和**语义覆盖度（coverage）**，以在 token 预算大幅缩减时仍保持语义完整性。SCOPE 将 token 选择建模为一个迭代贪心过程：每轮计算每个候选 token v 的 marginal coverage gain Δ(v; S)（v 加入当前已选集 S 后带来的额外覆盖度），然后乘以视觉 attention score A_v^α 作为显著性加权，得到 SCOPE score = Δ(v; S) · A_v^α。每轮选择 SCOPE score 最高的 token 加入 S，更新 coverage 状态，迭代 K 次得到最终 token 子集。

SCOPE 解决的关键问题：saliency-only 方法（如 FastV, VisionZip）仅按 attention 排序选 Top-K token，导致：(1) 语义完整性缺失——高 attention 的 token 集中在少数图像区域（如前景物体），背景和上下文信息被丢弃；(2) attention 分布偏斜——尾部分布的 token attention 值几乎均匀（flat tail），无法区分 informative vs redundant tokens。SCOPE 通过引入 coverage metric 和联合优化解决此问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SCOPE 在 MLLM pipeline 中的位置：Vision Encoder → SCOPE Token Selection → LLM。完整流程：

```
输入: 图像 I, 文本 T
1. V = CLIP_ViT(I)  → V ∈ R^{N×d}  (N=576 for LLaVA-1.5, N=2880 for LLaVA-Next)
2. A_v = Attention_CLS_to_v(V, layer=-2)  → A_v ∈ R^N  (saliency scores)
3. S_uv = cosine_sim(v_u, v_v)  → S ∈ R^{N×N}  (pairwise similarity matrix)
4. S = ∅, c_u = 0 ∀u ∈ V  (初始化: 空选集, coverage scores=0)
5. for t = 1 to K:  (K = 目标 token 数, 如 64/128/192)
     for each v ∈ V \ S:
       Δ(v; S) = Σ_{u∈V} max(S_uv, c_u) - c_u   (marginal coverage gain)
       score(v) = Δ(v; S) · A_v^α                 (SCOPE score, α=1.0)
     v* = argmax score(v)
     S = S ∪ {v*}
     c_u = max(c_u, S_{u,v*})  ∀u ∈ V            (更新 coverage)
6. LLM_input = Concat(S, Text_Tokens)
7. Output = LLM(LLM_input)  → autoregressive generation
```

复杂度：相似度矩阵 O(N²) 存储（576²≈332K 对），每轮选择 O(N²) 扫描 × K 轮 = O(K·N²)。论文报告在 4×A100 上，2880→160 tokens 时延迟从 601.9s 降至 188.8s（3.2× speedup），同时 POPE 保持 81.3%（vs full 86.4%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SCOPE 的实现方式：
- 集成位置：在 vision encoder 之后、LLM projector 之前插入 token selection 模块
- 显著性来源：使用 vision encoder 倒数第二层（layer -2）的 CLS token 到 visual token 的 attention scores
- 相似度：cosine similarity，预计算全量 N×N 矩阵
- 缩放因子 α：默认 1.0，通过消融实验确定最优值
- 框架：基于 lmms-evals 评估框架实现，支持 HuggingFace Transformers
- 开源代码：https://github.com/kinredon/SCOPE
- 评估：支持 LLaVA-1.5 (7B/13B), LLaVA-Next (7B/13B), Video-LLaVA, Qwen2-VL
- 与 FlashAttention 兼容（不依赖中间 attention map，仅依赖 encoder 输出 token embeddings），剪枝后 token 数减少使得后续 LLM attention 计算按 O(K²/N²) 缩放

涉及论文标题：
- SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs
