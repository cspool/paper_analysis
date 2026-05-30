## Guidance Prompt for KV Cache Compression（引导提示驱动的KV缓存压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Guidance Prompt for KV Cache Compression 是 StreamKV 提出的一种面向流式视频场景的 KV cache 压缩方法。与现有的基于用户问题（question-dependent）的 KV 压缩方法（如 FastV、SparseVLM、SnapKV）不同，该方法不依赖具体的用户问题，而是引入一个 guidance prompt 来捕获视频段内的关键语义元素，以此作为压缩的选择依据。Guidance prompt 覆盖五类语义元素：(1) salient entities（人物、物体、场景、关键视觉概念）；(2) key events and actions（发生了什么、何时、何地）；(3) temporal and causal relationships（事件时序和因果链）；(4) contextual cues（场景切换、对话、叙事变化）；(5) important numerical or factual details（计数、摘要、事实类信息）。压缩在每段编码完成后立即执行（非解码阶段离线压缩），使用 guidance prompt 的平均 query vector $\mathbf{g}^l = \frac{1}{N_g}\sum_{k} \mathbf{g}_k^l$ 作为层自适应 KV 选择模块的 selection criterion。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

Guidance Prompt KV 压缩的 pipeline 流程：

```
# 输入: 语义段 S^i 的 KV blocks B_l^i, 压缩率 θ, guidance prompt G

# Step 1: 将 guidance prompt G 送入 LLM，提取每层的 query vectors
for each layer l:
    g^l = (1/N_g) Σ_k g_k^l  # guidance prompt 平均 query vector

# Step 2: 计算总压缩预算
N = ⌈(1-θ) × T_i⌉ × L  # T_i 段帧数, L 层数

# Step 3: 层自适应选择
{I_l^i}_{l=1}^L = SelectKV({R_l^i, g^l}_{l=1}^L, N)  # Eq.(9)

# Step 4: 保留选中的 KV blocks + summary KV block
~B_l^i = [b_m^{i,l} | m ∈ I_l^i]  # 压缩后 KV blocks
B_l ← [B_l, ~B_l^i, b_s^{i,l}]    # 存入 KV Bank

# 关键: b_s^{i,l} (summary KV block) 不参与压缩，始终保留
```

与 question-dependent 压缩的对比：现有方法（FastV、SparseVLM）需要已知用户问题才能压缩，不适合 StreamingVQA 场景（问题未知、多轮对话）。Guidance prompt 使压缩聚焦于视频语义本身而非特定问题，更鲁棒。

术语一般如何实现？如何使用？

实现方式：guidance prompt 是一个预定义的文本模板（见论文 Appendix A），如 "Please identify the key semantic elements in this video segment, including salient entities, key events, temporal relationships, contextual cues, and important factual details"。将 guidance prompt 送入 Video-LLM，提取其在各 transformer 层的 query vectors 作为 selection criterion。适用场景：(1) 流式视频处理中未知用户问题的 KV 压缩；(2) 多轮对话场景下需要保留通用语义信息；(3) 任何需要 problem-agnostic 压缩的长上下文 LLM 推理场景。论文实验表明：60% 压缩率下 Overall 准确率 58.9%，甚至优于无压缩的 ReKV（53.5%），证明压缩不仅减少显存还通过去除冗余提升精度。

涉及论文标题：
- StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression
