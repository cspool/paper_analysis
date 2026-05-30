## Semantic Segment Partitioning for Video Streams（视频流语义分段划分）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Semantic Segment Partitioning for Video Streams 是一种基于视觉语义边界动态划分视频流的方法。与传统的均匀分段（每固定帧数一段）不同，该方法按视频内容的语义变化自适应确定段边界。核心流程：(1) 使用 ViT 编码器提取每帧的 patch 级 embedding $f_t \in \mathbb{R}^{P^2 \times D}$；(2) 计算相邻帧 embedding 的 cosine similarity $s_t = \frac{f_{t-1} \cdot f_t}{\|f_{t-1}\| \|f_t\|}$；(3) 将相似度低于阈值（如 0.99）的帧标记为语义边界；(4) 应用 exclusion window（最小段长 $m$）避免过短段；(5) 若段长超过上限 $M$，通过 segment merging 合并段内余弦相似度最高的相邻帧对（利用视频的时间冗余）。输出为语义段序列 $[\mathbf{S}^i]$，每段 $\mathbf{S}^i := [f_t^i]_{t=1}^{T_i}$ 满足 $T_i \in [m, M]$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

语义分段划分是 Streaming Video QA pipeline 的第一阶段，位于视频帧编码之前。它接收原始视频帧序列，输出语义边界标记的段划分结果。具体计算过程：

```
# 输入: 视频帧序列，ViT encoder
for t = 1 to T:
    f_t = ViT(frame_t)  # ∈ R^{P²×D}

# Step 1: 计算相邻帧相似度
for t = 2 to T:
    s_t = cos_sim(f_{t-1}, f_t)  # Eq.(1)
    if s_t < threshold:  # e.g., 0.99
        boundaries.append(t)

# Step 2: Exclusion Window 过滤
# 确保任意两个 boundary 之间距离 ≥ m（如 m=4）
boundaries = filter_by_window(boundaries, window_size=m)

# Step 3: Segment Merging
for each segment S^i:
    if len(S^i) > M:  # M = 64
        while len(S^i) > M:
            # 找到段内最相似的相邻帧对并合并
            (t1, t2) = argmax(cos_sim(S^i[t], S^i[t+1]))
            S^i[t1] = mean(S^i[t1], S^i[t2])  # 合并
            remove(S^i[t2])

# 输出: 语义段序列 [S^1, S^2, ...]
```

术语一般如何实现？如何使用？

实现方式：基于 ViT 编码器的 embedding 输出进行帧间相似度计算，不需要额外训练。阈值 $m$、$M$ 和 similarity threshold 是超参数（StreamKV 使用 m=4, M=64, threshold=0.99）。Segment merging 通过贪心合并最高相似度相邻帧对实现。适用场景：(1) 流式视频理解中需要在编码前确定段边界；(2) 避免均匀分段破坏语义连续性的任何长视频处理任务；(3) 可推广到其他需要内容感知分段的场景。与均匀分段相比，论文实验表明语义分段在所有压缩率下均获得更高准确率（Table 2: 50% 压缩率下 59.07% vs 57.32%）。

涉及论文标题：
- StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression
