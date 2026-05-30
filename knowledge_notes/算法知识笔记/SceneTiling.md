## SceneTiling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SceneTiling 是 VideoLLaMB 提出的无模型（model-free）视频语义分割算法，受 NLP 领域 TextTiling（Hearst, 1997）启发。其核心思想是：视频中相邻帧的 ViT [CLS] token 余弦相似度在场景边界处会出现显著下降（即边界两侧的语义内容差异最大），通过计算 depth score 检测这些"语义低谷"来分割视频。算法流程：(1) 计算相邻帧对 ViT [CLS] token 的余弦相似度序列 {c_1, ..., c_{n-1}}，c_i = CosineSim(ViT(v_i).cls, ViT(v_{i+1}).cls)；(2) 对每个位置 i 计算 depth score d_i = (cl_i + cr_i - 2c_i) / 2，其中 cl_i 和 cr_i 分别是 i 左侧和右侧的局部最大相似度——d_i 越大，说明 i 处的相似度相对周围越低，即语义边界越明显；(3) 计算 depth score 的均值 μ 和方差 σ，设定阈值 μ + α·σ（α 为超参数控制分割粒度），选取超过阈值的 K-1 个 depth score 对应位置作为分割点，将视频分为 K 个语义段 {s_1, ..., s_K}。SceneTiling 也可用于流式视频字幕生成：仅使用左侧相似度 d_i = (cl_i - c_i)/2 实时检测场景变化边界，无需预知完整视频。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SceneTiling 在 VideoLLaMB pipeline 中位于 vision encoder 之后、Memory Bridge 之前：

```
# === SceneTiling 伪代码 ===
# 输入: video V = {v_1, v_2, ..., v_n} (n frames), 超参数 α (默认值由论文经验设定)
# 输出: K 个语义段 {s_1, s_2, ..., s_K}

# Step 1: 提取帧级 CLS token
for i = 1 to n:
    f_i = ViT(v_i).cls_token  # ViT-L/14, dim=1024

# Step 2: 计算相邻帧余弦相似度
for i = 1 to n-1:
    c_i = CosineSimilarity(f_i, f_{i+1})
    # c_i ∈ [-1, 1], 值越高表示两帧越相似

# Step 3: 计算 depth score
for i = 1 to n-1:
    cl_i = max(c_1, ..., c_{i-1})  # 左侧最高相似度
    cr_i = max(c_{i+1}, ..., c_{n-1})  # 右侧最高相似度
    d_i = (cl_i + cr_i - 2 * c_i) / 2
    # d_i 高 → 帧i处的相似度显著低于周围 → 潜在场景边界

# Step 4: 确定分割阈值和分割点
μ = mean(d_1, ..., d_{n-1})
σ = std(d_1, ..., d_{n-1})
threshold = μ + α * σ
boundaries = {i | d_i > threshold}
# 选取 K-1 = len(boundaries) 个分割点

# Step 5: 分割视频
{s_1, s_2, ..., s_K} = split_by_boundaries(V, boundaries)
# 每个 s_j 是连续帧序列，内部语义一致

# 流式模式 (streaming caption):
for i = 1 to n-1:
    d_i = (cl_i - c_i) / 2  # 仅用左侧，实时检测
    if d_i > threshold:
        trigger_caption_generation()  # 场景变化时自动生成字幕
```

参数量化：n 帧视频的 SceneTiling 仅需 O(n) 次余弦计算 + O(n) 次 depth score 计算，额外开销极小。CLS token 维度为 ViT 隐藏维度（如 1024），余弦相似度计算为常数时间。α 控制分割敏感度：α 越大 → 阈值越高 → 分割点越少；α 越小 → 分割点越多但可能引入噪声分割。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SceneTiling 的实现完全基于 ViT 编码器的输出，无需训练任何额外参数。VideoLLaMB 开源实现中（github.com/bigai-nlco/VideoLLaMB），SceneTiling 作为预处理模块在特征提取后执行。核心代码使用 PyTorch 的 cosine_similarity 函数和简单的 NumPy/PyTorch 统计计算。流式模式下通过缓存左侧最大值 cl_i（而非全序列最大值）实现实时检测。局限性：(1) 依赖 ViT 编码质量，低质量/模糊帧可能导致不准确的相似度计算；(2) α 需要针对不同视频类型（快节奏 vs 慢节奏）调参；(3) 渐变场景过渡（fade/dissolve）可能不会被检测为边界，因为帧间相似度变化平缓；(4) 论文未提供 α 的最优经验值或自适应策略。

涉及论文标题：
- VideoLLaMB__Long-context_Video_Understanding_with_Recurrent_Memory_Bridges
