## Sparse Temporal Features / Dense Temporal Features / 稀疏与密集时间特征

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ReVisionLLM Hierarchical Adapter 输出的两种互补视频表示。Sparse Features S^i ∈ R^768：每个视频段（250帧/125s）通过 Cross-Attn + Self-Attn + FFN 压缩为 1 个 token，压缩比 250:1，保留段级语义信息但丢失精确帧级时刻，用于上层 hierarchy 高效扫描小时级视频。Dense Features D^i ∈ R^{250×4096}：每帧独立 Linear Projection (768→4096) 映射到 LLM space，保留全部帧级时间分辨率，仅在底层 hierarchy 已缩小的搜索范围内使用以精确定位秒级边界。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Sparse: 250 frame CLS tokens → 1 sparse token
C_tilde = CrossAttn(C_i, Q)      # (250, 768) 文本对齐
S_i = SelfAttn(concat([S_learnable, C_tilde]))[0]  # (768,)

# Dense: 250 frame CLS tokens → 250 LLM embeddings
D_i = Linear_768to4096(C_i)      # (250, 4096)

# 使用场景:
# Hierarchy 3: [S_1..S_100, prompt] → 100 tokens → 粗定位
# Hierarchy 1: [D_selected, prompt]  → 250 tokens → 精确边界
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
稀疏特征训练使用简化目标（Yes/No 存在性判断），密集特征训练使用完整定位目标（From s to e）。两种特征使用不同的 LoRA 模块进入 LLM。默认 ReVisionLLM 仅处理 57% 的视频帧——上层 hierarchy 使用 sparse features 大幅减少 token 数，底层仅对选定段使用 dense features。

涉及论文标题：
- ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos
