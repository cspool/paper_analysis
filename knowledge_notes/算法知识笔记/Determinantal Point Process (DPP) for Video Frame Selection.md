## Determinantal Point Process (DPP) for Video Frame Selection

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DPP (Determinantal Point Process, 行列式点过程) 是一种集合子集概率模型，选择概率 P(S) ∝ det(L_S)，其中 L 是 kernel matrix。在 TimeSearch-R 中 DPP 用作时序搜索的核心优化：在 [t_s, t_e] 内从 N 候选帧选 F≤8 帧，同时优化帧的 query relevance 和 interset diversity。Kernel 构造：L̃ = diag(r) · S · diag(r)，L̃_ij = r_i · r_j · h_i^T · h_j。r_i = SigLIP text-image 相关性分数 min-max 归一化到 [0,1]，S 是帧间 cosine similarity。DPP 的行列式编码质量和多样性权衡：高质量帧 (r_i→1) 优先被选，但相似帧被同时选中会降低行列式值被惩罚。TimeSearch-R 使用 fast greedy MAP inference (Chen et al., 2018) 近似求解 argmax det(L̃_S)，O(N·F²)。

从算法pipeline角度拆解：
```
# DPP 帧选择流程
h_i = SigLIP.encode(v_i) ∀ v_i ∈ F_cand      # [N, d]
q_emb = SigLIP.encode_text(query)             # [d]
S_ij = h_i^T · h_j                             # 帧间相似性
r_i = norm(q_emb^T · h_i)                      # query 相关性 [0,1]
L̃_ij = r_i · r_j · S_ij                       # DPP kernel
V* = greedy_MAP(L̃, F)                          # 选 F 帧最大化 det
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：(1) SigLIP-400M 提供帧/文本嵌入；(2) Greedy MAP 是近似算法（精确 MAP 为 NP-hard），实践中效率和质量充分；(3) 比 top-K by relevance 优势：避免选连续冗余帧，DPP 通过多样性惩罚覆盖不同内容。广泛用于信息检索、推荐系统、文档摘要、视频关键帧提取。

涉及论文标题：
- TimeSearch-R__Adaptive_Temporal_Search_for_Long-Form_Video_Understanding_via_Self-Verification_Reinforcement_Learning
