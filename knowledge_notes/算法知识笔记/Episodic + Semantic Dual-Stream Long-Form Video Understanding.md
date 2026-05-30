## Episodic + Semantic Dual-Stream Long-Form Video Understanding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Episodic + Semantic Dual-Stream Video Understanding 是 HERMES 提出的受人类认知双记忆系统启发（episodic memory + semantic memory）的长视频理解框架。人类认知中，episodic memory 负责回忆特定事件/经历（"昨天下午和母亲在电话里争论了什么"），semantic memory 负责存储一般性知识/概念（"家庭关系中常见的冲突模式"）。HERMES 将这一认知框架映射到视频理解中：(1) **Episodic Stream (ECO)** —— 以 window 为粒度在线处理视频帧，通过 global cosine-similarity merging 将帧压缩为最多 E 个 episode prototypes。ECO 保存时序细节和特定事件，类比"记得电影中具体发生了什么"。Episodic Q-Former 在 query 空间也进行 episode-level 聚合；(2) **Semantic Stream (SeTR)** —— 通过 stride-based 帧分组 + similarity merging 将 N 帧压缩为 N/k 帧语义代表。SeTR 提取跨整个视频的高层次主题和概念，类比"总结电影在讲什么"。Hierarchical Q-Former 两级（frame→video）增强语义表达；(3) **Fusion** —— 将两条流的输出 concat 后经 linear projection 馈入冻结 LLM (Vicuna-7B) 生成回答。双流互补：episodic stream 回答 "what happened when"，semantic stream 回答 "what is this about overall"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
HERMES 完整 dual-stream pipeline：
```
输入: long video V, instruction I
# === Episodic Stream (ECO + Episodic Q-Former) ===
windows = ViT_G_14(sample_frames(V, N=100), window=10)
M = []                                             # episode memory
for W_k in windows:
    M = ECO(M, W_k, max_episodes=20)               # online压缩
Q_0 = learned_queries                              # (32, 768)
Q_ep = Episodic_QFormer(Q_0, M)                    # episode-aware queries

# === Semantic Stream (SeTR + Hierarchical Q-Former) ===
F = concat(windows)                                # all features: (100, T, C)
F_prime = SeTR(F, keep_ratio=0.2)                  # semantic compression: (20, T, C)
Q_sem = Hierarchical_QFormer(F_prime)              # (32, 768)

# === Fusion + LLM Generation ===
U = Linear(concat([Q_ep, Q_sem]))                  # (64, LLM_dim)
answer = Vicuna_7B.generate(U, I)
```
关键设计：(1) 两条流可独立使用（作为 plugin 插入其他 VLM）；(2) 两条流均为 training-free（仅 Q-Former 和 adapter 可选微调）；(3) 仅需 100 帧（vs MA-LMM 2048 帧），22 FPS on V100。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
认知科学基础（论文 Section H.9）：基于 Tulving (1972) 的 episodic/semantic memory distinction、hippocampus 在 episodic memory consolidation 中的作用、neocortex 在 semantic knowledge 存储中的作用、event segmentation theory (Zacks et al., 2007)、gist extraction (Oliva, 2005)。在长视频理解中的优势：episodic stream 擅长捕捉角色关系变化（LVU Relationship +15.4% over S5）、semantic stream 擅长理解整体主题和场景分类。双流设计使 HERMES 在四个 benchmark 上达到 SOTA：MovieChat-1k +14.9%（zero-shot）、LVU +7.3%、Breakfast +2.2%、COIN +0.3%。

涉及论文标题：
- Bridging_Episodes_and_Semantics__A_Novel_Framework_for_Long-Form_Video_Understanding
