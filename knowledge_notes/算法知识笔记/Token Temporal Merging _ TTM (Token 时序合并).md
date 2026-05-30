## Token Temporal Merging / TTM (Token 时序合并)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Temporal Merging (TTM) 是一种在 Video LLM 推理的 prefilling 阶段，通过利用视频帧间的时序冗余（temporal redundancy）合并相似 visual token 来减少输入 token 数量的技术。核心假设是：视频中相邻帧包含大量相似或重复的视觉信息（如静态背景、连续动作的微小变化），可以通过帧间 token 级别的相似度计算来合并冗余 token。TTM 属于 training-free 方法，不需要额外训练或参数修改。DyCoke 的 TTM 采用滑动窗口（window=4 frames）、分组采样（Odd/Even 组）、余弦相似度度量和分层剪枝策略。相似方法：ToMe（Token Merging for ViT，通过 bipartite matching 合并相似 token）、TempMe（progressive merging across neighboring clips）、TESTA（temporal + spatial aggregation）、HoliTom（outer-LLM + inner-LLM holistic merging）、TTF（锚定帧 + 局部窗口相似度搜索）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DyCoke 的 TTM 在 prefilling 初始阶段执行，位于视觉编码器输出之后、LLM 输入之前：
```
# === Prefilling Stage: Token Temporal Merging (TTM) ===
# 输入: H_v' (visual tokens, shape: M_v*N_v × D)
# 超参数: K (保留比例 k%), window_size=4

# Step 1: 滑动窗口分组
for i in range(0, M_v, window_size):  # M_v=32 frames, window=4 → 8 windows
    window = H_v'[i*N_v : (i+4)*N_v]  # 4帧 × 196 tokens = 784 tokens
    # 分为 Odd 组 (帧0, 帧2) 和 Even 组 (帧1, 帧3)
    O_group = window[0*N_v:1*N_v] ∪ window[2*N_v:3*N_v]
    E_group = window[1*N_v:2*N_v] ∪ window[3*N_v:4*N_v]
    
    # Step 2: 计算 O/E 对应位置 token 余弦相似度
    for pos in range(N_v):
        S[pos] = cos_sim(O_group[pos], E_group[pos])  # Eq.3: h_i·h_j/(||h_i|| ||h_j||)
    
    # Step 3: 剪枝 E 组高相似 token (按 K 比例)
    threshold_E = percentile(S, 100 - k_E)
    E_pruned = E_group[S < threshold_E]
    
    # Step 4: O 组内 frame-0 全保留，其余帧与 frame-0 比相似度剪枝
    O_kept = O_group[0:N_v]  # 窗口第一帧全保留
    for f in [O_frame2]:
        S_o = cos_sim(O_kept[0:N_v], f)  # 与首帧比较
        O_kept ∪= f[S_o < threshold_o * K]

    merged_window = concat[O_kept, E_pruned]
    # 结果：4帧 → 约 K*4 帧等效 token 量

# Step 5: 输出压缩后的 visual tokens → concat 文本 tokens → LLM
H = concat[TTM(H_v'), H_q]  # Eq.4
# 32帧 × 196 = 6272 tokens → 约 1882 tokens (K=0.7 时保留约30%)
```
TTM 处理 32 帧输入仅需 < 10^{-3} 秒，开销可忽略。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DyCoke 通过 PyTorch 实现，预计算 token 间的 cosine similarity 矩阵。使用时通过 lmms-eval 传入 dycoke=True, dycoke_k=0.5~0.7 参数启用。TTM 作为 plug-and-play 模块嵌入在 vision encoder 和 LLM projector 之间，无需修改 LLM 结构。超参数 K 应随输入帧数增加而增大（更多帧 → 更多冗余 → 可更激进压缩）。DyCoke 实验显示 K=0.7 时保留约 30% visual tokens 仍能保持或提升性能。类似工具：HoliTom (github.com/cokeshao/HoliTom)、TTF (github.com/Cominder/ttf)。

涉及论文标题：
- DyCoke__Dynamic_Compression_of_Tokens_for_Fast_Video_Large_Language_Models
