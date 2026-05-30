## Dynamic Compression (动态压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Compression（动态压缩）是 D-CoDe 提出的核心方法组件之一，用于解决图像预训练 VLM 扩展到视频时的感知瓶颈（Perception Bottleneck）。它是一种 training-free 的自适应视觉信息压缩策略，包含时间和空间两个维度的操作：(1) **时间维度**：先均匀采样 ⌊α·N⌋ 帧（α=0.85），再利用 CLIP 的 global feature 计算帧间余弦相似度，迭代选择与已选帧语义最不相似的 supplementary frame，直到共选 N 帧——这种"均匀覆盖 + 多样性补充"策略优先保留语义不同的关键帧；(2) **空间维度**：对每帧的 visual tokens 按 ℓ2 norm 计算 salience 分数（activation magnitude），保留 top-⌊β·M⌋ 高激活 token（β=0.625），然后在保留的 token 中按余弦相似度（阈值 τ=0.9）使用贪婪算法合并冗余 token（anchor + cluster 取平均值作为代表 token）。核心创新在于将静态、均匀、无感知的压缩策略（uniform sampling + average pooling）替换为内容感知的动态策略。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Dynamic Compression 在 D-CoDe pipeline 中的执行流程：
```
# === 时间维度: 动态帧选择 ===
# 输入: 视频 V (T frames), 目标帧数 N, 均匀采样比 α=0.85
# Stage 1: 均匀采样
N_uniform = floor(α * N)
V_selected = uniform_sample(V, N_uniform)

# Stage 2: 基于语义多样性的补充帧选择
for k in 1..(N - N_uniform):
    # 计算每帧与已选帧的平均语义不相似度
    for each frame I_m in V \ V_selected:
        g_m = CLIP_visual(I_m)           # CLIP global feature
        avg_sim = mean(cosine_sim(g_m, g_n) for I_n in V_selected)
    # 选最不相似的帧（最大化多样性）
    I* = argmin(avg_sim)
    V_selected = V_selected ∪ {I*}

# === 空间维度: 动态 Token 压缩（每帧独立） ===
# 输入: 选中帧的 visual tokens F (M tokens × d dims)
# 参数: β=0.625 (保留比例), τ=0.9 (合并阈值)
for each frame in V_selected:
    F = VisualEnc(frame)                 # shape: (M, d)
    
    # Step 1: Salience-based Pruning
    a = [||f_i||_2 for f_i in F]        # ℓ2 norm salience
    F_pruned = TopK(F, key=a, k=floor(β*M))
    
    # Step 2: Greedy Similarity Merging
    sorted_idx = argsort(a, descending=True)
    merged = []
    merged_mask = [False] * len(F_pruned)
    for i in sorted_idx:
        if not merged_mask[i]:
            # 找与 anchor token f_i 相似度超过 τ 的未合并 token
            cluster = [i]
            for j in sorted_idx:
                if j > i and not merged_mask[j]:
                    sim = cosine_sim(F_pruned[i], F_pruned[j])
                    if sim >= τ:
                        cluster.append(j)
                        merged_mask[j] = True
            # 合并 cluster: 取平均值
            f_rep = mean(F_pruned[cluster], dim=0)
            merged.append(f_rep)
    
    frame_compressed = merged

# 最终拼接
F_final = concat([frame_compressed for frame in V_selected])
```

关键张量维度：
- CLIP global feature g_t: 由 CLIP 视觉编码器提取，维度取决于 CLIP 变体（~768 或 ~1024）
- Visual tokens F: 每帧经 LLaVA-NeXT 编码后的 token 序列，(M, d) 其中 M ~576 (336×336, 取决于 patch size)
- ℓ2 norm salience a_i: scalar，量化 token 对整体视觉表示的贡献
- 合并阈值 τ=0.9：只有 cosine similarity >= 0.9 的 token 对才被合并
- α=0.85, β=0.625：通过消融实验确定的最优超参数

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Dynamic Compression 在 D-CoDe 中作为 training-free 的前处理模块实现，不修改 LLaVA-NeXT 的任何权重。实现基于 HuggingFace Transformers + PyTorch，核心代码在 `Dcode.py` 的 `supp_frame_selection()`（帧选择）和 `token_select_and_merge()`（token 压缩）函数中。超参数通过 EgoSchema 消融实验确定：α=0.85（15帧中 ~13 帧均匀采样 + 2 帧补充）、β=0.625（保留 62.5% 高 salience token）、τ=0.9（仅合并高度相似的 token）。Dynamic Compression 单独使用将 EgoSchema accuracy 从 44.8% 提升至 51.8%（+7.0%），推理延迟从 3.927 s/sample 增加至 6.115 s/sample（+55.7%）。开源：https://github.com/hukcc/D-CoDe。

涉及论文标题：
- D-CoDe__Scaling_Image-Pretrained_VLMs_to_Video_via_Dynamic_Compression_and_Question_Decomposition
