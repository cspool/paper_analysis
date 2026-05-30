## Event-aware Temporal Agent（事件感知时序代理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Event-aware Temporal Agent 是 TSPO 中提出的轻量级可训练关键帧选择模块，负责从候选帧中概率化选择与文本查询最相关的关键帧子集。基于 CLIP-Large (400M 参数，冻结) 作为视觉和文本特征提取器，仅包含 3.5M 可学习参数（local window attention + MLP projector）。核心设计：(1) Event-aware 特征增强：通过 local window attention (窗口大小 w=12) 配合正弦位置编码捕获帧间时序依赖，学习"事件"级别的时序感知表示；(2) 双路径相似度融合：同时计算 event-level similarity (增强特征与文本) 和 frame-level similarity (原始特征与文本)，融合分数指导帧选择；(3) Gumbel-Softmax 概率化采样：输出帧选择概率和索引，支持 RL 探索。设计动机：现有 training-free keyframe selectors 使用大模型 (DINOv2-1B 或 MLLM-13B) 但无法针对下游 Video-MLLM 优化；Temporal Agent 保持轻量（3.5M vs 1B/13B）且可通过 GRPO 端到端训练。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Event-aware Temporal Agent 前向传播 ===
# 输入: V_c (T_c 帧候选视频), q (文本查询)
# 可训练参数: θ_agent = {W_q, W_k, W_v (local window attn), MLP_proj}
# 参数总量: 3.5M
# 冻结部分: CLIP-Large visual encoder, CLIP text encoder

# Step 1: CLIP 特征提取 (冻结, 无梯度)
F_f = CLIP_visual(V_c)          # [T_c, D], D 为 CLIP 特征维度
F_t = CLIP_text(q)              # [1, D]

# Step 2: 正弦位置编码 (注入时序信息)
PE = SinusoidalPositionalEncoding(T_c)  # [T_c, D]
F_f_pos = F_f + PE                      # [T_c, D]

# Step 3: Local Window Attention (事件感知)
# 每个帧 i 仅关注 [i-w/2, i+w/2] 范围内的帧
# w=12, 帮助学习局部事件依赖
F_e = LocalWindowAttention(
    Q = W_q @ F_f_pos,           # [T_c, d_k]
    K = W_k @ F_f_pos,           # [T_c, d_k]
    V = W_v @ F_f_pos,           # [T_c, d_v]
    window_size = 12,
    causal = False               # 双向窗口注意力
)                                # [T_c, D]

# Step 4: MLP 投影
F_e = MLP_proj(F_e)              # [T_c, D]

# Step 5: 双路径跨模态相似度
Sim_event = CosineSimilarity(F_e, F_t)   # [T_c], event-level
Sim_frame = CosineSimilarity(F_f, F_t)   # [T_c], frame-level
S = Sim_event + Sim_frame                # [T_c], 融合分数

# Step 6: Gumbel-Softmax 概率化采样
# 训练时: 加 Gumbel 噪声探索
γ ~ Gumbel(0, 1)                         # [T_c]
P = Softmax(S/τ + γ)                     # [T_c], 概率分布

# 推理时: 确定性采样 (无 Gumbel 噪声)
P = Softmax(S/τ)                          # [T_c]

# Step 7: TopK 选择
P_selected, I_selected = TopK(P, T_s)    # T_s 帧索引+概率
# log π_ts = Σ log P_selected[i]         # 用于 GRPO
```

Agent 的 RL 视角形式化：
- State: 输入长视频 V 和文本指令 Q
- Action: 关键帧选择，输出索引 I = {i_1,...,i_T_s} 和概率 P = {p_1,...,p_T_s}
- Policy: π_ts(V_s|q,V_c) = Π_{i=1}^{T_s} P_i(V_c, q)
- Reward: R_A + R_T (见 TSPO 术语)
- 优化: GRPO (仅更新 θ_agent)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/Hui-design/TSPO 中 src/open_tspo/ 目录下的模型实现。使用方式：(1) 下载预训练 TSPO-0.4B checkpoint (包含 CLIP-Large 权重 + 3.5M agent 参数)；(2) 通过 demo/llava_video_tspo.py 加载 agent + Video-MLLM backbone；(3) 推理流程：视频→1FPS 候选帧提取→CLIP 特征→Temporal Agent 选择 64 帧→Video-MLLM 生成回答。Agent 可跨 Video-MLLM 迁移：在 LLaVA-Video-7B 上训练的 agent 可直接用于 Qwen2.5VL-7B 等不同架构的 Video-MLLM。性能：仅 3.5M 可训练参数的 agent 使 LLaVA-Video-7B 在 LongVideoBench 提升 5.0% (58.9→63.9)，MLVU 提升 6.0% (70.3→76.3)。推理时帧提取仅需 1.2s (vs CoS 的 28.4s)。与类似模块的对比：LongVU 的 DINOv2-1B selector (1B 参数, training-free), CoS 的 MLLM-13B selector (13B, training-free), HORNet 的 MLP selector (<1M, GRPO-trained for Bernoulli mask), FrameVOYAGER 的 selector (training-based, random pre-processing sampling)。Temporal Agent 独特之处：轻量 (3.5M) + 可训练 (GRPO) + 事件感知 (local window attention) + 可迁移。

涉及论文标题：
- TSPO__Temporal_Sampling_Policy_Optimization_for_Long-form_Video_Language_Understanding
