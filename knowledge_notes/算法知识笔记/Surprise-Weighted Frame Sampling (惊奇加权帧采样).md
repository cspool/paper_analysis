## Surprise-Weighted Frame Sampling (惊奇加权帧采样)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Surprise-Weighted Frame Sampling 是 SPIKE 提出的一种 query-agnostic（不依赖查询）的自适应视频帧选择策略，用于替代 Video-LLM 的标准 uniform sampling。核心思想：将固定的帧预算 F 按各时间段的 Bayesian Surprise 得分比例重新分配——高 surprise 段获得更多帧（甚至被多次采样），低 surprise（routine）段获得较少帧。采样概率 p_i = softmax(S_i / τ_s)，其中 τ_s=0.7 控制分布集中度（小 τ_s 集中于 surprise 峰值，大 τ_s 趋于均匀）。这一方法的关键洞察是：surprising 事件恰好是人类叙事理解的关键信息点，在帧预算有限时优先采样这些帧能显著提升下游任务性能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Surprise-Weighted Frame Sampling
# 输入: 视频 X_{1:T}, 帧预算 F, surprise scores S_1..S_K
# 输出: 选中的 F 帧索引

# Step 1: 对视频均匀锚定 K 个采样段 (K ≤ F)
segments = uniform_sample_timesteps(T, K)

# Step 2: 每段用 SPIKE 计算 surprise score
for t in segments:
    S[t] = compute_bayesian_surprise(X, t)  # 见 Bayesian Surprise 条目

# Step 3: 将 surprise 转为采样概率 (τ_s 控制集中度)
τ_s = 0.7                                   # 论文实验值
p = softmax([S[t] / τ_s for t in segments])
# 如果所有 S_i 相等（完全无 surprise），则 p_i = 1/K (回退到 uniform)

# Step 4: 按概率采样 F 帧 (有放回，高 surprise 段可多次被选)
selected_frames = []
for f in range(F):
    seg = multinomial_sample(segments, p)    # 按 p 概率选段
    frame_idx = uniform_int_within(seg)      # 段内等概率选帧
    selected_frames.append(frame_idx)

# Step 5: 选中的帧送入 Video-LLM 进行下游任务
result = VideoLLM.infer(selected_frames, query)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SPIKE 的加权采样作为 Video-LLM 的 sampling layer 即插即用替代——不修改模型架构，仅改变帧选择策略。实现细节：(1) 时间锚点 K 也视为帧预算的一部分（K ≤ F），帧预算按视频时长按比例分配（≤1 分钟视频 8 帧基础预算，更长视频每增加 1 分钟预算翻倍）；(2) τ_s 可在推理时调节，实现从 "聚焦 surprise"（τ_s→0）到 "近似 uniform"（τ_s→∞）的连续控制；(3) 复杂度 O(F·N)，与推理时间 scaling 开销可比。与 SBD（shot boundary detection）方法（RGB Histogram、ECR、Optical Flow、Katna）的关键区别在于：SBD 依赖原始视觉变化（对相机运动/场景切换敏感），SPIKE 的 surprise score 基于语义理解（通过 Video-LLM 的先验/后验概率变化），更能对齐人类感知的关键时刻。

涉及论文标题：
- SPIKE-RL__Video-LLMs_meet_Bayesian_Surprise
