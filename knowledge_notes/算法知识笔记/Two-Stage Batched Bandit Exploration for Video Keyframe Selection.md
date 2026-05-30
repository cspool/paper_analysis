## Two-Stage Batched Bandit Exploration for Video Keyframe Selection

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Two-Stage Batched Bandit Exploration 是 FOCUS 将理论上串行的 bandit 算法（Algorithm 1: iterative UCB, 每步 pull 1 arm）转化为实际可高效并行批处理的策略（Algorithm 2）。串行算法要求每轮 pull 一个 arm → 观察 reward → 更新统计 → 决定下一 arm → repeat，这在 GPU 上意味着 BLIP 以 batch_size=1 串行前向，严重浪费 GPU 利用率。FOCUS 压缩为两次并行 batch：(1) Stage I Coarse——所有 M arm 各采 q 帧，一次性 batch BLIP forward → 计算 per-arm 统计 + optimistic UCB；(2) Stage II Fine——仅对 UCB 最高的 α*m arm 各采 z 帧，再一次性 batch → 用无偏经验均值选最终 top-m arm。与 batched bandit 文献（Perchet et al., 2016; Gao et al., 2019; Jin et al., 2024 的 Tri-BBAI 三批次最优 BAI）精神一致。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Algorithm 2: 两阶段批处理 (仅需 2 次 batch BLIP forward!)
# Stage I: Coarse Exploration
rewards = BLIP_batch(all M arms, q frames each)  # 第 1 次 batch, M*q 帧
for a in 1..M:
    μ̂_a, σ̂_a² = stats(rewards[a])
    β_a = sqrt(2*σ̂_a²*ln(n)/q) + 3*ln(n)/q
    μ̃_a = μ̂_a + β_a
A_coarse = TopM({μ̃_a}, α*m)    # optimistic UCB 粗选 α*m 个 arm

# Stage II: Fine-grained Exploitation
rewards_fine = BLIP_batch(A_coarse, z frames each)  # 第 2 次 batch
for a in A_coarse:
    update μ̂_a with rewards_fine[a]    # 合并两次采样更新经验均值
A_fine = TopM({μ̂_a}, m)              # 无偏经验均值精选 m 个 arm

# Frame Selection within A_fine
for a in A_fine:
    r̂_{a,t} = nearest_neighbor_interpolate(rewards_a)  # 插值所有帧
    p_a = softmax(r̂_{a,t})              # 构建采样分布
    K_a = sample_without_replacement(p_a, k_a)  # 不放回采样
return K = union(K_a)
```

消融（Table 7）：FOCUS-C（仅 coarse）= 62.3/58.4/62.3%, FOCUS-F（仅 fine）= 61.5/57.7/62.5%, FOCUS（两阶段）= 62.3/60.7/63.5%（Qwen2-VL/LLaVA-OV/LLaVA-Video）——两阶段互补，coarse 做全局定位，fine 做精准提取。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
两阶段批处理的核心优势：仅需 2 次 batch BLIP forward（vs 串行算法需 O(M²·log(1/δ)) 次），充分利用 GPU 并行性。FOCUS 在 LongVideoBench 上处理帧数仅占总帧数的 1.6%，5.5 GPU hours。Arm 数 M = video_duration / clip_length（如 1h / 16s = 225），α=0.25 默认。q 和 z 的具体值论文未明确说明。Batched bandit 理论保证（Theorem C.1）：两阶段算法以 ≥ 1-6(M-m)/n 概率输出 oracle top-m set。

涉及论文标题：
- FOCUS__Efficient_Keyframe_Selection_for_Long_Video_Understanding
