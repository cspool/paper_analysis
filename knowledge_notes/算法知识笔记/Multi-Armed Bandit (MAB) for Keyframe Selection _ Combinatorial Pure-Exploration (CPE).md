## Multi-Armed Bandit (MAB) for Keyframe Selection / Combinatorial Pure-Exploration (CPE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Armed Bandit (MAB) 是随机决策理论中的经典框架，建模在不确定环境下分配有限资源（pulls）给多个选项（arms）以达到某种最优目标的决策问题。FOCUS 将长视频 keyframe selection 创新性地建模为 MAB 中的 Combinatorial Pure-Exploration (CPE) 子问题：视频被划分为 M 个固定时长的 clip（每个 clip 为一个 arm），目标是选出最优的 m 个 arm 子集（即最 query-relevant 的 clip），然后从这些 arm 内进一步选出 k 个 keyframes。Pure-exploration 意味着目标不是最小化 regret，而是以高置信度找到最优 arm 子集——天然匹配 keyframe selection 的"选最优帧"目标。CPE 由 Chen et al. (NeurIPS 2014) 首次提出，核心算法 CLUCB 通过 confidence bound + oracle maximization 实现对任意组合结构的 arm 子集的高效识别。FOCUS 的决策类 S 定义为所有大小为 m 的 arm 子集。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 FOCUS 中，CPE bandit 建模流程：
```
# 将 keyframe selection 建模为 CPE bandit
Clips = partition(V, clip_length=l)  # M 个 clip = M 个 arm
def pull(arm_a):
    t ~ uniform(arm_a.start, arm_a.end)  # 随机采样 arm 内一帧
    return BLIP_ITM(x_t, q)              # frame-query relevance ∈ [0,1]

# CPE 目标: 找到最优 m-arm 子集
# S* = argmax_{|S|=m} sum_{a∈S} μ_a,  μ_a = E[r_t | t∈A_a]
# 决策类: S = {all size-m subsets of M arms}
```
分层设计的关键洞察：视频帧间强时间相关性（ACF > 0.5 for ~5s）意味着 clip 内帧高度相似，少量采样即可估计整个 clip 的平均 relevance。从帧级选择 C(T,k)（组合数巨大）降为 clip 级选择 × 帧级选择。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CPE bandit 在 FOCUS 中通过 Algorithm 2（两阶段 optimistic UCB）实现。理论保证：Bernstein confidence bound 保证 |μ̂_a - μ_a| ≤ β_a 以 ≥ 1-6/n 概率成立（Theorem B.1）；Algorithm 2 以 ≥ 1-6(M-m)/n 概率返回 oracle top-m set（Theorem C.1）。CPE 框架在 FOCUS 中用于 clip 级粗筛，随后在选中的 arm 内通过 nearest-neighbor 插值 + 概率采样完成帧级 fine selection。

涉及论文标题：
- FOCUS__Efficient_Keyframe_Selection_for_Long_Video_Understanding
