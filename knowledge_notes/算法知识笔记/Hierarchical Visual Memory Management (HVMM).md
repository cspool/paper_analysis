## Hierarchical Visual Memory Management (HVMM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Visual Memory Management (HVMM) 是 CurveStream 的记忆调度模块，负责在固定 token 预算（N_max = 20 frames）下对 CAS 输出的曲率分数序列进行动态路由决策。HVMM 的核心机制包含两个子组件：(1) Online Manifold Distribution Estimation —— 使用 Exponential Moving Average (EMA) 在线更新曲率分数的瞬态分布参数：μ_t = γ·μ_{t-1} + (1-γ)·CS_t, σ_t² = γ·σ_{t-1}² + (1-γ)·(CS_t - μ_t)²，其中 γ ∈ (0,1) 为动量因子，控制历史观测窗口大小；(2) K-Sigma Dynamic Dual Thresholds —— 基于当前分布生成两个自适应阈值：g1 = μ_t + k1·σ_t（模糊记忆下界，k1=0.0 默认）和 g2 = μ_t + k2·σ_t（清晰记忆下界，k2=1.0 默认），k1 < k2。HVMM 的核心设计理念是将记忆管理建模为"在线分布感知"过程而非静态规则——在非平稳流视频中（如静止观察后突然剧烈奔跑），分布参数 (μ_t, σ_t²) 通过 EMA 平滑适应场景节奏变化，阈值随之动态平移，确保对加速/减速场景均能有效区分高价值和低价值帧。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
HVMM 的在线更新与路由过程：
```
# 输入: 当前帧的 CS_t，历史分布参数 (μ_{t-1}, σ_{t-1}²)
# 参数: γ (EMA momentum), k1=0.0, k2=1.0, N_max=20

# Step 1: EMA 在线更新分布参数
μ_t = γ * μ_{t-1} + (1-γ) * CS_t
σ_t² = γ * σ_{t-1}² + (1-γ) * (CS_t - μ_t)²

# Step 2: 生成 K-Sigma 动态双阈值
g1 = μ_t + k1 * σ_t    # 模糊记忆下界 (k1=0.0)
g2 = μ_t + k2 * σ_t    # 清晰记忆下界 (k2=1.0)

# Step 3: 层级状态路由
if CS_t >= g2 or t == t_q:
    # 曲率尖峰 或 查询时刻帧
    s_t = Clear, r_t = High     # 保留原始分辨率
elif g1 <= CS_t < g2:
    # 中间过渡状态
    s_t = Blurred, r_t = Low    # 降采样 224×224
else:  # CS_t < g1
    # 低信息冗余
    s_t = Discard               # 直接丢弃

# Step 4: 更新记忆队列
M_t = M_{t-1}.append(I_t with (s_t, r_t))

# Step 5: FIFO 驱逐
if len(M_t) > N_max:
    evict oldest token  # 严格 FIFO 出队
```
HVMM 的 EMA 机制使得阈值能在非平稳场景中自适应用：当场景突然加速（curvature 整体抬高），μ_t 和 σ_t 通过 EMA 跟随上升，阈值 g2 也相应提高——防止正常高曲率帧被过度保留挤占 memory bank。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HVMM 在 CurveStream 中的默认超参数：γ（EMA momentum，论文未明确具体值）、k1=0.0、k2=1.0、N_max=20。HVMM 独立使用（无 CAS 动态评分，退化为均匀交替分配 Clear 和 Blurred Memory）在 StreamingBench 上带来 +9.76% 的绝对提升（Table IX），在 OVOBench 上带来 +4.69%（Table X）——表明即使无智能感知，二值层级记忆结构本身也比单层 FIFO 更有效。HVMM 的 Clear Memory 保留比例自适应维持在 ~50%（图 3b），这在 accuracy 和 token 成本之间达到最优 tradeoff（比 100% Clear 减少 ~40% token 开销同时保持或提升 accuracy）。HVMM 的 K-Sigma 超参数对性能鲁棒：当 k1, k2 在不同组合下变化时（图 4），accuracy 保持高度稳定，验证了动态阈值相对于静态阈值的优越性。

涉及论文标题：
- CurveStream__Boosting_Streaming_Video_Understanding_in_MLLMs_via_Curvature-Aware_Hierarchical_Visual_Memory_Management
