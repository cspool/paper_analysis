## Query-Aware Frame Selection (查询感知帧选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Query-Aware Frame Selection 是一类用于长视频理解的帧采样策略，核心思想是根据输入查询（query）的内容自适应地选择最相关的视频帧，而非使用统一的固定采样策略（如 uniform sampling）。传统 uniform sampling 以固定间隔抽取帧，对所有查询一视同仁；query-aware 方法通过相关性评估机制（如 CLIPScore、LMM 评分、object detection）判断每帧对当前查询的价值，仅保留高相关度帧。DIG 论文的关键贡献在于识别出 query-aware selection 并非对所有查询类型都必要——论文定义了 global query（需要全视频理解）和 localized query（针对特定时间段），并证明：(1) 对 global query，uniform sampling 已足够有效；(2) 对 localized query，query-aware selection 才真正带来显著收益。同时，DIG 指出基于 CLIPScore 的 query-aware 方法（如 Q-Frame、AKS）在复杂推理场景中不可靠，因 CLIPScore 仅依赖浅层特征匹配，无法捕捉需要上下文推理和世界知识的查询意图。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Query-Aware Frame Selection 通用 pipeline
# 输入: 视频 V, 查询 Q, 目标帧数 N

# Step 1: 候选帧生成
candidates = get_candidates(V)
# 方式: uniform sampling (Q-Frame, AKS), 或 CAFS (DIG)

# Step 2: 帧-查询相关性评估
for frame in candidates:
    relevance[frame] = score(frame, Q)
# score(): CLIPScore → cosine_sim(CLIP_I(f), CLIP_T(Q))
#          LMM Reward → LMM CoT推理 + {"reward": 0-100}

# Step 3: 基于分数的帧选择
selected = select_by_scores(candidates, relevance, N)
# 方式: Top-K, 或 iterative reward-guided (DIG, 无参数)

# Step 4: LMM 推理
answer = LMM(selected, Q)
```
三种代表性方法的对比（基于 DIG Table 1）：Q-Frame (CLIPScore+Top-K) 在 >32 frames 时退化至低于 uniform；AKS (BLIP+adaptive) 在 >64 frames 时退化；DIG (LMM Reward+iterative) 在 8-256 frames 上持续提升。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Query-Aware Frame Selection 在 DIG 中作为 training-free pipeline 实现：对 localized query 启动 CAFS+LMM Reward+Video Refinement，对 global query 回退到 uniform sampling。查询感知的核心计算开销来自相关性评估——DIG 使用 vLLM 加速 LMM 推理（reward assignment 占总选择时间 ~70%）。开源：https://github.com/Jialuo-Li/DIG。其他 query-aware 方法包括 BOLT（LLM agent 多轮搜索）、T*（temporal search + object detection）、MDP3（list-wise frame selection）等。

涉及论文标题：
- Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding
