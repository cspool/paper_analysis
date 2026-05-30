## Belief Tracking (信念追踪)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Belief Tracking（信念追踪）是认知科学中 Bayesian Theory of Mind (ToM) 框架的核心机制：智能体持续维护和更新对世界状态、其他智能体意图、或未来事件预测的内部模型（信念），当新证据出现时通过贝叶斯更新修正信念。在 NLP/AI 领域，Belief Tracking 传统上用于对话系统（追踪用户意图/槽位）和 Theory of Mind 推理（追踪角色心理状态）。SPIKE 论文将其扩展到 Video-LLM：信念被显式表示为可解释的文字假设 B_t = {b_{t,1}, ..., b_{t,N}}（如 "the delivery person will hand over the package"），每个假设包含先验概率 P_prior 和后验概率 P_post，形成完整的时间追踪链 {(B_1, S_1), (B_2, S_2), ..., (B_T, S_T)}。信念更新通过 Bayesian Surprise（KL/JSD）量化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Belief Tracking 在 SPIKE 中的实现
# 核心: 维护一个随时间演化的信念假设集合 + 历史文本摘要

# 初始化
H_t = ""                                    # 滚动文本摘要 (rolling memory)
B_history = []                              # 信念轨迹

for t in range(1, T+1):
    # 1. 从 Video-LLM 生成 N 个对未来事件的文字假设
    B_t = VideoLLM.generate(
        prompt="predict what will happen next",
        memory=H_t,                         # 历史事件摘要
        prior_frames=X_{t-W:t-1},           # 前 W=4 帧
        sampling="nucleus",                 # top_p nucleus sampling
        N=3                                 # 生成 3 个假设
    )

    # 2. 计算每个假设的 prior/posterior 概率
    O_t = X_t                               # 当前观察帧
    for b in B_t:
        NLL_prior[i] = -VideoLLM.log_prob(b | H_t, X_{t-W:t-1})
        NLL_post[i]  = -VideoLLM.log_prob(b | H_t, X_{t-W:t-1}, O_t)
    
    P_prior = softmax(-NLL_prior / τ)
    P_post  = softmax(-NLL_post / τ)

    # 3. Surprise = 信念分布变化量
    S_t = JSD(P_post, P_prior)

    # 4. 更新历史摘要 (追加当前帧描述，压缩到 ~200 词以内)
    event_desc = VideoLLM.caption(O_t, H_t, X_{t-W:t-1})
    H_t = BART_Large_CNN.summarize(H_t + event_desc)   # 滚动压缩

    # 5. 记录信念轨迹（可解释、可回溯）
    B_history.append((B_t, P_prior, P_post, S_t))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 SPIKE 中，Belief Tracking 通过以下组件实现：(1) Video-LLM（Qwen2.5-VL-7B）作为假设生成器——给定 H_t（文本内存）和 W_t（视觉前序帧），经 nucleus sampling 生成 N=3 个短假设（8-10 词）；(2) BART-Large-CNN 作为历史摘要压缩器——维持 ~200 词以内的滚动文本摘要，避免 prompt 过长；(3) 两次 forward pass（有无 O_t）获取 NLL 差值计算 surprise。这一机制的创新在于：传统 Video-LLM 无信念演化概念，将视频视为 "bag of frames"；SPIKE 赋予 Video-LLM 人类式的"预期→观察→更新"循环。未来可扩展至实时流处理、人机交互中的预期管理、异常行为预警。

涉及论文标题：
- SPIKE-RL__Video-LLMs_meet_Bayesian_Surprise
