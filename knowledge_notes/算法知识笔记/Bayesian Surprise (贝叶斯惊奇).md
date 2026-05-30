## Bayesian Surprise (贝叶斯惊奇)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bayesian Surprise 是 Itti & Baldi (2005, NIPS) 提出的一种基于信息论的注意力量化理论。其核心定义：Surprise = 新观测数据引入后，观测者对世界模型的信念分布从先验到后验的信息增益，即 KL 散度 D_KL(P_posterior || P_prior)。当新数据迫使信念发生剧烈变化时，KL 散度大，surprise 高；当新数据与先验一致时，KL 散度小，surprise 低。原论文在视觉显著性预测中证明 Bayesian Surprise 是人类注意力最强的已知吸引因子（72% 的 human gaze shifts 指向比均值更 surprising 的区域，多人一致性时为 84%）。SPIKE 论文首次将 Bayesian Surprise 引入 Video-LLM 推理：将模型的"信念"表示为对文字化假设（textual hypotheses，如 "the man will continue walking"）的概率分布，先验 P_prior 从历史文本摘要 H_t + 前序帧窗口 W_t 计算，后验 P_post 加入当前观察帧 O_t 后计算，surprise score S_t = D_KL(P_post || P_prior)（实际使用 JSD 替代 KL 以得 [0,1] 范围），用于指导后续的 surprise-weighted frame sampling。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Bayesian Surprise 在 SPIKE 中的计算流程
# 输入: 视频帧 X_{1:T}, prior_window_size W=4, hypotheses_N=3
# 输出: 每个时间步 t 的 scalar surprise score S_t ∈ [0,1]

# Step 1: 生成信念假设 (通过 Video-LLM nucleus sampling)
H_t = summarize(X_{t-C:t-W-1})              # 历史文本摘要 (BART-Large-CNN)
W_t = X_{t-W:t-1}                           # 前序帧窗口
O_t = X_t                                   # 当前观察帧

B_t = VideoLLM.generate("predict next frame", H_t, W_t, temperature, top_p)
      # B_t = {b_{t,1}: "the man walks away", 
      #         b_{t,2}: "the man trips and falls",
      #         b_{t,3}: "the man stops to look"}

# Step 2: 计算先验分布 P_prior (仅基于历史+前序帧，不包含当前帧)
for b in B_t:
    NLL_prior = -log P_M(b | H_t, W_t)       # Video-LLM 给出的负对数似然
P_prior = softmax(-[NLL_prior_i] / τ)       # τ 为温度参数

# Step 3: 计算后验分布 P_post (包含当前观察帧 O_t)
for b in B_t:
    NLL_post = -log P_M(b | H_t, W_t, O_t)  # 加入当前帧后重新评估
P_post = softmax(-[NLL_post_i] / τ)

# Step 4: Bayesian Surprise = JSD(P_post, P_prior)  (用 JSD 替代 KL)
M = 0.5 * (P_post + P_prior)
S_t = 0.5 * D_KL(P_post || M) + 0.5 * D_KL(P_prior || M)
# S_t ∈ [0, 1], 经 log_2 归一化
# S_t 大 = 当前帧 O_t 显著改变了模型对视频事件的理解
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Itti & Baldi 原实现使用 72 个空间滤波器，维护 432,000 个概率分布逐帧更新（计算量极大，2005 年硬件上 500 帧需数小时）。SPIKE 论文创新性地用 Video-LLM 的 token-level NLL 替代空间滤波器，通过文字化假设将 Bayesian Surprise 转化为语言模型可计算的量：不需要贝叶斯推断显式更新参数，只需 Video-LLM 在两个上下文（有无当前帧）下对同一假设文本做两次 forward pass 取 NLL 差值。这使 Bayesian Surprise 的计算开销降至 O(F·N)（F=帧预算，N=假设数，N=3），与推理时间 scaling 方法可比。应用场景：视频异常检测、surprise 定位、自适应帧采样、机器人异常监控等。

涉及论文标题：
- SPIKE-RL__Video-LLMs_meet_Bayesian_Surprise
