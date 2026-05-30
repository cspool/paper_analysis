## SPIKE-RL__Video-LLMs_meet_Bayesian_Surprise

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：(1) **SPIKE**：推理时框架，将 Video-LLM 的信念表示为显式概率分布（文字化信念假设），通过 KL 散度计算 Bayesian Surprise 得分，引导 surprise-weighted frame sampling。(2) **SPIKE-RL**：基于 GRPO 强化学习优化信念假设生成，LLM-Match 作为 reward signal，通过最终视频 caption 质量反向传播 credit 到中间信念序列。实验比较 surprise localization（三个 benchmark：Oops!、FunQA、Mr. Bean）和下游任务（BlackSwan、FunQA Task 2、ExFunTube、VideoMME-S、NextQA）上与 uniform sampling、RGB Histogram、ECR、Katna、Optical Flow 等 query-free 采样方法的性能。

- 硬件平台是什么，配置是什么。
  训练：4 × H100，单节点，DeepSpeed ZeRO-3 offload。推理：论文未明确单独给出推理平台，但使用 Qwen2.5-VL-7B-Instruct 作为 backbone，FlashAttention-2、bfloat16、PEFT。

- 模型是什么。数据集和bench分别是什么。
  模型：Backbone 为 Qwen2.5-VL-7B-Instruct（主要）和 Qwen2.5-VL-32B（扩展）；LLM-Match reward model 为 Olmo-7B-hf；历史摘要压缩使用 BART-Large-CNN。训练集：2000 个视频，30% surprising（Oops! 训练集）+ 70% unsurprising（ActivityNet Captions）。评测 benchmark：
  - Surprise Localization: Oops!（4,791 视频，精确时间戳）、FunQA（424 视频，标注最 surprising 片段）、Mr. Bean（48 视频，自定义，笑声轨道为标注）
  - 下游任务: BlackSwan Suite（Reporter-MCQ）、FunQA Task 2（解释生成）、ExFunTube（解释生成）、VideoMME-S（多模态推理，短视频无字幕）、NextQA（时间/常识/因果推理）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/sahithyaravi/SPIKE-RL。基于 Qwen2.5-VL（开源）训练 SPIKE-RL。

  算法 pipeline 核心流程：
  ```
  输入: 视频帧序列 X_{1:T}，帧预算 F
  输出: surprise-guided 采样的 F 帧

  # Step 1: 均匀采样 K ≤ F 个时间锚点
  timesteps = uniform_sample(T, K)

  For each timestep t = t_1, ..., t_K:
      # Step 2: 构建上下文
      W_t = X_{t-W:t-1}           # 前序帧窗口（W=4）
      H_t = summarize(X_{t-C:t-W-1})  # 历史文本摘要（BART-Large-CNN 压缩）

      # Step 3: 生成信念假设（N=3 个假设）
      B_t = {b_{t,1}, ..., b_{t,N}}
          = VideoLLM.generate(H_t, W_t, nucleus_sampling)

      # Step 4: 计算先验分布 P_prior
      For each hypothesis b_{t,i}:
          NLL_prior_i = -log P_M(b_{t,i} | H_t, W_t)
      P_prior = softmax(-NLL_prior / τ)   # τ 为温度参数

      # Step 5: 计算后验分布 P_post（加入当前帧 O_t = X_t）
      For each hypothesis b_{t,i}:
          NLL_post_i = -log P_M(b_{t,i} | H_t, W_t, O_t)
      P_post = softmax(-NLL_post / τ)

      # Step 6: Bayesian Surprise = KL(P_post || P_prior)
      S_t = D_KL(P_post || P_prior)
          = Σ_i P_post(b_{t,i}) * log(P_post(b_{t,i}) / P_prior(b_{t,i}))
      # 实际使用 JSD 替代 KL：S_t = JSD(P_post, P_prior) ∈ [0, 1]

  # Step 7: Surprise-weighted 采样
  p_i = softmax(S_i / τ_s), τ_s = 0.7  # 将 surprise 转为采样概率
  selected_frames = multinomial_sample(F, p)  # 高 surprise 段可被多次采样
  ```

  SPIKE-RL 训练流程（GRPO）：
  1. 对每个视频，执行 M=3 条 rollout 轨迹
  2. 每条轨迹：SPIKE 产生信念假设 + surprise 得分 → surprise-weighted 采样帧 → VideoLLM 生成 caption c
  3. LLM-Match 评估 caption c 与 ground truth 相似度，得 reward R
  4. 组内 Z-score 归一化：A^{(r)} = (R^{(r)} - μ_R) / σ_R
  5. 策略梯度优化：L = -1/M Σ_r A^{(r)} Σ_t Σ_k log p_θ(b_{t,k}^{(r)} | H_t^{(r)}, W_t^{(r)})
  6. 训练超参：LR=1e-6，GRPO β=0.1，N_hypotheses=3，max_prompt_length=8192，batch_size=4，epochs=1
