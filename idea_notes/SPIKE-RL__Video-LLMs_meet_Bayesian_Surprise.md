## SPIKE-RL__Video-LLMs_meet_Bayesian_Surprise

- baseline方法是什么？
  Baseline 是 Video-LLM 的 **uniform frame sampling**（均匀帧采样）：将视频帧按固定间隔均匀采样 F 帧输入模型，所有帧的权重相同，无论帧内容如何。全栈执行例子：
  - 模型推理算法层：Video-LLM（如 Qwen2.5-VL）将视频视为 "bag of frames"，均匀采样后一次性自回归生成结果（caption / QA 答案）。没有信念演化过程，没有帧选择策略，对 routine 和 surprising 帧一视同仁。
  - 系统框架层：论文未明确说明。标准 Video-LLM 推理 pipeline，直接调用模型 API 或本地推理。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：论文未明确说明。GPU 上标准 FlashAttention-2 + bfloat16 推理。
  - 硬件架构层：论文未明确说明。在 H100 GPU 上运行。

  核心缺陷：(1) **缺乏信念追踪**：模型不维护对视频故事的演化理解，无法区分 routine 和 surprising 帧；(2) **信息冗余**：均匀采样倾向于采样高频 mundane 帧，错过关键的 surprising 时刻；(3) **query-agnostic 但无原则**：虽不依赖查询，但采样策略毫无信息优先级的引导。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **SPIKE/SPIKE-RL** 将 Bayesian Surprise 引入 Video-LLM，通过显式信念追踪和 surprise 引导的帧采样解决 uniform sampling 的三个缺陷：
  
  **(1) 信念追踪替代 bag-of-frames**：SPIKE 在每个时间步维护显式概率分布 P(belief | context)，生成文字化信念假设 B_t = {b_{t,1}, ..., b_{t,N}}（如 "the man will continue walking"），然后计算 P_prior（仅用历史上下文 H_t + 前序帧 W_t）和 P_post（加入当前观察帧 O_t），通过 KL散度 D_KL(P_post || P_prior) 量化 surprise。这给 Video-LLM 注入了人类式的"预期-现实"对比机制。
  
  **(2) surprise-weighted 采样替代 uniform 采样**：将 F 帧预算按 surprise 得分比例分配：p_i = softmax(S_i/τ_s)，高 surprise 段可被多次采样，确保关键事件帧不被遗漏。τ_s 控制采样集中度（τ_s=0.7 实验设置）。
  
  **(3) GRPO 优化信念质量**：SPIKE-RL 用 RL 训练假设生成器，reward 来自最终 caption 与 ground truth 的 LLM-Match 相似度，通过策略梯度反向优化中间信念假设质量。

  全栈执行例子（对比 baseline）：
  - 模型推理算法层：Baseline → Qwen2.5-VL 均匀采样 F 帧一次性推理。SPIKE → 先对视频做 W 帧滑动窗口，每一步生成 N=3 个文字假设 + 计算 Bayesian Surprise（KL散度），再按 surprise 概率采样 F 帧送入 Qwen2.5-VL 推理。SPIKE-RL → 额外使用 GRPO 在 2000 视频集上训练假设生成器，3 条 rollout 轨迹、LLM-Match reward、Z-score 归一化 advantage。
  - 系统框架层：论文未明确说明。SPIKE 是即插即用的推理时模块，替换 Video-LLM 的 uniform sampling layer。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：论文未明确说明。训练使用 DeepSpeed ZeRO-3 offload，推理使用 FlashAttention-2。
  - 硬件架构层：论文未明确说明。训练在 4×H100 单节点上。
