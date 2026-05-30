## EVA__Efficient_Reinforcement_Learning_for_End-to-End_Video_Agent

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：EVA 是一个基于强化学习的端到端视频 Agent 框架，核心是 planning-before-perception 范式，通过迭代 summary–plan–action–reflection 循环实现自主视频理解。技术栈：(1) MDP 建模 —— s_t = {q, h_t, F_t}（query、历史、视觉证据），策略 π_θ(a_t | s_t) 参数化为 MLLM；(2) 灵活帧选择工具 —— 支持 start_time、end_time、nframes、resize 四个参数，同时控制时间和空间粒度；(3) 三阶段训练 pipeline —— SFT Cold-Start（10k 样本，训练 tool-call 格式、交错图文推理、帧级理解和基本帧选择策略，lr=2e-6, bs=8, 2 epochs）→ KTO 纠错（11k 标注策略，63% correct + 37% rejected，纠正猜测/欠采样/过采样等失败模式，lr=2e-6, β=0.1）→ Data-Enhanced GRPO 在线强化学习（9.6k 开放 QA + 1.1k MCQ，batch size=64, rollouts=8 per sample, lr=1e-6, 1 epoch，收集失败案例让 teacher MLLM 为 HD-VILA 新视频生成新 QA pairs）；(4) Reward —— Accuracy（MCQ: CSV self-verification r_csv；open-ended: ROUGE r_rouge = (R1+R2+RL)/3）+ Format Reward（tool call 但答案错误给 0.05 补偿抑制猜测）。训练用 32 H100 GPU。

  实验比较：(a) LSDBench —— 对比 Gemini-2.0-Flash (2700 frames/696.6K tokens)、LongVA、Qwen2-VL、Qwen2.5-VL、LongVila 等，EVA 用 76.9 frames/10.3K tokens 达到 51.0%（vs baseline Qwen2.5-VL* 49.2% with 32 frames/21K tokens）；(b) 长视频理解 —— LongVideoBench、MLVU、VideoMME、LVBench 上对比 GPT-4o、Gemini-1.5-Pro、Video-R1、VideoChat-R1、Qwen2.5-VL（static）和 VideoAgent、FrameThinker、VideoMTR（adaptive agent），EVA-GRPO 各 benchmark 分别达 55.0%、68.3%、60.2%、43.3%；(c) Video-Holmes 零样本视频推理 —— 7 子任务 vs GPT-4o、Gemini-2.0-Flash、InternVL2.5/3、Video-R1 等，EVA-GRPO 37.2%；(d) 消融 —— SFT→KTO→GRPO 阶段逐步增益、GRPO 数据组成（MC vs OE vs mixed）、ELV-Halluc（SAH Ratio 8.8%→5.0%）；(e) 效率 —— 总 token 可比或低于 uniform sampling，推理时间由自适应 compact visual tokens 决定。

- 硬件平台是什么，配置是什么。
  训练：32 × NVIDIA H100 GPU（GRPO 阶段）。推理评估：vLLM 框架（temperature=0），原始视频分辨率 720p。Base model：Qwen2.5-VL-7B-Instruct。Teacher MLLM：Qwen2.5-VL-72B（数据构造阶段）。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2.5-VL-7B-Instruct（主模型，支持多分辨率视觉输入）。Teacher MLLM: Qwen2.5-VL-72B。
  训练数据集（自建）：EVA-SFT（10k 样本，源 QA pairs 来自 llava-video 和 cgbench）、EVA-KTO（11k 标注帧选择策略）、EVA-RL（9.6k open-ended + 1.1k MCQ）。外部数据：HD-VILA（GRPO 数据增强阶段的新视频源）。
  Benchmarks（7 个）：(1) LSDBench —— 长视频采样困境 benchmark；(2) LongVideoBench —— 3763 videos/6678 QA pairs，最长 1h；(3) MLVU —— ~2600 QA pairs，平均 636s，9 任务；(4) VideoMME —— 900 videos/2700 QA pairs，30 子领域；(5) LVBench —— 1549 QA pairs，平均 4101s；(6) Video-Holmes —— 视频推理，7 子任务 SR/IMC/TCI/TA/MHR/PAR/CTI；(7) ELV-Halluc —— 语义聚合幻觉 benchmark。
  评价指标：Accuracy（全部 benchmark），Visual Token 数量（效率），SAH Ratio（ELV-Halluc）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/wangruohui/EfficientVideoAgent
  
  算法 pipeline 伪代码：
  ```
  # === EVA 推理流程（planning-before-perception） ===
  # s_0 = {q, h_0=[], F_0=[]}  # 初始仅 query

  for t in 1..max_rounds:
      # 1. Summary: 对当前帧生成详细描述
      summary_t = MLLM.summarize(F_{t-1})
      
      # 2. Planning: 基于 query + 历史 + summary 提出潜在 actions
      plan_t = MLLM.plan(q, h_{t-1}, summary_t)
      # 估算每个 action 的 token cost 和 expected outcome
      
      # 3. Action: 生成 frame_select tool call
      action_t = {
        "tool": "frame_select",
        "arguments": {"start_time": t0, "end_time": t1,
                      "nframes": N, "resize": r}
      }
      # resize=0.1 → 低分辨率全局浏览; resize=0.4~0.5 → 高分辨率聚焦
      F_t = F_{t-1} ∪ extract_frames(V, action_t.arguments)
      
      # 4. Reflection: 评估视觉信息是否充足
      if MLLM.reflect(q, F_t).is_sufficient: break

  answer = MLLM.answer(q, h_T, F_T)
  ```

  训练 pipeline 伪代码：
  ```
  # Stage 1: SFT Cold-Start (lr=2e-6, bs=8, 2 epochs)
  # 数据格式: Summary → Planning → Action → Reflection
  for batch in EVA-SFT:
      loss = CrossEntropy(MLLM(batch.input), batch.target)
      MLLM.backward(loss)
  
  # Stage 2: KTO Correction (lr=2e-6, β=0.1)
  # 单样本偏好: {trajectory, label ∈ {chosen, rejected}}
  for batch in EVA-KTO:  # 63% chosen + 37% rejected
      loss = KTO_loss(MLLM, batch, β=0.1, ref=MLLM_SFT)
      MLLM.backward(loss)
  
  # Stage 3: Data-Enhanced GRPO (lr=1e-6, bs=64, 8 rollouts/sample)
  for batch in EVA-RL:  # 90% OE + 10% MCQ
      # 采样 G=8 个候选响应
      responses = [MLLM.sample(q, v) for _ in range(8)]
      # 计算 reward
      for τ in responses:
          r_acc = r_csv(τ) if MCQ else r_rouge(τ)  # ROUGE = (R1+R2+RL)/3
          r_fmt = 0.05 if has_tool_call(τ) and not correct(τ) else 0
          R(τ) = w_acc * r_acc + w_fmt * r_fmt
      # GRPO advantage + policy update
      A_i = (R_i - mean(R)) / std(R)
      loss = GRPO_clip_loss(π_θ, π_ref, A)
      MLLM.backward(loss)
      # 每 N 步: 收集 failures, teacher 生成新 QA
      if step % N == 0:
          new_QA = teacher_MLLM(HD-VILA_new_video, 
                                in_context=failures)
          EVA_RL = EVA_RL ∪ new_QA
  ```
  
  使用例子：
  ```bash
  # vLLM serving + inference
  python serve_eva.py \
      --model_path /path/to/EVA-GRPO \
      --base_model Qwen2.5-VL-7B-Instruct
  
  # 评估 LSDBench
  python eval_lsdbench.py --model eva --checkpoint EVA-GRPO
  ```
