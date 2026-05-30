## LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：三阶段训练pipeline（1）Cold-Start SFT：在Qwen2.5-VL-7B基础上用247.9K SFT样本（含19,161条tool-augmented iMCoTT traces）进行监督微调，教会模型提出时间窗口、调用crop_video工具、基于retrieved frames推理；（2）Agentic RL (GRPO)：用1.6K RL样本，以联合奖励函数（answer accuracy + format compliance + temporal IoU）优化策略，让模型学会何时检查视频、裁剪多长、如何整合证据；（3）Agentic RFT：用15.4K自蒸馏高质量RL rollout traces进一步微调，稳定agentic行为。
  实验比较：LongVT-7B-SFT/RFT vs Qwen2.5-VL-7B、Video-R1-7B、VideoRFT-7B、Video-Thinker-7B，以及专有模型GPT-4o、Gemini 1.5 Pro。在VideoMME、VideoMMMU、LVBench、VideoSIAH-Eval四个长视频理解benchmarks和Charades-STA temporal grounding benchmark上评估。

- 硬件平台是什么，配置是什么。
  NVIDIA A800-SXM4-80GB GPUs：SFT用32卡，RL和RFT用64卡。推理评估用8卡A800。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2.5-VL-7B-Instruct作为基座模型。
  数据集：自建VideoSIAH数据套件（247.9K SFT样本 + 1.6K RL样本 + 15.4K RFT样本 + 652条VideoSIAH-Eval QA对），辅以LLaVA-CoT(54.6K)、OpenVLThinker(2.8K)、We-Math 2.0(602)、LongVideo-Reason CoT(5.2K)、Video-R1 CoT(165.6K)作为SFT数据。
  Benchmarks：VideoMME（平均1018s/视频）、VideoMMMU（平均506s/视频）、LVBench（平均4101s/视频）、VideoSIAH-Eval（平均1688s/视频）、Charades-STA（temporal grounding）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：代码、数据、模型权重公开在 https://github.com/EvolvingLMMs-Lab/LongVT。
  
  算法pipeline伪代码：
  ```
  # Stage 1: Cold-Start SFT (32 GPU, LMMs-Engine, stream packing)
  model = Qwen2.5-VL-7B-Instruct
  train_data = concat(SFT_w_tool(19.2K), SFT_wo_tool(228.8K))
  for step in 1..3000:
      batch = dynamic_stream_packing(train_data, buffer_size=51200)
      loss = -Σ log P(x_t | x_<t)   # next-token prediction
      update(model, AdamW(lr=5e-5, cosine, warmup=300))

  # Stage 2: Agentic RL via GRPO (64 GPU, verl + SGLang)
  model = load(SFT_checkpoint)  
  for step in 1..160:
      for prompt in batch(16):
          # 采样K=16个rollouts，每个rollout包含多轮tool calling
          for k in 1..16:
              y_k ~ π_θ_old(·|x, max_new_tokens=16384
                             , max_prompt_len=36000)
              # y_k = <think>... hypothesize window ...
              #       <tool_call>{"name":"crop_video",
              #         "arguments":{"start_time":t_s,"end_time":t_e}}
              #       </tool_call>
              #       <tool_response> cropped frames </tool_response>
              #       <think>... verify evidence ...</think>
              #       <answer> final answer </answer>
          
          # 计算联合奖励
          R_acc[k] = Judge_LLM(answer_k, answer_gt) ∈ {0, 0.5, 1}
          R_fmt[k] = 1 if format == <think>..<tool_call>..<answer> else 0
          R_time[k] = IoU([t_s,t_e], [t_s',t_e'])
          R[k] = R_acc[k] + R_fmt[k] + R_time[k]
          
          # GRPO advantages
          b = mean(R[1..K])
          A[k] = R[k] - b
          
          # 更新policy (KL-constrained)
          J = 1/K Σ A[k] Σ log π_θ(y_t|x,y_<t) - β*KL(π_θ||π_ref)
      update(model, AdamW(lr=1e-6, constant))

  # Stage 3: Agentic RFT (64 GPU)
  # 筛选高质量rollouts: answer正确 AND temporal IoU ≥ 0.3
  rft_data = filter(RL_rollouts, 
                    answer_correct & IoU(span_pred, span_gt) >= 0.3)
  model = load(best_RL_checkpoint)
  for step in 1..1600:
      batch = dynamic_stream_packing(rft_data, buffer_size=51200)
      loss = -Σ log P(x_t | x_<t)
      update(model, AdamW(lr=5e-5, cosine, warmup=160))
  ```

  张量计算示例（RL阶段，单rollout，B=1）：
  - 输入：prompt x + 512 frames video → 视觉编码器 → vision tokens (~512×256=131K tokens) + text tokens
  - 模型生成：max 16384 new tokens（含think + tool_call JSON + tool_response + think + answer）
  - crop_video执行：根据start_time/end_time从原始视频中重采样64帧 → 再次编码为vision tokens
  - KL divergence计算：每token位置计算 π_θ(·|x,y_<t) 与 π_ref(·|x,y_<t) 的KL散度
  - IoU计算：对predicted [t_s, t_e] 和 GT [t_s', t_e'] 计算 |intersection|/|union|
  - 多轮：up to 5 turns（T1到T5），每轮可再次调用crop_video精炼窗口
