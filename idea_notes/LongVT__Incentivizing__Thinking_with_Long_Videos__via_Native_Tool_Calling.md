## LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling

- baseline方法是什么？
  Baseline是Qwen2.5-VL-7B-Instruct，采用标准text-only Chain-of-Thought (CoT)推理配合uniform frame sampling的前向推理流程。模型对输入视频均匀采样64或512帧，一次性编码所有vision tokens后送入LLM进行单轮文本推理生成答案。这种passive frame consumption方式存在三个核心缺陷：(1) 均匀采样无法自适应捕获关键视觉证据——长视频中证据稀疏且时间上分散，uniform sampling容易错过fine-grained决定性时刻；(2) 纯文本CoT推理缺乏视觉grounding——模型在不确定时倾向"blindly rephrasing"而非回到视频中核实，导致幻觉；(3) SFT仅为imitation-driven，存在exposure bias，无法泛化到分布外query和未见视频模板。
  
  Baseline全栈执行例子（Qwen2.5-VL-7B，64-frame uniform sampling，单轮长视频QA推理）：
  - 算法层：Qwen2.5-VL-7B-Instruct，uniform采样64帧，visual encoder编码→projector→LLM decoder自回归生成文本，无tool calling能力，无temporal grounding supervision
  - 系统框架层：vLLM inference engine，continuous batching serving，无工具调用协议
  - 编译框架层：论文未明确说明
  - kernel调度层：标准FlashAttention/VLLM PagedAttention，无crop-resample pipeline
  - 硬件架构层：NVIDIA A800-SXM4-80GB 8卡推理

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：(1) iMCoTT (interleaved Multimodal Chain-of-Tool-Thought)：将LMMs的潜在temporal grounding能力激活为native video cropping tool，实现global-to-local推理——先全局skim形成假设时间窗，再调用crop_video(start_time, end_time)重采样细粒度帧进行验证，支持最多5轮self-reflection；(2) 三阶段训练pipeline：SFT cold-start教会模型tool调用范式→Agentic RL (GRPO)用joint answer-temporal grounding reward (R = R_acc + R_format + R_time)优化探索→Agentic RFT蒸馏高质量rollouts稳定行为；(3) VideoSIAH数据套件：半自动pipeline生成247.9K SFT样本 + 1.6K RL样本 + 15.4K RFT样本 + 652条VideoSIAH-Eval基准。

  解决Baseline缺陷的对应关系：
  - 对抗uniform sampling错过关键证据：iMCoTT通过crop_video工具实现on-demand temporal retrieval，模型可根据全局预览自主选择感兴趣时间段进行细粒度重采样，而非被动消费均匀帧。RL阶段的时间grounding reward (IoU)进一步优化窗口提案精度，使模型学会定位稀疏证据。
  - 对抗纯文本CoT幻觉：iMCoTT使推理过程grounded在实际视觉证据上——模型先think形成假设，再调用crop_video获取验证证据，基于新证据重新think，可自我纠正初始错误（如Figure 8案例中模型通过re-check将pink纠正为blue）。避免了"blindly rephrasing"导致的虚假回答。
  - 对抗SFT imitation-driven限制：GRPO-based RL的exploratory rollouts + joint奖励函数使模型超越SFT distribution——IoU reward抑制span inflation（对比Recall reward的reward hacking），RFT阶段用高质量rollouts (answer正确 AND IoU≥0.3) 提供in-distribution supervision稳定优化。Table 3证实SFT+RL+RFT全pipeline显著优于单一阶段。
  - 对抗inference latency：尽管有multi-turn tool interactions，LongVT-RFT inference速度反而快于单轮baselines（Table 4），因为证据grounded的回答更简洁，避免了hallucination-driven verbose generation。

  论文方法全栈执行例子（LongVT-7B-RFT，512 frames，多轮tool calling，长视频推理）：
  - 算法层：(1) 全局skim 64 frames → visual encoder → projector → LLM生成初始假设窗口 <think> [t_s, t_e] </think>；(2) 调用 <tool_call>{"name":"crop_video","arguments":{"start_time":t_s,"end_time":t_e}}</tool_call> → 外部executor从原始视频[t_s, t_e]段重采样64帧 → 再次visual encoder → projector → vision tokens返回；(3) LLM基于新vision tokens重新think验证证据 → <answer>最终答案</answer>；最多5轮。LongVT-7B-RFT模型经过SFT→RL→RFT三个阶段训练以优化此过程。
  - 系统框架层：LMMs-Engine (SFT训练，stream packing buffer 51200 tokens)，verl + SGLang (RL训练，multi-turn multimodal tool-augmented rollouts，16 rollouts/prompt)，vLLM (推理serving，MCP server + continuous batching)
  - 编译框架层：论文未明确说明（使用AdamW optimizer, Liger Kernel for SFT/RFT）
  - kernel调度层：crop_video执行在外部executor上，非kernel级别优化；visual encoder的FlashAttention处理vision tokens；SGLang prefix caching复用多次tool calling间的共享prefix tokens
  - 硬件架构层：NVIDIA A800-SXM4-80GB，SFT用32卡，RL用64卡，RFT用64卡，推理评估用8卡
