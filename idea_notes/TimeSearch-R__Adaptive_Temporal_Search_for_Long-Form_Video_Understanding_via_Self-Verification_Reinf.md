## TimeSearch-R__Adaptive_Temporal_Search_for_Long-Form_Video_Understanding_via_Self-Verification_Reinforcement_Learning

- baseline方法是什么？
  Baseline 是**手设计的时序搜索 (Hand-crafted Temporal Search)**，以 VideoAgent、T*、VideoTree 为代表。典型执行流程：
  - 模型推理算法层：VideoAgent 用 LLM (GPT-4) 作为中央 agent，通过 prompt 驱动多轮工具调用——先调用 VLM (GPT-4o) 做帧 captioning，再调用 CLIP 做帧检索，然后在纯文本模态中聚合信息做推理预测答案。T* 先用 VLM 从问题中提取目标物体，再调用目标检测模型 (YOLO-world-110M) 定位包含目标物体的关键帧，最后用检索到的帧集完成问答。VideoTree 引入树结构搜索来提高效率。所有方法均依赖人工设计的搜索工作流，缺乏端到端优化，搜索策略是次优的。
  - 系统框架层：VideoAgent 和 T* 均基于 API 模型 (GPT-4/GPT-4o) 做多轮调用，无训练过程。各 agent 通过 prompt engineering 编排 VLM/CLIP/YOLO 等子模型，非训练型框架。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：论文未明确说明。T* 使用 YOLO-world-110M (轻量检测器) 做帧检索，VideoAgent 使用 CLIP-1B 做相似度搜索，均为标准模型推理无自定义 kernel。
  - 硬件架构层：推理均在 GPU 上 (A100)，无硬件架构定制。

  核心缺陷：(1) **搜索策略次优**：手设计的搜索工作流无法泛化到不同问题/视频类型，搜索决策缺乏数据驱动的优化；(2) **搜索-推理割裂**：帧集在推理开始前固定，而实际视频推理是动态过程，中间推理结果应能驱动进一步搜索；(3) **多模型编排复杂**：VideoAgent 需协调 LLM agent + VLM captioner + CLIP retriever，T* 需 VLM + YOLO 检测，流水线脆弱且难以端到端优化；(4) **端到端延迟高**：VideoAgent 的 end-to-end latency 为 34.9s (Haystack-Ego4D)，多模型多次调用产生大量 overhead。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **TimeSearch-R** 将时序搜索重新定义为 text-video 交错的思维过程 (Interleaved Text-Video Thinking)，通过端到端 RL 从数据中学会最优搜索策略。核心机制 GRPO-CSV (Completeness Self-Verification) 补充原始 GRPO 仅奖励最终答案的缺陷，监督中间搜索步骤。

  对应解决四个缺陷：
  **(1) 端到端优化替代手设计工作流**：TimeSearch-R 将搜索策略学习建模为 policy optimization：policy model π_θ 在每个推理步 k 自主决定是否搜索、搜索哪个时间区间 [t_s^k, t_e^k]、用什么文本 query q^k、要多少帧 F。通过 GRPO 的 8 个 rollout 比较和 advantage-based 更新，模型从数据中学会最优搜索策略，无需人工规定搜索 pipeline。Temporal F1 从 baseline T* 的 2.5 提升到 8.1（3 倍以上）。
  **(2) 搜索-推理深度交错**：将搜索和推理融为一体——每轮 <think>...</think> 推理后可以跟 <tool_call>...</tool_call> 搜索请求，搜索结果直接追加到 CoT 供下一轮推理使用，实现了搜索和推理的循环迭代。这模拟了人类"假设驱动搜索"的认知模式：根据中间推理结果驱动进一步搜索。
  **(3) 单一的端到端模型**：仅需一个 Qwen2.5-VL-7B 模型完成所有推理和搜索决策，搜索函数 (SigLIP-400M + DPP) 仅作为轻量环境接口执行帧检索。不需要像 VideoAgent 一样协调多个异构模型。
  **(4) 更低延迟**：TimeSearch-R end-to-end latency 为 13.4s，比 VideoAgent 的 34.9s 降低 61.6%。因为搜索决策直接由 policy model 生成（无需额外 LLM agent 调度），且 DPP 帧选择比 CLIP retrieval 更高效。

  全栈执行例子（对比 baseline）：
  - 模型推理算法层：
    Baseline (VideoAgent) → LLM agent 分析问题 → 调用 VLM captioning 获取帧描述 → 调用 CLIP 检索相似帧 → 在文本空间聚合描述做推理 → 输出答案。搜索和推理在模态间转换，搜索策略由 prompt 固定。
    TimeSearch-R → 初始预览 Ṽ (768 frames @ 2fps) → π_θ 在 <think> 中推理 "I need to find when the person starts cooking" → <tool_call>{"name":"seek_video_frames","arguments":{"query":"person cooking in kitchen","start_time":120,"end_time":300,"num_frames":8}}</tool_call> → SigLIP-400M 计算候选帧嵌入 → DPP 选出 8 帧（兼顾相关性和多样性）→ 帧 + 时间戳返回追加到 CoT → π_θ 继续推理 "The cooking started at 180s-240s, now I need to check..." → 继续搜索或输出 <answer>。关键区别：搜索参数 (query, start_time, end_time) 由模型端到端学会，而非 hand-crafted；CSV 奖励确保搜索到的帧确实足以支撑正确答案。
  - 系统框架层：
    Baseline → 多模型编排 (GPT-4 agent + VLM + CLIP/YOLO)，API 调用频繁，无模型训练。
    TimeSearch-R → SFT (GPT-4o 生成交错 CoT 数据) → RL with GRPO-CSV on TRL library。训练：32 × A100 GPU，DeepSpeed ZeRO-3 Offload，vLLM colocate mode，Flash Attention 2.0，bfloat16。KL penalty β=0.005，batch size per GPU=1，gradient accumulation=2，AdamW lr=1e-6。推理：单模型 end-to-end，通过 <tool_call> 接口调用 SigLIP + DPP 搜索函数。
  - 编译框架层：论文未明确说明。使用 PyTorch native DDP + DeepSpeed ZeRO-3。
  - kernel 调度层：论文未明确说明。使用 Flash Attention 2.0 加速 attention 计算，DPP 搜索使用 standard matrix operations。
  - 硬件架构层：训练在 32 × NVIDIA A100 GPU 上，推理在 A100 GPU 上。无硬件架构定制。
