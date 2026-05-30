## Native Tool Calling in LMMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Native Tool Calling 指 LMM 通过端到端训练将工具调用策略完全内部化的能力——模型自主决定何时调用工具、传什么参数、如何整合工具返回结果，无需外部检索代理或专家模型辅助决策。"Native"（原生）的核心在于工具调用能力是通过 tool-integrated fine-tuning 内嵌于模型权重中的，而非依赖外部规则或 prompt engineering。在 LongVT 中，native tool 具体指 crop_video(start_time, end_time) 函数：模型在需要时生成结构化 JSON 调用该函数，从原始视频指定时间段重新采样细粒度帧。

从算法pipeline角度拆解术语。通过联网搜索让回答具体和精准。
Native Tool Calling 的推理执行流程：
```
# 模型生成结构化 tool call
output = llm.generate(prompt)
# output 包含: <tool_call>
#   {"name":"crop_video","arguments":{"video_path":"...","start_time":763.0,"end_time":995.0}}
# </tool_call>

# External executor 执行工具（非模型内部）
tool_response = crop_video_executor(video_path, start_time, end_time)
# tool_response = {frames: [resampled_64_frames]}

# 工具结果注入回上下文
new_prompt = prompt + output + format_tool_response(tool_response)
# 模型基于新视觉证据继续推理
output2 = llm.generate(new_prompt)
# <think> verify evidence... </think>
# <answer> final answer </answer>
```
训练流程：SFT 通过模仿 Gemini/Qwen 蒸馏的 tool-augmented traces 教会模型工具调用语法和语义；RL 通过联合奖励优化工具调用的时机和精度；RFT 通过高质量自蒸馏轨迹巩固工具使用模式。消融实验（Figure 3b）证明：直接 RL（无 cold-start SFT）导致 tool call 频率崩溃至零；一旦通过 SFT 建立基础能力，模型在 RL 中 tool-call 频率和 accuracy 同步提升。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Native Tool Calling 的实现通常需要：(1) 工具定义（function name, description, parameters schema）以 system prompt 或特殊 tokens 形式注入；(2) 训练数据包含完整的 tool call → tool response → reasoning 交互轨迹；(3) 推理时通过特殊分隔符（如 <tool_call>/</tool_call>）解析工具调用、执行外部函数、将结果注入上下文。与 MCP (Model Context Protocol) 等标准化协议结合时，可通过统一的 tool server 管理多种工具。LongVT 在评估时部署 MCP server + vLLM continuous batching 架构。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
- SAGE__Training_Smart_Any-Horizon_Agents_for_Long_Video_Reasoning_with_Reinforcement_Learning

SAGE 中的 Native Tool Calling 使用方式（6-tool multi-turn agent）：
SAGE 扩展了 tool set 至 6 种工具——web-search (Serper Google Search API)、parse-website、transcribe-speech (Whisper-large-v3)、ground-event (Qwen3-VL-30B-A3B-Instruct)、extract-video-parts 和 analyze (Qwen3-VL-30B-A3B-Instruct)——使 orchestrator SAGE-MM 可进行 knowledge-driven multi-turn reasoning。与 LongVT 单一 crop_video 工具不同，SAGE 的 tool diversity 要求 SAGE-MM 学会智能选择工具：例如知道 F1 2024 赛季排名后，通过 web-search 缩小 2025 赛季视频的搜索空间。Tool calling 通过 JSON action 格式实现（而非 LongVT 的 XML tag 格式）：Stage-1 和 Stage-2 都输出推荐工具调用的 JSON 对象（含 rationale、name、arguments）。RL 训练中 s_reasonable-tool 奖励（GPT-4o judge）惩罚不合理的工具调用。消融实验（Table 10）：移除 transcribe-speech 降 5.5%（verbal 问题降 36.5%），移除 extract-video-parts 降 5.0%（visual 问题降 5.4%），移除 web-search 降 2.5%。per-tool accuracy（Table 18）：transcribe-speech 单独最高 61.1%，extract-video-parts/ground-event 最弱 50.2-50.3%（依赖其他工具做局部处理）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Modality Pre-fusion（模态预融合）是 LLaVA-Mini (Zhang et al., 2025) 提出的核心机制——在 LLM backbone 之前，用额外 Transformer 块将 vision token 中的视觉信息提前融合进 text token，使后续可将 vision token 极端压缩（甚至到 1 个 token）而不损失性能。动机来自论文对 LLaVA 架构的逐层注意力分析：vision token 主要在 LLM 前几层被 text token 通过 attention attend 以"吸收"视觉信息，深层中 vision token 被关注的 attention 急剧下降（80%+ 转向 instruction token），因此深层中的大量 vision token 是冗余的。Pre-fusion 模块的 N_fusion 个 Transformer decoder 块与 LLM backbone 同构（相同结构和超参数），将全部 vision token 和 text token 拼接后做 self-attention，然后仅提取 text token 对应位置的输出作为"融合 token"（fusion token），这些 text token 已携带了所需的视觉信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Pre-fusion 在 LLaVA-Mini pipeline 中的计算过程：
```
# 输入
H_v = ViT(image)  → Projection    # [576, d_h] 全部 vision token
H_q = LLM_Embed(text)             # [l_q, d_h] text token

# Pre-fusion: N_fusion 层与 LLM 同构的 Transformer decoder blocks
concat = Concat(H_v, H_q)         # [576 + l_q, d_h]
for i in range(N_fusion):         # 默认 N_fusion=4
    concat = PreFusionDecoderBlock_i(concat, causal_mask)

# 提取 text token 位置的输出作为 fusion token
H_q_fused = concat[-l_q:]         # [l_q, d_h]

# 后续: 压缩后的 vision token (甚至仅 1 个) + fusion text token 输入 LLM
```
其中 PreFusionDecoderBlock 与 LLM backbone 的 Transformer block 完全相同（包括 dimensions、heads、FFN 结构），但不共享权重。消融实验（Table 6）：N_fusion=0 时 1-token 达 VQA-v2 72.4/GQA 54.2/MMB 57.7；N_fusion=4 时提升到 77.6/60.9/65.6（1.96T FLOPs），远超只增加 vision token 数（144 token w/o pre-fusion 仅 76.9/58.9/64.9 at 2.85T FLOPs）。在相同 FLOPs 下，增加 pre-fusion 层比增加 vision token 数收益更大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Pre-fusion 模块实现为与 LLM backbone 完全同构的 Transformer decoder 层（如 Vicuna-7B 的 decoder block），在模型代码中直接复用 LLM 的 TransformerBlock 定义，设置层数 N_fusion=4。训练时 pre-fusion 模块参与 Stage-2 Instruction Tuning 的端到端训练。关键设计考量：(1) 放在 LLM 外部而非内部——如在 LLM 第 L 层执行 fusion，vision 经过早期层后携带上下文信息反而不利于后续压缩；放在外部也保持了 LLM backbone 不变，兼容所有 LLM 加速框架。(2) Pre-fusion 仅取 text token 位置输出——因为目的是将视觉信息融入文本，而非保留 vision token。(3) 对于视频，每帧 text token 的 fusion 结果经 pooling 聚合为视频级 fusion token。开源实现：https://github.com/ictnlp/LLaVA-Mini。

涉及论文标题：
- LLaVA-Mini__Efficient_Image_and_Video_Large_Multimodal_Models_with_One_Vision_Token
