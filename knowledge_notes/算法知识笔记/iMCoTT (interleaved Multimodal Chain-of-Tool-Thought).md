## iMCoTT (interleaved Multimodal Chain-of-Tool-Thought)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
iMCoTT（交织多模态工具思维链）是 LongVT 论文提出的长视频推理范式。它将传统的纯文本 CoT (Chain-of-Thought) 扩展为 "推理步骤" 与 "视觉工具调用" 交织进行的循环过程。具体流程：(1) 模型首先对长视频进行全局 skim（均匀采样少量帧），形成关于证据所在时间段的粗粒度假设；(2) 模型以结构化格式 <tool_call>{"name":"crop_video","arguments":{"start_time":t_s,"end_time":t_e}}</tool_call> 调用原生视频裁剪工具，请求重采样指定时间窗口内的细粒度帧；(3) 工具返回裁剪后的视频帧（以 vision tokens 形式），模型基于新视觉证据重新 think，验证或修正假设；(4) 模型决定直接回答或进入下一轮工具调用（最多 5 轮）。iMCoTT 的核心创新在于将 LMM 的潜在 temporal grounding 能力通过 tool-integrated fine-tuning 激活，无需外部检索模型或专家模型。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
iMCoTT 的单次推理流程伪代码：
```
def iMCoTT(video, question, max_turns=5):
    # 全局 skimming: 均匀采样少量帧
    global_frames = uniform_sample(video, n=64)
    vision_tokens = visual_encoder(global_frames)
    context = [vision_tokens, question_text]
    
    for turn in range(max_turns):
        # Step 1: 模型思考，可能提出时间窗口
        output = llm.generate(context, stop=["</think>"])
        think_text = parse_think(output)
        
        # Step 2: 如果模型认为需要进一步检查
        if contains_tool_call(output):
            tool_args = parse_tool_call(output)
            # 调用 crop_video 工具
            cropped_frames = crop_video(video, 
                                        tool_args["start_time"],
                                        tool_args["end_time"])
            cropped_tokens = visual_encoder(cropped_frames)
            context.append(tool_response(cropped_tokens))
            continue  # 进入下一轮 think-verify
        
        # Step 3: 模型有足够证据，给出答案
        answer = parse_answer(output)
        return answer
    
    return answer
```
具体计算流程：输入 prompt = [system_prompt_with_tool_def] + [global vision tokens] + [question] → LLM decoder 逐 token 生成（最大 16384 tokens）→ 解析 <think>...</think>、<tool_call>...</tool_call>、<answer>...</answer> 标记 → 若解析到 tool_call，外部 executor 执行 crop_video → 重采样帧再编码为 vision tokens → 追加 <tool_response> 到上下文 → 继续生成下一轮。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
iMCoTT 通过三阶段训练实现：(1) Cold-Start SFT用 tool-augmented traces（由 Gemini 和 Qwen 模型蒸馏生成，含 <think>/<tool_call>/<tool_response>/<answer> 结构） 教模型工具调用范式；(2) Agentic RL (GRPO) 用联合奖励（answer accuracy + format compliance + temporal IoU）优化模型何时调用工具、裁剪多长时间、如何整合证据；(3) Agentic RFT 用 RL 阶段的高质量 rollout traces (answer 正确且 temporal IoU ≥ 0.3) 进一步微调稳定行为。推理时部署为 vLLM + MCP server 架构，通过特殊分隔标记解析多轮交互。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
