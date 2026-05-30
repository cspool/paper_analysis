## SGLang for Tool-Augmented RL Rollouts

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SGLang 是一个高效的开源 LLM serving 框架，核心理念是 Structured Generation Language——通过程序化接口将 LLM 调用、prompt 管理和生成控制结构化。在 LongVT 中，SGLang 被扩展用作 RL 阶段的 rollout 引擎：(1) 提供高效的并行 rollouts 生成能力（16 rollouts/prompt, 64 GPU）；(2) Native support for tool calling——支持模型生成结构化 JSON tool call、暂停生成等待外部工具执行、将 tool response 注入上下文后继续生成；(3) RadixAttention prefix caching 在多轮 tool calling 间自动复用共享的 prefix tokens；(4) continuous batching 实现高吞吐量的并发 rollouts。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
SGLang 在 LongVT RL 中的多轮 tool-calling rollouts 执行流程：
```
# SGLang Engine 配置
engine = sglang.Engine(
    model_path="Qwen2.5-VL-7B-SFT",
    tp_size=4,
    mem_fraction_static=0.85,
    max_total_tokens=36000  # 防止 OOM
)

# 单次 rollout 的多轮 tool calling
def generate_rollout(prompt, video_path):
    messages = [system_prompt_with_tool_def, user_prompt]
    for turn in range(5):
        # 1. SGLang 生成 (支持 continuous batching)
        output = engine.generate(messages, 
                                 max_new_tokens=16384,
                                 temperature=1.0)[0]
        
        # 2. 解析 output: 检测 <tool_call> 标记
        parsed = parse_delimited_output(output)
        
        # 3. 如果有 tool_call，执行并注入结果
        if parsed.tool_call:
            tool_result = execute_crop_video(video_path, **parsed.tool_args)
            messages.append(output)                    # 保留模型输出
            messages.append(format_tool_response(tool_result))  # 注入视觉结果
            continue  # 下一轮基于新证据推理
        
        # 4. 成功生成 answer
        return {"rollout": output, "turns": turn + 1}
    
    return {"rollout": output, "turns": 5}
```
SGLang 在 LongVT 中的关键价值：(1) RadixAttention prefix caching——全局帧在首轮编码后，后续轮次无需重新编码，仅在 tool_response 部分消耗新 tokens；(2) continuous batching——不同 prompt 的 rollouts 在同一 batch 中交错执行，GPU 利用率高；(3) 支持 structured output 解析（JSON tool call 格式验证）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SGLang 通过 PyPI (`pip install sglang`) 安装。在 LongVT 的 verl 集成中，SGLang 作为 rollout engine 被 Ray 管理：每个 rollout worker 启动一个 SGLang engine 实例，通过 HTTP/gRPC 接口接收 prompt 并返回生成的 rollouts。2025 年 SGLang 已获 a16z Open Source AI Grant，支持 400,000+ GPU 的全球部署，是 xAI (Grok 3)、Microsoft Azure (DeepSeek R1)、NVIDIA 等公司的生产推理引擎。SGLang 支持多种推理优化：speculative decoding (EAGLE/EAGLE3/NEXTN)、chunked prefill、PD disaggregation、MoE EP 并行等。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
