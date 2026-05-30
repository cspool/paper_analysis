## Multi-Turn Tool-Augmented Agentic Video Serving

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Turn Tool-Augmented Agentic Video Serving 是 LongVT 推理部署的核心系统架构：在标准的 LMM serving pipeline 中嵌入工具调用协议（MCP, Model Context Protocol），使模型能在多轮交互中动态调用 crop_video 工具读取视频片段。架构包含三个核心组件：(1) vLLM inference engine 处理 continuous batching 请求和 token 生成；(2) MCP server 监听模型输出中的 <tool_call> 标记，解析 JSON 参数后执行外部 crop_video 函数；(3) 特殊分隔符解析器（delimiter tags）从生成流中实时分离 reason step、tool invocation、final answer 三部分。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
单次推理的完整请求流程：
```
Client Request
  │  {video_path, question, max_turns=5}
  ▼
vLLM Engine (continuous batching)
  │  1. 加载视频，视觉编码 → vision tokens
  │  2. 构造 prompt: system_prompt(含tool_def) + vision_tokens + question
  │  3. 开始自回归生成
  ▼
Token Stream
  │  "<think>global preview suggests event at [t_s, t_e]</think>"
  │  "<tool_call>{\"name\":\"crop_video\",\"arguments\":{...}}</tool_call>"
  │  ← 检测到 </tool_call> → 暂停生成 → 触发 MCP server
  ▼
MCP Server
  │  1. 解析 tool_call JSON
  │  2. 调用 crop_video_executor(video_path, t_s, t_e)
  │  3. 重采样 64 frames → visual encoder → vision tokens
  │  4. 构造 <tool_response>...(cropped frames)...</tool_response>
  │  5. 追加到 vLLM 上下文
  ▼
vLLM Engine (恢复生成)
  │  1. 模型基于新 vision tokens 继续推理
  │  2. 可选：再次调用 crop_video (直到 max_turns)
  │  3. 生成 "<answer>final answer</answer>"
  ▼
Response to Client
  │  {answer, trace: [think, tool_calls, responses]}
```
关键系统设计：(1) 使用 vLLM PagedAttention 管理多轮交互中的 KV cache，避免重复编码全局帧；(2) 评估框架 LMMs-Eval 统一管理 benchmark 加载、推理调度和指标计算；(3) 推理 latency 实际上不高于 baseline——LongVT-RFT 的证据 grounded 回答更简洁避免 hallucination 导致的长输出。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 在 SFT 阶段使用 LMMs-Engine（PyTorch-based 训练框架，支持 stream packing 和 dynamic batching）；(2) RL 阶段使用 verl + SGLang（SGLang 提供 rollout 引擎，支持多轮 multimodal tool-augmented generation，16 rollouts/prompt）；(3) 推理评估阶段使用 vLLM (continuous batching) + MCP server + LMMs-Eval 评估框架。训练硬件：NVIDIA A800-SXM4-80GB GPU（SFT: 32卡, RL/RFT: 64卡），推理硬件：8 卡 A800。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
