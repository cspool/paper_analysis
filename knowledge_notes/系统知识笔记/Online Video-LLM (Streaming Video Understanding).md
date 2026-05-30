## Online Video-LLM (Streaming Video Understanding)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Online Video-LLM（在线视频大语言模型）是一类特殊的 Video Large Language Model，区别于传统的 Offline Video-LLM（输入完整视频后一次性生成答案），Online Video-LLM 需要：**始终在线（always-on）**、**逐帧接收视频流输入**、**具备时间决策能力（temporal decision-making）**——即在上下文合适的时刻主动生成响应，而非被动每帧输出或仅在用户提问时响应。其核心挑战是"响应-沉默"决策：模型必须在大量沉默帧（无需输出）和少数响应帧（需要输出）之间动态切换。代表模型包括 VideoLLM-online (EOS prediction)、VideoLLM-MoD (MoD token selection)、MMDuet (classification head)、LION-FS (dual-path routing)，以及 LiveStar (SVeD perplexity-based gating)。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 LiveStar 的 Serving 系统架构中，Online Video-LLM 的运行模式与传统 Offline VLM 的根本区别：

```
Offline Video-LLM (单次推理)：
  [Complete Video] → Vision Encoder → [All Frame Tokens] 
  → LLM Prefill → Decode → [Single Answer]

Online Video-LLM (流式推理)：
  for frame in stream:
      [Frame] → Vision Encoder → [16 tokens] → Append to Ctx
      → Verification/Silence Decision
      → (if Response) Decode [New Caption]
      → (if Silence) Maintain State
      → Memory Management (long videos)
  → [Stream of Time-Stamped Captions]
```

关键差异：(1) **Causal Constraint**：Online 模式下模型不能看到未来帧（严格因果），Offline 模式可看到完整视频；(2) **Incremental Processing**：Online 模式需维护跨帧状态（KV Cache），逐步积累上下文；(3) **Temporal Decision-Making**：Online 模式需自主决定何时输出，而非按外部指令输出；(4) **Memory Management**：Online 模式需在 10+ 分钟视频中管理 KV Cache 避免 OOM。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Online Video-LLM 的实现通常基于以下组件：(1) Streaming-capable vision encoder — 能够逐帧编码（如 InternViT, CLIP ViT），每帧产生少量 visual tokens（如 LiveStar 使用 16 tokens/frame）；(2) Streaming LLM backbone — 支持增量推理（如 InternLM2.5-7B），配合 KV Cache 复用历史上下文；(3) Response-Silence mechanism — 多种方案可选：EOS prediction（VideoLLM-online）、classification head（MMDuet）、perplexity gating（LiveStar/SVeD）；(4) Memory compression — 如 Peak-End (LiveStar)、FIFO、Uniform 等策略在长视频中调控 context 大小。评估 benchmark 包括 OmniStar (15 scenarios × 5 tasks)、Ego4D Narration Stream、SVBench 等，指标涵盖语义准确性（SemCor）、时间对齐度（TimDiff）、冗余度（TimRedun）和推理速度（FPS）。

涉及论文标题：
- LiveStar__Live_Streaming_Assistant_for_Real-World_Online_Video_Understanding
- StreamingEval__A_Unified_Evaluation_Framework_for_Streaming_Video_Understanding

StreamingEval 进一步从评估角度补充：Online Video-LLM 的流式在线问答任务定义为：在时刻 t，模型持续接收视频帧 V^t，维护交互历史 C^t，当用户 query Q^t 在 t0 时刻发出、编码在 t1 时刻完成时，模型基于 $p(R_{t1} | C_{t1}, V_{\text{enc}}[0,t1], Q_{t1})$ 自回归生成答案，强调严格因果约束（仅能访问已编码的过去帧）。StreamingEval 通过三进程异步 pipeline（Frame Player → Encoder-Memory Updater → Responder）模拟真实流式环境，以 MaxFPS、TTFT、Memory_bank、Accuracy 和综合 StreamingScore 评估模型的端到端部署能力。关键发现：离线 VideoLLM accuracy 普遍更高但 StreamingScore 不一定领先（因更高的延迟和资源消耗），揭示了 accuracy 和 deployability 之间的系统性 trade-off。
