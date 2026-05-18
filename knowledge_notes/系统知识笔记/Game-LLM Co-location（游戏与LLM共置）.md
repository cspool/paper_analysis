## Game-LLM Co-location（游戏与LLM共置）

术语是什么？通过联网搜索让回答具体和精准。

Game-LLM Co-location 指在单张消费级游戏 GPU 上同时运行游戏渲染任务和 LLM 推理任务，使 LLM 能在本地为游戏提供 AI 增强功能（如 NPC 控制、动作生成、对话生成），而无需额外的 GPU 硬件或依赖云端 LLM 服务。该问题是 LEGO 论文的核心场景：现代游戏（如 BlackMyth、FFXVI、RDR2）在 60 FPS 下每帧有约 16.6ms deadline，但实际渲染仅占用 GPU 时间的 47.6%-60.8%，剩余为 compute headroom；同时 LLM 需按 APM（Actions Per Minute）频率生成动作（100 APM=每 600ms、200 APM=每 300ms、300 APM=每 200ms），需利用这些碎片化的 GPU 空闲时间执行推理。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

共置执行的核心挑战与 LEGO 的应对：

**挑战 1：资源总量不足**
- 以 BlackMyth + Llama3-8B 为例：游戏渲染需 60.8% GPU time，100 APM 下 LLM 推理需 41.9% GPU time → 总计 102.7% 超限
- 200 APM 和 300 APM 下资源缺口更大（18 个场景中 14 个超过 GPU compute limit）
- LEGO 应对：通过 layer-skipping adaptor 降低 LLM 推理的计算需求，使其适配可用 headroom

**挑战 2：动态、碎片化的 headroom**
- Rendering task 执行时间波动显著（BlackMyth CDF 显示 7-10.1ms 范围）
- Headroom 分散在帧间（inter-rendering）和帧内（intra-rendering，因 game engine 的 batch rendering 优化产生）
- LEGO 应对：headroom-maximizing scheduler 预测总 headroom，将 LLM 推理拆分为 layer/sublayer 粒度 subtask 以填充两类空隙

**挑战 3：严格的实时约束**
- 游戏渲染 deadline 16.6ms（60 FPS），优先级最高
- LLM 动作生成 deadline 600ms/300ms/200ms（对应 100/200/300 APM）
- LEGO 应对：T_subtasks ≤ T_minimal safety constraint + sudden spike handling

**挑战 4：云端方案不可行**
- 云端 LLM 服务端到端延迟 20-110ms（同区域）到 300ms（跨洲），加上 OpenAI API 额外 300-700ms
- 200 APM SLO 为 300ms，300 APM SLO 为 200ms → 云端延迟不可接受
- LEGO 应对：全部在本地单 GPU 上执行

共置执行流程（以 BlackMyth + Llama3-8B + 200 APM on RTX 4090 为例）：
1. 游戏以 60 FPS 渲染，每帧 16.6ms。GPU 执行 rendering subtasks 和 auxiliary subtasks（不用 GPU）
2. 每 300ms（200 APM），LLM action 请求到达：从 game engine 提取场景状态、角色状态、历史动作构造 prompt（~512 tokens）
3. LEGO scheduler 用 LR 模型预测接下来 300ms execution window 内的总 rendering headroom
4. 根据预测 headroom 选择跳层策略和 adaptor
5. 推理执行：prefill 以 self-attention/FFN sublayer 粒度、decode 以 Transformer layer 粒度，填充 intra-rendering 和 inter-rendering headroom
6. LLM 输出动作描述（~16 tokens）→ game engine 执行对应技能/移动

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
- **框架集成**：llama.cpp front-end 集成到 Unreal Engine 4 中，修改 traversal function 加入调度逻辑。其他功能通过 dynamic library 调用
- **调度粒度**：decode 阶段以 Transformer layer (~0.4ms) 为粒度；prefill 阶段以 self-attention (~0.5ms) 和 FFN sublayer (~1.0ms) 为粒度
- **Safety constraint**：T_subtasks ≤ T_minimal，其中 T_minimal 为游戏所有 rendering task 中最小的 inter-rendering headroom
- **兼容性**：LEGO 不依赖 RTX 4090 特殊硬件特性，可部署到其他 gaming GPU
- **云游戏集成**：可集成到 NVIDIA GeForce NOW，利用 PilotFish 的 time-division 机制在云端 GPU 上分配 LLM inference time slice
- **多 AI agent 支持**：当游戏需要多个 AI agent（如 Dota-like 最多 9 个），LLM 推理需 batch 执行，LEGO 支持 Llama3-3B + adaptor 在 batch=9 下维持 100/200 APM

涉及论文标题：
- LEGO: Supporting LLM-enhanced Games with One Gaming GPU
