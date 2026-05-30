## StreamingEval__A_Unified_Evaluation_Framework_for_Streaming_Video_Understanding

- baseline方法是什么？
  现有流式视频理解的评估方法存在三个核心缺陷：(1) **评估设置不统一** —— 部分基准（如 VStream-QA, StreamingBench, OVO-Bench）使用 pseudo-streaming 设置：视频在 query 时间戳处截断但仍以离线方式处理，未模拟真实的增量帧到达和因果约束；(2) **指标单一** —— 仅关注 answer accuracy，完全忽略延迟（TTFT）、吞吐（MaxFPS）、资源消耗（memory usage）等部署关键指标；(3) **比较不公平** —— 在线模型和离线模型使用不同的评估协议（离线模型可访问全视频，在线模型受因果约束），且不同模型的 visual token 维度不同导致相同 token 数的实际内存占用不一致。

  Baseline 全栈执行例子（以典型 "pseudo-streaming" 评估 + offline VideoLLM 为例）：
  - **算法层**：加载预训练 VideoLLM（如 Qwen3-VL-8B），在评估时加载完整视频的所有帧到 GPU → Vision Encoder 一次性编码所有帧 → visual tokens + text tokens 拼接 → LLM 自回归解码。对于时间戳 query，仅使用 query timestamp 之前的帧子集，但仍以批量 offline 方式处理（可同时访问 future frames 做 context）。
  - **Serving/系统层**：标准 HuggingFace Transformers 推理 pipeline（model.generate），无帧级增量处理，无 memory budget 约束，KV cache 可无限增长直到 OOM。
  - **评估层**：仅计算 accuracy（如 OVO-Bench 的 Real-Time/Backward/Forward 三类任务平均分），不测量编码延迟、解码延迟、内存占用。
  - **kernel/硬件层**：FlashAttention-2 + Accelerate 加速，单卡 RTX 4090 BF16 推理。论文未明确说明 baseline 评估框架的细节。

  Baseline 的缺陷：
  1. Pseudo-streaming 无法反映真实部署条件：模型实际可一次性加载全视频做 context，与真实在线场景下帧逐帧到达、仅能访问过去的约束不一致。
  2. 仅用 accuracy 评估误导：离线模型 accuracy 高但实际部署时可能因 encoding 太慢（<1fps）无法跟上视频流，或 decoding 延迟过大破坏交互体验。例如 VideoChatOnline-4B 在 OVO-Bench accuracy 仅为 40.40 但 MaxFPS 仅 0.14（远低于 1fps 要求），声称 "在线" 但无法实际部署。
  3. Token 数预算不公平：不同模型 visual token embedding 维度不同，相同 "256 visual tokens" 内存占用差异可达 2-3×。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 StreamingEval，一套统一的流式视频理解评估框架。以评估 Qwen3-VL-8B 在 OVO-Bench 上为例，全栈执行例子：

  - **算法层**：加载预训练 VideoLLM（Qwen3-VL-8B），不做任何模型修改。离线模型通过 bounded-memory adapter 接入：视觉 encoder 逐帧编码 → MLP projector 映射到 LLM embedding 空间 → 写入固定容量 memory bank（FIFO 淘汰）。在线模型（如 Flash-VStream）保留原生 streaming mechanism（增量编码 + 记忆/状态更新 + retrieval policy）。

  - **Serving/系统层（核心创新）**：StreamingEval 实现三进程异步 pipeline：
    1. **Frame Player**：以 1fps 固定帧率提取和发送视频帧到下个进程。
    2. **Encoder & Memory Updater**：接收帧 → vision encoder 编码（$z_i = g_\theta(v_i)$）→ 按照模型特定更新规则更新 memory state $M_{\tau_i^+} = \mathcal{U}(M_{\tau_i^-}, z_i; B, \pi)$，其中 B 是字节级 memory budget，π 是淘汰策略（离线模型用 FIFO，在线模型用原生策略）。
    3. **Responder**：用户 query 到达时，编码 query，读取当前 memory snapshot $M_{t1}$，条件于对话历史 $C_{t1}$ 和 query $q_{t0}$ 做自回归生成 $R_{t1} \sim p_\phi(\cdot | q_{t0}, C_{t1}, M_{t1})$。
    三个进程通过 inter-process queues 通信，无额外同步阻塞。

  - **评估指标层**：四维指标 + 综合 StreamingScore：
    - MaxFPS：模型可维持的最大输入帧率（编码吞吐上限）
    - TTFT (Time-to-First-Token)：从 query 到达到首个 token 生成的 wall-clock 时间
    - Memory_bank：在线可用历史视觉缓存的字节预算
    - Accuracy：流式在线 QA 的正确率
    - StreamingScore = (MaxFPS^{w_f} × Acc^{w_a}) / (TTFT^{w_t} × M^{w_r})，其中 M = Mem × ln(Params)，权重可按部署偏好调节（Best Answer/Interaction First/Resource-Saving/Throughput First）

  - **字节级统一资源预算**：不按 visual token 数量约束，而是：
    $\text{Mem}_i(B) = B \cdot d_i \cdot s_{\text{emb}} + B \cdot 2L_i \cdot h_i^{\text{kv}} \cdot s_{\text{kv}}$
    计算 visual token embedding + 关联 KV cache 的总字节数，反推出模型特定的 visual token 上限 $B_i = \lfloor M_{\text{bytes}} / (d_i s_{\text{emb}} + 2L_i h_i^{\text{kv}} s_{\text{kv}}) \rfloor$。

  - **kernel/硬件层**：单卡 RTX 4090 48GB (BF16)，FlashAttention-2 + Accelerate 加速。三进程 pipeline 的 inter-process 通信开销可忽略。

  解决 Baseline 缺陷的对应关系：
  1. **真实因果约束** → 三进程异步 pipeline：Frame Player 按固定帧率发送 → Encoder 逐帧增量编码 → Responder 仅能访问 t1 时刻的 memory snapshot（不含未来帧）。严格保证了 streaming 评估的真实性，消除了 pseudo-streaming 设置的失真。结果：VideoChatOnline 的 MaxFPS 仅 0.14（远低于 1fps），揭示了声称 "在线" 的模型实际无法部署。
  2. **多维部署导向评估** → 四维指标 + StreamingScore：超越 accuracy 单一指标，同时量化延迟、吞吐、资源消耗。例如 Qwen3-VL 在 OVO-Bench accuracy 最高（58.00 vs StreamForest 55.57），但 StreamForest 的 StreamingScore 在特定权重下超过 Qwen3-VL（因更低延迟和内存），揭示了 accuracy 和 deployability 之间的系统性 trade-off。
  3. **统一资源预算** → 字节级 memory budget：将不同 embedding 维度的模型归一化到相同字节预算下比较，消除了 token 数预算的不公平。Memory_bank 从 0.1G→1.5G 的 sensitivity 实验表明 accuracy 在 1.0G 以上近乎饱和，为实际部署的资源分配提供了量化指导。
  4. **场景感知评估** → 可调权重的 StreamingScore：支持 Best Answer (w_a=0.4)、Interaction First (w_t=0.4)、Edge Resource-Saving (w_r=0.4)、Throughput First (w_f=0.4) 四种部署偏好，不同场景下模型排名可互换（如 Qwen3-VL 在 Best Answer 排第 1 但 Flash-VStream 在 Interaction First/Resource-Saving/Throughput First 均排第 1），但整体趋势统计稳健（Spearman ρ ∈ [0.972, 0.993]）。

> 注意：本论文属于评估框架/基准测试论文，非新的模型算法或系统实现。核心贡献是标准化评估协议，使不同模型在统一流式约束下可公平比较，并量化 accuracy-latency-throughput-resource 的多维 trade-off。
