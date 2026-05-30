## StreamingEval__A_Unified_Evaluation_Framework_for_Streaming_Video_Understanding

- 属于Serving调度的实现是什么？实验比较什么？
  实现：StreamingEval 是一套统一的流式视频理解评估框架，其 Serving 调度层创新体现在：(1) **三进程异步流式 pipeline** —— Frame Player（按固定帧率持续发送原始帧）、Encoder-and-Memory Updater（逐帧编码并更新视觉记忆）、Responder（接收 query 后加载当前记忆快照并调用 LLM 生成回答），三者通过 inter-process queue 通信，模拟真实在线系统中视频帧连续到达、模型持续更新、用户 query 随时发生的场景。(2) **固定容量 Memory Bank + FIFO 淘汰策略** —— 为离线模型引入统一的有界记忆适配器（bounded-memory adapter），将视觉表示通过投影层对齐到 LLM embedding 空间后写入固定容量 memory bank，超出预算时按 FIFO 淘汰最旧内容，使离线模型可在严格在线约束下运行。(3) **字节级统一资源预算** —— 不按 visual token 数量约束，而是按字节计算存储成本（visual token embedding + 关联的 KV cache），将不同模型的 memory budget 标准化为统一的 byte-level cap，避免 embedding 维度不同导致的不公平比较。(4) **多维 StreamingScore 综合指标** —— 将 MaxFPS、Accuracy、TTFT、Memory 整合为单一可调权重的 StreamingScore，支持不同部署场景（最佳答案/交互优先/资源节省/吞吐优先）的偏好评估。

  实验比较：
  评估 12 个代表性模型（6 个离线 VideoLLM: Qwen3-VL-8B, InternVL3.5-8B, Llava-OV1.5-8B, MiniCPM-V4.5-8B, VideoLLaMA3-7B, VideoChat-7B；6 个在线 VideoLLM: Flash-VStream-7B, Flash-VStream-7B*, ReKV-7B, StreamForest-7B, TimeChat-Online-7B, VideoChatOnline-4B）。在 OVO-Bench（12 任务/2800 标注）和 StreamingBench（18 任务/900 视频/4500 QA）上，输出 per-task accuracy、MaxFPS、TTFT、Memory 和 StreamingScore。关键发现：离线模型 accuracy 普遍高于在线模型，但在线模型 StreamingScore 更高（更低延迟/更稳定在线节奏）；Memory_bank 从 0.1G→1.5G 时 accuracy 接近饱和；分辨率从 224→448 时 accuracy 提升但 computation cost 增加。

- 硬件平台是什么，配置是什么。
  单卡 NVIDIA RTX 4090 48GB（VRAM 48GB, 带宽 1008.10 GB/s, 峰值 40.32 TFLOPS）。CPU: Intel Xeon Platinum 8570（20 cores, 48GB RAM）。PCIe 16 lanes, 带宽 31.50 GB/s。所有实验在 BF16 精度下运行，使用 FlashAttention-2 + Accelerate 加速，流式输入统一 1fps。

- 开源Serving框架是什么。修改了什么。
  论文未使用现有开源 Serving 框架（如 vLLM、SGLang），而是自建 StreamingEval 流式评估 pipeline（基于 PyTorch + HuggingFace Transformers）。核心修改/构建包括：
  (1) 自定义三进程异步 pipeline：Frame Player → Encoder-Memory Updater → Responder，通过 inter-process queues 通信，无额外同步阻塞。
  (2) 自定义 memory bank 管理：固定容量 FIFO 淘汰策略 + 字节级 budget 计算（公式：$B_i = \lfloor M_{\text{bytes}} / (d_i s_{\text{emb}} + 2L_i h_i^{\text{kv}} s_{\text{kv}}) \rfloor$），统一不同模型的 visual token 预算上限。
  (3) 自定义评估指标计算：MaxFPS（编码吞吐上限）、TTFT（首个 token 延迟）、Memory_bank（历史上下文预算）、StreamingScore（可调权重综合分）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源地址：https://github.com/wwgTang-111/StreamingEval

  StreamingEval 框架输入到硬件执行全过程（以评估 Qwen3-VL-8B 在 OVO-Bench 上为例）：

  ```
  # ===== 初始化阶段 =====
  # 1. 加载模型 & 配置 memory budget
  model = load_model("Qwen3-VL-8B")  # BF16 on RTX 4090
  M_bytes = 0.5 * 1e9  # 0.5GB memory budget
  B_i = floor(M_bytes / (d_i * s_emb + 2 * L_i * h_kv * s_kv))
  # 计算模型特定的 visual token 上限
  memory_bank = FixedCapacityFIFO(max_tokens=B_i)
  
  # ===== 三进程异步 Pipeline =====
  # Process 1: Frame Player（固定帧率 1fps）
  def frame_player(video_path, fps=1):
      for frame in extract_frames(video_path, fps):
          frame_queue.put((frame, timestamp))
          time.sleep(1/fps)
      frame_queue.put(None)  # EOF signal
  
  # Process 2: Encoder & Memory Updater
  def encoder_memory_updater():
      while True:
          item = frame_queue.get()
          if item is None: break
          frame, timestamp = item
          # 视觉编码
          z_i = vision_encoder(frame)  # → visual tokens
          z_proj = projector(z_i)      # → LLM embedding space
          # 更新 memory bank (FIFO)
          memory_bank.append(z_proj)
          while memory_bank.byte_size > M_bytes:
              memory_bank.evict_oldest()  # FIFO 淘汰
          memory_snapshot_queue.put(
              (timestamp, memory_bank.snapshot()))
  
  # Process 3: Responder（query 触发式）
  def responder(query, query_timestamp):
      # 编码 query
      q_embeds = tokenizer(query)
      t1 = now()  # query 编码完成时刻
      # 读取 t1 时刻的 memory snapshot
      M_t1 = memory_snapshot_queue.get_latest(t1)
      # 拼接输入: [visual_tokens, text_tokens, query]
      input_ids = concat(M_t1.tokens, q_embeds)
      # LLM 自回归生成
      ttft_start = now()
      first_token = model.generate(input_ids, max_tokens=1)
      ttft = now() - ttft_start
      # 继续生成直到 EOS
      answer = model.generate(input_ids, 
          past_key_values=first_token.kv_cache)
      return answer, ttft
  
  # ===== 硬件执行映射 =====
  # GPU (RTX 4090 48GB):
  #   - Vision Encoder forward: GPU Tensor Cores, BF16
  #   - LLM Decoder: FlashAttention-2 on GPU Tensor Cores
  #   - KV Cache: GPU VRAM 存储, PagedAttention 管理
  #   - Memory Bank: GPU VRAM 中 visual token embeddings
  # CPU (Intel Xeon Platinum 8570):
  #   - Frame extraction & preprocessing
  #   - Inter-process queue management
  #   - Tokenizer 处理
  # PCIe 16× (31.50 GB/s):
  #   - CPU→GPU: frame data transfer (预处理后帧)
  #   - GPU→CPU: 极少（仅 final answer text）
  ```

  关键作用：StreamingEval 框架将流式视频理解评估标准化为三进程异步 pipeline，实现：(1) 严格因果约束（仅访问已处理的过去帧），(2) 统一 memory budget 下公平比较不同模型，(3) 多维指标（MaxFPS/TTFT/Memory/Accuracy/StreamingScore）量化 accuracy-latency-throughput-resource 权衡。

> 注意：本论文属于评估框架/基准测试论文，主要贡献是评估协议而非模型或系统实现。最接近的层次是 Serving调度（流式pipeline + 资源约束 + 延迟/吞吐评估），次要接近算法pipeline（评估模型算法在流式条件下的表现）。
