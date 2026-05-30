## LiveStar__Live_Streaming_Assistant_for_Real-World_Online_Video_Understanding

- 属于Serving调度的实现是什么？实验比较什么？
  实现：LiveStar 的 Serving 调度层创新体现在三个方面：(1) **SVeD (Streaming Verification Decoding)** 响应-沉默解码框架 —— 在流式视频推理中，每个 incoming frame 通过单次 forward pass 验证当前字幕的 perplexity，当 PPL 变化超过阈值 α（默认 1.03）时触发解码生成新字幕，否则保持沉默。相比 EOS-based 方法（VideoLLM-online 等需要完整解码输出 EOS token 来标记沉默），SVeD 避免了对每个帧都进行完整解码，大幅降低流式推理延迟。(2) **Streaming KV Cache（双级缓存架构）** —— 包含 intra-dialogue KV cache（帧级别处理缓存）和 inter-dialogue streaming cache（跨对话长上下文保留），消除新视频帧处理时的历史上下文重复计算。支持 SVeD 中 swap 操作后的 cache 序列完整性维护，以及 Peak-End Memory Compression 后的动态长度适配。实现 1.53× 推理加速（5 分钟视频）。(3) **Peak-End Memory Compression** —— 对 10+ 分钟视频（3 fps）进行在线记忆管理，通过概率剪枝低重要性旧帧（基于 PPL 和时间衰减），将推理上下文窗口维持在可控范围内，支持长时间流式服务不 OOM。

  实验比较：
  OmniStar-RNG 在线评估中，对比 VideoLLM-online（EOS-based streaming）、VideoLLM-MoD（MoD token selection）、MMDuet（classification head），LiveStar 在 SemCor（3.19 vs 1.68/1.66/1.63）、TimDiff（1.91 vs 2.67/2.54/2.32）、FPS（3.82 vs 3.37/3.41/0.91）上全面领先。Ablation: KV Cache 双级策略（Both vs w/o Inter-Dialog vs Neither）FPS 从 2.50→2.92→3.82；Memory 压缩策略对比 Uniform/FIFO/Peak-End。

- 硬件平台是什么，配置是什么。
  训练：8× NVIDIA A800 GPU。推理：论文使用相同 A800 GPU 平台进行在线流式推理。FPS 测试在 5 分钟视频（3 fps）下进行。

- 开源Serving框架是什么。修改了什么。
  LiveStar 基于 PyTorch + HuggingFace Transformers 构建，未使用现有开源 Serving 框架（如 vLLM、SGLang），而是自建流式推理逻辑。核心修改包括：
  (1) 自定义 streaming decoding loop：替代标准 autoregressive generate()，实现 frame-by-frame 流式输入处理 + SVeD 响应-沉默决策。
  (2) 双级 KV Cache 管理：修改标准 KV cache 以支持 (a) intra-dialogue 帧级 cache 复用、(b) inter-dialogue 跨对话 cache 持久化、(c) SVeD swap 操作后的 cache 序列一致性维护、(d) Peak-End 剪枝后的动态 cache 收缩。
  (3) 流式视频输入 pipeline：实现 1-4 FPS 可配帧率下的持续帧提取-编码-推理 pipeline。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源地址：https://github.com/yzy-bupt/LiveStar。

  框架输入到硬件执行全过程（以 OmniStar-RNG 实时叙述生成为例）：

  ```
  # ===== Serving 输入阶段 =====
  # 视频上传 → 流式帧提取 (1-4 FPS 可配)
  stream = VideoStream(video_path, fps=3)
  
  # ===== 逐帧处理循环（核心 Serving 循环）=====
  Ctx, Dec, PPL_cache = [], None, {}
  t_i = 0  # 上次解码时间戳
  
  for timestamp, raw_frame in stream:
      # 1. Vision Encoding (InternViT, 冻结)
      frame_embed = vision_encoder(raw_frame)  # → [16, D_llm]
      frame_tokens = mlp_projector(frame_embed)  # → [16, D_llm]
      Ctx.append(frame_tokens)
      
      # 2. SVeD 响应-沉默决策
      if Dec is not None:
          # 单次 forward pass: 计算 perplexity
          # 复用 inter-dialogue KV cache 避免重算历史
          PPL_new = model.forward_perplexity(Dec, Ctx, 
                      use_kv_cache="inter_dialogue")
          if PPL_new > 1.03 * PPL_cache[t_i]:
              # 语义变化 → 触发解码
              new_tokens = model.generate(Ctx)
              Dec = tokenizer.decode(new_tokens)
              Ctx.append(tokenizer.encode(Dec))
              t_i = timestamp
              PPL_cache[timestamp] = model.forward_perplexity(
                  Dec, Ctx)
          else:
              # 沉默: 将 Dec 移到 Ctx 末尾
              swap_last_two(Ctx)  
      else:
          Dec = model.generate(Ctx)
          Ctx.append(tokenizer.encode(Dec))
          t_i = timestamp
          PPL_cache[timestamp] = model.forward_perplexity(Dec, Ctx)
      
      # 3. Peak-End Memory Compression（每 W=40 帧触发）
      if len(Ctx) > window_size * 16:  # 16 tokens/frame
          # 按 PPL 和 elapsed time 概率剪枝旧帧
          Ctx = peak_end_prune(Ctx, PPL_cache, window=40)
          # 同步更新 KV cache
          kv_cache = prune_kv_cache(kv_cache, Ctx)
  
  # ===== 硬件执行 =====
  # - InternViT forward: GPU (A800) 处理 448×448 帧 → 16 tokens
  # - SVeD verification: 单次 forward pass (无 token generation)
  # - Decoding: 仅在语义变化时触发，减少 GPU compute
  # - KV Cache: GPU HBM 存储，双级管理避免重算
  # - FPS: 3.82 (5分钟视频, 对比 baseline 2.50-3.41)
  ```

  Serving 调度核心优势：
  - **按需解码**：SVeD 仅在 perplexity 变化显著时触发解码（而非每帧），减少约 90%+ token generation 次数（在 1 分钟视频 @3fps 中，若仅 5 个语义变化段，则 180 帧中仅需 5 次解码）
  - **KV Cache 复用**：双级 cache 使历史帧的 K/V 无需重算，在 5 分钟视频上实现 1.53× 加速
  - **内存管控**：Peak-End 压缩使 KV cache 大小可控，支持 10+ 分钟视频持续流式推理

## LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos

- 属于Serving调度的实现是什么？实验比较什么？
  实现：MM-SP（Multi-Modal Sequence Parallelism）推理模式。在HuggingFace Transformers基础上实现分布式序列并行推理，替代原有的Pipeline Parallelism推理方式。核心创新：(1) 所有GPU并发参与计算（vs HF Pipeline逐层串行，仅1 GPU同时活跃），加速比与GPU数量成正比；(2) 内存均匀分布到所有设备（vs HF Pipeline首卡存储全部输入embedding和图像导致内存瓶颈）；(3) 推理模式下管理动态变化的tensor（input tokens和position encodings逐步变化），检测持有last token的机器信号来正确终止分布式进程；(4) 两阶段sharding策略：Stage1按图像数均衡分布帧用于视觉编码，Stage2按token数均衡切分用于LLM解码。

  实验比较：
  (a) 推理延迟：MM-SP推理 vs HuggingFace Pipeline Parallelism推理，单节点8×H100 GPU，8B模型。MM-SP实现8.2×加速（所有GPU并发 vs Pipeline仅1 GPU活跃）。
  (b) 最大支持序列长度：MM-SP支持2.9×更长的序列（96K序列下HF Pipeline首卡存80GB activations而其余卡仅18GB导致OOM，MM-SP均匀分布）。
  (c) 训练吞吐量：vs ZigZag-RingAttn（2.1×-5.7×加速）、vs Megatron-LM CP（3.1×-4.3×加速）、vs Megatron-LM CP+TP hybrid（1.1×-1.4×加速）、vs DeepSpeed-Ulysses（持平），在32 H100 GPU上。
  (d) 最大训练序列长度：MM-SP 2D-Attention支持2M+ tokens on 256 GPUs（vs Ulysses受限于attention heads数量32，约8×少；vs Megatron-LM支持显著更短的序列）。
  (e) 64 H100 GPU扩展性 (Table 8)：578K序列2D-Attention 16.9s/iter vs ZigZag 77.2s/iter。
  (f) FSDP vs Zero-3内存效率 (Table 7)：FSDP在256K序列2D-Attention 7.04s/iter vs Zero-3 OOM，证明FSDP更高效。
  (g) 两阶段sharding ablation (Table 5)：long captioning任务上1%-7%加速（8 GPU: 1.12s vs 1.20s/iter）。
  (h) Communication overlap副作用 (Table 2)：Ring-style SP的通信-计算重叠设计占据SM资源，导致attention kernel变慢（forward +4.2%-18.6%, backward +0.5%-5.8%）。

- 硬件平台是什么，配置是什么。
  H100节点：每节点8×H100 80GB，NVLink 900 GB/s (intra-node)，InfiniBand 50 GB/s single path (inter-node)，intra/inter带宽差异18×。最大序列长度实验：32×A100节点（256 GPUs，每节点8×A100 80GB）。推理实验：单节点8×H100 80GB。

- 开源Serving框架是什么。修改了什么。
  开源Serving框架：HuggingFace Transformers。通过monkey-patching方式集成MM-SP，无需修改Transformers核心代码。
  修改内容：(1) 替换HF Pipeline Parallelism推理为MM-SP Sequence Parallelism推理，所有GPU并发计算而非逐层串行；(2) 实现两阶段sharding策略：视觉编码阶段按帧数均衡分配，LLM解码阶段按token数均衡切分（含dummy token padding确保均匀可分）；(3) 实现2D-Attention通信模式：构建N_head × N_ring通信mesh（如8 GPU=4×2），intra-node用All-to-All按head dim重分布QKV，inter-node用P2P传输KV chunks；(4) 推理模式下动态管理位置编码和输入token，检测终止信号；(5) 集成Flash-Attention2作为注意力后端，Triton实现自定义kernel。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源：github.com/NVlabs/VILA/tree/main/longvila
  
  Serving框架使用与执行全过程（以8 GPU推理256 frames视频为例）：
  ```
  输入：长视频(256 frames) + 用户prompt文本
  ↓
  [HuggingFace Transformers - monkey-patched MM-SP推理]
  1. Tokenization: 文本→text_tokens；frames→<img> placeholder tokens
  2. MM-SP Stage1 Sharding (视觉编码):
     构建SP通信组: 8 GPUs, 4×2 mesh (intra-node A2A=4, inter-node P2P=2)
     distribute_frames(256 frames, 8 ranks) → 每rank 32 frames
     各rank并行: vis_feats = vision_encoder(32 local_frames)  [balanced load]
  3. MM-SP Stage2 Sharding (LLM推理):
     all_gather(vis_feats) → 全局视觉特征汇总
     concat(vis_feats, text_tokens) → 完整多模态序列
     shard_by_token_count(full_seq, 8 ranks) → 每rank seq_len/8 tokens [balanced]
     pad with dummy tokens → 均匀可分
  4. LLM Decoder with 2D-Attention (逐层):
     for each transformer layer:
       Q,K,V = project(local_tokens)               [本地Linear计算]
       A2A(Q,K,V) within node (4 GPUs): 
         all_to_all scatter by head_dim → 重分布QKV按attention head
         # 利用NVLink 900 GB/s高带宽
       P2P(K,V) across nodes (2 groups):
         send_recv KV_chunks to next ring neighbor   [InfiniBand 50 GB/s]
         # 仅传输KV block，不传Q
       local_attn = FlashAttention2(Q_local, K_all, V_all, causal_mask)
       A2A(attn_output) reverse → 恢复原始head分布
       FFN(local_tokens)                  [本地计算，无通信]
  5. Decoding循环:
     每步生成1 token，位置编码递增
     持有last token的rank broadcast token给所有rank
     所有rank更新KV cache
     检测EOS或max_len → all_reduce终止信号
  6. 输出: 长视频描述/问答文本
  ↓
  硬件执行映射:
  - 视觉编码器: 每GPU各自计算，负载均衡（32 frames/GPU × 256 tokens/frame）
  - A2A通信: NVLink 900 GB/s, 4 GPU full mesh
  - P2P通信: InfiniBand 50 GB/s, ring topology
  - Attention计算: FlashAttention2 on Tensor Cores
  - FFN计算: cuBLAS GEMM on Tensor Cores
  ```
  
  关键作用：相比HF Pipeline（单GPU活跃，首卡内存瓶颈），MM-SP推理实现：(1) 8 GPU并发→8.2×加速；(2) 内存均匀分布→2.9×更长序列；(3) 可线性扩展至更多GPU。

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
