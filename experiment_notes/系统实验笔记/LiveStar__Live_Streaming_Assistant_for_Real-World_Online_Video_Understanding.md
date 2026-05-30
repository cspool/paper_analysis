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
