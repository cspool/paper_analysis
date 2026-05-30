## Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **Flood**，一个高效的离线推理框架。核心设计：(1) **全流水线并行 (PP) 架构**：放弃主流框架（如 vLLM）广泛使用的 Tensor Parallelism (TP)，因为它依赖高带宽互联（如 NVLINK），在没有 NVLINK 的低规格设备上通信开销可能占总执行时间一半以上。Flood 纯用 PP 在节点间和节点内实现，无需张量切分。(2) **多对一进程映射**：在单加速器上部署多个进程，每个进程绑定独立 CUDA stream，实现多进程共享单加速器，零 CPU 开销。(3) **Segment Cache**：替代 vLLM 的 block-based KV cache（PageAttention），在连续内存空间中分配 KV cache 为 `[max_token_num, num_head, head_dim]` 形状的大 block，避免小 block 导致的计算资源低效利用。支持超长输出时的 extend/append/wait 策略，以及原生 prefix caching。(4) 单机 8 加速器部署 9 个进程，确保始终有一个进程在等待 pipeline 第一阶段加速器可用，消除空闲时间。

  实验比较：Flood vs vLLM (v0.6.6.post2) 在 shareGPT benchmark 上的吞吐量（tokens/s）对比——Ling-Lite 在 1×Device E 上 5869 vs 4355 (1.35×)；Ling-Lite 在 1×Device C 上 5451 vs 3576 (1.52×)；Ling-Plus 在 16×Device B 上 4857 vs 2331 (2.08×)；Ling-Plus(FP8) 在 8×Device E 上 6569 vs 2742 (2.40×)。

- 硬件平台是什么，配置是什么。
  五种异构 AI 加速器：Device A (370 TFLOPS, 64GB)、Device B (120 TFLOPS, 96GB)、Device C (312 TFLOPS, 80GB)、Device D (989 TFLOPS, 80GB)、Device E (147 TFLOPS, 96GB)。推理性能测试使用 1×、8×、16× 配置，缺乏 NVLINK 等高速互连。

- 开源Serving框架是什么。修改了什么。
  开源框架：**Flood**（[https://github.com/alipay/PainlessInferenceAcceleration](https://github.com/alipay/PainlessInferenceAcceleration)），是全新设计的离线推理框架。对比如 vLLM，Flood 的修改/创新：(a) 完全放弃 TP 转而采用全 PP 架构；(b) 用 Segment Cache 替代 PageAttention/block table；(c) 多对一进程-加速器映射替代一对一映射；(d) 多 stream 隔离替代单 stream。Flood 未修改 vLLM，而是独立设计。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源：[https://github.com/alipay/PainlessInferenceAcceleration](https://github.com/alipay/PainlessInferenceAcceleration)。Flood 全 PP 离线推理全流程：

  ```
  === 系统初始化 ===
  单机 8 加速器场景:
    启动 9 个进程（>加速器数量）
    Pipeline Stage 分配: DP_0→Acc0, DP_1→Acc1, ..., DP_7→Acc7
    每个进程绑定独立的 CUDA stream
    第 9 个进程为备选，等待任意 stage 空闲

  === Segment Cache 初始化 ===
  对每个请求预分配连续 KV cache:
    cache_tensor = [max_token_num, num_head, head_dim]   # 连续内存
    # vs vLLM: [num_blocks, block_size, num_head, head_dim]  # 分散小 block

  === 单请求推理流程 ===
  Input: prompt_tokens (请求文本)

  Step 1 - Prompt 处理:
    # 分配连续 segment 给 prompt
    segment = allocate_contiguous_memory(prompt_len + max_output_len)
    # TP 模式需切分 tensor 到多卡; PP 模式无此开销
    for stage in pipeline_stages:  # 各加速器顺序执行
      hidden = model_layer(hidden)
      # PP 天然支持 batch 内 prefix: 共享前缀仅存一份

  Step 2 - Token 生成循环:
    while not done:
      hidden = model_layer(hidden)
      logits = lm_head(hidden)
      next_token = sample(logits)

      # Segment Cache 写 KV: 连续地址直接索引
      kv_cache[seq_pos] = (key, value)

      # 若实际生成超出预分配 segment:
      if seq_pos > segment.max_len:
        if adjacent_free():         extend segment   # 策略1
        elif other_free():          append segment   # 策略2
        else:                       wait_list()      # 策略3

      if eos_token: break

  === 并发调度（多对一映射） ===
  加速器 0 上进程 A: 执行 pipeline stage 0 of batch_1 (stream A)
  加速器 0 上进程 B: 执行 pipeline stage 0 of batch_2 (stream B)
  # 两进程通过不同 stream 并发，GPU 资源持续利用
  # 无需跨进程 tensor 通信（vs TP 每层需 AllReduce）
  ```

  关键优势：PP 无需 NVLINK，在低规格/异构设备上 TP 通信开销 >50%，PP 零通信开销换取更高吞吐。Segment Cache 避免 small-block overhead，同时天然支持 prefix caching（共享前缀的多请求复用一个 segment）。
