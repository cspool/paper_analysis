## ProMoE: Fast MoE-based LLM Serving using Proactive Caching

- baseline方法是什么？
  Baseline 方法是 **reactive caching with LRU/static cache + expert offloading**，即通过 LRU 或 static policy 将频繁访问的 experts 缓存于 GPU memory，未命中时从 CPU memory 被动按需加载（reactive cache miss）。Baseline 的两种变体：(a) Transformers Offloading (TO)——仅 expert 参数 offload 到 CPU，inference 时按需 cudaMemcpy 加载；(b) Llama.cpp Offloading (LO)——同时 offload expert 参数和计算到 CPU。
  全栈执行例子（Baseline: LRU cache + transformers, DS-1 FP16, RTX 4090, 50% cache rate, single token decode）：
  ```
  # 算法层：传统 MoE decoder layer
  token → embedding → for layer 1..28:
      # 系统框架层：HuggingFace transformers
      self_attention(x) → RMSNorm → gate: softmax(W_gate@x) → TopK=6 experts
      # Kernel调度层：cuBLAS GEMM for each expert FFN
      for each selected expert:
          if expert NOT in GPU cache:        # reactive cache miss!
              cudaMemcpy(CPU→GPU, expert_weight)  # BLOCKS critical path
          FFN_expert(x)  # SwiGLU: gate_proj→SiLU→×up_proj→down_proj
      weighted_sum(outputs)
  # 硬件架构层：RTX 4090 + PCIe 4.0 32GB/s + Intel i9-14900K
  ```
  核心缺陷：(1) **Reactive cache miss on critical path**——cache miss 时 expert 加载与 GPU 计算串行，DS-1 50% cache 时 decode 阶段 60.4% 时间用于等待 expert 加载，prefill 阶段达 82.7%；llama.cpp 更严重（prefill 94.2%, decode 79.0%）因推理速度更快使等待时间占比更高；(2) **Modern decoder-only MoE 的 uniform access pattern**——现代 MoE（DS/QW/Mixt）通过 Device-Limited Routing 和 Expert-Level Balance Loss 训练避免 routing collapse，导致 expert 访问分布均匀（low skewness），LRU cache hit rate 受限（不同于早期 encoder-decoder MoE 如 Switch Transformer 的 power-law 分布）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 ProMoE，通过 **proactive caching** 将 expert 加载从被动反应式（reactive）变为主动预测式（proactive），核心设计：(1) Learned Predictor——二层 MLP 学习 hidden state → expert selection 映射，accuracy 84.7%（vs token-based 58.3%, skip-based 66.9%）；(2) Stride Prefetching——预测与 prefetch pipeline 并行，将预测延迟隐藏，最大化 CPU-GPU 带宽利用率；(3) Chunked Prefetching + Early Preemption + Reordered Inference——三级协调机制消除被动 cache miss，最大化 prefetch 与 inference 重叠。
  全栈执行例子（ProMoE + transformers, DS-1 FP16, RTX 4090, 50% cache rate, single token decode）：
  ```
  # 算法层：ProMoE 增强的 MoE decoder layer
  token → embedding → for layer 1..28:
      # === 以下 3 步与 GPU self_attention 并行 ===
      # 算法层：Learned Predictor (CPU, ~200μs)
      h_prev_cpu = clone_to_cpu(x)              # 前一层 hidden state
      pred_logits = MLP_predictor[l](h_prev_cpu) # 2-layer MLP, ~2M params
      pred_experts = TopK(softmax(pred_logits), k=6)
      
      # 系统框架层：Prefetcher PushPredictedExperts (LOW priority)
      for e in pred_experts:
          if e not in GPU cache:
              queue.push(Task(layer=l, expert=e, chunk=0..2, pri=LOW))
      
      # 系统框架层：GPU self_attention
      x = self_attention(x)
      
      # Kernel调度层：gate function → hook
      gate_logits = W_gate @ x
      precise_experts = TopK(softmax(gate_logits), k=6)
      
      # 系统框架层：Early Preemption + Reordered Inference
      queue.remove_low_pri_tasks(layer=l)       # clear LOW tasks for this layer
      reordered = sort_by_cache_status(precise_experts) # cached → prefetching → none
      for e in reordered:
          if e not fully prefetched:
              queue.push(Task(layer=l, expert=e, chunk=0..2, pri=HIGH))
      
      # 硬件架构层：GPU computation ←→ PCIe prefetching pipeline
      for e in reordered:
          wait_until_chunks_ready(e)            # wait for prefetch completion
          output += gate_weight[e] * FFN_e(x)   # SwiGLU FFN on GPU
      # Prefetcher worker thread (CPU) concurrently:
      #   while True: task = queue.pop() → cudaMemcpyAsync(CPU→GPU, chunk)
  ```
  解决 Baseline 缺陷的方式：
  1. **针对"reactive cache miss on critical path"**：Proactive caching 通过 predictor 提前预测 + prefetcher 异步传输，将 expert 加载从关键路径移除。Chunked prefetching（3 chunks per expert）使高优先级任务等待延迟≤1 chunk。Early preemption 将 cache miss 检测提前到 gate 完成时刻（而非 expert 访问时刻）。Reordered inference 让已缓存 experts 先执行，同时异步 prefetch 缺失 experts。最终将关键路径加载时间从 69.68% 降至 30.96%（QW-2, 50% cache）。
  2. **针对"uniform access pattern 限制 cache hit rate"**：ProMoE 不依赖 expert access skewness——predictor 直接从 hidden state 预测 expert 选择，而非从历史访问频率推测。learned predictor 在 uniform access pattern 下仍维持 84.7% accuracy。stride prefetching 确保即使预测有 15.3% 误差，FetchRate 仍高（因预测与传输 pipeline 并行）。
  3. **整体效果**：vs offloading baselines: prefill 平均 2.20× (up to 3.21×), decode 平均 2.07× (up to 5.02×)。vs hand-crafted caching (LRU/static): prefill 1.78×, decode 1.34×。
