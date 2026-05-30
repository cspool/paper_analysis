## Fast Inference of Mixture-of-Experts Language Models with Offloading

- 属于Serving调度的实现是什么？实验比较什么？
  - 论文构建了 MoE 专用的 expert offloading 调度系统，替代 naive offloading（HuggingFace accelerate 的 device_map="auto"），实现更高效的 batch size 1 交互式推理调度。核心调度策略包括：
    1. **Expert LRU Cache 调度**：每个 MoE 层维护 k 个最近使用 expert 的 GPU 缓存，避免每 token 都从 host RAM 重新加载。k=2（12GB GPU）或 k=4（16GB GPU）。
    2. **Speculative Expert Prefetching**：在当前层 expert 加载完成后立即启动投机预取——将下一层 MoE gate 应用于当前层 hidden states 预测下一层最可能使用的 expert，在后台异步加载。
    3. **内存分割调度**：当 host RAM 无法容纳完整模型时（如 Google Colab），expert 在 host RAM 和 GPU memory 之间按 LRU 策略动态换入换出，换出时回写到 host RAM。
    4. **异步多 buffer 架构**：分配 b=4 个共享 device buffer 用于异步拷贝和预取，所有 MoE 层复用，减小内存足迹。
  - 实验比较：
    - Full algorithm vs w/o expert pre-loading vs w/o LRU cache & pre-loading vs Naive offloading (accelerate)
    - 在 T4/RTX 3060/RTX 3080 Mobile/A100 上测 tokens/sec，batch size=1

- 硬件平台是什么，配置是什么。
  - T4 (Google Colab free-tier): 16GB VRAM, PCIe Gen.3, host-to-device 8-16GB/s
  - RTX 3080 Mobile (gaming laptop): 16GB VRAM, PCIe Gen.4
  - RTX 3060 (midrange desktop): 12GB VRAM, PCIe Gen.3
  - A100-80GB-SXM: 用于对比参考（可无 offloading 运行）
  - 约束：GPU VRAM 仅容纳 non-expert layers + k 个缓存 expert，全部 expert 参数需存储于 host RAM

- 开源Serving框架是什么。修改了什么。
  - **基线框架**：HuggingFace accelerate（naive offloading with device_map="auto"），按层整体加载/卸载到 GPU
  - **论文自建 offloading 系统**：不修改已有 serving 框架，而是基于 PyTorch 构建专用的 MoE offloading 调度器，替代 accelerate 的默认 offloading
  - 修改/新增内容：
    - **Per-expert offloading**（替代 per-layer offloading）：将每个 MoE 层的 8 个 expert 独立 offload，仅加载 top-2 所需的 expert 到 GPU
    - **LRU cache 管理**：在 GPU 侧维护 per-layer expert cache，跟踪使用顺序
    - **投机预取逻辑**：使用当前层 hidden states 推测下一层 expert 选择，异步启动 host-to-device 传输
    - **内存管理**：expert 参数连续 pinned memory 分配 + b=4 个共享 device buffer

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：代码开源在 https://github.com/dvmazur/mixtral-offloading
  - **Serving 框架执行全过程（以 Mixtral-8x7B-Instruct 在 T4 16GB + 2-bit experts 为例）**：

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. 模型加载阶段                                                  │
    │    - Attention/embedding/norm 层: 4-bit HQQ 量化, 常驻 GPU       │
    │    - Experts (45.1B params): 2-bit HQQ 量化, 常驻 pinned RAM    │
    │    - GPU 侧预分配: k×32_layers 个 expert slot + 4 个共享 buffer │
    │    - 初始化 LRU cache: 每层随机加载 k 个 expert 到 GPU           │
    │           ↓                                                     │
    │ 2. 用户输入 prompt tokens [T₁, T₂, ..., Tₙ]                      │
    │    Prompt 处理 (prefill): 逐层计算, 每层 expert 加载一次          │
    │    (与生成阶段的逐 token 加载不同, prefill 相对高效)               │
    │           ↓                                                     │
    │ 3. Token 生成循环 (autoregressive decode, batch=1)               │
    │    for each new token:                                          │
    │      for layer l in 0..31:                                      │
    │        ┌─ Attention block ──────────────────────────────────┐   │
    │        │  已常驻 GPU, 直接计算, 输出 h_attn                  │   │
    │        └────────────────────────────────────────────────────┘   │
    │        ┌─ MoE Gate ────────────────────────────────────────┐   │
    │        │  gate_scores = W_gate[l] @ h_attn  (常驻GPU)      │   │
    │        │  top2_experts = topk(gate_scores, 2)               │   │
    │        └────────────────────────────────────────────────────┘   │
    │        ┌─ Expert Loading (GPU cache check) ────────────────┐   │
    │        │  for e in top2_experts:                            │   │
    │        │    if e in GPU_cache[l]:                           │   │
    │        │      expert_weights = GPU_cache_buf[e]  // 命中    │   │
    │        │      mark_recent(e)                                │   │
    │        │    else:                                           │   │
    │        │      evict = LRU_evict(C_l)  // cache miss         │   │
    │        │      copy GPU_cache[evict] → host_pinned[evict]   │   │
    │        │      copy host_pinned[e] → GPU_cache_slot          │   │
    │        │      (单次 contiguous mem copy, PCIe 8-16GB/s)      │   │
    │        └────────────────────────────────────────────────────┘   │
    │        ┌─ Speculative Prefetch (后台异步) ─────────────────┐   │
    │        │  # 用当前 h_attn 预测下一层 expert                │   │
    │        │  pred_gate = W_gate[l+1] @ h_attn                  │   │
    │        │  pred_top1, pred_top2 = topk(pred_gate, 2)        │   │
    │        │  async_copy host_pinned[pred_top1] → shared_buf   │   │
    │        │  async_copy host_pinned[pred_top2] → shared_buf   │   │
    │        │  (在独立的 CUDA stream 上执行, 与当前层计算重叠)    │   │
    │        └────────────────────────────────────────────────────┘   │
    │        ┌─ Expert FFN Computation ──────────────────────────┐   │
    │        │  out = 0                                          │   │
    │        │  for e, w in zip(top2_experts, gate_weights):     │   │
    │        │    out += w * SiLU(W_gate_e @ h) * (W_up_e @ h)  │   │
    │        └────────────────────────────────────────────────────┘   │
    │      → 下一 token 生成                                         │
    │           ↓                                                     │
    │ 4. 输出: generated tokens, 2-4 tokens/sec                       │
    └─────────────────────────────────────────────────────────────────┘
    ```

    **关键性能数据（Table 2, 2-bit experts）**：
    | Hardware | Full algo | w/o pre-load | w/o cache & pre-load | Naive (accelerate) |
    |----------|-----------|-------------|---------------------|-------------------|
    | A100 | 3.06 tok/s | 2.92 tok/s | 2.27 tok/s | 1.39 tok/s |
    | 3080 Mobile | 2.66 tok/s | 2.23 tok/s | 1.76 tok/s | 1.06 tok/s |
    | RTX 3060 | 2.28 tok/s | 2.05 tok/s | 1.55 tok/s | 0.92 tok/s |
    | T4 (Colab) | 2.09 tok/s | 1.57 tok/s | 1.17 tok/s | 0.66 tok/s |

    从 Naive offloading 到 Full algorithm 加速约 2.2×（T4）到 3.2×（RTX 3060, 3-bit）。Pre-loading 在 RTX 3060 上效果最显著（因 k=2 的较小 LRU cache）。
