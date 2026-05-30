## HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - HybriMoE 在 kTransformers 之上实现了一套面向资源受限环境的混合 CPU-GPU MoE 推理调度系统，核心调度优化包括：
    1. **Dynamic Intra-Layer Hybrid Scheduling（Section IV-B）**：引入三条优先级规则简化 expert-to-hardware 映射问题——GPU 优先计算已缓存的高负载 expert（降序），CPU 优先计算未缓存的低负载 expert（升序），PCIe 传输优先移动高负载未缓存 expert 到 GPU。将调度问题形式化为 `argmin max(CPU_TIME(cpu_expert), GPU_TIME(gpu_expert))` 的分配问题。执行前通过仿真阶段迭代填充 CPU/GPU/PCIe 时间线来评估调度策略，选择最小化延迟的配置。
    2. **Impact-Driven Inter-Layer Prefetching（Section IV-C）**：利用残差连接导致相邻层 hidden state 高度相似的特点，复用后续层的 gating 信息预测 next-3-layers 的 expert 激活，通过仿真评估预取每个 expert 对整体调度效率的潜在影响（impact），贪心选择收益最高的 expert 进行预取。
    3. **Score-Aware Caching / MRS Replacement Policy（Section IV-D）**：提出 Minus Recent Score (MRS) 替换策略，利用 expert routing score 作为缓存优先级信号。公式 S = α × TopP(s) + (1-α) × S，仅累积 top-p 个 expert 的 score（p 通常为 2× 激活 expert 数）。利用高 score expert 在后续 iteration 中更可能被重用的观察。
  - 实验比较：llama.cpp（静态按层映射到 CPU/GPU）、AdapMoE（SOTA GPU-centric MoE 调度，自适应 prefetching+caching）、kTransformers（SOTA CPU-GPU hybrid MoE 调度，按历史激活频率静态映射）。指标：TTFT（prefill 阶段）、TBT（decode 阶段）。消融实验对比 Scheduling、Prefetching、Caching 各组件的独立贡献。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA RTX A6000
  - CPU：Intel Xeon Gold 5220R，限制使用 10 cores（模拟边缘部署场景）
  - 通过调整 GPU expert cache ratio（25%, 50%, 75%）评估不同硬件配置下的性能和扩展性

- 开源Serving框架是什么。修改了什么。
  - 基于 **kTransformers** (https://github.com/kvcache-ai/ktransformers) 和 **llama.cpp** kernels。
  - kTransformers 提供灵活的基础设施用于 kernel injection 和混合 CPU-GPU 执行。
  - 核心修改：
    1. **Hybrid Scheduler**：在 kTransformers 的 expert 执行路径中插入优先级规则驱动的调度逻辑，运行时动态分配 expert 到 CPU 或 GPU。仿真阶段在 warmup 中完成，收集 CPU/GPU 处理速度和数据传输延迟。
    2. **Parallel Execution Engine**：利用 fine-grained CUDA stream 调度实现 CPU、GPU、PCIe transfer 三者的并行执行。修改 C++ kernels 直接处理 expert 计算任务分配，消除 Python 开销。
    3. **Prefetching Module**：在每层计算时利用后续层 gating 信息预测并预取 expert，与当前层计算并行。
    4. **MRS Cache Manager**：替换 kTransformers 原有的 LFU 缓存策略为 score-aware MRS 策略。
    5. **Marlin Quantization**：集成 llama.cpp 的 Marlin 4-bit 量化 kernel 提升计算效率和降低内存使用。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：代码开源在 https://github.com/PKU-SEC-Lab/HybriMoE
  - **HybriMoE MoE 推理全流程（单 MoE layer，以 Mixtral-8x7B 为例）**：

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Warmup 阶段                                                    │
│    - 测量 CPU expert computation speed (per token latency)        │
│    - 测量 GPU expert computation speed                            │
│    - 测量 PCIe transfer latency (CPU↔GPU expert weight copy)      │
│    - 初始化 MRS cache: 随机加载 k 个 expert 到 GPU cache          │
│           ↓                                                       │
│ 2. 用户输入 prompt tokens [T₁, T₂, ..., Tₙ]                       │
│    Prefill + Autoregressive decode loop:                           │
│           ↓                                                       │
│ 3. 每层 MoE 执行                                                  │
│    ┌─ Attention Block ────────────────────────────────────────┐   │
│    │  Non-expert 权重常驻 GPU，直接在 GPU 计算                  │   │
│    └──────────────────────────────────────────────────────────┘   │
│    ┌─ MoE Gating/Router ──────────────────────────────────────┐   │
│    │  gate_logits = W_gate @ h  (常驻 GPU, 轻量级)             │   │
│    │  topk_experts, gate_weights = topk(softmax(logits), K)   │   │
│    └──────────────────────────────────────────────────────────┘   │
│    ┌─ Hybrid Scheduling Decision ─────────────────────────────┐   │
│    │  1. 检查每个 activated expert 是否在 GPU cache 中          │   │
│    │  2. 构建 GPU Queue: 已缓存 expert，按 load 降序排列        │   │
│    │  3. 构建 CPU Queue: 未缓存 expert，按 load 升序排列        │   │
│    │  4. Simulation Phase: 迭代填充 CPU/GPU/PCIe 时间线          │   │
│    │     while 未完成所有 expert:                               │   │
│    │       选最早完成的 timeline → 执行对应操作:                │   │
│    │         - GPU: 从 GPU Queue 取最高 load cached expert      │   │
│    │         - CPU: 从 CPU Queue 取最低 load uncached expert    │   │
│    │         - PCIe: 从 CPU Queue 取最高 load uncached expert   │   │
│    │           → 传输完成后插入 GPU Queue (按 load 降序)        │   │
│    │  5. 选择 min max(CPU_TIME, GPU_TIME) 的调度方案            │   │
│    └──────────────────────────────────────────────────────────┘   │
│    ┌─ Parallel Execution (3-way: CPU + GPU + PCIe) ───────────┐   │
│    │  CUDA Stream 0 (GPU compute):                              │   │
│    │    for cached expert e (sorted by load desc):              │   │
│    │      Marlin 4-bit quantized GEMM:                          │   │
│    │        gate_out = SiLU(W_gate_4bit @ h)                    │   │
│    │        up_out = W_up_4bit @ h                              │   │
│    │        out += gate_weight * gate_out * up_out @ W_down_4bit│   │
│    │  CUDA Stream 1 (PCIe transfer):                            │   │
│    │    for high-load uncached expert:                          │   │
│    │      cudaMemcpyAsync(CPU_weight → GPU_buffer, PCIe)        │   │
│    │  CPU Thread Pool (CPU compute):                            │   │
│    │    for low-load uncached expert:                           │   │
│    │      llama.cpp C++ kernel:                                 │   │
│    │        CPU GEMM → expert FFN output                        │   │
│    │    (CPU 端首 expert 计算慢，后续 expert 因 cache 利用更快)  │   │
│    └──────────────────────────────────────────────────────────┘   │
│    ┌─ Impact-Driven Prefetching (与当前层计算并行) ────────────┐   │
│    │  1. 读取 layer l+1, l+2, l+3 的 gating weights             │   │
│    │  2. 计算预测的 expert activation:                          │   │
│    │     pred_experts_{l+i} = topk(W_gate_{l+i} @ h_l, K)       │   │
│    │  3. 对每个候选 prefetch expert:                            │   │
│    │     模拟 "若预取该 expert" 对调度效率的影响                 │   │
│    │     (复用 Section IV-B 的仿真逻辑)                          │   │
│    │  4. 贪心选择 impact 最高的 expert → async 预取             │   │
│    └──────────────────────────────────────────────────────────┘   │
│    ┌─ MRS Cache Update (每 iteration 结束) ───────────────────┐   │
│    │  1. 获取当前 iteration 的 routing scores s                 │   │
│    │  2. 更新 priority scores:                                  │   │
│    │     S = α × TopP(s) + (1-α) × S                            │   │
│    │     (仅 top-p=2K 个 expert score 被累积)                    │   │
│    │  3. 若需 evict: 选择 S 最低的 expert 从 GPU cache 移除    │   │
│    │  4. 将新激活的 expert 写入 GPU cache                       │   │
│    └──────────────────────────────────────────────────────────┘   │
│           ↓                                                       │
│ 4. Expert FFN 聚合输出                                            │
│    out = Σ gate_weights[e] × ExpertFFN_e(h)                       │
│    (CPU 计算的结果通过 PCIe 拷回 GPU)                              │
│           ↓                                                       │
│ 5. 输出: next token logits → 采样 → 下一个 token                  │
└─────────────────────────────────────────────────────────────────┘
```

  - **关键技术原理**：
    - **优先级规则的设计动机**：GPU 优先高负载缓存 expert（减少传输开销），CPU 优先低负载未缓存 expert（CPU 延迟与 load 线性相关），PCIe 优先高负载未缓存 expert（最大化 GPU 利用率）
    - **仿真调度原理**：warmup 阶段测量 CPU_TIME_per_expert、GPU_TIME_per_expert、TRANSFER_TIME → 运行时基于实际 expert activation 构建优先级队列 → 迭代式仿真（贪心 fill timelines）→ 输出最优 expert-to-device 分配方案
    - **MRS vs LRU/LFU**：LRU/LFU 不考虑 MoE expert 的 routing score 预测信号。MRS 利用"高 score expert 更可能在下一 iteration 被重用"的观察（图 3b），通过指数移动平均累积 score 信号

  - **关键性能数据**：
    | Stage | Model | Cache Ratio | HybriMoE Speedup vs kTransformers |
    |-------|-------|-------------|-----------------------------------|
    | Prefill (avg) | All models | 25%-75% | 1.33× |
    | Decode (avg) | All models | 25%-75% | 1.70× |

  - **Ablation Study (Qwen2, 25% cache ratio)**:
    | Technique | Prefill Latency(s) | Prefill Speedup | Decode Latency(s) | Decode Speedup |
    |-----------|-------------------|-----------------|-------------------|----------------|
    | Baseline (kTransformers) | 1.47 | — | 0.21 | — |
    | +Scheduling | 1.17 | 1.26× | 0.14 | 1.46× |
    | +Prefetching | 1.39 | 1.06× | 0.18 | 1.15× |
    | +Caching | — | — | 0.15 | 1.38× |
    | All | 1.13 | 1.31× | 0.11 | 1.86× |

  - **MRS Cache Hit Rate vs LRU (Figure 9)**:
    | Model | 25% Cache (MRS vs LRU) | 75% Cache (MRS vs LRU) |
    |-------|------------------------|------------------------|
    | Mixtral | 36.2% vs 30.2% (+6%) | 83.3% vs 80.6% |
    | DeepSeek | 52.7% vs 47.7% (+5%) | — |
    | Qwen2 | 52.8% vs 45.0% (+7.8%) | — |
