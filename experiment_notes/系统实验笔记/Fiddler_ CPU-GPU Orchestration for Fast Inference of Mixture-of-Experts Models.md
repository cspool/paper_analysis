## Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models

- 属于Serving调度的实现是什么？实验比较什么？
  - Fiddler 实现了一个面向资源受限环境的 MoE 推理 CPU-GPU 协同调度系统。核心调度策略包括：
    1. **动态执行策略选择（Algorithm 1）**：每个 expert 根据输入 token 数量 s 在三种策略中动态选择——(a) 若 expert 权重已在 GPU memory，直接在 GPU 执行；(b) 若 `cpu_lat(s) > gpu_lat(s) + trans_lat()`，则从 CPU memory 拷贝权重到 GPU 并在 GPU 执行；(c) 否则从 GPU memory 拷贝 activation 到 CPU memory 并在 CPU 执行。决策基于 latency model：GPU 延迟恒定（受限于参数加载），CPU 延迟随输入量线性增长。
    2. **Expert 热门度导向的 GPU 放置**：离线 profiling 统计各 expert 激活频率，按热门度降序将尽可能多的 expert 放入 GPU memory，最大化 GPU cache hit rate。在 Env1（56/256 expert on GPU）下 hit rate 从随机 21.9% 提升至 25.2%，Env2（125/256）从 48.8% 升至 53.0%。
    3. **AVX512_BF16 CPU 专用计算 kernel**：利用 Intel AVX512_BF16 指令集实现 CPU 端 expert 计算 kernel，PyTorch 原生不支持该指令集。
  - 实验比较：
    - Fiddler vs DeepSpeed-MII (ZeRO-Infinity) vs Mixtral-Offloading vs llama.cpp
    - 场景 a：不同输入/输出长度的 end-to-end 单 batch 推理吞吐
    - 场景 b：长上下文 prefill 的 TTFT（512-4096 input tokens）
    - 场景 c：beam search（width 4-16）end-to-end 延迟

- 硬件平台是什么，配置是什么。
  - Environment 1：NVIDIA Quadro RTX 6000（24576MiB VRAM），Intel Xeon Gold 6126（48 cores），PCIe Gen3 x16（32GB/s）
  - Environment 2：NVIDIA RTX 6000 Ada（49140MiB VRAM），Intel Xeon Platinum 8480+（112 cores），PCIe Gen4 x16（64GB/s）
  - 约束：两块 GPU 的显存均无法容纳 Mixtral-8x7B 全部 90GB+ 参数，Env1 仅可放 56/256 expert，Env2 可放 125/256 expert

- 开源Serving框架是什么。修改了什么。
  - Fiddler 基于 PyTorch 构建，不修改已有开源 serving 框架，而是自建 CPU-GPU 协同调度系统。
  - 核心修改/新增：
    - **Per-expert 执行策略决策**：替代 per-layer 统一执行，对每个 expert 独立判断 CPU/GPU 执行策略
    - **Expert 权重分配**：initialization 阶段将 non-expert 层+热门 expert 放 GPU memory，其余 expert 放 CPU memory
    - **Latency model 校准**：initialization 阶段测量 cpu_lat(s)、gpu_lat(s)、trans_lat() 为运行时决策提供参数
    - **CPU AVX512_BF16 kernel**：自定义 C++ kernel 替代 PyTorch 默认 CPU GEMM

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：代码开源在 https://github.com/efeslab/fiddler
  - **Serving 框架执行全过程（以 Mixtral-8x7B 16-bit 在 Environment 1, single-batch inference 为例）**：

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. 初始化阶段                                                    │
    │    - 加载 Mixtral-8x7B (47B params, 16-bit, >90GB)               │
    │    - Non-expert 层 (Attention/Embedding/Norm): 常驻 GPU (~2B)     │
    │    - Expert 层 (32 layers × 8 experts = 256 experts):             │
    │      热门度 profiling (ShareGPT calibration) → 按热门度排序      │
    │      GPU 放置 top-56 热门 expert，其余 200 个 expert 放 CPU RAM   │
    │    - 测量 microbenchmark: cpu_lat(s), gpu_lat(s), trans_lat()    │
    │           ↓                                                       │
    │ 2. 用户输入 prompt tokens [T₁, T₂, ..., Tₙ]                       │
    │    Prefill + autoregressive decode loop:                          │
    │           ↓                                                       │
    │ 3. 每层 MoE 执行 (Forward)                                        │
    │    for each layer l in 0..31:                                     │
    │      ┌─ Attention block ────────────────────────────────────┐    │
    │      │  Weights 常驻 GPU, 直接在 GPU 计算                     │    │
    │      └──────────────────────────────────────────────────────┘    │
    │      ┌─ MoE Gate ───────────────────────────────────────────┐   │
    │      │  gate_scores = W_gate[l] @ h  (常驻 GPU)              │   │
    │      │  top2_experts, gate_weights = topk(softmax(scores),2) │   │
    │      │  infl_size[j] = count(tokens routed to expert j)      │   │
    │      └──────────────────────────────────────────────────────┘   │
    │      ┌─ Fiddler Algorithm 1: Per-expert 执行决策 ────────────┐  │
    │      │  for j = 1 to 8:                                       │  │
    │      │    s = inp_size[j]                                     │  │
    │      │    if s == 0: continue                                 │  │
    │      │    if is_at_gpu(l, j):                                 │  │
    │      │      → GPU 直接执行 (无数据传输)                        │  │
    │      │    elif cpu_lat(s) > gpu_lat(s) + trans_lat():         │  │
    │      │      → CPU→GPU copy expert weight (300MB/expert, PCIe) │  │
    │      │      → GPU 执行 expert FFN                             │  │
    │      │    else:                                               │  │
    │      │      → GPU→CPU copy activation (s×4096 floats, PCIe)   │  │
    │      │      → CPU AVX512_BF16 执行 expert FFN                 │  │
    │      │      → CPU→GPU copy output activation                  │  │
    │      └──────────────────────────────────────────────────────┘   │
    │      ┌─ Expert FFN 聚合 ────────────────────────────────────┐   │
    │      │  out = Σ gate_weights[j] * SiLU(W_gate_e @ h)         │   │
    │      │       * (W_up_e @ h)   // 各 expert 独立计算后加权加和  │   │
    │      └──────────────────────────────────────────────────────┘   │
    │           ↓                                                       │
    │ 4. 输出: generated tokens                                         │
    └─────────────────────────────────────────────────────────────────┘
    ```

    **关键性能数据**：
    | Scenario | Fiddler vs Best Baseline | Env1 | Env2 |
    |----------|--------------------------|------|------|
    | Single batch (avg) | 1.26× vs llama.cpp | — | — |
    | Long prefill TTFT (avg) | 1.07× vs DeepSpeed-MII | — | — |
    | Beam search (avg) | 11.57× vs llama.cpp | — | — |

    **三种执行策略的 latency model**：
    - Strategy (a): latency = gpu_lat(s) ≈ constant（GPU 直接从显存执行）
    - Strategy (b): latency = gpu_lat(s) + trans_lat() ≈ constant + weight_transfer（GPU 计算+PCIe 传权重，约 2-5× 计算时间）
    - Strategy (c): latency = cpu_lat(s) + negligible_act_copy ≈ linear in s（CPU 计算，activation 拷贝 <1% 总延迟）
    - 决策阈值：当 s 较小时 cpu_lat(s) < gpu_lat(s) + trans_lat() 选 (c)；当 s 较大时选 (b)
