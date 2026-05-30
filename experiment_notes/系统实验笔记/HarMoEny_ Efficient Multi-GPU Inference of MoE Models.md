## HarMoEny: Efficient Multi-GPU Inference of MoE Models

- 属于Serving调度的实现是什么？实验比较什么？
  - HarMoEny 实现了一个面向多 GPU MoE 推理的动态负载均衡 Serving 调度系统，核心调度优化包括：
    1. **Dynamic Token Rebalancing（Section 4.2, Algorithm 2）**：每个 batch 中所有 GPU 交换轻量级 metadata（token-to-expert 分配摘要，约 4kB），构建全局 token 分布视图。贪心调度算法确定性地识别最过载 GPU g_max 和最大贡献源 GPU g_from，将 token 从过载 GPU 重路由到欠载 GPU g_min，逐步迭代直至负载平衡。引入 token threshold q 控制最小 offload 粒度。
    2. **Asynchronous Expert Prefetching（Section 4.3）**：当 token rebalancing 将 expert 重分配到未持有该 expert 的 GPU 时，通过独立 CUDA stream 异步从 system memory 预取 expert 权重，直接覆写已完成的 expert 内存位置（无需先写回 system memory），overwrite 加速 5.5×（11ms→2ms on V100）。
  - 实现量：1115 行 PyTorch 代码，使用 NVIDIA CUDA stream 实现异步 expert 加载。MoE 层实现为 PyTorch nn.Module，可应用于任意 PyTorch 模型。
  - 实验比较：DeepSpeed（Tutel enabled, round-robin EP）、FastMoE、FasterMoE（dynamic shadowing）、ExFlow（integer programming expert placement）共 4 个 baseline。
  - 指标：Throughput（tokens/s）、Mean TTFT（time-to-first-token）。ablation study 额外包含 time breakdown（CUDA Events 细粒度分析）和不同 load balancing policies 对比（Round-robin、ExFlow policy、Even Split）。

- 硬件平台是什么，配置是什么。
  - NVIDIA DGX1 机器，8× NVIDIA V100 GPU（每 GPU 32GB VRAM），NVLink 互联，500 GB system memory。

- 开源Serving框架是什么。修改了什么。
  - HarMoEny 直接基于 PyTorch 实现（不基于已有 serving 框架修改），开源在 https://github.com/sacs-epfl/HarMoEny。
  - 核心修改/新增（作为 PyTorch nn.Module 插入现有模型）：
    1. **MoE Layer 重写**：Algorithm 1 定义的 6 步 MoE forward 流程——token routing → metadata exchange → token scheduling (rebalancing) → scatter tokens → expert processing + async loading → gather tokens。替代标准 MoE 层的 "router → all-to-all dispatch → expert FFN → all-to-all combine" 流程。
    2. **Token Scheduler**：Algorithm 2 的贪心负载均衡调度器，在 metadata exchange 后同步计算出全局最优 token-to-GPU schedule S。
    3. **Expert Prefetching 协议**：异步 expert 权重传输——当一个 expert 完成计算后，立即检查是否有下一个需要运行但未加载的 expert，通过独立 CUDA stream 异步 prefetch 权重直接覆写已完成 expert 的内存。
    4. **Configurable Router Skew**：可配置的人工 expert 流行度偏斜机制（参数 α ∈ [0,1]），支持可控的 token 分布倾斜实验。
  - 需要 Gurobi license 来运行 ExFlow baseline 对比实验（ExFlow 使用 integer programming）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：代码完整开源在 https://github.com/sacs-epfl/HarMoEny（212 commits, 82.5% Python, 16.5% Shell, 1.0% Dockerfile）。包含 Docker 支持、EC2 setup 脚本、experiments 目录下的可执行实验脚本。
  - **HarMoEny MoE 推理全流程（以 Switch128, 8×V100, batch inference 为例）**：

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 模型加载与配置                                                │
│    PyTorch 加载 Switch128 (12 MoE layers, 128 experts/layer)      │
│    - Self-Attention + Router: 复制到所有 8 GPU (data parallelism) │
│    - Experts: 初始 round-robin 分布到 8 GPU (EP=8)               │
│    - 每 GPU 持有 16 个 expert (128/8)，每 expert 18MB             │
│    HarMoEny 通过 replace_moe_layer() 注入自定义 MoE 层            │
│           ↓                                                      │
│ 2. 用户输入 batch tokens [B tokens]                              │
│           ↓                                                      │
│ 3. 每层 MoE forward (Algorithm 1, 6 steps)                       │
│    ┌─ Step 1: Token Routing ─────────────────────────────────┐   │
│    │  各 GPU 独立计算 self-attention → Router(W_gate @ h)     │   │
│    │  → m_expert: token-to-expert assignment tensor           │   │
│    │  (各 GPU 复制执行, 无通信)                                │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ Step 2: Metadata Exchange ────────────────────────────┐   │
│    │  SENDMETADATATOGPUs(m_expert)                           │   │
│    │  每 GPU 广播本地 token-expert 分配 (~4KB metadata)       │   │
│    │  → m_all: 全局 token-to-expert assignment                │   │
│    │  (negligible overhead)                                   │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ Step 3: Token Scheduling (Algorithm 2) ───────────────┐   │
│    │  S_initial = INITIALASSIGN(m_all)                       │   │
│    │  S = REBALANCE(S_initial):                               │   │
│    │    t_avg = total_tokens / |G|                            │   │
│    │    while any GPU has tokens > t_avg:                     │   │
│    │      g_max = most overloaded GPU                        │   │
│    │      g_from = GPU contributing most tokens to g_max     │   │
│    │      e_max = expert from g_from sending most to g_max   │   │
│    │      t_move = tokens to transfer                         │   │
│    │      if t_move < q: stop (insufficient to amortize)     │   │
│    │      g_min = least loaded GPU                           │   │
│    │      transfer min(t_move, t_avg - t_g[g_min]) tokens    │   │
│    │      from e_max on g_from → g_min                        │   │
│    │  → S: rebalanced 3D schedule [src_GPU, expert, dst_GPU] │   │
│    │  (各 GPU 独立并行计算, 因 metadata 全局一致, 结果相同)    │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ Step 4: Scatter Tokens ───────────────────────────────┐   │
│    │  SENDTOKENSTOGPUS(x, m_expert, S)                       │   │
│    │  All-to-all communication: 各 GPU 按 rebalanced S 发送  │   │
│    │  token 到目标 GPU → receive x' from all other GPUs      │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ Step 5: Expert Processing + Async Loading ────────────┐   │
│    │  for each expert e assigned to this GPU:                │   │
│    │    if e not in GPU memory:                              │   │
│    │      async CUDA stream: copy e weights from sys mem     │   │
│    │      overwrite completed expert's memory (5.5× faster)  │   │
│    │      18MB / (PCIe bandwidth) ≈ 2ms (V100)              │   │
│    │    compute: x''_e = e(x'_e)  // expert FFN forward      │   │
│    │    (async prefetch overlaps with current expert compute)│   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ Step 6: Gather Tokens ────────────────────────────────┐   │
│    │  SENDTOKENSBACKTOGPUs(S, x'')                           │   │
│    │  All-to-all communication: 各 GPU 返回处理后的 token    │   │
│    │  → RECONSTRUCT(S, y, m_all): 恢复原始 token 顺序       │   │
│    └────────────────────────────────────────────────────────┘   │
│ 4. 输出: next token logits → generated tokens                   │
│    90% skew workload: 186 tok/s (vs 106 tok/s ExFlow, +75%)     │
│    无 skew workload: 213 tok/s (vs ~200 tok/s baselines)        │
└─────────────────────────────────────────────────────────────────┘
```

  - **Token threshold q 的数学推导（Section 4.4）**：
    q 由硬件规格决定，与动态 workload 无关：q > φ·d_type / (2·β)
    其中 φ = GPU FLOPS, d_type = 元素字节数, β = PCIe 带宽。
    物理含义：确保 expert 计算时间 > expert 加载时间，使 prefetch 可被计算掩盖。

  - **关键性能数据（Table 1 model specs）**：
    | Model | MoE Layers | Experts | Expert Size |
    |-------|-----------|---------|-------------|
    | Switch128 | 12 (alternating) | 128 | 18 MB |
    | Qwen 1.5 MoE | 24 | 60 | 33 MB |

  - **Throughput 对比（Switch128, 90% skew, Constant dataset）**：
    | System | Throughput | HarMoEny Speedup |
    |--------|-----------|-----------------|
    | HarMoEny | 186 tok/s | 1.0× |
    | ExFlow | 109 tok/s | 1.7× |
    | FasterMoE | 109 tok/s | 1.7× |
    | FastMoE | 124 tok/s | 1.5× |
    | DeepSpeed | 20 tok/s | 9.1× |

  - **Ablation: time breakdown（Switch128, 90% skew, MoE layer 1）**：
    - No rebalancing: mean latency 289ms, GPU idle 82.6%
    - Rebalancing only: mean latency 149.5ms (-48.3%), GPU idle 2.6%
    - Rebalancing + async prefetch: mean latency 136.6ms (-8.6% over sync)
    - Scheduler overhead: 30.8% of mean latency (Switch128), 20.3% (Qwen)

  - **Real-world datasets throughput (Switch128)**：
    HarMoEny: 201 tok/s (steady across datasets)
    FasterMoE/FastMoE: 92-98% of HarMoEny throughput
    ExFlow: inconsistent due to inability to adapt to dynamic skew
    DeepSpeed: very low due to input padding strategy
