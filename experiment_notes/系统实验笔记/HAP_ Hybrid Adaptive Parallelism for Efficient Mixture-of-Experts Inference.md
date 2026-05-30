## HAP: Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - HAP 在 DeepSpeed-FastGen 上构建了一套面向 MoE 推理的动态混合并行策略自动选择系统。核心调度优化包括：
    1. **Module Decomposition（Section III-B）**：将 MoE 模型分解为 Attention 模块和 Expert 模块两个独立计算单元，各配备专用的推理延迟仿真模型（基于 FLOPs 的计算仿真模型 + 基于数据量的通信仿真模型），使用随机森林回归拟合仿真系数 η 和 ρ，计算仿真误差 <10%，通信仿真误差 <5%。
    2. **ILP-based Hybrid Parallel Strategy Search（Section III-C）**：构建 Attention 模块（DP/TP/DP+TP 混合）和 Expert 模块（EP/TP/EP+TP 混合）的并行策略搜索空间，将最小化端到端推理延迟问题形式化为整数线性规划（ILP），通过 Python PuLP 库求解最优混合并行配置。搜索在典型 8-GPU 单机配置下 <1 秒完成。
    3. **Dynamic Parallelism Transition Strategy（Section III-D）**：prefill 和 decode 阶段使用不同并行策略时，维护 INT4（per-group 量化）备份权重于 CPU memory，通过多 stream 异步流水线上传并反量化恢复为 BF16 精度。过渡策略根据仿真在 weight redistribution（集合通信）与 quantized upload+dequant 之间选择开销更低的方案。Per-group 量化保持 MMLU 67.7%（与原版一致）、GSM8K 58.0%（vs 原版 58.3%）。
  - 实验比较：HAP-based inference vs TP-based inference（baseline），端到端延迟对比。四种推理场景：短上下文约束输出（256 in + 64 out）、短上下文扩展输出（256 in + 2048 out）、长上下文约束输出（4096 in + 64 out）、长上下文扩展输出（4096 in + 2048 out）。

- 硬件平台是什么，配置是什么。
  - NVIDIA A100 80GB（节点内 NVLink，高带宽），4×A100 及 8×A100 配置。
  - NVIDIA A6000 48GB（节点内 PCIe，低带宽），4×A6000 配置。
  - NVIDIA V100（节点内 PCIe），8×V100 配置。
  - 单节点多 GPU 部署场景。

- 开源Serving框架是什么。修改了什么。
  - **Serving 框架**：DeepSpeed-FastGen（https://github.com/microsoft/DeepSpeed），基于 MII 和 DeepSpeed-Inference 的高吞吐文本生成框架，原生支持 TP 等静态并行策略。
  - **修改内容**：
    1. **并行策略搜索引擎**：在 DeepSpeed-FastGen 的模型加载/初始化阶段集成 ILP 求解器（Python PuLP），基于硬件规格（GPU 数、显存、带宽）和模型配置（hidden dim、expert 数、层数）自动搜索最优混合并行策略。
    2. **动态策略切换机制**：在 prefill→decode 过渡点插入策略切换逻辑——若 Expert 模块在 prefill 和 decode 使用不同策略，触发 INT4 量化权重上传 + GPU 反量化，或 weight redistribution via AllGather/AllToAll。
    3. **计算/通信仿真模型校准**：initialization 阶段运行 microbenchmark 收集计算和通信延迟数据，训练随机森林回归模型以精确估计各策略组合的延迟。
    4. **内存约束感知**：ILP 约束中包含 KV cache、attention 权重、expert 权重、activation 的 per-device 内存占用约束，对 EP 采用保守上限估计（2× TP activation footprint）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：论文未明确给出 HAP 独立开源仓库。基于开源 DeepSpeed-FastGen 构建。
  - **HAP Serving 框架执行全过程（以 Mixtral-8x7B 在 4×A6000、4096-token context + 64-token generation 为例）**：
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. 初始化阶段                                                    │
    │    HAP 读取模型配置 (Mixtral-8x7B: 32 layers, hidden=4096,      │
    │    experts=8, MoE_inter=14336) 和硬件配置 (4×A6000, PCIe)       │
    │    → Microbenchmark 收集计算/通信延迟数据                        │
    │    → 训练随机森林回归仿真模型 (计算 η, 通信 ρ)                    │
    │    → ILP 求解器 (PuLP) 搜索最优混合并行策略:                      │
    │      Attention 模块: DP=4 (避免 TP 的 AllReduce 通信开销)        │
    │      Expert 模块 prefill: EP=4 (低通信量)                        │
    │      Expert 模块 decode: TP=4 (避免 EP 负载不均衡)               │
    │    → 准备 INT4 量化 expert 权重备份于 CPU memory                  │
    │           ↓                                                      │
    │ 2. Prefill 阶段 (4096 tokens 并行)                               │
    │    for layer l in 0..31:                                         │
    │      ┌─ Attention (DP=4, 各 GPU 独立计算) ───────────────────┐  │
    │      │  各 GPU 持有完整 attention 权重                         │  │
    │      │  处理 1/4 batch tokens, 无通信                          │  │
    │      └──────────────────────────────────────────────────────┘  │
    │      ┌─ MoE Gate (各 GPU 复制执行) ──────────────────────────┐  │
    │      │  gate_logits = W_gate @ h  [1×(64×4096) → 1×8]       │  │
    │      │  top2_experts = topk(softmax(logits), 2)              │  │
    │      └──────────────────────────────────────────────────────┘  │
    │      ┌─ Expert FFN (EP=4, 每 GPU 2 experts) ─────────────────┐  │
    │      │  GPU0: experts 0,1; GPU1: experts 2,3; etc.          │  │
    │      │  All-to-All dispatch: tokens 路由到 expert 所在 GPU   │  │
    │      │  各 GPU 计算本地 expert FFN (SwiGLU)                   │  │
    │      │  All-to-All combine: 聚合输出                         │  │
    │      └──────────────────────────────────────────────────────┘  │
    │           ↓                                                      │
    │ 3. Prefill→Decode 过渡（动态策略切换）                            │
    │    Expert 模块策略: EP=4 → TP=4                                 │
    │    HAP 仿真评估过渡开销:                                          │
    │      T_reshard (AllGather+AllToAll) vs T_upload+T_dequant       │
    │    → 选择 INT4 量化权重上传方案 (更低开销):                       │
    │      CPU→GPU async copy INT4 权重 (multi-stream)                │
    │      → GPU 端 per-group dequant 恢复 BF16                       │
    │      → 过渡开销与 prefill 计算重叠（T_dequant < T_attn+T_comm） │
    │           ↓                                                      │
    │ 4. Decode 阶段 (逐 token autoregressive, 64 tokens)              │
    │    for each new token:                                           │
    │      ┌─ Attention (DP=4, 同 prefill) ───────────────────────┐  │
    │      │  各 GPU 独立计算，batch=1 per GPU                      │  │
    │      └──────────────────────────────────────────────────────┘  │
    │      ┌─ MoE Gate (各 GPU 复制执行) ──────────────────────────┐  │
    │      └──────────────────────────────────────────────────────┘  │
    │      ┌─ Expert FFN (TP=4, 各 GPU 持有完整 expert 的 1/4) ───┐  │
    │      │  Expert 权重沿中间维度切分 (14336→3584 per GPU)       │  │
    │      │  各 GPU 计算部分输出 → AllReduce 聚合                  │  │
    │      │  decode 阶段通信量小 (单 token)，TP 负载均衡优势明显   │  │
    │      └──────────────────────────────────────────────────────┘  │
    │           ↓                                                      │
    │ 5. 输出: 64 个 generated tokens                                  │
    │    端到端延迟 vs TP baseline: 1.68× speedup on A6000            │
    └─────────────────────────────────────────────────────────────────┘
    ```

    **关键设计原理**：
    - 长上下文 prefill 场景：通信是瓶颈（TP 的 AllReduce 通信量大），HAP 为 Attention 选 DP（无通信）、Expert 选 EP（All-to-All 通信量低于 TP 的 AllReduce）
    - Decode 场景：计算是瓶颈（单 token），EP 的负载不均衡会浪费 GPU 计算资源，HAP 为 Expert 切换为 TP
    - 短上下文场景：TP 在多数配置下已接近最优，HAP 搜索后可能仍选 TP，实现不低于 baseline 的延迟

    **关键性能数据**：
    | Scenario | Model | Hardware | HAP Speedup vs TP |
    |----------|-------|----------|-------------------|
    | 256in+64out | Mixtral-8x7B | 4×A100 | 1.16× |
    | 256in+64out | Qwen1.5-MoE | 4×A6000 | 1.37× |
    | 4096in+64out | Mixtral-8x7B | 4×A6000 | up to 1.68× |
    | 4096in+64out | Mixtral-8x7B | 4×A100 | up to 1.77× |
    | 4096in+64out | Qwen2-57B | 4×A100 | up to 1.52× |
    | 2048in+64out | Mixtral-8x7B | 8×V100 | 1.57× |
    | 4096in+2048out | Mixtral-8x7B | 4×A100 | up to 1.13× |

    HAP 优势最大出现在长上下文+约束输出场景（prefill 主导延迟，通信瓶颈严重），短上下文+扩展输出场景加速最小（decode 主导延迟，TP 已是最优）。
