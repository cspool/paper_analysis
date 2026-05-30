## Faster MoE LLM Inference for Extremely Large Models

- 属于Serving调度的实现是什么？实验比较什么？
  - 论文基于 sglang serving 框架对 fine-grained MoE 模型（DeepSeek-V2-Lite、DeepSeek-V3）进行推理效率分析，核心调度层面的实现和实验包括：
    1. **MoE batch effect 分析（Section 4）**：通过 PyTorch + torch.compile 模拟实验，分析 MoE 层在不同 sequence length 下的 latency 和 throughput，对比 FFN，量化 MoE 因额外 expert 参数加载导致的"弱化批次效应"（weakened batch effect）。
    2. **Expert skipping 效率实验（Section 5）**：在 sglang 上修改 expert 激活逻辑，使每层仅激活 na（2 到原始值）个 expert，测量不同并发度（2-768）下的 throughput 变化和加速比。
    3. **Expert pruning 效率实验（Section 6）**：在 sglang 上减少总 expert 数量 ne（从 64 降到 8-48），测量不同并发度（2-784）下的 throughput 和加速比。
    4. **Expert parallelism 效率分析（Section 4.2）**：理论分析 EP vs TP 的通信开销——EP 在 fine-grained MoE 中通过 group-constrained routing 可将跨节点通信从 2(nd-1)Ld 降至 2naLd。
  - 实验比较：
    - Section 4：MoE vs FFN 在不同 sequence length L 下的 per-token latency 和 AI
    - Section 5：不同 na（2-6/2-8）在不同并发度下的 throughput（token/s）和 speedup ratio
    - Section 6：不同 ne（8-64）在不同并发度下的 throughput 和 speedup ratio（up to 2.3×）
    - Section 5.2：不同 inter-layer expert 分配策略（ascending/descending/peak/valley）下的 benchmark 性能

- 硬件平台是什么，配置是什么。
  - **Section 4 模拟实验**：1× NVIDIA Tesla A800 80G PCI-e, Intel Xeon Silver 4314 CPU @ 2.40GHz (24 cores), 15×16GB ECC DDR4@2666MHz
  - **DeepSeek-V2-Lite (Section 5 & 6)**：2× NVIDIA Tesla A800 80G PCI-e, Intel Xeon Silver 4314 CPU @ 2.40GHz (24 cores), 15×16GB ECC DDR4@2666MHz
  - **DeepSeek-V3 (Section 5 & 6)**：8× NVIDIA Tesla H200 141G SXM5, Intel Xeon Platinum 8558 CPU @ 2.10GHz (48×2 cores), 32×64GB ECC DDR4@2666MHz
  - 效率测试约束：固定 1024 input tokens + 1024 output tokens

- 开源Serving框架是什么。修改了什么。
  - **Serving 框架**：sglang build v0.4.4 post 1 (commit ad4e58bf67ec833ff4d036af5129ec6e1633efc4)
  - **Profiling 工具**：sglang.bench
  - **修改内容**：
    1. **Expert skipping 修改**：在 sglang 的 MoE expert 调度中，将所有 MoE 层的激活 expert 数 na 从默认值（V2-Lite=6, V3=8）统一降低到 2 至原始值之间的某个值。修改涉及 MoE layer 的 top-k 选择逻辑——在 router gate 输出后，将 topk 的 k 参数替换为缩减后的 na。
    2. **Expert pruning 修改**：在模型加载阶段，根据选择策略（random/structured/activate count/soft count）从 ne 个 expert 中选择 ne' 个保留，其余不加载到显存。修改涉及模型权重加载路径——仅加载选中的 expert 参数。
    3. **Section 4 模拟**：使用 PyTorch + torch.compile + HuggingFace Transformers (MixtralModel)，实现 MoE 和 FFN 的 latency 模拟，不修改 sglang。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：论文未提供独立开源仓库。使用开源框架 sglang (https://github.com/sgl-project/sglang) 和 PyTorch、HuggingFace Transformers。
  - **Serving 框架执行全过程（以 DeepSeek-V2-Lite, expert skipping na=2, 并发度=512, 2×A800 为例）**：

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. 模型加载与配置                                                │
    │    sglang 加载 DeepSeek-V2-Lite (16B, 64 experts, na 从 6→2)     │
    │    - Attention/Embedding/Norm: 常驻 GPU 0                        │
    │    - Shared Expert (ds=10944): 常驻 GPU 0                        │
    │    - Routed Experts (64×de=1408): 分布在 2×A800 (EP=2)           │
    │    - 每层仅激活 na=2 个 routed expert (从 6 降至 2)               │
    │           ↓                                                      │
    │ 2. 请求到达与调度                                                │
    │    sglang scheduler 接收 512 个并发请求                          │
    │    每个请求: 1024 input tokens + 1024 output tokens              │
    │    Continuous batching: 动态合并请求                              │
    │           ↓                                                      │
    │ 3. Prefill 阶段 (1024 tokens 并行处理)                           │
    │    for each MoE layer:                                           │
    │      ┌─ Attention (MLA) ───────────────────────────────────┐     │
    │      │  常驻 GPU, 1024 tokens 并行计算                       │     │
    │      │  KV cache 写入 sglang 的 RadixAttention 管理         │     │
    │      └────────────────────────────────────────────────────┘     │
    │      ┌─ MoE Gate ──────────────────────────────────────────┐    │
    │      │  gate_logits = W_r @ h  (1024×64 tensor)             │    │
    │      │  topk_indices = topk(sigmoid(gate_logits), k=2)      │    │
    │      │  (原 k=6, 现改为 k=2, 减少 expert 加载和计算)         │    │
    │      └────────────────────────────────────────────────────┘     │
    │      ┌─ Expert FFN (仅 top-2, EP=2) ───────────────────────┐    │
    │      │  GPU0: expert e₀,e₁ 的 FP16 权重                      │    │
    │      │  GPU1: expert e₂,e₃ 的 FP16 权重                      │    │
    │      │  各 GPU 仅计算分配给自己的 expert                      │    │
    │      │  all-reduce 聚合结果                                   │    │
    │      └────────────────────────────────────────────────────┘     │
    │      ┌─ Shared Expert (常驻 GPU) ──────────────────────────┐    │
    │      │  out += SharedExpert(h)  (ds=10944, 不参与 routing)   │    │
    │      └────────────────────────────────────────────────────┘     │
    │           ↓                                                      │
    │ 4. Decode 阶段 (逐 token autoregressive)                        │
    │    for each new token (共 1024 tokens):                         │
    │      同 prefill 的 MoE 流程，但 batch=1 per request              │
    │      sglang 的 continuous batching 将多请求的 decode 合并批处理    │
    │      na=2 时：memory I/O 减少 (仅需加载 2 个 expert 参数)         │
    │           ↓                                                      │
    │ 5. 输出: generated tokens + throughput metric (token/s)          │
    │    结果 (Table 8): na=2 相比 na=6 在 concurrency=512 时           │
    │    throughput 从 9379→10954 tok/s (16.8% 提升)                   │
    └─────────────────────────────────────────────────────────────────┘
    ```

    **expert pruning 执行全过程（ne=64→16, concurrency=512, 2×A800）**：
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. Pre-inference Expert Selection                                │
    │    Soft Count 方法：记录每个 expert 在 calibration data 上的      │
    │    激活次数，选 top-ne' 个最常用 expert 保留。                      │
    │    从 64 个 expert 中选 16 个，其余 48 个不加载。                  │
    │           ↓                                                      │
    │ 2. 模型加载（仅加载选中的 expert）                                │
    │    sglang 加载 DeepSeek-V2-Lite，但每层仅 ne'=16 个 expert        │
    │    总参数量从 16B 降至约 16×(16/64)×(routed_expert_ratio)         │
    │    GPU 显存占用减少，但 FLOPS 不变（单 token 仍需 na=6 个 expert    │
    │    计算，只是可选的 expert pool 缩小）                              │
    │           ↓                                                      │
    │ 3. 推理执行                                                      │
    │    相同 sglang 推理流程，但 expert pool 从 64→16                  │
    │    每个 expert 的计算强度 (compute intensity) 提高                 │
    │    → 低并发时 memory I/O 瓶颈缓解 → up to 2.3× speedup            │
    │    高并发时（192+）throughput 可能下降（sglang 内部策略 bug？）      │
    │           ↓                                                      │
    │ 4. 性能影响                                                      │
    │    ne'=48 (25% 减少): 最佳方法 soft count, Avg 64.2 vs 66.0       │
    │    ne'=32 (50% 减少): 最佳方法 soft count, Avg 57.8 vs 66.0       │
    │    ne'=16 (75% 减少): 最佳方法 soft count, Avg 47.8 vs 66.0       │
    │    随机选择 ne'=16/32 几乎丧失语言能力                             │
    └─────────────────────────────────────────────────────────────────┘
    ```

    **关键数据（Table 8, Figure 3a, DeepSeek-V2-Lite）**：
    | Concurrency | na=6 (baseline) | na=5 | na=4 | na=3 | na=2 | Speedup (na=2 vs 6) |
    |-------------|-----------------|------|------|------|------|----------------------|
    | 2 | 479 | 511 | 544 | 583 | 631 | 1.32× |
    | 32 | 2345 | 2412 | 2529 | 2716 | 3069 | 1.31× |
    | 128 | 5591 | 5660 | 5812 | 5960 | 6126 | 1.10× |
    | 512 | 9379 | 9783 | 9950 | 10102 | 10954 | 1.17× |
    | 768 | 9453 | 9694 | 10043 | 10249 | 10968 | 1.16× |
