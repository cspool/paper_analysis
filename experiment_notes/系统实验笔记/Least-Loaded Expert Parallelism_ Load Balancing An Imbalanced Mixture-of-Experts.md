## Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts

- 属于Serving调度的实现是什么？实验比较什么？
  - LLEP 提出一种新的 Expert Parallelism (EP) 负载均衡算法，核心实现是动态将超载 GPU 的多余 token 和 expert 权重路由到欠载 GPU，保证所有 GPU 在最短集体延迟内完成计算并满足内存约束。具体实现包括：
    1. **Least-Loaded Assignment (LLA) 算法（Alg. 2）**：按 expert 负载降序，对每个 expert 判断其原生 GPU 是否可容纳所有 token。若不能，则通过 LLAS（Alg. 3）子程序将多余 token 溢出（spill）到负载最轻的 GPU，直至全部 token 分配完毕。同时构建对应的权重传输计划（weight transfer plan）。
    2. **自适应阈值 λ 切换**：当全局 expert 负载的 max/mean 比值低于 λ 时，LLEP 回退到标准 EP，因为此时负载已足够均衡，LLA 的额外开销不经济。
    3. **Backward-pass 支持**：反向传播时，溢出 expert 权重的梯度会被传回其原生设备，与原生梯度累加。LLEP 保持 MoE 的精确数学计算（exact computation），不改变模型行为。
    4. **容量与效率约束**：α 因子控制每 GPU 最大 token 容量（m_α = α · Σl_i / P），m 为每个 GEMM 的最小 token 数以保持计算效率。低于 m 的溢出量被强制本地计算（避免低效的微小 GEMM）。
  - 实验比较：
    - **受控实验（Section 5.1）**：LLEP vs Standard EP，在 gpt-oss-120b、DeepSeek-V3、Kimi-K2 三种 MoE 配置下，模拟 30%-95% token 集中到 1/4/16 个 expert 的不均衡场景。指标：Forward pass speedup 和 peak memory per GPU。
    - **端到端实验（Section 5.2）**：LLEP vs Standard EP，使用真实预训练的 gpt-oss-20b 和 gpt-oss-120b，在 Megatron-Math 数据集上测量全模型吞吐量。训练实验用 gpt-oss-20b 全参数 SFT（Zero-3 + CPU offloading），指标为 AIME'25 accuracy vs wall-time。
    - **消融实验（Section 5.3）**：消融 batch size B、factor α、adaptive ratio λ、hidden size D/H、expert 数量 N 对 speedup 的影响。
    - 结果：MoE 层最高 6.1× speedup（gpt-oss-120b, 95%→1 expert），峰值内存降低 5×。端到端全模型 gpt-oss-120b 加速 1.88×，gpt-oss-20b 加速 2.2×。

- 硬件平台是什么，配置是什么。
  - **受控实验（Section 5.1）**：8× NVIDIA H200 GPU（单节点），batch size per GPU：32K tokens (gpt-oss) / 16K tokens (DeepSeek-V3, Kimi-K2)。
  - **端到端实验（Section 5.2）**：
    - gpt-oss-20b：1-8 GPU（扩展性测试），论文未明确 GPU 型号（使用与受控实验相同硬件，H200）。
    - gpt-oss-120b：多 GPU 扩展性测试。训练实验使用 Zero-3 + CPU offloading for gradients and optimizer states。
  - **消融实验（Section 5.3）**：NVIDIA H200 GPU。论文未明确说明 CPU、内存、互联（NVLink/NVSwitch）等具体配置，但使用 PyTorch NCCL 后端进行 GPU 间通信。

- 开源Serving框架是什么。修改了什么。
  - LLEP **不修改现有 Serving 框架**（如 vLLM、SGLang），而是在 PyTorch 分布式训练/推理基础上实现独立的 EP 负载均衡模块。
  - 基线框架：PyTorch 分布式（torch.distributed）+ NCCL 后端，用于标准 EP 的 All-to-All 和 P2P 通信。
  - 修改/新增内容：
    1. **LLA/LLAS 算法模块**：纯 Python 实现，接收全局 expert token 负载统计，输出 token 分配计划和权重传输计划。不修改 PyTorch autograd 或 NCCL primitive。
    2. **LLEP dispatch_combine 流程（Alg. 4）**：在标准 EP 的 dispatch-combine（Alg. 1）基础上新增：λ 阈值判断 → LLA 规划 → 构建含 foreign expert chunk 的 All-to-All → P2P 权重传输 → 对 native + foreign experts 执行 Grouped-GEMM → All-to-All reverse 结合。
    3. **权重传输**：通过 NCCL P2P（peer-to-peer）操作在 GPU 间传输 expert 权重矩阵，仅在 LLA 判定需要溢出时触发。
    4. **Backward 梯度回流**：溢出 expert 的反向梯度通过 P2P 传回原生 GPU 并与本地梯度累加，确保训练正确性。
    5. **可优化方向（论文提出但未实现）**：C++/Triton kernel 融合通信操作、避免 memory-intensive index select 的直接 All-to-All on unsorted tensors、compute/communication overlap、intra-node 优先的 spill 策略。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：代码开源在 https://github.com/SalesforceAIResearch/LeastLoadedEP
  - **LLEP 执行全过程（以 8×H200, gpt-oss-120b MoE layer, 128 experts / 4 active experts, EP=8, 每 GPU 16 experts 为例）**：

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. 输入 & Router 阶段（所有 GPU 本地执行）                       │
    │    每 GPU 持有 B_p=32K tokens，hidden dim D, K=4                 │
    │    Router 计算: s_i = softmax_i(u^T W_r) → top-K gate weights   │
    │    → 全局收集 per-expert token 负载 l ∈ R^N (含 128 experts)    │
    │           ↓                                                      │
    │ 2. 自适应判断（Alg. 4, Line 214）                               │
    │    if max(l) / mean(l) < λ (=1.3):                              │
    │      执行标准 EP (Alg. 1)，跳过 LLA                             │
    │      → 正常 dispatch-combine → 输出                             │
    │    else:                                                         │
    │           ↓                                                      │
    │ 3. Least-Loaded Assignment (Alg. 2, CPU 侧 Python 执行)          │
    │    - 按 expert 负载降序排序 l → 最大负载 expert 先分配           │
    │    - m_α = α × Σl_i / P = 1 × total_load / 8 (每GPU容量上限)    │
    │    - 对每个 expert i:                                            │
    │      · Case 1: 原生 GPU 容量足够 → 全部分配给原生 GPU            │
    │      · Case 2: 原生 GPU 部分容纳 → 容纳部分，剩余溢出到最轻GPU   │
    │      · Case 3: 原生 GPU 已满 → 全部溢出到最轻GPU                 │
    │    - LLAS (Alg. 3): 按 g_a[g]+g_p[g] 排序其他GPU, 贪心分配      │
    │    - 输出: A (token 分配计划) + W (权重传输计划)                 │
    │           ↓                                                      │
    │ 4. LLEP Dispatch（Alg. 4, Lines 217-226）                       │
    │    ┌─ Token 重排 ───────────────────────────────────────────┐   │
    │    │  按路由索引 sort → chunk B_p 和 G_p 到 per-GPU 段       │   │
    │    │  每个 GPU 的 chunk 含: native expert tokens +            │   │
    │    │  foreign expert tokens (由 LLA 分配)                     │   │
    │    └───────────────────────────────────────────────────────┘   │
    │    ┌─ All-to-All Dispatch (NCCL) ───────────────────────────┐   │
    │    │  GPU p 发送: {B_i, G_i | i ∈ assigned experts}         │   │
    │    │  GPU p 接收: {B̂_i, Ĝ_i | i ∈ [pM,(p+1)M-1] ∪ S}       │   │
    │    │  其中 S = foreign experts assigned to GPU p              │   │
    │    └───────────────────────────────────────────────────────┘   │
    │    ┌─ P2P 权重传输 (NCCL P2P) ──────────────────────────────┐  │
    │    │  for each j ∈ S (foreign experts):                      │  │
    │    │    W_j: GPU_native(j) → GPU_p (via P2P Send/Recv)       │  │
    │    │    传输量: D × H per expert (如 8192×8192=64M floats)    │  │
    │    └───────────────────────────────────────────────────────┘   │
    │           ↓                                                      │
    │ 5. Grouped-GEMM 计算（Alg. 4, Line 229）                        │
    │    ┌─ GPU p 上的 GEMMs ─────────────────────────────────────┐   │
    │    │  for expert i ∈ native experts ∪ S:                     │  │
    │    │    Ĥ_i = Ĝ_i ⊙ B̂_i W_i  (SwiGLU: 3×GEMM per expert)    │  │
    │    │  使用 cuBLAS 独立 GEMM kernel (非 fused grouped-GEMM)    │  │
    │    │  cuBLAS 的硬件优化 GEMM 比 Triton grouped-GEMM 更快     │  │
    │    └───────────────────────────────────────────────────────┘   │
    │           ↓                                                      │
    │ 6. LLEP Combine（Alg. 4, Lines 231-237）                        │
    │    ┌─ All-to-All Reverse (NCCL) ────────────────────────────┐   │
    │    │  将专家输出 {Ĥ_i} 传回各 token 的原始设备               │   │
    │    └───────────────────────────────────────────────────────┘   │
    │    ┌─ Token 还原 ───────────────────────────────────────────┐   │
    │    │  concat({H_i}) → reverse_sort → reshape(B_p, K, H)     │   │
    │    │  H'_p = sum over K dim                                 │   │
    │    └───────────────────────────────────────────────────────┘   │
    │           ↓                                                      │
    │ 7. Backward Pass（若训练）                                       │
    │    - 梯度反向传播：All-to-All reverse (forward combine)          │
    │      → expert FFN backward → All-to-All reverse (forward dispatch)│
    │    - Foreign expert 梯度: P2P 传回原生 GPU 与原生梯度累加       │
    │    - LLA 算法不在计算图中，不影响 autograd                       │
    │           ↓                                                      │
    │ 8. 输出: MoE layer 输出 H'_p                                     │
    └─────────────────────────────────────────────────────────────────┘
    ```

    **关键性能数据**：
    | 场景 | 模型 | 不平衡度 | LLEP Speedup | Memory 节省 |
    |------|------|---------|-------------|------------|
    | 受控实验 | gpt-oss-120b | 95%→1 expert | 6.11× | 5× |
    | 受控实验 | DeepSeek-V3 | 95%→1 expert | ~4.5× | ~3× |
    | 受控实验 | Kimi-K2 | 95%→1 expert | ~4× | ~3.5× |
    | 端到端 | gpt-oss-120b | 自然分布 | 1.88× | — |
    | 端到端 | gpt-oss-20b | 自然分布 | 2.2× | — |
    | 训练 (SFT) | gpt-oss-20b | 自然分布 | 1.25× | — |
