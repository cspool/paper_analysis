## Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts

- baseline方法是什么？
  - **标准 Expert Parallelism (EP)**：将 MoE 模型的 N 个 expert 等分到 P 个 GPU 上（每 GPU 持有 M=N/P 个 expert 权重）。推理/训练时，每 GPU 的 tokens 通过 router 计算 top-K expert 选择，经由 All-to-All 通信将 token 分发到对应 expert 所在的 GPU（dispatch），各 GPU 用本地 expert 权重执行 FFN 计算，再通过 All-to-All 将结果合并回原始设备（combine）。EP 假设所有 GPU 的计算负载均衡，但在实际 imbalanced routing 下，少数 expert 可能接收大多数 token，导致持有这些 expert 的 GPU 过载（高延迟 + OOM 风险），而其他 GPU 空闲。
  - 全栈执行例子（标准 EP，gpt-oss-120b MoE layer, 128 experts / K=4 激活, EP=8, 8×H200）：
    - **算法 Pipeline**：token batch → Router (u^T W_r) → top-4 gating → softmax → per-expert token assignment。MoE 输出: h = Σ g_i · SwiGLU_i(u)。标准 EP 不做任何负载均衡干预，tokens 严格按 routing indices 分发。
    - **系统框架**：PyTorch distributed + NCCL。EP=8, 每 GPU 16 experts。Alg. 1 dispatch_combine: sort routing indices → index_select to reorder tokens → All-to-All dispatch (NCCL) → 本地 Grouped-GEMM (cuBLAS, 16 experts per GPU) → All-to-All combine (NCCL reverse) → reverse_sort → reshape → sum over K。不使用任何 serving 框架（vLLM/SGLang），基于 PyTorch 原生分布式训练范式。
    - **编译框架**：论文未明确说明。PyTorch eager mode, cuBLAS GEMM。
    - **Kernel 调度**：NCCL All-to-All collective（padded 或 unpadded）+ cuBLAS GEMM kernel per expert。GEMM 效率随 B_i (per-expert token 数) 增大而提高：少量大 GEMM > 大量小 GEMM。论文 Fig. 8 显示 cuBLAS 独立 GEMM 优于 Triton fused grouped-GEMM。
    - **硬件架构**：8× NVIDIA H200 GPU，单节点 NVLink/NVSwitch 互联。B_p=32K tokens per GPU。95% token 集中在 1 个 expert 时：所有 token 汇聚到 1 个 GPU → GPU 处理时间 4.6× 慢 → peak memory 4× 增长 → 可能 OOM crash。

  - Baseline 的核心缺陷：
    1. **无负载均衡**：标准 EP 严格按 routing indices 分发 token，对 expert 负载不均衡无任何干预。极端情况下 95% token 汇聚到 1 个 GPU（持有热门 expert）。
    2. **内存不可控**：过载 GPU 的 token buffer 随不均衡度线性增长（Eq. 4），peak memory 可达平衡时的 4×，直接导致 OOM crash。
    3. **GPU 计算资源浪费**：过载 GPU 成为 straggler，决定整个 EP group 的 collective latency（max_i time-of-GPU_i），其余 GPU 计算完成后空闲等待。
    4. **Naive 缓解方案均有问题**：减小 batch size 降低吞吐量；chained gradient checkpointing 仍有硬内存上限；EPLB (Liu et al., 2024) 复制热门 expert 增加内存、仅用于推理且极端情况仍 OOM；预留额外内存 (Huang et al., 2024) 增加 CPU/GPU 内存开销。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Least-Loaded Expert Parallelism (LLEP)**：在标准 EP 的 dispatch 阶段之前，通过 LLA 贪心算法预先计算全局 optimal token-to-GPU 分配方案，将超载 GPU 的多余 token 和对应的 expert 权重（Wi）传输到负载最轻的 GPU 上执行，使所有 GPU 几乎同时完成计算。
  - 对应解决 Baseline 缺陷：
    1. **解决无负载均衡**：LLA 算法（Alg. 2+3）按 expert 负载降序分配，优先让原生 GPU 计算本地 expert 的 token（减少权重传输），超出容量 m_α 的部分溢出到 g_a+g_p 最小的 GPU。α 因子控制每 GPU 容量上限，m 控制最小 GEMM 效率阈值。最终所有 GPU 负载接近均值，消除 straggler。
    2. **解决内存不可控**：通过 m_α = α · Σl_i / P 硬限制每 GPU token 数，每个 expert 的 token 被拆分到多个 GPU 并行处理。peak memory 保持稳定（图 4 bottom row），不随不均衡度增长，最多节省 5× 内存。
    3. **解决 GPU 资源浪费**：LLEP 将计算负载均匀分布到所有 P 个 GPU，最小化 max_i time-of-GPU_i，实现 collective latency 最小化。在 batch size 越大时加速越明显（图 6a），因为大 batch 饱和各 GPU 容量后 LLA 开销被摊薄。
    4. **优于 naive 缓解方案**：不降低 batch size（反而可利用均衡内存提高 batch size）；不做 expert 复制（无额外内存）；支持训练和推理；保持 exact computation（不改变模型输出）；支持 backward pass（梯度回流）。

  - 全栈执行例子（LLEP，gpt-oss-120b MoE layer, 128 experts / K=4, EP=8, 8×H200，假设 80% token 集中到 4 个 expert）：
    - **算法 Pipeline**：MoE 数学计算与标准 EP 完全相同（exact computation）。Router → top-K gate → 但 token-to-GPU 分配不再严格遵循 routing index，而是按 LLA 计划重新分配。对每个被路由到 expert e 的 token，由 LLA 决定由哪个 GPU 执行该 expert 对该 token 的 FFN 计算。所有 expert 的输出被正确聚合且 gate weight 不变。
    - **系统框架**：PyTorch distributed + NCCL。Alg. 4 LLEP dispatch_combine: 收集全局 expert 负载 l → max(l)/mean(l) ≥ λ=1.3 → 触发 LLA → Python CPU 侧执行 Alg. 2 (LLA) + Alg. 3 (LLAS) 计算分配计划 A + 权重传输计划 W → 按 A 构建 per-GPU token chunks（含 foreign expert tokens）→ All-to-All dispatch (NCCL) → P2P 权重传输 W_i: overloaded GPU → underloaded GPU (NCCL P2P Send/Recv) → Grouped-GEMM (cuBLAS, native+foreign experts) → All-to-All combine (NCCL reverse) → reverse_sort → reshape → sum。Backward: foreign expert 梯度通过 P2P 返回原生 GPU 累加。
    - **编译框架**：论文未明确说明。PyTorch eager mode, cuBLAS GEMM。
    - **Kernel 调度**：NCCL All-to-All (tokens) + NCCL P2P (expert weights) + cuBLAS GEMM (per expert)。P2P 权重传输开销取决于 D×H 大小——hidden size 越大，每个 expert 权重传输成本越高，但同时 GEMM 效率也越高（Fig. 7b 显示 LLEP 在更大 hidden size 下加速比更高）。LLA 算法时间为 Python CPU 计算，token 量级大时开销可忽略。
    - **硬件架构**：8× NVIDIA H200 GPU，单节点 NVLink/NVSwitch。参数设定：λ=1.3, α=1, m=1024。80% token 集中在 4/128 experts → 持有这些 expert 的 GPU(s) 超载 → LLA 将多余 token + 对应 W_i 溢出到负载最轻的 GPU → 最终 8 个 GPU 负载接近 = 总 token 数/8。加速效果：MoE layer 3-5× speedup（vs 标准 EP），全模型 gpt-oss-120b 1.88× speedup。
