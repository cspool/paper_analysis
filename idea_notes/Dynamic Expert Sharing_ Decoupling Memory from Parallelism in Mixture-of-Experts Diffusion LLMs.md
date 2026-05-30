## Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs

- baseline方法是什么？
  **Vanilla MoE dLLM 并行解码**：在 diffusion LLM 的 block-based parallel decoding 中，每个 token 通过独立的 gating/routing function 选择 Top-K experts（标准 softmax gating + TopK selection）。MoE 层输出为 MoE(x) = Σ_{i∈S} (G(x)_i / Σ_{j∈S} G(x)_j) · E_i(x)，其中 S = TopK(G(x), K)。N 个并行 token 导致 unique expert load |∪_{n=1}^N S_n| 近乎线性增长（"expert explosion"），HBM→SRAM weight fetching cost 主导延迟。

  全栈执行例子（LLaDA2.0-Mini 16B, block size 32）：
  - **算法pipeline层**: 每 token 独立 softmax gating → Top-8 selection → weighted sum of 8 expert FFNs。32 tokens × 8 experts/token → unique activated experts ≈84 per layer。
  - **系统框架层**: dInfer inference framework + Fast-dLLM KV cache（0.9 confidence threshold）。HBM 加载 84 个 expert 权重（~0.98 GB/layer MoE component），memory-bound 运行。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: PyTorch native kernels 执行 gating、softmax、topk、scatter/gather 等碎片化算子链（12+ kernels）。
  - **硬件架构层**: NVIDIA B200 GPU，HBM→SRAM bandwidth 瓶颈。Expert weight fetching cost b >> per-token compute cost a，导致 memory-bound。

  现有 expert skipping baseline（NAEE、MC-MoE）的缺陷：token-centric 优化仅减少 per-token compute（a 项），不减少 unique expert load（b 项）。在 dLLM 场景下 accuracy 严重退化（LLaDA2.0-Mini 上仅保留 ~46% relative accuracy），因为静态阈值无法适应并行 token 间多样的 gating 分布。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Dynamic Expert Sharing（DES）** 将优化从 per-token pruning 转为 sequence-level coreset selection。核心洞察：dLLM 并行解码的 token 共享语义上下文，expert 需求存在显著重叠。通过识别 compact、high-utility expert coreset C，限制所有 token 仅在 C 内路由，最大化 expert 复用。

  全栈执行例子（DES-Vote, β=0.15, LLaDA2.0-Mini 16B, block size 32）：
  - **算法pipeline层**: 
    1. **Saliency-Aware Voting**：mask 每 token 的非 Top-K router scores → 跨序列聚合投票 V_i = Σ_{n=1}^N Masked(I_{n,i}) → Top-M_core experts 组成 coreset C（M_core = β×M = 0.15×M）。
    2. **Constrained Local Routing**：每 token 从 C 中选择 Top-K experts，重新归一化 gate weights。
    3. 结果：unique experts 从 84→38（-55%），accuracy 保留 99.5%。
    对比 baseline：Token-centric expert skipping（NAEE/MC-MoE）仅跳过低分 experts 减少 compute（a 项），但每 token 独立选择意味 |∪S_n| 几乎不变；DES 通过跨 token 共享最大化了 |∪S_n| 的降低，直接减少 weight-fetching cost（b 项）。
  - **系统框架层**: dInfer + Fast-dLLM，DES 在每 MoE 层插入 coreset selection step。Memory footprint 从 0.98 GB/layer 降至 0.45 GB/layer。DES-Vote 的 β 参数（连续值）提供灵活的 budget 控制，可绕过 DES-Seq 每 token 至少 1 expert 的下限。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 自定义 fused CUDA kernel 将 12 个碎片化算子（softmax + topk + masked reduction + topk）融合为 2 个 kernel：Kernel 1 利用 register-level 计算 + atomic instructions 完成 per-token softmax/ TopK/weighted vote accumulation；Kernel 2 执行 threshold-governed final ranking。实现 6× speedup over PyTorch baseline。
  - **硬件架构层**: NVIDIA B200 GPU。MoE layer latency 降低 38.0%（LLaDA2.0-Mini），end-to-end GPU kernel time 降低 8.2-14.3%。DES-Vote 在不同 block sizes（8/16/32/64）下保持恒定低 expert count，彻底解耦了 memory overhead 与并行度的绑定关系。
