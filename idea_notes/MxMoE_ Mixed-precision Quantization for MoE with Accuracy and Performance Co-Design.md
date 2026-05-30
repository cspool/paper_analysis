## MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

- baseline方法是什么？
  - **GPTQ（weight-only uniform quantization）**：对所有 linear block 使用统一 bitwidth 的 weight-only 量化（per-group, group size 128, asymmetric min-max），配合 random Hadamard 变换预处理（incoherence processing）。不考虑 MoE block 内不同 linear block 的量化敏感度差异，也不利用 expert 激活频率差异优化计算效率。
  - **QuaRot（weight-activation uniform quantization）**：统一 4-bit weight + 4-bit activation 量化，使用 Hadamard 旋转消除 outlier。在 W4A4 下精度严重退化（DeepSeek-V2-Lite WikiText2 PPL 8.44 vs FP16 5.92，Qwen2-MoE PPL 110.66 vs FP16 5.84）。
  - **全栈执行例子（GPTQ baseline, DeepSeek-V2-Lite, RTX 4090）**：
    - 算法层：加载 FP16 MoE 模型 → 逐 MoE block 校准：128 seqs WikiText2 → 对每个 expert 的所有 linear block 统一使用 GPTQ 3.25-bit per-channel asym quantization → 得到量化权重。所有 linear block 位宽相同。
    - 系统框架层：基于 PyTorch + HuggingFace Transformers，CUDA kernel 执行 GEMM。调用 VLLM-Marlin-MoE 或 HQQ kernel 处理 low-precision GEMM。
    - 编译框架层：论文未明确说明。
    - kernel调度层：VLLM-Marlin-MoE kernel 顺序调用 Marlin kernel 处理每个 expert 的 GEMM（每次一个 expert），kernel launch overhead 和 GPU under-utilization 严重。HQQ kernel 不做 dequantization fusion，性能更差。两者均不支持混合精度——所有 expert 使用相同精度 kernel。
    - 硬件架构层：NVIDIA RTX 4090 GPU，无自定义硬件。
  - **Baseline 核心缺陷**：
    1. **统一位宽忽视 MoE 内 linear block 的异构量化敏感度**：同一 expert 内 gate_proj 和 down_proj 量化敏感度差异显著（Fig. 1a），统一位宽要么对不敏感 block 浪费 bit 预算，要么对敏感 block 精度不足。在 2.25-bit 下 GPTQ 的 Qwen1.5-MoE WikiText2 PPL 达 11.19（vs FP16 6.79），Mixtral PPL 达 5.69（vs FP16 3.88）。
    2. **不利用 expert 激活频率差异优化计算效率**：expert 激活频率差异超过 10×（Fig. 1b），部分 expert 形成的 GEMM 是 memory-bound（低激活频率，tokens 少），部分是 compute-bound（高激活频率）。统一位宽无法选择性对 memory-bound expert 用 W4A16、对 compute-bound expert 用 W8A8。
    3. **现有 low-precision kernel 缺乏混合精度支持**：VLLM-Marlin-MoE 和 HQQ 均不支持同一 MoE block 内混合精度 GEMM 并行执行，顺序处理导致 GPU under-utilization（Fig. 2）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **MxMoE = 硬件感知 linear-block 级 bitwidth 分配 + 自动混合精度 Group-GEMM kernel 生成**：
    - **解决缺陷 1（linear-block 粒度混合精度）**：
      - 对每个 expert 的每个 linear block 独立评估量化敏感度 Δ_{i,j,k}（校准集上的输出 Euclidean distance），通过 ILP 联合优化 L（输出扰动求和）和 T（tile 级执行时间求和），在内存预算 M 约束下求解最优 {x_{i,j,k}}（为每个 linear block 分配一个量化方案 k ∈ S）。
      - Linear-block 粒度 vs expert 粒度：同一 expert 内 gate_proj/down_proj 敏感度不同，expert 级分配只能折中，linear-block 级分配可针对性优化。实验验证 linear-block 分配 consistent 优于 expert 级（Table 3，DeepSeek-V2-Lite PPL 6.11 vs 6.32）。
    - **解决缺陷 2（硬件感知优化）**：
      - 目标函数 T 基于 tile 级 profiling + roofline model。expert 激活频率 f_i 影响其 GEMM shape（token 数决定 m 维度），进而影响 arithmetic intensity（A≈m），决定该 GEMM 是 memory-bound 还是 compute-bound。
      - 硬件感知分配自动对 memory-bound GEMM 分配 W4A16（减少 memory traffic）、对 compute-bound GEMM 分配 W8A8（利用 Tensor Core 高吞吐），同时保证精度约束。W4.25A15.5 在 memory-bound 下比 uniform W4A16 快 up to 25%。
    - **解决缺陷 3（自动混合精度 Group-GEMM kernel）**：
      - Micro-kernel specialization 为每种精度实现专用 CUDA device function（如 W2A16 fused dequant+bit-manip, W4A4-g128 multistage pipeline），避免 universal kernel 的性能损失（unified kernel 比 specialized 慢 13-38%，Table 6）。
      - Resource configuration 强制统一 warp count + shared memory 按最大需求分配 + k-dimension tiling (slice-K)，使不同精度 micro-kernel 可在同一 kernel launch 中水平融合。
      - Tile scheduler 使用 greedy LPT 启发式调度，消除顺序 expert 处理的 kernel launch overhead。
  - **全栈执行例子（MxMoE W5A5, Qwen1.5-MoE, RTX 4090）**：
    - 算法层：离线校准（128 seqs）→ 计算每 linear block 的 {Δ_{i,j,k}} → 统计 expert 激活频率 → ILP 求解最优 {x_{i,j,k}}（r=0.75）→ 各 linear block 按分配方案量化（randomized Hadamard transform + GPTQ）→ 激活运行时动态量化。结果：Qwen1.5-MoE W5A5 WikiText2 PPL 7.01（vs QuaRot W4A4 18.44，+11.43 PPL 提升），Avg Acc 66.72（vs QuaRot 43.47，+23.25%）。
    - 系统框架层：基于 PyTorch + CUDA/CUTLASS。MxMoE kernel generator 自动编译融合 kernel，替代 VLLM-Marlin-MoE/HQQ。
    - 编译框架层：论文未明确说明。
    - kernel调度层：自动生成混合精度 Group-GEMM kernel。一个 kernel launch 内并行处理 MoE block 的所有 expert GEMM（不同精度），tile scheduler 按 greedy LPT 分配 tile 到 SM。消除 VLLM-Marlin-MoE 的 per-expert kernel launch overhead。W5A5 比 FP16 快 3-3.4×（compute-bound），比 uniform W8A8 快 29.4%。
    - 硬件架构层：NVIDIA RTX 4090 GPU。无自定义硬件。
  - **关键设计选择映射**：
    - Baseline 缺陷1（统一位宽忽视敏感度）→ Δ_{i,j,k} 量化扰动建模 + ILP 逐 linear block 优化
    - Baseline 缺陷2（不利用激活频率差异）→ tile 级执行时间建模 T = Σ c·y·x + roofline-guided hardware-aware allocation
    - Baseline 缺陷3（kernel 缺混合精度支持）→ micro-kernel specialization + resource config + tile scheduler 自动生成
