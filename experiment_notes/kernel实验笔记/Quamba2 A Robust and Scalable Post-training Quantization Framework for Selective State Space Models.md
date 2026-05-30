## Quamba2 A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Quamba2 实现了完整的 4-bit/8-bit CUDA kernel 栈，覆盖 Mamba1 和 Mamba2 block 的所有关键算子：(1) **4-bit/8-bit matmul kernels**：基于 CUTLASS 实现 W4A8、W4A16、W8A8 三种投影层 kernel，权重按 per-group scaling factors 重排以最大化 Tensor Core 加载吞吐，output scaling factor 融合到 input scaling: s_fused = s_X/s_Y，使得 Ȳ = s_W × s_fused × W̄ × X̄；(2) **W4A8/W4A16 fused matmul-transpose kernels**：专为 Mamba1 block 实现 fused matmul+transpose；(3) **W8A8 causal convolution kernel**：将 causal depthwise conv1d 适配为 8-bit 激活和 8-bit 权重；(4) **8-bit selective scan 和 SSD kernels**：修改 Mamba 原版 selective scan/SSD kernel，接受量化权重、8-bit 激活和对应 scaling factors，加载 8-bit cached states 以减少显存压力（latency 提升约 1.18× at L=1024）；(5) **量化 Fast Hadamard Transform kernel**：在 fast Hadamard transform CUDA kernel 中集成 scaling factor s_y，执行 ȳ^H = (1/s_y) × H_n × y，避免额外量化计算开销；(6) **4-bit/8-bit embedding kernel** 和 **4-bit/8-bit lm_head kernel**：支持 head-to-toe 量化。所有 kernel 针对 auto-regressive 推理场景优化：生成阶段（memory-bound）使用 4-bit weight kernel 减少显存带宽压力，prefill 阶段（compute-bound）使用 8-bit activation kernel 利用 Tensor Core INT8 算力。
  - 实验比较：(a) SSD kernel latency: FP16 vs INT8 activations at L=256/512/1024/2048（Table 3）；(b) W8A8/W4A8/W4A16 end-to-end TPOT/TTFT on A5000 and Orin Nano 8G（Table 5）；(c) batch size scaling TPOT b=1/32/64/128/256 on A5000（Figure 11）；(d) roofline model 分析各 bit-width 的 compute/memory bound 特性（Figure 10）；(e) TTLT vs batch size Pareto 分析（Figure 12）。

- 后端平台是什么，配置是什么。
  - NVIDIA A5000 GPU 24GB（cloud），NVIDIA Orin Nano 8G（edge）。CUDA kernel 基于 CUTLASS（Thakkar et al. 2023）实现。4-bit/8-bit matmul kernel 适配自 Xiao et al. 2023、Frantar et al. 2024（Marlin）、Zhang et al. 2024、LY 2024b/a（CUDA HGEMM/HGEMV）。Fast Hadamard Transform CUDA kernel 集成自 Dao 2024b。Causal Conv1d CUDA kernel 集成自 Dao 2024a。Selective Scan/SSD kernel 修改自 Gu and Dao 2024 / Dao and Gu 2024 官方实现。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估框架：基于 vLLM（Kwon et al. 2023）进行 serving latency 测量，集成 Quamba2 自定义量化 kernel。修改内容：(a) 将 vLLM 的投影层替换为 Quamba2 的 W4A8/W4A16/W8A8 matmul kernel；(b) SSD/selective scan kernel 修改为接受 8-bit activations + scaling factors；(c) causal conv1d kernel 修改为 W8A8；(d) embedding/lm_head 替换为 4-bit/8-bit kernel；(e) 集成量化 fast Hadamard transform。Latency profiling：warm-up iterations + 100 iterations 平均，逐 operator 记录 latency。Model size profiling：统计所有量化参数和 buffers 的显存占用。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/enyac-group/Quamba（论文声明 will be released）
  - Kernel 输入到性能输出全过程（以 W4A8 Mamba2 SSD block 为例）：
    1. **Input projection (W4A8)**：输入 u_t ∈ R^D (FP16) → per-group 量化到 8-bit → ū_t。4-bit weights W̄_in 和 per-group scales s_W 预加载到 shared memory → Tensor Core 执行 INT8 matmul: (x̄_t, B̄_t, C̄_t, Δ̄_t) = dequant(W̄_in, s_W) @ ū_t。Output scaling 融合: s_fused = s_X/s_Y，kernel 输出 Ȳ（INT8）+ s_fused（FP16）。
    2. **Online Hadamard + Sort-and-cluster**：x̄_t^H = FWHT_kernel(x̄_t)（in-place Hadamard transform, O(d log d)）。按 pre-computed cluster indices 重排 channel → 分组 → 各组 quantize: x̄_t^s[c] = clamp(round(x_t^H[c]/s_{m,n}), -127, 127)。
    3. **Causal Conv1d (W8A8)**：x̄_t^s 与 8-bit conv weight → conv1d_kernel 执行 INT8 depthwise conv: y_conv[t,c] = Σ_{k=0}^{K-1} W̄_conv[c,k] × x̄^s[t-k,c]。
    4. **SSD scan (8-bit states)**：从 HBM 加载 8-bit h_{t-1}（cached state）→ 加载 Ā_t, B̄_t^g, C̄_t^g（8-bit）→ SSD_kernel 执行: h_t = Ā_t ⊙ h_{t-1} + B̄_t^g ⊗ x̄_t^s, y_ssd = C̄_t^g ⊙ h_t → 写回 8-bit h_t 到 HBM 作为下步 cached state。8-bit memory traffic 降低约 2× vs FP16。
    5. **Output projection (W4A8)**：quantize y_ssd ⊙ SiLU(z_t) → ȳ → FWHT → ȳ^H = (1/s_y) × H_n × ȳ（Hadamard kernel 内联 scaling）。加载 4-bit W̄_out → dequant + matmul → ȳ_out。
    6. **性能输出**：kernel profiler 记录每个 operator 的 GPU time (ms) → 累加得 TTFT（prefill 1024 tokens）和 TPOT（generation per token）。Memory profiler 记录 HBM 占用：weights（4-bit）+ cached states（8-bit）+ activations + scaling factors + buffers。
