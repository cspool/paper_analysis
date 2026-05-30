## AVX512_BF16 CPU Expert Computation Kernel for MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

AVX512_BF16 CPU Expert Computation Kernel 是一种利用 Intel AVX-512 BF16 指令集在 CPU 上高效执行 MoE expert FFN 矩阵乘法的自定义计算 kernel。Fiddler 论文中发现 PyTorch 的默认 CPU GEMM 不支持 BF16 的 AVX-512 指令（仅使用 FP32 路径），因此在 CPU 执行 expert FFN（Strategy c）时性能次优。Fiddler 手动实现了利用 VDPBF16PS 指令的 CPU kernel——该指令每周期可执行 32 个 BF16 乘加操作（MAC），相比 FP32 GEMM 提供显著吞吐提升。

在 Fiddler 的 Strategy (c) 中，此 kernel 加速 CPU 端 expert FFN 的三次矩阵乘法：(1) gate projection: [s, 4096] × [4096, 14336] → [s, 14336]；(2) up projection: 同尺寸；(3) down projection: [s, 14336] × [14336, 4096] → [s, 4096]。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

AVX512_BF16 Expert FFN Kernel 的计算流程：

```
// CPU Expert FFN with AVX512_BF16
// 输入: activation [s, 4096] (bf16), expert_weights (bf16, pinned memory)
// 输出: output [s, 4096] (bf16)

// Step 1: Gate projection
// W_gate: [4096, 14336] (bf16, 3 matrices per expert)
gate_out = avx512_bf16_matmul(activation, W_gate)   // [s, 14336]

// Step 2: Up projection
up_out = avx512_bf16_matmul(activation, W_up)       // [s, 14336]

// Step 3: SiLU activation (element-wise, FP32 for precision)
gate_act = SiLU(gate_out)  // x * sigmoid(x)

// Step 4: Gated fusion (element-wise multiply)
fused = gate_act * up_out  // [s, 14336]

// Step 5: Down projection
output = avx512_bf16_matmul(fused, W_down)          // [s, 4096]

// --- AVX512_BF16 GEMM 内部实现 (tile 分块) ---
// 每次迭代:
//   1. 从内存加载 32 个 BF16 值到 ZMM 寄存器 (512-bit)
//      _mm512_load_epi32() / _mm512_maskz_loadu_epi32()
//   2. VDPBF16PS 指令: 计算 ZMM_A (bf16) × ZMM_B (bf16) 的点积
//      _mm512_dpbf16_ps(ZMM_acc, ZMM_A, ZMM_B)
//      → 累加到 FP32 accumulator (保护数值精度)
//   3. 重复直至 tile 完成
//   4. FP32 → BF16 截断写回内存
//      _mm512_cvtneps_pbh()

// Tile size 选择:
//   - 小于 L2 cache 以最小化 cache miss
//   - 对 [s, 4096] × [4096, 14336], tile 沿 M/N/K 各维度划分
//   - 典型 tile: M_tile=64, N_tile=256, K_tile=256
```

Fiddler 的 runtime 调度中 CPU kernel 被调用的条件：

```
// Algorithm 1 中的 Strategy (c) 触发路径:
if s > 0 AND NOT is_at_gpu(i,j):
    if cpu_lat(s) <= gpu_lat(0) + trans_lat():
        // 小 s → CPU execution
        activation_cpu = cudaMemcpyAsync(act_gpu → pinned_cpu, PCIe)
        output_cpu = avx512_bf16_expert_ffn(activation_cpu, W_cpu)
        cudaMemcpyAsync(output_cpu → output_gpu, PCIe)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **硬件要求**：Intel Xeon 支持 AVX-512 BF16 的 CPU（Cooper Lake 4th Gen+ 或 AMD Zen 4 EPYC+）
  - Fiddler Env2 使用 Intel Xeon Platinum 8480+ (112 cores, Sapphire Rapids, 支持 AVX512_BF16 + AMX)
  - Fiddler Env1 使用 Intel Xeon Gold 6126 (Skylake, AVX-512 F/CD/VL/BW/DQ 但不支持 BF16 子集) — 论文未说明 Env1 是否回退到 FP32
- **性能提升**：相比 PyTorch 默认 CPU GEMM（仅 FP32），AVX512_BF16 kernel 通过 BF16 将内存带宽需求减半 + VDPBF16PS 指令的双发射能力（端口 0+5 各一条），提供约 2× 理论吞吐提升
- **开源实现**：KTransformers 和 vLLM-CPU-AVX512BF16 提供类似实现；Fiddler 代码开源在 https://github.com/efeslab/fiddler（包含自定义 CPU kernel）
- **与其他 CPU 优化的关系**：与 AMX（Intel Sapphire Rapids+, 矩阵乘法专用加速器）正交——AMX 提供更高吞吐但需要 tile 配置/释放操作，AVX512_BF16 更轻量

涉及论文标题：
- Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models
