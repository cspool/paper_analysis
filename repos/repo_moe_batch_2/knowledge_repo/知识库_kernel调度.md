## Stacking Computer (Batched Gating for MoE Expert Prefetching)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Stacking Computer 是 HOBBIT / MoE-APEX 中用于加速 MoE expert 预取预测的批量 gating 计算技术。在逐层预测后续层所需 expert 时，naive 方法需要逐层执行 gating 计算（线性增长开销）。Stacking Computer 利用 gating 权重矩阵的一维为 expert 数量（通常很小：8/16/64），将所有后续层的 gating 权重堆叠成 [N, M, E] 张量（N=层数, M=d_model, E=experts），与 hidden state x ∈ R^M 做一次批量矩阵乘 matmul(x, W_stacked)，结合 top-k 选择，利用 GPU 并行性实现接近单层 gating 的计算速度。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Stacking Computer: 批量 gating 计算
// 输入: x [1, M] (当前 token hidden state)
//       W_gate[l] [M, E] (第 l 层 gating 权重, l=0..L-1)
// 输出: pred_experts[l] [K] (第 l 层预测的 top-K experts)

// Step 1: 堆叠所有后续层 gating 权重
// naive: for l in next_layers: gate_logits[l] = x @ W_gate[l]
// stacking: 一次批量计算

W_stacked = stack([W_gate[l] for l in range(cur_layer+1, L)])  // [N, M, E]
                                 // N = L - cur_layer - 1

// Step 2: 批量矩阵乘 (GPU 高度并行)
x_expanded = x.unsqueeze(0)               // [1, 1, M]
gate_logits_all = x_expanded @ W_stacked  // [1, N, E]
gate_probs_all = softmax(gate_logits_all, dim=-1)  // [1, N, E]

// Step 3: 批量 top-k 选择
pred_experts = topk(gate_probs_all, k=K, dim=-1)  // [1, N, K]

// Step 4: 自适应层数选择
// 从最近层开始，若所有 pred experts 已在 cache 则继续下一层
for l_idx in range(N):
    needed = pred_experts[0, l_idx]
    if not all_in_cache(needed):
        prefetch(needed)  // 发起预取
        break             // 或继续检查（取决于配置）
```

堆叠计算的关键：
- W_stacked 维度 [N, M, E]，M 是 hidden dimension（如 4096），E 很小（8/16/64）
- 批量 matmul 的 FLOPS 与单层类似（N 层 × E 小 → 总计算量 ≈ 单层大矩阵乘）
- GPU 上 N×E 并行执行，latency 接近常数（不随 N 线性增长）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 实现方式：在 Llama.cpp 中，将所有后续层的 gating Linear 层权重在初始化时预堆叠为扁平张量。推理时用 CUDA batched GEMV 或 GEMM（M=1 时为 GEMV batch）。
- 堆叠范围：建议 1-3 层 ahead（HOBBIT 推荐 p=1~3）。更深层预测准确率下降但仍有 ~90%，收益递减。
- 开销：堆叠操作的 overhead 在 Mixtral-8x7B 上 <0.1ms，相比 expert 加载 (~10ms) 可忽略。
- 与混合精度预取的配合：预取时同时加载低精度 expert，即使预测错误也仅浪费 1/4 带宽。

涉及论文标题：
- HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference
- MoE-APEX: An Efficient MoE Inference System with Adaptive Precision Expert Offloading

## Sparse GEMV Kernel for MoE Expert (Triton-based, Column-Major, Selective Loading)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sparse GEMV Kernel 是 FloE 提出的针对 MoE expert FFN 的自定义稀疏矩阵-向量乘法 kernel，基于 Triton 语言实现（参考 CATS kernel 修改）。核心设计：(1) 将 W_down 转置为列主序存储（W_down^T），使其列与 W_gate 的列对齐——同一 intermediate neuron 对应的 gate 列和 down 列在内存中连续，共享相同的稀疏掩码；(2) 根据 up projection 输出的幅值掩码（mask = |x @ W_up| ≥ t），选择性仅加载 W_gate 和 W_down^T 中被掩码选中的列，跳过其余列的读取；(3) 将 SiLU 激活和 element-wise multiply（Hadamard 积）融合到每个 Triton block 中执行，避免中间结果 x' 的多次 global memory 读写，减少 kernel launch 次数。在 RTX 3090 上，90% 稀疏度时单 expert 计算延迟从 0.524ms 降至 0.263ms（1.99× 加速）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// FloE Sparse GEMV Kernel 伪代码 (Triton-based, Algorithm 1)
// 输入: hidden state x [1, d_hidden=4096]
//        sparse threshold t_ij (per-expert, offline determined)
//        expert weights E_ij = {W_gate, W_down^T, W_up}
//        W_gate:  [d_hidden, d_intermediate], row-major, 仅 mask 选中列驻留 GPU
//        W_down^T: [d_hidden, d_intermediate], column-major (转置后), 仅 mask 选中列
//        W_up:    [d_hidden, d_intermediate], INT2 quantized → dequant on CPU

// GPU Kernel 入口 (单个 expert, batch=1):
function sparse_expert_gemv(x, t_ij, E_ij):
    // Step 1: up projection (全精度, 密集 GEMV)
    v = x @ W_up_deq                    // [1, d_intermediate], W_up 已解量化
    
    // Step 2: 生成稀疏掩码 (element-wise)
    mask = (|v| >= t_ij)                // bool[d_intermediate], True≈10% at 90% sparsity
    
    // Step 3: 融合 sparse gate GEMV + SiLU + Hadamard (Triton fused)
    //         每个 Triton block 处理若干选中的列
    x_prime = fused_sparse_gate(v, mask, x, W_gate_cols)
    // fused_sparse_gate 内部:
    //   for each col j where mask[j] is True:
    //       gate_j = SiLU(dot(x, W_gate[:, j]))
    //       x_prime[j] = gate_j * v[j]
    
    // Step 4: sparse down GEMV
    //         W_down^T[:, mask] 列主序, 列宽 d_hidden
    y = sparse_down_gemv(x_prime, mask, W_down_T_cols)
    // sparse_down_gemv 内部:
    //   对每个输出维度 k:
    //       y[k] = sum_{j where mask[j]} x_prime[j] * W_down[j, k]
    //              = sum_{j where mask[j]} x_prime[j] * W_down^T[k, j]
    return y

// Triton kernel 关键优化:
// 1. W_down^T 列主序: 对每个选中的列 j, W_down^T[:, j] 在内存中连续
//    配合 W_gate[:, j] 连续, 两列可一次合并读取
// 2. 列选择性加载: 通过 mask 索引数组 indptr 定位选中列
//    load(W_gate_base + indptr[j] * d_hidden) 而非遍历所有 d_intermediate 列
// 3. 融合操作: SiLU + multiply 在寄存器中完成, 无需写回 global memory
```

FloE Table 1 单 expert 执行延迟 (ms) 对比：
| GPU | Dense (0%) | 50% sparse | 70% sparse | 90% sparse |
|-----|-----------|-----------|-----------|-----------|
| RTX 3090 | 0.524 | 0.379 (1.43×) | 0.305 (1.72×) | 0.263 (1.99×) |
| A6000 | 0.524 | 0.365 (1.44×) | 0.305 (1.72×) | 0.277 (1.89×) |
| A100 | 0.253 | 0.195 (1.30×) | 0.176 (1.44×) | 0.155 (1.63×) |
| H100 | 0.253 | 0.134 (1.26×) | 0.176 (1.44×) | 0.155 (1.63×) |

注：RTX 3090 和 A6000 在 90% 稀疏度下达 ~2× 加速；H100/A100 受 kernel launch overhead 限制，高稀疏度下加速递减。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 基于 Triton (Tillet et al., 2019, https://github.com/triton-lang/triton) 实现，参考 CATS (Lee et al. 2024a) 的 sparse kernel 设计
- W_down 的转置+列主序存储是性能关键——传统 row-major 下，稀疏化使每行仅部分列有效，导致非连续访问；转置后按列加载，每列连续，内存合并效率高
- 配合 compact weights layout (gate 列 + down 列 co-locate in DRAM)，一次 PCIe 传输即可获得两列数据
- 在 consumer GPU (RTX 3090) 上稀疏加速比显著，在数据中心 GPU (H100) 上因计算吞吐极高，稀疏化的相对收益递减
- 该 kernel 的输入 x 为单个 token (batch=1)，是典型的 latency-sensitive 场景；若增加 batch size，稀疏化的不规则访存可能成为瓶颈

涉及论文标题：
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU

## Compact Asynchronous Transfer (SIMD Multi-threaded CPU-to-GPU Data Transfer)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Compact Asynchronous Transfer 是 FloE 提出的高效 CPU-to-GPU 数据传输机制，用于将 DRAM 中压缩后的 expert 权重高效传输到 GPU VRAM。包含三个协同优化：(1) **Compact Weights Layout**：在 DRAM 中将 gate projection 的列和 down projection 的对应行（转置为列）co-locate 到连续内存，chunk 大小从 d_hidden×num_bytes 增加到 2×d_hidden×num_bytes，减少内存碎片和 DMA 请求数量；(2) **AVX-512 SIMD 打包**：CPU 端使用 AVX-512 指令集并行处理多个权重 group 的打包（解量化+拷贝到 pinned memory），利用 512-bit 寄存器一次处理 16 个 FP32 或 32 个 FP16 元素；(3) **Multi-threaded + Multi-stream 异步传输**：多 CPU 线程并行打包不同 expert 的权重到 pinned memory，使用多个 CUDA stream 异步发起 cudaMemcpyAsync 传输请求，最小化 PCIe 总线空闲时间。在 RTX 3090 + PCIe 4.0 上，达到峰值带宽的 88%，比 PyTorch 原生实现快 12.6×。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Compact Asynchronous Transfer 流程
// DRAM 布局 (Co-located Gate column + Down row):
//   Expert E_ij 在 DRAM 中按 chunk 组织:
//     Chunk_k = [W_gate[:, k] | W_down^T[:, k]]  // 2 * d_hidden * num_bytes
//     (仅 mask[k]==True 的列才被组织为 chunk)
//   每个 chunk 在 DRAM 中连续, 大小 = 2 * 4096 * 2 = 16KB (FP16)

// CPU 端多线程打包 (per thread):
function pack_and_transfer(expert_indices, mask, DRAM_base, pin_buffers, streams):
    for each thread t in parallel:
        thread_chunks = partition(chunks_of_expert, t, num_threads)
        for chunk_idx in thread_chunks:
            // AVX-512 SIMD 拷贝 gate 列和 down 列到 pinned memory
            src_gate = DRAM_base + chunk_idx * chunk_size
            src_down = DRAM_base + chunk_idx * chunk_size + d_hidden * num_bytes
            dst = pin_buffers[t] + offset
            
            // AVX-512: 每次 512-bit (64 bytes) = 32 × FP16
            for i in 0..(chunk_size/64):
                _mm512_store_si512(dst + i*64, _mm512_load_si512(src_gate + i*64))
            for i in 0..(chunk_size/64):
                _mm512_store_si512(dst + chunk_size/2 + i*64, 
                                   _mm512_load_si512(src_down + i*64))
            offset += chunk_size
        
        // 异步传输该线程的 pinned buffer 到 GPU
        cudaMemcpyAsync(GPU_buf + thread_offset, pin_buffers[t],
                        thread_chunks_size, cudaMemcpyHostToDevice, streams[t])

// 传输延迟分析 (FloE Figure 7):
// Chunk size 1:  高延迟 (~1.2ms) — 大量小 DMA 请求, API/cudaLaunch overhead
// Chunk size 50: 最低延迟 (~0.37ms) — API overhead 与 DRAM 打包时间平衡
// Chunk size 200: 高延迟 (~0.52ms) — DRAM 打包时间超过传输重叠收益
// 最优 chunk size = 50 (在 FloE 的硬件配置下)

// PyTorch 原生实现:
//   for col in selected_cols:
//       gate_col = W_gate[:, col].contiguous()  // 多次小内存拷贝
//       down_col = W_down[col, :].contiguous()
//       cudaMemcpyAsync(GPU_gate + offset, gate_col, ...)
//       cudaMemcpyAsync(GPU_down + offset, down_col, ...)
//   → 大量非连续小传输, ~7% PCIe 带宽利用率
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- AVX-512: Intel CPU (Skylake-X 及更新) 支持，512-bit 向量寄存器，一条指令处理 16×FP32 或 32×FP16。在 FloE 的 64 核 CPU 上，多线程 + AVX-512 提供足够 CPU 吞吐来喂饱 PCIe 4.0 带宽
- Pinned (page-locked) memory: 使用 `cudaHostAlloc()` 或 `torch.tensor.pin_memory()` 分配，GPU DMA engine 可直接访问，无需 CPU staging。pinned memory 大小需谨慎——过大会挤占 OS 可用内存
- 多 CUDA stream: 每个 stream 维护独立的命令队列和内存拷贝引擎，允许 PCIe 传输与 GPU kernel 计算重叠。FloE 使用多 stream 使得不同 expert 的传输可并行
- 紧凑布局的 trade-off: co-locate gate 列和 down 列需要 DRAM 中重新组织权重——这可在模型加载时一次性完成（offline），不增加推理运行时开销
- 自 PyTorch 2.0+ 起，`torch.compile` 和 CUDA graphs 可以部分实现类似的传输优化，但 FloE 的手动 AVX-512 实现仍显著优于 PyTorch 原生

涉及论文标题：
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU

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

## CPU vs GPU Batching Effects in MoE Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

CPU vs GPU Batching Effects 是 Fiddler 论文通过 microbenchmark 揭示的关键性能特征差异，是其动态执行策略决策的基础。核心发现：
- **GPU 端**：expert FFN 执行延迟近乎恒定，与输入 batch size s 无关（s=1..64 范围内延迟变化 <10%）。这是因为 GPU 的并行计算能力使执行延迟受限于从显存加载参数的时间（memory-bandwidth bound），而非计算时间。
- **CPU 端**：expert FFN 执行延迟随输入 batch size s 近乎线性增长。这是因为 CPU 的计算能力远弱于 GPU，延迟受限于计算（compute-bound），参数加载时间被计算时间完全掩盖。
- **PCIe 传输**：weight copy (CPU→GPU, ~300MB/expert) 延迟恒定，是 GPU computation 的 2-5×；activation copy (GPU→CPU, s×4096×2 bytes) 延迟极小，<1% of single-input CPU latency。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Fiddler microbenchmark 的测量方法和建模结果（Appendix A, Figure 7）：

```
// Microbenchmark 测量 (32 layers × 多次, 报告 mean ± std):
// 
// GPU execution latency vs input size s:
//   GPU(s=1):  基准时间
//   GPU(s=2):  ~1.0× 基准
//   GPU(s=4):  ~1.0× 基准
//   GPU(s=8):  ~1.0× 基准
//   GPU(s=16): ~1.0× 基准
//   GPU(s=32): ~1.0× 基准
//   GPU(s=64): ~1.0× 基准
//   → Model: gpu_lat(s) = gpu_const
//   (Env1 s=1 因 PyTorch 单 batch 不同实现有 ~10% 差异，可忽略)
//
// CPU execution latency vs input size s:
//   CPU(s=1):  基准时间
//   CPU(s=2):  ~2.0× 基准
//   CPU(s=4):  ~4.0× 基准
//   CPU(s=8):  ~8.0× 基准
//   ...
//   → Model: cpu_lat(s) = cpu_slope × s
//
// PCIe transfer latency:
//   Weight copy (CPU→GPU): constant, 2-5× GPU computation
//   → Model: trans_lat() = trans_const
//
//   Activation copy (GPU→CPU): <1% of CPU(s=1)
//   → Model: ignored in latency model

// Algorithm 1 中的决策函数:
gpu_lat(s) = gpu_const                   // 恒定
cpu_lat(s) = cpu_slope × s              // 线性
trans_lat() = trans_const               // 恒定

// 决策阈值:
// cpu_lat(s) < gpu_lat(s) + trans_lat()
// → cpu_slope × s < gpu_const + trans_const
// → s < (gpu_const + trans_const) / cpu_slope = s_threshold
//
// s < s_threshold → Strategy (c): CPU 执行
// s ≥ s_threshold → Strategy (b): GPU+transfer 执行
```

延迟构成的 breakdown（以 Mixtral-8x7B expert 为例）：
| Component | Latency | Bound | s-dependence |
|-----------|---------|-------|-------------|
| GPU expert FFN | ~T_gpu | Memory BW | Constant |
| CPU expert FFN | ~s × T_cpu_per_token | Compute | Linear in s |
| PCIe weight copy | ~T_wcopy (300MB) | PCIe BW | Constant |
| PCIe activation copy | ~s × T_acopy (negligible) | PCIe BW | Linear in s (但可忽略) |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **Profiling 方法**：初始化阶段对每个 operation 运行 32 次（每层 1 次），记录 mean ± std，一次性完成
- **适用条件**：batching effects 的差异来源于 GPU 和 CPU 的架构根本差异——GPU 的 SIMT 大规模并行 vs CPU 的少量大核心——因此该特征在各类 GPU/CPU 组合中普遍成立
- **对调度的影响**：正是这种 "GPU 恒定 vs CPU 线性" 的差异使得动态策略选择有意义——若两者都是线性的或都是恒定的，则总有单一最优方案
- **Fiddler 利用方式**：在 initialization 阶段测量三个常数（gpu_const, cpu_slope, trans_const），runtime 仅需查询 s 并比较

涉及论文标题：
- Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models

## GroupedGEMM (Grouped General Matrix Multiplication)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GroupedGEMM（Grouped General Matrix Multiplication）是一种将多个不同尺寸、不同转置方式和不同缩放因子的矩阵乘法操作合并为单次 kernel launch 的批量 GEMM 操作。在 fine-grained MoE 推理中，GroupedGEMM 是 expert 层的核心计算原语：每个 expert 需要对分配给它的 token 执行独立的 FFN 矩阵乘法（W_up, W_gate, W_down），由于各 expert 接收的 token 数量不同（门控路由不均衡），这些 GEMM 操作具有不同的 M 维度（token count × d_model），但共享 K 和 N 维度（expert 权重矩阵维度）。

与普通批量 GEMM（所有子任务形状相同）相比，GroupedGEMM 的关键特性是支持**异构子任务形状**——每个 expert 的 token 数量可能不同，甚至该 expert 可能没有收到任何 token。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Fine-grained MoE 中 GroupedGEMM 的计算过程（以 Cutlass 实现为例）：

```
# GroupedGEMM for MoE expert computation
# 输入: hidden_states [total_tokens, d_model], token_to_expert mapping
# 输出: expert_outputs [total_tokens, d_model]

# Step 1: Token Routing & Grouping
for token i in range(total_tokens):
    topk_experts[i] = Router(hidden_states[i], k=6)

# Step 2: Build GroupedGEMM problem for FC1 (W_up)
groups = []
for expert_j in range(num_experts):
    tokens_for_j = [i for i where expert_j in topk_experts[i]]
    if len(tokens_for_j) > 0:
        groups.append({
            'A': hidden_states[tokens_for_j],  # [n_j, d_model]
            'B': W_up[j],                       # [d_model, d_ff]
            'C': output_up[tokens_for_j]        # [n_j, d_ff]
        })

# Step 3: Single kernel launch - all groups in parallel
cutlass_grouped_gemm(groups, alpha=1.0, beta=0.0)

# Step 4: Activation + FC2 similarly via GroupedGEMM
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

IFMoE 论文讨论了三种 GroupedGEMM 实现：
1. **Triton GroupedGEMM**：Triton 语言编写，灵活性最高但性能可能稍逊
2. **Cutlass GroupedGEMM**：NVIDIA Cutlass 库实现，IFMoE 的实际选择（因为 PyTorch 与 CUDA 12.5 版本冲突，无法使用 cuBLAS 版本）
3. **cuBLAS GroupedGEMM**：CUDA 12.5 新增的 GroupedGEMM API，预期性能最优但受限于 PyTorch/CUDA 兼容性

IFMoE 中的性能瓶颈分析：GroupedGEMM 是 memory-bound 操作——单个 expert 的 memory footprint 较小，但当 batch size 增大（激活 expert 数线性增长），总 memory pressure 上升。同时，MoE 动态路由使 Torch Compile 和 CUDA Graph 无法优化，进一步加剧延迟。

涉及论文标题：
- IFMoE: An Inference Framework Design for Fine-grained MoE

## Expert Parallelism (EP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Parallelism (EP) 是 MoE 模型分布式训练/推理的核心并行策略。在 EP 中，不同 expert 的权重被分配到不同 GPU 上（每个 GPU 持有部分 expert），token 通过 All-to-All 通信被路由到对应 expert 所在的 GPU 进行计算，结果再通过 All-to-All 返回原 GPU。EP 解决了 MoE 模型参数量巨大（如 256 experts × per-expert FFN）无法放入单 GPU 的问题，但也引入显著的 All-to-All 通信开销。

EP 在训练和推理中的关键差异：(1) 训练时通信常跨节点（多机），All-to-All 是主要瓶颈；(2) 推理时通信通常在节点内（单机多卡），NVLink 带宽充足，All-to-All 不再是首要瓶颈，反而是共享参数的每卡全量复制造成的内存浪费成为主要问题。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

经典 EP 推理流程（IFMoE 论文所述 Baseline）：

```
# EP=4, 每 GPU 持有不同 expert subset
# 每 GPU 复制完整 Attention + Norm + Shared Expert 参数

# Dispatch phase:
each GPU:
    router_outputs = Router(hidden)        # [local_tokens, num_experts]
    topk_indices = TopK(router_outputs, k=6)
    # All-to-All: send tokens to expert GPUs
    tokens_by_expert = AllToAll_Scatter(hidden, topk_indices)

# Compute phase:
each GPU:
    for expert_j in local_experts:
        expert_out[j] = ExpertFFN(tokens_by_expert[j])  # GroupedGEMM

# Combine phase:
each GPU:
    output = AllToAll_Gather(expert_out)   # return to origin GPU
    output += SharedExpert(hidden)         # dense
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

EP 的通信模型：每个 token 传输 d_model 维数据两次（dispatch + combine），总通信量 = 2 × total_tokens × d_model × sizeof(dtype)。典型实现中 EP 与 DP（数据并行）正交组合——EP group 内切分专家，DP group 内复制模型。

IFMoE 对 EP 推理的改进：用 EP+TP hybrid（共享参数用 TP 切分，expert 参数用 EP），通信从 All-to-All 改为 double All-Gather（因节点内通信带宽充足，All-Gather 通信量与 All-to-All 相当但内存效率更高）。

**Lancet 对 EP 的通信开销量化**（Lancet, MLSys 2024）：

Lancet 在 GPT-2 MoE 训练的实验中量化了 EP 的 all-to-all 瓶颈：(1) all-to-all 通信占总训练时间最高 40%；(2) all-to-all 执行时间可达 expert 计算时间的 3.36x。因此传统仅重叠 all-to-all+expert 的方案（Tutel, FasterMoE）只能隐藏 expert 计算，all-to-all 仍主导 critical path。Lancet 通过全图重叠解决此问题——前向 pipelining non-MoE 计算（self-attention, FFN）与 all-to-all 重叠，反向调度 weight gradient computation (dW) 与 all-to-all 重叠。在 A100/V100 集群上减少 non-overlapped communication 最多 77% vs Tutel，端到端加速 1.3x。

涉及论文标题：
- IFMoE: An Inference Framework Design for Fine-grained MoE
- Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping
- LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts
- Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement
- Llama 3 Meets MoE: Efficient Upcycling

**LLEP 对 EP 的核心洞察与扩展**：

LLEP 指出标准 EP 的设计假设（每 GPU 负载始终近似均衡）在实践中不成立——训练良好的 MoE 模型会表现出持续的不均衡路由（专家专业化），且这种不均衡在 domain-specific post-training 或推理中是正确的/可取的。LLEP 从系统层面缓解不均衡，而不修改模型行为：
- **Least-Loaded Assignment (LLA)**: 在 dispatch 前，通过贪心算法将超载 GPU 的多余 token + expert 权重溢出到欠载 GPU。使用容量因子 α 硬限制每 GPU token 数，最小 GEMM token 数 m 约束避免低效微小 chunk 传输。
- **自适应 λ**: 当 max(l)/mean(l) < λ（如 λ=1.3）时回退标准 EP，避免不必要的 LLA 开销。
- **Backward 支持**: foreign expert 梯度通过 P2P 传回原生 GPU 累加，支持训练。
- **性能**: MoE 层 up to 6.1× speedup, 5× memory 节省 (H200, gpt-oss-120b)；端到端 gpt-oss-120b 1.88× speedup。
- **代码**: github.com/SalesforceAIResearch/LeastLoadedEP

**Lazarus 对 EP 的扩展分析**：

Lazarus 指出传统 EP 的两个关键缺陷：(1) Expert load 不均衡——gate network 动态路由导致某些 expert 收到远多于其他 expert 的 token（up to 87% tokens routed to 2 experts），等分 expert 到 GPU 导致 GPU 间计算不均衡；(2) 无弹性——EP 要求 GPU 数为 EP size 的整数倍，故障后可能有多余 GPU 空闲。Lazarus 通过 adaptive expert replica allocation（为 popular experts 分配更多 replicas 和 GPUs）+ flexible token dispatcher（CUDA kernel 处理非对称 placement 下的 token dispatch）+ flexible all-to-all（无 padding）来解决这些缺陷，允许任意 GPU 数下完全利用所有资源。

**LatentMoE 对 EP 的分析扩展**：

LatentMoE 从 hardware-software co-design 角度量化分析 EP 在不同 deployment regime 下的瓶颈（Section 2.1-2.2, GB200 NVL72）：

Memory BW Regime（低延迟，latency-critical）：
- Per-GPU memory traffic per MoE layer: M_exp = d·m + t_exp·(d+m) per expert
- 需要 t_exp ≥ 1418 (for Qwen3-235B, d=4096, m=1536) 才进入 compute-bound
- 典型 latency-critical serving 中 t_exp ~ 数百 → firmly memory BW bound

Communication Regime（高吞吐，throughput-oriented）：
- Communication cost per GPU: M_comm = 2.5·(N/EP)·t_exp·d (FP4+BF16 mixed precision)
- Communication-to-compute ratio ≈ 9:1 for GB200 + Qwen3-235B
- All-to-All 是主要瓶颈，占总执行时间的 ~90%

LatentMoE 的解决方案：通过在 latent space ℓ 中进行 EP 的 All-to-All 通信，per-token message size 从 d 降至 ℓ = d/α，同时通过增加 K'=αK 保持总通信量不变（ℓ-MoE_acc）或降低 α 倍（ℓ-MoE_eff）。Expert 权重加载的 memory BW 从 d·m 降至 ℓ·m（降低 α×）。

**Llama 3 Meets MoE 对 EP 的实践分析**：

论文在 128-512 H100 GPU 上使用 EP=8 训练 Llama 3-E8T2，总结了 EP 的关键调优实践：
- EP 通信是每层的 All-to-All token dispatch + combine，将其保持在 NVLink 域内（单节点 8 GPU）可最小化延迟
- MoE 层 EP 性能优于 TP（expert 独立计算，EP 仅需 token dispatch），TP 更适合 Attention 层
- AllToAll-based token dispatcher 对 TopK=1-4 更高效（vs AllGather-based）
- 通过 MoE Parallel Folding 实现 Attention (TP1CP2) 和 MoE (EP8) 的异构并行映射，将 TP/CP group 折叠到 EP group 的 NVLink 域内
- 总 5-D Hybrid Parallelism: TP=1, EP=8, CP=2, PP=4, VPP=8, DP with ZeRO-1

**LSH-MoE 对 EP 通信瓶颈的分析与优化**：

LSH-MoE 在 V100 (100Gbps) 和 A100 (200Gbps) 集群上对 EP 的 all-to-all 通信瓶颈进行了详细 profiling：GPT-MoE (15B) 的 all-to-all 占训练总时间约 30%，RoBERTa-MoE 约 40%，Swin-MoE-L 约 70%，平均约 45%。LSH-MoE 的 scalability analysis 表明，该比例在更大模型和更多 GPU 下保持恒定——$\frac{T_{all\_to\_all}}{T_{compute}} = \frac{\text{FLOPs}}{6B_{inter}} \times \frac{k}{1+2k} \times \frac{w-1}{wh}$，其中 h 增长缓慢而 l 和 expert 数量增长较快。LSH-MoE 通过 LSH 聚类压缩 all-to-all 通信数据量（仅传 centroids 而非全部 tokens），实现 1.28×-2.2× 端到端加速，同时保持模型精度。

涉及论文标题：
- IFMoE: An Inference Framework Design for Fine-grained MoE
- LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts
- Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement
- Llama 3 Meets MoE: Efficient Upcycling
- LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing

## All-to-All Communication in Distributed MoE Inference

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 NCCL 中，A2A 通过 `ncclAllToAll` 或等价的 `ncclSend`/`ncclRecv` 点对点通信实现：

```
# EP=4, Top-1 MoE, 8 tokens, experts 在 GPU 0..3
# GPU 0 上有 tokens: {t0->E0, t1->E2, t2->E1, t3->E0, t4->E3, t5->E0, t6->E1, t7->E2}

# A2A Dispatch:
GPU0: send [t0,t3,t5] to GPU0(self), [t2,t6] to GPU1, [t1,t7] to GPU2, [t4] to GPU3
      recv from all GPUs: tokens routed to E0

# A2A Combine (after expert compute):
GPU0: send expert outputs back to origin GPU
      recv expert outputs from other GPUs for tokens originally on GPU0

# NCCL API (simplified):
ncclGroupStart()
for each peer:
    ncclSend(sendbuf[peer], count[peer], peer, comm, stream)
    ncclRecv(recvbuf[peer], count[peer], peer, comm, stream)
ncclGroupEnd()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- NCCL 2.21.5 提供 GPU 间的高效 A2A 实现（基于 NVLink 或网络）
- A2A 通信量：每个 token 传输 d_model 维的数据，总通信量 = num_tokens × d_model × sizeof(dtype)
- A2A 通信时间和 expert 计算时间的比率是决定 overlapping 效率的关键——FOLDMOE 通过引入 attention 计算增大可重叠的计算量
- FOLDMOE 在 100 Gbps 跨节点网络（AWS g5.48xlarge）上评估，A2A 带宽是主要瓶颈
- Figure 1 显示 A2A 曲线的斜率大于 expert computation 曲线，证明其是 scaling bottleneck

FSMoE 将 A2A 通信与节点内通信（ESP-AllGather/ESP-ReduceScatter）以及专家计算进行流水线调度。在 MP 和 ESP group 对齐节点内 GPU 数的常见配置下，节点内通信（NVLink）速度快于节点间通信（InfiniBand），因此可将 ESP-AllGather/ESP-ReduceScatter 与 AlltoAll 在不同 chunk 上重叠执行。FSMoE 使用线性模型（α+β·n/r）建模各通信操作按 pipeline degree r 切分后的耗时，通过 SLSQP 求解器确定最优 r。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism

FlowMoE 将 A2A 通信统一纳入全 Transformer block 流水线调度——A2A dispatch/combine 按层流动顺序与 MHA、gating、expert 计算交错编排（而非仅在 MoE 层内独立调度）。反向传播中 A2A 任务优先级高于 all-reduce chunk，A2A 优先执行，AR chunk 填充 A2A 间隙。在 16× RTX 3090 上 vs ScheMoE 1.14-1.28× 加速，vs Tutel 1.29-1.42× 加速。

FarSkip-Collective 在 Megatron-LM 训练中将 A2A Dispatch 和 Combine 改为异步执行（async_op=True），利用修改后的模型连接性使通信与 attention 计算和 shared expert 计算重叠。单节点 EP=8 下前向重叠率 87.6-92.9%，反向通过 Sequence Number hijacking 实现 84.1-89.0% 重叠率。仅 routed experts 和 gating 的计算不可重叠（它们依赖通信的输入/输出）。

FasterMoE（PPoPP'22）从网络拥塞角度分析了 A2A 的瓶颈：在树形拓扑的集群中，跨节点 A2A 流量 T_n = M(N-1)/N · BH 约为节点内流量 T_w = (MN-1)/MN · BH 的 M 倍，导致上层链路严重拥塞。FasterMoE 提出 grouped pairwise exchange 替代粗粒度 all-to-all，将 workers 分割为 n 个 group，形成环结构逐 stride 交换数据，并使用独立的 CUDA comm stream 和 comp stream 异步执行 S/C/R 三步操作序列，打破同步 barrier。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion
- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism

**LLEP 中的 All-to-All 使用与扩展**：

LLEP 在标准 EP 的 dispatch-combine All-to-All 基础上进行了扩展：dispatch 的 target GPU 集合从仅 native experts ([pM, (p+1)M-1]) 扩展为 native ∪ foreign experts (由 LLA 分配)。All-to-All 的 send/recv buffer 根据 LLA 分配计划 A 构建——每个 GPU 可能需要发送 tokens 到更多 target GPU（如果 LLA 将部分 expert load 溢出到多个 GPU）。关键特性：
- LLA 倾向于将溢出 load 分配给 intra-node GPU（减少跨节点通信开销）。
- Weight transfer 通过 NCCL P2P (Send/Recv) 而非 All-to-All——每次仅传输一个 expert 权重矩阵（D×H 大小）。
- All-to-All 通信量大时（大 B）效率更高，摊薄 LLA CPU 计算和 P2P 权重传输开销。图 6a 显示 LLEP speedup 随 batch size 增大而增加。

Hecate 通过拓扑感知的 token dispatching 使 All-to-All 通信时间相比 EP 减少 12.3×。其 dispatcher 优先 intra-node (NVLink/NVSwitch) 通信路径，仅在 node 内无 expert replica 时才跨 node dispatching，同时均匀分配 tokens 到多个 replica devices 防止新的 straggler。拓扑感知 dispatching 与 sparse materialization 的拓扑感知 placement search 协同工作。

HierMoE 提出 Hierarchical Token Deduplication AlltoAll (HierD-AlltoAll)，将 AlltoAll 分解为 D 维（D≤4 对应集群拓扑层数），每层按 expert group 进行 token 去重（通过 bitwise OR 聚合同一 group 内的 expert 选择）。使用线性模型 t = α + n·β 拟合 7 种 AlltoAll 变体的性能（r² > 0.997），自动选择使总通信时间最小的维度 d*。在 32-GPU A6000 集群上实现 vs Megatron-LM AlltoAll 1.99×-2.72× 加速，vs Tutel-2DH 2.34×-3.32× 加速。核心权衡：高层（小 group 数）去重收益大但低带宽 IB 传输量大；低层（大 group 数）去重收益小但 NVLink 带宽高；d* 自动选择最优折中点。

涉及论文标题：
- HierMoE: Accelerating Mixture of Experts Training with Hierarchical Token Deduplication and Expert Swap
- LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts
- Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement

**Lazarus 对 All-to-All 的扩展——Flexible All-to-All**：

Lazarus 提出 flexible all-to-all，将传统 EP 的 padded all-to-all 替换为无 padding 的 all-to-all。由于 expert replicas 的非对称放置（每个 rank 可能持有不同数量的专家 replicas），token dispatcher CUDA kernel 根据每个 rank 对每个 expert 的处理容量计算 dispatch schedule，使每个 rank 发送/接收恰好需要数量的 token（s_j tokens to rank j），无需 padding。相比传统 padded all-to-all（padding 到最大 routed expert 的 token 数），flexible all-to-all 不传输任何 padding token，在 expert load 高度不均衡时显著减少无效通信。

**LatentMoE 对 All-to-All 通信成本的建模**：

LatentMoE 从 hardware-software co-design 角度推导了 All-to-All 通信成本公式（Section 2.2），并据此指导架构设计。

通信量公式（GB200 NVL72, EP=64, FP4+BF16 mixed precision）：
- Per GPU per MoE layer: M_comm = 2.5·(N/EP)·t_exp·d
  - Factor 2.5: 0.5 bytes (FP4 dispatch) + 2 bytes (BF16 aggregation)
  - Equivalent: M_comm ∝ t_total·K·d/EP
- 通信/计算比: t_comm/t_comp = 5·F/(4·m·BW_NVL) ≈ 9 (GB200 + Qwen3-235B)
- → Communication is ~9× more expensive than expert computation

关键设计洞察：M_comm ∝ K·d，即通信量由 K 和 d 的乘积决定。LatentMoE 通过将 rounted expert 的通信从 d-space 移到 ℓ-space（per-token size ↓α×），再用增加的 K'=αK 补偿总通信量（ℓ-MoE_acc 保持通信量不变），或保持 K 不变以降低 α× 通信量（ℓ-MoE_eff）。

HybridEP 分析了 A2A 与 All-Gather (AG) 在跨 DC MoE 训练中的根本差异和转换关系。A2A 的延迟特征：$Lat_{comm}^{A2A} = \frac{D(|G^{A2A}|-1)}{B|G^{A2A}|} \approx O(1)$，即随 GPU 数增加保持恒定；AG 的延迟特征：$Lat_{comm}^{AG} = \frac{P_E(|G^{AG}|-1)}{B} \propto O(n)$，随 GPU 数线性增长。当 A2A 的一个数据块传输被转换为 AG（即本地 GPU 已通过 AG 获得目标 expert），A2A 流量减少 $\frac{D}{G}$ 而 AG 流量增加 $P_E$。这构成了 A2A↔AG 转换的 trade-off 基础。HybridEP 的 Stream-Based Modeling 通过求解 $\min_p Lat_{final}(p) = Lat_{comp} + Lat_{comm} - Lat_{ovlp}$ 得到最优混合比例 p（A2A 传输的数据比例），当 p=1 时退化为纯 A2A（标准 EP），当 p=0 时为纯 AG。在跨 DC 低带宽场景下，当 expert 参数较小（$P_E$ 相对 D 较小时），AG 替代 A2A 更为有利，因为 expert 的高可压缩性（50× via SR compression）使实际 AG 流量远小于理论值。

涉及论文标题：
- HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid ExpertData Transmission
- IFMoE: An Inference Framework Design for Fine-grained MoE
- LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing

**LSH-MoE 对训练中 All-to-All 通信的压缩方法**：

LSH-MoE 在 MoE 训练中对 All-to-All 通信采取了独特的压缩方法（与上述调度/重叠/路由优化正交且可叠加）：在通信前使用 LSH 在线聚类 token，仅传输 cluster centroids（m 个 centroids vs N 个原始 tokens，压缩率 m/N），接收端通过 residual-based error compensation 恢复近似输出。压缩率由 hash 函数数量控制（默认 6 个，约 20% 压缩率时精度无损）。通信量从 N×h 降至 m×h（Swin-MoE 中压缩率 11.7%，1.28× 加速）。该方法不修改 gate 机制或模型结构，可插入任何 EP 训练框架。

## All-Gather (AG) Communication for Expert Distribution in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

All-Gather (AG) 是一种 NCCL 集合通信原语，在 MoE 训练中用于将 expert 参数从一个 GPU 分发到一组 GPU。与 All-to-All 不同，AG 的特点是每个 GPU 的发送数据被收集到通信组中所有 GPU 上（各 GPU 输出 = 所有 GPU 输入的全集拼接）。在 HybridEP 的混合 EP 方案中，AG 替代传统的跨 DC A2A 通信来传输 expert 参数：域内 GPU 通过 AG 收集彼此的压缩 expert 参数，使得每个 GPU 都拥有域内所有 expert 的完整副本，从而消除了域内跨 GPU 的 token 数据传输（因为所有 expert 都已在本地可用）。AG 的关键优势：(1) Expert 的可压缩性——expert 权重分布比 activation data 更集中、outlier 更少，可从 P_E 压缩 50× 至 P_E/50；(2) 异步潜力——expert 不依赖 token data 即可传输，AG 通信可与 pre-expert 计算（Attention）完全重叠。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

HybridEP 中 AG 用于域内 expert 分发的伪代码：

```
# AG for Expert Distribution in HybridEP (S_ED=4, 4 DCs in Expert Domain)
# 每个 GPU 持有 1 个压缩 expert (P_E/CR), CR=50

# Asyn-comm 阶段：AG 与 pre-expert computation 重叠
# GPU i 的 CUDA stream:
stream_comm:
    # 1. SREncode 结果已在 Send Queue (与上一 iteration optimizer.step 融合)
    compressed_expert = send_queue.pop()  # 压缩后: value-index 格式, P_E/CR
    
    # 2. NCCL All-Gather: 域内所有 GPU 收集彼此的压缩 expert
    #    输入: 每个 GPU 贡献 compressed_expert_i [P_E/CR]
    #    输出: 每个 GPU 获得所有 GPU 的压缩 expert [S_ED * P_E/CR]
    all_compressed = NCCL_AllGather(compressed_expert, group=domain_group)

stream_compute:
    # 同时执行 pre-expert computation (Attention + FFN 前向)
    attn_output = attention(local_tokens)  # 与 AG 完全重叠

# 同步后:
stream_comm:
    # 3. SRDecode: 恢复完整 expert = shared_expert + decompress(residual)
    for i in range(S_ED):
        expert_i = SRDecode(all_compressed[i], shared_expert)
        recv_queue.push(expert_i)  # 供 expert FFN 使用

# Expert Computation:
for expert in recv_queue:
    output += gate_weight * expert_ffn(expert, tokens)
```

AG 通信量分析: 未压缩时 $V^{AG} = P_E * (S_{ED} - 1)$，压缩后降至 $P_E/CR * (S_{ED} - 1)$。以 Mistral-Small P_E=4.7MB, S_ED=8, CR=50: V_AG ≈ 4.7/50 * 7 ≈ 0.66MB per GPU（vs 未压缩的 32.9MB）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- NCCL 实现：`ncclAllGather(sendbuf, recvbuf, sendcount, datatype, comm, stream)`，HybridEP 调用 NCCL-2.10 实现。
- 与 NCCL All-to-All 的区别：AG 每个 GPU 的发送量小（仅自己的 expert）但接收量大（所有 GPU 的 expert 合集）；A2A 每个 GPU 的发送和接收量通常更平衡但需要更复杂的 send/recv 配对。在跨 DC 低带宽场景下，AG 的发送数据可以充分压缩（expert 残差的 Top-k），而 A2A 的 token data 无法同等程度压缩。
- 异步实现关键：利用 CUDA stream 将 AG 通信放在独立 stream 中，通过 CUDA event 同步。HybridEP 的 Asynchronous Communicator 管理 Send Queue（编码后 expert）和 Recv Queue（解码后 expert），实现 AG 通信与计算的完全异步重叠。
- AsyncEP (ZeRO-Prefill, 2026) 采用类似思路：用异步 weight AllGather 替换 per-layer activation AllToAll，weight streaming 与 prefill compute 完全重叠，在长序列 prefill 场景取得 1.35-1.37× 加速。
- SHARP (NCCL 2.27+) 支持 AllGather 的 In-Network 计算，可将 AG 延迟降低 2.5×（NVSwitch 系统），对 AG 的使用有显著利好。

涉及论文标题：
- HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid ExpertData Transmission
- IFMoE: An Inference Framework Design for Fine-grained MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Parallelism (EP) 是 MoE 模型训练的专用并行策略（Shazeer et al., 2017）。在 EP 中，MoE 层的多个专家被分配到不同的 GPU 设备上，每个 GPU 持有部分专家的参数。训练时，每个 GPU 上的 token 根据 gate 路由被 dispatch 到持有目标专家的 GPU，计算完成后结果 combine 回原 GPU。与 Data Parallelism (DP) 不同，EP 中各 GPU 持有不同的模型参数（不同专家），而非相同参数的副本。EP 的关键通信代价是 all-to-all (A2A)。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FOLDMOE 中的 EP 配置：

```
配置: EP=16 (16 GPUs, 每 GPU 1 expert)
并行组合: attention 层用 DP=2 + TP/SP=8, MoE 层用 EP=16

单 GPU 上的 MoE layer forward with EP:
while True:
    # 1. Gate 计算 (本地, 无通信)
    gate_scores = gate(local_tokens)      # [num_local_tokens, num_experts=16]
    routes = topk(gate_scores, k=1)        # 每个 token 路由到 1 个 expert

    # 2. A2A Dispatch (跨 GPU 通信)
    for each expert e:
        tokens_to_e = gather(routes == e)  # 收集路由到 expert e 的所有 token
        send(tokens_to_e, dst=gpu_of_expert[e])
    remote_tokens = recv(from all GPUs)    # 接收其他 GPU 发来需要本 GPU 处理的 token

    # 3. Expert Compute (本地, 无通信)
    output = expert(local_expert, remote_tokens, gate_scores)

    # 4. A2A Combine (跨 GPU 通信)
    for each origin GPU:
        send(output_for_tokens_from_origin, dst=origin)
    combined_output = recv(from all GPUs)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- EP 通常与 DP/TP 混合使用：DP 在节点间，TP+SP 在节点内 attention 层，EP 在所有 GPU 上的 MoE 层
- EP 通信量 = 2 × num_tokens × d_model × sizeof(dtype)（dispatch + combine 各一次）
- 扩展：每 GPU 可持有多个专家（n_expert_per_gpu > 1），以减少 A2A 通信但增加每 GPU 计算量
- FOLDMOE 每 GPU 1 expert，这是 EP 通信压力最大的配置，突出了 A2A 瓶颈
- FSMoE 在 EP 基础上结合 Expert-Sharding Parallelism (ESP)，当 GPU 数量超过 expert 数量时，每个 expert 被进一步分片到多个 GPU (ESP group)。EP 和 ESP 的组合引入了额外的 ESP-AllGather 和 ESP-ReduceScatter 通信操作，FSMoE 通过协同调度将这些额外通信与 AlltoAll 重叠，进一步提升训练效率。

FUSCO 通过融合数据变换和通信来优化 A2A 性能。FUSCO 将 A2A 通信建模为 structured segments（token 级别的逻辑单元），使用 Segment Descriptor（{addr, size} 对数组）捕获每个 segment 的源/目标内存布局。在 dComm 引擎中，A2A Dispatch 的发送端 GPU kernel 根据 descriptor 从非连续内存（expert-major layout）gather 数据到 NIC ring buffer，inline 完成 layout transformation；接收端直接 scatter 到 expert activation tensor 的最终位置。dComm 还实现了 Hierarchical Routing：对同一目的节点的多个 expert，sender 仅发送一份 token 拷贝给 forwarder GPU，forwarder 再经 intra-node NVLink 分发，消除跨节点重复传输。

- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models

HAP 对 EP 在 MoE 推理中阶段差异的分析：prefill 阶段（长序列，batch×seqlen 大），EP 的 All-to-All 通信量低于 TP 的 AllReduce——EP 仅传输路由到对应 expert 的 token 数据而非全量输出。因此 EP 在通信瓶颈场景（PCIe 低带宽如 A6000/V100）下优于 TP。但 decode 阶段（单 token），EP 的负载不均衡问题突出——热门 expert 所在 GPU 繁忙而其他 GPU 空闲，TP 因权重均匀切分无此问题。HAP 利用这一阶段差异，允许 Expert 模块在 prefill 用 EP、decode 切换为 TP，通过动态策略切换实现取两者之长、避两者之短。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models
- FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training
- HAP: Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism
- HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs
- HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid ExpertData Transmission

HybridEP 揭示了 EP 在跨数据中心（cross-DC）场景下的根本性瓶颈和一种新的解法。在跨 DC 部署中，inter-DC 带宽极低（10Gbps Ethernet vs intra-DC PCIe 128Gbps），EP 的 A2A 通信可占训练总时间的 50%-90%（Figure 2b），且通信时间（数十 ms）远超计算时间（<1ms），导致传统的计算-通信重叠策略（FasterMoE, Tutel, SmartMoE）完全失效——通信根本无法被隐藏。HybridEP 通过将 EP 改造为 Hybrid Expert/Data Transmission 来结构化解决此问题：(1) 引入 AG 通信替代部分 A2A——利用 expert parameter 的高可压缩性（50× via SR compression）和异步传输潜力，将跨 DC 的低带宽通信从 token data 转向 expert weight；(2) Stream-Based Modeling 自动决定最优的 A2A/AG 混合比例 p；(3) Domain-Based Partition 将 p 映射到层级 GPU 拓扑（域内 AG，域间 A2A）；(4) 当 p=0（纯 AG）时，标准 EP 的跨 DC A2A token 传输完全消除。本质上，HybridEP 将 EP 泛化为一种更灵活的混合通信范式——标准 EP (p=1) 只是其特例。在 1000 DC 仿真中，HybridEP 相比 EP 最高 1.45× 加速（固定 domain size），相比 Tutel/FasterMoE/SmartMoE 最高 5.6× 加速（cross-DC 低带宽训练）。

HeterMoE 揭示了 EP 在异构 GPU 集群上的局限：EP 不区分 GPU 型号，将 attention 和 expert 统一分配，导致旧 GPU（V100）也被迫执行 attention（不支持 FlashAttention，64K 时仅 A40 的 27% attention 性能）。HeterMoE 提出 Zebra Parallelism 替代 EP——ZP group 内 attention 仅在新 GPU 复制，expert 仅分布在旧 GPU，通过 microbatch-level 跨 GPU 流水线实现 overlap。

FarSkip-Collective 的工作将 EP 概念扩展到推理侧（vLLM/SGLang），使用了不同于训练的 EP 实现方式。在 vLLM/SGLang 推理 EP 中，activation 在所有 rank 上复制，仅 expert 权重按 EP 分布，使用 all-reduce（而非 all-to-all）聚合结果。FarSkip 将此 all-reduce 异步化，利用架构修改后的依赖断裂点实现通信-计算重叠（all-reduce 重叠率 95.3-97.6%）。

Hecate 量化了 EP 的 **straggler effects**：imbalanced expert load 下，最重载 device 决定了整个 MoE layer 的计算延迟（其他 device 等待），同时该 device 的入站 All-to-All 通信量也最大。在 AWS V100 cluster 上评估，相比 balanced load 分布，imbalanced load 可使训练性能下降 5.18×。Hecate 的 FSSDP 通过 SparseAllGather/SparseReduceScatter 替代 EP 的静态 expert 分布，每 iteration 从零构建临时 placement，使 expert load 的 straggler 效应被稀疏物化机制消除。在 All-to-All 层面，Hecate 的拓扑感知 dispatching（优先 intra-node）使 A2A 通信时间相比 EP 减少 12.3×。

## SparseAllGather

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SparseAllGather 是 Hecate/FSSDP 提出的稀疏通信原语，用于在 MoE 训练中从 sharded expert parameters 按需物化 (materialize) 一个临时的 expert placement。形式上，SparseAllGather 操作在 logical input buffer（划分为等大小的 chunks C = {C_0, C_1, ...}，每个 chunk = 一个 expert 的参数）上，从 pre-condition placement 𝒫₀ 转换到 post-condition placement 𝒫₁（𝒫₀ ⊆ 𝒫₁，即物化目标是 shard 的超集）。𝒫₀ 为 surjective（每个 chunk 唯一归属于某 source device）。其通信量上界 O(λS)，其中 λ = |Ĉ|/|C| 为需跨 device 通信的 expert 比例（稀疏度），S 为总参数大小。当 λ << 1 时，远小于 FSDP AllGather 的 O(S)。

在 NCCL 实现中，SparseAllGather = ncclGroupStart/End 包裹的一组 ncclBroadcast：对每个 (expert, target_device) 需要物化的对，从持有该 expert 的 source device 向需要该 expert 的所有 target devices broadcast。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
SparseAllGather NCCL 实现 (spAG(P_0, P_1)):
-------------------
输入:
  P_0: 初始分片 placement (|D| 个 source device 各持有一部分 expert)
  P_1: 目标 placement (P_0 ⊆ P_1)
  expert 参数分布在各 device 上, 每个 expert 参数大小 = expert_size

输出: P_1 中每个 device 获得所需 expert 的参数副本

NCCL 执行:
  ncclGroupStart()
  for each chunk c (expert) that needs to be materialized:
      // 需要从 source device 发送到 ≥1 个 target device
      if (c, d_target) in P_1 and (c, d_target) not in P_0:
          d_src = unique source device from P_0 holding chunk c
          // Broadcast: 从 d_src 到所有需要 c 的 target devices
          sub_comm = NCCL subgroup containing d_src and all relevant targets
          ncclBroadcast(chunk_c_data, root=d_src, comm=sub_comm)
  ncclGroupEnd()

通信量分析:
  Ĉ = {c | c 至少需发送到一个新 device}
  λ = |Ĉ| / |C|  (稀疏度, 通常 λ << 1)
  vol(spAG) = O(λ·S)  // S = |C| × expert_size
                       // 最坏: bottleneck device 接收 λ·S 数据
  vs FSDP AllGather: O(S) vs FSSDP SparseAllGather: O(λS)
  当 λ << 1: O(λS) << O(S)

Scheduling 约束:
  t = T_attn_fwd × bw / expert_size
  // t = 可在 Attention forward 时间内隐藏通信的最大 expert 数
  // spAG 延迟 ≤ T_attn_fwd → 完全重叠, 零 critical path 开销
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Hecate 用 NCCL group calls 的 Broadcast 实现 SparseAllGather。每个 Broadcast 操作针对一个 expert chunk 到一个 sub-communicator（包含 source device 和所有 target devices）。
- 通信与 Attention computation 重叠（不在 critical path 上）：Forward 中 spAG 重叠于 Attention forward。Backward 中 spAG（下一层 re-materialize）重叠于 Attention backward。
- 更高效的实现可利用数据稀疏性和网络拓扑信息（如 TACCL、GC3 等 collective synthesizer），动态生成针对当前 placement 和拓扑优化的 sparse collective 算法（留作 future work）。
- FSSDP 的 Calibration stage 可选地追加一次 on-critical-path 的 spAG：MoE gate 输出后对比预测 load vs 实际 load，若追加物化的收益 > 通信开销则执行。

涉及论文标题：
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism

## SparseReduceScatter

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SparseReduceScatter 是 Hecate/FSSDP 提出的稀疏通信原语，与 SparseAllGather 对称配对使用，用于将 MoE training backward pass 中 replicated expert 产生的 gradients reduce（求和）回持有对应 MoE shard 的 source device。形式上，spRS(𝒫₀, 𝒫₁) 从 pre-condition 𝒫₀（gradients 分布在多个 device 上）转换到 post-condition 𝒫₁（每个 chunk 的 reduce 结果唯一存在于一个 device，𝒫₁ surjective 且 𝒫₁ ⊆ 𝒫₀）。通信量上界 O(λS)，其中 λ 为需跨 device reduce 的 expert 比例。

在 NCCL 实现中，SparseReduceScatter = ncclGroupStart/End 包裹的一组 ncclReduce：对每个需 reduce 的 chunk c，在持有 c 的 replica 的所有 device 之间执行 reduce 操作，结果写入持有该 MoE shard 的 root device。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
SparseReduceScatter NCCL 实现 (spRS(P_0, P_1)):
-------------------
输入:
  P_0: gradient 分布 placement (包含 replica, 每个 expert 的梯度可能在多个 device)
  P_1: 目标 placement (每个 expert 的梯度 reduce 到唯一 source device)
  |P_0| = 总 expert gradient 副本数 (≥ |P_1|)

输出: P_1 中每个 source device 持有其 MoE shard 中 experts 的 reduced gradients

NCCL 执行:
  ncclGroupStart()
  for each chunk c (expert) that has replicas:
      if (c, d_src) in P_1:  // d_src 是该 expert 的 gradient 目标
          // Reduce: 所有持有 c 梯度副本的 device → d_src
          sub_comm = NCCL subgroup containing d_src and all devices with replica of c
          ncclReduce(chunk_c_grad_data, root=d_src, comm=sub_comm, op=SUM)
  ncclGroupEnd()

通信量分析:
  // 与 spAG 对称且等价
  vol(spRS) = O(λ·S) = vol(spAG)
  总 FSSDP 通信量 = vol(spAG) + vol(spRS) = O(2λS)

与 Rearrangement AllReduce 比较:
  // 同一 placement P' 下, rearrangement 系统需要 AllReduce 同步 DP group
  Vol(AllReduces) = Σ_i 2(|D_i|-1)/|D_i| · S/|C|
  当 |D_i| 大时 → O(2λS) = vol(spAG) + vol(spRS)
  // FSSDP 实现相同 placement 的通信量等价于 AllReduce
  // 但消除了 rearrangement 的 expert 参数+优化器状态迁移开销

Scheduling:
  spRS(layer l) + spAG(layer l+1) 同时重叠于 Attention backward
  (Attention backward 约 2× Forward → 足够隐藏两个 sparse collective)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Hecate 用 NCCL group calls 的 Reduce 实现 SparseReduceScatter。每个 spRS 调用与其对称的 spAG 配对——spAG(𝒫, 𝒫') 物化 placement，spRS(𝒫', 𝒫) 将梯度 reduce 回 source。
- 在 MoE layer backward 中，expert backward 计算完成后立即执行 spRS（可与 Attention backward 重叠，如上所述）。
- spRS 的通信量取决于 placement 的稀疏度 λ。Hecate 的 topology-aware sparse materialization (Algorithm 1) 在搜索 placement 时最小化跨 node replica，间接降低 spRS 的通信开销。

涉及论文标题：
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism

## Topology-Aware Token Dispatching (in MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Topology-Aware Token Dispatching 是 Hecate 的 Dispatcher 组件使用的 token 路由策略。当 sparse materialization 使同一个 expert 的参数在多个 device 上有副本时，每个 token 需要从多个候选 target device 中选择一个来 dispatch。Hecate 的 dispatching 算法优先 intra-node 通信路径：若 token 的 source device 所在 node 内有该 expert 的 replica，则优先选择该 node 内的 device；仅当 node 内无 replica 时才跨 node dispatch。当有多个候选 device 时，均匀分配 tokens 以平衡负载。

这种拓扑感知策略减少了跨 node 的 All-to-All 通信量，因为 inter-node 带宽（如 100 Gbps NIC）通常远低于 intra-node 带宽（如 600 GB/s NVSwitch）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Topology-Aware Dispatching (Hecate Dispatcher):
-------------------
输入:
  routes: token → expert assignment (来自 MoE gate)
  P': 当前 materialized expert placement
       (哪些 expert 在哪些 device 上有副本)
  topology: node/device 映射

输出: dispatch_plan: 每个 token → 目标 device

for each token t on source device d_src:
    expert_e = routes[t]
    candidates = {d | (expert_e, d) in P'}  // expert 的所有 replica device

    // Priority 1: 同 device
    if d_src in candidates:
        dispatch_plan[t] = d_src  // 本地计算, 零通信

    // Priority 2: 同 node (intra-node NVLink/NVSwitch)
    else if exists d_candidate in candidates where node[d_candidate] == node[d_src]:
        // 均匀分配到同 node 的候选 devices
        dispatch_plan[t] = least_loaded(candidates_in_same_node)

    // Priority 3: 跨 node (inter-node NIC)
    else:
        // 均匀分配到所有候选 devices
        dispatch_plan[t] = least_loaded(all_candidates)

// 效果: 最小化跨 node All-to-All 通信量
// Hecate 实验中 A2A 时间相比 EP 减少 12.3×
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 在 Hecate 的 Executor 中，MoE gate 输出 token assignment 后调用 Dispatcher 生成 dispatching plan。
- Dispatcher 需要 topology map（node-device 隶属关系），在 cluster setup 时建立。
- 均匀分配（least_loaded）防止同一 expert 的所有 tokens 涌入同一 device 造成新的 straggler。
- 拓扑感知 dispatching 结合 sparse materialization 的拓扑感知 placement search（Algorithm 1 优先 intra-node placement），两者协同减少跨 node 通信。

涉及论文标题：
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism

## Grouped Pairwise Exchange

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Grouped Pairwise Exchange 是 FasterMoE（PPoPP'22）提出的细粒度 all-to-all 通信调度算法，将粗粒度的同步 all-to-all 操作拆分为 n 个 group 的逐 stride pairwise exchange 序列。n 个 group 形成环结构，在第 j 步（j=0,1,...,n-1），group i 向 group (i+j) mod n 发送数据并从 group (i-j) mod n 接收数据（stride 递增）。Group 分配采用启发式：将拓扑邻近的 workers 放入同一 group，使得 group 内（stride=0）通信最快。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Grouped Pairwise Exchange in one MoE layer (forward)
# n groups of workers, arranged in a ring
# Comm stream 和 Comp stream 独立并行

# Comm stream:
for j in 0..n-1:
    S_{i,j}:  send tokens to group (i+j) mod n
              recv tokens from group (i-j) mod n

for j in 0..n-1:
    R_{i,j}:  recv expert outputs from group (i+j) mod n
              send local token outputs to group (i-j) mod n

# Comp stream (与 Comm stream 并行执行):
for j in 0..n-1:
    C_{i,j}:  compute on tokens from group (i-j) mod n using local experts

# 依赖关系:
# C_{i,j} 依赖 S_{i,j} 完成 (token 已到达)
# R_{i,j} 依赖 C_{i,j} 完成 (计算结果可用)
# S_{i,j} 完成前不能启动 C_{i,j}

# 智能调度: 最快操作放在首尾
# S_{i,0}: group内通信, 无上层连接 → 最快 → 第一位
# R_{i,n-1}: ring通信, 全带宽利用 → 第二位快 → 末位
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 FasterMoE 中基于 FastMoE 扩展实现。Group 大小通过环境变量 `FMOE_FUSE_GRAN` 控制。NCCL 用于底层 pairwise 通信。在 *johnny* 和 *trevor* 集群上评估，智能调度单独加速 1.40×，与影子化联合加速 2.20×（johnny）/ 5.72×（trevor）。理论加速上界为 `(Lat_comm + Lat_comp) / max{Lat_comm, Lat_comp}`，某些层达到理论上界的 99%。

涉及论文标题：
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models

## Computation-Communication Overlap (via Smart Scheduling)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Computation-Communication Overlap 是通过在独立 CUDA stream 上并行执行计算 kernel 和通信操作来隐藏通信延迟的系统级优化技术。FasterMoE 将其实现为 Smart Scheduling 策略：将粗粒度 all-to-all 通信拆分为 n 个 fine-grained 操作序列，在 comm stream 和 comp stream 上重新排列 S（send）、C（compute）、R（receive）操作，尊重数据依赖的同时最大化并行度。核心思想来自 DDL-Roofline 分析的结论——同步执行（半理想曲线）下 end-to-end 延迟 = Lat_comp + Lat_comm，而通过重叠执行可逼近理想曲线（P̄_ideal = P_w · min{1, R_CC}）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Smart Scheduling: 两 stream 调度 (n groups, worker i)
# 数据依赖: C_{i,j} 等 S_{i,j} 完成, R_{i,j} 等 C_{i,j} 完成

# Timeline (图 8b/c 示意, n=4):
# Comm stream:  |S0|S1|S2|S3|     |R0|R1|R2|R3|
# Comp stream:      |C0|C1|C2|C3|    (Cx等对应Sx完成)

# 对比同步执行 (图 8a):
# |S0|S1|S2|S3|C0|C1|C2|C3|R0|R1|R2|R3|

# 延迟分析 (n groups):
# 同步: ΣS_j + ΣC_j + ΣR_j
# 重叠: max(S_0 + ΣS_j, C_0) + ... + max(R_{n-1}, C_{n-1})
# 优化: 将最快的 S_{i,0}(group内通信) 和 R_{i,n-1}(ring通信) 放首尾
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 FasterMoE 中基于 FastMoE 的 CUDA stream 基础设施实现，每个 worker 创建独立的 comm stream 和 comp stream。调度逻辑按 step j 展开 S/C/R 操作序列。在 *johnny* 和 *trevor* 上实测：智能调度单独加速 1.40×，与 dynamic shadowing 联合加速 2.20×（johnny）/ 5.72×（trevor）。理论上界为 (Lat_comm + Lat_comp) / max{Lat_comm, Lat_comp}，大模型和更多 worker 下实际加速比更接近理论上界（因启动开销相对更低）。

涉及论文标题：
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models
- FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training

FlowMoE 通过 Unified Pipeline Scheduling 将重叠范围从 MoE 层内扩展到整个 Transformer block——MHA 计算与 A2A 通信重叠（Pipe-AT 贡献 +10.3%），all-reduce chunk 与 A2A 通信间隙重叠（Pipe-AR 贡献 +24.6%），使计算-通信重叠率达到全 block 级别。

## FlashAttention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlashAttention（Dao et al., 2022, 2023）是一种 IO-aware 的精确注意力计算 fused kernel。传统 attention 计算需要物化完整的 N×N attention matrix（O(N²) 内存），而 FlashAttention 通过分 tile（tiling）和在线 softmax（online softmax）技术，在不物化完整 attention matrix 的情况下计算精确的 attention 输出，将内存访问从 O(N²) 降低到 O(N)。FlashAttention-2 进一步优化了 work partitioning，减少非矩阵乘法的 FLOPs。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FlashAttention 的核心算法（简化伪代码）：

```
def flash_attention(Q, K, V):  # Q,K,V: [N, d]
    # 将 Q 分为块大小 Br 的 tile, K/V 分为块大小 Bc 的 tile
    O = zeros(N, d)
    l = zeros(N, 1)    # 归一化常数 (log-sum-exp)
    m = zeros(N, 1)    # running max

    for i in 0..Tr-1:                           # Q tiles (外循环)
        Q_i = Q[i*Br : (i+1)*Br]
        O_i = zeros(Br, d), l_i = zeros(Br, 1), m_i = -inf

        for j in 0..Tc-1:                       # K/V tiles (内循环)
            K_j = K[j*Bc : (j+1)*Bc]
            V_j = V[j*Bc : (j+1)*Bc]
            S_ij = Q_i @ K_j^T                  # [Br, Bc] on-chip
            m_ij = rowmax(S_ij)                 # running max update
            P_ij = exp(S_ij - m_ij)             # softmax numerator
            l_ij = rowsum(P_ij)                  # softmax denominator
            # 在线更新 (避免存储完整 attention matrix)
            O_i = diag(exp(m_i - m_ij)) @ O_i + P_ij @ V_j
            l_i = exp(m_i - m_ij) * l_i + l_ij
            m_i = m_ij

        O_i = diag(1/l_i) @ O_i                 # 最终归一化
        O[i*Br : (i+1)*Br] = O_i

    return O
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- FlashAttention GitHub: https://github.com/Dao-AILab/flash-attention
- 在 FOLDMOE 中，FlashAttention 用于每个 attention micro-batch 的计算，因为 micro-batch causal attention 与全序列 causal attention 产生相同的 mask pattern 和输出
- 内存节省：N×d 而非 N×N（对于 32K seqlen 节省 ~1000×）
- 速度：通常 2-4× 加速 vs 标准 attention（尤其长序列）

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining

## Tensor Parallelism (TP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Tensor Parallelism (TP, Shoeybi et al., 2019) 是 Megatron-LM 提出的模型并行策略，将单层 Transformer 内的矩阵乘法算子沿特定维度切分到多个 GPU。对于 attention 层的 QKV 投影和 FFN 层，TP 将权重矩阵按列切分，每 GPU 计算部分输出；对于后续的 output projection，按行切分，每 GPU 先本地计算再 all-reduce 求和。TP 优点是不需要 layer-wise 的通信流水线（PP 需要），但每层都需要一次 all-reduce。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Megatron-LM TP（2-way TP 简化示例）：

```
# Attention layer with TP=2 (沿列切割 W_q, W_k, W_v)
GPU0: Q0 = X @ W_q[:half_cols], K0 = X @ W_k[:half_cols], V0 = X @ W_v[:half_cols]
GPU1: Q1 = X @ W_q[half_cols:], K1 = X @ W_k[half_cols:], V1 = X @ W_v[half_cols:]

# 各自计算 attention (本地，无通信)
GPU0: Z0 = attention(Q0, K0, V0)
GPU1: Z1 = attention(Q1, K1, V1)

# Output projection (沿行切割 W_o), 需要 all-reduce
GPU0: Y0 = Z0 @ W_o[:half_rows]^T  →  Y = all_reduce(Y0 + Y1)
GPU1: Y1 = Z1 @ W_o[half_rows:]^T
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- FOLDMOE 在 attention 层使用 TP=8（intra-node，同一节点内 8 GPU）
- TP 与 FOLDMOE 的 attention-MoE pipelining 正交：TP 切分算子，FOLDMOE 沿 sequence 维度切分数据
- TP 通信为 all-reduce（通常通过 NVLink，intra-node），与 EP 的 A2A 通信（可能跨节点）独立
- FOLDMOE 将 TP 和 EP 组合使用，充分利用节点内高带宽 NVLink 和节点间网络
- HAP 对 TP 的推理性能分析：TP 在长上下文 prefill 场景下因 AllReduce 通信量 ∝ batch×seqlen 成为瓶颈——在 PCIe 低带宽（A6000/V100）下，TP 通信开销严重。TP 在短序列 decode 场景下因单 token 通信量极小且无负载不均衡，是最优策略。HAP 的 ILP 搜索在通信瓶颈场景下倾向于为 Attention 选 DP（无通信）、为 Expert prefill 选 EP（All-to-All 通信量低于 TP 的 AllReduce），在计算瓶颈场景下仍选 TP。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
- HAP: Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference
- IFMoE: An Inference Framework Design for Fine-grained MoE

## Sequence Parallelism (SP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sequence Parallelism (SP, Korthikanti et al., 2023) 是 Megatron-LM 对 Tensor Parallelism 的扩展。在 TP 中，attention 层的 LayerNorm 和 Dropout 操作在每个 GPU 上对完整的 sequence 副本执行，浪费内存。SP 将 sequence 维度沿 TP group 切分，使 LayerNorm 和 Dropout 只操作部分 token（而非完整序列），减少激活内存。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# TP+SP 组合 (2-way)
# Transformer block forward:
GPU0 (seq half 0):   GPU1 (seq half 1):
  LN(X[0:N/2])          LN(X[N/2:N])          # SP: seq 切分, 各算一半
  Dropout(...)           Dropout(...)           # SP: seq 切分
  # gather full seq for attention (通信)
  X_full = all_gather(X_half)  ←→  all_gather
  # Attention with TP (算子切分)
  Z_half = TP_Attention(X_full)                 # TP: 算子切分
  # reduce-scatter for next SP
  Z_half = reduce_scatter(Z_full)
  LN(Z_half)             LN(Z_half)             # SP: seq 切分
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- FOLDMOE 中 attention 层使用 TP+SP=8（intra-node）
- SP 仅作用于 token-wise 操作（LayerNorm、Dropout），不作用于 attention 计算和 MoE 计算
- 这意味着 SP 的 sequence 切分不影响 FOLDMOE 的 attention-MoE pipelining 数据完整性
- SP 减少激活内存，使 FOLDMOE 能在更长序列或更大模型上运行

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining

## NCCL

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

NCCL (NVIDIA Collective Communications Library) 是 NVIDIA 开发的高性能多 GPU 集合通信库，提供优化的 all-reduce、all-gather、reduce-scatter、broadcast、all-to-all 等集合通信原语。NCCL 为 NVIDIA GPU 拓扑（NVLink、NVSwitch、InfiniBand）做了专门优化，使用 ring、tree、collnet 等算法。在 MoE 训练的 EP 中，NCCL 提供 A2A 通信的实现。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FOLDMOE 训练中的 NCCL 通信流程：

```
# 训练配置: 2 nodes × 8 A10G GPUs, 100 Gbps network
# NCCL 通信分解:
# Intra-node (NVLink/NVSwitch): TP all-reduce, SP all-gather/reduce-scatter
# Inter-node (100 Gbps network): EP A2A dispatch/combine, DP all-reduce (gradients)

# MoE Layer A2A via NCCL:
for each Transformer block with MoE:
    # A2A Dispatch (inter-node bottleneck)
    ncclGroupStart()
    for peer in EP_group:
        ncclSend(tokens_for_peer, peer, stream=comm_stream)
        ncclRecv(tokens_from_peer, peer, stream=comm_stream)
    ncclGroupEnd()

    # Expert Compute (on compute stream, overlaps with above)
    ...

    # A2A Combine
    ncclGroupStart()
    for peer in EP_group:
        ncclSend(results_for_peer, peer, stream=comm_stream)
        ncclRecv(results_from_peer, peer, stream=comm_stream)
    ncclGroupEnd()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- NCCL 是 PyTorch 分布式训练的默认通信后端（通过 `torch.distributed` 调用）
- FOLDMOE 使用 NCCL 2.21.5 + CUDA 12.4
- NCCL 通信可与 CUDA kernel 在分离的 stream 上重叠（FOLDMOE 核心依赖此特性）
- 跨节点 A2A 带宽（100 Gbps）是 FOLDMOE 评估中的主要瓶颈——这也是为什么 FOLDMOE 需要 attention-MoE pipelining 来隐藏此通信
- FSMoE 使用 NCCL 2.12 + CUDA 11.3 + PyTorch 1.12，支持 4 种 AlltoAll 算法（NCCL-A2A、1DH-A2A、2DH-A2A），通过 Dispatch/Combine 子模块的抽象实现即插即用切换。FSMoE 的在线 profiler 使用 nccl-tests 微基准测量各通信原语的 α/β 参数。
- FUSCO 构建在 NCCL 2.26.3 transport 层之上，复用 NCCL 的设备注册、连接管理和底层网络协议栈（TCP/IP、InfiniBand/RoCE），在其 network abstraction layer 之上实现 Fused Data+Communication。FUSCO 约 2000 行 C++/CUDA 实现 dComm runtime（包括 on-device descriptor interpretation、pipeline coordination 和 fused communication），作为独立 collective primitive（类似 send/recv/allgather）暴露。FUSCO 的关键洞察是 NCCL 的 all-to-all 原语对 MoE token 的 logical structure 和 routing 语义无感知——它将数据视为无结构的字节流，迫使上层框架在通信前后做显式 permute/repack。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion
- LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing

## Expert-Sharding Parallelism (ESP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert-Sharding Parallelism (ESP / 专家分片并行) 是 MoE 训练中当 GPU 数量超过 expert 数量时引入的并行策略。当 P > E 时纯 EP 会导致部分 GPU 闲置。ESP 将每个 expert 的权重沿 hidden dimension 切分到 ESP group 内多张 GPU（类似 MP），使所有 GPU 参与计算。ESP 引入 ESP-AllGather（expert 计算前收集 token 分片）和 ESP-ReduceScatter（expert 计算后聚合输出并切分）两个集合通信操作。当 ESP group 对齐节点内 GPU 数时（如 8 卡 DGX），这两个操作为节点内通信（NVLink），与节点间 AlltoAll 可重叠执行。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FSMoE 中 EP+ESP 组合的 kernel 执行流程（N_ESP=4）：

```
# 输入 tokens 已通过 A2A Dispatch 到达 expert 所在 GPU group
# Step 1: ESP-AllGather (intra-node)
for gpu in ESP_group:
    local_shard = tokens_on_gpu[gpu]
    full_input = AllGather(local_shard)   # [T/N_ESP, M] → [T, M]

# Step 2: Expert Computation (各GPU算自己的权重分片)
# W1 [M, H] 沿 H 维切分为 [M, H/N_ESP]
local_out = full_input @ W1_shard        # [T, H/N_ESP]
local_out = activation(local_out)
local_out = local_out @ W2_shard         # [T, M]

# Step 3: ESP-ReduceScatter (intra-node)
combined = ReduceScatter(local_out)      # 聚合+切分 → [T/N_ESP, M]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DeepSpeed-MoE 和 Tutel 均支持 ESP。FSMoE 通过 ExpertBase 抽象支持，用户设置 N_ESP 参数，调度器自动管理通信 placement。ESP 通信量随 N_ESP 增大而增加，FSMoE 在 N_ESP=N_MP=节点内 GPU 数的最常见配置下重点优化其与 AlltoAll 的协同调度。

涉及论文标题：
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models

## ESP-AllGather / ESP-ReduceScatter

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

ESP-AllGather 和 ESP-ReduceScatter 是 ESP 引入的两个集合通信操作。ESP-AllGather 在 expert 计算前将 ESP group 内各 GPU 上的 token 分片收集到所有 GPU；ESP-ReduceScatter 在 expert 计算后将各 GPU 计算的输出分片聚合并按 token 分配切分。当 ESP group 对齐节点内 GPU 数时，此二操作为节点内通信（NVLink），与节点间 AlltoAll（InfiniBand）物理隔离，可实现无竞争重叠。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FSMoE 中 ESP-AllGather/ESP-ReduceScatter 的流水线调度时间线：

```
# r=4 chunks, 节点内通信 (ESP-AG/RS) 与节点间通信 (A2A) 重叠
t:  0    1    2    3    4    5    6    7    8
    | C0:AG|    | C1:AG|    | C2:AG|    | C3:AG|
    | C0:A2A | C1:A2A | C2:A2A | C3:A2A | GAR  |
    |    | C0:RS | C1:RS | C2:RS | C3:RS |      |
    |    | C0:Exp| C1:Exp| C2:Exp| C3:Exp|      |
```

线性模型：t_{ag,r} = α_{ag} + n_{ag}/r · β_{ag}，在 Testbed-A 上 α_ag=3.37e-1, β_ag=2.32e-6（NVLink 高带宽），对比 AlltoAll 的 β_a2a=2.21e-7（InfiniBand 低带宽），节点内通信显著快于节点间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FSMoE 通过 NCCL ncclAllGather/ncclReduceScatter 实现，在线 profiler 用 nccl-tests 微基准测量 α/β。FSMoE vs FSMoE-No-IIO 实验显示此重叠贡献约 5-6% 额外加速。

涉及论文标题：
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models

## Gradient-AllReduce in MoE Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Gradient-AllReduce 是数据并行（DP）中梯度同步的集合通信操作。在混合并行训练中，Gradient-AllReduce 为节点间通信，与 AlltoAll 共享 InfiniBand 带宽。若不加优化，会与 MoE 层的 AlltoAll 争用网络导致额外延迟。FSMoE 的自适应梯度分区将 Gradient-AllReduce 与 MoE 层协同设计，最大化隐藏梯度同步开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FSMoE 两阶段梯度分区算法：

```
# Phase 1: 贪心分配
for each layer i (from last to first):
    t_olp_i = overlappable_time(layer_i)  # MoE层空闲时间
    n_grad_i = g_grad_inv(min(t_grad(remaining), t_olp_i))
    remaining -= n_grad_i

# Phase 2: 差分进化优化剩余梯度
if remaining > 0:
    minimize Σ f_moe^i(t_grad(x_g^i))  # f_moe^i: Algorithm 1
    subject to 0 ≤ x_g^i < n_rem^i + Σ(n_rem^j - x_g^j)
```

性能模型：t_{ar}(n) = α_{ar} + n·β_{ar}，在 Testbed-A 上 α_ar=5.11e-1, β_ar=4.95e-6。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

标准实现为 PyTorch DDP 的 `all_reduce(grad, SUM)`。PipeMoE+Lina 用固定 chunk size (30MB) 切分梯度，但无法适应不同配置。FSMoE 的自适应分区根据各层 overlappable parts 的实际时间动态分配，对比 Tutel-Improved 额外加速 5-7%。

涉及论文标题：
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training

FlowMoE 将 Gradient-AllReduce 与 MoE 层的 A2A 通信协同调度：将每层 all-reduce 梯度切成 S_p 大小的 chunk，赋予低于 A2A 的优先级，在 A2A 通信间隙填充执行。BO 自动调优 S_p 以平衡重叠增益和系统开销。 (α-β Linear Model)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

α-β 线性模型将通信/计算操作耗时分解为固定启动开销 α 和每字节/每计算单元的可变开销 β：t(n) = α + n·β。按 pipeline degree r 切分后：t_r = α + n/r·β。FSMoE 使用此模型预测不同 r 下的执行时间以确定最优流水线度。拟合精度 R² > 0.998。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Profiling (训练前一次, <100s)
for msg_size in range(2^18, 24·2^18, 2^18):
    for op in [AlltoAll, AllGather, ReduceScatter, AllReduce]:
        t = nccl_test(op, msg_size)
α, β = least_squares(msg_sizes, times)  # <10ms

# 最优r求解 (4 cases, 平均193ms per config)
for c in {1,2,3,4}:
    r_c, t_c = SLSQP(minimize f_c(r), constraints=c)
r_opt = argmin(t_1, t_2, t_3, t_4)
```

Testbed-A 参数：GEMM α=4.26e-2, β=2.29e-11; AlltoAll α=2.87e-1, β=2.21e-7; AllGather α=3.37e-1, β=2.32e-6; ReduceScatter α=3.95e-1, β=2.34e-7; AllReduce α=5.11e-1, β=4.95e-6。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FSMoE profiler 基于 nccl-tests + torch.matmul 微基准，训练前执行一次。换集群时重新 profiling 一次即可。α-β 模型假设线性关系——在 FSMoE 测量范围内（2^18~12·2^19 float elements）被实验验证。SLSQP 求解器使用 scipy.optimize.minimize(method='SLSQP')，二次收敛速度。

涉及论文标题：
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models

## Segment Descriptor (FUSCO dComm)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Segment Descriptor 是 FUSCO 的 Data-Fused Communication Engine (dComm) 的核心抽象，灵感来源于操作系统虚拟内存管理中的段描述符机制。在 FUSCO 中，Segment Descriptor 是一个 {memory_address, size_in_bytes} 对，描述通信 payload（如 MoE token）在 GPU 内存中的一段连续区域的地址和大小。一个 descriptor list（descriptor 数组，连续存放于 GPU global memory）描述一次通信的所有 segments，发送端 descriptor 指定从哪些非连续内存位置 gather 数据，接收端 descriptor 指定将收到的数据 scatter 到哪些目标位置。通过这种统一元数据，dComm 可在单次传输中完成端到端的 structured data layout transformation——将 expert-major layout 的 token 直接转换为通信所需的 device-major layout，无需额外 permute kernel。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// descriptor_list 连续存放于 GPU global memory
// 每个 descriptor: {uint64_t addr, uint32_t size}
// dComm GPU Producer Kernel (per-slice launch):

__global__ void dcomm_gather_kernel(
    Descriptor* desc_list, int num_descs,
    char* ring_buffer, int slice_id, int slice_size)
{
    int slice_start = slice_id * slice_size;
    int bytes_copied = 0, cumsum = 0;
    
    for (int i = 0; i < num_descs && bytes_copied < slice_size; i++) {
        if (cumsum + desc_list[i].size <= slice_start) {
            cumsum += desc_list[i].size;
            continue;
        }
        int offset_in_seg = max(0, slice_start - cumsum);
        int to_copy = min(desc_list[i].size - offset_in_seg,
                          slice_size - bytes_copied);
        // GPU copy: non-contiguous segments → contiguous ring buffer
        // Layout transformation inline during this copy
        cudaMemcpyAsync(ring_buffer + bytes_copied,
                        (char*)desc_list[i].addr + offset_in_seg,
                        to_copy, cudaMemcpyDeviceToDevice);
        bytes_copied += to_copy;
        cumsum += desc_list[i].size;
    }
    __threadfence_system();
    *slice_ready_flag = 1;  // 通知 NIC consumer: slice 就绪
}
```

接收端 scatter 逻辑镜像上述：`desc.addr` 指向 expert activation tensor 的最终目标偏移，数据从 receive buffer 直接写入最终 layout，无需二次重排。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FUSCO 的 Communication Planner 构建两级 descriptor：
- **Node-Level**：基于 token-node 矩阵，每个 destination node 仅一份 token 拷贝（deduplication），发送端 descriptor 指向原始 token 地址，接收端 forwarder descriptor 指向 receive buffer 偏移。
- **Expert-Level**：基于 token-expert 矩阵，将 forwarder 上已收 token 的 local address 映射到各 expert GPU 上 expert activation tensor 的 exact offset。

Descriptor 数组通过累计已传输字节数定位当前 active segment（O(1)），无需端点间协调。Slice 将多个 segment 打包为较大传输单元（远大于 4-14KB 的单个 token），amortize descriptor 处理开销并确保持续填充 NIC。

涉及论文标题：
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion

## DeepEP

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

DeepEP 是 DeepSeek 团队开发的 MoE 专用通信库（https://github.com/deepseek-ai/DeepEP），为 DeepSeek-V3 训练和推理提供通信后端。构建在 NVSHMEM 之上，通过 warp specialization 和全流水线 IB-NVLink 数据路径实现高效通信。使用 NVSHMEM one-sided put/get 操作进行跨节点通信，配合 IBGDA (InfiniBand GPUDirect Async) 实现 GPU 直接访问远程内存，减少 CPU 介入和软件开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// DeepEP 通信模式 (NVSHMEM one-sided)
// Sender 直接 RDMA put 到 remote GPU 的对称内存:
nvshmem_putmem_nbi(remote_buffer, local_data, size, peer_rank);
// Receiver 不需要 recv —— 数据已由 RDMA 写入 local buffer
nvshmem_fence();
nvshmem_quiet();  // 等待所有 outstanding put 完成

// Warp specialization: SM 的 warp 分两组
//   - 通信 warp: 专职 NVSHMEM put/get + 轮询 completion
//   - 计算 warp: 在通信进行中执行 expert GEMM
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 优化与 InfiniBand、NVLink、IBGDA 紧密耦合，portability 受限
- Small message 场景（如 4K seqlen）下 NVSHMEM one-sided 开销低于 NCCL two-sided——FUSCO 在低序列长度下相对 DeepEP 优势较小
- Token deduplication 是局部和静态的，不如 FUSCO hierarchical routing 灵活
- FUSCO 在 real-world 16K seqlen 下比 DeepEP 快 1.13-1.34×，在 single-node routed 场景下快 1.95-2.01×

涉及论文标题：
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion

## NVSHMEM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

NVSHMEM 是 NVIDIA 基于 OpenSHMEM 标准的 GPU 加速通信库，为 GPU 集群提供 Partitioned Global Address Space (PGAS) 编程模型。核心原语是 one-sided put/get 操作——发送端 GPU 直接向远程 GPU 的对称内存区域写入数据，无需远程端显式参与。通过 IBGDA 实现 GPU 直接访问 RDMA-capable 网络，消除传统 two-sided MPI 通信中的 CPU 中转。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// 初始化对称内存
int *buf = (int*)nvshmem_malloc(sizeof(int) * N);
// buf 在所有 PE (Processing Element = GPU) 上有对称地址

// One-sided put: GPU→GPU direct write
nvshmem_int_put(dest, source, count, peer);
// One-sided get: GPU→GPU direct read
nvshmem_int_get(dest, source, count, peer);

// Non-blocking + 同步
nvshmem_putmem_nbi(dest, src, size, peer);  // non-blocking
nvshmem_fence();   // 确保同一 PE 的操作 ordering
nvshmem_quiet();   // 等待该 PE 的所有 outstanding 操作完成
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- One-sided 语义使 GPU 可自行发起 RDMA 传输，减少延迟 jitter——DeepEP 核心依赖此特性
- 需要显式对称内存管理（nvshmem_malloc）和 barrier/fence/quiet 同步
- 与特定硬件（InfiniBand、IBGDA）紧密耦合
- FUSCO 选择基于 NCCL 而非 NVSHMEM，以保持跨网络（TCP/IP、RoCE、IB）portability

涉及论文标题：
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion
- FlashMoE: Fast Distributed MoE in a Single Kernel
- JANUS: Disaggregating Attention and Experts for Scalable MoE Inference

## All-Reduce in MoE Inference (vLLM/SGLang EP Style)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

在 vLLM 和 SGLang 的 Expert Parallelism (EP) 推理中，MoE 层的跨 GPU 通信使用 all-reduce 而非训练中的 all-to-all（Dispatch/Combine）。这是因为推理采用"replicated activations + distributed expert weights"方式：所有 GPU 上持有输入 activation 的完整副本，但 expert 权重按 EP 分布（每个 GPU 持有 E/EP 个 expert）。各 GPU 在本地计算自己的 experts 后，通过一次 all-reduce 聚合所有 GPU 的部分结果。这种方式消除了 Dispatch/Combine 所需的 token permutation 步骤。

训练 vs 推理的 EP 通信对比：
- **训练 EP**：Dispatch all-to-all（发送 token）+ routed expert 计算 + Combine all-to-all（收集结果）
- **vLLM/SGLang 推理 EP**：本地 expert 计算 + all-reduce（聚合部分结果）

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

vLLM EP 推理 MoE 层异步化（FarSkip 实现）：

```
# 常规同步模式:
for expert_id in local_experts:
    local_out[expert_id] = ExpertMLP(activation, expert_weights[expert_id])
full_out = all_reduce(sum(local_out))  # 阻塞! GPU 空闲

# FarSkip 异步模式:
local_out = fused_moe(activation, local_expert_weights)
all_reduce_handle = all_reduce(local_out, async_op=True)  # 立即返回

# 不等待 all-reduce，继续执行 attention (FarSkip 架构保证不依赖)
attn_out = attention(hidden_states)

# 同步点——在需要完整输出之前
all_reduce_handle.wait()
output = local_out + shared_expert_out + attn_out + residual
```

CUDA Stream 层面的执行（兼容 HIP/CUDA graphs）：

```
comm_stream = torch.cuda.Stream()
with torch.cuda.stream(comm_stream):
    handle = torch.dist.all_reduce(expert_out, async_op=True)  # PyNCCL
# compute stream 继续执行 → all-reduce 与计算 overlap
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 通信量：all-reduce 仅传输 d_model 维度的 output activation（而非 full token 的 d_model × num_tokens），推理中 activation 已复制
- CUDA Graphs 兼容：使用 PyNCCL（Python NCCL C API binding）替代标准 torch.dist，支持 graph capture
- 重叠率：Llama-4 95.3%, DeepSeek-V2 97.6%（FarSkip 论文数据）

涉及论文标题：
- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models

## Asynchronous Collective Communication (async_op / CUDA Stream Overlap)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Asynchronous Collective Communication 是在 GPU 计算进行期间启动并执行集合通信操作的技术，通信在后台运行而不阻塞计算流。PyTorch 中 `handle = dist.all_reduce(tensor, async_op=True)` 启动通信后立即返回 handle，通信在 GPU 上异步执行，需要结果时 `handle.wait()` 同步。

关键机制：
- **CUDA Stream 分离**：通信 kernel 和计算 kernel 在不同 Stream 执行，GPU SM 调度器同时从多 Stream 取指令
- **通信仅占用部分 SM**：NCCL/RCCL 通信 kernel 使用部分 CUDA cores 做数据打包/解包，Tensor Cores 和大部分 CUDA Cores 仍可用于计算
- **原地完成**：all-reduce 通常是 in-place 的，结果直接写入原 tensor 内存

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FarSkip-Collective 训练中的异步 all-to-all（含 Sequence Number Hijacking）：

```
# 前向: 异步 Dispatch + Combine
class AsyncAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, group):
        handle = dist.all_to_all(output, input, group=group, async_op=True)
        ctx.handle = handle  # 存储供 backward 使用
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.empty_like(grad_output)
        handle = dist.all_to_all(grad_input, grad_output, group=ctx.group, async_op=True)
        ctx.backward_handle = handle
        return grad_input, None

# Sequence Number Hijacking (反向传播优先级重排):
# 子块计算节点: 高 Sequence Number → autograd 优先执行
# 通信输入节点: 低 Sequence Number → autograd 延后执行
# 效果: 在通信等待期间先执行子块计算，最大化重叠窗口
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- PyTorch: `dist.all_reduce(t, async_op=True)` → `Work` handle → `handle.wait()`
- CUDA Stream: `with torch.cuda.stream(comm_stream): handle = dist.all_reduce(t)`
- CUDA Graphs: 使用 PyNCCL 的 graph-compatible API (标准 torch.dist 不支持 graph capture)
- Overlap 前提：(1) 架构存在依赖断裂点；(2) 通信与计算无数据依赖；(3) 通信时间 ≤ 可重叠计算时间

涉及论文标题：
- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models

## Unified Pipeline Scheduling for MoE Training (FlowMoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Unified Pipeline Scheduling 是 FlowMoE (NeurIPS 2025) 提出的分布式 MoE 训练调度策略。其核心是将流水线调度边界从"仅 MoE 层内部"扩展到"整个 Transformer block"，统一编排 MHA 计算、gating、expert 计算、A2A 通信和 all-reduce 通信。传统方法（Tutel/ScheMoE/PipeMoE）仅对 MoE 层内的 A2A 通信和 expert 计算做 token-level 流水线重叠，MHA 和 gating 占单次迭代时间的 29.8%-36.1% 却完全串行，all-reduce 在反向传播结束后集中执行。FlowMoE 证明这些"被忽略"的任务可以通过统一流水线调度实现全重叠。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// 任务定义（每层 l，每子块 r ∈ [1, R]，R 通常=2）:
// AT_r(l): MHA + gating 计算子任务
// D_r(l):  Dispatch A2A 通信子任务
// E_r(l):  Expert 计算子任务
// C_r(l):  Combine A2A 通信子任务
// AR(l):   All-reduce 梯度（切成 S_p 大小的 chunk）

// 前向调度顺序（计算与 A2A 交错）:
// AT_1→AT_2→...→AT_R→E_1→...→E_R→AT_1(l+1)→...
// D_1→D_2→...→D_R→C_1→...→C_R→D_1(l+1)→...

// 反向调度顺序:
// E_R(l+1)→...→AT_1(l+1)→E_R(l)→...→E_1(l)→AT_R(l)→...→AT_1(l)
// C_R(l+1)→...→D_1(l+1)→C_R(l)→...→C_1(l)→D_R(l)→...→D_1(l)
// AR chunk 在 A2A 任务间隙插入（优先级: A2A > AR）

// 消融实验 (M=8192, H=8192, 16 GPU):
// Pipe-MoE only (Tutel):  1.46× vs vanillaEP
// + Pipe-AT (MHA+gating): 1.61× → MHA+gating 贡献 +10.3%
// + Pipe-AR (w/ BO):      1.82× → AR 贡献 +24.6%
// Full FlowMoE:           2.05×
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 基于 PyTorch + Tutel 实现，三个队列（DataQueue/A2AQueue/ARQueue）+ 后台通信池管理器
- 类继承扩展 Tutel 的 MoE 层，修改 token 切分和 CUDA stream 调度
- R=2 保持与 Tutel/ScheMoE 相同的流水线度，通过扩展调度范围而非增大 R 来提升重叠
- 开源: https://github.com/ZJU-CNLAB/FlowMoE
- 在 675 个自定义 MoE 层配置和 4 个真实 MoE 模型上验证，所有有效配置下均快于 ScheMoE
- 硬件: 16× RTX 3090 (100Gb/s) + 8× RTX 2080Ti (10Gb/s)

涉及论文标题：
- FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training

## Tensor Chunk-Based Priority Scheduling for All-Reduce in MoE Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlowMoE 提出的反向传播梯度 all-reduce 调度策略。将每层的 all-reduce 梯度张量切成大小为 S_p 的 chunk，放入通信任务池，赋予低于 A2A 通信的优先级。运行时仅当 A2A 队列为空时才执行 AR chunk，使 all-reduce "见缝插针"地填充 A2A 通信间隙，实现全重叠。Theorem 1 证明在 A2A 任务间隙插入 AR chunk 可减少反向传播总时间；Theorem 2 证明理想无启动开销下 S_p→0 时 per-iteration time 最小化，实际 S_p 需平衡系统开销（NCCL kernel launch、小 chunk 低带宽利用率），由 BO 自动搜索。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// 通信池管理器（后台线程）:
class CommPoolManager:
    A2AQueue: PriorityQueue    // 优先级: HIGH
    ARQueue: PriorityQueue     // 优先级: LOW

    def run():
        while training_active:
            if not A2AQueue.empty():
                execute(A2AQueue.pop())
            elif not ARQueue.empty():
                execute(ARQueue.pop())
            else:
                wait()

// All-Reduce 切分:
chunk_size = S_p  // BO 搜索, BERT-Large-MoE 上 ~2.5MB
for c in range(num_chunks):
    ARQueue.push(grad[c*S_p : (c+1)*S_p], priority=LOW)

// Timeline (反向):
// A2A: |C_1|    |D_2|    |C_2|    |D_1|
// AR:      |chk1|    |chk2|    |chk3|
// Comp: |AT_1'|AT_2'|E_1'|E_2'|
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 仅在反向传播期间激活（前向无 all-reduce 需求）
- S_p 最优值高度依赖硬件（GPU 型号、网络带宽、模型配置），不可跨集群复用
- BO 采样约 8 次（每次测 10 轮迭代平均时间），总开销 < 1% 训练迭代时间
- 硬件环境变化时需重新 profiling
- 与 DeAR (reduce-scatter + all-gather 两阶段) 正交，chunk 方法可应用于 DeAR 的两个阶段

涉及论文标题：
- FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training

## All-to-All Communication in Distributed MoE Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

All-to-All Communication 是分布式 MoE expert parallelism (EP) 的核心集合通信操作。每 MoE 层需要两轮 All-to-All：(1) Dispatch——将各 GPU 的 tokens 按 gate 选择的 expert index 发送到 hosting GPU；(2) Combine——将 expert 计算完成的输出 tokens 返回原 GPU。通信量与 token 数、hidden dimension、EP degree 成正比。GRACE-MoE 实测 multi-node 场景中 cross-node All-to-All 占单 MoE 层执行时间的 70%+，端到端延迟约 40%。Flat global All-to-All 在 heterogeneous 链路（NVLink + Ethernet）中产生 straggler effect——所有 ranks 等待最慢 link，amplifying synchronization overhead。

从kernel调度角度拆解：

```
# Flat All-to-All (NCCL alltoallv, blocking):
sent = alltoall_dispatch(local_tokens, counts_per_rank)  # GPU SM idle
expert_out = fused_moe(sent, local_experts)
combined = alltoall_combine(expert_out, counts_per_rank) # GPU SM idle

# GRACE-MoE HSC (hierarchical + sparse):
# Stage 1: cross-node sparse P2P (global group, zero-padded)
#   GPU aggregates tokens by dest node → single send
# Stage 2: intra-node NVLink redistribution (overlapped w/ routing comp)
# Combine: symmetric reverse

# 通信时间模型:
# T_flat ≈ 2 × n_token × d_model / BW_cross_node
# T_HSC ≈ n_unique_dest × d_model / BW_cross_node
#        + max(0, n_token × d_model / BW_intra_node - T_routing)
```

术语一般如何实现？如何使用？

- NCCL AlltoAll/AlltoAllv 是 PyTorch distributed 的底层实现（ring/tree topology）
- MoE 专用库：DeepEP（NVSHMEM, one-sided put/get + warp specialization）、FUSCO（transformation-communication fusion）
- EP degree 通常 = 节点数（节点内 GPU 分配给其他并行维度）
- GRACE-MoE HSC 在 Megablocks 上用 NCCL global group + manual zero-padding 实现 logical sparsity
- 优化原则：minimize cross-node traffic（dedup, affinity grouping），maximize intra-node BW utilization

涉及论文标题：
- GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference

## Hybrid TP-EP Parallelism for MoE on 3D NMP

术语解释
Hybrid TP-EP Parallelism 是 HD-MoE 提出的一种将 Tensor Parallelism (TP) 和 Expert Parallelism (EP) 混合使用的自动并行策略，专为 3D NMP 分布式架构设计。与 GPU 集群上的 Hybrid TP-EP（将 mesh 划分为子区域，区域内 TP + 区域间 EP，通过复制 hot expert 缓解不均衡）不同，HD-MoE 的 Hybrid TP-EP 允许**单个 expert 在不同节点间部分切分**（连续变量 P_ic ∈ [0,1]），high-frequency expert 使用 TP 模式分担计算负载，low-frequency expert 使用 EP 模式避免通信开销。

术语是什么？
Hybrid TP-EP Parallelism 是一种结合两种并行策略的 MoE 推理部署方法：(1) Tensor Parallelism (TP)：将单个 expert 的权重沿 intermediate dimension 切分到多个节点，各节点计算部分输出后通过 all-reduce 聚合（计算均衡，通信开销大）；(2) Expert Parallelism (EP)：将完整 expert 分配给单一节点，token 通过 all-to-all 路由到对应节点（通信少，负载不均衡）。HD-MoE 的关键创新是将 expert 分配形式化为连续变量 P_ic（expert i 在节点 c 的分配比例），使得 hot expert 可以同时分配到多个节点（TP 模式），cold expert 保持完整分配（EP 模式），且这一分配由 LP 求解器自动搜索得到。

从kernel调度角度拆解术语
HD-MoE Hybrid TP-EP 的执行伪代码：
```
# 输入：LP 求解的连续 placement matrix P_ic (E x D)
# 输入：token-to-expert activation (B tokens, each activates e experts)
# 输入：logic cluster to physical node mapping (from BO)
for each MoE layer:
    # 1. Token Dispatch (All-to-All)
    for each token t:
        for each activated expert e_i in t.experts:
            # 选择持有 expert i 的物理节点
            candidates = {c | P_ic > 0}  # 多个节点可能持有部分 expert
            target = argmin_c(load[c]) among candidates
            send(t.hidden_state, src=current_node, dst=target)
    
    # 2. Expert Computation (per node)
    for each node c in parallel:
        assigned_tokens = received_tokens[c]
        for each expert i where P_ic > 0:
            # 如果 P_ic = 1: 完整 EP 模式，本地计算完整 FFN
            # 如果 0 < P_ic < 1: TP 模式，计算 1/|holders| 的中间维度
            partial_output = expert_i.ffn(assigned_tokens, slice=P_ic)
        # 3. Result Aggregation
        for each token t:
            if expert i has P_ic < 1:  # TP 模式需要 all-reduce
                all_reduce(partial_output, group=holders_of_expert_i)
```
关键设计：P_ic 连续值允许 hot expert (高 f_i) 部分切分以平衡计算（多个节点分担），cold expert (低 f_i) 完整分配以避免通信。LP 目标函数 min(t_comp + 2γ·t̂_comm) 同时优化计算均衡和通信量。

术语一般如何实现？如何使用？
GPU 集群上的 Hybrid TP-EP（如 DeepSeek-V3 部署 DeepSeek-R1）采用 EP + hot expert replication 方式，即 EP 为主要策略，但将高频 expert 复制到多个节点以避免负载不均衡。但这在 3D NMP 上不适用（内存受限无法复制完整 expert）。HD-MoE 的连续 P_ic 方案是 3D NMP 特化的实现，通过 LP 求解器（如 PuLP、Gurobi、CPLEX）离线搜索最优 P_ic。代码开源：https://github.com/angerybob/HD-MoE

涉及论文标题：
- HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing

## α-β Communication Model for Optimal Expert Broadcast

术语解释
α-β 通信模型是分布式计算中描述 point-to-point 消息传递延迟的经典模型：发送大小为 m 的消息所需时间 = α + β·m，其中 α 是消息启动延迟（latency per message），β 是每字节传输时间的倒数（1/bandwidth）。HD-MoE 使用该模型推导 MoE expert 预广播的最优 chunk size c，以在给定 runtime window 内最大化广播效率。

术语是什么？
α-β 模型（也称 Hockney 模型或 postal model）将通信分解为两个独立成本：(1) α（latency term）：每次通信的固定开销（包括软件协议栈开销、路由建立延迟等），与消息大小无关；(2) β（bandwidth term）：每单位数据的传输时间 = 1/bandwidth。总延迟 T(m) = α + β·m。该模型适用于消息大小适中的场景，对于极小消息（α 主导）和极大消息（β 主导）需要更复杂的 LogP/logGP 模型。

从kernel调度角度拆解术语
HD-MoE 使用 α-β 模型推导 expert 预广播的最优 chunk size：
```
# 给定：expert 大小 = h·IS, mesh sqrt(D)×sqrt(D), 可广播时间窗口 k iterations
# latency  = α · (2√D + h·IS/c)  # α: per-hop + per-chunk overhead
# bandwidth = β · (h·IS + 2c√D)   # β: 1/BW
# t_pre_b = latency + bandwidth

# 下界（当 chunk size c 最优时）：
# t_pre_b ≥ h·IS·β·k + 2·α·√D + 2·√(2√D·β·k·α·h·IS)

# 最优 chunk size：
c* = √(α·h·IS / (2·β·k·√D))
```
在 batch=512, 5 TFLOPS/50 GB/s 配置下，上层推理时间允许预广播 2 个 expert；在 2.5 TFLOPS/75 GB/s 配置下可预广播 5 个 expert。预广播的 expert 被分成 c* 大小的 chunk，利用多跳路径并发传输以最小化链路拥塞。

术语一般如何实现？如何使用？
α-β 模型广泛应用于 MPI collective 通信建模（Hockney 1994）、GPU 间通信建模（NCCL 性能模型）和分布式训练通信优化（DeepSpeed、Megatron）。α 和 β 值通过 microbenchmark 测量得到：发送不同大小消息，拟合 T(m) = α + β·m 线性回归。HD-MoE 的 α 对应 NoC hop latency (~0.1-5µs per hop)，β = 1/BW (~0.013-0.04 ns/byte for 25-75 GB/s)。

涉及论文标题：
- HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing
- HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap

HierMoE 将 α-β 模型扩展到多维度分层 AlltoAll 通信建模。对于 D 维分层 AlltoAll：t_d = Σ_{i=1}^{d-1} (n_inter_i · β_inter(i) + α_inter(i)) + n_intra · β_intra(d-1) + α_intra(d-1)。通过 nccl-tests 一次性测量 7 种 AlltoAll 变体的 α, β（r² > 0.997, <300s），训练期间无需重新校准。该模型驱动最优维度 d* 选择和 expert swap 决策矩阵 Q_d*。
## Re-Index Vector (for Expert-Specific CUDA Kernels)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Re-Index Vector 是 HEXA-MoE 中为 Expert-Specific Operators（ESMM、ESS、ESTMM）CUDA kernel 提供 I/O 指导的辅助数据结构。由于 MoE 的 token 到 expert 分配是动态和不规则的，直接做 expert-wise 计算无法利用 GPU 的合并内存访问（coalesced memory access）和 Tensor Core。Re-Index Vector 通过将 routing choice 信息编码为排序后的 token index 序列，使同 expert 的 token 在逻辑上连续排列，从而在 CUDA kernel 中实现规则的内存访问模式。

构造过程：(1) 统计每个 expert 的 token 数量 ctr[e]（atomicAdd）；(2) 将 ctr[e] 向上取整到 tiling size BLK 的倍数；(3) 计算累积偏移 idx[e]（prefix sum，idx[0]=0，idx[e]=Σ_{j<e} ctr[j]）；(4) 按 routing choice 将原始 token index 写入 v[idx[R[i]]++] = i；(5) 每 expert 的 BLK 对齐尾部填充 -1（表示跳过）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Re-Index Vector 构造 (Algorithm 1, CUDA pseudocode)
Input: R [N] (routing choice, 0 ≤ R[i] < E)
Output: v [N'] (re-index vector, N' ≥ N, divisible by BLK, -1 padding)
        idx [1+E] (start index per expert, idx[E] = N')

// Step 1: 统计 per-expert token 数
ctr[0..E-1] = {0}
parallel for i = 0 to N-1:
    atomicAdd(ctr[R[i]], 1)

// Step 2: 对齐到 BLK
parallel for i = 0 to E-1:
    ctr[i] = BLK * ceil(ctr[i] / BLK)
N' = sum(ctr[0..E-1])

// Step 3: prefix sum → idx
idx[0..E-1] = prefix_sum(ctr[0..E-1])  // idx[1..E] 为各 expert 起始位置

// Step 4: 写入 token indices
parallel for i = 0 to N-1:
    pos = atomicAdd(idx[R[i]], 1)
    v[pos] = i
// v 中尾部填充位置保持 -1

// ESMM Kernel 中使用 Re-Index Vector:
parallel for i in range(0, N', BLK):    // i 步进 BLK
    exp = R[v[i]]                         // 当前 BLK 对应的 expert
    parallel for j in range(0, D2, BLK):  // 输出维度 tiling
        c = b[exp, j:j+BLK].repeat(BLK, 1)  // 加载 bias
        for k in range(0, D1, BLK):       // 输入维度 tiling
            parallel for t = 0 to BLK-1:
                xsub[t] = (v[i+t] != -1) ? x[v[i+t], k:k+BLK] : 0
            wsub = w[exp, k:k+BLK, j:j+BLK]
            c += xsub @ wsub              // Tensor Core MMA (16×16×16)
        parallel for t = 0 to BLK-1:
            if v[i+t] != -1:
                y[v[i+t], j:j+BLK] = c[t]  // 原位写回
```

Re-Index Vector 的关键作用：(1) 将同 expert 的 token 聚集到连续区域，使 thread-block 只需加载一次 expert 权重（而非每个 token 加载一次）；(2) 通过 padding -1 使所有 expert 的 token 数对齐到 BLK，保证规则的内存访问 pattern；(3) 输出按 v 中的原始 index 写回，保持与输入相同的 token 顺序。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Re-Index Vector 在每个 MoE 层的前向和反向传播中动态构建，开销为 O(N) 的 atomic 操作。由于 BLK 对齐引入了少量 padding（最多 BLK-1 per expert），padding 位置在 kernel 中通过检查 v[i+t] != -1 跳过计算和写回，不产生冗余 FLOPs。ESTMM 中两输入共享同一 re-index vector（因为它们来自同一个 ESMM 输出），进一步减少构造开销。开源实现：https://github.com/UNITES-Lab/HEXA-MoE。

涉及论文标题：
- HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy

## Expert-Specific Fused Kernel (ESFK)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert-Specific Fused Kernel (ESFK) 是 HEXA-MoE 中将 MoE 层反向传播中的三个 expert-specific 算子——ESS（expert-wise 求和）、ESTMM（expert-wise 转置矩阵乘法）、ESMM（expert-wise 矩阵乘法）——融合为单一 CUDA kernel 的技术。动机：三个算子在 backward pass 中计算不同梯度（ESS 计算 bias 梯度、ESTMM 计算 weight 梯度、ESMM 计算 input 梯度），独立启动 kernel 会产生多次 global memory 读写和 kernel launch overhead。ESFK 通过统一 thread-block shape 和扩展 thread-grid 维度将三者融合，使单 MoE 层 backward 仅需 2 个 fused kernels + 1 个 element-wise dot product。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

融合策略（Table 6）：
- 各算子原始 thread-block shape 统一为 (WARP, TIMES)
- ESS 二维 grid (E, ⌈D/(TIMES·BLK)⌉) 扩展第三维为 1
- ESMM 二维 grid (⌈N'/BLK⌉, ⌈D/(TIMES·BLK)⌉) 扩展第三维为 1
- ESTMM 三维 grid (E, ⌈D1/(TIMES·BLK)⌉, ⌈D2/(TIMES·BLK)⌉)
- ESFK 聚合 grid: dim-3 = ⌈N'/BLK⌉ + ⌈D2/BLK⌉ + ⌈D2/(TIMES·BLK)⌉

```
// ESFK 执行流程:
__global__ void ESFK(x, y1, ∂ℓ/∂y, W1, W2, R, v, idx, ...):
    gid_z = blockIdx.z
    
    if gid_z < ⌈N'/BLK⌉:
        // ESMM: 计算 ∂ℓ/∂x (input gradient)
        ESMM_block(x, ∂ℓ/∂y1, W1^T, R, v, ...)
    
    elif gid_z < ⌈N'/BLK⌉ + ⌈D2/BLK⌉:
        // ESS: 计算 ∂ℓ/∂b (bias gradient)
        ESS_block(∂ℓ/∂y, R, v, idx, ...)
    
    else:
        // ESTMM: 计算 ∂ℓ/∂W (weight gradient)
        ESTMM_block(y1, ∂ℓ/∂y, R, v, idx, ...)
```

一次 kernel launch 完成三种梯度计算，消除 kernel launch overhead 和中间结果的 global memory 往返。消融实验（Figure 9b）显示 ESFK 可有效减少 latency，且对 memory footprint 无影响。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要求三个算子的 thread-block shape 一致（统一为 WARP×TIMES），通过 shape transposing 或 dim expanding 对齐。CUDA 实现中通过 `blockIdx.z` 判断当前 thread-block 应执行哪个算子。开源实现：https://github.com/UNITES-Lab/HEXA-MoE。

涉及论文标题：
- HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy

## Megablocks (Block-Sparse Kernel for MoE Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Megablocks (Gale et al. 2022) 是斯坦福大学提出的面向 MoE 高效训练的 block-sparse 矩阵乘法系统。核心贡献是实现了高效的 block-sparse 矩阵乘法 CUDA kernel，使得不同 expert 可以拥有不同的尺寸（不同 hidden dimension），仍能在一个 kernel 中批量计算，而无需传统的 token padding/dropping。传统 MoE 训练使用 expert parallelism + all-to-all dispatch/combine，各 expert 计算统一使用 GEMM——这要求所有 expert 接收的 token batch 大小相同（通过 capacity factor padding），引入大量冗余 FLOPs。Megablocks 通过两级 block-sparse 设计消除这一限制：(1) 外层 block-sparse 矩阵高效定位每个 token 对应的 expert；(2) 内层使用高速 CUDA kernel (CUTLASS/cuBLAS) 执行各 expert 的局部密集矩阵乘法，但所有 expert 的计算融合在单一 kernel 中，无需多次 kernel launch。

在 HMoE 中，Megablocks 的关键意义在于：异构 MoE 的不同 expert 具有不同的 FFN hidden dimension（如 2304→5888），无法使用统一 shape 的 GEMM。Megablocks 的 block-sparse kernel 原生支持混合形状的批量矩阵乘法，使 HMoE 的训练从工程层面成为可能。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Megablocks 的核心 kernel 设计（两级稀疏）：

```python
# Megablocks 处理 MoE 层 forward 的简化流程

# 输入: tokens [N, D]  按 expert assignment 排序后
# expert_offsets: [E+1]  各 expert 的 token 范围
#   expert_i tokens = tokens[expert_offsets[i] : expert_offsets[i+1]]

# Step 1: 构建 Block-Sparse 矩阵
# 将 E 个 expert 的 GEMM 表达为 Block Diagonal 矩阵:
# B = [[W_1,   0,   0]
#      [  0, W_2,   0]
#      [  0,   0, W_3]]
# 其中各 W_i 尺寸不同: W_1 [D, H_1], W_2 [D, H_2], ...

# Step 2: Block-Sparse GEMM (单 CUDA kernel)
# 使用 dCUDA block-sparse matrix multiply
# 各 CUDA block 负责一个 expert 的计算:
for each expert e with token range [start, end]:
    n_e = end - start
    if n_e > 0:
        # 局部 dense GEMM: [n_e, D] @ [D, H_e] → [n_e, H_e]
        CUBLAS_GEMM(tokens[start:end], W_e, output[start:end])

# Step 3: 输出 [N, ΣH_e]  — 各 expert 输出 concatenated
```

在 HMoE 的异构设置中，Megablocks 的优势尤为突出：
- Arithmetic distribution (2304→5888): 8 个不同形状的 W_e，传统方法需要 8 次独立 GEMM launch 或 pad 到 max(H_e)
- Megablocks: 单次 kernel launch，内部各 CUDA block 处理不同形状的子 GEMM

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Megablocks 开源实现：https://github.com/stanford-futuredata/megablocks。用于 MoE 训练框架中替代传统的 expert parallelism + all-to-all + GEMM pipeline。在 HMoE 的训练流程中，Megablocks 作为 expert 计算的后端，处理异构 expert 的不规则 GEMM。配合 DeepSpeed Zero2（参数分片）和 gradient checkpointing（激活检查点）实现高效训练。ES-MoE (Kim et al. 2024) 是对 Megablocks 的补充——通过 expert-wise offloading 到 CPU memory 并按需加载回 GPU，进一步缓解异构 expert 导致的 GPU memory 不均衡。

涉及论文标题：
- HMoE: Heterogeneous Mixture of Experts for Language Modeling

## Pipeline-Shared Cache (for Data-Centric MoE Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Pipeline-Shared Cache 是 HEXA-MoE 在 data-centric 并行配置下提出的 GPU HBM 内存管理机制，用于解决 data-centric MoE 训练中 backward pass 的内存膨胀问题。此前的 data-centric 方法（如 Janus）在 forward pass 中预取每层所需参数，但为 backward pass 保存了所有层的完整 gathered 参数在 GPU HBM 中，导致巨大的内存占用。Pipeline-Shared Cache 在每设备 HBM 上分配一块额外的共享缓存区域，动态缓存当前 pipeline stage 所需的 gathered MoE shards——forward 时写入，backward 时读取，同一 cache 区域在不同层之间复用。配合 all gather 通信与 attention/router 计算的 overlap，实现通信和内存的双重优化。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Pipeline-Shared Cache 的工作流程:
// HBM 布局: [model_params] [activations] [optimizer_states] [pipeline_cache]
// Cache 大小 = max_per_layer_gathered_params
//           = E × D_i × D_mid × 2 × sizeof(dtype)

for layer in 0..L-1:
    // Stream COMPUTE: Attention + Router
    attn_out = attention(layer_norm(x))
    
    // Stream COMM (overlap): All gather MoE shards → cache
    all_gather_into_cache(local_moe_shards, pipeline_cache)
    sync_streams()
    
    // Forward: ESMM 使用 cache 中的完整参数
    y = ESMM(x, pipeline_cache.W1, pipeline_cache.b1, R(x))
    y = ESMM(activation(y), pipeline_cache.W2, pipeline_cache.b2, R(x))

// Backward: 每层重新 all gather 到 cache (与 attention backward 重叠)
for layer in L-1 down to 0:
    all_gather_into_cache(local_moe_shards, pipeline_cache)
    ∂ℓ/∂W2 = ESTMM(y2, ∂ℓ/∂y, R(x))  // 使用 cache 中的完整参数
    ...
```

消融实验（Figure 9a）：data-centric 配置下若不加 pipeline-shared cache，内存占用会超过 Tutel baseline；加入后 data-centric 的内存占用略高于 model-centric 但明显优于 baseline（10%-48% 节省）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GPU HBM 上预分配 contiguous buffer，大小根据所有 MoE 层中最大参数总量计算。All gather 使用 NCCL 原语，结果直接写入 cache buffer。Backward 时从 cache 读取完整参数做 ESTMM，写回梯度仅需 local shard 部分。开源实现：https://github.com/UNITES-Lab/HEXA-MoE。

涉及论文标题：
- HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy

## CUDA Stream Async Expert Overwrite Loading (HarMoEny)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

CUDA Stream Async Expert Overwrite Loading 是 HarMoEny 的异步 expert 预取机制（Section 4.3），通过独立 CUDA stream 从 system memory（CPU RAM）异步传输 expert 权重到 GPU memory，直接覆写已完成的 expert 所占内存。核心洞察：expert 权重在推理中不变（read-only），无需先写回 system memory。Overwrite-based loading 比传统 "write-back + load" 快 5.5×（11ms vs 2ms on V100, 18MB Switch128 experts）。

从 kernel 调度角度拆解术语：

```
# HarMoEny 双 CUDA stream 模型 (Algorithm 1, Step 5)
# compute_stream: 执行 expert FFN GeMM
# load_stream: 异步 system→GPU expert weight transfer

gpu_expert_slot = [slot0, slot1]  # ping-pong 双 slot

for idx, expert_e in enumerate(assigned_experts):
    curr_slot = gpu_expert_slot[idx % 2]
    next_slot = gpu_expert_slot[(idx + 1) % 2]

    if expert_e not in GPU_memory:
        # load_stream: 异步预取下一 expert 权重
        with torch.cuda.stream(load_stream):
            # 直接覆写 next_slot (已完成的 expert)
            # 无需 write-back → 5.5× faster
            next_slot.copy_(system_mem[expert_e.offset], non_blocking=True)
            # Transfer: 18MB / 32GB/s ≈ 0.56ms theoretical, ~2ms actual (V100)

    with torch.cuda.stream(compute_stream):
        # 当前 expert 计算 (与 load_stream 的传输重叠)
        output += gate_weights[e] * expert_ffn(expert_e, tokens_e)
        # Expert FFN: 2× GeMM (W1[x] @ x → activation → W2[x] @ result)

    load_stream.wait_stream(compute_stream)  # 确保 slot 读写无冲突

torch.cuda.synchronize()
```

Annotations:
- **load_stream**: 独立 stream 执行 cudaMemcpyAsync (system→GPU)
- **compute_stream**: 主 stream 执行 expert FFN GeMM
- **双 slot ping-pong**: 仅需 2 个 expert slot（compute 用 1 个 + prefetch 用 1 个）
- **Overlap condition**: computation_time > transfer_time → 传输完全隐藏
- **5.5× origin**: 传统 offloading 需先 GPU→CPU write-back (9ms) + CPU→GPU load (2ms) = 11ms; overwrite 仅需 load (2ms)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

HarMoEny 在 PyTorch 中使用 `torch.cuda.Stream()` 创建独立 load stream + `.copy_(non_blocking=True)` 异步传输。需至少 2 个 expert slot fit in GPU memory（大多数 MoE serving 已满足）。由 token threshold q 保证 computation > transfer（防止传输无法隐藏的场景）。HarMoEny 1115 行 PyTorch 代码中实现，开源：https://github.com/sacs-epfl/HarMoEny。

涉及论文标题：
- HarMoEny: Efficient Multi-GPU Inference of MoE Models

## Zebra Parallelism (ZP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Zebra Parallelism (ZP) 是 HeterMoE 提出的面向异构 GPU 集群的 MoE 训练并行策略，替代传统 Expert Parallelism (EP)。在 ZP group 中，expert 模块分布在 N 个 expert GPU（older generation），attention blocks 和其余模块（gate、embedding）复制在 M 个 attention GPU（newer generation）。ZP 将 input batch 分为 R 个 microbatch，attention GPU 和 expert GPU 同时处理不同 microbatch，形成 "zigzag" 式的跨 GPU 流水线。与 Pipeline Parallelism 在 layer 级别切分不同，ZP 在单个 transformer layer 内部切分 attention 和 expert 模块到不同 GPU。

ZP 的关键特征：(1) 不引入额外通信——EP 本就通过 all-to-all 交换 token，ZP 仅将 exchange 从 "attention GPU↔attention GPU" 变为 "attention GPU↔expert GPU"，数据总量不变；(2) 每 GPU 内 3 个 CUDA stream（2 通信 + 1 计算）并行执行 dispatch、combine 和 compute，dispatch 和 combine 方向相反在独立 stream 上不发生带宽竞争；(3) Theorem 1 证明了最优 task ordering——bipartite 通信（M 个 attention GPU 与 N 个 expert GPU 之间）的 ZP schedule 为最小化总迭代时间的最优调度。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
ZP group: M attention GPUs + N expert GPUs, R microbatches

Optimal Forward Schedule (Theorem 1):
  Attention GPU compute: (A_{1,1}^F...A_{1,R}^F)...(A_{L,1}^F...A_{L,R}^F)
  Expert GPU compute:    (E_{1,1}^F...E_{1,R}^F)...(E_{L-1,1}^F...E_{L-1,R}^F)

  A_{i,j}^F: layer i, microbatch j, attention forward
  E_{i,j}^F: layer i, microbatch j, expert forward

Stream Architecture (per GPU):
  Stream 0 (compute): attention/expert 计算
  Stream 1 (comm D):  dispatch all-to-all (Attn→Exp 方向)
  Stream 2 (comm C):  combine all-to-all  (Exp→Attn 方向)
  Sync via CUDA events between streams

依赖约束:
  t(A_{i,j}^F) ≥ t(C_{i-1,j}^F) + T_C            (数据依赖)
  |t(A_{i,j}^F) - t(A_{i',j'}^F)| ≥ T_A   (stream 顺序执行)

Overlap 示例 (R=3, forward):
  Attn GPU: [Disp0][A_{1,0}^F][Comb0][Disp1][A_{1,1}^F]...
  Exp GPU:  [==== E_{1,0}^F ====][Disp1][E_{1,1}^F]...
  // Dispatch 和 Combine 方向相反，在独立 stream 上无带宽竞争
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 基于 PyTorch v2.2 + DeepSpeed v0.14（3K 行 Python）
- ZP engine 初始化时创建 3 CUDA stream，为每个 microbatch 分配 receive buffer
- 创建分离的 NCCL dispatch/combine all-to-all group
- 通过 PyTorch NCCL all-to-all wrapper 传入不等 split size（因 Asym-EA 可能导致不同 GPU 处理不同数量 tokens）
- Gate backward 特殊处理：gate 的 top-k confidence scores 形式 "residual" 连接，backward 分两路传播——一路经 confidence scores 到 gate weights，另一路经 expert outputs。HeterMoE 在 attention outputs 处停止第二分支的 backward，等 expert GPU 梯度后 accumulated
- ZP 可与 data parallelism 组合（多 ZP group 间做 DP）
- 论文声明将开源（截至分析时未找到公开代码仓库）

涉及论文标题：
- HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs

## Asymmetric Expert Assignment (Asym-EA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Asymmetric Expert Assignment (Asym-EA) 是 HeterMoE 中消除 Zebra Parallelism 流水线气泡的细粒度负载均衡机制。当 expert GPU 计算慢于 attention GPU（常见于短序列），attention GPU 产生 idle bubbles。Asym-EA 将部分 expert 计算迁回 (offload) 到 attention GPU 以 balance 计算时间。

核心算法为 "gather and squeeze"（Algorithm 1）：accumulate 跨多层的 bubble（T_gather = T_E^Exp - T_A^Attn，每 microbatch 每层 expert GPU 比 attention GPU 多花的时间），直到累积量 ≥ T_squeeze（offload 一个最小 chunk 可消除的 bubble），然后在 accumulation 最多的层 squeeze。最小 offload chunk: n_1 = max(1, N/M) 个 experts per attention GPU 获得，n_2 = n_1·M/N 个 experts per expert GPU 被 offload。考虑 memory 约束：α 系数 enforce 上限 n_max（attention GPU 内存），β 系数 enforce 下限 n_min（expert GPU 内存），α 和 β 至多一个激活。

效率：Asym-EA 在 4K 序列上提供 1.14-1.20× 额外加速，在 >20K-28K 序列上不再需要（T_A^Attn ≥ T_E^Exp）。

从kernel调度角度拆解术语：

```
Algorithm 1: Gather and Squeeze
Input: n (experts), L (layers), M, N (GPU ratio)
       T_A^Attn, T_E^Attn, T_E^Exp (profiled per-microbatch times)
Output: O = {o_1,...,o_L} (experts to offload per layer)

n_1 ← max(1, N/M)                          // per-attn-GPU min acquire
n_2 ← n_1 · M/N                            // per-exp-GPU min offload
T_gather ← T_E^Exp - T_A^Attn              // bubble per layer
T_squeeze ← T_E^Exp·N/n·n_1 + T_E^Attn·N/n·n_2

α = min(⌊n_max/n_2⌋·T_squeeze/(L·T_gather), 1)  // memory upper bound
β = max(⌈n_min/n_2⌉·T_squeeze/(L·T_gather), 1)  // memory lower bound

t_bubble ← 0
for l ← 1 to L:
    t_bubble += α·β·T_gather
    if t_bubble ≥ T_squeeze:
        o_l ← ⌊t_bubble/T_squeeze⌋ · n_2
        t_bubble -= o_l/n_2 · T_squeeze

// 可整除要求: M | N 或 N | M（与 EP 中 GPU 数须整除 expert 数类似）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Asym-EA optimizer 依赖 Profiler 提供的 T_A^Attn, T_E^Attn, T_E^Exp 和 n_min, n_max
- Profiler 在每个 setup 上运行一次
- Offload 后 attention GPU 先完成所有 microbatch attention，再计算被 offload 的 experts
- 选择性逐层 offload——不同层 offload 不同数量的 experts，避免简单统一 offload 导致气泡转移到 expert GPU
- 仅在可整除的 GPU 比例下有效（如 4:2, 4:4, 4:8），其他比例（如 4:3）无法使用 Asym-EA
- 使用 profiled forward 时间优化，backward 时间成比例减少

涉及论文标题：
- HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs

## Persistent Kernel / Megakernel for Distributed MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Persistent Kernel（持久内核）是一种仅需一次 CPU launch、在 GPU 上持续运行直到完成全部计算任务的 GPU kernel 设计范式。FlashMoE 将这一概念推向极致，构建了一个 Megakernel（巨型融合内核），将分布式 MoE operator 的全部阶段——Gate routing、Token Dispatch、Expert FFN (2×GEMM)、Expert Combine、跨 GPU 通信——融合为单一持久 kernel。传统实现（DeepSpeed-MoE、Megatron-LM）每个 MoE layer 需要 33-550 次独立 kernel launch（Table 1），每次 launch 由 CPU 串行调度，产生 CPU-GPU 同步间隙和非确定性 kernel start time。FlashMoE 的 megakernel 仅需 1 次 launch，kernel 内 GPU 自主管理所有任务调度和执行。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// === 单次 CPU launch (仅1次!) ===
LaunchKernel<<<N_blocks, 128_threads>>>(A, X, O, N);

// === Kernel 内部 (GPU 自主执行) ===
// Phase 0: FusedGate (所有 block 参与)
T_φ, G_φ ← FusedGate(A)

// Phase 1: Role Assignment
if blockId + 1 < N:
    Dispatch(T_φ, A)               // 准备 dispatch packet
    while interrupt == False:      // 持久循环!
        awaitTask_from_Scheduler() // 等待 task assignment
        switch task.Type:
            case GEMM0: fGET_GEMM0()
            case GEMM1: fGET_GEMM1()
            case Combine: combine()
else:
    if warpId == 0: scheduler_loop()
    else: subscriber_loop()

// Phase N: Interrupt → 所有 actor 退出 → kernel 返回
```

Kernel 在整个 MoE layer 执行期间持续活跃（不返回 CPU），通过 while 循环和 doorbell 信号机制维持。Scheduler 持续 sweep doorbells → 发现新 task → 分配 → processor 执行 → 结果写回 → notify subscriber → subscriber 解码下一轮 task。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- FlashMoE kernel 实现指标（Table 4）：6820 LOC, 0 B stack frame, 0 spill, 46 KB shared memory/block, 255 registers/thread, 2 max active blocks/SM, 53s compilation, 29 MB binary
- 单一 kernel 的核心优势：消除 launch 间隙 → SM utilization 93.17%（vs DeepEP 14%）；消除 kernel boundary global memory round-trip；确保确定性的 GPU-native 调度时序
- 与 CUDA Graphs 的区别：CUDA Graphs 仅消除 CPU launch overhead（仅适用 static workload），不解决 kernel boundary round-trip 或 host-device synchronization

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel

## Actor Model for GPU Kernel — Warp/Block Specialization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Actor Model 是并发计算模型（Carl Hewitt, 1973）。FlashMoE 将其移植到 GPU kernel 内部：将 GPU thread block 和 warp 特化为三种独立 actor 角色——Processor（N-1 个 block，执行 GEMM/element-wise）、Scheduler（1 个 warp，多线程 work-conserving 调度）、Subscriber（3 个 warp，解码 remote packet 为 task descriptor）。每个 actor 通过共享/全局内存交换消息（doorbell 信号 + task descriptor），以非阻塞、松耦合方式并发。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
GPU SM 分配:
┌────────────────────────────────────────────┐
│ SM_0: Proc 0  │ SM_1: Proc 1  │ ...       │
│ SM_{N-1}: OS Block (Scheduler + Subscriber)│
│   Warp 0: Scheduler (32 threads 并行 sweep)│
│   Warp 1-3: Subscriber (并发 decode flags) │
└────────────────────────────────────────────┘

Actor 间通信:
- Scheduler ↔ Subscriber: shared memory (同 block 内)
- Scheduler → Processor:   global memory doorbell + task queue
- Remote GPU → Subscriber: NVSHMEM one-sided put + signal

Scheduler 循环 (Algorithm 3):
while scheduled < taskBound:
    do in parallel: sweep doorbells → tqState
    WarpInclusiveSum(counts, offset, total)
    while total > 0:
        repopulate ready_queue
        do in parallel: signal processors
    taskBound = AtomicLoad(taskBound)  // dynamic

Subscriber 循环 (Algorithm 4):
while interrupt == False:
    do in parallel: atomically claim dispatch flags
        if set: decode → GEMM0 tasks → notify scheduler
    do in parallel: atomically claim combine flags
        if set: decode → combine tasks → notify scheduler
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- OS block 仅占 1/N 资源做管理，N-1 个 block 全力计算
- Scheduler work-conserving + multithreaded：有 task 就分配，32 threads 并行 sweep
- Subscriber 用 3 warps（非 1）并行处理 dispatch + combine 信号
- 传统 GPU kernel 所有 block 执行相同对称代码；FlashMoE 将 GPU 视为分布式系统，block/warp 是独立"处理节点"

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel

## Symmetric Tensor Layout for Conflict-Free One-Sided Access

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Symmetric Tensor Layout（对称张量布局）是 FlashMoE 为跨 GPU 非阻塞 one-sided 访问设计的内存组织方案。定义 L ∈ R^{P×R×B×E×C×H}，P = world size, R = 2 rounds (dispatch+combine), B = 2 buffers (outgoing+incoming), E = local experts, C = capacity, H = hidden dim。核心是 temporal buffering：4× overprovision（2 rounds × 2 buffers）使每个数据流有独立 buffer。Theorem 3.1 证明 layout write-write conflict-free。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Layout: L[P][R][B][E][C][H]
// Write rules: inter-device → p*=p_s, b=1; intra-device staging → b=0, p_s=p_t
// Conflict-free proof: p_s1 ≠ p_s2 → p*_1 ≠ p*_2 → i1 ≠ i2
// Size(L) ≈ 4 × Size(T), Memory overhead ≤ 2.15%

// 数据流:
// GPU i dispatch: write L[i,0,0,:,:,:] → NVSHMEM put → GPU j L[i,0,1,:,:,:]
// GPU j compute: Subscriber decode → Processor GEMM0→GEMM1
// GPU j combine:  write L[j,1,0,:,:,:] → NVSHMEM put → GPU i L[j,1,1,:,:,:]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- NVSHMEM 对称内存分配（`nvshmem_malloc`）确保地址对称
- Capacity upscaling 对齐 tile block size bM=128
- 2-round × 2-buffer overprovision 确保 producer-consumer 无需同步
- Memory overhead ≤ 2.15%（Mixtral 8x7B, 32K seqlen），DeepSeek-V3 仅 0.11%

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel

## Payload-Efficient Communication / In-Place Padding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Payload-Efficient Communication 是 FlashMoE 消除 MoE dispatch 网络带宽浪费的技术。传统 AlltoAll 的对称性约束迫使零填充 token 参与通信和计算。FlashMoE 用 In-Place Padding（本地对齐 tile size bM=128）后用 NVSHMEM one-sided put 仅发送有效 token，消除 null padding 的网络传输和后续无效计算。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Traditional: AlltoAll(padded) → null tokens 占带宽!
// FlashMoE:
for each expert e on remote GPU j:
    if actual_tokens > 0:
        nvshmem_put(
            &L[j, DISPATCH, incoming, e, 0, :],   // remote
            &L[0, DISPATCH, outgoing, e, 0, :],   // local
            actual_tokens × H × sizeof(float),     // 仅有效数据
            peer = j
        )
    // In-place padding (对齐 bM=128) 仅在本地 buffer, 不传网络
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 前提是 one-sided (R)DMA——每 GPU pair 独立决定传输量，不依赖 collective symmetry
- Capacity upscaling（对齐 bM=128）确保 Processor coalesced read

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel

## In-Kernel Work-Conserving Task Scheduler

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlashMoE megakernel 的核心管理组件，由 OS block 中 1 个 warp（32 threads）实现。属性：(1) work-conserving——有 task 就分配；(2) multithreaded——32 threads 并行 sweep doorbells；(3) in-kernel——运行 GPU 上，无需 CPU 干预。Doorbell 是 monotonic counter（非 binary flag），避免丢失 concurrent 更新。taskBound 由 Subscriber atomic increment 动态增加。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
while scheduled < taskBound:
    // 并行 sweep doorbells
    do in parallel: local_counts[tid] = count_pending(doorbells)
    
    // Warp inclusive sum (~5 cycles via __shfl_up_sync)
    WarpInclusiveSum(local_counts, &offset, &total)
    
    // 分发 task 给空闲 processor
    while total > 0:
        repopulate ready_queue
        do in parallel: signal processors about task indices
    
    // 动态更新
    taskBound = WarpBroadcast(AtomicLoad(taskBound))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Warp shuffle `__shfl_up_sync` (~5 cycles) vs shared memory atomic (更慢)
- 传统 GPU kernel 按预分配数据范围处理；FlashMoE 按 runtime readiness 动态分配
- 类似于 OS 进程调度器的设计哲学，但运行在 GPU warp 上

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel

## Tile-Level Parallelism in Fused MoE Kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Tile-Level Parallelism 是 FlashMoE 将大矩阵分解为细粒度、可独立调度的 tile 计算单元的策略。每个 128×64 tile 对应一个独立 task descriptor。MoE FFN (2×GEMM + activation) 和 Combine 统一为 task 抽象：t = (M, ⋆, φ)，执行 F_t(A, B, C, D) := C ← φ(A ⋆_t B + D)。⋆ 为 · (GEMM) 或 ⊙ (Hadamard)。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Tile: 128×64, 128 threads/block
// Task 抽象:
// FFN:  t₁ = (M, ·, SiLU)     → C₁ ← SiLU(A·W₁ + b₁)
//       t₂ = (M, ·, identity)  → C₂ ← A·W₂ + b₂
// Combine: t₃ = (M, ⊙, identity) → C ← A⊙S + C

// Task struct (128-byte cache line aligned):
struct Task {
    const byte* aData;
    array<const byte*, 2> bData;  // W1, W2
    array<byte*, 2> cData;        // output
    uint M, tileIdx, batchIdx, peerIdx, expertIdx;
    TaskType taskType;  // GEMM0 | GEMM1 | Combine
};

// Processor GEMM0: CUTLASS gemm + SiLU epilogue + stage to shared memory
// Processor GEMM1: CUTLASS gemm + NVSHMEM put (if remote)
// Processor Combine: Hadamard product + accumulation
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 128×64 tile = 8192 elements; FP32 tile = 32 KB
- CUTLASS in-kernel device-side GEMM 执行 tile 级矩阵乘
- 同一 expert 的多个 tiles 可由不同 Processor block 并行处理
- Tile dimension selection balance: register usage + shared memory + SM occupancy

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel

## CUTLASS (CUDA Templates for Linear Algebra Subroutines)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

CUTLASS 是 NVIDIA 开源的 CUDA C++ 模板库（https://github.com/NVIDIA/cutlass），通过 C++ 模板将 GEMM 分解为可组合的抽象层次（tile → warp → thread），编译期生成针对特定数据类型、矩阵布局和 GPU 架构优化的 kernel。FlashMoE 使用 CUTLASS 作为 in-kernel BLAS，在持久 kernel 内直接调用 device-side GEMM。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// CUTLASS 三级分块:
// 1. Thread Block Tile → shared memory (TILE_M×TILE_K + TILE_K×TILE_N)
// 2. Warp Tile → registers (WARP_M×WARP_N), iterate K
// 3. Thread Tile → Tensor Core mma.sync (M16N8K16)

// FlashMoE Processor 内调用:
fused_device_gemm(
    A = input_tile[128, 2048],
    B = expert_W1[2048, 2048],
    C = output_tile[128, 2048],
    epilogue = SiLU,  // fused in registers
    bias = expert_bias
);
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- cuBLAS = 预编译库（每次调用需 launch）；CUTLASS = 模板库（编译期嵌入）
- CUTLASS 3.x Cute 抽象提供更轻量 layout + tiling algebra
- FlashMoE 255 registers/thread 部分归因于 CUTLASS register-intensive GEMM
- Megatron-LM CUTLASS backend: 85 kernel launches; FlashMoE: 仅 1 次

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel

## Activated-Expert-Balanced Scheduling (AEBS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

AEBS 是 JANUS 提出的 MoE 层激活 expert 调度算法，实现为 GPU kernel，在每 MoE 层每 decode step 运行。其核心思想是：MoE 层 latency 由所有 MoE instance 中 distinct activated expert 数最多的那个 instance（即 a_max = max_i a_i）决定。因此，调度目标不是平衡 token counts 或 routing probabilities（如 EPLB），而是直接 minimize a_max。

AEBS 是 synchronization-free 的——每个 MoE instance 独立运行相同的 deterministic kernel，通过确定性算法保证所有 instance 产生相同的调度决策，无需跨 GPU 协调或 CPU-GPU 同步。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Algorithm 1: AEBS (CUDA Kernel, per MoE layer, per decode step)

Input (GPU global memory):
  T: token count, k: top-k, n_e: MoE instance count
  L(i,j): logical expert ID for token i, expert j    [T × k]
  R(e): replica count for expert e                    [E]
  G(e): set of instances hosting replicas of expert e [E]
  P(e,g): physical replica ID of expert e on instance g [E × n_e]

// Step 1: Collect activated expert set (GPU parallel)
Parallel for (i in 0..T-1, j in 0..k-1):
    atomicOr(E_bitmap, L(i,j))  // bit vector marking activated experts

// Step 2: Initialize per-instance load counters
load[g] = 0 for g = 1..n_e   // distinct expert count per instance

// Step 3: Assign single-replica experts (forced placement)
for e in E_active where R(e) == 1:
    g = unique_instance(G(e))
    actRep[e] = P(e,g)
    atomicAdd(load[g], 1)

// Step 4: Assign multi-replica experts (greedy load balancing)
for e in E_active where R(e) > 1:
    g* = argmin_{g ∈ G(e)} load[g]  // instance with fewest activated experts
    actRep[e] = P(e, g*)
    atomicAdd(load[g*], 1)

// Step 5: Rewrite token routing (GPU parallel)
Parallel for (i in 0..T-1, j in 0..k-1):
    O(i,j) = actRep[L(i,j)]  // logical EID → physical RID

// Step 6: Dispatch (performed by each MoE instance independently)
// Each instance reads O to determine which tokens to process locally

Key invariants:
  - ALL n_e instances run identical kernel with identical input
  - AEBS is deterministic → same output on all instances
  - No cross-GPU communication needed for scheduling
  - No CPU-GPU synchronization (pure GPU kernel)
```

Performance characteristics:
- Scheduling overhead: <20μs (batch=64) to <90μs (batch=4096), plateaus when most experts activated
- AEBS vs EPLB: reduces a_max by 2-5 experts → MoE layer latency reduction
- Scales well with MoE instances (8→16: only small overhead increase)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 实现为 CUDA kernel (~300 行 CUDA/C++)，作为 SGLang MoE layer 的一部分
- Input data (top-k routing results) 已在 GPU global memory (gating kernel 输出)，无需 CPU 访问
- Replica mapping metadata 更新频率低 (仅在 reconfiguration 时，~15min 间隔)，可放入 GPU constant memory
- 所有 MoE instances 使用相同 input 独立运行（通过 NVSHMEM broadcast 或共享的 routing data 保证一致性）
- 适用于任何有 expert replica 冗余的分布式 MoE 推理/训练系统

涉及论文标题：
- JANUS: Disaggregating Attention and Experts for Scalable MoE Inference

## Flexible Token Dispatcher (CUDA Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Flexible Token Dispatcher 是 Lazarus 为 MoE 训练设计的 CUDA kernel，实现非对称 expert placement 下的高效 token dispatch。在传统 EP 中，每 expert 仅有一个 replica，token dispatch 是简单的：将 token 发送到持有该 expert 的 GPU。但 Lazarus 为 popular experts 分配了不同数量的 replicas 在不同 GPU 上（非对称 placement），需要决定每个 token 具体发往哪个持有目标 expert replica 的 GPU，同时平衡各 GPU 负载。

该 kernel 对所有 E 个 experts 和 N 个 ranks 并行计算 dispatch schedule。核心逻辑：(a) 计算每个 expert 的每 replica 应处理的 token 数 p_e = t_e / r_e（负载均衡）；(b) 计算每个 rank 对每个 expert 的处理容量 P_{e,j} = p_e × R_{e,j}；(c) 优先将 rank j 本地已有的 token 分配给自身（min(P_{e,j}, T_{e,j})）；(d) 将超出本地容量的剩余 token 按各 rank 剩余容量比例分发（proportional distribution）；(e) 根据 schedule 将 input activations reshuffle 为连续 buffer，使 routed to same expert + dispatched to same rank 的 token 连续排列，供后续 flexible all-to-all collective 使用。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Flexible Token Dispatcher 的 CUDA kernel 执行流程（Algorithm 1）：

```
Input: N GPUs, i (current rank), R_{e,j} (replicas for expert e at rank j),
       T_{e,j} (#tokens routed to expert e at rank j), h (input activations)
Output: h' (reshuffled activations for all-to-all), s_j (tokens to rank j)

// Step 1: All-gather T_{e,j} from all ranks (E integers per rank, negligible)
// Step 2: Compute dispatch schedule (parallel across experts and ranks)
for e ← 0 to E in parallel:
    r_e = Σ_j R_{e,j}              // total replicas for expert e
    t_e = Σ_j T_{e,j}              // total tokens routed to expert e
    p_e = t_e / r_e                // tokens each replica should handle
    
    for j ← 0 to N in parallel:
        P_{e,j} = p_e × R_{e,j}    // rank j's processing capacity for expert e
        P_{e,j} -= min(P_{e,j}, T_{e,j})  // subtract locally processed tokens
    
    D_{e,i} = p_e × R_{e,i} - P_{e,i}  // tokens processed locally by rank i
    
    for j ← 0 to N, j ≠ i in parallel:
        // Distribute remaining tokens proportionally to residual capacity
        D_{e,j} = (T_{e,i} - D_{e,i}) × P_{e,j} / Σ_{k≠j} P_{e,k}

// Step 3: Compute dispatch counts per rank
for j ← 0 to N in parallel:
    s_j = Σ_e D_{e,j}

// Step 4: Reshuffle input activations
for j ← 0 to N in parallel:
    for e ← 0 to E in parallel:
        start = Σ_{0..j-1} s_{j'} + Σ_{0..e-1} D_{e',j}
        end = start + D_{e,j}
        // Copy D_{e,j} tokens of expert e from h to h'[start..end]
        // Tokens are sorted by (target_rank, expert_id)

// Step 5: Perform flexible all-to-all with s_j tokens to each rank j
//         (no padding — each rank sends exactly s_j tokens)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Lazarus 用 ~500 LoC CUDA 实现该 kernel，在 MoE block 的 forward path 中替代传统 DeepSpeed MoE 的 dispatch 逻辑。由于 collective communication operations 需要所有参与 rank 的同步，kernel 执行前需先 all-gather 所有 rank 的 T_{e,j}（E integers per rank，overhead 可忽略），确保所有 rank 有全局一致的 dispatch schedule 信息。该 kernel 使用 shared memory 处理 per-rank 的 per-expert capacity 计算，通过原子操作协调跨 warps 的 token 分配。

在 RTX 3090 (10 emulated nodes) 上，当 workload 完全 balance (1:1 load ratio) 时，flexible dispatcher 引入的 overhead 极小（Lazarus 吞吐几乎等于 DS baseline）。当 load ratio 变为 4:1（imbalanced）时，Lazarus 保持恒定吞吐，而 DS 吞吐急剧下降。

涉及论文标题：
- Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement

## Grouped-GEMM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Grouped-GEMM 是将多个具有相同形状但不同数据的矩阵乘法（GEMM）批量执行的 kernel 技术。在 MoE 推理/训练中，每个 expert 的 FFN 计算是独立的 GEMM：B_i W_i（token batch × expert weight matrix）。当每 GPU 持有多个 experts 时，传统方案是 for-loop 逐个 expert 调用 cuBLAS GEMM（N 次 kernel launch，N 次 kernel launch overhead），而 Grouped-GEMM 将所有 expert 的 GEMM 合并为单次 kernel launch，消除 N-1 次 launch overhead。

LLEP 论文 Fig. 8 对此进行了基准测试：在相同总 FLOPs (B=65536 tokens 均匀分配到 N experts, D=H=8192) 下，cuBLAS 独立 GEMM（for-loop 多次 launch）vs Triton fused Grouped-GEMM（单次 launch + TMA）。结果显示 cuBLAS 的多次 launch 仍然快于 Triton 的单次 fused kernel，因为每个 cuBLAS GEMM 是硬件特定的高度优化实现（针对 NVIDIA GPU 架构级别的优化），而 Triton 版本是通用实现。这说明即使消除 launch overhead，对 hardware-optimized 的 GEMM 来说，数据布局和 tile 策略的重要性超过 kernel launch 数量。

从kernel调度角度拆解术语：

MoE 中 Grouped-GEMM 的两种实现方式对比：

```
方式 1: cuBLAS 独立 GEMM (for-loop, N 个 experts)
  for i in range(N):
      if B_i is not empty:
          output[i] = cublasGemmEx(B_i, W_i)  // 每次 launch + 硬件优化
  // N 次 kernel launch, 但每次硬件高度优化
  // 时间 = N × T_overhead + Σ(B_i × T_Bi,D,H)

方式 2: Triton fused Grouped-GEMM (单次 launch)
  @triton.jit
  def grouped_gemm_kernel(B_ptrs, W_ptrs, output_ptrs, B_sizes):
      // 单次 kernel launch, TMA 加速数据加载
      for i in range(N):
          // 所有 expert 在同一 kernel 内计算
  // 1 次 kernel launch, 但通用实现未针对硬件 tuning
```

LLEP 的发现：在 D, H 固定时，B_i 越大 GEMM 效率越高（T_B1,D,H < T_B2,D,H when B1 > B2）。因此给定固定 FLOPs，少量大 GEMM（少量 experts 大量 tokens）远快于大量小 GEMM（大量 experts 少量 tokens）。EP 和 LLEP 均利用此原理——将 experts 分布到多 GPU，每 GPU 仅计算少数 experts 的大 batch GEMM。

术语一般如何实现？如何使用？

主流 Grouped-GEMM 实现选项：
- **cuBLAS** (NVIDIA proprietary): 硬件优化的独立 GEMM，通过 `cublasGemmEx` 调用。对单个大 GEMM 效率最高。
- **CUTLASS GroupedGEMM**: NVIDIA 开源模板库，支持单次 kernel launch 执行不同形状的 GEMM（不同 B_i 和相同 W_i 形状）。
- **Triton Grouped-GEMM**: 通用实现，可搭配 TMA (Tensor Memory Accelerator) 加速 H100+ 上的数据加载。
- **MegaBlocks**: 专为 MoE 设计的 sparse GEMM kernel，支持 block-sparse 和 grouped GEMM 操作。

选择建议：大 B_i（> 1000 tokens per expert）用 cuBLAS 独立 GEMM；小 B_i 且多 experts 用 CUTLASS GroupedGEMM 减少 launch overhead；H100+ 平台上可考虑 Triton+TMA 的融合 kernel。

涉及论文标题：
- Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts

## MoE Parallel Folding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE Parallel Folding 是 NVIDIA 在 Megatron-Core 中提出的异构混合并行策略，核心思想是**解耦 Transformer 中 Attention 层和 MoE 层的并行映射**。传统分布式训练中，所有层共享同一套并行配置（如 TP=2, PP=4），但实际上 Attention 层和 MoE 层的最优并行策略不同：Attention 层受益于高 TP/CP 处理密集序列计算，MoE 层受益于高 EP 处理稀疏 expert 计算。Parallel Folding 允许 Attention 层使用独立的 TP×CP×DP×PP 四维并行映射，MoE 层使用 Expert-TP×EP×Expert-DP×PP 四维并行映射。通过将通信密集型并行操作"折叠"到 NVLink 高带宽域内，减少跨节点通信开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Parallel Folding 的关键在于通信域的折叠映射：

```
# Attention 层并行配置: TP=2, CP=2
# → Attention 的 TP×CP group = 4 GPUs (同一节点 NVLink 域内)

# MoE 层并行配置: EP=8
# → MoE 的 EP group = 8 GPUs (1 节点或跨节点)

# Folding: Attention 的 TP×CP group (4 GPUs) "折叠"到
#           MoE 的 EP group (8 GPUs) 中
# → 将 Attention TP/CP 通信限定在 NVLink 高带宽域
# → 避免跨节点通信扩大到 Attention 层
```

实际训练配置示例（128 H100, 46.8% MFU）：

```
# 最优配置:
Attention: TP=1, CP=2  (TP×CP=2 GPUs, NVLink 域内)
MoE:       EP=8         (EP=8, 单节点 8 GPU 的 NVLink 内)
PP=4, VPP=8, DP 自动

# 效果:
- TP=1 避免 Attention 层的跨节点 TP 通信
- EP=8 最大化 expert 间并行
- CP=2 分担长序列内存压力
- PP=4 跨 4 个 pipeline stage
```

配置搜索调优实践（论文总结）：
1. TP 和 EP 保持在 NVLink 域内 —— TP/EP 每层都有通信，NVLink 带宽远超 InfiniBand
2. MoE 层 EP 性能优于 TP —— expert 独立计算，EP 仅需 All-to-All token dispatch
3. AllToAll-based token dispatcher 对 TopK=1-4 更高效（vs AllGather-based）
4. CP 配合 GQA 可重叠通信与计算，降低 KV cache 通信量
5. 跨节点扩展用 PP+DP，VPP 减少 pipeline bubble size
6. 早期训练阶段对 MoE 层启用 recomputation 缓解负载不均 OOM

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现于 Megatron-Core (https://github.com/NVIDIA/Megatron-LM)，通过 NeMo 框架调用：
- 为 Attention 和 MoE 层分别创建独立的 process groups
- 通信域映射：Attention TP/CP groups 被映射为 MoE EP group 的子集
- 传统限制 EP ≤ DP 被打破，允许 EP 独立设置
- 需要 nccl 支持灵活的子通信域创建

已知性能：Mixtral 8x22B 达 49.3% MFU, Qwen2-57B-A14B 达 39.0% MFU, Llama 3-E8T2 达 46.8% MFU (128 H100)。扩展至 1024 GPUs, 128K 序列长度。

涉及论文标题：
- Llama 3 Meets MoE: Efficient Upcycling

## 5-D Hybrid Parallelism

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

5-D Hybrid Parallelism 是 Megatron-Core 中用于大规模 MoE 模型训练的**五种并行策略的任意组合**，具体包括：

1. **Tensor Parallelism (TP)**：将单层内的权重张量沿 hidden/column 维度切分到多个 GPU，每 GPU 计算部分结果后通过 AllReduce 合并。适合 Attention 层的 QKV 投影和 FFN 权重矩阵
2. **Expert Parallelism (EP)**：将不同 expert 的权重放置到不同 GPU 上，token 通过 All-to-All 通信路由到对应 expert 所在 GPU 计算后再返回。适合 MoE 层
3. **Pipeline Parallelism (PP)**：将模型按层切分为多个 stage，每个 stage 放置在不同 GPU 上，通过 micro-batch pipeline 流水线执行。配合 Virtual Pipeline Parallelism (VPP) 减少 pipeline bubble
4. **Context Parallelism (CP)**：将长序列沿序列维度切分到多个 GPU，减少单 GPU 的激活内存。配合 Ring Attention 或 Blockwise Transformers
5. **Data Parallelism (DP) with ZeRO-1**：每个 DP rank 持有完整模型副本但分片 optimizer states，处理不同 batch 数据后梯度 allreduce

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

五种并行的通信模式和切分方式：

```
# TP: 沿 hidden/column 维度切分
W [d_model, d_ffn] → shard along d_ffn → W_0 [d_model, d_ffn/tp], W_1 [...]
Forward: y_partial = x @ W_i  →  AllReduce(y_partial)
Backward: 同上，梯度沿相同维度 reduce

# EP: 按 expert 分配到不同 GPU
experts {E_0,...,E_7}, EP=4 → GPU0: {E_0,E_1}, GPU1: {E_2,E_3}, ...
Forward: AllToAll_Scatter(tokens) → ExpertCompute → AllToAll_Gather
通信量: 2 × total_tokens × d_model × sizeof(dtype)

# PP: 按层切分 stage
Layers {0..63}, PP=4 → Stage0: {0..15}, Stage1: {16..31}, ...
Forward: GPU0→GPU1→GPU2→GPU3 (send/recv activations)
VPP: 每 GPU 交替执行多个 virtual stage，填充 bubble

# CP: 沿序列长度切分
SeqLen=8192, CP=2 → GPU0: tokens[0:4096], GPU1: tokens[4096:8192]
Attention: RingAttention 交换 KV 块完成跨段 attention

# DP: 复制模型权重，独立 batch
B_total = dp_size × micro_batch_size
ZeRO-1: optimizer states 分片，梯度 AllReduce 后各 rank 更新自己的分片
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现于 Megatron-Core + NeMo：
- 用户指定 (TP, EP, PP, CP, DP) 和 global batch size
- Megatron-Core 自动建立对应的 NCCL process groups 和通信拓扑
- 关键约束：EP ≤ DP (传统)，但 MoE Parallel Folding 可打破此约束
- 实际使用以 TP×CP 不跨节点、EP 保持 NVLink 域内、PP 跨节点、DP 跨所有节点为最佳实践
- 已知性能：46.8% MFU (128 H100, Llama 3-E8T2), 49.3% MFU (Mixtral 8x22B)

涉及论文标题：
- Llama 3 Meets MoE: Efficient Upcycling

## HCCL (Huawei Collective Communication Library)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

HCCL 是华为开发的高性能集合通信库，基于 Ascend NPU 平台，类似 NVIDIA NCCL 的角色。提供单节点多卡和多节点多卡的集合通信原语，包括 All-to-All、All-Gather、All-Reduce、Reduce-Scatter、Broadcast 等。支持多种通信算法：ring、mesh、HD (Hierarchical Decomposition)、ring+HD、mesh+HD。通信底层基于 PCI-E、HCCS（节点内）和 RoCE（节点间）高速链路。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 LocMoE 的 PanGu-Σ MoE 训练中，HCCL 执行以下通信模式：

```
# MoE 层的通信-计算流程（使用 HCCL 原语）
# 假设 128 Ascend 910A NPU, EP=16, TP=8

# Phase 1: All-to-All Dispatch（将 token 发送到对应 expert 的设备）
# Group-wise: 按 TP 域拆分 All-to-All
for tp_group in range(num_tp_groups):
    # 每个 TP group 内的 device 负责 EP domain 的部分通信
    local_tokens = tokens[tp_group * local_batch : ...]
    # HCCL All-to-All: 在 EP domain 内交换 token 数据
    dispatched = HCCL.all_to_all(local_tokens, expert_idx, ep_group)
    
# Phase 2: All-Gather in TP domain
# 利用 HCCS 256GB/s 高带宽在 TP 域同步
all_tokens = HCCL.all_gather(dispatched, tp_group)

# Phase 3: Expert FFN 计算（与通信重叠）
# FFN kernel 切片与下一轮 All-to-All 重叠
for micro_batch in split(local_tokens):
    expert_output = expert_ffn_kernel(micro_batch)  # AI Core 执行
    # HCCL All-to-All combine 与计算流水线重叠
```

论文图 2 显示了 HCCL 各通信算子在 64N/128N/256N 下的算法带宽。随着节点数增加，All-to-All 带宽瓶颈加剧（跨节点 RoCE vs 节点内 HCCS 的带宽差异）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

HCCL 通过 CANN 驱动层与硬件交互，对上提供标准集合通信 API。在 MindSpore 中，HCCL 通信原语通过框架的通信后端自动调用，用户通过配置并行策略（EP/TP/DP）间接使用。HCCL 的算法选择（ring vs mesh vs HD）可由环境变量或配置文件控制，不同算法在不同通信模式和集群拓扑下有各自的性能优势。

涉及论文标题：
- LocMoE: A Low-overhead MoE for Large Language Model Training

## Deterministic FlashAttention Gradient (FAG / 确定性 FlashAttention 梯度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Deterministic FlashAttention Gradient (FAG) 是 LongCat-Flash 训练基础设施中实现的确定性后向传播 kernel。默认 FlashAttention 的 backward pass 使用 atomicAdd 对 dQ/dK/dV 沿不同维度进行归约——原子操作不保证执行顺序，导致同一输入在不同 run 间产生 bitwise 不同的梯度。这种非确定性使得：(1) 训练无法精确复现，(2) SDC (Silent Data Corruption) 检测困难（缺少 bitwise 一致的 baseline）。

LongCat-Flash 的 deterministic FAG 方案：使用有限 extra workspace 按确定性顺序累积各 tile 的部分梯度，替代默认的 atomicAdd unordered reduction。通过 double-buffer pipelining、tuned tiling schedules 和 load balancing 三项协同优化性能。结果：达到原始确定性版本的 1.6x 速度，非确定性版本的 0.95x。

LongCat-Flash 是首批在整个训练 pipeline 中实现端到端确定性的 LLM training system——包括 computation 和 communication 两部分。确定性确保任意 training step 可被多次重跑并产生 bitwise identical loss，使 SDC 检测成为可能。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Deterministic FAG Kernel (简化伪代码)

输入: Q, K, V, dO (BF16 tensors), 所有 shape 已知

// 将 dQ/dK/dV 计算划分为 tiles
tiles = partition(Q.shape, tile_size_per_SM)

// 每个 SM 分配 tiles (load balancing)
for sm_id, tile_set in balanced_split(tiles, num_SMs):
    sm_workspace = alloc_workspace()  // 确定性累加用 workspace

    // Double-buffer pipelining: 两组 buffer 交替使用
    buf_a = alloc_buffer()
    buf_b = alloc_buffer()

    for i, tile in enumerate(tile_set):
        cur_buf = buf_a if i % 2 == 0 else buf_b
        prev_buf = buf_b if i % 2 == 0 else buf_a

        // 异步加载当前 tile 数据 (TMA/async copy)
        load_async(tile.Q, tile.K, tile.V, tile.dO)

        // 如果上一 tile 完成, 将结果写入确定性累积的 workspace
        if i > 0:
            deterministic_accumulate(sm_workspace, prev_buf.dQ)
            deterministic_accumulate(sm_workspace, prev_buf.dK)
            deterministic_accumulate(sm_workspace, prev_buf.dV)

        // 计算当前 tile
        cur_buf.dQ, cur_buf.dK, cur_buf.dV = flash_attention_backward_tile(
            tile.Q, tile.K, tile.V, tile.dO
        )

    // 最后 tile 的归约
    deterministic_accumulate(sm_workspace, cur_buf.dQ)
    deterministic_accumulate(sm_workspace, cur_buf.dK)
    deterministic_accumulate(sm_workspace, cur_buf.dV)

// SM 间合并: 按确定性顺序 (如 SM ID 升序)
for sm_id in sorted(range(num_SMs)):
    merge_into_global_output(global_dQ, sm_workspaces[sm_id].dQ)
    ...
```

关键优化：
- **Deterministic accumulation**: 按 tile 顺序依次累加到 workspace，替代 atomicAdd
- **Double-buffer pipelining**: 当前 tile 计算 + 上一 tile 结果写回重叠执行
- **Tuned tiling**: 按 H800 SM 数量和 shared memory 大小优化 tile 尺寸
- **Load balancing**: 在各 SM 间均匀分配 tile 计算量

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. Workspace 开销：需要存储各 tile 的部分梯度，开销约 O(num_tiles × tile_output_size)，在 H800 80GB 上可接受。
2. 性能 tradeoff：确定性 vs 速度。LongCat-Flash 的 0.95x non-deterministic 性能水平是 SOTA——此前确定性实现通常有 1.5x-2x 减速。
3. 与 SDC 检测的集成：FAG 是最敏感的 SDC 检测点（同时混合 tensor 和 vector 计算），通过 on-chip in-place recomputation 对比 bitwise 结果检测 SDC。
4. 应用于训练 full pipeline：FAG + Deterministic ScatterAdd + 确定性通信（pipelined all-gather/reduce-scatter 代替 all-to-all）共同实现端到端确定性。

涉及论文标题：
- LongCat-Flash Technical Report

## SwapAB MoE GEMM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SwapAB MoE GEMM 是 LongCat-Flash 推理中针对小 batch MoE 解码场景的 GEMM 优化技术。传统 MoE GEMM 使用 token activations 作为左矩阵 A (M×K)、expert weights 作为右矩阵 B (K×N)，公式为 C = A × B。在 decoding 阶段，M（token 数）通常很小（每个 GPU 上可能只有几十个 token），需要 padding 到 M 维度的 64 元素最小对齐（Tensor Core 要求），padding overhead 显著。

SwapAB 反转矩阵角色：将 expert weights 作为左矩阵（N×K）、token activations 作为右矩阵（K×M），利用 N 维度（expert intermediate dim, 通常 2048）的 8 元素对齐粒度。因为 N >> M（在 small batch 下），N 维度的 padding overhead 可忽略。计算 $C^T = B^T \times A^T$ 而非 $C = A \times B$，结果在内存中按需 reinterpret。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// SwapAB MoE GEMM 原理

// 传统 MoE GEMM: C[m, n] = A[m, k] × B[k, n]
//   m = token_count (小 batch 时如 64), 需要 padding 到 64 对齐 → 浪费
//   k = expert_hidden_dim (e.g., 6144)
//   n = expert_intermediate_dim (e.g., 2048)

// SwapAB: C'[n, m] = B_T[n, k] × A_T[k, m]
//   B_T: [n, k] = transpose(expert_weights)  → 左矩阵
//   A_T: [k, m] = transpose(activations)     → 右矩阵
//   n 维度对齐粒度为 8 (vs m 维度的 64) → padding overhead 低得多

// 伪代码:
// 输入:
//   activations: [m, k] (BF16/FP8)
//   weights: [k, n] (BF16/FP8)

// 内存中 reinterpret (无物理转置):
B_T = reinterpret_as([n, k], weights)    // 形状变化，无数据拷贝
A_T = reinterpret_as([k, m], activations) // 形状变化，无数据拷贝

// Tensor Core GEMM: C' = B_T × A_T
C_T = tiled_gemm(B_T, A_T)  // [n, m]

// 内存中 reinterpret 回原始形状:
output = reinterpret_as([m, n], C_T)     // 形状变化，无数据拷贝
```

Swapping 的效果：当 m=64, n=2048 时，传统方法 M 维度无 padding 但需 exact 64；当 m=63 时需 padding 1 个 token（1.6% overhead）；当 m=32 时需 padding 32 个 token（50% overhead）。SwapAB 使 padding 粒度从 M 的 64 降到 N 的 8，大幅减少 small-batch 下的计算浪费。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. 基于 DeepGEMM (https://github.com/deepseek-ai/DeepGEMM) 修改实现。DeepGEMM 设计为 right-hand B matrix 为 weight，SwapAB 反转此约定。
2. 内存 reinterpret 而非物理转置：Python/PyTorch 层面用 `view()` / `as_strided()` 改变形状，GPU kernel 内用指针偏移访问。
3. 适用场景：MoE decoding 阶段（small token count per expert per GPU）。Prefilling 阶段（large token count）M 维度足够大，padding overhead 占比小，SwapAB 的收益递减。
4. 与 quantization 协同：LongCat-Flash 使用 FP8 block-wise quantization (activations [1,128], weights [128,128])，SwapAB 不影响量化方案——quantization/dequantization 发生在 GEMM 之前/之后，与矩阵维度无关。

涉及论文标题：
- LongCat-Flash Technical Report

## NVLink Sharp Communication Kernels (NVSwitch 硬件加速通信 Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

NVLink Sharp Communication Kernels 是 LongCat-Flash 推理系统中基于 NVSwitch 硬件加速的 collective communication 实现。传统 NCCL all-gather/reduce-scatter 依赖 GPU SM（Streaming Multiprocessor）执行数据搬移和 reduction，占用 SM 资源和内存带宽。NVLink Sharp (NVSwitch) 提供硬件加速的原语——broadcast (multimem.st) 和 in-switch reduction (multimem.ld_reduce)——在 NVSwitch 内部完成数据传输和规约，无需大量占用 GPU SM。

LongCat-Flash 使用 inline PTX assembly 直接调用这些硬件指令，实现仅需 4 个 thread blocks 的 all-gather/reduce-scatter kernel。性能在 4KB-96MB message size 全范围超越 NCCL 和 MSCCL++。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// NVLink Sharp All-Gather Kernel (简化 PTX 伪代码)

// 输入: local_data [local_size] 在 GPU i 上
// 输出: full_data [total_size] 在所有 GPU 上

// 1. 每个 GPU 发布自己的数据 (multimem.st)
for offset in range(0, local_size, chunk):
    // inline PTX: 存储本地数据到 NVSwitch 共享内存
    asm volatile(
        "multimem.st [%0], %1;"
        :: "l"(switch_addr + gpu_id * local_size + offset),
           "r"(local_data[offset:offset+chunk])
    );

// 2. 等待所有 GPU 完成发布 (barrier in NVSwitch)
__syncwarp();

// 3. 每个 GPU 读取所有数据 (load from NVSwitch)
for gpu in range(num_gpus):
    for offset in range(0, per_gpu_size, chunk):
        asm volatile(
            "ld.global.ca.b32 %0, [%1];"
            : "=r"(full_data[gpu * per_gpu_size + offset])
            : "l"(switch_addr + gpu * per_gpu_size + offset)
        );

// 仅需 4 thread blocks 执行——其余 SM/thread blocks 可用于计算
```

```
// NVLink Sharp Reduce-Scatter Kernel (multimem.ld_reduce)

// 输入: full_data [total_size] 在 GPU i 上 (部分数据)
// 输出: reduced_data [local_size] 在 GPU i 上 (对应分片的规约结果)

for offset in range(0, local_size, chunk):
    // inline PTX: NVSwitch 从所有 GPU 读取数据并执行 in-switch reduction
    asm volatile(
        "multimem.ld_reduce.add.f32 %0, [%1];"
        : "=f"(reduced_data[offset:offset+chunk])
        : "l"(switch_addr + gpu_id * local_size + offset)
    );
    // NVSwitch 内部: 从所有 GPU 的 offset 位置读取 →
    //   执行 FP32/BF16 reduction (add/min/max) →
    //   返回规约结果到请求 GPU
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. **PTX inline assembly**：需要手动管理寄存器分配和内存对齐。CUDA C++ 层面难以表达 multimem 操作，需要 PTX 级别控制。
2. **Uniform vs Non-uniform token distribution**：LongCat-Flash 的 kernel 支持均匀和非均匀两种 token 分布。非均匀场景（如 imbalanced EP routing）下需要额外的 metadata 传递（各 GPU 的 per-rank 数据量）。
3. **4 thread blocks 效率**：相比 NCCL 需要数十个 thread blocks，仅 4 个 block 大幅减少 SM 占用——剩余 SM 可用于计算（如 Dense FFN），提高 overlap 效率。
4. **适用场景**：(1) ScMoE Dense FFN 的 TP all-gather/reduce-scatter（intra-node NVLink）；(2) 与 MoE inter-node RDMA 通信并发执行（GPUDirect RDMA），最大化网络总利用率。
5. **硬件依赖**：仅在 Hopper 架构 (H100/H800) 且有 NVSwitch 的系统中可用（如 HGX H100 8-GPU baseboard）。论文未明确说明是否支持其他架构。

涉及论文标题：
- LongCat-Flash Technical Report

## Fused GemmAdd (融合的 GEMM 加法)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fused GemmAdd 是 LongCat-Flash 训练基础设施中将 FP32 gradient accumulation 融合到 Grouped GEMM epilogue 中的 kernel 优化。在 MoE training backward pass 中，dw（weight gradient）计算通过 GEMM 完成，随后需要与 optimizer state 或 existing gradient 做 FP32 加法规约——这个加法步骤原本作为独立 kernel，产生额外的 HBM write-back 和 re-read，成为 bandwidth-bound 瓶颈。

Fused GemmAdd 将 FP32 addition 嵌入到 GEMM 的 epilogue 阶段（在 Tensor Core 输出数据还未写入 HBM 前、仍在寄存器/SMEM 中时完成加法），消除中间 write-back，并通过 tile GEMM pipeline 隐藏加法延迟。此外避免 BF16 数据写入 HBM 后重新读取时的精度损失。LongCat-Flash 在 fused GroupedGemmAdd benchmark 上取得 3.12x-3.86x 加速。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Fused GemmAdd 原理

// 传统流程 (2 kernels):
// Kernel 1: GEMM
C_bf16 = GEMM(A_bf16, B_bf16)  // BF16 GEMM 输出 [m, n] in HBM
// Kernel 2: Accumulate (独立 kernel, bandwidth-bound)
C_fp32 = convert_to_fp32(C_bf16)  // Load from HBM
C_fp32 += existing_gradient_fp32   // FP32 add (HBM → Register → HBM)

// Fused GemmAdd 流程 (1 kernel):
// GEMM + Addition fused in epilogue:
for each tile in tiles:
    // Tensor Core: C_tile_bf16 = A_tile_bf16 × B_tile_bf16
    C_tile_bf16 = tc_mma(A_tile_bf16, B_tile_bf16)

    // Epilogue (融合阶段):
    // 不写回 HBM，直接在寄存器/SMEM 中完成:
    C_tile_fp32 = bf16_to_fp32(C_tile_bf16)        // 精度提升
    existing_fp32 = load_from_hbm(existing_gradient, tile_offset)  // 加载已有梯度
    C_tile_fp32 += existing_fp32                     // FP32 加法
    store_to_hbm(C_tile_fp32, tile_offset)           // 写回 HBM

    // 下一 tile 的 GEMM 与当前 tile 的 epilogue 通过 double-buffer 重叠
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. Epilogue fusion：需要修改 GEMM kernel 的 epilogue 阶段。CUTLASS/Triton 均支持自定义 epilogue（如 bias addition, activation, quantization），GemmAdd 本质上是 "+ existing gradient" 的 epilogue 算子。
2. 精度保持：FP32 epilogue 避免了 BF16 → HBM 写回 → BF16 → FP32 转换的精度损失。MoE training 对梯度精度敏感（专家数量多、每个 expert 的 token 少 → per-expert gradient magnitude 小）。
3. 适用场景：(1) Grouped GEMM 的 epilogue（每个 expert 独立梯度累加）；(2) ScatterAdd 替代——某些场景下 fused GEMM+add 可替代 ScatterAdd 的梯度聚合。

涉及论文标题：
- LongCat-Flash Technical Report

## Deterministic ScatterAdd (确定性分散累加 Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Deterministic ScatterAdd 是 LongCat-Flash 训练基础设施中的确定性梯度聚合 kernel。ScatterAdd 在 MoE backward pass 中承担关键角色——将各 expert 处理的 token 梯度按原始 token 位置聚合回去。默认 CUDA 实现因 input-output operand count 不匹配（多个 expert 可能向同一 token 位置写入梯度），强制单 compute unit 串行执行，导致最高 50x 减速。

LongCat-Flash 的 Deterministic ScatterAdd 使用 hierarchical reduction algorithm：先将梯度按 token 位置分组，然后在各 processor 间并行规约，再按确定性顺序合并。结果在保证确定性的同时，性能达到与非确定性版本持平。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Deterministic ScatterAdd 原理

// 输入:
//   grad_per_expert: [num_experts, num_tokens_per_expert, d_model]  # 各 expert 产生的梯度
//   token_to_expert_map: [num_experts, num_tokens_per_expert]  # token → expert routing info
//   original_token_order: [batch, seq_len]

// Step 1: 按 destination token 分组 (parallel across processors)
for proc_id in range(num_processors):
    local_buckets = [[] for _ in range(max_token_id)]
    for expert_grad, token_id in my_assigned_range:
        local_buckets[token_id].append(expert_grad)

// Step 2: 每个 processor 内规约
for token_id in local_buckets.keys():
    local_reduced[token_id] = sum(local_buckets[token_id])

// Step 3: Processor 间按确定性顺序合并 (hierarchical reduction)
// 而不是 atomicAdd (非确定性)
sorted_procs = sort_by_id(processors)  // 确定性顺序
for token_id in range(max_token_id):
    result = zeros(d_model)
    for proc_id in sorted_procs:
        result += local_reduced[proc_id][token_id]
    output[token_id] = result
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. 与默认 CUDA ScatterAdd（如 `scatter_add_()`）单 compute unit 串行执行对比，hierarchical reduction 将工作分配到所有 processor → 消除 50x 减速。
2. 确定性保证：按 processor ID 升序合并（而非依赖硬件 timing），确保相同输入在不同 run 下 bitwise 一致。
3. 在 LongCat-Flash 中的地位：与 Deterministic FAG 一起构成端到端确定性训练的 backward pass 组件。
4. 通用性：hierarchical reduction 思想不仅适用于 MoE token 梯度聚合，也可用于其他以 ScatterAdd 为 bottleneck 的操作。

涉及论文标题：
- LongCat-Flash Technical Report

## Fused Triton Kernel for MoE Router Post-Processing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fused Triton Kernel for MoE Router Post-Processing 是 LYNX 用于在 MoE 推理的 critical path 中高效执行 batch 级 expert selection 的 GPU kernel 实现。LYNX 将 confidence analysis、adaptive expert scoring、expert pruning 和 expert remapping 四个步骤融合为 4 个 Triton kernel，替代原本需要超过 700 个 PyTorch 小算子的 naive implementation。

四个 kernel 的分工：
1. **Kernel 1 (Token-wise Binning)**：对 batch 中所有 token 并行计算 log-ratio 并做 AffinityBinning 离散化，同时计算 top-k weight sums
2. **Kernel 2-3 (Batch-wise Scoring & Expert Pruning)**：对每个 expert 做 batch 级别指数加权评分，基于分数分布动态确定 active expert 集
3. **Kernel 4 (Expert Remapping & Compaction)**：将 low-confidence token 重映射到 reduced expert set，compaction 重排映射表，renormalize weights 并重新计算 top-k

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
┌── LYNX Kernel Launch Sequence (per MoE layer, decode iteration) ─┐
│                                                                    │
│  [Kernel 1: Token-wise Binning]                                   │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Grid: (B, ) 即 batch_size 个 thread blocks                    │ │
│  │ 每个 block 处理 1 个 token:                                    │ │
│  │                                                                │ │
│  │ %token_logits = load(router_logits + token_id * N)  // [N]    │ │
│  │ top1_logit = max(%token_logits)                    // scalar  │ │
│  │ for e in topk_indices[token_id]:                              │ │
│  │     log_ratio = %token_logits[e] - top1_logit                 │ │
│  │     bin[token_id][e] = clamp(floor(log_ratio * α), -β, 0)    │ │
│  │ store(bin_out + token_id * k, bin[token_id])                  │ │
│  │                                                                │ │
│  │ Fusion: subtract + multiply + floor + clamp → 1 kernel       │ │
│  │ Replaces: ~200 PyTorch ops (index_select, sub, mul,           │ │
│  │           floor, clamp, scatter)                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  [Kernel 2-3: Batch-wise Scoring & Expert Pruning]                │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Kernel 2: Score accumulation                                  │ │
│  │ Grid: (N, ) 即 N 个 thread blocks (每个 expert 一个)          │ │
│  │ 每个 block:                                                     │ │
│  │   score = 0.0                                                  │ │
│  │   for t in range(B):                                          │ │
│  │       for rank in range(k):                                   │ │
│  │           if topk_idx[t][rank] == expert_id:                  │ │
│  │               score += pow(B, bin[t][rank])  // B^{bin}      │ │
│  │   store(scores_out + expert_id, score)                        │ │
│  │                                                                │ │
│  │ Kernel 3: Threshold & Pruning                                 │ │
│  │ Grid: (1, )  single block                                     │ │
│  │   sorted_scores = sort(scores, descending=True)               │ │
│  │   threshold = determine_by_distribution(sorted_scores,       │ │
│  │                bin_width, max_bins)                            │ │
│  │   active_mask = scores >= threshold                           │ │
│  │   store(active_mask_out, active_mask)                         │ │
│  │                                                                │ │
│  │ Fusion: reduce + pow + sort + threshold → 2 kernels          │ │
│  │ Replaces: ~300 PyTorch ops                                    │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  [Kernel 4: Expert Remapping & Compaction]                        │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Grid: (B, )  batch_size 个 thread blocks                      │ │
│  │ 每个 block 处理 1 个 token:                                    │ │
│  │                                                                │ │
│  │   // 对 high-confidence token: 保留 origin top-k               │ │
│  │   // 对 low-confidence token: remap lower-ranked experts      │ │
│  │   for rank in range(k):                                       │ │
│  │       if confidence(token) >= threshold OR rank == 0:         │ │
│  │           new_expert[rank] = original_topk[rank]              │ │
│  │       else:                                                    │ │
│  │           new_expert[rank] = find_best_alt_in_active_set(...) │ │
│  │                                                                │ │
│  │   // Compaction: 将 sparse expert indices 映射为 dense        │ │
│  │   compact_expert = active_expert_map[new_expert]              │ │
│  │                                                                │ │
│  │   // Renormalize: 重新计算 softmax                            │ │
│  │   new_weights = softmax(router_logits[compact_expert])        │ │
│  │   store(mapping_out, compact_expert)                          │ │
│  │   store(weights_out, new_weights)                             │ │
│  │                                                                │ │
│  │ Fusion: gather + scatter + softmax + topk → 1 kernel         │ │
│  │ Replaces: ~250 PyTorch ops                                    │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  总计: 4 Triton kernels 替代 700+ PyTorch ops                      │
│  Overhead: <4% 总体 decode latency                                 │
│  CUDA Graph 兼容: 所有 kernel 保持静态控制流                        │
└────────────────────────────────────────────────────────────────────┘
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LYNX 的 4 个 fused kernel 使用 Triton 语言编写，编译为 CUDA PTX。关键设计选择：(1) 每个 kernel 将数据保持在 registers 或 shared memory 中，消除 intermediate tensor 的 global memory 读写；(2) 静态控制流确保 CUDA Graph capture 兼容——这是 vLLM 等 serving engine 的关键优化需求；(3) 4 个 kernel launch 的开销远小于 700+ 个细粒度 PyTorch kernel launch 的累积开销（每个 launch ~5-10μs）；(4) Kernel 参数（α, β）在模型加载时计算一次，作为 kernel constant 传入。

涉及论文标题：
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection

## Expert Cache with Priority-based Eviction（优先级专家缓存与驱逐）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Cache with Priority-based Eviction 是 FineMoE 在 GPU memory 中管理 expert weights 的缓存机制，包括两个优先级计算：**prefetching priority**（决定哪些 experts 优先从 CPU 加载到 GPU）和 **eviction priority**（决定 GPU cache 满时驱逐哪些 experts）。两种 priority 均基于 searched expert map 中的概率分布 p_{l,j} 计算，使 GPU cache 在有限的显存约束下（6GB-96GB）最大化 expert hit rate。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Expert Prefetching Priority:
PRI^{prefetch}_{l,j} = p_{l,j} / (l - l_now)

其中:
  p_{l,j}: searched expert map 中 expert j 在 layer l 的被选概率
  l - l_now: 距离当前 layer 的层数
  → 近层 (small l-l_now) + 高概率 (large p) = 高优先级

Expert Eviction Priority:
PRI^{evict}_{l,j} = 1 / (p_{l,j} × freq_{l,j})

其中:
  p_{l,j}: searched expert map 中的概率
  freq_{l,j}: expert 被访问的历史频率 (LFU-like)
  → 低概率 (small p) + 低频 (small freq) = 高 eviction 优先级 (先被踢出)

GPU Cache 管理流程（每次专家访问时）:

# Prefetching
for each E_{l,j} in E_prefetch:
    priority = p_{l,j} / (l - l_now)
    插入 GPU task pool (按 priority 排序的 priority queue)

# GPU Task Pool 异步执行:
while task_pool not empty:
    task = pop_highest_priority()
    cudaMemcpyAsync(host_ptr→device_ptr, expert_size, stream=prefetch_stream)
    ExpertCache[expert_id] = device_ptr

# Eviction (Cache 满时)
while GPU_cache_memory > budget:
    worst_expert = argmax_{E_{l,j} in cache} 1/(p_{l,j} × freq_{l,j})
    cudaFree(ExpertCache[worst_expert])
    从 ExpertCache 中移除 worst_expert

# On-Demand Loading (Expert Miss, 最高优先级)
if expert_not_found in ExpertCache:
    暂停所有 pending prefetch tasks
    cudaMemcpyAsync(host→device, expert_weights, stream=on_demand_stream)
    synchronize(on_demand_stream)  # 等待加载完成
    恢复 prefetch tasks
```

关键设计选择：
- 不使用 LRU (Least Recently Used)：expert usage 是 layer-wise sequential 的（一层接一层），"recently used" 的 experts 不会再被近期使用（因已跳过该层），LRU 不适合 expert offloading 场景。
- LFU + probability：在 MoE-Infinity LFU 基础上集成 expert map probability，使高频 + 高概率 experts 获得最强缓存保护。
- On-demand loading 可抢占 prefetch：expert miss 直接影响 forward 能否执行，优先级高于 speculative prefetch。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FineMoE 基于 MoE-Infinity 代码库，Expert Cache 以 C++ CUDA Runtime API 实现。GPU task pool 使用 C++ 异步线程 + CUDA streams 管理。Expert to GPU mapping 使用 hash map (Python dict → C++ map)，multi-GPU 场景下按 round-robin 分配 experts 到不同 GPUs 以均衡负载。Prefetch/eviction priority 在每次 expert map search 完成时更新，确保缓存策略实时反映最新的 prediction confidence。

实验表明 FineMoE 的 priority-based caching 在 expert hit rate 上超越纯 LFU 和 LRU（图 14b 消融实验）。在 limited GPU cache（6GB）场景下效果最显著：TPOT 比 MoE-Infinity (LFU) 降低 29%。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

## Asynchronous Expert Prefetching in GPU Task Pool（GPU 异步专家预取任务池）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Asynchronous Expert Prefetching in GPU Task Pool 是 FineMoE 的 GPU-side 专家预取执行机制：Expert Map Searcher 确定需要预取的 expert 集合后，将 prefetch 任务提交到 GPU space 的 task pool（priority queue），由异步线程调度 CUDA async memory copy 将 expert weights 从 CPU 传输到 GPU。任务按 prefetching priority 排序执行。关键设计：prefetch tasks 与 inference computation 使用独立的 CUDA streams，使 CPU→GPU 数据传输与 forward pass 重叠执行。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
GPU Task Pool 执行流程（CUDA Stream 级别）:

CUDA Stream 分配:
  stream_compute: inference forward pass (attention, gate, expert GEMM)
  stream_prefetch: async CPU→GPU expert weight transfer
  stream_ondemand: emergency on-demand loading (expert miss)

Task Pool 数据结构:
  task_pool = PriorityQueue[
    {expert_id: "l_j", priority: p/(l-l_now), size: expert_weight_bytes, action: "prefetch"},
    ...
  ]

异步执行流程:
┌─ stream_compute ──────────────────────────────────────────────┐
│ Layer 1: attn → gate → expert compute                        │
│ Layer 2: attn → gate → expert compute                        │
│ Layer 3: attn → gate → expert compute                        │
│ ...                                                           │
└───────────────────────────────────────────────────────────────┘
    ▲ (不等待 prefetch 完成)

┌─ stream_prefetch ─────────────────────────────────────────────┐
│ cudaMemcpyAsync(host→dev, expert_L4_w0, stream_prefetch)     │
│ cudaMemcpyAsync(host→dev, expert_L5_w1, stream_prefetch)     │
│ ...                                                           │
└───────────────────────────────────────────────────────────────┘

Expert Miss 处理 (抢占机制):
  if 当前 layer 需要的 expert 不在 cache:
    1. 暂停 stream_prefetch 上的所有 pending tasks
    2. 在 stream_ondemand 上: cudaMemcpyAsync(host→dev, missed_expert)
    3. synchronize(stream_ondemand) → forward 该层
    4. 恢复 stream_prefetch 上的 pending tasks
```

与 MoE-Infinity synchronous prefetching 对比：
- MoE-Infinity: 每层 forward 前同步等待 expert prediction + prefetch → prefetch latency 直接加在 critical path
- FineMoE: forward 与 prefetch 异步重叠 → prefetch latency 不进入 critical path → overhead <1% iteration time

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FineMoE 的 GPU task pool 基于 MoE-Infinity C++ CUDA Runtime API 实现。Multiple CUDA streams 用于分离 compute 和 data transfer。Task priority queue 使用 C++ std::priority_queue 实现，按 PRI^{prefetch} = p/(l-l_now) 排序。On-demand loading 通过 flag 机制抢占：设置 global flag 暂停 prefetch dispatcher，等待 on-demand load 完成后清除 flag 恢复。此设计使 FineMoE 即使在高 expert miss rate 场景下也能最小化 on-demand loading 对 critical path 的影响。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading


## FusedMoE (vLLM Fused MoE Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FusedMoE 是 vLLM 推理框架中针对 MoE 模型的高性能融合 kernel，将 MoE 层的多个操作（token routing、expert dispatch、grouped GEMM、activation、weighted combine）融合为少量 GPU kernel 调用，减少 kernel launch overhead 和 HBM 访存次数。核心实现包括 Triton-based grouped GEMM（TritonExperts）和 CUTLASS/DeepGemm 等多种 backend。FusedMoE 是 vLLM 高效支持 Mixtral、DeepSeek-V2、Qwen-MoE 等 MoE 模型推理的基础组件。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// FusedMoE Kernel 调度流程
// 输入: hidden_states [B, L, H], 输出: expert_output [B, L, H]

// Step 1: Gate + Top-K Routing (GPU kernel, lightweight)
gate_logits = matmul(hidden_states, W_gate)        // [B, L, N_experts]
topk_vals, topk_idx = topk(gate_logits, k)          // 选 top-k experts
topk_weights = softmax(topk_vals)                   // 归一化

// Step 2: Token-to-Expert Sorting (Triton kernel)
// moe_align_block_size: 按 expert ID 排序 tokens
// 将同一 expert 的 tokens 分组到连续的 BLOCK_SIZE_M 块中
sorted_tokens, expert_offsets = sort_by_expert(
    hidden_states, topk_idx, block_size=64
)

// Step 3: Grouped GEMM - W1 (Triton Grouped GEMM kernel)
// 每个 expert 独立执行一次 GEMM，但 batch 在一起减少 kernel launch
// expert_i: sorted_tokens[offset_i:offset_{i+1}] @ W1_i
for expert_i in active_experts:
    h_i = sorted_tokens[offset_i:offset_{i+1}]  // [N_i, H]
    inter_i = h_i @ W1_i.T                       // [N_i, 4H] (gate+up projection)
    
// Step 4: SiLU Activation (fused in same kernel)
    gate, up = split(inter_i, 2)                  // gate + up projection
    act_i = up * silu(gate)                       // SwiGLU activation

// Step 5: Grouped GEMM - W2 + Reduce (Triton Grouped GEMM kernel)
    out_i = act_i @ W2_i.T                        // [N_i, H]
    // Scatter back to original token positions
    output[expert_i_tokens] += topk_weights[expert_i_tokens] * out_i
```

LExI 在 FusedMoE 上的修改最小：仅改变每层的 top-k 参数。减少 k 值直接减少 Step 2-5 的处理 token 数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

vLLM FusedMoE 位于 `vllm/model_executor/layers/fused_moe/`，核心类 `FusedMoE` 支持多种 kernel backend（Triton/CUTLASS/FlashInfer/Marlin/DeepGemm），通过 `FusedMoE.select_experts_implementation()` 自动选择。支持 FP16/BF16/FP8/INT4/INT8 量化。vLLM v0.12 引入 MoE chunking：将长序列 tokens 分块，允许 expert 计算与 all-to-all 通信重叠执行。对于 DeepSeek-V2/V3 的 shared expert，`SharedFusedMoE` 支持 shared expert 与 routed expert dispatch 的 overlap 执行。LExI 在此基础上的修改：加载模型后，修改每个 MoE layer 的 `self.top_k` 参数为进化搜索得到的 k_j 值。

涉及论文标题：
- LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference

## Irregular All-to-All (All-to-Allv) for MoE Partitioning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Irregular All-to-All（MPI 术语中称为 All-to-Allv）是 MoE 分布式训练中的通信模式，其中每个设备向不同目标设备发送不同数量的数据（非均匀 partition）。在 Lancet 的算子分区方案中，input batch 沿 batch 维度分区为 micro-batch，每个 micro-batch 经过 gating 后，向某个 expert 发送的 token 数从 0 到 C（expert capacity）不等，但所有 partition 的总 token 数之和等于 C。这种不规则性源于：special gating operator 在 partition 间传递容量信息——当第一个 partition 使用 3/4 C 容量时，后续 partition 动态调整 remaining capacity 为 1/4 C。Lancet 使用双趟 All-to-All 实现：第一趟交换各 GPU 间实际传输的 data size，第二趟按已知 size 传输实际数据。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Lancet 的 Irregular All-to-All 实现（论文 Fig. 10）：

```
# 设 G 个 GPU 参与 all-to-all, 每 GPU 分配 E^l 个 expert (G = E/E^l)
# Input Buffer: [G, C] 固定形状, 仅部分填充
# Output Buffer: [G, C] 固定形状

def irregular_all_to_all(tokens_by_expert, gating_result):
    # tokens_by_expert[g][e]: GPU g 上属于 expert e 的 tokens (variable size)
    # gating_result 包含每个 (src_gpu, dst_expert) 的 token count
    
    # Phase 1: 交换 data sizes
    # send_sizes[g] = [count_0, count_1, ..., count_{G-1}]  (发给每个 GPU 的 token 数)
    send_sizes = compute_send_sizes(tokens_by_expert)
    recv_sizes = all_to_all_sizes(send_sizes)  # 第一趟 all-to-all: 只交换 size 信息
    
    # Phase 2: 传输实际数据
    # 基于 recv_sizes 知道从每个 src 收多少数据
    # 基于 send_sizes 知道向每个 dst 发多少数据
    for dst_gpu in range(G):
        if send_sizes[dst_gpu] > 0:
            ncclSend(tokens_buffer[dst_gpu], size=send_sizes[dst_gpu], target=dst_gpu)
    for src_gpu in range(G):
        if recv_sizes[src_gpu] > 0:
            ncclRecv(output_buffer[src_gpu], size=recv_sizes[src_gpu], source=src_gpu)
    
    return received_tokens
```

与 Uniform All-to-All 的对比：

```
# Uniform: 每 GPU 向每 GPU 发送固定 C 个 token
# 总通信量: G * G * C * token_size

# Irregular: 每 GPU 向 GPU g 发送 s_g 个 token, Σ s_g = G*C
# 总通信量: G * C * token_size (实际数据) + G*G*sizeof(int) (size info)
# 不规则的总通信量更低（不传输 padding tokens）
```

Pipeline 调度中的不规则 All-to-All：

```
# Pipeline Stage 中的 3-partition 例子:
# Partition 0: NonMoE₀ → [IrregA2A₀ out] → ...
# Partition 1:        NonMoE₁ → [IrregA2A₁ out] → ...
# Partition 2:               NonMoE₂ → [IrregA2A₂ out] → ...

# IrregA2Aₖ 的通信量取决于 gating output 在 partition k 中的 token 分布
# PipelineScheduler 使用 static-shape approximation (C/k) 预估时间
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Lancet 基于 NCCL Send/Recv primitives 实现（grouped communication），不使用 `ncclAllToAll`。Input/Output buffer 按最大容量（C×G）静态分配，运行时仅部分填充。该实现不传输 padding tokens，因此总通信量可低于 uniform all-to-all。类似的 irregular all-to-all 实现在 DeepSpeed-MoE、FasterMoE、Lina、Tutel 等系统中也有出现，各自有不同的优化策略（如 Tutel 的 2D-Hierarchical All-to-All 利用 NVLink intra-node 和 network inter-node 的分层拓扑）。Lancet 的 static-shape approximation 虽不能精确预测不规则 all-to-all 的绝对时间，但误差仅 3.83%，足够引导 DP 搜索选择正确的 partition range。

涉及论文标题：
- Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping
