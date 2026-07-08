## Async Copy Backend（异步拷贝后端）

术语是什么？

Async Copy Backend 是 Pipelined Sharding 中实现 CPU compute 与 GPU weight streaming 重叠执行的关键机制。它利用 CUDA 的异步内存拷贝（`cudaMemcpyAsync`）将模型权重从 pinned sysRAM 传输到 GPU scratch buffer，同时 CPU 线程池继续执行 CPU-resident layers 的计算。Async Copy Backend 与 CUDA stream 管理协同：weight transfer 在独立的 copy stream 上执行，GPU kernel 在 compute stream 上执行，通过 CUDA event 同步。该机制是 Dynamic Plan 相比 Static Plan 获得额外性能提升的来源。

从 kernel 调度角度拆解术语：

Async Copy Backend 的执行伪代码（以 Dynamic Plan 下 decode 一步为例）：

```
// CPU 线程池执行 CPU-resident layers
cpu_thread_pool:
  for layer in cpu_layers:
    output = cpu_kernel(layer.weights_cpu, input)

// 同时，copy stream 异步传输下一个 GPU-resident layer 的权重
cuda_stream_copy:
  for layer in gpu_layers:
    cudaMemcpyAsync(layer.scratch_gpu, layer.weights_pinned, size, H2D, stream_copy)

// GPU compute stream 等待权重就绪后执行
cuda_stream_compute:
  for layer in gpu_layers:
    cudaEventSynchronize(event_weights_ready[layer])
    gpu_kernel<<<grid, block, 0, stream_compute>>>(layer.scratch_gpu, input, output)
    cudaEventRecord(event_compute_done[layer], stream_compute)

// 主线程同步：等待 CPU 和 GPU 均完成
cpu_thread_pool.wait()
cudaStreamSynchronize(stream_compute)
```

术语一般如何实现？如何使用？

在 llma.cpp 中，Async Copy Backend 使用 Pinned Memory（`cudaMallocHost`）作为 sysRAM 侧的 weight buffer，以保证 DMA 传输的最大带宽。Copy stream 通常使用非默认 CUDA stream 以与默认 compute stream 并行。Weight 传输以 layer 为粒度（而非整个模型），允许 fine-grained pipelining：当 GPU 正在执行 layer_i 时，copy stream 已开始传输 layer_{i+1} 的权重。论文 Fig. 16 展示了 PCIe 代际（Gen4 vs Gen5）对性能的显著影响，验证了 async copy 在带宽受限场景下的关键性。

涉及论文标题：
- Efficient, VRAM-Constrained Cross-Lingual Model Inference on Client Devices

---

## Vision Tensor Offload（视觉张量卸载）

术语是什么？

Vision Tensor Offload 是 VLMOpt 中的 VLM 显存优化技术：将 CLIP/ViT vision encoder 的权重固定在 sysRAM（CPU 内存），仅在图像编码时按需 stream 到 GPU。由于 vision encoder 仅在 prefill 阶段执行一次（单张图像），其权重无需常驻 VRAM。该技术与 Serialized Vision Teardown 协同：vision encoder 完成后权重 scratch buffer 立即释放，归还 VRAM 给 language model。对 CR1（Cosmos-Reason1），vision encoder 权重约 2GB，offload 后节省等量 VRAM。

从 kernel 调度角度拆解术语：

Vision Tensor Offload 的执行流程：
1. 模型加载时：vision encoder 权重仅分配在 pinned sysRAM，不在 GPU 分配持久存储。
2. 图像 prefill 阶段：
   - 分配 GPU scratch buffer（大小 = 单层 vision encoder 最大权重）
   - 逐层 stream vision weights（sysRAM → GPU scratch）
   - 执行该层的 GPU kernel（conv/attention/LN）
   - 释放中间激活，重用 scratch buffer 给下一层
3. Language model 阶段：vision scratch buffer 完全释放，GPU 显存全部用于 language model。

术语一般如何实现？如何使用？

在 llma.cpp 中，Vision Tensor Offload 复用与 Pipelined Sharding 相同的 Async Copy Backend 和 pinned memory 基础设施。与 LLM 的 Dynamic Plan 不同，vision offload 不需要 CPU/GPU compute overlap（vision encoder 全在 GPU 执行），重点是 minimization of VRAM residency。论文未详细说明 vision encoder 内部的 layer-by-layer offload 粒度。

涉及论文标题：
- Efficient, VRAM-Constrained Cross-Lingual Model Inference on Client Devices

---

## Tiled FlashAttention in Vision Encoder（视觉编码器中的分块 FlashAttention）

术语是什么？

Tiled FlashAttention 是 VLMOpt 中将标准 FlashAttention 的分块策略应用于 vision encoder self-attention 的技术。Vision encoder（特别是高分辨率 ViT）的 self-attention 产生 O(N²) 大小的 KQ 中间张量，其中 N = (image_height/patch_size) × (image_width/patch_size)。对于 1440p 图像（2560×1440），使用 patch_size=14 时 N ≈ 18,800，KQ 张量可达数 GB（远超 VRAM budget）。Tiled FlashAttention 对 Q 维度进行 tile（分块），每次仅加载一个 Q-tile 到 SRAM/VRAM，将其与完整 K 做 attention，限制峰值中间张量大小。论文声称 1440p VLM attention 峰值显存降至 2 GB 以下。

从 kernel 调度角度拆解术语：

Tiled FlashAttention 的计算流程（以 ViT self-attention 为例）：

```
// 标准 FlashAttention 对 Q 做 outer loop tile
for q_tile in tiles(Q, tile_size_q):    // Q: [N, d_head]
    load q_tile to SRAM                  // [tile_size_q, d_head]
    o_tile = zeros(tile_size_q, d_head)
    l_tile = zeros(tile_size_q)
    m_tile = -inf

    for kv_tile in tiles(KV, tile_size_kv):  // K,V: [N, d_head]
        load kv_tile to SRAM
        s = q_tile @ K_tile^T               // [tile_size_q, tile_size_kv]
        s = s * scale
        m_new = max(m_tile, rowmax(s))
        p = exp(s - m_new)
        l_new = exp(m_tile - m_new) * l_tile + rowsum(p)
        o_tile = diag(exp(m_tile - m_new)) * o_tile + p @ V_tile
        m_tile = m_new
        l_tile = l_new

    o_tile = diag(1/l_tile) * o_tile
    store o_tile to HBM
```

关键参数 `tile_size_q` 决定峰值显存：tile_size_q 越小显存越低但 tile 数越多（更多 HBM↔SRAM 往返）。论文未给出具体 tile_size_q 值。

术语一般如何实现？如何使用？

在 llma.cpp 中，Tiled FlashAttention 修改 vision encoder 的 attention kernel，增加 Q-tile outer loop。该修改仅影响 vision encoder（不影响 language model decoder 的 FlashAttention），且仅在图像分辨率超过阈值时启用。论文声称 small perf cost（tile 往返开销），但未给出具体 overhead 数值。

涉及论文标题：
- Efficient, VRAM-Constrained Cross-Lingual Model Inference on Client Devices

---

## Roofline Model for Schedule Selection（面向调度选择的 Roofline 模型）

术语是什么？

Roofline Model for Schedule Selection 是 Pipelined Sharding 的 Planner 中用于估算每种 plan cost 的性能模型。传统 roofline model 将 kernel 分类为 compute-bound 或 memory-bound，此处 Planner 扩展 roofline 以建模异构 CPU-GPU 执行：对每个 kernel，基于 profile database 中的实测延迟（而非理论峰值 FLOPS/bandwidth）估算在 CPU 和 GPU 上分别的执行时间，再叠加 PCIe weight transfer 时间。Planner 使用此 cost model 比较 GPU-only / Static / Dynamic 三种 plan 的总延迟，选择最低 cost plan。

从 kernel 调度角度拆解术语：

Planner 的 cost estimation 公式：

```
Cost(plan, tier) = Σ layer_cost(layer, plan, tier)

layer_cost(layer, GPU-only):
    = profile_db[layer.op][GPU].latency
    + weight_size / PCIe_effective_bw   // weight streaming from sysRAM

layer_cost(layer, Static):
    = min(profile_db[layer.op][CPU].latency_with_contention,
          profile_db[layer.op][GPU].latency)
    + (intermediate_size / PCIe_effective_bw if cross-device else 0)

layer_cost(layer, Dynamic):
    = Static_cost
    - overlap_savings  // 估算 CPU compute || GPU streaming overlap 可隐藏的延迟
```

其中 `overlap_savings` 估计为 `min(CPU_compute_time, GPU_stream_time) * overlap_efficiency_factor`。

术语一般如何实现？如何使用？

Planner 中的 roofline model 不追求绝对精度（不需要精确预测 latency），只需正确排序三种 plan 的相对 cost 即可。论文 Fig. 17 显示 scheduler picks vary across configurations，验证了 cost model 能正确区分不同硬件配置下的最优 plan。论文未详细说明 `overlap_efficiency_factor` 的校准方法。

涉及论文标题：
- Efficient, VRAM-Constrained Cross-Lingual Model Inference on Client Devices

## CUDA Multi-Process Service (MPS)

术语解释
CUDA Multi-Process Service (MPS) 是 NVIDIA 提供的 GPU 多进程共享技术，允许多个 CPU 进程同时向同一 GPU 提交 CUDA kernel，通过硬件级的 context sharing 实现低开销的 GPU 资源共享。相比传统的多进程时间分片（time-slicing），MPS 通过将多个进程的 CUDA context 合并为一个共享 context，消除了 context switch 的开销（~10-20μs per switch），实现了 kernel 级并发执行。

术语是什么？
MPS 的核心工作机制：
1. **MPS Server**：一个系统级 daemon 进程，管理 GPU 资源的共享
2. **MPS Client**：每个用户进程通过 libcuda 连接到 MPS Server，提交 CUDA 操作
3. **Shared Context**：所有 client 的 GPU 操作在同一个 CUDA context 中执行，volatile GPU state（如 local memory、barrier）被分区保护
4. **Concurrent Kernel Execution**：不同进程的 kernel 可在 GPU 的不同 SM 上并发执行，或通过 CUDA streams 实现 overlap

从kernel调度角度拆解：
PRISM 中 MPS 的使用场景（computation + I/O 双进程架构）：

```
Process Architecture:
  ┌─────────────────────────┐    ┌──────────────────────┐
  │ Computation Process     │    │ I/O Process           │
  │ (PyTorch, Python)       │    │ (Libuv, C)            │
  │                         │    │                       │
  │ CUDA MPS Client #1      │    │ CUDA MPS Client #2    │
  │  ├─ cudaStreamCompute   │    │  ├─ cudaStreamH2D     │
  │  │   (cuBLAS GEMM)      │    │  │  (weight prefetch)  │
  │  │   (attention kernel) │    │  │  (hidden state I/O) │
  │  └─ cudaStreamD2H       │    │  └─ ...               │
  │      (output transfer)  │    │                       │
  └──────────┬──────────────┘    └──────────┬────────────┘
             │                              │
             └──────────┬───────────────────┘
                        ▼
              ┌─────────────────┐
              │  MPS Server     │
              │  Shared Context │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  NVIDIA GPU SM  │
              │  Concurrent     │
              │  Kernel Exec    │
              └─────────────────┘

时间线（单层前向中的 kernel overlap）：

Computation Process              I/O Process              GPU SM
     │                               │                      │
     ├─ layer_i GEMM ────────────────┤                      ├─ GEMM tiles
     │  (streamCompute)              │                      │
     │                               ├─ cudaMemcpyAsync ────┤ H2D: L_i+1 weights
     │                               │  (streamH2D)         │   (DMA engine)
     ├─ layer_i attention ───────────┤                      ├─ attn kernel
     │  (streamCompute)              │                      │
     │                               ├─ cudaMemcpyAsync ────┤ D2H: prev hidden states
     │                               │  (streamD2H)         │   (DMA engine)
     ├─ layer_i FFN ─────────────────┤                      ├─ FFN GEMM
     │  (streamCompute)              │                      │
     ▼                               ▼                      ▼
```

无 MPS 时的替代方案与对比：
| 方案 | GPU 共享方式 | Context Switch | Max Concurrent Processes |
|------|-------------|----------------|--------------------------|
| Time-slicing (default) | 时间片轮转 | ~10-20μs/switch | 无限制（串行） |
| MPS | Shared context | ~0 (kernel concurrent) | 48 (NVIDIA limit) |
| MIG (A100/H100) | 物理分区 | N/A | 7 (A100) / compute-slice partitions |
| CUDA Streams (单进程) | 同 context, streams | N/A | 1 process, N streams |

术语一般如何实现？如何使用？
- 启用方式：
  ```bash
  # 启动 MPS daemon
  nvidia-cuda-mps-control -d
  
  # 验证状态
  nvidia-smi -q | grep -A 10 "Processes"
  # 应显示多个进程共享同一 GPU context
  
  # 关闭
  echo quit | nvidia-cuda-mps-control
  ```
- PRISM 使用方式：在双进程启动前启用 MPS，使 computation process 和 I/O process 的 CUDA 操作可在 GPU 上并发执行
- 关键限制：
  - MPS 不支持 GPU 内存隔离（一个进程的 OOM 会影响所有 client）
  - 不支持 CUDA IPC（inter-process communication）的某些高级特性
  - 需要所有进程使用相同的 CUDA 版本
  - 某些 NVIDIA 消费级 GPU（如部分 GeForce）不支持 MPS
- PRISM 的 benefit：通过 MPS 使 I/O 进程的 cudaMemcpyAsync 与 computation 进程的 GEMM kernel 真正并发（而非串行化），使 overlapped layer streaming 的"重叠"在硬件层面成立

涉及论文标题：
- On-device Semantic Selection Made Low Latency and Memory Efficient with Monolithic Forwarding

---

## Arm Compute Library (ACL)

术语解释

Arm Compute Library (ACL) 是 Arm 官方开源的底层机器学习计算库，为 Arm Cortex-A CPU、Arm Neoverse 和 Arm Mali GPU 提供优化的算子实现。库提供 C++ API，支持卷积、GEMM、激活函数、归约、归一化等常见 ML 算子的高性能 kernel，是 Arm 平台上运行 ML 推理的核心基础设施。

术语是什么？

ACL 由三层组成：
1. **Core 层**：张量数据结构、硬件特定 kernel 实现（NEON/SVE/OpenCL）、数学工具函数。INT8 GEMM 通过 `arm_gemm` 子系统实现，使用手写 NEON 汇编 kernel（如 `a64_gemm_s8_8x12`），利用 Armv8.2-A+ 的 SDOT/UDOT 指令实现 S8×S8→S32 的 4 路并行点积。
2. **Runtime 层**：内存管理（`TensorAllocator`）、算子调度（`NEScheduler`）、张量生命周期管理、高层 Function 封装（如 `NEGEMMLowpMatrixMultiplyCore`）。
3. **Graph 层**（可选）：神经网络图表示、算子融合、优化 pass。

ACL 的 INT8 GEMM kernel 选择使用启发式方法：根据 CPU 特性（是否有 DotProd、SVE、MMLA）、矩阵维度（M/N/K）和 CPU 型号（如 A53 特殊处理）在运行时选择最优 kernel。例如：支持 DotProd 且 K>32 → `a64_gemm_s8_8x12`（8×12 tile），K≤32 → `a64_smallK_hybrid_s8s32_dot_8x4`，无 DotProd → `a64_gemm_s8_4x4` 回退。

从kernel调度角度拆解术语：

ACL 作为 kernel 执行引擎，一次 INT8 注意力的 kernel 调度流程（以 IntAttention 论文的 `bench_speed.cpp` 为例）：

```
用户调用 bench_speed --pipe 3 --L 1024 --d 128

ACL 内部 kernel 调度：
  1. 内存分配：
     TensorAllocator::allocate(Q_s8, [L, d])
     TensorAllocator::allocate(K_s8, [L, d])
     TensorAllocator::allocate(V_s8, [L, d])
     TensorAllocator::allocate(S_s32, [L, L])  // logits buffer
     TensorAllocator::allocate(P_u8, [L, L])   // probability buffer
     TensorAllocator::allocate(O_s32, [L, d])  // output buffer

  2. Kernel 1: NEGEMMLowpMatrixMultiplyCore::run(Q, K, S)
     ├── Method selection: check CPU features
     │   └── RK3588S2 (Cortex-A76): selects a64_gemm_s8_8x12 (DotProd)
     ├── Interleave Q: reshape for cache-friendly access
     ├── Transpose K: columns → contiguous memory
     ├── Tiled GEMM: process 8×12 tiles
     │   └── Inner loop: SDOT v0.4s, v1.4s, v2.4s, v3.4s  // 4 int8 dot-products
     └── Output: S_s32 buffer

  3. Kernel 2: NEIndexSoftmax::run(S, P)  // 论文新增的融合 kernel
     ├── NEArithmeticOps::max_reduce(S)   // 行内 max
     ├── NEArithmeticOps::subtract(S, max) // 逐元素减
     ├── NEComparison::greater_than(shifted, -c_int) // 生成 mask
     ├── LUT lookup via NEON TBL           // 并行查表
     └── Integer normalize via NEON umull+ushr  // 定点归一化
     Output: P_u8 buffer

  4. Kernel 3: NEGEMMLowpMatrixMultiplyCore::run(P, V, O)
     └── 同 Kernel 1，但 A=U8（reinterpret as S8），B=S8
     Output: O_s32 buffer

  5. (可选) NEQuantizationLayer::run(O_s32, O_s8)
     将 int32 输出 requantize 回 int8 供下游使用
```

关键 kernel 调度特性：
- **Tile 粒度**：INT8 GEMM 的 8×12 tile 在 A76 的 2×128-bit NEON 单元上实现最佳寄存器利用率
- **LUT 常驻寄存器**：IndexSoftmax 的 32×UINT8=32B LUT 完全放入 NEON 寄存器（32×128-bit = 512B 寄存器文件），查表零内存访问
- **融合 kernel**：IndexSoftmax 将 clip+LUT+normalize 三步融合，减少 2 次中间 buffer 的 L1/L2 cache 往返
- **多线程**：`GemmInterleaved` 按 M 和 N 维度分片，多线程并行执行独立 tile，通过 `NEScheduler` 管理线程池

术语一般如何实现？如何使用？

1. **构建与集成**：
   - 构建：`scons arch=arm64-v8.2-a neon=1 opencl=0`（CPU only）
   - 链接：`-larm_compute-static -larm_compute_core-static`
   - 头文件：`#include "arm_compute/runtime/NEON/functions/..."`

2. **使用方式**（以 IntAttention 为例）：
   - 打补丁：`git apply add_impl_for_ACL.patch` 在 ACL 中添加 `NEIndexSoftmax`
   - 自定义 kernel 注册到 ACL 的 Function 体系中
   - `bench_speed.cpp` 直接调用 ACL 的 `NEGEMMLowpMatrixMultiplyCore` + 自定义 `NEIndexSoftmax`

3. **与 CUDA 的对比**：
   | 维度 | ACL (NEON) | CUDA (cuBLAS) |
   |------|-----------|---------------|
   | 目标硬件 | ARM Cortex-A/Neoverse CPU | NVIDIA GPU (SM) |
   | SIMD 宽度 | 128-bit NEON (4×FP32, 16×INT8) | 32-thread warp |
   | INT8 GEMM | SDOT/UDOT 指令 | INT8 Tensor Core |
   | kernel 选择 | 运行时启发式 | 静态选择 |
   | 量化支持 | 原生 QASYMM8/QASYMM8_SIGNED | 需手动 scale |
   | LUT 支持 | TBL 指令（寄存器内查表） | 无等效指令 |

4. **限制**：
   - 仅支持 Arm 平台，不可移植到 x86
   - NEON 仅 128-bit 宽度（vs AVX-512 的 512-bit），单指令吞吐有限
   - 无原生 FP8/FP4 支持（需软件模拟）
   - 动态 kernel 选择有运行时开销（约 μs 级，对短序列推理可能显著）
   - 修改 ACL 需打补丁维护自己的 fork，升级 ACL 版本时需重新适配

涉及论文标题：
- IntAttention Fully Integer Attention Pipeline for Edge LLM Inference

## LUT-Based Dequantization via vlut16（基于vlut16查表的反量化）

术语是什么？
LUT-Based Dequantization via vlut16 是使用 Hexagon NPU 的 HVX 向量查表指令 `vlut16` 实现 INT4→FP16 高效反量化的技术。传统反量化需要 unmask（拆分高低 4-bit）→ unpack（扩展为 16-bit）→ convert（整数→浮点）三步指令序列，每一步消耗向量寄存器和指令槽。`vlut16` 指令将 16-entry LUT 存储在寄存器中，对每个输入字节（8-bit index）通过查表直接输出对应的 16-bit 值（FP16），单指令完成 INT4 index → FP16 weight 转换。同时通过 `vlut16` 将 4 组 FP16 scale 广播到整个向量寄存器（将 scale 作为 LUT 内容、预设常数索引），替代传统的 per-element scale 乘法。

从 kernel 调度角度拆解术语：
vlut16 反量化 kernel 的伪代码（Q4_0 对称量化，group_size=32，super-group=8 groups）：

```
// 输入: INT4 qweight[256] (128 bytes, 1 HVX register)
//       FP16 scales[8] (16 bytes, 在 HVX reg 中)
//       LUT dequant_table[16] = {-8f16, -7f16, ..., 7f16} 
// LUT 预计算（系统初始化时，一次）:
//   # 对 Q4_0: LUT[i] = (i & 0x7) - (i >> 3 ? (i & 0x7) - 8 : 0) 
//   # 转换为 qfloat 内部格式以避免后续转换开销

// Step 1: INT4 → FP16 值（vlut16 查表）
HVX_VReg qweight_v = vmem_load(qweight_addr);        // 128B INT4 packed
HVX_VRegPair result_pair = vlut16(qweight_v, lut_reg);// 每个 byte index → 16-bit value
// result_pair.even: 低 64 个 INT4 的 FP16 值
// result_pair.odd:  高 64 个 INT4 的 FP16 值

// Step 2: Scale 广播（vlut16 查表）
// 预设: scale_lut[16] = {s0, s0, s1, s1, s2, s2, s3, s3, s4, s4, s5, s5, s6, s6, s7, s7}
// 使用常量索引数组 [0,0,1,1,...,7,7] 作为 vlut16 输入
HVX_VRegPair scale_pair = vlut16(const_idx_reg, scale_lut_reg);

// Step 3: 反量化 = INT4_value × scale
HVX_VReg deq_even = vmpy(result_pair.even, scale_pair.even);
HVX_VReg deq_odd  = vmpy(result_pair.odd, scale_pair.odd);

// 输出: 256 个 FP16 weights (512 bytes, 4 HVX registers)
// 可连续写入 TCM（HMX tile layout）
```

与传统三步方案的指令数对比（256 个 INT4 值）：
| 操作 | 传统方案指令数 | vlut16 方案指令数 |
|------|-------------|-----------------|
| INT4 unpack (高低 4-bit 分离) | 4 (vand, vshr, etc.) | 0 |
| INT16→FP16 convert | 2 (vcvt) | 0 |
| Scale 广播 | 2-4 (vsplat, vshuff) | 1 (vlut16) |
| Scale 乘法 | 2 (vmpy) | 2 (vmpy) |
| qfloat→IEEE 转换 | 2 (vconvert) | 0 (LUT 预输出 qfloat) |
| **总指令数** | **12-14** | **5** |

术语一般如何实现？如何使用？
- 实现位置：htp-ops-lib 的 dequantization GEMM kernel 中。LUT table 在系统初始化时预计算并保持在 HVX 寄存器中。
- 适用量化格式：Q4_0（对称，每 32 元素共享 1 个 FP16 scale）、IQ4_NL（非对称查表，仅需换 LUT 内容）、Q8_0（每 32 元素共享 1 个 FP16 scale，int8→FP16 查表）。
- 关键细节：(1) LUT 内容预编码为 qfloat 格式，消除 HVX 的 qfloat→IEEE 转换开销；(2) vlut16 的 LUT 仅有 16 个条目（每条目 16-bit），对于 Q4_0（16 个唯一值）刚好够用；(3) Super-group coalesce 后 256 个 INT4 值精确填充 1 个 HVX 寄存器——单次 vlut16 处理整个 super-group。
- 论文效果：Dequantization GEMM 整体加速 9.65-19.04× vs baseline scatter 方案。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

---

## LUT-Based Softmax via vgather（基于vgather查表的Softmax）

术语是什么？
LUT-Based Softmax via vgather 是使用 Hexagon NPU HVX 的 `vgather` 向量指令实现 Softmax 中指数函数（exp）高效计算的技术。利用 Safe Softmax（所有 exp 输入 ≤0）的性质，预计算 32768-entry FP16 LUT（64 KiB，常驻 TCM），将 exp 计算替换为一次 vgather 查表。`vgather` 指令单次可从 TCM 中不连续地址 gather 64 个 2-byte 元素，将 FP16 input → FP16 exp output 的映射计算时间降至接近零。相比传统的多项式展开 exp（有序列依赖，VLIW 下 ILP 受限）或标量 `expf` 调用，LUT-based 方案大幅降低 Softmax 延迟。

从 kernel 调度角度拆解术语：
LUT-Based Softmax 在 FP16 FlashAttention 中的执行流程（Algorithm 1 中的 `LUT_Exp` 步骤）：

```
// 预计算（系统初始化时）:
//   lut_exp[0..32767] (FP16, 64 KiB, 驻留在 TCM)
//   for i in range(32768):
//       x = -float16(i)  # 仅存非正值 (Safe Softmax)
//       lut_exp[i] = exp2(x * log2(e))  # 以 exp2 近似 exp
//   # exp 的系数 log2(e) = 1.4427 被吸收到 QK^T scaling 中

// 运行时 LUT_Exp (FlashAttention inner loop):
// 输入: S ∈ FP16 [B_q, B_kv], 每元素 ≤ 0 (已减 rowmax)
// 输出: P ∈ FP16 [B_q, B_kv]

lut_exp(S_row):
  // 忽略 MSB (sign bit, 因为 Safe Softmax 保证 ≤0)
  // 左移 1 bit → byte offset = |value| * 2
  S_abs = vand(S_row, 0x7FFF)     // 清除 sign bit → |S|
  byte_offset = vshl(S_abs, 1)     // ×2 → byte offset (FP16 = 2 bytes)
  
  // vgather: 从 TCM 不连续地址并行 gather 64 个 FP16 值
  // 单指令耗时 24-48 个指令包 (V75)
  P_row = vgather(tcm_lut_base, byte_offset)  // 64 FP16 values/instruction
  return P_row

// 在 FlashAttention 内调用:
// m_new = max(m_old, rowmax(S))
// P = LUT_Exp(S - m_new)          // LUT 替代 polynomial exp
// l_new = exp(m_old - m_new) * l_old + rowsum(P)  // FP32 accum
// O = diag(exp(m_old - m_new)) * O + P @ V     // HMX GEMM
```

vgather 指令开销与优化：
- V75 上 vgather 延迟：24-48 指令包（高延迟但高吞吐：每指令 64 元素）
- 对比 FP16 polynomial exp：每元素需 ~10+ 指令（乘加序列 + 指数/尾数操作），且有序列依赖 → VLIW 下仅 1-2 条/包
- LUT 存储成本：32768 FP16 entries = 64 KiB，仅占 TCM 的 0.8%
- 地址限制：vgather 最大 byte offset 为 65536 → 32768 entries × 2B = 65536 bytes，刚好达到上限。利用 Safe Softmax (非正输入) 仅存 ≤0 的半边 LUT，解决容量限制

术语一般如何实现？如何使用？
- 实现位置：htp-ops-lib 的 `flash_attn.c` 中。LUT 在系统初始化时预计算并常驻 TCM（固定 64 KiB 区域，不参与动态分配）。
- 精度考量：(1) LUT 使用 FP32 精度计算中间结果后截断为 FP16（比 FP16 polynomial 精度更高）；(2) `exp2` 的系数 `log2(e)` 吸收到 `1/√d` scaling 中，无精度损失。
- 局限性：(1) vgather 在 V79 架构上有浮点错误（论文建议 V79 禁用 vgather exp，回退到 FP16 polynomial）；(2) LUT 固定占用 64 KiB TCM——对 TCM 压力大的 kernel 需权衡。
- 论文效果：Softmax 加速 1.26-2.19× vs FP32 exp，up to 1.60× vs FP16 polynomial exp。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

---

## Hardware-Aware Tile Quantization（硬件感知分块量化）

术语是什么？
Hardware-Aware Tile Quantization 是一种针对 NPU 矩阵单元（HMX）内存布局定制的细粒度分组量化方案。包含两个核心步骤：(1) **Tile-Group Quantization**：在量化前将权重矩阵预先 permute 为 HMX 期望的 tile 内存布局（外层级 column-major tile 排序 + 每 tile 内每两行交错 shuffle），然后按此布局中连续的 32 元素为一组做分组量化（而非传统列主序布局下的连续 32 元素）。量化后权重自然对齐 HMX tile 布局，反量化后可直接连续写入 TCM。(2) **Group Coalesce for HVX**：将 8 个量化组合并为 super-group（256 个 INT4 元素 = 128 bytes），重新排列 INT4 值和 scale 值，使 256 个 INT4 值精确填充 1 个 HVX 128-byte 向量寄存器，最大化 HVX 向量利用率。

从 kernel 调度角度拆解术语：
Offline 权重转换 pipeline（一次量化 + 重排）：

```
// 输入: FP16 Weight Matrix W [K, N] (K=hidden_dim, N=proj_dim)
//       采用 column-major 存储 (llama.cpp CPU backend 默认)
// 输出: INT4 Quantized Weight W_q (已排列为 HMX tile layout + super-group coalesce)

Step 1: Pre-quantization Permute (目标: HMX tile layout)
  // HMX tile layout: 外层级 column-major tile 排列 + 内层每两行 shuffle
  W_perm = zeros(K, N)
  for n_tile in range(0, N, 32):                # 外层级: col-major tiles
    for k_tile in range(0, K, 32):
      tile_2d = W[k_tile:k_tile+32, n_tile:n_tile+32]  # [32, 32]
      for r in range(0, 32, 2):                 # 内层: 每两行交错
        W_perm[..., n_tile*K + k_tile*32 + ...] = 
          interleave(tile_2d[r], tile_2d[r+1])  # 2×32 → transposed-like

Step 2: Tile-Group Quantization
  // 在 permuted layout 中，每 32 个连续元素为一个 group
  group_size = 32
  for g_idx in range(0, K*N/group_size):        # g_idx 按 permuted 内存序
    group = W_perm[g_idx*32 : (g_idx+1)*32]
    scale = max(abs(group)) / 7.0               # Q4_0 symmetric
    qgroup[i] = round(group[i] / scale)          # INT4: [-8, 7]
    // 存储: 16 bytes INT4 + 2 bytes FP16 scale (AoS layout)

Step 3: Post-quantization Group Coalesce (目标: HVX 128-byte 寄存器对齐)
  // 合并 8 个相邻 group (256 INT4 values = 128 bytes)
  for super_idx in range(0, num_groups/8):
    // Step 3a: 提取 8 groups 的 INT4 值
    for g in range(8):
      int4_block[g*32 : (g+1)*32] = qgroup[super_idx*8 + g].values
    // int4_block 现在为 256 个连续的 INT4 值 = 128 bytes = 1 HVX 寄存器
    
    // Step 3b: 提取 8 groups 的 FP16 scales
    for g in range(8):
      scale_block[g] = qgroup[super_idx*8 + g].scale  // 8 个 scales
    
    // Step 3c: 打包为 super-group 格式
    super_group.values = int4_block   // 128 bytes (1 HVX reg)
    super_group.scales = scale_block  // 16 bytes (vlut16 broadcast ready)
```

内存对齐效果对比：
| 方案 | INT4 值布局 | Scale 布局 | HVX reg 利用率 |
|------|-----------|-----------|---------------|
| Naive AoS (group_size=32) | 16B INT4 + 2B scale (interleaved) | 每组独立 | 16/128 = 12.5% |
| Super-group coalesce | 128B INT4 (sequential) | 16B scales (block) | 128/128 = 100% |

术语一般如何实现？如何使用？
- 工具链：`extras/convert_hf_to_gguf_htp.py`（Python 脚本）执行 Step 1 的 HMX layout permutation。`llama-quantize` 配合 `REPACK_FOR_HVX=1` 环境变量执行 Step 2-3 的量化和 super-group coalesce。
- 关键约束：(1) 量化必须在 permuted layout 上执行（而非原始 column-major layout）——否则 scatter 问题未被解决；(2) Tile-group 与 Common-group 的精度差异极小（Table 4: WinoGrande 62.6 vs 63.3, MMLU 35.5 vs 35.3, Wiki PPL 10.2 vs 10.2），因为预训练权重大致服从零均值高斯分布，tile 内统计量与列连续组相似；(3) 权重排列仅需离线执行一次（转换脚本），推理时无额外开销。
- 论文效果：HMX layout 单独贡献 vs baseline 的加速（消除 scatter）；Super-group coalesce 额外贡献 1.82-3.45× 加速（消除 HVX 寄存器低利用率）。整体 dequantization GEMM 加速 9.65-19.04×。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

---

## dequantize-softmax-requantize Bottleneck（去量化-softmax-再量化瓶颈）

术语解释

dequantize-softmax-requantize 瓶颈是指在 INT8 量化 Transformer 推理中，对矩阵乘法（QK^T 和 PV）做完 INT8 GEMM 加速后，中间的 softmax 路径因需要从 INT32 转回 FP32 计算 softmax（exp + sum + div），再转回 INT8 供下游 GEMM 使用，产生额外类型转换和浮点运算开销，成为注意力延迟主要瓶颈的现象。

术语是什么？

在标准 INT8 量化注意力 pipeline 中（"Quant-Only" 方案），数据流为：
```
S8×S8 → S32 (QK^T INT8 GEMM)
  → dequantize: S32 → FP32
  → softmax: FP32 exp + row_sum + div
  → requantize: FP32 → S8
  → S8×S8 → S32 (PV INT8 GEMM)
```

当 INT8 GEMM 将 QK^T 和 PV 矩阵乘加速 2× 以上后，中间的三步浮点路径的相对开销急剧上升。IntAttention 论文在 RK3588S2 ARM CPU 上实测，这条路径占注意力延迟 ≤65%。具体分解：
- dequantize（S32→FP32）：逐元素类型转换，NEON 可向量化，但引入 FP32 buffer 的写操作（~O(L²) 内存写）
- softmax（FP32 exp + sum + div）：exp 为标量库函数 `expf()`，每个 logit ~12-20 cycles；sum 和 div 可向量化
- requantize（FP32→S8）：逐元素类型转换 + 饱和处理，NEON 可向量化，但引入 FP32 buffer 的读操作

这三次操作之间产生 3 次 L1/L2 cache 往返：dequant 写 FP32 buf → softmax 读写 FP32 buf → requant 读 FP32 buf 写 S8 buf。在 memory-bound 的边缘设备上，这些额外的内存往返显著放大延迟。

从kernel调度角度拆解术语：

```
以 RK3588S2 (ARM Cortex-A76) 上 INT8 Quant-Only 方案的 attention 为例：

Time  Kernel                                     Memory Ops
──────────────────────────────────────────────────────────────
 t0   NEGEMMLowp Q×K^T (S8×S8→S32)              R: Q_s8, K_s8  W: S_s32
      [~35% latency]

 t1   NEDequantizationLayer S32→FP32              R: S_s32       W: S_fp32
      [~10% latency]
      └── NEON vcvtq_f32_s32: 4 lanes/cycle, 每次写 FP32 buffer

 t2   NEExpLayer (softmax)                        R: S_fp32      W: P_fp32
      [~40% latency — 主要瓶颈]
      ├── max reduction (NEON vmaxvq)
      ├── expf() 标量库调用 (12-20 cycles × 1024 elements)
      ├── row_sum (NEON vaddq)
      └── division (NEON vdivq, ~10 cycles/div)

 t3   NEQuantizationLayer FP32→S8                 R: P_fp32      W: P_s8
      [~15% latency]
      └── NEON vcvtnq_s32_f32 + vqmovn + saturation

 t4   NEGEMMLowp P×V (S8×S8→S32)                 R: P_s8, V_s8  W: O_s32
      [~35% latency]

 总延迟 = t1 + t2 + t3 + t4
 其中 softmax 路径 (t1+t2+t3) ≤ 65% total

 关键瓶颈分析：
 - 3 次 cache round-trip between t1→t2, t2→t3, t3→t4
 - expf() 标量无 NEON 向量化（各 lane 数据依赖不同）
 - 所有中间 buffer (S_fp32, P_fp32) 为 FP32，占用 4× 带宽 vs UINT8
```

术语一般如何实现？如何使用？

要消除此瓶颈，有以下方案：

1. **全整数 softmax 替换**（IntAttention 的方案）：将 t1+t2+t3 三步融合为一个整数 kernel（IndexSoftmax），实现 S32→U8 直通。消除所有类型转换和中间 FP32 buffer。延迟占比从 ≤65% 降至可忽略。

2. **LUT-based softmax**（EXAQ、Bi-LUT 等方案）：在 FP16/INT16 域用 LUT 替代 exp 计算，虽仍需 dequant/requant，但避免了昂贵的标量 expf() 调用。延迟降低 ~30-50%。

3. **Clipped-linear softmax (HCCS)**：完全替换指数为线性函数，无 LUT 无 exp，精度损失略大但速度最快。

4. **Log-domain softmax**（REXP 等）：用对数域运算替代直接 exp，配合 LOD（Leading-One Detection）简化计算。

5. **瓶颈识别方法**：在目标平台上分步 benchmark attention 的每个子 kernel：
   - `bench_speed.cpp --pipe 2`（INT8 Quant-Only）→ 测总延迟
   - 单独测 dequant + softmax + requant 路径的延迟
   - 计算占比 = softmax_path_latency / total_attention_latency
   - 若占比 >30%，则存在显著瓶颈，值得优化

涉及论文标题：
- IntAttention Fully Integer Attention Pipeline for Edge LLM Inference

## Co-driver Design / NPU Control-Data Plane Separation（NPU 控制-数据面分离的协同驱动设计）

术语是什么？

Co-driver Design（协同驱动设计）是 TZ-LLM 提出的一种 NPU 驱动架构，将传统的单片式 NPU Linux 驱动分离为两个独立的安全域：(1) **REE 控制平面（Control Plane）**：保留在 Linux 内核中的全功能驱动（~60K LoC），负责 NPU job 调度排队、电源管理（DVFS）、频率控制、Linux 设备框架集成；(2) **TEE 数据平面（Data Plane）**：部署在 TEE 用户态的精简驱动（~1K LoC），仅包含 NPU job 执行的最小闭包——初始化执行上下文（I/O page table、register commands、I/O buffers）、MMIO 启动作业、处理 completion 中断。两个平面通过 smc 指令协作：控制平面决定"何时执行哪个 job"，数据平面负责"如何安全地执行安全 job"。

从 kernel 调度角度拆解术语：

Co-driver 的 job 执行流程伪代码：

```
// === REE 控制平面（Linux NPU 驱动）===
func ree_npu_scheduler():
    while true:
        job = unified_queue.dequeue()  // 混合安全 shadow jobs 和非安全 jobs
        if job.is_secure_shadow:
            smc_call(SMC_NPU_SCHEDULE, job.id)  // 通知 TEE 接管 NPU
            wait_for_smc_response()              // 阻塞等待 TEE 完成
            discard_shadow_job(job)
        else:
            launch_npu_job(job)                  // 正常启动非安全 job
            wait_for_completion_interrupt()
        schedule_next()

// === TEE 数据平面（TEE NPU 驱动, ~1K LoC user-mode）===
func tee_npu_handle_smc(job_id):
    job = lookup_initialized_job(job_id)
    if job == null or job.already_issued: return ERROR  // 防重放
    if job.seq_num != current_exec_seq: return ERROR    // 防重排序
    // NPU 安全模式切换（严格顺序）
    tzpc_config(NPU_MMIO, SECURE)     // Step 1: 禁止 REE 访问 NPU MMIO
    while npu_is_busy(): spin()       // Step 2: 等待非安全 job 完成
    tzasc_grant_dma(NPU, job_regions) // Step 3: 授权 NPU DMA 安全内存
    gic_route_irq(NPU_IRQ, TEE)       // Step 4: NPU 中断路由到 TEE
    mmio_launch(job)                  // 启动安全 NPU job
    wait_for_secure_interrupt()       // 等待 completion
    tzasc_revoke_dma(NPU)             // 归还 NPU
    gic_route_irq(NPU_IRQ, REE)
    tzpc_config(NPU_MMIO, NON_SECURE)
    current_exec_seq += 1
    smc_return(SUCCESS)
```

术语一般如何实现？如何使用？

- **TCB 最小化**：TEE NPU 驱动以 user-mode TA 运行，TEE OS 仅映射 NPU MMIO 区域到其地址空间 + TZASC 限制 NPU 仅可 DMA 访问指定区域。即使 NPU 驱动被攻破，攻击者无法访问其他安全内存。
- **Qualcomm 可行性**：作者调查了 Qualcomm NPU 的开源 Linux 驱动，确认可提取类似的数据平面闭包。
- **切换开销**：TZPC+TZASC+GIC 配置 + smc 通信总计占 TTFT 的 1.6%∼2.7%（prefill）和 decoding 的 2.3%∼5.7%。

涉及论文标题：
- TZ-LLM

## Shadow Job Scheduling（影子作业调度）

术语是什么？

Shadow Job Scheduling 是 TZ-LLM co-driver 设计中实现 REE-TEE NPU 统一调度的机制。LLM TA 需要执行安全 NPU job 时，TEE 驱动向 REE 驱动的调度队列提交一个**影子作业（Shadow Job）**——包含元数据（job ID、优先级）但不含真实执行上下文的占位符 job。Shadow job 的作用是"占位排队"——让 REE 调度器知道有安全 job 等待 NPU 资源，但不暴露 job 内容。当 shadow job 被调度到时，REE 驱动通过 smc 将 NPU 控制权移交 TEE 驱动。TEE 驱动在安全模式下执行真实 job，完成后通知 REE，REE 丢弃 shadow job 并调度下一个。

从 kernel 调度角度拆解术语：

Shadow Job 调度的时间线（以 YOLOv5 与 LLM job 混合调度为例）：

```
REE App    REE NPU Driver    TEE NPU Driver    NPU Hardware
  │  YOLO job    │                  │               │
  │─────────────>│─launch──────────>│──────────────>│ exec YOLO
  │              │                  │  shadow job   │
  │              │<──smc────────────│               │
  │              │ enqueue shadow   │               │
  │<──YOLO done──│<─completion──────│<──────────────│
  │              │ dequeue shadow   │               │
  │              │──smc(SCHEDULE)──>│               │
  │              │                  │ TZPC+TZASC cfg│ hw switch
  │              │                  │─launch secure>│ exec secure
  │              │                  │<──secure IRQ──│ done
  │              │                  │ TZASC restore │ hw restore
  │              │<──smc_return─────│               │
  │              │ discard shadow   │               │
```

术语一般如何实现？如何使用？

- **防重放/重排序**：每个安全 job 分配单调递增序列号，启动前验证序列号一致性。
- **统一调度公平性**：TEE-REE NPU 分时下 REE NN 应用（YOLOv5/MobileNet）吞吐额外损失 ≤3.8%。
- **与 StrongBox 的对比**：StrongBox [33] 使用 EL3 monitor 保护 GPU jobs，扩展了高特权 TCB。TZ-LLM 通过 user-mode TEE 驱动 + shadow job 避免扩展 EL3 TCB。

涉及论文标题：
- TZ-LLM

## Priority-based Preemptive Pipeline Scheduling（基于优先级的抢占式流水线调度）

术语是什么？

Priority-based Preemptive Pipeline Scheduling 是 TZ-LLM 为流水线参数恢复设计的 CPU operator 调度算法。流水线中多类 operators（CPU computation、CMA allocation、AES decryption）竞争 CPU 资源。调度器的 greedy 策略：优先执行 ready 的 CPU computation operator；若无，则执行与最早 computation operator 关联的 restoration operator。Allocation 和 decryption operators 被划分为 micro-operators，每完成一个 micro-operator 检查是否有 computation operator 变为 ready——若有则立即抢占切换到 computation。

从 kernel 调度角度拆解术语：

调度伪代码：

```
struct Operator {
    type: {CPU_COMPUTE, ALLOC, DECRYPT}
    topo_index: int  // 计算图拓扑序
    parent_index: int  // restoration operator 所属的 computation operator index
}

func scheduling_priority(op):
    if op.type == CPU_COMPUTE:
        return (0, op.topo_index)       // priority_class=0 (highest)
    else:
        return (1, op.parent_index)     // priority_class=1

func scheduler():
    while has_ready():
        op = priority_queue.pop()
        if op.type in {ALLOC, DECRYPT}:
            for micro in decompose(op):
                execute(micro)
                if any_CPU_COMPUTE_ready():  // preemption checkpoint
                    preempt()
                    break
        else:
            execute(op)  // CPU compute: run to completion
```

术语一般如何实现？如何使用？

- **调度效果**：与理论下界差距 0.01%∼9.9%（有内存压力）、10.4%（无内存压力，I/O 唯一瓶颈路径时的最坏情况）。
- **I/O 和 NPU 调度**：各自独立、按拓扑序最优——它们使用不同硬件引擎，不存在资源竞争。
- **限制**：greedy 策略未显式建模三种可能关键路径的切换——当 loading/I/O 是瓶颈时（无内存压力场景），应优先 allocation operators 以减少 loading 气泡，但 greedy 策略仍优先 CPU computation，导致 suboptimal。

涉及论文标题：
- TZ-LLM
