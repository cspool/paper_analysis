## MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MxMoE 自动生成混合精度 Group-GEMM GPU kernel，包含三个核心组件：
    1. **Micro-Kernel Specialization**：为每种量化精度（W2A16, W4A16, W4A4-g128, W8A8 等）实现可配置的 CTA 级 CUDA device function，利用 CTA index independence 实现水平融合。例如 W2A16 micro-kernel 使用 fused dequantization + bit manipulation 优化 int-to-float 转换；W4A4-g128 使用 multistage software pipelining 严格遵循 128 量化 group 约束。Memory access 模式针对每种量化方案手工调优 compute-to-memory access pipeline。
    2. **Resource Configuration**：为水平融合的混合精度 Group-GEMM kernel 配置计算资源。强制所有 micro-kernel tiles 使用相同 warp count（满足 CUDA 编程模型的 uniform resource 要求），shared memory 按融合操作中最大需求分配。为减少因 tile size 差异导致的 shared memory 浪费，引入 k-dimension tiling（slice-K）对较小 tile 增加 k 维并行度，同时减少 warp under-utilization。
    3. **Tile Scheduling**：因不同精度和 tile shape 组合的执行时间差异显著，tile 调度顺序直接影响总完成时间。MxMoE 使用 greedy 启发式优先调度计算密集 tile，在 MoE block tiles 数远大于 SM 数时实现近最优性能（符合 Graham 1966 的 bound）。
  - 实验比较：因缺乏已建立的 low-precision Group-GEMM baseline，比较 MxMoE 生成的 uniform-bitwidth 和 mixed-precision kernel vs CUTLASS 16-bit Group-GEMM。评估 memory-bound（512 tokens）和 compute-bound（8192 tokens）两种 workload 下的 MoE block 计算吞吐量。

- 后端平台是什么，配置是什么。
  - NVIDIA RTX 4090 GPU（Ada Lovelace 架构）
  - CUDA/CUTLASS 框架

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估方式：从 WikiText-2 随机采样 512/8192 token 序列，测量 MoE block 计算吞吐量（仅计 expert GEMM 计算，gate/topk/sort 等开销可忽略）。
  - 修改内容（混合精度 Group-GEMM kernel 设计）：
    1. Micro-kernel：每种精度独立实现为 CUDA device function，模板参数指定资源配置
    2. Kernel generator：根据 ILP 分配的方案，自动组合多个 micro-kernel 为统一 kernel，生成 precision-aware routing logic
    3. Tile scheduler：greedy LPT（Longest Processing Time first）启发式
    4. K-dimension tiling (slice-K)：W4A16 tile 比 W8A8 tile 显著更小，slice-K 将 W4A16 的 k 维切分为多个 sub-tile，增加 SM 利用率
  - 对比 kernel：
    - HQQ kernel（不融合 dequantization，性能差）
    - VLLM-Marlin-MoE kernel（顺序调用 Marlin W4A16 kernel，suboptimal GPU utilization）
    - CUTLASS 16-bit Group-GEMM（full-precision baseline）

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源地址：https://github.com/cat538/MxMoE
  - Kernel 执行全过程（以 DeepSeek-V2-Lite MoE block W4.25A15.5 在 RTX 4090 上为例）：

  **阶段 1: Kernel 编译时生成**
  - 输入：混合精度方案 {x_{i,j,k}}（如某些 expert 的 down_proj 用 W4A16，gate_proj 用 W8A8）
  - Step 1a: 对方案中出现的每种精度，选择对应的 micro-kernel（如 W4A16 → Marlin-style fused dequant kernel, W8A8 → standard INT8 GEMM kernel）
  - Step 1b: Resource Configuration — 所有 micro-kernel 统一 warp count（如 4 warps/CTA），shared memory 取所有方案中的最大值
  - Step 1c: K-dimension tiling — 对 tile size 较小的方案增加 k-dim split（如 W4A16 的 k=256 分为 2×128）
  - 输出：编译后的 fused mixed-precision Group-GEMM kernel

  **阶段 2: 运行时执行**
  - 输入：MoE block 输入 X ∈ R^{T×d}（T tokens），各 expert 的 INT4/INT8 packed 权重 + scale + zero-point
  - Step 2a: Gating → 每个 token 分配到 top-k expert → 按 expert 分组 token，得到 per-expert X_e
  - Step 2b: Tile Scheduler 构建 tile list：
    对每个启用的 expert e，对每个 linear block j（gate/up/down），根据其分配精度 k 和 tile config t，将 GEMM (X_e, W_{e,j}) 分解为 tiles {(c, n_t)}，所有 tiles 汇总为全局调度队列
  - Step 2c: Greedy LPT 调度 — 按 tile 执行时间 c 降序排列，依次分配到有空闲 SM 的 tile slot
  - Step 2d: SM 执行 micro-kernel：
    - W4A16 tile：从 global memory 加载 INT4 packed W + FP16 scale → shared memory → fused dequantization（INT4→FP16）→ Tensor Core MMA → FP16 accumulator
    - W8A8 tile：加载 INT8 W + INT8 activation → Tensor Core IMMA → INT32 accumulator → dequant to FP16
  - Step 2e: 所有 tiles 完成后，reduction 得到 MoE block output

  **阶段 3: 性能输出**
  - Memory-bound（512 tokens）：W4.25A15.5 比 FP16 快 1.6-2.7×，比 uniform W4A16 快 up to 25%（Qwen1.5-MoE）
  - Compute-bound（8192 tokens）：W5A5 比 FP16 快 3-3.4×，比 uniform W8A8 快 up to 29.4%
  - 混合精度优势来源：hardware-aware bitwidth allocation 将低精度 activation 分配给高频激活 expert（compute-bound），保持高频 expert 高精度
