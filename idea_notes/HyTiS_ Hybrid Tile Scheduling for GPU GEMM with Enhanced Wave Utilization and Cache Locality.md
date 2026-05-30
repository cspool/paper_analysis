## HyTiS: Hybrid Tile Scheduling for GPU GEMM with Enhanced Wave Utilization and Cache Locality

- baseline方法是什么？
  Baseline 是 cuBLAS（NVIDIA 闭源 GEMM 库，使用 homogeneous tile scheduling——所有 output tiles 采用统一的 tile size，通过 precompiled kernel specializations 执行）。以 H100 GPU 为例，给定 GEMM 问题 M×N×K=1672×1024×4096：

  **全栈执行例子**：
  - **模型推理算法层**：以 GPT/LLM 中 GEMM 为例，输入为 activation tensor A (M×K=1672×4096) 和 weight matrix B (K×N=4096×1024)，需计算 C=A×B。cuBLAS 选择预编译的 fixed-size tile kernel（如 bM×bN×bK=128×128×64）。
  - **系统框架层**：PyTorch 调用 torch.matmul → dispatch 到 cuBLAS cublasGemmEx。cuBLAS 的 tile 调度是 homogeneous 的——所有 output tiles 使用相同 micro-kernel。
  - **编译框架层**：cuBLAS 使用预编译 kernel，无运行时编译。Inductor-Triton baseline 使用 Triton compiler 生成约 20 种候选 tile 配置的 kernel，但 tile layout 固定为 group-M with group_size=8。
  - **kernel调度层**：H100 有 132 个 SM。1672×1024 GEMM 在 bM×bN=128×128 下产生 ceil(1672/128)×ceil(1024/128) = 14×8 = 112 个 output tiles。112 tiles 在 132 SM 上形成 0 个 full wave（112 < 132），全部成为 1 个 partial wave，SM 利用率仅 112/132 ≈ 85%，剩余 20 个 SM 闲置——这是 wave quantization 问题。cuBLAS 的 homogeneous tile 无法应对此情况，性能随 M 变化剧烈波动（在 M=640→704 和 M=1664→1728 处出现 36% 和 21% 的性能陡降，Figure 1）。
  - **硬件架构层**：H100 SM 内 Tensor Core 执行 wgmma 指令进行 MMA 计算，SM 通过 L1/SMEM 做 local data staging，所有 132 SM 共享 50MB L2 cache。Partial wave 中部分 SM 空闲导致 Tensor Core 利用率不足。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  HyTiS 提出两层级联的混合 tile 调度 + 自适应 tile layout 选择，在 Triton 上实现。以同一 1672×1024×4096 GEMM 为例：

  **全栈执行例子（对比 baseline）**：
  - **模型推理算法层**：同一 GEMM 问题，HyTiS 不改变算法——仍是 dense FP16 matmul with FP32 accumulation。关键区别在于 scheduling 层。
  - **系统框架层**：PyTorch 调用 hytis.matmul(a, b) → HyTiScheduler 接收 (M,N,K)=(1672,1024,4096) → 检查 tuning cache → 执行 auto-tuning 在 TO×LO 搜索空间中选择最优 (K1, K2, layout)。
  - **编译框架层**：HyTiS 使用 Triton 作为后端 compiler，不改动 Triton 编译流程。Triton 负责 intra-tile 优化（memory coalescing、thread swizzling、shared memory allocation、TMA instruction emission on H100）。
  - **kernel调度层（核心创新）**：
    1. Offline profiling：目标 GPU 上预先对所有候选 micro-kernel 执行 profiling（H100 约 19 min），构建 S^TO（吞吐量导向，large tiles，在 full waves 中最大化 compute/memory ratio）和 S^LO（延迟导向，fine-grained tiles，在 partial wave 中最小化 per-wave latency）两个候选集。按资源约束（SMEM≤SMEM_0, REG_spill==0）、指令约束（H100 wgmma 要求 bM%64==0）、SMEM 利用约束（one-tile-per-SM，确保 tile 足够大以有效利用 shared memory）过滤，再按 threshold l1/l2 保留性能接近最优的候选。
    2. 两层级联调度：第一级用 TO micro-kernel K1（如 128×128×64）处理 full waves。1672×1024 问题没有 full wave（112 tiles < 132 SM），退化为 LO-only scheduling。此时第二级用 LO micro-kernel K2（如 64×64×32）处理所有 tiles。更典型的例子是 M=1800, N=1024 → ~144 tiles：第一级 TO kernel 产生 1 个 full wave（132 tiles），第二级 LO kernel 处理剩余 12 tiles，fine-grained tiling 使 partial wave 执行时间显著缩短。
    3. 对比 Split-K/Stream-K：Split-K 和 Stream-K 沿 K 维度分割 partial wave 的 workload 到更多 SM，但引入 reduction sync 和额外 workspace（Stream-K 比 cuBLAS 多消耗 70%+ device memory）。HyTiS 无需沿 K 拆分——通过两级不同 tile size scheduling 直接避免 partial wave 的 SM 浪费，零同步开销。
    4. 自适应 tile layout：在 tile size 确定后，通过分析模型计算第一 wave 的 DRAM→L2 流量 V_1 选择最优 group size s_opt（s_opt^GM = min(ceil(sqrt(N_SM·bN/bM)), ceil(M/bM))）；计算所有 wave 的总流量 V_tol = ΣV_i，在 GM 和 GN 布局中选择 V_tol 更小的。实测 DRAM read 量减少（H100 上 low region 从 HyTiS(STL) 的 46% 降至 20%，high region 从 15% 升至 28%）。
  - **硬件架构层**：SM 内 Tensor Core（H100 wgmma）执行 micro-kernel 的 MMA 计算。TMA 指令做 asynchronous global→shared memory 数据搬运（H100 only）；A100 上使用传统 ldmatrix 指令 + data-parallel launch。L2 cache 数据复用受益于 layout 调度：选取最优 (GM/GN, s) 后，同一 wave 内相邻 SM 对矩阵 A/B 的访问在 L2 中命中率更高。

  关键设计选择映射到 baseline 缺陷：
  - wave quantization → SM 利用率低：两级别联调度，full waves 用大 tile 保吞吐，partial wave 用小 tile 降延迟。
  - fixed tile layout → 次优 L2 cache affinity：分析模型自适应选择 GM/GN 布局和 group size，最小化 wave 粒度 DRAM→L2 流量。
  - Split-K/Stream-K 的同步开销：HyTiS 不沿 K 维度拆分，免 reduction sync。
  - Inductor-Triton fixed search space：offline profiling 构建 architecture-aware 候选集 + runtime adaptive search space（l1/l2 阈值动态调整）。
