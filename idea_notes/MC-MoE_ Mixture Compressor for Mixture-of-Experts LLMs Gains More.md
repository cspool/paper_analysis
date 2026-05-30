## MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

- baseline方法是什么？
  - Baseline 为 GPTQ uniform bit-width quantization（"Uni"），即对所有 expert 使用相同位宽（如 2-bit）做 PTQ 量化，各 expert 重要性差异被忽略。另一对比 baseline 为 BSP（Block Score Predictor, Li et al. 2024），它基于 block 级别做混合精度分配（25% MoE layer 为 4-bit，其余 2-bit），但仍是 layer 粒度而非 expert 粒度。动态剪枝方面，baseline 为 weight-only pruning（Lu et al. 2024），仅基于 routing weight ratio w₁/w₀ 剪枝低分 expert，不保护重要 token。
  - 以 GPTQ 2-bit uniform quantization + weight-only pruning 为 baseline，全栈执行路径为：
    - **算法层**：MoE-LLM 推理时，对于每个 token t 在每个 MoE layer：Router G(t) 计算 softmax(W_g · t) 生成 N 个 expert 的 routing scores → 取 Top-2 expert {E₀, E₁} 及对应权重 {w₀, w₁} → 若 w₁/w₀ < μ 则仅计算 y = w₀·E₀(t)，否则 y = w₀·E₀(t) + w₁·E₁(t) → 所有 expert 权重 W 均被 2-bit uniform GPTQ 量化。expert 存储：每层 8 个 expert 均占用相同 ~2-bit/param。注意力模块保持 16-bit。
    - **系统框架层**：标准 PyTorch + HuggingFace Transformers 推理，无特殊 Serving 框架修改。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：使用 HQQ 保存量化权重和反量化，CUDA kernel 基于 HQQ 实现，无 expert-specific 优化。
    - **硬件架构层**：NVIDIA A100-80GB GPU，Mixtral 8×7b 需 2 卡（FP16 约 96.8 GB），量化后 ~13.6 GB 仍需 at least 1 卡 A100。
  - Baseline 核心缺陷：
    1. **忽略 expert 异质性**：不同 expert 的激活频率、routing weight、量化敏感度差异巨大，uniform 量化导致重要 expert 欠保护而冗余 expert 过保护，2-bit uniform 量化在 8 个 benchmark 上平均准确率下降 28.6%（71.29% → 42.67%）。
    2. **BSP 的 layer 粒度粗放**：仅区分 layer 不区分 expert，某些 layer 内仍有高重要性 expert 被 2-bit 量化损坏，某些低重要性 expert 浪费 4-bit 预算。
    3. **Weight-only pruning 引发 attention decay**：仅依赖 routing weight 剪枝时，某些关键 token 对应的 expert 被错误剪枝，导致后续层 attention map 畸变，PPL 从 ~5.9 升至 ~6.5（约 10% 相对退化），且 15% 剪枝率下 LM-Eval 降约 10%。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - MC 通过"expert 显著性驱动的静态混合精度量化（PMQ）+ token 重要性感知的动态剪枝（ODP）"双阶段设计解决上述缺陷：
  - 全栈执行路径（以 Mixtral 8×7b, k=2.54-bit 为例）：
    - **算法层 — PMQ 阶段（pre-loading）**：
      1. 在 C4 校准数据上对原始 16-bit Mixtral 8×7b 做一次前向推理，为每个 MoE layer 的每个 expert i 计算三维重要性向量：(a) 访问频率 ϕᵢ = nᵢ/N；(b) 激活权重和 wᵢ = Σσᵢʲ/N；(c) 各候选位宽 j∈{1,2,3} 下的量化重构 F-norm ϵᵢⱼ = ‖F(θ) − F(θ[eᵢ→Q(eᵢ,j)])‖_F。
      2. 构建 Integer Programming 模型：目标函数 MIN ΣᵢΣⱼ ϕᵢᵅ·wᵢᵝ·(ϵᵢⱼ·xᵢⱼ)ᵞ，约束为平均位宽 = k、每个 expert 唯一分配、至少 1 个 3-bit 和 1 个 2-bit expert。求解得到的 xᵢⱼ 给出每个 expert 的最优位宽 Bᵢ ∈ {1,2,3}。
      3. 按 Bᵢ 用 GPTQ 量化每个 expert：对 2/3-bit 使用线性量化 + Hessian 误差补偿；对 1-bit 使用 B̃ = (sign(W)+1)/2 映射到 {0,1} + scaling factor s = ‖W‖_ℓ₁/(d×m)。Attention/gating 等非 expert 参数统一 4-bit。量化耗时 ~90 分钟（Mixtral 8×7b）。
    - **算法层 — ODP 阶段（online inference）**：
      1. 对每个输入 token，在当前 MoE layer 之前，基于上一层 attention map A = softmax(KᵀQ/√dₖ) 计算 token importance：Iⱼ = ‖tⱼ‖₁ · (Σ_{i≥j} Aⱼᵢ)/(L−j)，结合 token 特征范数和被关注度。
      2. 对 top-2% 高重要性 token 启用完整保护：保留 Router 分配的 Top-2 expert 计算，不做剪枝。
      3. 对非保护 token，仍执行 routing-score-based pruning：当 w₁/w₀ < μ（μ 为 calibration 中位数）时仅保留 primary expert，跳过 secondary expert。
      4. 1-bit expert 反量化使用位运算加速：s · xB = s(Σ_{B̃ᵢⱼ=1} xⱼ − Σ_{B̃ᵢⱼ=0} xⱼ)，MACs 从 dm 降至 m。
    - **系统框架层**：使用 HQQ 工具保存混合精度量化权重并执行反量化，设计了 1-bit 权重紧凑存储格式（bit-change transformation B̃ ∈ {0,1}）。CUDA kernel 基于 HQQ 适配。未修改特定 Serving 框架。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：HQQ 提供的 CUDA kernel 处理不同位宽权重的反量化和矩阵乘法。1-bit 权重使用加法树替代乘累加。未引入新的 kernel 调度策略。
    - **硬件架构层**：单张 NVIDIA A100-80GB（或 RTX 3090）即可运行压缩后 Mixtral 8×7b，无需多卡。FP16 baseline 需 2×A100。
  - 对比 baseline 的改进映射：
    - **Uniform quantization 忽略 expert 异质性 → PMQ 三层 expert 重要性驱动位宽分配**：ϕᵢ（频率）× wᵢ（路由权重）× ϵᵢⱼ（量化误差）三维建模每个 expert 的真实重要性。Integer Programming 在平均位宽约束下自动将高位宽分配给关键 expert（如某些 layer 中高频高权 expert 获 3-bit），低位宽分配给冗余 expert（如几乎不被激活的 expert 获 1-bit）。结果：2.54-bit PMQ 在 8 benchmark 上 Avg 67.50%（仅降 3.8%），远超 BSP 的 49.07%（降 22.2%）和 Uni 2-bit 的 42.67%（降 28.6%）。
    - **BSP layer 粒度粗放 → PMQ expert 粒度精细**：BSP 在 layer 级别决策（25% layer 4-bit, 75% layer 2-bit），PMQ 在 expert 级别决策（每个 expert 独立 1/2/3-bit）。同一 layer 内不同 expert 可获不同位宽，精度-效率 trade-off 更优。
    ## MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism

- baseline方法是什么？
  - Baseline 为 **FastMoE**（primitive expert parallelism）和 **FasterMoE**（pipeline parallelism + expert shadowing）。以 FasterMoE 为例说明全栈执行路径：
    - **算法层**：MoE 训练时，对每个 mini-batch 的 All-to-All dispatch → Expert FFN（Linear1 + GeLU + Linear2）→ All-to-All collect 三个阶段**串行**执行，通信阶段 GPU 空闲等待，计算阶段网络带宽空闲。FasterMoE 引入了 pipeline parallelism，但有两个关键缺陷：(1) 按 **node 维度**切分 batch（而非 batch 维度），将 All-to-All 拆解为多组 P2P 通信，丧失 NCCL 的 All-to-All 优化能力（如 ring/tree topology 聚合），且在异构带宽下同步等待造成资源浪费；(2) pipeline granularity **固定**，无法适应动态变化的 batch size 和网络条件。同时 FasterMoE **未考虑内存优化**，activation tensors 和 temporary buffers 占据大量 GPU DRAM，限制了可训练的 batch size 和模型规模。
    - **系统框架层**：基于 PyTorch + NCCL + CUDA 实现，使用 NCCL 的 All-to-All 原语进行 token dispatching。FasterMoE 通过 NCCL group calls 将 All-to-All 拆分为多路 P2P 通信，每个 group 内独立执行。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：默认 PyTorch CUDA kernel launch，未对 communication/computation/memory copy 的 stream 并行做系统性优化，三种操作串行执行或简单重叠，不考虑资源竞争（带宽竞争、SM 竞争）导致的 slowdown。
    - **硬件架构层**：NVIDIA A100 40GB / V100 16GB GPU，200 Gbps / 56 Gbps HDR InfiniBand，第 3 代 / 第 2 代 NVLink。
  - FasterMoE 的缺陷映射：
    1. **通信效率低**：按 node 维度切分导致 NCCL All-to-All 退化，pipeline granularity 受限于 node 数（通常 2-8），无法精细调优。
    2. **自适应能力差**：pipeline granularity 固定，无法随 batch size、模型规模、集群规模变化而动态调整。
    3. **内存占用高**：activation tensors（Equation 2：M_act = 4*B*M + B*H）和 temporary buffers（Equation 3）未优化，如 1.5B GPT-2 在 batch size=32、seq len=1K 时需要约 60GB GPU 内存。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - MPMoE 通过三个核心机制联合解决上述缺陷：
  - **全栈执行路径**（以 MoE-GPT-XL 在 Valor 集群 n=6 为例）：
    - **算法层 — 自适应 Micro-Batch Pipeline**：
      1. 将 mini-batch T_I(N, B, M) 沿 batch 维度切分为 n 个 micro-batch T_I[i](N, B/n, M)，而非沿 node 维度。
      2. 对每个 micro-batch 执行 pipeline: S(i) → C(i) → R(i)，三个阶段重叠执行。S(i+1) 在 C(i) 启动后并发开始，R(i) 在 C(i) 完成后启动，同时 S(i+2) 开始。交替调度 S 和 R stage 以增强内存访问局部性（Figure 7）。
      3. 自适应确定最优 n：MPMoE-pb 通过 Algorithm 1 的 profile-based 搜索（利用"n 单调递增于 B"和"性能关于 n 呈抛物线"的两个假设减少搜索空间），MPMoE-pm 通过 piecewise 性能模型估算（图 8 的 3 种 paradigm + 图 9 的 piecewise 速度拟合 + Section 2.3 的 α 干扰因子）。
    - **算法层 — Memory Reuse**：
      1. 识别"memory bubbles"：不同 micro-batch 的 T_DI[i]、T_M[i]、T_DO[i] 在不同时间激活，可共享 buffer。n 个 partition 的 buffer 需求从 O(n) 降为 O(1)（Figure 6）。
      2. 4 种恢复策略（S1-S4, Table 2）：按需组合 CPU offload、通信重放、重计算三种机制恢复后向所需的被覆盖 tensors，根据当前 N（GPU 数量）和 B（batch size）选择开销最小的策略。
    - **系统框架层**：
      1. 基于 PyTorch 1.9 + CUDA 11.1 + NCCL 2.7 实现，修改了 MoE layer 的 forward/backward 实现。
      2. 沿 batch 维度切分保留原始 All-to-All 语义（不降级为 P2P），充分利用 NCCL 对 All-to-All 的优化（合并小消息、ring/tree topology 聚合）。
      3. 使用 Tensor Cores 加速 Expert FFN 中的矩阵乘法（Linear1 和 Linear2）。
    - **kernel 调度层**：
      1. 使用多个 CUDA stream 并行执行 computation（C）、communication（S/R）、memory copy（M/D2H/H2D），通过 α(y,x) slowdown 因子（Section 2.3, Figure 3）量化并行操作间干扰。
      2. 建立 3 种 pipeline paradigm（图 8）的性能模型：P0-P4 五阶段分解，每个阶段由瓶颈 stream 的执行时间决定。
      3. Profiling 微基准（图 9）获取 W_comp, W_comm, W_mem 的 piecewise 速度函数（区分小体积/大体积数据的不同硬件利用率）。
    - **硬件架构层**：
      1. Adira: 64×A100 40GB + 200 Gbps InfiniBand + NVLink 3.0。
      2. Valor: 16×V100 16GB + 56 Gbps InfiniBand + NVLink 2.0。
      3. 自适应配置考虑了不同集群的硬件特性差异：Adira 上 MPMoE-pb 优于 MPMoE-pm（因网络波动大导致性能模型精度下降），Valor 上两者性能相当（网络稳定）。
  - 对比 baseline 的改进映射：
    - **FasterMoE 按 node 切分 → MPMoE 按 batch 切分**：保留了 NCCL All-to-All 的集体通信优化能力，避免 P2P 拆解带来的 kernel launch 开销和同步等待，pipeline granularity n 不受 node 数限制（可从 2 到 8 灵活选择），micro-benchmark（Figure 13）验证了更低的 dispatch/recovery 通信延迟。
    - **固定 pipeline → 自适应 pipeline（profile-based + performance model）**：MPMoE-pb 通过 Algorithm 1 的动态搜索缓存（G 和 C）在运行时学习最优 n，搜索次数随训练进行收敛；MPMoE-pm 通过性能模型在零 profiling 开销下估计（<1% overhead, Figure 16），在稳定网络环境（Valor）下与 pb 性能可比。
    - **未考虑内存优化 → Memory Reuse + 自适应策略选择**：4 种策略可根据 N（GPU 数）和 bottleneck 类型（计算瓶颈 vs 通信瓶颈）自适应选择——N 小时 CPU offload 更优（S1, S2），N 大时 recompute 更优（S4，避免 PCIe/memory bandwidth 竞争）。最终实现最高 53% 内存节省（Figure 11），达理论上限的 ~95%（Figure 12），同时可增加 batch size 以提升 GPU 利用率（减少 bubble overhead）。：Iⱼ = ‖tⱼ‖₁ · mean_attention 同时考虑 token 自身特征强度和跨 token 注意力，保护仅 2% 的关键 token 即可将 PPL 从 6.46 降至 6.24（≈3.4% 改善），且激活参数压缩比仅从 15.1% 降至 14.8%（几乎无损）。同时可进一步剪枝低重要性 token 的所有 expert（2% token masking → CR 15.8%, PPL 6.35），实现效率与精度双赢。
