## QMoE Sub-1-Bit Compression of Trillion-Parameter Models

- baseline方法是什么？
  - Baseline 是 Round-To-Nearest (RTN) 量化——直接将 bfloat16 权重按量化网格进行最近邻舍入，无数据依赖校准。RTN 在 2-bit 精度下尚可运行但 loss 显著增加（base128: 1.73→2.27, c2048: 1.18→1.33），三元精度下几乎崩溃（base128: 4.54）。另一个 baseline 是"不压缩"——c2048 的 bfloat16 推理需 3.2TB 存储，对应 >65 A6000 或 >130 3090 GPU，对于普通硬件完全不可行。
  - 全栈执行例子（Baseline: RTN 三元量化 + 朴素 sparse format, SwitchTransformer-c2048, bfloat16 inference, 需 >65 A6000）：
    - **算法层**：RTN 对每行权重独立量化——w_min = min(W_row), w_max = max(W_row)，三元网格 {w_min, 0, w_max}，每个权重映射到最近的值。无 calibration data 参与，无法补偿跨层误差累积，不做 Hessian 校正。量化后权重具有高稀疏度（~88.6% 零值），但直接使用 CSR/bitmask 等 sparse format 存储时：bitmask 占 1 bit/param（几乎抵消三元 2-bit 表示的压缩），column index 占 10-13 bit/param（反而膨胀）。
    - **系统框架层**：HuggingFace Transformers 加载完整的 bfloat16 checkpoint (3.2TB)，标准 autoregressive decoding 流程。每层 MoE: router → Top-1 expert selection → expert FFN (2×GEMM + GELU) → combine。若无压缩，需通过模型并行（tensor parallelism + expert parallelism）将 expert 分片到 ~65 张 A6000 或 ~130 张 3090 GPU。单 GPU 无法装入完整模型，涉及多机通信开销。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准 cuBLAS bfloat16 GEMV kernel。在 uncompressed 场景下接近内存带宽理想利用率（因矩阵较小、访存模式规则）。压缩后的 RTN + sparse format 场景下，由于 metadata 开销（bitmask/column index），全局内存读取量并不比 uncompressed 减少多少，且 sparse kernel 的开销（gather/scatter、非连续访存）显著降低有效带宽。
    - **硬件架构层**：>65×A6000 GPU 组成多机集群，通过 NVLink（节点内）+ Infiniband/网络（节点间）互联。各 rank 持有 expert shards，all-to-all 通信将 token 路由到对应 expert 所在 GPU。多机部署的核心成本在内存（3.2TB 模型权重）和通信（跨机 token dispatch）。
  - Baseline 核心缺陷根因（三个）：
    1. **Post-training 量化精度不足**：RTN 无法突破 ~3 bit/param 的"可用精度墙"——对于 MoE 模型虽比 dense 模型更鲁棒（不崩溃），但三元精度下 loss 增加太大无法实用。根本原因是 RTN 只做逐元素的最近邻映射，不利用 layer-wise Hessian 信息来补偿量化误差的跨层累积。
    2. **Sparse format metadata 开销抵消压缩收益**：三元量化后虽 ~88.6% 权重为零，但存储这些零的位置信息（bitmask 或 column index）比压缩后的 2-bit 权重本身还大，直接使用 sparse representation 无法实现 sub-1-bit 压缩。
    3. **Scaling 瓶颈**：现有 data-dependent 量化方法（如 GPTQ）针对 dense 模型优化——每层 few large matrices 的 GPU 利用率良好，但 MoE 有 1000× 更多的小层，导致现有实现 GPU 利用率极差、内存需求巨大（需存储 >100× 的 calibration data）、可靠性问题（10000+ 层中大概率 hit 数值不稳定 edge case）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文提出 QMoE，通过系统-算法-格式-内核的垂直协同设计，将万亿参数 MoE 压缩到 <1 bit/param 并实现高效推理。核心创新链：(1) Scalable GPTQ for MoEs → (2) 利用自然稀疏的低熵字典编码 → (3) Warp-per-row 字典解码 CUDA kernel。
  - 全栈执行例子（QMoE, SwitchTransformer-c2048, 4×A6000）：
    - **算法层（解决"Post-training 量化精度不足"和"Scaling 瓶颈"缺陷）**：
      - **Scalable GPTQ Adaptation for MoEs**：将 GPTQ 的 layer-wise Hessian-based 量化扩展到 MoE 场景——(a) Activation Offloading：calibration data 的中间激活存于 CPU RAM 的 list buffer 数据结构（大连续 buffer + delimiter indices），GPU 仅按需取小块数据，实现单卡 A6000 处理 160K calibration samples；(b) Expert Grouping：将 16 个 expert 组批处理，权重和 Hessian 堆叠为 3D tensor，batched GPTQ 算法同时压缩组内所有 expert，实现约 6× 加速（vs per-expert 串行）；(c) Lazy Weight Fetching：3.2TB 原始模型权重按需从磁盘加载到 GPU，压缩后写回并释放内存——不必同时加载全部权重到 RAM；(d) Robustness Mods：10× 提高 Hessian dampening (δ=0.1)、对不可逆 Hessian layer 退化为 RTN、token cap 为均值 4× 防 OOM、特殊 token premasking（MLM 的 mask token 从校准数据中排除，因模型对其过于鲁棒、误差补偿无益）。
      - 对比 baseline：RTN 三元精度 c2048 loss 从 1.18→2.15 (+82%)；QMoE (GPTQ) 三元精度 loss 仅 1.18→1.26 (+6.7%)。RTN 等价于 Hessian = Identity（无误差补偿），GPTQ 通过二阶 Hessian 信息逐列校正量化误差的跨层传播。
    - **系统框架层（解决"Scaling 瓶颈"缺陷）**：
      - 基于 PyTorch + HuggingFace Transformers 实现，所有修改通过运行时动态 patch（无需修改官方安装）。修复 HuggingFace 对大 MoE 的两个 bug：(a) 配置和模型设置 fix；(b) 跳过无 token 分配的 expert 的（空）CUDA kernel launch——>10× 加速大模型推理。
      - Compression pipeline 的 CPU-GPU 协同设计：list buffer in CPU RAM + lazy weight fetching from disk → 单 A6000 可在 ~16h 内压缩 c2048（1.6T params）。原始模型仅需加载到磁盘（3.2TB），不要求同时装入 RAM。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层（解决"Sparse format metadata 开销抵消压缩收益"缺陷）**：
      - **Dictionary-Based Sub-1-Bit Encoding**：利用三元量化后 88.6% 自然稀疏度带来的低熵——P(0)≈0.886, P(1)≈P(2)≈0.057——设计 fixed-length codeword → variable-length sequence 的字典编码。算法 1（max-priority queue）生成 2^16 个最高概率的三元对序列（每序列 ≤14 pairs = 28 weights），存入 UINT16→2×UINT32 的字典。编码时不直接存储零的位置信息（bitmask/column index），而是用概率最高的 codeword 更频繁地表示"多个零连续出现"的 pattern——信息论上接近熵编码但不需要变长码字的序列依赖。
      - c2048 实现 20.07× (MoE-only) / 19.81× (full model) → 0.807 bits/param。与理论极限 25.40× 仅差 ~20%，换取快速 GPU 解码。
      - 对比 baseline：Baseline 的 bitmask (1 bit/param) + 2-bit ternary = 3 bits/param，无 net compression；column index (10-13 bits) 更差。QMoE 的字典编码直接压缩到 0.8 bits/param，不存储任何 per-weight metadata。
      - **Custom CUDA Decoding Kernel (Sub1MatVec)**：warp-per-row 并行，每 warp 通过 coalesced load 取 32 个 UINT16 codewords → 查 GPU L2 cache 中的 512KB 字典 → 28/32 threads 并行提取 2-bit ternary 权重 → shared memory dequant lookup table（复制 32× 避免 bank conflict）→ FMA 累加 → warp shuffle reduction。字典按概率降序排列，高频 codeword 自动 L1 cache prefetch。Global memory 读取量仅约 1/20 of bfloat16（0.8 vs 16 bits/param），bit ops 开销被 global memory latency（~200 cycles vs ~1 cycle）完全隐藏。性能上，压缩 kernel 在所有矩阵形状上比 uncompressed cuBLAS bfloat16 GEMV 更快（最高 35% speedup）。
    - **硬件架构层**：4×A6000 (48GB) 或 8×3090 (24GB) 单服务器。核心变化：baseline 的 c2048 bfloat16 推理需 >65 A6000 / >130 3090 GPU（多机集群），QMoE 压缩后仅需 4 或 8 卡（单服务器）。每 GPU 约 40GB compressed model weights (160GB total / 4)。Uncompressed baseline 因 GPU 数量多、跨机通信开销大；QMoE 在单服务器内 GPU 间 NVLink 互联、无可感知通信瓶颈。端到端推理仅 <5% 开销 vs 理想化 uncompressed baseline（该 baseline 估计将所有 expert 指向同一权重数据来回避内存限制——实际部署需 20× 更多 GPU 及对应通信开销，因此 <5% 是下界估计）。
  - 解决 Baseline 缺陷的方式总结：
    1. **针对"RTN 精度不足"**：QMoE 将 GPTQ 的 data-dependent Hessian-based 量化适配到 MoE 场景（通过 expert grouping 批量化 + activation offloading + robustness mods），三元精度下 loss 增加仅 6.7%（vs RTN 的 82%）。关键 insight：GPTQ 用每层 Hessian 矩阵的二阶信息逐列校正量化残差，RTN 等价于 Hessian=Identity（无校正）。
    2. **针对"sparse format metadata 开销"**：不直接使用 sparse format，而是通过字典编码利用低熵（而非直接 sparsity）实现压缩——高概率 codeword 编码"频繁出现的零模式"，不存储 per-weight 位置信息。字典优化为 c2048 权重分布，c2048 压缩率 20.07×，与独立同分布模型仅差 ~5%。
    3. **针对"Scaling 瓶颈（GPU 利用率差、内存需求大、可靠性差）"**：Activation offloading（list buffer + CPU RAM）+ Lazy weight fetching（磁盘按需加载）+ Expert grouping（16 expert batch = 6× 加速）+ Robustness mods（dampening、fallback RTN、token cap、premasking）→ 实现单卡 A6000 在 <1 天内压缩 c2048 (1.6T params)。
    4. **针对"压缩后解码效率"**：Dictionary code 的 fixed-length codewords + warp-per-row 并行 + L2-cache resident 字典 + shared memory dequant + coalesced access → 解码开销被 global memory latency 完全隐藏，压缩 kernel 比 uncompressed cuBLAS 更快。
