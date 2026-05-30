## Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems

- baseline方法是什么？
  **Baseline 为两类**：
  
  1. **MoNDE [18]（GPU-NDP, context-agnostic expert placement）**：基于全局历史专家激活频率统计进行静态 hot/cold 分类——hot experts 常驻 GPU HBM（FP16），cold experts 驻留 CXL-NDP 设备执行。当 router 选中的 expert 不在 GPU 时，触发 on-demand 迁移（expert weights 从 NDP → GPU via PCIe）或由 NDP 执行。核心问题：
     - **静态/全局频率忽略 context dependence**：同一 expert 对不同输入序列的 heat 程度不同，全局统计无法捕捉 per-sequence 的激活变化
     - **on-demand expert migration**：decoding 期间频繁触发参数传输（expert weight ~数百 MB/次），PCIe 带宽成为瓶颈，GPU 利用率下降
     - **NDP compute pressure 未解决**：cold experts 以 FP16 在 NDP 上执行，NDP 计算吞吐有限（64×(4×4) systolic arrays），成为系统瓶颈
  
  2. **HOBBIT [31]（GPU-only mixed-precision offloading）**：所有 experts 和 attention 均在 GPU 执行，使用混合精度（FP16 + INT4）从 CPU memory 加载 expert weights。缺陷：无 NDP 近数据执行优势，expert weights 仍需通过 PCIe 传输到 GPU。

  **Baseline 全栈执行例子（以 MoNDE, Mixtral-8×7B, 4 GPU/4 NDP experts/layer 推理一个序列为例）**：
  
  - **算法层**：输入 tokens → Self-Attention → Router (top-2 gating) → 每 token 选择 2 个 experts → 若选中 expert 在 GPU → 本地 FP16 FFN；若选中 expert 在 NDP → 若 on-demand policy → expert weight (FP16, ~每 expert ~170MB for Mixtral-8×7B) 从 NDP DDR 经 PCIe → GPU HBM → GPU FFN 计算，或 activation 经 PCIe → NDP → NDP FP16 计算 → activation 回传 GPU
  - **系统框架层**：MoNDE runtime → GPU memory manager 维护 hot expert cache → cold experts 在 NDP DDR → expert migration triggered per layer per decoding step → 迁移延迟 90%+ of transformer block time
  - **编译框架层**：论文未明确说明（标准 PyTorch + custom runtime）
  - **Kernel调度/运行时计算层**：GPU 端标准 cuBLAS GEMM；NDP 端 systolic array FP16 GEMM → NDP compute units bottleneck（64 arrays × 16 MAC 仅 ~1 TOPS vs H100 989 TFLOP/s）
  - **硬件架构层**：H100 GPU + DDR-based NDP via PCIe Gen4 ×16 → PCIe BW ~32 GB/s vs HBM3 ~3.35 TB/s → 参数传输主导延迟

  **Baseline 的核心缺陷**：
  1. **Context-agnostic placement 忽略 expert activation dynamics**：不同输入序列、不同 decoding step 的 expert 激活模式不同，静态/频率统计无法捕捉
  2. **Parameter Movement 开销巨大**：Expert weight 传输数百 MB vs activation 仅数 KB —— MoNDE 将计算延迟转化为参数传输延迟
  3. **NDP compute pressure**：FP16 在受限 NDP 硬件（64× systolic arrays）上执行成为瓶颈，低量化位宽是释放 NDP 潜力的关键
  4. **频繁 migration 的带宽争用**：on-demand swapping 在 decoding 期间持续触发 GPU↔NDP 传输，挤占 pipeline 效率
  5. **Prefill 信息被浪费**：Prefill 阶段自然产生的 expert 激活统计信息未被利用，decoding 从零开始做决策

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：Context-Aware MoE Inference on CXL-NDP = **Prefill-Guided Expert Placement** + **Context-Aware Mixed-Precision Quantization for NDP**。核心洞察：prefill 阶段的 expert 激活分布与 decoding 阶段高度相似（cosine similarity avg=0.89），因此 prefill 统计可作为 decoding 阶段 placement 和 quantization 的可靠先验。

  **Defect → Design 映射**：

  | Baseline 缺陷 | 论文设计选择 | 解决机制 |
  |---|---|---|
  | Context-agnostic placement 忽略动态性 | Prefill-guided placement: S_{l,e} = αP̃_{l,e} + (1-α)W̃_{l,e} 基于 prefill 统计 | Prefill-decode activation 相似度 0.89 → per-sequence once placement，捕捉 context-dependent hot/cold 模式 |
  | Parameter Movement 开销（~数百 MB/expert） | NDP 近数据执行 + single migration: expert 仅在 prefill 后迁移一次 | 转化为 Activation Movement (~8KB/token) → 数据移动量减少 10^4-10^5× |
  | NDP compute pressure (FP16 瓶颈) | Context-aware mixed precision: 1-4 bit per-expert based on importance | NDP 量化执行 ~5-8× latency reduction，硬件计算压力降低 |
  | 频繁 migration 带宽争用 | Once-per-sequence fixed placement: decoding 期间 zero migration | PCIe 仅用于 activation 传输，无 expert weight 争用 |
  | Prefill 信息被浪费 | Prefill 阶段即收集 (P_{l,e}, W_{l,e})，驱动 placement + bitwidth 决策 | 零额外推理开销（counter update 可忽略），信息复用最大化 |
  | 统一量化忽略 expert heterogeneity | Prefix-structured 分配：重要 expert 高 bitwidth，次要 expert 低 bitwidth | 在固定平均 bitwidth budget 下最大化量化增益，Ours-2bit + selector vs w/o selector: +3.2% avg accuracy |

  **论文方法全栈执行例子（以 Mixtral-8×7B, K=4 GPU/4 NDP experts/layer, Ours-3bit, 一个推理请求为例）**：

  - **算法层**：
    1. Prefill: tokens → Attention → Router → 收集 (P_{l,e}, W_{l,e}) for all 32 layers, 8 experts/layer
    2. Importance: S_{l,e} = 0.5P̃_{l,e} + 0.5W̃_{l,e} → per-layer top-4 → H_l (GPU), bottom-4 → C_l (NDP)
    3. Bitwidth: 4 NDP experts, b_bar=3 → R=8 increments → prefix-structured search → e.g., (n4=2, n3=2): top-2 NDP experts → 4-bit GPTQ, bottom-2 → 2-bit GPTQ
    4. Decoding: per token → Router → top-2 experts → if GPU → H100 FP16 FFN; if NDP → activation sent via PCIe → NDP systolic array (3-bit/2-bit) → result back → weighted sum
    5. Overlap: GPU computing layer l hot experts while NDP computing layer l-1 cold experts → pipeline efficiency

  - **系统框架层**：
    - Prefill 统计收集器：轻量级 per-layer counter array (8 counters × 2 metrics × 32 layers = 512 values)
    - Expert Placement Module：prefill 后执行一次 O(L·E) 排序，单次 expert migration (PCIe weight transfer)
    - Expert Bitwidth Selector：O(L·E_NDP^2) 前缀枚举搜索，per-sequence 执行一次
    - Decoding Runtime：固定 placement，在 GPU 和 NDP 间按 router 结果分派计算

  - **编译框架层**：论文未明确说明。

  - **Kernel调度/运行时计算层**：
    - GPU 端：H100 tensor cores 执行 FP16 GEMM（hot experts），标准 cuBLAS
    - NDP 端：量化 GEMM on 64×(4×4) systolic arrays——不同 bitwidth 的 effective throughput 不同（4-bit ~4× faster than FP16 equivalent, 1-bit uses XNOR+popcount）
    - PCIe activation transfer：per-token 8KB (4096-dim FP16) vs per-expert weight ~170MB — 约 2×10^4× 减少
    - GPU-NDP pipeline overlap：GPU stream 1 (hot FFN) || NDP stream (cold FFN via PCIe)

  - **硬件架构层**：
    - H100 GPU: 80GB HBM3, 132 SMs, 989.4 TFLOP/s → 处理 hot experts + attention + router
    - CXL-NDP: 512GB DDR, 512 GB/s internal BW, 64×(4×4) systolic arrays @ 1 GHz → 处理 cold experts (量化)
    - PCIe Gen4 ×16: ~32 GB/s → activation movement 通道（非 parameter movement 通道）
    - 关键：NDP 512 GB/s 内部带宽 >> PCIe 32 GB/s → 近数据执行利用 NDP 高内部带宽

  **对比 Baseline 的核心改进路径**：
  Baseline (MoNDE, context-agnostic):
  Prefill (no stats) → Decoding: per step per layer: Router → if cold expert needed → [Parameter Movement: ~170MB expert weight NDP→GPU via PCIe] OR [NDP FP16 compute: systolic arrays bottleneck] → GPU wait → FFN → next layer
  瓶颈: PCIe parameter transfer OR NDP compute (FP16)

  Ours (Context-Aware):
  Prefill → [Collect (P,W) stats] → [Importance + Placement + Bitwidth, once] → [Single expert migration, once] → Decoding: per step per layer: Router → if GPU: local FP16 FFN; if NDP: [Activation Movement: ~8KB via PCIe] → [NDP b-bit compute, 5-8× faster] → [Activation back via PCIe]; GPU hot FFN || NDP cold FFN (overlap)
  优势: Activation Movement + NDP low-bit compute + zero decoding migration + overlap

  **关键设计决策对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | Prefill-guided once-per-sequence placement | 消除 decoding 期间 expert migration | 6.6-8.3× speedup vs MoNDE |
  | Context-aware mixed-precision (1-4 bit) | 降低 NDP compute pressure | NDP latency 5× (3-bit) / 8× (2-bit) reduction |
  | Prefix-structured bitwidth allocation | 在固定 avg bitwidth 下最大化精度 | +3.2% accuracy (Ours-2bit w/ selector vs w/o) |
  | Prefill statistics (activation freq + routing score) | 捕捉 context-dependent expert importance | Prefill-decode cosine sim 0.89 |
  | Parameter → Activation Movement | 从 ~170MB→~8KB per expert invocation | PCIe 带宽使用大幅降低 |
  | GPU-NDP pipeline overlap | NDP compute hidden behind GPU compute | 端到端 8.7× decoding throughput vs MoNDE |
  | 保留部分 FP16 experts on GPU (K per layer) | 关键 experts 无损精度 | 仅 0.13% avg accuracy drop (Ours-3bit) |
