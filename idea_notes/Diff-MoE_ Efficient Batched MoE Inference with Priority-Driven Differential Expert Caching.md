## Diff-MoE: Efficient Batched MoE Inference with Priority-Driven Differential Expert Caching

- baseline方法是什么？
  - Baseline 是 MoE 推理中现有的 offloading 方案，分为两类：(a) Prefetch-based（如 Pre-gated MoE）——提前预取下一 MoE 层的全部或预测 experts，重叠传输与计算；(b) Cache-based（如 MoE-Infinity、LRU-based caching）——在 GPU memory 中缓存频繁激活的 experts，基于 LRU 或估计重用概率进行驱逐。
  - 全栈执行例子（Baseline: MoE-Infinity，batch_size=64，Switch-Base，XSum，5% cache ratio）：
    - **算法层**：MoE gating network 对每个 token 计算 Softmax(LinearGate(X)) → TopK (默认 k=1)。batch 中约 34 个不同 experts 被激活。
    - **系统框架层**：MoE-Infinity 在 FasterTransformer 基础上实现全局共享缓存（所有 layers 共享一个 cache pool），按估计的 expert 重用概率管理驱逐。batch_size=64 时，每次迭代激活的 ~34 experts 几乎占满 5% 缓存容量（36 experts），导致缓存被整批刷新，下一迭代命中率骤降至 <0.1%。每个 miss 触发 host→GPU PCIe 传输（128 GB/s），通信时间主导延迟。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：FasterTransformer 的 MoE kernel 在 GPU 上并行执行 FFN 计算。Expert 参数的 host→GPU 加载由 FasterTransformer 的 offload 模块按需触发 cudaMemcpy。计算时间（batch matmul）基本恒定（约 2-2.5 ms for batch=64），而通信时间随 batch size 线性增长至 ~15-20 ms，占 per-iteration 延迟的 97% 以上。
    - **硬件架构层**：H200 GPU (141 GB HBM) ↔ Host DRAM (1 TB) 通过 PCIe 5.0 (128 GB/s) 连接。每个 expert 参数约 85 MB（7B/128/2，考虑 MoE 占约一半层）。batch=64 时 34 experts 的总传输量约 2.9 GB，占 PCIe 带宽 ~23 ms 传输时间。Prefetch-based 方案最多只能隐藏 1-2 个 expert 的传输（对应 ~2.6 ms 计算窗口），其余无法重叠。
  - Baseline 缺陷根因：在 batched inference 下，随着 batch size 增大：(1) Prefetch-based 方法的通信时间增长远超计算时间增长（batch 1→16，通信 6.53× vs 计算 1.55×），可隐藏的传输比例急剧下降；(2) Cache-based 方法的缓存命中率随 batch 增大而崩塌（batch 1→16, 5% cache ratio 下 miss rate 从 6.91% → 68.84%），因为单次迭代激活的 experts 集合频繁超过缓存容量，导致大量替换和重复传输。两种方法的共同根因是**没有利用 expert 激活的全局和时间局部性（global & temporal locality）**来差异化缓存管理。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：Diff-MoE 通过观察发现 MoE 推理中 expert 激活具有两种局部性——(a) Global Locality：每层中少数 experts 在全部推理过程中频繁激活（top-6 experts 占 33%-83% 激活）；(b) Temporal Locality：某些 experts 在短解码窗口内（3-6 次迭代）反复激活（1.13×–2.40× reactivation）。基于此提出三层差分缓存层级 + GRU predictor。
  - 全栈执行例子（Diff-MoE, batch_size=64, Switch-Base, XSum, α=5%）：
    - **算法层**：
      - Priority Initialization：离线微调阶段统计每个 MoE layer 的 expert 激活频率，top-N（默认 N=2, α=5%）标记为 globally hot → p=MaxP=2, 永久锁定在 HPCi。
      - Dynamic Priority Update：在线推理每层 MoE 执行时，按公式 p_k^i = clip(p_k^i + Δ_inc if activated | p_k^i - Δ_dec_in if inactive+cached | p_k^i - Δ_dec_out if inactive+uncached) 更新 non-global experts 的优先级。
      - Locality-Preserving Replacement：当 activated expert 优先级 ≥ threshold_hot(=1) 且不在 MPCi 时，按优先级降序排列候选；MPCi 中优先级 < 1 的 resident 按优先级升序排列；最高候选替换最低 resident。
      - GRU Predictor：6 层 GRU，以当前层 expert IDs 为输入，输出下一层各 expert 的激活概率分布。batch 内多个样本聚合 → top-2 uncached experts 预取到 LPC。
    - **系统框架层**（FasterTransformer 修改）：
      - 在初始化阶段将每个 MoE layer 的 globally hot experts 永久加载到 HPCi（各层 2 个，共 12 experts，约 1 GB）。
      - 创建 per-layer MPCi（各层 4 experts，共 24 experts，约 2 GB）和跨层共享 LPC（临时缓冲当前层激活 experts + 预取 experts）。
      - 在 gating 后拦截 expert 加载：先查 HPCi → MPCi → LPC，仅对三级都 miss 的 expert 触发 host→GPU 传输。
      - 在当前层计算期间，异步预取预测的下一层 experts 到 LPC 预取缓冲区，与计算重叠。
      - 当前层计算完成后：LPC 中的 locally hot experts 晋升到 MPCi，其余驱逐。LPC 仅保留已预取的下一层 experts。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：FasterTransformer MoE kernel 不变。差异在于：Baseline 的 per-iteration expert 传输量由 batch 激活的全部 experts 决定（batch=64 约 34 experts × 85 MB = 2.9 GB），而 Diff-MoE 通过 HPC/MPC 命中削减了大量传输——28.5% cache hit rate（vs MoE-Infinity <0.1%），实际传输量大幅减少。Predictor 的 top-1 accuracy 在 batch≥16 时稳定 >90%，top-2 accuracy 则更高。预取的 1-2 个 experts（约 170 MB）传输完全被 GPU 计算时间覆盖。
    - **硬件架构层**：同一 H200 + PCIe 5.0 硬件。区别在于：Baseline 的 memory hierarchy 是 Host DRAM → GPU LPC 的简单二级结构；Diff-MoE 引入 Host DRAM → GPU LPC → GPU MPC → GPU HPC 的四级结构（LPC 为临时层，MPC 和 HPC 为持久层），且 HPC/MPC 按 layer 隔离而非全局共享。这保证在 batched 场景下每层都有独立的缓存空间（不会因其他层的 experts 挤占而驱逐本层热点），避免全局缓存竞争导致的命中率崩塌。
  - 解决 Baseline 缺陷的方式：
    1. **针对 Prefetch-based 方案（通信增长远超计算增长）**：Diff-MoE 通过 HPC 和 MPC 的差分缓存消除冗余传输——globally hot experts 一次性加载后永驻 GPU，locally hot experts 在短窗口内复用不反复传输。只有 cold experts 才按需加载，大幅削减传输总量，使有限的 PCIe 带宽不再被重复流量占满。
    2. **针对 Cache-based 方案（命中率随 batch 增大崩塌）**：Diff-MoE 用 per-layer 独立缓存 + 优先级驱动替换取代全局共享缓存 + LRU。per-layer 设计保证每层有稳定缓存容量（不被其他层挤占）；优先级机制综合频率和时效信息（LRU 仅看 recency），保护了短窗口内即将复用的 locally hot experts，避免了 LRU 在单次迭代刷新全部缓存的问题。
    3. **通用性**：Diff-MoE 不修改 MoE 模型架构本身（不改变 gating、expert 权重、top-K），使其可直接适用于任何 MoE-based LLM，部署成本低。
