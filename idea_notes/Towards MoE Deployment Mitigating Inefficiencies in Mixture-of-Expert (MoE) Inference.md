## Towards MoE Deployment Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference

- baseline方法是什么？
  Baseline 是 fairseq [23] 中基于 **Static Gating（静态容量门控）** 的 MoE Transformer 推理部署方案。Baseline 的核心缺陷：(1) **静态容量造成巨大 waste factor**——对于 LM（E=512, C=0.05），每个 token 需要 top-2 gating（仅 2 个专家真正计算），但每个专家固定处理 ECS = 512×0.05×S = 25.6S 个 token，waste factor = 12.8×；对于 MT（E=128, C=1），waste factor = 64×；(2) **dispatch mask 矩阵乘法的内存开销**——每次 MoE 层都构造大小为 (E, S, S×C) 的稀疏 dispatch mask，通过 batch 矩阵乘法（BMM）将 tokens 分发到各专家，LM batchsize=8 时激活内存高达 6.29GB；(3) **专家参数完全驻留 GPU**——所有 512 个 expert FFN（LM 52B 模型）的参数全部占用 GPU HBM，静态内存达 18.88GB（单 GPU），限制了 batch size 扩展；(4) **专家负载极度不均衡**——hot experts 承载大量 tokens，cold experts 几乎空闲，且某些 experts 完全不被激活（MT Decoder 约 75% experts 不活跃），但不活跃的 experts 仍占据 GPU 内存并处理空 placeholder。

  Baseline 全栈执行例子（以 LM 推理，batch_size=8，单节点 8×V100 为例）：
  - **算法层**：Transformer decoder-only LM，24 layers，每 MF=2 层中 1 层为 MoE 层（含 512 个 expert FFNs）。Gating 为 top-2，static capacity C=0.05。token emb → MHA → Static Gating → Dispatch Mask BMM → All-to-All (fixed size) → Expert FFN → All-to-All → Combiner BMM。每 expert 固定处理 25.6S tokens，大部分为零填充（placeholder 向量），产生 token dropping 风险（超出容量的 token 被丢弃，仅靠残差连接保留信息）。
  - **框架层**：fairseq MoE Transformer + expert parallelism（每个 GPU 持有 64 个 experts for LM E=512/8 GPU）。PyTorch 实现，通信使用 NCCL all-to-all collective（单节点内 NVLink 300GB/s，多节点 InfiniBand）。
  - **编译框架层**：论文未明确说明（使用 PyTorch eager mode / fairseq 默认编译路径）。
  - **Kernel 层**：cuBLAS 矩阵乘法（expert FFN 的小 batch GEMM）、NCCL all-to-all（使用 NVLink/IB 的 RDMA 原语）、dispatch/combiner batch 矩阵乘法（cublasGemmBatchedEx 或 cublasGemmStridedBatchedEx）。大量时间消耗在 dispatch BMM 和 fixed-size all-to-all 上。
  - **硬件层**：NVIDIA V100 GPU（32GB HBM2），SM 执行 FFN 的 MAC 运算，HBM 存 expert 参数和中间激活，NVLink 互联用于 all-to-all 通信。CPU（Xeon E5）仅用于数据预处理，不参与推理计算。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文通过三项互补优化解决 baseline 缺陷：
  **(1) Dynamic Gating** 消除静态容量约束：用 argsort + 两阶段 all-to-all 替代 dispatch mask BMM。不再传输空 placeholder，只传输实际分配给 expert 的 tokens。直接解决了 waste factor（12.8× 或 64×）和 dispatch mask 的激活内存（LM batchsize=8 从 6.29GB 降至 1.28GB）。同时，token 不再被丢弃（因为不再有容量限制），提升了模型鲁棒性。
  **(2) Expert Buffering** 减少静态参数内存：利用 expert 激活的时序局部性（同一 expert 连续 batch 活跃），在 GPU 内存中只缓存 hot experts，其余放在 CPU 内存按需加载。通过 LIFO 淘汰策略 + Memcopy 与 all-to-all 通信 overlap 隐藏延迟。MT Decoder 静态内存减少 2.25GB（1.47× reduction）。
  **(3) Load Balancing** 改善设备间负载分布：基于历史激活数据，将 expert-to-GPU 映射形式化为 multi-way number partitioning（NP-hard），用 greedy 算法近似求解，将高负载和低负载 expert 混搭分配到各 GPU，减少瓶颈设备。LM 的 Max Load 从 0.6 降至 0.4 以下。

  论文方法全栈执行例子（以 LM 推理，batch_size=64，单节点 8×V100，所有优化启用为例）：
  - **算法层**：同样 Transformer decoder-only LM 结构。变化点：Gating → top-2 仍输出 expert assignments；**Dynamic Gating**：不再构造 dispatch mask，而是 argsort(flat_expert_indices) → 得到最优 token 排列 idx → bincount 统计每 expert token 数 → 两阶段 all-to-all（先传 sizes，再传 data）→ 各 GPU 的 expert FFN 只处理实际分配到的 tokens → index-based gather 恢复顺序。整个过程 O(S log S + SD)，远小于原方案 O(S²EDC)。**Expert Buffering**：接收到 all-to-all sizes 后，检查本 GPU 上的 active experts 是否在 GPU cache 中 → hit 直接使用，miss 触发 CPU→GPU Memcopy（与 all-to-all 并行）。**Load Balancing**：初始化时按 greedy 算法将 512 experts 分配到 8 GPU（每 GPU 64 experts），高负载+低负载混合。
  - **框架层**：仍是 fairseq，但 gating 模块改写：新增 DynamicGatingLayer（含 argsort、bincount、two-phase all-to-all wrapper）、ExpertCache（含 GPU buffer、CPU parameter store、LIFO eviction、Memcopy overlap 逻辑）、LoadBalancedExpertPlacement（含 greedy/anti-correlation assignment 模块）。Python+PyTorch+CuPy 实现。
  - **编译框架层**：论文未明确说明。
  - **Kernel 层**：核心变化——(a) dispatch/combiner 的 BMM 被 index-based gather/scatter 替代（大幅减少内存访问和计算量）；(b) all-to-all 通信量从固定 EC tokens 降至实际 token 分配量（waste factor 消除），通信量降至原方案的 1/12.8（LM）或 1/64（MT）；(c) 新增 CUDA Memcopy kernel 用于 CPU→GPU expert 参数传输，与 NCCL all-to-all 通过 CUDA stream 并行 overlap。
  - **硬件层**：同一 V100 平台。但 batch size 可从 8 扩展到 64（单节点 LM），得益于：(a) 动态内存从 6.29GB 降至 1.28GB；(b) 静态内存通过 Expert Buffering 降低 2.25GB（MT Decoder）。多节点时性能提升更显著（LM 多节点吞吐提升达 11.55×），因为减少的通信量在跨节点场景下影响更大。瓶颈从 GPU 内存容量和通信带宽转移到 CPU-GPU 带宽（Expert Buffering 的 Memcopy 路径），实测 CPU-GPU 带宽饱和在 12GB/s（PCIe 3.0）。
