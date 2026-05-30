## GPU Memory Hierarchy (HBM vs SRAM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GPU Memory Hierarchy（GPU内存层次结构）是现代GPU的多级存储体系，由不同容量和速度的内存层级组成。以NVIDIA A100为例：(1) **HBM (High Bandwidth Memory)**——off-chip DRAM，40-80GB容量，带宽1.5-2.0TB/s，是GPU的主内存，存储模型参数、激活值和中间张量；(2) **L2 Cache**——on-chip，40MB，连接所有SM（Streaming Multiprocessors）；(3) **On-chip SRAM (Shared Memory + L1 Cache)**——per SM，共192KB（可配置split between shared memory and L1），带宽约19TB/s（~10× faster than HBM）；(4) **Registers**——per thread，最快但最小（每SM 65536 registers = 256KB）。越靠近计算单元的内存越快但越小，呈现经典的memory hierarchy pyramid。FlashAttention的关键洞察：attention的中间结果（N×N矩阵）太大无法fit in SRAM，因此必须反复在HBM↔SRAM间搬运，HBM带宽成为瓶颈。通过tiling将计算组织为"每次仅加载fit in SRAM的block"的streaming模式，使数据尽量在SRAM中复用后再写回HBM。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

GPU memory hierarchy在attention计算中的数据流（以A100为例）：
```
HBM (40-80GB)
  │  BW: 1.5-2.0 TB/s
  │  Latency: ~400-800 cycles
  ▼
L2 Cache (40MB, shared across all SMs)
  │  BW: ~7 TB/s
  │  Latency: ~200 cycles
  ▼
SM On-Chip Memory (per SM):
  ├── Shared Memory (configurable, up to 164KB): explicitly managed by programmer
  │     BW: ~19 TB/s, Latency: ~20-30 cycles
  └── L1 Cache (configurable, remainder of 192KB): hardware-managed
       BW: ~19 TB/s
  ▼
Registers (256KB per SM, 65536 × 32-bit)
  BW: ~40 TB/s, Latency: 0 cycles (immediate)

Data flow for FlashAttention:
  HBM → L2 → Shared Memory (K_j, V_j, Q_i tiles)
  Shared Memory → Registers (for compute)
  Registers → Tensor Core (MMA) → Registers → Shared Memory (O_acc)
  Shared Memory → HBM (final O, m, l)
```
关键硬件设计约束：(1) SRAM容量极小（192KB/SM），N×N矩阵根本无法放入；(2) 数据在HBM→SRAM间传输由程序员通过`__shared__` memory和coalesced global memory loads/writes显式管理；(3) 每SM可同时resident多个thread block（受限于SRAM/register/thread budget），SM occupancy影响延迟隐藏能力。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GPU memory hierarchy的使用策略：(1) 将经常复用的数据放入shared memory（如FlashAttention中K_j tile在inner loop中被所有Q_i复用）；(2) 使用coalesced global memory access最大化HBM带宽利用率（128-byte alignment）；(3) 通过`__syncthreads()`确保shared memory写入对所有threads可见；(4) 在Hopper+架构上使用TMA（Tensor Memory Accelerator）进行异步HBM→shared memory拷贝（FlashAttention-3的核心优化）。Memory hierarchy-aware design已成为GPU kernel优化的标准方法论——NVIDIA的CUDA Best Practices Guide和Roofline Model均以此为核心理念。

涉及论文标题：
- FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness
