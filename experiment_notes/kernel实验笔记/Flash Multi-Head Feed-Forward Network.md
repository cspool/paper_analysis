## Flash Multi-Head Feed-Forward Network

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现SRAMFFN——I/O-aware fused kernel用于FlashMHF的multi-head FFN计算。Analogous to FlashAttention的online softmax，SRAMFFN通过blockwise computation（沿着d_ff维度分块）避免在HBM中materialize大的intermediate activation tensor (SiLU(QK^T) ⊙ (QU^T))。两个实现版本：（1）Triton版本（Algorithm 1-3，forward+backward two-pass DK/DU/DV），可在consumer GPU（RTX3090等）上高效运行；（2）ThunderKittens/TK版本（Algorithm 4-5，forward+backward），针对Hopper架构，利用asynchronous data movement（TMA）、warp-group specialization（consumer/producer分工）、stage/ring buffer。Forward pass中producer warpgroup预取K/U/V tiles到SRAM stage buffer，多个consumer warpgroup并行处理不同x-block（sequence tile），各自独立迭代所有inter-tiles，仅在每个sub-network开始时同步router R。
  实验比较：FlashMHF vs SwiGLU FFN vs MH-FFN的（i）peak memory consumption（MB）across sequence lengths 192-16128（Figure 8a, Table 5）；（ii）latency（ms）across sequence lengths（Figure 8b, Table 5）。FlashMHF vs SwiGLU达到3-5x memory reduction和1.00x-1.08x inference speedup。

- 后端平台是什么，配置是什么。
  - NVIDIA H100 GPU (Hopper架构)，单卡benchmark，batch size=8
  - 配置：d_e=384, H=16, E=22, d_h=128，sequence length从192到16128（Table 5）
  - Triton kernel版本可在consumer GPU（RTX3090）上运行但不在Hopper上高效
  - 对比baseline：cuBLAS优化的标准SwiGLU FFN kernel

- 评估性能的软件/脚本是什么。修改了什么。
  - 自编benchmark脚本测量memory和latency，对比FlashMHF kernel vs SwiGLU FFN vs MH-FFN
  - Triton实现（Algorithm 1-3）：编写SRAMFFN-FORWARD-TRITON、SRAMFFN-BACKWARD-TRITON(DQ,DR)、SRAMFFN-BACKWARD-TRITON(DK,DU,DV)三个Triton kernel。Grid并行化over batch/head/sequence blocks。Forward pass在inner loop中blockwise累积输出O_acc（公式15）；backward pass分两轮：第一轮计算dQ和dR，第二轮计算dK/dU/dV
  - ThunderKittens/TK实现（Algorithm 4-5）：Hopper架构上使用warp-group specialization——1个producer warpgroup异步预取Q/R/K/U/V tiles，CON_WARPGRPS≥2个consumer warpgroup并行计算。Producer维护NUM_STAGES ring buffer实现流水线预取；consumer在stage间wait/signal实现producer-consumer同步
  - 核心优化：kernel避免materialize intermediate activation (SiLU(QK^T)⊙(QU^T)) in HBM，改为blockwise SRAM累加；output在单个fused kernel中直接生成

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文声明代码将公开于 https://anonymous.4open.science/r/FlashMHF-9395（当前匿名审阅）。将包含Triton和ThunderKittens两套kernel实现，可直接替换标准FFN module。

  评估原理与流程（以Hopper上FlashMHF单层memory/latency benchmark为例，d_e=384, H=16, E=22, d_h=128, L=4096, bs=8）：

  1. **Input**：Q ∈ R^{B×H×L×d_h}（输入query tensor），K/U/V ∈ R^{H×E×d_e×d_h}（sub-network参数），R ∈ R^{B×H×L×E}（router gating weights）。这些均在GPU HBM中。

  2. **SRAMFFN-FORWARD-TK kernel launch**（Algorithm 4）：
     Grid配置：x = ceil(L / (BLOCK_SEQ · CON_WARPGRPS)), y = H, z = B
     a. **Warmup (producer)**：prefetch所有consumer的Q tiles → prefetch sub-network e=0的router R → prefetch首批NUM_STAGES (K,U,V) inter-tiles到SRAM stage buffer
     b. **Producer loop**：遍历所有inter_tile = NUM_STAGES ... E·(d_e/BLOCK_INTER)-1。每个iteration先wait consumer完成当前stage（释放buffer），再prefetch下一个(K,U,V) tile。若inter_tile是new sub-network的第一tile，额外prefetch router R rows
     c. **Consumer warpgroup c ∈ {0,...,CON_WARPGRPS-1}**（独立并行）：load自己的Q tile（x-block）→ O_acc=0 → 遍历所有inter_tiles → wait producer填充当前stage → M=Q_blk·K_tile^T; N=Q_blk·U_tile^T → S=SiLU(M)⊙N → S=S⊙r（apply router row）→ O_acc+=S·V_tile → signal producer → 最后store O_acc to global memory
     d. **关键**：中间tensor M/N/S仅驻留在SRAM中，不写入HBM。O_acc在SRAM中累积，最终才write to HBM

  3. **Memory profiling**：torch.cuda.max_memory_allocated()测量peak memory。FlashMHF peak memory = O(d_model·L) vs SwiGLU = O((d_ff+d_model)·L) vs MH-FFN = O((d_ff·H+d_model)·L)。L=16128时FlashMHF≈3016MB, SwiGLU≈9966MB, MH-FFN→OOM

  4. **Latency profiling**：CUDA event记录kernel execution wall time。FlashMHF vs SwiGLU latency对比（Table 5）。L=4032时FlashMHF=126.40ms vs SwiGLU=127.44ms (~1.01x)；L=16128时FlashMHF=497.40ms vs SwiGLU=535.20ms (1.08x)。Speedup源于消除HBM↔SRAM的intermediate activation读写

  5. **Output**：Memory (MB)和Latency (ms)随L变化的对比表/图。FlashMHF在所有L下memory均优于SwiGLU（3-5x），latency从略慢（短序列cuBLAS更优）到略快（长序列消除I/O瓶颈）

  6. **Backward pass**（Algorithm 5, SRAMFFN-BACKWARD-TK）：producer prefetch Q/dO/R for sequence tiles → 两个consumer warpgroups各自处理不同的inter-tile (A/B)，并行计算dK/dU/dV → 每个sequence tile t处交换并累加dQ^(A)+dQ^(B)和dR^(A)+dR^(B) → store到global memory
