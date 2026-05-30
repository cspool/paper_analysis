## FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning

- baseline方法是什么？
  **FlashAttention v1（Dao et al., 2022）**：IO-aware exact attention，通过tiling + online softmax将attention计算融合为单个CUDA kernel，避免$N\times N$ attention矩阵在HBM的materialization。前向在block-wise计算中每次内迭代都做`diag(ℓ)^{-1}` rescale维护正确输出，后向存储row-wise max m和row-wise sum ℓ用于softmax梯度重计算。

  全栈执行例子（GPT-style training, N=8K, d=128, 32 heads, batch=2, A100 80GB）：
  - **模型推理算法层**：Exact softmax(QK^T/√d)V。FlashAttention v1通过tiling + online softmax保证数值等价，Algorithm 1的forward loop结构：外循环over KV blocks, 内循环over Q blocks。
  - **系统框架层**：PyTorch extension (`flash_attn_func`)，替换HuggingFace/Megatron-LM中标准attention调用。框架无需修改。
  - **编译框架层**：论文未明确说明。手写CUDA kernel，非编译器自动生成。
  - **kernel调度层（关键缺陷）**：
    (a) **并行度不足**：仅parallel over batch和head维度（1 thread block per head），对长序列场景（batch小, head少）occupancy低——例如N=8K, head_dim=128, 32 heads, batch=2时仅64 thread blocks，远低于A100的108 SMs，GPU利用率不足。
    (b) **Warp划分低效（split-K）**：在一个thread block内，FlashAttention v1将K和V split到4个warp，Q对所有warp可访问。每个warp计算部分QK^T后需将各自的partial results写入shared memory、同步、然后累加——额外的shared memory reads/writes成为瓶颈。
    (c) **Non-matmul FLOPs开销**：每次内迭代都对已累积的output做`diag(ℓ)^{-1}` rescale（elementwise multiply, non-matmul），且后向需同时存储m和ℓ（2×N个scalars per head）。Non-matmul FLOP虽占比小但耗时（A100上non-matmul吞吐仅19.5 TFLOPs/s vs matmul 312 TFLOPs/s，贵16×），拖累整体吞吐，forward仅达30-50% peak，backward仅25-35% peak。
  - **硬件架构层**：A100 GPU，108 SMs，192KB SRAM/SM，HBM带宽1.5-2.0TB/s。FlashAttention v1 block size受SRAM限制（B_c ≈ M/(4d), B_r = min(B_c, d)），长序列下每个thread block循环次数T_c = ceil(N/B_c)增大。

  Baseline缺陷：
  - (a) **Occupancy不足**：仅batch×heads个thread blocks，长序列场景下远少于SM数量，GPU compute units空闲。
  - (b) **Shared memory通信开销**："split-K" warp划分导致warp间需通过shared memory同步和累加partial results。
  - (c) **Non-matmul FLOPs过多**：每次迭代rescale output，减少可用于matmul的时间比例（matmul:non-matmul吞吐比16:1）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlashAttention-2**：三种改进分别对应三个缺陷：
  (i) **Algorithm tweak（对应缺陷c）**：前向改维护"un-scaled" output，所有KV blocks处理完后一次性`diag(ℓ)^{-1}` rescale，消除每次迭代的output rescale non-matmul操作。后向仅存logsumexp `L = m + log(ℓ)` 替代 (m, ℓ)，减少register使用和non-matmul计算。效果：non-matmul FLOPs减少，更多时间花在matmul上。
  (ii) **Sequence length parallelism（对应缺陷a）**：前向外循环embarrassingly parallel，不同thread block处理不同row block（对应不同sequence position的output chunk），无需同步。后向不同thread block处理不同column block，仅dQ更新需atomicAdd。thread block数从batch×heads增至batch×heads×T_r（或T_c），例如N=8K, B_r=128时T_r=64，总thread blocks = 2×32×64 = 4096 >> 108 SMs，occupancy大幅提升。
  (iii) **Avoid split-K warp partitioning（对应缺陷b）**：前向改为split Q across 4 warps（K/V所有warp共享），每个warp计算其Q slice的完整output，无需warp间通信。后向同样避免split-K。消除shared memory读写瓶颈。

  全栈执行例子（同样GPT-style, N=8K, d=128, 32 heads, batch=2, A100 80GB）：
  - **模型推理算法层**：Exact softmax(QK^T/√d)V定义不变。Algorithm 1 tweak：un-scaled output维护 + final rescale + 仅存logsumexp L，数学上等价（Theorem 1, Dao et al. 2022）。
  - **系统框架层**：同FlashAttention v1，`flash_attn_func(q,k,v,causal=True)`作为drop-in replacement。新增GPT端到端训练验证（HuggingFace GPT-1.3B/2.7B），框架无需修改。
  - **编译框架层**：论文未明确说明。CUTLASS 3.x提供底层building blocks（TileIterator, Collective，etc），手写kernel而非编译器自动生成。论文讨论了未来方向：让compiler自动做这些优化。
  - **kernel调度层（三层改进的集中体现）**：
    - **Thread block调度**：Forward: 4096 thread blocks（batch×heads×T_r = 2×32×64），每个处理1个row block的所有KV blocks（j=1..64）。Backward: 4096 thread blocks（batch×heads×T_c），每个处理1个column block的所有row blocks（i=1..64），dQ通过atomicAdd合并。
    - **Warp内划分**：Forward时4 warps per thread block，Q按row split到4 warps（各32 rows of B_r=128），K_j/V_j在shared memory所有warp可见。每个warp: `S_warp = Q_warp @ K_j^T`（32×128, Tensor Core）→ rowmax/exp/rowsum（CUDA core）→ `O_warp = diag(exp(m_old-m_new))·O_warp + P_warp @ V_j`（Tensor Core）。**零warp间通信**——每个warp独立产出output slice。vs FlashAttention v1 split-K需写partial results到shared memory、同步、累加。
    - **Memory**：SRAM layout: Q_i slice per warp in registers, K_j[128,128] (32KB) + V_j[128,128] (32KB) in shared memory, O_tilde[128,128] (32KB) across registers, ℓ[128]/m[128] in registers。S_warp[32,128]和P_warp[32,128]在registers中。Total ~96KB shared memory + registers，fit 192KB SRAM。
    - **解码阶段**：KV cache loading split到多个thread blocks并行加载以saturate HBM bandwidth。写partial results到HBM后通过separate reduce kernel合并（因为thread blocks间无法直接通信）。
  - **硬件架构层**：相同A100。改进不在硬件，而在kernel调度充分利用已有硬件：通过更多thread blocks填满108 SMs，通过避免split-K减少shared memory traffic，通过减少non-matmul FLOPs增加Tensor Core利用率。

  效果量化（forward pass, A100, head_dim=128, causal mask）：
  - FlashAttention v1: ~30-50% peak FLOPs/s（~94-156 TFLOPs/s）
  - FlashAttention-2: ~50-73% peak FLOPs/s（~156-228 TFLOPs/s），约2× speedup
  - End-to-end GPT-2.7B 8k context: 225 TFLOPs/s (72% model FLOPs utilization) vs FlashAttention v1 175 TFLOPs/s vs 无FlashAttention 80 TFLOPs/s
