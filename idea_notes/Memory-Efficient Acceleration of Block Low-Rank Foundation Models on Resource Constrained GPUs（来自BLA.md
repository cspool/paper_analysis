## Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs（来自BLAST repository Lee et al. 2024和Monarch repository Dao et al. 2022）。这些baseline实现虽然在理论上减少了FLOP和模型大小（2×-3×压缩），但在多token推理场景下性能反而退化——Monarch比dense慢1.14-1.68×，BLAST慢2.63-4.31×。退化根因是BLR的block结构产生了dense baseline中不存在的中间张量（Monarch: b×n×r, BLAST: 2个b×n×r中间量），这些中间数据通过global memory传递，加上block维度排列(b₂↔b₁, r'↔b₂)在contiguous维度上的uncoalesced access，将原本compute-bound的dense线性层推入memory-bound区域（roofline α从高于breakpoint降到低于breakpoint）。

  全栈执行例子（以Llama-7B QKVproj层Monarch baseline在A40上，n=1024, i=o=4096, r=1024, b=16为例）：
  - 算法层：Monarch BLR——dense权重W[4096×4096]替换为16×16块的块低秩分解，每块rank r'=64，参数从16.8M降至4.2M（4×压缩），FLOP从34.4G降至17.2G（2×减少）。但在n=1024多token时，b=16块结构产生中间张量Z[16×1024×1024]=64M个元素（BF16: 128MB），而dense线性层中间量为0。
  - 系统框架层：PyTorch eager mode dispatch Monarch forward的多个kernel：X_blocks reshape → bmm(X, V^T) → permutation kernel 1 (r'↔b₂) → permutation kernel 2 (b₂↔b₁) → bmm(Z_perm, U) → final permutation (b₂,n,q)→(n,q,b₂)。每个kernel都有独立launch overhead，中间张量全部通过global memory传递。
  - 编译框架层：torch.compile()尝试fuse操作但受限于BLR block结构——bmm和permutation的复杂index manipulation使compiler难以生成fused kernel，尤其permutation on innermost dimension导致uncoalesced memory access pattern，torch.compile()无法通过layout推导消除。
  - kernel调度层：A40上6MB L2 cache无法容纳128MB中间张量，导致频繁DRAM spill。Permutation kernel的uncoalesced loads使DRAM bandwidth利用率远低于峰值（696 GB/s）。bmm kernel本身是compute-bound但被permutation memory traffic拖累。
  - 硬件架构层：NVIDIA A40 GPU（6MB L2 cache），中间张量远大于L2容量→每步permutation都是DRAM round-trip。Jetson Orin Nano上更严重：DDR bandwidth仅68 GB/s，且L2仅4-6MB。

  Baseline核心缺陷总结：
  1. **中间数据移动**：BLR block结构产生的b×n×r中间张量在global memory中多次往返，dense baseline无此开销。
  2. **排列开销**：Monarch和BLAST的block维度重排需要独立的kernel launch，且排列在contiguous（innermost）维度上造成uncoalesced memory access。
  3. **编译器局限性**：即使torch.compile()也难以自动fuse bmm+permutation和优化BLR-specific memory layout。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文通过**硬件感知的Triton kernel设计**解决BLR多token推理的性能退化。核心策略：partial fusion（非全融合）、operation reordering、tailored memory layouts。不对BLR算法本身做任何修改（保持相同准确率），仅在kernel实现层面消除冗余数据移动。

  **Monarch优化①②③（联合使用）：**
  - ① V重排布（offline）：将V存储从contiguous along b₂ then r'改为r' first then b₂，消除推理时r'↔b₂ permutation kernel → 离线一次性操作消除运行时开销
  - ② 排列融合（kernel级）：将b₂↔b₁ permutation与第一个bmm融合为单个Triton kernel。每个thread block计算输出tile时直接计算目标b₂索引和调整后的r'偏移，用swapped indices写出 → 消除一个完整kernel launch + 一次global memory中间张量往返
  - ③ 避免最终permutation（offline）：当Monarch线性层输出被静态权重消费时，pre-permute该权重rows → 消除推理时的(b₂,n,q)→(n,q,b₂) kernel

  **BLAST优化⑤（推荐，TPC utilization高）：**
  - ⑤ 仅排列融合+Tensor Core：转置S和U为S^T/U^T从左侧乘，在每个kernel内transpose中间输出tiles。n保持contiguous，r/b₁/b₂依次作为三个kernel的batch维度 → 零permutation kernel launch，所有bmm保持tensor core执行（via Triton dot()），permutation开销完全吸收到bmm内部

  **BLAST优化④（备选）：**
  - ④ bmm部分融合：每个thread block内循环b₁维度计算S-weighted累加，消除V→S之间的中间permutation和第一个bmm输出的global memory物化。但第二个bmm用CUDA cores batched outer product → 牺牲tensor core 16×吞吐量优势，仅适用于小rank或极端memory-bound场景

  全栈执行对比baseline（以Llama-7B QKVproj层Monarch ①②③优化在A40上，同n=1024）：
  - 算法层：权重参数化和FLOP与baseline完全相同（Monarch BLR, 4.2M参数, 17.2G FLOP）——算法层面无变化，精度不变。
  - 系统框架层：从6-7个PyTorch kernel launches缩减为2-3个Triton kernel launches（fused perm+bmm → bmm → optional final perm/pre-permuted downstream matmul）。kernel launch overhead减少60-70%。
  - 编译框架层：不使用torch.compile()做高层fusion——直接用Triton编写fused kernel，对BLR-specific dataflow有完全控制。Triton compiler负责lower-level优化（shared memory allocation, warp scheduling, memory coalescing）。
  - kernel调度层：A40上：
    - Kernel 1 (fused perm+bmm): grid=(b₁×ceil(n/t_n)×ceil(r/t_r)), t_n=64, t_r=128, t_p=64。X和V tiles从global memory coalesced加载到shared memory → tensor core dot() → 直接写入permuted output layout。**消除了baseline中的两个独立permutation kernel和中间128MB张量的global memory往返**。
    - Kernel 2 (second bmm with U): 标准batched bmm with tensor cores。
    - (可选Kernel 3已消除): final permutation → 若接residual则仍需此kernel（论文承认无法避免），但大多数QKVproj→attention路径可通过pre-permute attention weight避免。
  - 硬件架构层：NVIDIA A40 GPU。优化后arithmetic intensity从memory-bound区域回升。关键数据流：X_tile [L2→shared] + V_tile [L2→shared] → tensor core MMA → result直接写入permuted layout [shared→L2]——无单独permutation kernel的数据移动。A40 6MB L2中仅需容纳正在处理的tile（64×256 BF16 ≈ 32KB），远小于baseline的128MB中间张量。

  **效果量化：**
  - Monarch ①②③综合：1.46-2.37× layer-wise speedup over Monarch baseline
  - BLAST ⑤：DiT-XL/2 QKVproj up to 7.15× over BLAST baseline on Jetson
  - BLAST ⑤ end-to-end：1.13-1.48× over dense baseline across models（注意：这是相对于dense的加速，不是相对BLR baseline）
  - 关键tradeoff：BLAST ⑤（tensor core）> BLAST ④（CUDA core）in >90% cases，因为tensor core 16× throughput优势远超消除permutation开销的收益
  - BLAST ⑤ > BLAST ④ 的例外：仅在极端memory-bound且b极小的场景（论文中④的GPT2-S on Jetson表现优于某些情况）

  设计思路核心：论文证明BLR压缩的"理论FLOP减少≠实际加速"的gap可以完全由软件/系统层面的kernel优化填补——**不改变压缩算法、不牺牲精度、不依赖新型硬件**，仅通过partial fusion、operation reordering和tailored memory layout三个策略，在现有GPU上实现BLR的理论加速变为实际加速。关键洞察是BLR的额外中间数据移动（而非额外计算）是瓶颈，且PyTorch compiler的通用优化无法处理BLR-specific的permutation-bmm交织pattern，需要手工Triton kernel设计来直接控制tile-level数据流。
