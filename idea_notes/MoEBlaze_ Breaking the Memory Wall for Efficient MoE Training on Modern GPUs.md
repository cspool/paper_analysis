## MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

- baseline方法是什么？
  Baseline 是 MegaBlocks（Gale et al., 2023），将 MoE 计算重新表述为 block-sparse 操作以避免 padding 和 token dropping。MegaBlocks 的 token dispatch 依赖 **基于排序的方法**：将所有 token 的 top-k 选择展平为 (expert_id, token_id) 元组 → 按 expert_id 做 multi-pass radix sort 分组 → index recovery 重建 token 顺序并计算 per-expert range。此方法存在两个核心瓶颈：(1) **激活内存膨胀**——分配 per-expert materialized token buffer（大小 L×K×d，在 DeepSeek 规模下约 94GB），以及 compact 后的 FFN 中间激活（约 98GB），总 activation memory 可达数百 GB；(2) **dispatch 开销**——sorting 需要多次 global memory passes（radix sort 的 pass 数与 key width 成正比），强制 multi-kernel dispatch pipeline（multi-pass sort + segmented scan + index recovery），kernel launch latency 高且 GPU 资源利用率低。

  全栈执行例子（MegaBlocks + DeepSeek 配置，单 H100）：
  - **算法Pipeline 层**：Gate = softmax(W_g · x) → TopK → 为每个 expert e 分配容量 C ≈ γ·LK/E 的固定 buffer → 按 gate score 排序 token 并打包入 buffer（超出容量的 token drop 或路由到 residual path）。
  - **系统框架层**：PyTorch + CUDA custom kernels。Token dispatch 通过 [CUB radix sort](https://nvlabs.github.io/cub/) 实现——flatten top-k 结果 → sort by expert_id → compute offsets → bucketize tokens。
  - **编译框架层**：论文未明确说明。MegaBlocks 使用自定义 CUDA kernel（block-sparse matrix multiplication），无编译器框架修改。
  - **kernel调度层**：dispatch kernel 流程：(1) radix sort kernel（≈4 passes for 16-bit expert_id key）→ 每次 pass 需 read + write L×K 个 (expert_id, token_id) pair，即 O(LK) global memory traffic； (2) segmented scan kernel 计算 per-expert offsets；(3) scatter kernel 将 token 写入 per-expert buffer。FFN kernel：(1) 加载 materialized routed token buffer → W1 GEMM；(2) 存储中间激活 (L×h) → activation function → W2 GEMM → 输出 buffer。激活内存 = L×K×d（routed token buffer）+ L×h（FFN 中间激活），对 SwiGLU 额外 ×2。
  - **硬件架构层**：NVIDIA H100 GPU。Memory bandwidth bound 是主要瓶颈——activation function 为 point-wise 操作，在 tall-and-skinny 矩阵（L≫d）下 memory bandwidth bound；radix sort 的多次 global memory pass 受限于 HBM bandwidth（3.35 TB/s）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MoEBlaze 通过三个协同设计打破 memory wall：(1) **索引替换 materialized buffer**——用四组轻量级索引数据结构（expert_token_indices、expert_token_offsets、token_expert_indices、token_index_map，各 L×K 个 int32，总计 4×L×K×4 bytes vs MegaBlocks 的 L×K×d×2 bytes）替代 per-expert materialized token buffer；(2) **atomic-free 并行 dispatch 构建**——3-step kernel（dense map 构建 → warp-level reduction 计计数 → tile-level scan + location map 写入）替代 multi-pass radix sort，利用 shared memory prefix sum 和 warp-level reduction，避免 atomics 和多次 global memory pass；(3) **kernel fusion + activation checkpoint**——fused SwiGLU kernel 将 W1/W2 两个 GEMM + SiLU + element-wise multiply 融合为单 kernel，forward 中仅保存 a, b, y_swi（SiLU(a) 不保存），backward 时 recompute SiLU。

  全栈执行例子（MoEBlaze 对比 MegaBlocks）：
  - **算法Pipeline 层**：
    - **MegaBlocks**：Gate → sort-based dispatch → 写入 materialized routed buffer → W1 → 存储中间激活 → act → W2 → 输出 → 反向需要 (L,d)→(L×k,d) 展开。
    - **MoEBlaze**：Gate → 构建 4 组索引数据结构（仅 int32 IDs） → on-the-fly gather x[token_ids] → fused W1+W2+SwiGLU（单 kernel） → 仅保存 a, b, y_swi → W3 → on-the-fly reduction via token_index_map → 反向通过 scatter（逆向索引）直接映射梯度，无需展开。
    - 关键差异：MoEBlaze 不分配 L×K×d 的 routed token buffer（节省 ~94GB for DeepSeek scale），仅分配 4×L×K 个 int32 索引（~16MB for L=2M, K=4）。FFN 计算中 recompute SiLU 进一步节省 L×h 的激活存储。
  - **系统框架层**：PyTorch 2.0.1 + CUDA 12.1 自定义 kernel。替换 MegaBlocks 的 sort-based dispatch pipeline 和 block-sparse FFN 为 MoEBlaze 的 index-based dispatch + fused FFN kernel。触发时机相同（每个 MoE layer 的 forward/backward），但 memory footprint 和 kernel launch chain 大幅缩短。
  - **编译框架层**：论文未明确说明。所有优化在 CUDA kernel 层面，无编译器框架修改。
  - **kernel调度层**：
    - **MegaBlocks dispatch**：radix sort（≈4 global memory passes）× 每次 O(LK) + segmented scan + scatter → kernel launch chain length ≈ 6-8，总 global memory traffic ≈ 8×L×K×(8+4) bytes（sort key + value）。
    - **MoEBlaze dispatch**：3-step kernel chain：(1) dense map fill（1 pass, L×E writes）；(2) warp-level count（1 pass, E reductions）；(3) tile-level scan + location write（1 pass, L×K writes）。Kernel launch chain length ≈ 3，总 global memory traffic ≈ L×E + L×K（int32 writes），无 sort 的多次 full pass。
    - **MegaBlocks FFN**：W1 GEMM → store a, b → SiLU compute（load a, compute, store SiLU(a)）→ element-wise multiply（load SiLU(a), b, compute, store y_swi）→ W3 GEMM → backward 需要 a, b, σ(a), SiLU(a), y_swi 全部在 HBM 中。
    - **MoEBlaze FFN**：fused kernel 内：load x once → stream through W1 GEMM, W2 GEMM simultaneously → compute SiLU(a) in register → y_swi = SiLU(a)⊙b in register → store a, b, y_swi only → W3 GEMM。Backward：recompute SiLU(a) from a（element-wise O(L×h)，memory bandwidth bound → recompute cost ≈ 直接读 HBM 的 cost）。SwiGLU 下节省 5 个中间 tensor 的 HBM write/read（a, b, σ(a), SiLU(a), y_swi_product），内存节省最显著（最高 4× reduction，conf3 从 40GB→10GB）。
  - **硬件架构层**：NVIDIA H100 GPU。利用 WGMMA（warp-group matrix multiplication）和 TMA（Tensor Memory Accelerator）加速 fused GEMM。Fused kernel 将 computation 从 memory-bound domain 推向 compute-bound domain——原本 activation function 受限于 HBM bandwidth（3.35 TB/s），融合后 point-wise ops 在 register/shared memory 完成，仅需读 x 一次（vs 两次）且无中间 global write。Speedup 在 SwiGLU 下更显著（2×–6.2× vs 1.4×–3.7× for SiLU），因为 SwiGLU 的中间激活更复杂、记忆节省更大。
