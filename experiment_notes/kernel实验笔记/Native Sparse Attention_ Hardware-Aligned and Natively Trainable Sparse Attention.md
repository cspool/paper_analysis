## Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  NSA 在 Triton 上实现了硬件对齐的稀疏注意力 kernel，专为 GQA/MQA 架构设计。核心 kernel 设计针对 selection attention（compression 和 sliding window 直接复用 FlashAttention-2 kernel）。关键优化：(1) Group-Centric Data Loading：不同于 FlashAttention 按时间连续 query block 加载到 SRAM，NSA kernel 对每个 query 位置 t，将同一 GQA group 内所有 H 个 query head 的 Q ∈ R^{[H, d_k]} 一同加载到 SRAM，因为它们共享相同的稀疏 KV block 索引 I_t；(2) Shared KV Fetching：在内循环中按 I_t 顺序加载连续的 key/value block K ∈ R^{[B_k, d_k]}, V ∈ R^{[B_k, d_v]} 到 SRAM（B_k 为 kernel block size 且 B_k | l'），消除同一 group 内 head 间的冗余 KV 传输；(3) Outer Loop on Grid：将 query/output 循环放到 Triton 的 grid scheduler 中，因为各 query block 的 inner-loop 长度（正比于选中的 block 数 n）几乎一致，有利于 GPU SM 间负载均衡。kernel 通过消除冗余 KV 传输和均衡 SM 负载实现近最优 arithmetic intensity。

  实验比较：(a) NSA Triton kernel vs FlashAttention-2 Triton kernel（同一后端），测量 forward 和 backward latency（8k/16k/32k/64k context），forward 最高 9.0× speedup @64k，backward 最高 6.0× speedup @64k；(b) 解码阶段 memory access volume 对比（Table 4），NSA 在 64k 时仅需加载 ~5632 等效 token 量（vs Full Attention 65536），预期 11.6× speedup。

- 后端平台是什么，配置是什么。
  NVIDIA A100 GPU（8-GPU 系统）。kernel 配置：GQA group=4，每 group heads H=16，d_q=d_k=192，d_v=128。NSA 超参：compression block size=32，stride=16，selected block size l'=64，selected block count n=16，sliding window=512。kernel block size B_k 整除 l'。

- 评估性能的软件/脚本是什么。修改了什么。
  使用 Triton（Tillet et al., 2019）实现 NSA kernel。对比 baseline 为同样用 Triton 实现的 FlashAttention-2 kernel（保证同后端公平比较）。修改内容：(a) 实现了新的 Triton kernel for grouped-query sparse attention，核心修改 query 加载方式——从 FlashAttention 的「按时间连续 query block 加载」改为「按 GQA group 的 query heads 加载」；(b) 内循环按 I_t 索引顺序加载 KV blocks，每条 KV cache line 加载一次后供 group 内所有 heads 共享；(c) 将 query 遍历外移到 Triton grid scheduler 中。compression attention 和 sliding window attention 直接复用 FA2 kernel 无需修改。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文出自 DeepSeek-AI，kernel 使用 Triton 实现但未在论文中提供显式开源链接。kernel 执行原理如下：

  **Kernel 输入**：Query tensor Q_{t} ∈ R^{[H, d_k]}（同一 GQA group 在位置 t 所有 heads 的 query）；稀疏 KV block 索引 I_t（由算法层的 Top-n selection 预先计算）；全局 K/V cache（HBM 中）。

  **Kernel 执行流程**（参见 Figure 3）：
  ```
  Grid Loop (Triton grid scheduler, 每个 program 处理一个 query 位置 t):
    1. 加载 Q_{t} ∈ R^{[H, d_k]} 到 SRAM （Group-Centric Loading）
    2. 初始化 o ∈ R^{[H, d_v]} = 0, l ∈ R^{H} = 0 （online softmax 状态）
    Inner Loop (遍历 I_t 中每个选中的 KV block):
      3. 从 HBM 加载连续 KV block：K_blk ∈ R^{[B_k, d_k]}, V_blk ∈ R^{[B_k, d_v]} 到 SRAM
         （Shared KV Fetching — 同一 group 所有 H heads 共享此加载）
      4. 计算 S = Q_{t} @ K_blk^T / sqrt(d_k) ∈ R^{[H, B_k]}
      5. Online Softmax 更新：m_new = max(m_old, rowmax(S))
         l_new = exp(m_old - m_new) * l_old + rowsum(exp(S - m_new))
         o_new = exp(m_old - m_new) * o_old + exp(S - m_new) @ V_blk
    End Inner Loop
    6. 写出 o = o_new / l_new ∈ R^{[H, d_v]} 到 HBM
  End Grid Loop
  ```

  **关键性能原理**：(a) 算术强度优化——每个 inner loop iteration 中，HBM 加载量 = B_k × (d_k + d_v) 个元素，计算量 = H × B_k × (2d_k + 3d_v) FLOPs（含 online softmax），H=16 时算术强度约为 16× (2d_k+3d_v) / (d_k+d_v) ≈ 14，超过 A100 的 critical arithmetic intensity，从 memory-bound 变为 compute-bound；(b) 消除冗余 KV 传输——同一 GQA group 的 H 个 query heads 共享相同 KV block，FlashAttention 方式会让每个 head 独立加载，NSA kernel 一次加载供 H 个 heads 使用，减 少 H-1 倍冗余；(c) Grid 负载均衡——所有 query position 的 inner loop 长度 = n × (l'/B_k) 几乎恒定，无 warp divergence。
