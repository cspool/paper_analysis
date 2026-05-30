## InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  使用 Triton 语言实现了三个核心 GPU kernel：(1) **Pruning Stage Kernel**：单一 Triton kernel 实现了完整的剪枝 stage（SelectRep + chunk score estimation + top-K chunk selection），通过参数化设计可复用于所有 stage（不同 l_c、k、b_q），利用 key sequence dimension 并行度（类似 FlashDecode 的 split-KV）避免全局同步，相比 HiP Attention 的迭代式 top-k 算法消除了内部 global thread synchronization；(2) **Block Sparse Attention Kernel**：基于 FlashAttention 风格实现 prefill 的 BSA kernel，基于 FlashDecoding 风格实现 decoding 的 BSA kernel，利用 PagedAttention 管理 KV cache 内存，仅对 ~2K-4K 选中的 key token 计算完整注意力；(3) **UVM Offloading Kernel**：实现基于 Nvidia UVM 的动态 KV cache 加载/驱逐，在 attention kernel 执行期间通过 PCIe 访问 CPU memory，维护 GPU key bank（两个独立 bank：mask-selection 用和 BSA 用）和 page table（global-to-local index mapping），LRU 驱逐策略，整个 offloaded attention 操作实现为 CUDA graph capturable。实验比较：(1) kernel-level attention latency：vs FA2 (1M window)、InfLLM (12K)、HiP (1K) 的 prefill/decoding 延迟，Triton kernel 内各阶段耗时拆解（Stage 0/1/2/BSA/Extra 占比）；(2) decoding latency with KV offloading：vs FA2 (estimated)、InfLLM 在 256K/512K/1024K 下的带 offloading 解码延迟，含 mask hit ratio 和 SA hit ratio 分析。

- 后端平台是什么，配置是什么。
  (1) NVIDIA RTX 4090 24GB（PCIe 4.0 x8），搭配 AMD Ryzen 7950X + 128GB DDR5 + Ubuntu 22.04；(2) NVIDIA L40S 48GB（AWS g6e.48xlarge）。KV cache offloading 的 PCIe 带宽：PCIe 4.0 x8 约 16 GB/s，访存延迟较 VRAM 高 31.5×。attention latency 测试使用 AWQ Llama 3.1 8B + FP8 KV cache。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 SGLang（https://github.com/sgl-project/sglang）的推理框架进行 kernel-level 和 end-to-end 评测。核心修改：
  1. **Pruning Stage Triton Kernel**：用 Triton 语言重写 HiP Attention 的层次化剪枝算法，核心改进是将 SelectRep 的迭代二分搜索展开为无全局同步的单 kernel 实现——由于每次迭代仅访问 2 个 token（左右分支首 token），消除了 HiP 中 internal top-k 导致的 global thread sync
  2. **BSA Triton Kernel**：实现 FlashAttention-style（prefill: tiling + recompute）+ FlashDecoding-style（decoding: split-KV parallel）+ PagedAttention（block-based KV memory）的 block sparse attention
  3. **UVM Offloading**：基于 CUDA UVM 实现运行时 KV cache 动态换页，通过 page table 管理 GPU↔CPU 的 token 迁移，LRU 驱逐策略
  4. **Mask 缓存与刷新**：实现 per-stage mask cache，refresh interval 设为 16/8/4（fast: 32/16/8, flash: 96/24/8）

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：(1) hip-attention: https://github.com/DeepAuto-AI/hip-attention/；(2) SGLang 集成: https://github.com/DeepAuto-AI/sglang/

  **Kernel 评估原理与全流程（以单层 attention decoding 为例）**：

  ```
  输入：Query Q ∈ R^(H×1×d)（单 token decoding）、Key K ∈ R^(H×T_kv×d)、Value V（同）
       当前 mask 缓存状态（各 stage 的 I^(l,i) 和 counter c^(i)）
  输出：Attention output O ∈ R^(H×1×d)

  Step 1: Mask Cache Check
    For each stage i = 1..3:
      if c^(i) % n_refresh^(i) == 0:
        运行 PruningStage kernel → 生成/更新 I^(l,i)
        记录 GPU cache miss → fetch missing key from CPU UVM
      else:
        复用缓存的 I^(l,i)（mask temporal locality）

  Step 2: PruningStage Kernel（Triton, 单 kernel）
    - 输入: Q block, K indices I^(i-1), stage params (l_c, b_q, k)
    - Kernel 内操作（以 Stage 1 为例，l_c=256, k=32K）:
      a) 将 I^(i-1) 划分为 n_chunk = |I|/256 个 chunk
      b) 对每个 chunk j (parallel over j):
         - SelectRep: log₂(256)=8 次迭代，每次取 2 token 与 Q 做点积
         - 收敛到代表 token r_j，计算 s_j = max_{head,t}(q_t^T * k_{r_j})
      c) Top-K chunk selection: 保留分数最高的 K = 32000/256 = 125 个 chunk
    - 时间复杂度: O(T_q * T_kv) for Stage 0, O(T_q) for Stage 1-2
    - 关键优化: SelectRep 每次迭代仅 2 次点积，无需全局同步，key sequence dim 并行

  Step 3: Block Sparse Attention Kernel（Triton, FlashDecoding-style）
    - 输入: Q (1 token), K/V (仅 I^(3) 中约 2K-4K tokens)
    - 使用 PagedAttention block-based KV 管理
    - Tiling over key sequence dim (类似 FlashDecoding split-KV)
    - Online softmax rescaling (FlashAttention 风格)
    - 复杂度: O(H * d * k^(3))，其中 k^(3) ≈ 2K-4K

  Step 4: KV Cache Management
    - 记录 BSA 过程中的 GPU cache miss
    - LRU eviction: 驱逐 cold token → CPU UVM，加载 miss token → GPU bank
    - Page table 更新: 维护 global_idx → local_bank_idx 的映射
  ```

  **Attention Latency 拆解（Table 3, 1M context decoding, RTX 4090, 3K preset）**：
  - Total (AR, 无 mask cache): 936 µs
  - Stage 0: 28.2% (264 µs) — 最昂贵的初始剪枝
  - Stage 1: 4.0% (37 µs)
  - Stage 2: 5.3% (50 µs)
  - BSA: 2.2% (21 µs) — 仅对 2K-4K token 计算
  - Extra: 60.3% (565 µs) — UVM offloading 的 PCIe 传输开销

  **Mask Cache 加速效果（Table 4, 256K decoding）**：
  - No cache（3 stage 全重算）: 9,803 µs/token
  - All cache（仅 BSA + offload）: 110 µs/token → **89× speedup**

  **性能对比总结（1M context decoding vs baselines）**：
  - vs FA2: 18.95× faster（245 µs vs 4,645 µs）
  - vs InfLLM (12K): 4.98× faster
  - vs HiP (1K): 92% faster（245 µs vs 450 µs）
