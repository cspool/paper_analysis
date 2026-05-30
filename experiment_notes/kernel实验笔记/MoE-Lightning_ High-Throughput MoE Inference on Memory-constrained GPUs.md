## MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MoE-Lightning 实现了一个基于 Intel MKL 的自定义 **CPU Grouped Query Attention (GQA) kernel** 和 **Paged Weight Transfer** 机制：
    1. **CPU GQA Kernel**：在 decode 阶段，每层 transformer 的 attention softmax + weighted sum 在 CPU 上执行（而非 GPU），避免将 KV cache 从 CPU 传输到 GPU 占用 PCIe 带宽。Kernel 利用 Intel MKL batch GEMM 进行 query·key 和 attention-weights·value 矩阵乘法。GQA 场景下多个 query head 共享 KV head，通过 MKL 批量矩阵运算高效实现。
    2. **Paged Weight Transfer**：每层 MoE FFN expert weights 存入 CPU memory，分 n 页（n = 微批次数）。传输采用两阶段流水线：CPU memory → pinned memory (memcpy) → GPU memory (cudaMemcpyAsync)，两阶段在连续 pages 间重叠。GPU kernel 通过 page table 查找对应 weight pages 的 GPU 地址。
  - 实验比较：(1) 单层 CPU attention kernel latency vs KV cache transfer latency (CPU→GPU) vs MoE FFN GPU kernel latency 在不同微批次大小 μ 和 context length 下的对比 (Fig. 9)；(2) CPU attention kernel 在所有测试配置下比 KV cache transfer 快 3-4×，接近 CPU memory bandwidth 与 PCIe bandwidth 的比值；(3) MoE FFN kernel latency 在不同 μ 下基本不变（memory-bound）。

- 后端平台是什么，配置是什么。
  - CPU: Intel Xeon @ 2.20GHz (S2) / 2.30GHz (S1/S6-S9), 24-core/32-core。CPU attention kernel 利用 Intel MKL (oneAPI Math Kernel Library) 加速矩阵运算。
  - GPU: NVIDIA T4 (16GB HBM, ~65T FLOPS FP16) / L4 (24GB) / Multi-T4，通过 PCIe Gen3/4 连接。
  - Memory hierarchy: GPU HBM (16-64GB) → PCIe (~16-32 GB/s) → CPU pinned memory → CPU DRAM (192-416GB)。
  - Ablation (Fig. 9) 使用 L4 GPU + 24-core Intel Xeon @ 2.20GHz 测量单层 kernel latency。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估基于 MoE-Lightning 自建系统（Python + C++），CPU GQA kernel 和 paged weight transfer 为自研 C++ 模块，编译为 PyTorch extension。
  - 修改/创新：
    - 替代 GPU attention + KV cache transfer 路径：传统方法（如 FlexGen $S_4$）将 KV cache 从 CPU cudaMemcpy 到 GPU 再做 Flash Attention → MoE-Lightning 改为直接在 CPU 上执行 MKL GQA kernel，仅将 hidden states (shape: [μ, d]，远小于 KV cache [μ, s, d]) 通过 H2D 传输回 GPU。
    - Paged weight transfer 替代整层一次性传输：传统方法整层所有 experts weights 一次性 H2D → MoE-Lightning 分 n 页交错传输，避免阻塞后续微批次的 hidden states H2D。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源**：https://github.com/caoshiyi/artifacts/tree/asplos25（ASPLOS 2025 artifact）
  - **CPU GQA Kernel 评估（Fig. 9 原理）**：
    1. **Kernel 输入**：
       - Query vectors: shape [μ, n_q, d]，μ 为微批次大小，n_q=32 (Mixtral 8x7B)，d=128 (per-head dim)。
       - KV cache blocks: 存储在 CPU pinned memory 中的历史 key/value vectors，shape [s, n_kv, d]，s 为 context length，n_kv=8 (GQA 分组)。
       - GQA group size = n_q / n_kv = 4。
    2. **Kernel 计算流程**：
       - Step 1: 对每个 query head，使用 Intel MKL SGEMM 计算 query·key^T → attention scores [μ, n_q, s]。
       - Step 2: Softmax over sequence dimension（向量化实现）。
       - Step 3: 使用 MKL SGEMM 计算 attention weights × value → attended outputs [μ, n_q, d]。
       - GQA reuse: n_q=32 query heads 共享 n_kv=8 KV heads，实际计算量 = n_kv × attention per KV group。
    3. **Kernel 输出**：attended hidden states [μ, n_q×d] = [μ, 4096]，作为 PostAttn (O projection) 的输入通过 H2D 传输回 GPU。
    4. **性能指标**：kernel latency (ms/层)，在 μ ∈ {32, 64, 128, 256}、context length ∈ {128, 256, 512, 1024, 2048} 下测量。
    5. **关键结论**：CPU attention 比 KV cache H2D 快 3-4×，接近 CPU BW (~200 GB/s) / PCIe BW (~50 GB/s) ≈ 4× 的理论比值。随着 μ 和 context length 增大，CPU attention 可能成为瓶颈。
  - **Paged Weight Transfer 原理**：
    1. **输入**：存储在 CPU DRAM 中的 expert FFN weights [n_e=8, h1×h2×2 = 4096×14336×2]。
    2. **分页**：weights 被分为 n_pages = n_ub (微批次数) 页，每页大小 = total_expert_weights / n_pages。
    3. **两阶段流水线传输**：Page k 在 CPU→pinned (memcpy) 传输时，Page k-1 同时在 Pinned→GPU (cudaMemcpyAsync)。GPU PostAttn kernel 通过 page table 查表访问对应 pages。
    4. **性能输出**：paged 传输消除了整层一次性传输导致的 hidden states H2D 阻塞，使 GPU 利用率更高。
