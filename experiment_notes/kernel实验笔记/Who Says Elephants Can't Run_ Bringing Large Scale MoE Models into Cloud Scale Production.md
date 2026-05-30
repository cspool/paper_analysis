## Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：(1) **基于 CUB radix sort 的 GPU Token Routing**——将 MoE token-to-expert 路由实现为 GPU-friendly 的 radix sort：对每个 token 的 (expert_scale, expert_idx, row_idx) 三元组按 expert_idx 排序，将同 expert 的 tokens 连续排列，用 CUTLASS Grouped GEMM 并行计算所有 expert 的矩阵乘法。(2) **Fused GEMM+Dequantize CUDA Kernel**——将 weight dequantization（INT4/INT8→FP16）融合进 CUTLASS Grouped GEMM，避免单独 dequantize kernel 的额外内存读写。核心优化：用 FP16 bit-trick 序列（mantissa 直接编码 + 0x6400 构造 + FP16 减法）替代原生 IntToFloat (I2F) 指令。(3) **INT8/INT4 优化的 GPU Dequantize**——INT8：一次加载 4 个 INT8 到 32-bit 寄存器，并行构造 2 个 FP16；INT4：weight layout 重排减少 bit 操作指令量。
  - 实验比较：(a) MoE GEMM 归一化吞吐量：FP16 vs INT8 native I2F vs INT8 optimized I2F vs INT4 optimized I2F，在不同 active experts（1~32）下共 40 tokens（Table 1）；(b) 各种 kernel 组合的端到端 throughput（Table 3）；(c) Batch pruning 的加速效果（1.14×）。

- 后端平台是什么，配置是什么。
  - 单卡 NVIDIA PCIE V100（Volta 架构），CUDA 11.6，nvcc + gcc/g++ 9.3编译
  - 生产部署：单卡 NVIDIA T4（Turing 架构，16GB，INT4 支持，无 NVLink）

- 评估性能的软件/脚本是什么。修改了什么。
  - NVIDIA FasterTransformer 推理框架：扩展以支持 MoE layers（encoder+decoder），使用 CUTLASS Grouped GEMM + CUB radix sort + fused dequantize
  - 修改内容：(a) 实现 MoE token routing kernel——基于 CUB DeviceRadixSort，对每个 token 的 (expert_scale, expert_idx, row_idx) 三元组排序，permute activation matrix 使同 expert tokens 连续；(b) 实现 CUTLASS Grouped GEMM 调用——为每个 expert 构造子矩阵指针（sub-matrix start pointer + weight pointer + bias pointer），单 kernel 并行执行所有 expert matmul；(c) 实现 Fused GEMM+Dequantize——在 CUTLASS GEMM kernel 的 weight load 阶段 fused dequantize，替换原生 I2F 为 FP16 bit-trick 序列；(d) 实现 batch pruning——在 decoder token routing 中跟踪 active_tokens 计数。
  - 开源情况：核心组件 FasterTransformer、CUTLASS、CUB 均为开源；论文自身未提供独立开源仓库。

- 基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 以单次 MoE Decoder Layer 的 expert computation 为例，评估原理和全过程：
    1. **输入**：decoder 生成的 hidden states H，形状 (B, S, 1024)，其中 B=batch_size=32, S=beam_size（beam search 当前步的有效 token 数），FP16。
    2. **Gating/Routing**：Router（FP16 GEMM + softmax）为每个 token 生成 (expert_scale, expert_idx) 元组（top-1 gating）。若某句子已完成（生成了 EOS），Batch Pruning 将 expert_idx 设为 INT_MAX。
    3. **Token Routing Kernel**（CUB radix sort）：
       - 对每个 token 构造三元组 (expert_scale, expert_idx, row_idx)
       - CUB DeviceRadixSort::SortPairs 按 expert_idx 排序 → sorted order + permutation indices
       - Gather kernel：按 permutation indices 从 H 中 gather rows → H_permuted，同 expert tokens 连续排列
       - Offsets kernel：扫描排序后的 expert_idx 数组，计算每个 expert 的 (start_offset, num_tokens) → expert_ptr
       - 仅前 active_tokens 行参与计算
    4. **Fused GEMM+Dequantize Kernel**（CUTLASS Grouped GEMM）：
       - 输入：H_permuted 的子矩阵（各 expert 的 token batch, FP16）+ 各 expert 的 INT4/INT8 weights (1024×4096) + FP16 scales + FP16 biases
       - Weight load 阶段 fused dequantize：4 个 INT8 → 1 个 32-bit reg → 构造 `0x6400 | val`（FP16 1024+val）→ FP16 减 1152 → 乘 scale → FP16 dequantized weight
       - 标准 FP16 GEMM 计算（使用 Tensor Cores 若可用）
       - 所有 experts 的 GEMM 在单个 kernel launch 中并行执行（CUTLASS Grouped GEMM）
    5. **Un-permute + Scale**：按逆排列将输出 rows 恢复原始顺序，每行乘以对应 expert_scale
    6. **输出**：MoE layer output H_out，形状 (B, S, 1024)，FP16
    7. **评估原理**：Throughput 测量翻译 1000 tokenized English sentences（约 40K tokens）的总时间，计算每秒处理 input tokens 数。GEMM 级别的性能评估通过固定 total tokens=40（解码阶段典型值），测量不同 active experts 数下的 GEMM 延迟，归一化为 FP16 baseline 的倍数（Table 1）。
