## Flex Attention: A Programming Model for Generating Optimized Attention Kernels

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  FlexAttention 在 kernel 调度层面的核心实现包括：(i) BlockMask 数据结构——将 score 矩阵按 block（默认 128）分割，通过 kv_num_blocks [B,H,Num_Row] 和 kv_indices [B,H,Num_Row,Num_Col] 两个紧凑张量编码 block 级别稀疏性，内存开销 O(⌈Q_LEN/BS⌉ × ⌈KV_LEN/BS⌉) 远小于完整 score 矩阵 O(M×N)；(ii) Full/Partial Block 优化——区分 Full Blocks（无 score 被 mask，可跳过 mask_mod）和 Partial Blocks（部分被 mask，需逐元素执行 mask_mod），对 causal mask 等模式获得约 15% 性能提升；(iii) BlockMask 引导的间接内存访问——通过 kv_indices 映射跳过完全 masked 的 block，无需修改 kernel 即可支持 sliding window、local-global attention 等多种稀疏模式；(iv) 数据预取 Pipeline——SM 沿 Q_LEN tile 并行，每 SM 沿 KV_LEN 迭代，当前 score block 计算时预取下一 KV tile（HBM→SRAM），BlockMask 消除了条件分支检查从而允许高效 pipeline；(v) Paged Attention 支持——通过 BlockMask 转换合并 page table 的间接内存访问与 BlockMask 的稀疏跳过访问，无需重写 kernel，overhead <1%。

  实验比较：(1) 7 种 attention 变体的 kernel 速度 vs FAv2/FAv3/FAKV/SDPA，训练和推理；(2) block sparsity 的加速效果（proportional to sparsity）；(3) Paged Attention overhead vs FlashAttn-v2；(4) 端到端 torchtune 训练（Llama3-8B on Alpaca）和 gpt-fast 推理（Llama3.1-8B/70B）。结果：training forward 0.68×-1.43× vs FAv2, backward 0.86×-1.05×；inference 0.93×-1.45× vs FAKV；GQA+alibi 场景 5.37× vs FAKV；Paged Attention overhead <1%。

- 后端平台是什么，配置是什么。
  Nvidia H100 GPU（功率限制 650W，内存带宽限制 2.4TB/s），Nvidia A100 GPU（功率限制 330W），Nvidia A6000 GPU。KV size 固定 256 MiB，head dimension 64，数据类型 bfloat16。BlockMask 默认 block size=128。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 PyTorch 核心 + Triton（https://github.com/triton-lang/triton）的 attention kernel。评估脚本使用 PyTorch benchmark 工具测量 kernel wall-clock time。
  
  修改/新增内容：
  - BlockMask 数据结构：两个张量 kv_num_blocks（每行非零 block 数）和 kv_indices（非零 block 列索引），由 create_block_mask() 通过 torch.vmap 自动生成
  - Full/Partial Block 分类逻辑：编译时判定 block 是否全部可见（full）、部分可见（partial）或全部 masked（oblivious），运行时对 full block 跳过 mask_mod 仅执行 score_mod
  - 间接内存访问策略：GPU block 的 workload 根据 kv_num_blocks 调整，通过 kv_indices 映射到下一个 block（可不连续，支持非连续 token 访问）
  - 数据预取 pipeline：在 Triton kernel 模板中插入预取逻辑，当前 block 计算时预取下一 KV tile
  - Paged Attention 集成：通过将 page table 的物理-逻辑映射融入 BlockMask 的 kv_indices，实现 fused indirect memory access

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  FlexAttention 已集成至 PyTorch 2.5+（https://github.com/pytorch/pytorch），attention-gym（https://github.com/pytorch-labs/attention-gym）提供工具。

  Kernel 调度评估原理与流程（以 H100 causal mask, training forward, QKV_LEN=16k, head_dim=64, bf16 为例）：

  1. **BlockMask 生成（编译时）**：
     - 输入：causal_mask_mod(b, h, q_idx, kv_idx) → return q_idx >= kv_idx
     - create_block_mask 用 torch.vmap 对 Q_LEN=16384, KV_LEN=16384, block_size=128 批量评估 mask_mod
     - 输出：score 矩阵被切分为 (16384/128)×(16384/128) = 128×128 个 block
     - Full blocks：对角线上方，全部 kv_idx ≤ q_idx 的 block（约 50% 总 block 数，对这些 block 运行时跳过 mask_mod）
     - Partial blocks：对角线上的 block（少量，需逐元素 mask_mod）
     - Oblivious blocks：对角线下方，全部 kv_idx > q_idx，完全跳过计算
     - kv_num_blocks: [B,H,128]，每行的非 oblivious block 数从 1 到 128
     - kv_indices: [B,H,128,128]，记录每行非 oblivious block 的列索引

  2. **GPU 调度**：
     - Q_LEN=16384, 每个 SM 处理一个 Q tile（Q_BLOCK_SIZE tokens，由模板决定）
     - 每个 SM 沿 KV_LEN 维度迭代处理一"行"block
     - SM 读取 kv_num_blocks[row] 确定该行需处理的 block 数
     - SM 通过 kv_indices[row, :] 获取非 oblivious block 的索引
     - 当前 block 计算时，通过预取管线加载下一 KV tile（HBM→SRAM）

  3. **Per-Block 计算**：
     - Full block：加载 Q tile + K tile → QK^T GEMM → score_mod（如 Alibi bias） → softmax（online rescaling） → PV GEMM。**不执行 mask_mod**。
     - Partial block：加载 Q tile + K tile → QK^T GEMM → mask_mod 逐元素 mask（设为 -inf）→ score_mod → softmax → PV GEMM。
     - Oblivious block：完全跳过（通过 kv_indices 自动排除）。

  4. **数据预取 Pipeline**：
     - 时间线：while iterating blocks: prefetch(KV_tile[i+1]) || compute_score(KV_tile[i]) → score_mod → online_softmax_update
     - 因为 BlockMask 消除了条件分支（不需要逐元素检查是否 masked），pipeline 可以高效流水线化

  5. **性能测量**：
     - CUDA event timing 测量 kernel wall-clock time
     - 吞吐量 = (effective FLOPs) / time，其中 effective FLOPs 仅计算非 masked block 的 FLOPs
     - Speedup = FlexAttention time / baseline（FAv2/FAKV）time
     - 对于 causal mask（50% sparsity）：forward 1.00×-1.22× vs FAv2
     - 对于 sliding window（更高 sparsity）：speedup 更显著

  6. **Paged Attention overhead 测量**：
     - 对比 FlexAttention with paged attention vs FlexAttention without paged attention vs FlashAttn-v2 without paged attention
     - Batch size=32, head_dim=64, num_heads=16, 变化 seq_len 和 page size
     - 通过将 page table 映射融入 kv_indices 实现 fused indirect memory access
     - 结果：平均 overhead <1%，远低于 vLLM 报告的 20-26% attention kernel overhead
