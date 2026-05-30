## Accelerating MoE Model Inference with Expert Sharding

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 MoEShard 的 **expert computation kernel 融合优化**，包含两层：
  1. **Token Concatenation Fusion**: 将同一 expert 的来自所有 GPU 的 token concatenate 为单个大 tensor 进行矩阵乘法，将 kernel launch 数从 |E|×|G| 降至 |E|（独立于 GPU 数量）。
  2. **MegaBlocks Block-Sparse Matrix Multiplication**: 将所有 expert shard 计算融合为单次大规模稀疏矩阵乘法（variable-sized sparse MM，基于 Gale et al. MegaBlocks [19]），使 kernel launch 数独立于 expert 数量。

  实验比较（Ablation, Section 4.4）:
  - **Varying experts (8→256, batch=250)**: MoEShard w/ MegaBlocks vs MoEShard w/o MegaBlocks。expert<64 时 MegaBlocks kernel 创建开销导致性能略低；expert≥64 时 MegaBlocks 优势递增。
  - **Varying batch size (10→450, 128 experts)**: MegaBlocks 版在所有 batch size 下均优于无 MegaBlocks 版，因 128 experts 时 MegaBlocks 效率更高。

- 后端平台是什么，配置是什么。
  4× NVIDIA A100 GPU（每卡 80GB HBM），NVLink 互联（双向 600 GiB/s），同一节点。CUDA 12.6。CPU: AMD EPYC 7543 32-core。

- 评估性能的软件/脚本是什么。修改了什么。
  **评估工具**: PyTorch + CUDA 12.6，在 BookCorpus 数据集上执行 forward pass，100 iterations averaged per layer。
  
  **修改/优化内容**:
  1. **Per-Expert Token Concatenation**: 在 MoEShard 的 Step 4 expert computation 中，对每个 expert e，将 W[0][e], W[1][e], ..., W[|G|-1][e] concatenate 为单个 tensor，只执行一次矩阵乘法（而非 |G| 次小乘法），计算完成后拆分回 per-GPU result。伪代码：`tokens_concat = cat([W[g][e] for g in G]); result_concat = tokens_concat @ W_i_shard @ W_o_shard; split result_concat back to W[g][e]`。
  2. **MegaBlocks Sparse MM**: 将所有 expert 的 (tokens, W_i_shard, W_o_shard) 打包为 block-sparse 格式，调用 MegaBlocks 的 variable-sized sparse matrix multiplication kernel 一次完成全部计算。这利用了 block-sparse 数据结构将多个独立的小矩阵乘法合并为一个 GPU kernel 调用。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。

  **开源**: https://github.com/sacs-epfl/moe-inference (Python + PyTorch, 13 commits on main)。

  **Kernel 调度评估原理（以 4 GPU, 128 expert, batch=250, seq=120 为例）**:

  1. **Without Fusion (baseline kernel scheduling)**: Step 4 中双重循环 `for g in G: for e in E:` → 每 (g, e) pair 触发一次矩阵乘法 kernel launch → 共 4×128 = 512 kernel launches。每个 kernel 处理小批量 token（部分 expert 可能 0 token），大量 GPU SM 空闲。

  2. **Token Concatenation Fusion**: 外层 `for e in E`，内层将在所有 GPU 上目标为该 expert 的 token 全部 cat → 每个 expert 一次 kernel launch → 共 128 kernel launches（独立于 GPU 数）。每个 kernel 处理的 token 量增大，GPU 利用率提升。

  3. **MegaBlocks Sparse MM Fusion**: 将所有 128 个 expert shard 的 token 组织为 block-sparse 格式（每 expert shard 为一个 sparse block，token 数可变），一次 `cublasGemmEx` 风格调用完成 → 1 kernel launch。MegaBlocks 内部使用 custom CUDA kernel 遍历 non-zero blocks 并分派到 SM 执行。

  4. **性能输出**: 每个 forward pass 中 `torch.cuda.Event` 记录开始/结束时间 → 100 iterations 取 per-layer 平均 TTFT。对比 with/without MegaBlocks 的 TTFT 差异衡量 kernel fusion 收益。
