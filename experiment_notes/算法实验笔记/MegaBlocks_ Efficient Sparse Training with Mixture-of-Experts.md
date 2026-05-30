## MegaBlocks: Efficient Sparse Training with Mixture-of-Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 **dropless-MoE (dMoE)**，将 MoE 层的 expert 计算从 batched matrix multiplication 重新表述为 block-sparse matrix multiplication。核心思想：标准 MoE 使用 batched GEMM 要求所有 expert 分配相同数量 tokens（导致 token dropping 或 padding 浪费）；MegaBlocks 将 expert 计算视为 block diagonal matrix multiplication，允许可变大小 block（即负载不均衡的 token 分配），通过将大 block 分解为多个 128×128 固定大小小 block 的 block-sparse 矩阵乘法来实现。
  - 实验比较：
    - dMoE (MegaBlocks) vs dMoE (Tutel, dynamic capacity factor) vs Dense Transformer (Megatron-LM)：在 The Pile 上训练 decoder-only Transformer 语言模型 (MoE-XS/Small/Medium)，比较端到端训练时间和 validation loss（Figure 7）。MegaBlocks 实现 1.38×、2.0×、4.35× 加速。
    - dMoE (MegaBlocks) vs token-dropping MoE (Tutel, capacity_factor=1/1.5/2)：在相同模型配置下，以 loss-equivalent Pareto 前沿比较训练时间（Figure 8）。MegaBlocks 减少训练时间 1.18×–1.38×。
    - Block-sparse matrix multiplication kernel micro-benchmarks vs cuBLAS batched GEMM（Figure 9）：18 种问题配置（3 模型 × 6 operations），平均达到 cuBLAS 98.6% 吞吐量。
    - MoE layer forward pass vs Megatron-LM SwitchMLP（sequential expert, Appendix A）：num_experts=128 时 20× 加速。
    - Block-sparse kernels vs Triton Blocksparse（Appendix C）：平均 9× 吞吐量优势。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A100 SXM4 80GB。单卡实验（§3 motivation）用 1×A100。端到端训练实验（§6.1）用 8×A100 SXM4 80GB（8-way expert model parallelism for MoE layers + data parallelism for other layers）。
  - 软件：CUDA 11.5、CUTLASS 2.5、PyTorch + Megatron-LM。Mixed-precision training (FP16 + FP32 accumulation)。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Decoder-only Transformer 语言模型，配置如表 1 & 2：
    - Transformer-XS (46M), Transformer-Small (125M), Transformer-Medium (356M), Transformer-Large (760M), Transformer-XL (1316M)
    - MoE-XS (839M), MoE-Small (3,693M), MoE-Medium (13,041M)：将 Transformer 的 FFN 层替换为 64-expert MoE 层，top-1 routing，每个 expert 为 2 层 MLP（ffn_hidden_size=4×hidden_size）
    - 所有模型 vocabulary_size=51200, sequence_length=1024, attention_head_size=64
  - 数据集：The Pile（Gao et al. 2020），使用 GPT2 tokenization（Radford et al. 2019）。训练 10B tokens，batch size 512 sequences。训练/验证集划分按 The Pile 标准划分。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/stanford-futuredata/megablocks，Apache-2.0 许可证
  - 算法pipeline 伪代码（dMoE forward，对应 Figure 4）：

```
# x.shape: (num_tokens, hidden_size)
def dmoe_forward(self, x):
    # (1) Router: Assign tokens to experts
    # indices.shape: (num_tokens), weights.shape: (num_tokens)
    indices, weights = router(x)  # top-k greedy selection

    # (2) Create block-sparse matrix topology
    # Constructs the variable-size block diagonal matrix
    # as many 128x128 fixed blocks (Figure 3C)
    topology = make_topology(indices)

    # (3) Permute tokens to group by expert assignment
    # Pad each expert batch to multiple of block size (128)
    x = padded_gather(x, indices)

    # (4) Compute expert layers via block-sparse ops
    # self.w1.shape: (hidden_size, ffn_hidden_size * num_experts)
    # self.w2.shape: (ffn_hidden_size * num_experts, hidden_size)
    # SDD: Sparse = Dense x Dense (Figure 3C forward)
    x = sdd(x, self.w1, topology)       # output: block-sparse
    # DSD: Dense = Sparse x Dense (second layer)
    x = dsd(x, self.w2)                 # output: dense

    # (5) Un-permute tokens and scale by router probabilities
    x = padded_scatter(x, indices)
    return x * weights
```

  - 关键：SDD 操作中，sparse output matrix 对应图 3C 的 block diagonal structure。每个 expert 的 token batch 被分解为 ceil(num_tokens_expert/128)×128 行的多个 block。w1 和 w2 的列维度按 expert 拼接（concatenated），使得单次 block-sparse 矩阵乘法等价于并行计算所有 expert。
  - 向后传播：对 MLP expert (2-layer FFN)，需要 SDD^T (第二层 weight grad)、DS^T D (第二层 data grad)、DSD^T (第一层 data grad)、DD^T S (第一层 weight grad) 四种操作。
