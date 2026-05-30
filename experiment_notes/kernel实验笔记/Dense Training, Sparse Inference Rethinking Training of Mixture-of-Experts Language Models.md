## Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  论文使用 SimpleMoE (Tan et al. 2024) 的 ParallelLinear 操作实现 MLP 层的稀疏推理（top-K expert 选择和执行），使用 torch.nn (PyTorch) 实现 Attention 层的密集推理。核心运行时决策：(1) MLP 层使用 sparse inference kernel（ParallelLinear），因为 MLP 层 sparsity 高（激活比例 <30-40%），sparse kernel 在计算量减少上的收益超过动态路由开销；(2) Attention 层使用 dense inference kernel（torch.nn），因为 Attention 层 sparsity 低（激活比例 >60%），sparse kernel 的动态路由开销反而导致更慢。实验比较不同 expert sampling 策略（Threshold / TopK / Threshold-TopK）对 WikiText PPL 和 active param count 的 trade-off。

- 后端平台是什么，配置是什么。
  NVIDIA A100-80GB GPU 和 H100-80GB GPU。训练使用 H100-80GB。

- 评估性能的软件/脚本是什么。修改了什么。
  使用 SimpleMoE (Tan et al. 2024, arXiv:2403.08245) 的 ParallelLinear 操作进行稀疏推理。使用 torch.nn (PyTorch) 进行 dense inference。论文未修改 kernel 实现本身，而是做出 runtime 层面的调度决策——通过观察 Figure 5 发现 Attention 层 sparsity > 60% 时 sparse inference 反而比 dense inference 慢（因动态路由的中间 token 复制和 expert 输出聚合 overhead），因此采用混合策略：MLP 用 sparse，Attention 用 dense。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文代码未开源。SimpleMoE (Tan et al. 2024) 是开源项目 [ScatterMoE](https://github.com/shawntan/scattermoe)，提供了 ParallelLinear 的原生 PyTorch 实现。

  **Kernel 层面的输入-性能-输出流程**（以 DS-MoE-6B MLP 层, D_emb=4096, N_ffd=32, D_ffd=512, TopK=4 为例）：

  1. **Token 路由**：
     ```
     X: [B, 4096]                # input hidden states batch
     S = Softmax(X @ W_r)        # W_r: [4096, 32], Router 计算
     top_idx: [B, 4]             # 每 token 选 top-4 experts
     ```
  
  2. **ParallelLinear Sparse Forward（SimpleMoE）**：
     ```
     # 等价于对每个 selected expert 执行 matmul
     # 使用 torch.index_select 或 bmm 实现 batched expert computation
     for each expert e in top_idx:
       # 收集分配给 expert e 的 tokens
       X_e = gather tokens assigned to expert e    # [T_e, 4096]
       # 执行 FFN: W_up [4096, 512], W_down [512, 4096]
       H_e = GeLU(X_e @ W_up_e)                     # [T_e, 512]
       O_e = H_e @ W_down_e                         # [T_e, 4096]
       # 按 Router score 缩放
       O_e = S[token_to_e, e].unsqueeze(-1) * O_e
     # Scatter 回原 token 顺序
     O = scatter sum of all O_e back to [B, 4096]
     ```
     Note: ParallelLinear 内部通过 `torch.nn.functional.linear` 的 group 机制批量处理 experts，避免逐 expert Python loop 开销。

  3. **性能对比的 Kernel 层面原理**：
     - **Dense FFN**：`O = GeLU(X @ W_up) @ W_down` where W_up: [4096, 16384] (32×512), W_down: [16384, 4096]。计算量：2 × B × 4096 × 16384 = ~134M FLOPs/token。
     - **Sparse FFN (TopK=4)**：4 个 expert, W_up_e: [4096, 512], W_down_e: [512, 4096]。计算量：4 × 2 × B × 4096 × 512 = ~16.8M FLOPs/token。计算量减少 8×。
     - **Sparse Overhead**：token routing（gather/scatter）+ Router 计算。当 sparsity 高（MLP, ~70% tokens 无需计算）时，overhead << 节省的 FLOPs。但当 sparsity 低（Attention, ~30%）时，dynamic routing overhead 可能 > dense 节省。
  
  4. **混合策略的 Kernel 调度依据**（Figure 5, DS-MoE-3B ε=0.48）：
     - MLP 层：平均 active experts ~6-8 / 32，sparsity ~75-80%。使用 ParallelLinear sparse。
     - Attention 层：平均 active experts >5 / 8（60%+），且 attention 的 KV cache 计算已为 computation-heavy。使用 torch.nn dense（torch.nn.functional.scaled_dot_product_attention）。
     - 论文发现：Attention 层在 sparsity >60% 时 sparse inference 因 dynamic routing overhead 而比 dense 更慢。

  5. **Expert Sampling 策略对 Kernel 路由效率的影响**（Figure 4, DS-MoE-3B）：
     - Threshold：per-token 独立决定激活 expert 数（不同 token 可能不同 K），batch inference 时有 padding 浪费。
     - TopK：固定 K 值，统一 batch 内所有 token 的 expert 数，GPU 利用率高。
     - Threshold-TopK：先统计 batch 内平均激活 expert 数，用统一 K 值。兼顾自适应和 batch 效率。
     - 实验结论：Threshold 在 PPL/效率 trade-off 最优，但 TopK 和 Threshold-TopK 更适合实际部署。
