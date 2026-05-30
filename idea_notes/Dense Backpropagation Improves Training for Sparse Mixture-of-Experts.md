## Dense Backpropagation Improves Training for Sparse Mixture-of-Experts

- baseline方法是什么？
  Baseline 是 **Standard TopK MoE routing with sparse backward pass**。在标准 TopK MoE 中，Router 通过线性变换 Wx 产生 logits，经 Softmax 得到 expert weights π，TopK 选择 K 个 expert 处理 token，输出 y = Σ_{i∈A} π_i E_i(x)。在反向传播时，由于 TopK 选择是离散不可微操作，使用 straight-through estimator 绕过：∂y/∂π_i = E_i(x) if i∈TopK else 0。未被选中的 experts (N-K 个) 对 Router 的梯度贡献为 0，Router 无法从这些 experts 获得反馈信号。这导致：(1) Router 学习效率低——只能根据已激活 experts 的输出调整路由策略，无法评估未激活 experts 是否更合适；(2) 训练不稳定——Router 更新不完整，在 load imbalance 时容易产生 loss spike；(3) 最大 stable learning rate 受限——因为只有部分 Router 参数得到更新，较大 LR 会导致少数被更新的行产生过大变化。Baseline 全栈执行例子：训练时每个 token x → Router 计算 π = Softmax(Wx) → TopK 选择 expert {i_1,...,i_K} → 仅 K 个 expert 计算前向 E_i(x) → 前向输出 y = Σ_{i∈A} π_i E_i(x) → 反向传播时 ∂y/∂π_i = E_i(x) for i∈A, 0 for i∉A → dL/dW = (dL/dy) · Σ_{i∈A} E_i(x) · (∂π_i/∂W) → Router 仅根据 K 个已激活 expert 的输出更新，N-K 个未激活 expert 对应的 W 行不参与梯度更新。训练框架使用 gpt-neox + Megablocks + liger kernel (Triton)，dropless MoE，global-batch auxiliary loss (0.01)，AdamW optimizer。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **DefaultMoE**：为每个 expert 维护一个 EMA default vector Ê_i = EMA(E_i(x))，在反向传播时将未激活 expert 输出替换为 default vector，使 Router 收到 dense gradient。直接解决 Baseline 的三大缺陷：

  **(1) Router 梯度不完整** → 解决：Default vector 填充缺失梯度。Standard TopK 的 Router 梯度误差为 ε_TopK = (∂L/∂y) Σ_{i∉A} E_i(x) · (∂π_i/∂W)，即丢失了 N-K 个未激活 experts 的梯度项。DefaultMoE 使用 Ê_i ≈ E[E_i(x)] 替代，梯度误差变为 ε_default = (∂L/∂y) Σ_{i∉A} (E_i(x) - E[E_i(x)]) · (∂π_i/∂W)，在期望上为 0（因为 E[E_i(x) - E[E_i(x)]] = 0）。通过实验验证：DefaultMoE 的 Router gradient 在所有 K 值下都与 dense gradient（K=N 全激活）更相似，尤其在 K=1 的前几层（Router entropy 高时）差异最为显著。

  **(2) 训练稳定性差** → 解决：更完整的 Router 更新允许更大的 stable learning rate。Baseline 在 LR=9×10⁻⁴ 时会出现 loss spike（某次迭代 load 极不均衡导致单步大幅更新），DefaultMoE 在相同 LR 下稳定训练，且 DefaultMoE 在所有 LR 下均优于 baseline。这是因为 DefaultMoE 同时更新所有 N 个 expert 对应的 Router 行，避免了个别行更新过大。

  **(3) 收敛速度慢** → 解决：DefaultMoE 在 160B token 训练中比 TopK baseline 减少约 9~15% 的 token 需求（达到相同 target PPL），且最终 PPL 更低。在 12 个下游 benchmark 上，8c1 平均提升 2.1%，8c2 提升 5.0%。该方法对所有 MoE 配置（8c1~32c4）、所有模型规模（557M~7.33B）、所有 sparsity 比例均有效。

  DefaultMoE 全栈执行例子（8c1, 1.96B model, training step）：
  - **算法层**: token x 经 Router 得到 π = Softmax(Wx) → TopK=1 选择 expert 3 → 前向时 E_3(x) 真实计算，E_0,E_1,E_2,E_4,E_5,E_6,E_7 用 EMA buffer Ê_i 替代 → y = Σ π_i [if i=3: E_3(x) else Ê_i] → 反向传 Router: ∂y/∂π_i = E_3(x) for i=3, Ê_i for i≠3 → dL/dW[i,:] += (dL/dy) · Ê_i · x^T for all i ≠ 3 → 所有 8 个 expert 对应的 Router 行均获得梯度更新。EMA 同步更新: Ê_3 = β·Ê_3 + (1-β)·E_3(x)，其他 expert 的 EMA 保持不变。
  - **系统框架层**: gpt-neox + Megablocks 实现 dropless MoE 训练（data-parallel only），64 GPUs AWS，global aux loss 跨节点 reduce。
  - **编译框架层**: 论文未明确说明（gpt-neox eager execution + Triton JIT compiled kernels via liger kernel）。
  - **Kernel调度层**: Megablocks 的 sparse matmul kernel 处理 expert FFN 的批量计算，liger kernel (Triton) 提供优化的 cross-entropy 等 loss kernel。EMA 更新和 default vector 替换均在 PyTorch eager 层面完成，额外开销：O(1) memory per expert × hidden_dim（如 1024 维 × 8 experts × 16 layers ≈ 0.03% 参数增量）。
  - **硬件架构层**: 64 GPUs AWS 集群，无特殊硬件要求。Throughput overhead: 1.96B 模型 1 GPU 上 26,393 (TopK) vs 25,913 (DefaultMoE) tokens/sec = -1.85%；7.33B 模型 per-node 1,393 vs 1,391 tokens/sec = -0.18%（统计噪声级别）。随模型增大，matmul 占比增加，EMA 开销占比趋近于 0。
