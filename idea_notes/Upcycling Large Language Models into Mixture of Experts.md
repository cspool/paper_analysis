## Upcycling Large Language Models into Mixture of Experts

- baseline方法是什么？
  Baseline 有两种：(1) **续训稠密模型（Continued Dense Training）**——在预训练后的稠密 checkpoint 上继续用新数据训练，保持稠密 MLP 架构不变；(2) **从头训练 MoE（Training from Scratch）**——随机初始化 MoE 架构并从头训练，不使用预训练权重。
  
  全栈执行例子（Baseline 续训稠密模型）：
  - **算法**：稠密 Transformer decoder，每层一个 FFN (MLP)。NVIDIA Nemotron-4 15B，SwiGLU 激活 + RoPE，预训练 8T tokens 后继续训练 1T tokens。
  - **系统框架**：Megatron-LM 分布式训练，data parallelism + tensor parallelism。
  - **编译框架**：论文未明确说明（Megatron-LM 内部使用 NCCL + cuBLAS/cuDNN）。
  - **kernel 调度**：论文未明确说明（标准的 GEMM kernel，无 expert 路由开销）。
  - **硬件架构**：NVIDIA GPU（论文未明确型号），标准 GPU 集群。
  - 执行流程：Token → Attention (QKV GEMM + MHA) → MLP FFN (W1 GEMM → SwiGLU → W2 GEMM) → 下一层。每 token 经过相同的单个 FFN，总激活参数 = 总参数 = 15B。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Upcycling + Virtual Group Init + Weight Scaling + Softmax-then-TopK Routing** 方法，将预训练稠密模型转换为 MoE 模型，在相同或略高计算量下获得更好的模型质量。

  全栈执行例子（论文方法 E8G1T2 Upcycling）：
  - **算法**：从 Nemotron-4 15B 稠密 checkpoint 出发，将每层 MLP 权重复制 8 份初始化 8 个 expert，使用 Virtual Group Init（if fine-grained）+ Weight Scaling。Router 随机初始化，使用 softmax-then-topK。每 token 路由到 top-2 experts，以 Router 概率加权求和。训练时使用 Load Balancing Aux Loss (coeff=1e-2)。
  - **系统框架**：Megatron-LM (expert parallelism) + NeMo。Upcycling 代码在 Megatron-LM moe/upcycling 模块中。
  - **编译框架**：论文未明确说明（Megatron-LM 内部 all-to-all 通信 + expert GEMM）。
  - **kernel 调度**：论文未明确说明（all-to-all token dispatch/combine 通信 + expert FFN GEMM）。
  - **硬件架构**：NVIDIA GPU（论文未明确型号），支持 DP + TP + EP。
  - 执行流程：Token → Attention → Router (softmax W_r @ x → top-2) → token dispatch (all-to-all by expert ID) → Expert_1 FFN (W1 GEMM → SwiGLU → W2 GEMM) / Expert_2 FFN → weighted sum (p1*o1 + p2*o2) → token combine (all-to-all back) → 下一层。总参数 ≈ 15B * (1 + (E-1)*2/3) ≈ 85B，激活参数 ≈ 15B (top-2 下约 30B FLOPs 等价)，总训练 FLOPs 约高于 dense 续训。

  **设计思路核心映射**（Baseline 缺陷 → 论文方法设计选择）：

  - 缺陷(1) **续训稠密模型 plateau 早，无法利用更多参数提升容量** → 方案：**Upcycling into MoE**。通过将单 MLP 扩展为多 expert MoE，增加模型容量（参数量扩大 ~5.7× 但不按比例增加 FLOPs）。Figure 4a 定量验证：Nemotron 2B 续训迅速 plateau，而 upcycling 持续改善，最终 loss 低 1.1%。Table 1 大规模验证：Nemotron-4 15B 续训 1T tokens 后 MMLU 65.3 vs upcycling E8G1T2 MMLU 67.6。

  - 缺陷(2) **Fine-grained MoE upcycling 初始 forward pass 与 dense 模型不等价，loss 爆炸无法收敛** → 方案：**Virtual Group Initialization + Weight Scaling**。原因分析：(a) fine-grained expert 被切分为更小 shard，每个 expert 只贡献 dense 输出的 1/G，router 需要恰好从每个 shard 选一个副本才能重建完整输出；(b) 随机 router 的 softmax 概率 ≈ 1/N (N=E×G)，导致每个 expert 输出被缩放 1/N。Virtual Group Init 通过将 router 权重在组内复制，保证 TopK 恰好覆盖所有 G 个 shard。Weight Scaling 通过 ³√(E×G²/T) 缩放因子补偿输出缩放。Figure 9 定量验证：w/ weight scaling loss 低 1.5%。

  - 缺陷(3) **TopK-then-softmax Router 丢失绝对值信息，导致 sub-optimal 训练** → 方案：**Softmax-then-TopK Router**。TopK-then-softmax (Mixtral 方案) 对 top-K logits 重新做 softmax，丢失非 top-K logits 的绝对值信息（softmax of single element = constant 1, no gradient）。Softmax-then-topK 保留完整的 softmax 分布，Router 能感知所有 expert 的相对重要性。Section 3.4 实验验证 softmax-then-topK 一致优于 topK-then-softmax。

  - 缺陷(4) **Upcycling 使用 fine-tuning 式的小学习率使模型停留在稠密模型的局部最小值，expert 无法分化** → 方案：**学习率重置策略**。将学习率从 pretraining 最低 (2e-5) 重新 warmup 到峰值 (2e-4)，再 cosine decay。这会降低 upcycled MoE 与 base dense model 的权重 cosine similarity（从 ≈1 降至 0.6-0.7, Figure 6），帮助模型逃离 dense 局部最小值，促进 expert 分化。Figure 5 定量验证：constant LR 快速 plateau，重置 LR 最终 loss 更低。

  - 缺陷(5) **小 batch size 下 MoE 每个 expert 只收到 1/E 的 tokens，梯度噪声大，load balancing loss 不稳定** → 方案：**大批量训练（4M tokens batch）**。通过增大 global batch size，每个 expert 收到的有效 token 数增加，梯度更稳定，load balancing loss 更准确。Figure 7 定量验证：batch size 1024 (4M tokens) 收敛快于 512 (2M tokens)，且 GPU FLOP utilization 更高。

  关键定量对比（Baseline vs 论文方法）：
  | 对比维度 | 续训稠密 15B (1T tokens) | Upcycling E8G8T8 (1T tokens) | Upcycling E8G1T2 (1T tokens) |
  |---------|------------------------|-------------------------------|-------------------------------|
  | val loss | 1.377 | 1.320 (-4.1%) | 1.306 (-5.2%) |
  | MMLU (5-shot) | 65.3 | 66.2 (+0.9) | 67.6 (+2.3) |
  | 总参数 | 15B | ~85B (64 experts, 1/8 hidden) | ~85B (8 experts) |
  | 训练 FLOPs | baseline | 等 FLOP (iso-FLOP) | 更高（top-2 增加计算量） |

  论文核心洞察：Upcycling 不仅利用预训练权重的知识加速训练，更重要的是 MoE 架构本身提供了比 dense 模型更高的容量上限。在足够长的 token horizon (1T+) 下，MoE 相对于 dense 的优势持续扩大。同时揭示了 upcycling 与 fine-tuning 本质不同——它需要更大的学习率来摆脱局部最优、需要大批量来稳定 expert 训练。
