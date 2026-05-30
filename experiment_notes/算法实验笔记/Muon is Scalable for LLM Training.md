## Muon is Scalable for LLM Training

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：对 Muon 优化器进行三项关键扩展，使其适用于大规模 LLM 训练：(1) 引入 AdamW 风格的 weight decay（λ=0.1），解决原始 Muon 在大规模训练中权重 RMS 持续增长超出 bf16 范围的问题；(2) 提出 Consistent Update RMS 机制，对每个矩阵参数按 √(max(A,B)) 缩放更新量，使不同 shape 的矩阵参数具有一致的更新尺度，避免大矩阵（如 MLP 矩阵 [H, 2.6H] 或 [H, 4H]）更新过小而限制模型容量、小矩阵（如 GQA/MLA 中独立 KV head）更新过大导致训练不稳定；(3) 将 Muon 更新 RMS 匹配到与 AdamW 相同的 ~0.2 范围（scale factor = 0.2），使得 Muon 可以直接复用为 AdamW 调优的 learning rate 和 weight decay。分布式方面提出 Distributed Muon（Algorithm 1），基于 ZeRO-1 在 DP 组上分片 optimizer state，引入 DP Gather（bf16 全矩阵收集）和 Newton-Schulz 迭代在 bf16 精度下计算全矩阵更新，通信量为 Distributed AdamW 的 1~1.25 倍。
  - 实验比较：(a) Scaling Law 实验：在 399M~1.5B 参数 Llama 架构密集模型上，按 compute-optimal 设置对比 Muon vs AdamW，Muon 仅需约 52% 训练 FLOPs 即可匹配 AdamW 性能；(b) 大规模预训练：基于 DeepSeek-V3-Small 架构训练 3B/16B MoE 模型 Moonlight（5.7T tokens），对比 Moonlight-A（同架构+AdamW）和业界模型（Llama3.2-3B, Qwen2.5-3B, DSV2-Lite），Moonlight 在 MMLU 达 70.0 vs DSV2-Lite 58.3；(c) 消融实验：对比 Baseline（仅匹配 AdamW RMS）、Update Norm（直接归一化）和 Adjusted LR（按 shape 缩放）三种更新 RMS 控制策略；(d) SFT 实验：验证 pretrain 和 SFT 阶段优化器互换性，以及在 Qwen2.5-7B 上 SFT 时 Muon vs AdamW；(e) Spectral Analysis：通过 SVD entropy 分析，Muon 训练的权重矩阵具有更高的 SVD entropy，验证其提供更多样化的优化方向。

- 硬件平台是什么，配置是什么。
  - GPU 集群，支持 Megatron-LM 的 TP/PP/EP/DP 并行策略。具体 GPU 型号、数量、集群规模论文未明确说明。训练使用 bf16 混合精度。分布式 Muon 的 Newton-Schulz 迭代在 bf16 下计算，通信量相比 fp32 减半。

- 模型是什么。数据集和bench分别是什么。
  - Scaling Law 模型：Llama 架构密集模型，参数量从 399M 到 1.5B（不含 embedding），hidden size 1536~2560，层数 12~20，训练 tokens 8.92B~38.91B，batch size 96~256（8K context length）。Learning rate 8.3e-4~9.5e-4。
  - Moonlight 模型：基于 DeepSeek-V3-Small 架构的 MoE 模型，2.24B activated / 15.29B total params（含 embedding 为 3B/16B），使用 SwiGLU MLP、GQA、MLA。修改：去除 MTP、修改 auxfree bias 更新规则为 b_i = b_i + u × (sign(e_i) − sign(e).mean())、gate scaling factor=2.446。
  - 预训练数据：Moonshot AI 自研数据集（参见 K. Team 2025），最大 context length 8K。SFT 数据：tulu-3-sft-mixture（Lambert et al. 2024, 4K seq length）。
  - Benchmarks：English (MMLU 5-shot, MMLU-pro 5-shot, BBH 3-shot, TriviaQA 5-shot), Code (HumanEval pass@1, MBPP pass@1), Math (GSM8K 4-shot, MATH, CMATH), Chinese (C-Eval 5-shot, CMMLU 5-shot)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：分布式 Muon 实现将以 PR 形式贡献给 Megatron-LM（https://github.com/NVIDIA/Megatron-LM）；预训练 checkpoint、SFT checkpoint 和中间 checkpoint 均已发布。
  - 算法 Pipeline 核心（Muon + Weight Decay + Consistent Update RMS）：
  ```
  # 对每个矩阵参数 W ∈ R^{A×B}，每步迭代：
  # Nesterov momentum: 先外推再计算正交化
  M_t = mu * M_{t-1} + ∇L(W_{t-1})

  # Newton-Schulz 迭代 (N=5, a=3.4445, b=-4.7750, c=2.0315)
  X_0 = (mu * M_t + ∇L(W_{t-1})) / ||·||_F    # 注意：Nesterov 风格
  for k=1 to 5:
      X_k = a*X_{k-1} + b*(X_{k-1} @ X_{k-1}^T) @ X_{k-1}
            + c*(X_{k-1} @ X_{k-1}^T)^2 @ X_{k-1}
  O_t = X_5  # ≈ (M_t M_t^T)^{-1/2} M_t = U V^T

  # 更新：Matching AdamW RMS + Weight Decay
  W_t = W_{t-1} - lr * (0.2 * O_t * sqrt(max(A,B)) + lambda * W_{t-1})
  ```
