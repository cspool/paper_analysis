## Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  论文提出 DS-MoE（Dense Training, Sparse Inference）框架，属于 MoE 模型训练范式的创新——训练阶段所有 expert 全部参与前向/反向计算（dense training），推理阶段仅激活 top-K 个 expert（sparse inference）。实验比较：(1) 与同参数量 Dense 模型比性能和计算效率；(2) 与同性能 Sparse MoE (SMoE, 传统 sparse training) 比参数效率；(3) 在 vLLM 上与 Mistral-7B、DeepSeekMoE-16B、Qwen1.5-MoE-A2.7B 比吞吐量；(4) ablate MI loss weight α 和 expert sampling strategy (Threshold / TopK / Threshold-TopK)。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 80GB × 8（1B-scale 训练，24h）、H100 80GB × 32（3B/6B-scale 训练，64h/124h）。推理评估使用 NVIDIA A100-80GB 和 H100-80GB。

- 模型是什么。数据集和bench分别是什么。
  模型规模：DS-MoE-1B (1067M)、DS-MoE-3B (2846M)、DS-MoE-6B (6343M)。每个 MLP 层有 32 个 expert（D_ffd 分别为 256/384/512），每个 Attention 层有 8-16 个 MoA expert。训练数据：Pile 子集，1B-scale 用 30B tokens，3B/6B-scale 用 100B tokens，tokenizer 使用 CodeGen tokenizer。Benchmarks：PiQA、HellaSwag、WinoGrande、SciQ、Arc-e、Arc-c（zero-shot），WikiText perplexity（language modeling）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文代码未在 GitHub 公开。使用 SimpleMoE (Tan et al. 2024) 的 ParallelLinear 实现稀疏推理，使用 dMoE (Gale et al. 2023, MegaBlocks) 实现 SMoE baseline。vLLM 部署部分使用开源 vLLM (Kwon et al. 2023)。

  **算法 Pipeline 核心流程**：
  
  1. **Dense Training（前向）**：
  ```
  # Router 计算所有 N 个 expert 的分数
  S = Softmax(h(X))           # S ∈ R^N
  # 计算所有 N 个 expert 的输出
  for i in 1..N:
    O_i = e_i(X)              # Expert_i 的前向
  # 加权求和所有 expert（而非仅 top-K）
  O = sum_i(S_i * O_i)        # dense weighted sum
  ```
  
  2. **Dense Training（反向）**：
  ```
  # 传统稀疏训练的 Router 梯度（有 mask M ∈ {0,1}^N）:
  ∇S = [e_1(X), ..., e_N(X)]^T ∇O ⊙ M   # 仅激活 expert 有梯度
  # DS-MoE 的 Dense 梯度：
  ∇S = [e_1(X), ..., e_N(X)]^T ∇O        # 所有 expert 梯度保留
  ∇e_i(X) = S_i · ∇O                      # 每个 expert 获得完整梯度
  ```
  
  3. **MI Loss 负载均衡**：
  ```
  # 最大化 expert 分布熵 H(e) 促进负载均衡
  H(e) = -Σ_{i=1..N} p(e_i) log p(e_i)
  # 最小化条件熵 H(e|X) 促进专家集中
  H(e|X) = -Σ p(e_i|x) log p(e_i|x)
  # 总 MI Loss
  L_MI = -H(e) + (1/|X|) Σ_{x∈X} H(e|x)
  # 总 loss
  L = L_LM + α · L_MI
  ```
  
  4. **Sparse Inference**：
  ```
  # 方法一：固定 TopK
  A = argtopK(S, K)           # 取分数最高的 K 个 expert
  O = Σ_{i∈A} S_i · e_i(X)   # 仅计算选中的 expert
  # 方法二：动态阈值
  p_norm_i = S_i · N           # 归一化概率
  A = {i | p_norm_i > ε}       # 分数超过阈值 ε 的 expert
  ```
  
  5. **Mixture of Attention Head (MoA)**：
  ```
  # 每个 MoA expert i 计算 N_head 个 query vectors
  Q_i = W_q_i @ X              # Q_i ∈ R^{N_head × d_head}
  # 共享的 KV cache
  K, V shared among all experts
  O_ij = Softmax(Q_ij @ K^T) @ V @ W_o_j
  # 最终输出：top-K experts 的加权和
  O = Σ_{k=1..K} S_{A_k} · Σ_{j=1..N_head} O_{A_k,j}
  ```
  
  张量计算细节（以 DS-MoE-3B, D_emb=3072, N_ffd=32, D_ffd=384 为例）：
  ```
  X: [B, 3072]                  # 输入 hidden states
  Router: W_r [3072, 32]        # Router 权重
  S: [B, 32] = Softmax(X @ W_r) # Router scores
  # Dense Training Forward
  E_i(X) = GeLU(X @ W_up_i + b_up_i) @ W_down_i + b_down_i  # Expert FFN
  O = Σ_i S_i · E_i(X)          # [B, 3072]
  # Sparse Inference (TopK=6)
  top_idx = argtopK(S, 6)       # [B, 6]
  O = ParallelLinear(X, top_idx, all_expert_weights)  # SimpleMoE
  # 活跃参数: 6 × (3072×384 + 384×3072) ≈ 14M per layer
  # 活跃hidden比例: 14M/40M ≈ 34%
  ```
  
  关键超参数：α_MoA: 3.5e-4 (1B) / 2e-4 (3B/6B), α_MoE: 6.3e-4 (1B) / 4e-4 (3B) / 2e-4 (6B)。Sparse inference threshold ε=0.48 用于主实验。Optimizer: AdamW, lr=3e-4, cosine schedule, warmup 1B/2B tokens, weight decay 0.01, gradient clip 1.0, batch size 0.5M/2M tokens, seq_len 2048, FSDP + activation checkpointing.
