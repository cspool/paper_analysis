## Cost-Optimal Grouped-Query Attention for Long-Context LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  提出一种基于 scaling law 的 GQA 配置搜索方法，通过三步搜索过程找到给定目标 loss 和上下文长度下的 cost-optimal GQA 配置（nh, nkv, N）。核心设计：(1) **解耦 head 数量与 hidden size**：传统 GQA Transformer 强制 nh × dh = d（dh 固定为 64），本文解除此约束，使 nh 成为独立超参数自由控制 attention 计算 FLOPs（time-variant cost）；(2) **联合优化模型大小与 GQA 配置**：同时调整模型大小 N（time-invariant cost）、query head 数 nh 和 KV head 数 nkv（time-variant cost），实现推理资源在 attention 和非 attention 组件之间的最优分配；(3) **三步搜索过程（Step 1→3）**：Step 1 定义候选 GQA 配置集 H_cand = {nh=1,2,4,...,max(d)/dh} × {nkv=1,2,4,...,max(d)/dh}（满足 nkv ≤ nh），共 21 个候选；Step 2 对每个 H 训练系列不同大小模型（3M→1.2B），用 power-plus-constant 函数 L(N;H) = (a/N)^b + E 拟合 scaling curves（R² > 0.999）；Step 3 对目标 loss L* 和上下文长度 T，求解 N*(H) = a/(L* - E)^{1/b} 并计算硬件感知成本 Z = λM_infer^α + (1-λ)C_infer^β（λ=0.9, α=1/2, β=1/3），选择 Z 最小的 (N*, H*)。实验比较：(1) Loss vs. inference costs（M_infer, C_infer, Z）对不同 GQA 配置的 tradeoff 曲线（T=8K/16K/32K/64K/128K/512K）；(2) cost-optimal GQA vs Llama-3 GQA（nh=d/dh, nkv=8）在 T=128K 下的 training/inference throughput 和下游性能（common-sense reasoning + NIAH）；(3) 对齐 training FLOPs 的 comparison（用更少 head 的配置获得更多训练数据）；(4) nh 和 nkv 对 loss 的 power-plus-constant scaling law 验证（Section 5.4 + Appendix I）。

- 硬件平台是什么，配置是什么。
  NVIDIA A800 GPU，8 GPU 集群，BF16 训练，FP16 评估。单张 A800 测试 downstream throughput（batch_size=1, T=128K）。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3 架构（RoPE θ=500,000 + SwiGLU FFN + RMSNorm + pre-norm，无 bias/dropout），使用 GPT-2 tokenizer（V=50,304），dh=64，dff≈8d/3。训练 3M→1.2B 参数模型，具体配置见 Table 9（L/d 从 4/256 到 36/1536）。数据集：SlimPajama（627B tokens，RedPajama 的去重版本），每 batch 512K tokens，训练数据与参数量之比为 20:1（Chinchilla law）。Benchmark：zero-shot common-sense reasoning（8 任务：ARC-Challenge, ARC-Easy, BoolQ, HellaSwag, Lambada, PIQA, SocialIQA, Winograd，使用 LM-Evaluation-Harness）、Needle-in-a-Haystack（RULER benchmark，1K-128K context）。Training 含两阶段——第一阶段 T=4K 20B tokens + 第二阶段 T=128K 1B tokens。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/THUNLP/cost-optimal-gqa

  **算法 Pipeline 详解（三步搜索 + 推理）**：

  **前提：解耦 nh 与 d**
  传统 GQA: nh × dh = d（nh = d/dh，不可调），本文解除此约束：
  ```
  # 传统 GQA（Llama-3）: nh 由 d 决定
  d = 1536, dh = 64 → nh = d/dh = 24（四舍五入到最近 power-of-2 → 32）

  # 本文: nh 独立于 d，可自由调整
  d = 1536, dh = 64, nh ∈ {1,2,4,8,16,32}  # 任意选择
  # 每个 head dimension 仍为 dh=64
  # 输出投影 W_o^h ∈ R^{d×dh} 仍映射回 d 维
  ```
  解耦后 QKV 投影维度变为 [d, nh×dh]（而非 [d, d]），当 nh < d/dh 时参数量减少。

  **Step 1: Candidate Selection**
  ```
  Input: max(d)/dh = 32  # 最大模型的 hidden size 对应
  H_cand = []
  for nh in {1, 2, 4, 8, 16, 32}:
      for nkv in {1, 2, 4, 8, 16, 32}:
          if nkv <= nh:
              H_cand.append((nh, nkv))
  # |H_cand| = 21 个候选配置
  ```

  **Step 2: Scaling Curves Fitting**
  ```
  # 对每个 H ∈ H_cand，训练系列模型（3M→1.2B）with T=8K
  for (nh, nkv) in H_cand:
      for N in [3M, 19M, 85M, 150M, 200M, 470M, 680M, 1.2B]:
          model = build_model(N, nh, nkv)
          loss = train(model, SlimPajama, ratio=20:1)
          record (N, loss)

      # 拟合 power-plus-constant scaling law
      L(N) = (a/N)^b + E
      # a, b: 配置相关参数
      # E: 语言的自然熵（跨配置共享）
      # 拟合 R² > 0.999
  ```

  **Step 3: Cost Minimization**
  ```
  Input: target loss L*, context length T
  # 推理成本公式
  C_infer(T) = 2N + 4TL dh nh          # 时间不变 FLOPs + 时间相关 FLOPs
  M_infer(T) = N + 2TL dh nkv          # 参数内存 + KV cache 内存

  for (nh, nkv) in H_cand:
      # 从 scaling law 求解满足 L* 的最小 N
      N*(nh,nkv) = a(nh,nkv) / (L* - E)^{1/b(nh,nkv)}

      # 计算硬件感知成本
      Z(nh,nkv) = 0.9 * M_infer(T)^(1/2) + 0.1 * C_infer(T)^(1/3)

  # 选择 Z 最小的配置
  H* = argmin Z
  return (N*, nh*, nkv*)
  ```

  线性插值计算 N* 对应的精确 (L,d) —— 使用 Table 7 的预定义配置插值。
  实际部署时选择最接近 (N*, nh*, nkv*) 的实际整数配置。

  **推理流程（以 H=8,1 at T=128K 为例）**：
  ```
  # 模型配置: N=1.8B (vs Llama-3 1.2B), nh=8 (vs 32), nkv=1 (vs 8)
  # Prefill 阶段:
  for each layer l in 1..L:
      Q = X @ W_Q  # [T, nh*dh] = [T, 8*64] = [T, 512]
      K = X @ W_K  # [T, nkv*dh] = [T, 64] — 仅 1 个 KV head!
      V = X @ W_V  # [T, nkv*dh] = [T, 64]
      # GQA: 8 个 query head 共享 1 组 KV
      for g in 1..8:  # nh/nkv = 8 groups
          Q_g = Q[:, (g-1)*64 : g*64]  # [T, 64]
          attn_g = softmax(Q_g @ K^T / sqrt(64))  # [T, T]
          out_g = attn_g @ V  # [T, 64]
      output = concat([out_1, ..., out_8]) @ W_O

  # Decode 阶段 (single token):
      # KV cache 大小: 2 * L * T * dh * nkv = 2 * 36 * 128K * 64 * 1
      # vs Llama-3 GQA: 2 * 36 * 128K * 64 * 8 = 8× 更大!
  ```

  **推理成本对比（T=128K, L*=2.615, BF16）**：
  | 配置 | N | nh | nkv | M_infer | C_infer | 节省 |
  |------|---|----|-----|---------|---------|------|
  | Llama-3 GQA | 1.2B | 32 | 8 | baseline | baseline | — |
  | Cost-Optimal | 1.8B | 8 | 1 | **-50.8%** | **-57.8%** | 同 loss |

  **Throughput 实测（Table 5, A800）**：
  - Training: Llama-3 GQA (32,8) = 18,655 tok/s → Ours (8,1) = 31,260 tok/s（+67.6%）
  - Inference: Llama-3 GQA (32,8) = 12,921 tok/s → Ours (8,1) = 20,643 tok/s（+59.8%）

  **Key hyperparameter constraints**：
  - dh = 64（固定），V = 50,304（固定），dff ≈ 8d/3
  - AdamW optimizer (β1=0.9, β2=0.95, weight_decay=0.1, gradient_clip=1.0)
  - WSD LR scheduler (10% warmup, 20% cosine decay to 0.1× max_lr)
  - Max LR: grid-searched per model size on MHA baseline (1e-3 for 3M/19M/85M, 5e-4 for 150M/200M/470M, 2e-4 for 680M/1.2B)
  - Precision: BF16 (training), FP16 (evaluation)
