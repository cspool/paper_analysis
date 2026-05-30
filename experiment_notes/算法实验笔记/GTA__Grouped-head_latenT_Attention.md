## GTA__Grouped-head_latenT_Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  论文提出 **GTA (Grouped-head latenT Attention)**，一种新的注意力机制，包含两个组件：(1) **Shared Attention Map**：将 query 和 key 按 heads 分组，同一 group 内的 heads 共享同一套 query-key 注意力计算，从而减少 MHA 中每个 head 独立计算的冗余——每个 head 映射到某个 Q group 和某个 K group，仅对 (Q_{q(i)}, K_{k(i)}) 计算注意力分数，而非每个 head 独立计算 Q_i K_i^T；(2) **Nonlinear Value Decoder**：引入一个压缩的 latent value representation C ∈ R^{N × n_c × d_l}（d_l ≥ d_h），对 latent value 先用 head-specific 投影矩阵 W_{P,i} ∈ R^{d_l × d_h} 生成 value，再通过 element-wise 乘以 sigmoid gate Sigmoid(x_t W_{G,i})（对当前 token 输入自适应）调制，实现非线性、上下文自适应的 head-specific value 生成。实验比较 MHA、GQA、MLA 三种 baseline，评估下游任务准确率、预填充/解码时延、KV cache 大小。

- 硬件平台是什么，配置是什么。
  训练：4 节点 × 8 NVIDIA A800 80GB GPU（共 32 GPU），分布式训练，支持 1~4 节点弹性扩展。
  推理（LLM-Viewer 模拟）：NVIDIA A100 40GB、A100 80GB、H100 80GB、H100 PCIe 80GB。
  推理（实际部署）：NVIDIA H100 80GB、NVIDIA A800 80GB、NVIDIA RTX 3060 12GB、Apple M2、BCM2712（移动处理器）。

- 模型是什么。数据集和bench分别是什么。
  模型：160M 参数（24 层，hidden=768，n_h=12），500M 参数（24 层，hidden=1280，n_h=20），1B 参数（54 层，hidden=1280，n_h=20）。GTA 变体 GTA1~GTA4 采用不同 n_q/n_k/n_c 分组数（如 GTA1: n_q=3, n_k=1, n_c=1, d_l=128；GTA4: n_q=10, n_k=1, n_c=2, d_l=256）。
  预训练数据集：C4（160M/500M 验证实验）、smollm-corpus 220B tokens（1B 扩展实验）。
  微调数据集：tulu3-sft-mixture。
  Benchmark：PIQA、HellaSwag、ARC-e、ARC-c、Winogrande、BoolQ、MathQA、TruthfulQA、SIQA、LogiQA、BBH、MBPP、IFEval、Wikitext PPL。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/plm-team/GTA。训练框架基于自定义 PyTorch 实现，微调使用 LlamaFactory [39]，评估使用 lm-evaluation-harness [25]。

  **GTA 算法详细张量计算流程（以 1B 模型为例，n_h=20, n_q=5, n_k=1, n_c=1, d_h=64, d_l=128）：**

  **Step 1: 输入投影（Eq 5）**
  ```
  X ∈ R^{N × 1280}   # N tokens, hidden=1280

  Q = X @ W_Q        # W_Q ∈ R^{1280 × 320},  Q ∈ R^{N × 320}  (5 Q groups × 64 head_dim)
  K = X @ W_K        # W_K ∈ R^{1280 × 64},   K ∈ R^{N × 64}   (1 K group × 64 head_dim)
  C = X @ W_C        # W_C ∈ R^{1280 × 128},  C ∈ R^{N × 128}  (1 C group × 128 latent_dim)
  ```

  **Step 2: 按分组映射 head**
  ```
  q(i): {0..19} → {0..4}   # 20 heads 映射到 5 Q groups
  k(i): {0..19} → {0}      # 20 heads 共享 1 套 key
  c(i): {0..19} → {0}      # 20 heads 共享 1 套 latent value
  ```

  **Step 3: 非线性 Value 解码（Eq 6）**
  ```
  for each head i in 0..19:
      # 从共享 latent value C 投影生成 head-specific value
      V_i = (C @ W_{P,i}) ⊙ sigmoid(x_t @ W_{G,i})
      # W_{P,i} ∈ R^{128 × 64}: d_l → d_h
      # W_{G,i} ∈ R^{1280 × 64}: H → d_h, x_t 是当前 token
      # ⊙ 为 element-wise 乘法
  ```

  **Step 4: 注意力计算（Eq 7, 等价公式 Eq 8）**
  ```
  for each head i in 0..19:
      # 共享 attention map：使用 Q group 的 query 对 K group 的 key 计算 score
      Q_group = Q[q(i) * 64 : (q(i)+1) * 64]    # (N, 64)
      K_group = K                                 # (N, 64), 所有 heads 共享
      C_group = C                                 # (N, 128)

      # Eq 8: 先对 latent value 做 attention，再 gate
      attn_scores = Q_group @ K_group^T / sqrt(64)     # (N, N)
      attn_weights = softmax(attn_scores)               # (N, N)
      # 在 latent 空间计算 attention
      O_i_raw = attn_weights @ C_group                   # (N, 128)
      # 投影到 head_dim 并 gate
      O_i = (O_i_raw @ W_{P,i}) ⊙ sigmoid(x_t @ W_{G,i}) # (N, 64)
      # 输出投影
      O_i = O_i @ W_{O,i}                                # (N, 1280)
  ```

  **Step 5: 合并输出**
  ```
  O = sum(O_i for i in 0..19)    # (N, 1280)
  ```

  **KV Cache 写入**：仅 cache K (64 dims/token) 和 C (128 dims/token)，共 192 dims/token/layer vs MHA 的 2560 dims/token/layer (=7.5%)。decode 时仅需追加 1 个 token 的 K 和 C 并重新计算 gate。

  **关键优化 tricks：**
  - Equation 8 的 reformulation 使得 decode 时无需重新计算所有历史 V_i（仅需存 C 并执行 latent-space attention），大幅减少 decode 的 FLOPs。
  - Shared attention matrix 减少了 QK^T 计算次数：从 MHA 的每个 head 独立计算（n_h 次）降至每个 Q-K group 组合的计算（n_q 次），当 n_q << n_h 时显著节省。
  - 数值结果（1B 模型）：预填充 FLOPs 降至 GQA 的 37.5%，KV cache 降至 GQA 的 30%，预填充时延 2× 加速，解码时延也显著改进。
