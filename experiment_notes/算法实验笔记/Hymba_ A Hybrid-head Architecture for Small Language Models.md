## Hymba: A Hybrid-head Architecture for Small Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - **Hymba**：一种融合 attention 和 SSM（Mamba）的 hybrid-head 并行架构。核心创新包括四部分：(1) **Hybrid-Head Module**：在同一层内并行放置 attention heads 和 SSM heads，两者处理相同输入，输出经 learnable β₁, β₂ 归一化重缩放后取均值融合；(2) **KV Cache 优化**：仅 3 层使用 global attention（首/中/末层），其余层使用 sliding window attention (SWA)；相邻层间共享 KV cache（cross-layer KV sharing）；(3) **Meta Tokens**：128 个可学习 token 前置到输入序列，作为 learned cache initialization 引导 attention 关注有意义 token，减轻 attention sink 现象；(4) **Scaling**：模型从 300M 消融扩展到 1.5B（32 layers, hidden=1600, 25 attn heads, 5 query groups），训练 1.5T tokens。
  - 实验比较包括：(a) 与 SOTA sub-2B 小模型对比（SmolLM2-1.7B、Qwen2.5-1.5B、Llama-3.2-1B/3B、Phi-1.5、h2o-danube2、OpenELM 等）；(b) Apple-to-apple 架构对比：同参数/同数据/同训练 recipe 下对比 Llama3、Mamba2、Mamba2 w/ FFN、Samba（sequential Mamba-Attn）在 300M 和 1B 两个规模的各项任务表现；(c) Needle-in-a-Haystack 长上下文检索对比；(d) Instruction-tuned 模型对比（Llama-3.2-1B-Instruct、Qwen2.5-1.5B-Instruct、SmolLM-1.7B-Instruct）；(e) 消融实验：Attention/SSM head 比例、parallel vs sequential fusion、local/global attention ratio、KV cache sharing、meta tokens、fusion strategy（mean vs concat）。

- 硬件平台是什么，配置是什么。
  - 训练：128× NVIDIA A100 GPU（pretrain 1.5B 模型 1.5T tokens）
  - 推理吞吐测试：NVIDIA A100 GPU，sequence length=8K，batch size=128，PyTorch（OOM 时减半 batch size 直到不 OOM）
  - 后训练（SFT + DPO）：论文未明确说明 GPU 数量/型号

- 模型是什么。数据集和bench分别是什么。
  - 模型：Hymba-125M (24 blocks, hidden=512, 8 attn heads)、Hymba-350M (32 blocks, hidden=768, 12 attn heads)、Hymba-1.5B (32 blocks, hidden=1600, 25 attn heads, 5 query groups)
  - 训练数据：DCLM-Baseline-1.0 + SmolLM-Corpus + NVIDIA 内部高质量数据集（Hymba-1.5B: 1.5T tokens total，其中 public data only 版本为 DCLM 1T + SmolLM 500B）
  - Ablation (300M)：100B tokens 训练，序列长度 1K/2K；FineWeb 数据集（A.3 apple-to-apple at 300M）
  - Apple-to-apple (1B)：100B tokens SmolLM-Corpus
  - 评估 Benchmark：
    - Commonsense Reasoning: MMLU (5-shot), ARC-Easy/C (0-shot), PIQA (0-shot), HellaSwag (0-shot), Winogrande (0-shot), OBQA (0-shot), TruthfulQA (0-shot), SIQA (0-shot), LAMBADA (0-shot)
    - Recall-Intensive: SQuAD-C (1-shot), SWDE
    - Language Modeling: WikiText-2 perplexity, LMB perplexity
    - Instruction-tuned: GSM8K (5-shot), GPQA (0-shot), IFEval, BFCLv2 (Berkeley Function-Calling Leaderboard), RoleBench
  - 评估框架：lm-evaluation-harness（主评估），HuggingFace/LightEval（小模型评估）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源模型：HuggingFace 上发布 Hymba-1.5B-Base 和 Hymba-1.5B-Instruct
  - 基于 Mamba 和 Llama 架构实现，使用 PyTorch 框架

  **Hymba Hybrid-Head 算法pipeline（单层前向）**：

  ```
  # 输入: X ∈ R^{N×d_model}（原始 token 序列，N=tokens, d_model=hidden_dim）
  # 先 prepend meta tokens R ∈ R^{m×d_model}, m=128
  X̃ = concat([R, X], dim=0)  # X̃ ∈ R^{(N+m)×d_model}

  # Step 1: 输入投影
  # W_in_proj = [W^Q, W^K, W^V, W^{SSM}, W^G]
  Q = W^Q @ X̃  # attention queries
  K = W^K @ X̃  # attention keys
  V = W^V @ X̃  # attention values
  X_ssm = W^{SSM} @ X̃  # SSM input features
  G = W^G @ X̃  # SSM gates

  # Step 2a: Attention head 输出（sliding window 或 global）
  # 滑动窗口时 causal mask 限制为 window_size 内
  M_attn = softmax(Q @ K^T / √d_head) @ V    # Y_attn = M_attn @ X̃
  # 若使用 GQA: K, V 的 head 数少于 Q 的 head 数

  # Step 2b: SSM head 输出（Mamba-style, 逐 token recurrent）
  # B = W_B @ X_ssm, C = W_C @ X_ssm
  # Δ = Softplus(W_Δ @ X_ssm)
  for i in 1..N+m:
      # Discretize continuous SSM
      Ā_i = exp(Δ_i ⊗ A)  # A ∈ R^{d_state×d_state}, Δ_i ∈ R^{d_inner}
      B̄_i = Δ_i ⊗ B_i
      # Recurrent update
      h_i = Ā_i ⊙ h_{i-1} + B̄_i ⊙ X_ssm[i]   # h ∈ R^{d_inner×d_state}
      y_i = C_i @ h_i
  Y_ssm = G ⊙ Y  # element-wise gate

  # Step 3: 融合（归一化 + 重缩放 + 平均）
  Y_attn_norm = norm(Y_attn)
  Y_ssm_norm = norm(Y_ssm)
  Y_fused = β₁ ⊙ Y_attn_norm + β₂ ⊙ Y_ssm_norm
  # β₁, β₂ ∈ R^{d_model} 是可学习 per-channel 缩放向量

  # Step 4: 输出投影
  Y = W_out_proj @ Y_fused

  # 注：实际实现中，每层有多个 attention heads 和 SSM heads，
  # 如 1.5B: 25 attn heads × (d_head=64), SSM heads 占据剩余维度
  # attn:mamba 参数比约 1:5.23（最终配置含 GQA 和 KV sharing 后）
  ```

  **KV Cache 优化配置**：
  - 仅第 1 层、中间层、最后 1 层使用 global full attention（共 3 层）
  - 其余 29 层使用 sliding window attention（window_size=1024）
  - 每 2 个连续层共享同一 KV cache（cross-layer KV sharing）
  - 结果：8K 序列下 cache size 从 Llama 的 414.7MB 降至 39.4MB（10.5× reduction）

  **Meta Tokens 推理流程**：
  ```
  # 离线预计算（仅一次）
  K_meta = W^K @ R     # meta tokens 的 K
  V_meta = W^V @ R     # meta tokens 的 V
  X_ssm_meta = W^{SSM} @ R  # meta tokens 的 SSM 输入
  # 存储这些值作为 "learned cache initialization"

  # 在线推理
  X̃ = concat([K_meta_cache, K_input], dim=0)  # 在 K cache 维度
  # 后续计算同上述 pipeline，meta tokens 部分的 K/V/SSM 状态从预计算值加载
  ```

  **训练配置**：
  - LR scheduler: Warmup-Stable-Decay (WSD)，warmup=1% steps，stable peak lr=3e-3，decay to 1e-5 over 20% steps
  - Sequence length: 2K（最后 100B tokens 增至 8K，同步调整 ROPE base）
  - Batch size: 2M tokens
  - 后训练：FFT (lr=5e-5) → DPO (lr=3e-6)，LMFlow toolkit，packed samples (block_size=8192 for SFT, 2048 for DPO)

  **关键结果**：
  - Hymba-1.5B avg accuracy 61.06% vs SmolLM2-1.7B 60.04%（+1.02%），cache 79MB vs 1573MB（19.91× reduction），throughput 664 vs 238 tok/s（2.79×）
  - 超越 Llama-3.2-3B：avg +1.32%，cache 11.67× smaller，3.49× faster
  - Apple-to-apple 1B avg 54.57% vs Llama3 52.82%（+1.75%），vs Samba 52.83%（+1.74%）
