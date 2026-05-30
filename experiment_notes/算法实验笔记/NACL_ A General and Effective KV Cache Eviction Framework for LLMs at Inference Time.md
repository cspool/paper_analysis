## NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time

- 属于算法pipeline的实现是什么？实验比较什么？
  NACL 提出一种混合 KV cache 淘汰框架，在 encoding 阶段一次性完成全局最优淘汰（而非逐 token step-by-step 贪心淘汰），包含两个核心策略：(1) **PROXY-TOKENS EVICTION**：选取输入末尾的任务相关 token（如用户问题）作为 proxy tokens，利用 proxy tokens 对所有 prefix token 的全局 attention score 求和作为 token 重要性评分（F_score = Σ_{x_p∈P} Softmax(A(x_p, *))），保留 top-C_p 高分 token。相比 H2O 的全体 token 累加 attention score（引入冗余信息）和 MSRNN 的单当前 token attention（信息不足），proxy tokens 提供更精准的任务相关重要性估计。(2) **RANDOM EVICTION**：将 PROXY-TOKENS EVICTION 的评分经 Softmax 归一化后作为概率分布 P_prompt，从该分布中采样 C_r 个 token 保留，每个 attention head 和每层使用不同随机种子。这种 per-head per-layer 的多样化采样使信息在更多维度上被保留（LLaMA-7B 32层×32头，budget=20% 时 token 在至少一个 head 中保留概率达 99.92%）。最终 KV cache budget C = C_p + C_r，proxy tokens 默认约 10% budget。NACL 将淘汰建模为一次 encoding phase 全局操作，时间复杂度从 O(p+T) 降至 O(1)（long-context 下 T ≪ p）。

  实验比较：(a) short-text tasks (5-shot/25-shot)：NACL vs Attention Sink、H2O、MSRNN 在 lm-eval-harness 七个任务上的 accuracy；(b) long-text tasks (LongBench)：NACL vs Scissorhands、H2O、MSRNN 在 budget 10%/20%/30% 下的七个任务 accuracy；(c) KV cache 内存使用 vs sequence length（NACL 20% vs H2O 20%）；(d) 消融实验：移除 PROXY-TOKENS EVICTION（-28.1 short-text）、移除 RANDOM EVICTION（-1.2 short-text / -9.2 long-text）、uniform 采样替代 attention-score 采样（-0.8 short-text / -1.1 long-text）、step-by-step global eviction 替代 one-eviction（-1.3 short-text）、per-layer 替代 head-wise eviction（-2.1 short-text / -2.7 long-text）。

- 硬件平台是什么，配置是什么。
  单张 NVIDIA A100 80GB GPU（bfloat16 精度）。FlashAttention-2 用于加速注意力计算。Reduce Attention Scores CUDA kernel 实现兼容 FlashAttention-2 的 128K long-text 推理。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA2-7B-base、LLaMA2-7B-Chat（decoder-only Transformer，32 layers，32 heads per layer，d=4096）。
  
  数据集/Benchmark：
  - 短文本（lm-eval-harness）：PiQA（5-shot: 319 tokens）、COPA（118 tokens）、OpenBookQA（97 tokens）、Winogrande（160 tokens）、SciQA（508 tokens）、ARC-Easy（296 tokens）、ARC-Challenge（239 tokens）。25-shot 设置下 token 数约 5×。
  - 长文本（LongBench，4K context）：PassageRetrieval-Zh、PassageRetrieval-En、RepoBench-P、HotpotQA、NarrativeQA、TriviaQA、QMSum。
  - 辅助：perplexity 在 OpenBookQA 上计算。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2024-NACL

  **NACL 单层 encoding phase 前向 pipeline（张量级，LLaMA2-7B, 32 heads, d=4096, C=20%, C_p=6%, C_r=12%, proxy_budget≈10%）**：

  ```
  Step 1 - Prefill Q/K/V:
    x_prompt ∈ R^{p×4096}（p=4K tokens）
    for each head h (0..31):
      Q_h = x_prompt @ W_Q^h     # W_Q^h ∈ R^{4096×128}, Q_h ∈ R^{4096×128}
      K_h = x_prompt @ W_K^h     # K_h ∈ R^{4096×128}
      V_h = x_prompt @ W_V^h     # V_h ∈ R^{4096×128}

  Step 2 - Attention Scores（全量）:
    A_h = Q_h @ K_h^T / sqrt(128)     # A_h ∈ R^{4096×4096}, causal masked

  Step 3 - Proxy Tokens Selection:
    # 默认取输入末尾 ~10% 的 token 作为 proxy tokens
    # 实际应用中 proxy tokens = 用户问题部分（位于 prompt 末尾）
    P = {p * 0.9, ..., p-1}           # |P| ≈ 410 tokens (10% of 4K)
    P_h = P                           # head h 的 proxy token 索引

  Step 4 - PROXY-TOKENS EVICTION (F_score):
    # 仅用 proxy tokens 行求和作为 token importance
    F_score = Σ_{x_p ∈ P_h} Softmax(A_h[x_p, :])    # ∈ R^{4096}
    # F_score[j] 表示 token j 对所有 proxy tokens 的综合重要性

  Step 5 - Top-K selection for proxy eviction:
    R = x_prompt \ P_h                              # 非 proxy token 集合
    C_p = 0.06 * p ≈ 246 tokens                     # proxy eviction budget
    u_score = TopK(F_score[R], C_p)                 # 选非 proxy 中最高分 C_p 个
    u_score = u_score ∪ P_h                         # proxy tokens 默认保留

  Step 6 - RANDOM EVICTION:
    # 从 F_score 构建概率分布
    P_prompt_h = Softmax(F_score)                   # ∈ R^{4096}, 概率分布
    C_r = 0.12 * p ≈ 492 tokens                     # random eviction budget
    u_random_h ~ Multinomial(P_prompt_h, C_r, seed=h)  # per-head 不同 seed

  Step 7 - 合并保留集:
    S_encoding^h = u_score ∪ u_random_h             # 共 C = 738 tokens (20%)
    K_cache^h = K_h[S_encoding^h]                   # [738, 128]
    V_cache^h = V_h[S_encoding^h]                   # [738, 128]
    # 淘汰率 80% (4K → 738)

  Step 8 - Generation Phase（逐 token decoding）:
    for each new token z_t:
      for each head h:
        K_cache^h = [K_cache^h, z_t @ W_K^h]       # 追加新 token KV
        V_cache^h = [V_cache^h, z_t @ W_V^h]
        if t % m == 0:                              # 每 m 步淘汰一次
          A_t = (z_t @ W_Q^h) @ K_cache^h^T / sqrt(128)
          S_t^h = Eviction(A_t, C)                  # 重复 Step3-7
          K_cache^h, V_cache^h = K_cache^h[S_t^h], V_cache^h[S_t^h]
  ```

  **与 baselines 的关键差异**：
  - H2O: F_score = Σ_{all tokens} Softmax(A[i, :])（全量累加 attention → 冗余信息 + attention bias）
  - MSRNN: F_score = Softmax(A[current_token, :])（仅当前 token → 信息不足）
  - NACL: F_score = Σ_{x_p∈P} Softmax(A[x_p, :])（proxy tokens 子集 → 精准 + 抗 bias）+
    head-wise RANDOM EVICTION（概率采样 → 增加信息多样性）

  **KV cache 压缩效果**：
  - LLaMA2-7B, batch=4, seq_len=32K, bf16: 64GB → NACL 20% ≈ 12.8GB (5× reduction)
  - NACL 20% short-text avg: 63.8 vs Full 64.6 (-0.8), H2O 20%: 60.3 (-4.3) — 80% improvement
  - NACL 20% long-text avg: 30.8 vs Full 31.5 (-0.7), H2O 20%: 28.6 (-2.9) — 76% improvement
  - NACL 30% long-text: PR-Zh=6.8 (H2O=3.7), PR-En=9.0 (H2O=5.0) — NACL 在 passkey retrieval 上显著优于 H2O
