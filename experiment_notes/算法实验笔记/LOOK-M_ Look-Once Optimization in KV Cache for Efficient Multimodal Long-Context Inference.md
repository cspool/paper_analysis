## LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  LOOK-M 是一种免微调（fine-tuning-free）的多模态 KV cache 压缩方法，核心实现分为两部分：(1) **Text-Prior KV Pair Eviction**：在 prompt prefilling 阶段，利用累积注意力分数（cumulative attention scores）动态更新 KV cache，但区别于传统的无差别积累方法（如 H2O），LOOK-M 对文本 token 赋予 text-prior 值 T_p = Max(A_s) 加到其累积注意力分数上，确保文本 token 在 eviction 阶段优先保留，图像 token 中仅保留 attention score 最高的 top-N 个；(2) **KV Pairs Merging**：对被 evicted 的 KV pair 使用 many-to-one nearest-neighbor matching 找到其最相似的 conserved token，然后通过三种合并策略（averaged merging、pivotal merging、weighted merging）将 evicted token 的信息融入 conserved token 中。实验比较 LOOK-M（含 A-Merge/W-Merge/P-Merge 三种合并策略，以及叠加 text-prior TP 的组合共 6 种变体）与 text-only KV cache eviction baselines（H2O、SnapKV、RoCo）和 Full Cache 在 MileBench 基准上的准确率/ROUGE-L，以及不同 KV cache budget（5%-100%）、不同压缩率比例（α¹:α²）、不同模型架构下的性能，和 decoding latency/GPU memory 效率。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 (80GB) 和 RTX 3090 (24GB) GPU。延迟和 GPU 内存测试在 RTX 3090 单卡上进行。FlashAttention-2 加速注意力计算。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-v1.5-7B、LLaVA-v1.5-13B（LLM backbone: Vicuna-7B/13B）、InternVL-v1.5-7B、MobileVLM-V2-3B。
  数据集/Benchmark：MileBench，含 4 类子任务——T: Temporal Multi-image Tasks（T1-T4，含 Action Localization/Prediction/Sequence, Object Existence/Interaction, Egocentric Navigation, Counterfactual Inference/State Change 等）、S: Semantic Multi-image Tasks（S1-S5，含 Webpage QA/Textbook QA, Slide QA/OCR/Document QA, Visual Change Captioning, Multimodal Dialogue, Space Understanding）、N: Needle in a Haystack Tasks（N1 Text Needle, N2 Image Needle）、I: Image Retrieval。评估指标为 Accuracy 和 ROUGE-L。默认 recent ratio α¹=0.1，important ratio α²=0.1。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/SUSTechBruce/LOOK-M

  **算法 Pipeline（基于 Section 3 Methodology）**：

  **输入**：多模态 prompt X = {X₁^T, X₁^I, ..., X_N^T, X_M^I}，L_prompt 为 prompt 长度，D 为 hidden dimension。最近窗口大小 M = α¹ × L_prompt，重要 token 数 N = α² × L_prompt。

  **Phase 1: Multimodal Prompt Encoding（Prefilling）**：
  ```
  for each transformer layer:
      K = X @ W_K   # shape: [L_prompt, D]
      V = X @ W_V   # shape: [L_prompt, D]
      Q = X @ W_Q   # shape: [L_prompt, D]
      A_p = softmax(Q @ K^T / sqrt(D))   # [L_prompt, L_prompt], causal
  ```

  **Phase 2: Text-Prior KV Pair Eviction（3.2 节，公式 4-8）**：
  ```
  # Step 1: 计算累积注意力分数
  A_s = sum(A_p[i,:] for i in 0..L_prompt)   # [L_prompt], 沿 query 维求和

  # Step 2: Text-Prior 增强——文本 token 获得优先级
  T_p = max(A_s)                              # 取最大 attention score 作为 text-prior
  for each textual token index t in T:
      A_s[t] = A_s[t] + T_p                   # 文本 token 分数 += text-prior

  # Step 3: 选择保留 token
  recent_kv = KV[-M:]                         # 最近 M 个 KV pair 始终保留
  I = Top_N(A_s[:-M], N)                      # 从前 L_prompt-M 个 token 中选 top-N
  K_c = concat(K[I], recent_K)                # conserved K: [N+M, D]
  V_c = concat(V[I], recent_V)                # conserved V: [N+M, D]
  K_e = K \ K_c                               # evicted K
  ```

  **Phase 3: KV Pairs Merging（3.3 节，公式 9-12）**：
  ```
  # Step 1: 计算 similarity matrix between K_e and K_c
  for i in evicted_indices I_e:
      for j in conserved_indices I_c:
          s_ij = cosine_sim(K[i], K[j])       # k_i^T k_j / ||k_i|| ||k_j||

  # Step 2: 对每个 conserved token j，找其 maximum similarity set
  for j in I_c:
      k_sim[j] = {K_e[i] | argmax_i matches j}

  # Step 3: 合并策略（三选一）
  # (a) Averaged Merging: 直接平均
  k_c[j] = 1/(L_sim + 1) * (k_c[j] + sum(k_sim[j]))

  # (b) Pivotal Merging: 先融合 evicted↔closest，再平均
  pivotal = avg(k_e[i], k_closest)            # 对每个 evicted token
  k_c[j] = 1/(L_sim + 1) * (k_c[j] + sum(pivotal))  # 再加权 conserved

  # (c) Weighted Merging: 基于 similarity 矩阵动态加权
  k_c[j] = 1/(L_sim + 1) * (k_c[j] + sum(k_sim[i] * S[x][y]))

  # Value 合并使用与 Key 相同的 similarity matrix 和权重（alignment property）
  ```

  **Phase 4: Token Generation（Decoding）**：
  ```
  for each new token x_t:
      q_t = x_t @ W_Q
      k_t, v_t = x_t @ W_K, x_t @ W_V
      K = concat(K_c, k_t)   # compressed KV + new token
      V = concat(V_c, v_t)
      x_t_out = softmax(q_t @ K^T / sqrt(D)) @ V
  ```

  **默认配置**：recent ratio α¹=0.1，important ratio α²=0.1（总 cache budget = 20%），最佳合并策略为 TP + P-Merge（text-prior + pivotal merging）。在 extreme compression ratio 99%（仅保留 1%）下 LOOK-M 仍维持接近 Full Cache 的性能。

  **关键效率数据（Table 4, RTX 3090）**：
  - Full Cache (100%): Decoding Latency 28.16 ms/token, GPU Memory 1.52 GiB
  - LOOK-M (20% budget): Decoding Latency 20.98 ms/token, GPU Memory 0.32 GiB (≈80% memory reduction)
  - LOOK-M (5% budget): Decoding Latency 18.22 ms/token, GPU Memory 0.13 GiB (≈92% memory reduction)
