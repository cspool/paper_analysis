## MEDA: Dynamic KV Cache Allocation for Efficient Multimodal Long-Context Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  MEDA 是一种免训练（training-free）的动态层间 KV cache 分配方法，专为多模态长上下文推理设计。核心实现包含三部分：(1) **跨模态注意力熵（Cross-Modal Attention Entropy）**：每层计算文本→视觉（A_TV）和视觉→文本（A_VT）的跨模态注意力矩阵，计算其注意力熵 E_CM^l = -(E_TV^l + E_VT^l)，以此量化该层注意力分布的不确定性和分散程度。熵越低表示注意力越集中于关键跨模态 token 对，该层对 KV cache 需求较小；熵越高表示注意力越分散，需要更多 KV cache。基于此通过 inverse entropy softmax allocation 公式 S_l = α_l · S, α_l = exp(E_CM^l) / Σ_k exp(E_CM^k) · L · ρ 动态分配各层的 KV cache 大小。(2) **多模态 KV Pair 选择（Multimodal KV Pair Selection）**：在 prefill 阶段计算累积注意力分数 A_s，对文本 token 的注意力分数加 max(A_s) 偏置优先保留文本 KV pairs，保留最近 M 个 token 的上下文窗口，从剩余 token 中选取 top-N 个最高注意力分数的 token 组成保守 cache (K_c, V_c)。(3) **多模态 KV Pair 合并（Multimodal KV Pair Merging）**：对未选中的 less important tokens，通过 many-to-one nearest-neighbor matching 基于 cosine similarity 匹配到最近的保守 token，使用平均合并策略 k_j ← (k_j + Σ_{i∈N_j} k_i) / (|N_j| + 1) 合并 KV pair，保留全局上下文信息而非简单丢弃。实验比较在 MileBench（多图像多文本 benchmark）、Video-ChatGPT、DREAM-1K、WorldQA 上与 H2O、SnapKV、PyramidKV（text-centric baselines）和 LOOK-M（multimodal KV cache baseline）在不同压缩比 ρ 下的 accuracy/ROUGE-L/F1 等性能指标以及 decoding latency 和 GPU memory usage。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 GPU（单卡）。实验环境详见于附录 A.2：AMD EPYC 7643 48-Core Processor + NVIDIA A100 GPU。精度测试使用 HuggingFace Transformers。速度测试使用 DREAM-1K 前 20 个 YouTube 视频样本。解码速度测量为 decoding 阶段时间除以总生成 token 数。KV cache 内存使用计算：Memory = (input_len + decoding_len) × 2 × 32 × 32 × 128 × 2 / (1024³) GiB，其中 2 表示 FP16 精度 2 bytes，32 为 attention head 数和 layer 数，128 为每 head 维度，第二个 2 表示 K 和 V 各一份。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-v1.5-13B（32 layers）、LLaVA-NeXT-7B（32 layers）、InternVL-v1.5-7B（用于 multi-images 任务）；LLaVA-Video-7B/32B、LongVA-7B、LongVILA-8B（用于 long-video 任务）。数据集：(1) MileBench——6440 个多模态长文本样本，平均 15.2 张图片和 422.3 words/sample，含四类子任务：Temporal Multi-image Tasks (T)、Semantic Multi-image Tasks (S)、Needle in a Haystack Tasks (NH)、Image Retrieval Tasks (IR)，评估指标为 accuracy 和 ROUGE-L；(2) Video-ChatGPT——基于 ActivityNet-200 的视频描述 benchmark，GPT-3.5 评估正确性、细节、上下文理解、时序理解四个维度；(3) DREAM-1K——1000 个视频片段（真人电影、动画、stock footage、YouTube、TikTok），AutoDQ 指标评估 F1/Precision/Recall；(4) WorldQA——开放式 QA 数据集，GPT-4 评估生成质量。压缩比 ρ 在 multi-images 任务默认为 0.1，long-video 任务默认为 0.2。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/AIoT-MLSys-Lab/MEDA（论文中声明）

  **算法 Pipeline 核心流程**：

  **Stage 1 - Prefill + 跨模态注意力熵计算（Prompt Encoding）**：
  ```
  输入：多模态 prompt X ∈ R^{L_prompt × D}（含 text tokens X_n^T, image tokens X_m^I, video tokens X_q^V）
  
  for each layer l = 1..L:
      # 标准 QKV 投影
      Q^l = X W_Q^l, K^l = X W_K^l, V^l = X W_V^l    # [L_prompt, D]
      
      # 获取 text / visual token 子集索引
      Q_T^l = Q^l[text_indices]    # [n_T, D]
      K_T^l = K^l[text_indices]    # [n_T, D]
      Q_V^l = Q^l[visual_indices]  # [n_V, D]
      K_V^l = K^l[visual_indices]  # [n_V, D]
      
      # 跨模态注意力计算
      A_TV^l = Softmax(Q_T^l (K_V^l)^T / √D)    # [n_T, n_V]
      A_VT^l = Softmax(Q_V^l (K_T^l)^T / √D)    # [n_V, n_T]
      
      # 跨模态注意力熵（公式 5-6）
      E_TV^l = (1/|T|) Σ_i Σ_j A_TV^l[i,j] · log(A_TV^l[i,j])
      E_VT^l = (1/|V|) Σ_i Σ_j A_VT^l[i,j] · log(A_VT^l[i,j])
      E_CM^l = -(E_TV^l + E_VT^l)
  
  # 动态层间 KV cache 分配（公式 7）
  for each layer l:
      α_l = exp(E_CM^l) / Σ_k exp(E_CM^k) · L · ρ   # ρ: 压缩比
      S_l = α_l · S                                   # S: 总 KV cache budget
  ```

  **Stage 2 - KV Pair 选择与合并**：
  ```
  对每层 l，分配 budget S_l：
      # 1. 累积注意力分数（公式 8）
      A_p = Attn(Q_p K_p^T)                    # prefill 阶段的 attention
      A_s = Σ_i A_p[i, :]                      # 沿 query 维度求和 [L_prompt]
      
      # 2. Text-prior 偏置（公式 9）
      A_s[T] = A_s[T] + max(A_s)               # T = text token 索引
      
      # 3. 保留最近 + 选 top-N 重要 token（公式 10）
      I = Top_N(A_s[:-M])                       # 排除最近 M 个 token
      K_c = [K[I, :]; K[-M:, :]]               # 保守 cache
      V_c = [V[I, :]; V[-M:, :]]
      
      # 4. Many-to-one 最近邻匹配（公式 11）
      for each less important token i in I_less:
          for each conserved token j in I_c:
              u_{i,j} = cos_sim(k_i, k_j)      # key token 间的余弦相似度
  
      # 5. 平均合并（公式 12）
      for each j in I_c:
          N_j = {i | j = argmax cos_sim(k_i, k_j)}
          k_j ← (k_j + Σ_{i in N_j} k_i) / (|N_j| + 1)
          v_j ← (v_j + Σ_{i in N_j} v_i) / (|N_j| + 1)
  ```

  **Stage 3 - Decoding with Compressed KV Cache**：
  ```
  for each new token x_t:
      q_t = x_t W_Q
      # 使用压缩后的 K_c, V_c 计算 attention
      x_{t,out} = Softmax(q_t K_c^T / √D) V_c
      # 新 token 的 KV 追加到 cache
      K_c ← [K_c, x_t W_K], V_c ← [V_c, x_t W_V]
  ```

  **关键参数**：β₁ : β₂ = 3 : 1（recent context tokens M 和 important tokens N 的比例），memory overhead per layer 与 β₁ + β₂ 成正比。ρ 为总压缩比（如 0.1 即保留 10% KV cache）。所有实验在单张 A100 上完成，精度为 FP16。

  **复杂度**：跨模态熵计算仅在 prefill 阶段执行一次（O(n_T · n_V) per layer），KV pair 选择为 O(L_prompt) TopK 操作，合并为 O(L_less · L_c) 最近邻搜索。与 prefill 的 O(L_prompt²) 相比额外开销极小。Decoding 阶段直接使用压缩后的 KV cache，内存和延迟均降低。

  **关键效果**：
  - MileBench, LLaVA-NeXT-7B, ρ=0.1：MEDA 在所有 11 个 sub-task 上均优于或接近 Full Cache，整体显著优于 H2O/SnapKV/PyramidKV/LOOK-M
  - LLaVA-Video-7B, ρ=0.2：F1 31.3 vs Full Cache 32.5（H2O: 27.7, SnapKV: 28.8）
  - 20% budget 下 GPU memory 从 2.42 GiB 降至 0.67 GiB（72% 减少），decoding latency 从 14.61 ms/token 降至 8.23 ms/token
  - 5% budget 下 decoding latency 降至 5.18 ms/token（2.82× speedup）
  - 无需任何 fine-tuning，即插即用兼容所有 MLLM
