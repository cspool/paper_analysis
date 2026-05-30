## Cross-Self KV Cache Pruning for Efficient Vision-Language Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 Cross-Self Pruning (CSP)，一种 training-free 的 KV cache 剪枝方法，专为多模态视觉语言模型 (VLM) 设计。核心设计包含两部分：(1) **Cross-Self Attention Decomposition**：将原始注意力矩阵 A ∈ [0,1]^{L×L} 分解为 intra-modality attention（同一模态内的 self-attention: A^{st} ∈ [0,1]^{L_t×L_t}, A^{sv} ∈ [0,1]^{L_v×L_v}）和 inter-modality attention（跨模态的 cross-attention: A^{ct} ∈ [0,1]^{L_v×L_t}, A^{cv} ∈ [0,1]^{L_t×L_v}）。对两类注意力分别沿 query 轴求和得到重要性分数 A^s 和 A^c，然后独立进行 top-K 选择得到 M^s 和 M^c，最终 mask M = M^s ∧ M^c 取交集（即 token 必须在 intra- 和 inter- 两个维度都被判定为重要才保留）。同时使用 observation window O（最近 O 个 query token）和 recent window R 来剪裁注意力矩阵 A[-O:, :-R]，聚焦于最近上下文的实际需求；(2) **n-Softmax 平滑恢复**：剪枝后 attention 分布的 denominator 从 Σ_{j∈I^+ ∪ I^-} e^{O_j} 变为 Σ_{j∈I^+} e^{O_j}，导致注意力分数变得更加尖锐（sharpness-shift）。引入 n-softmax：A_i = e^{O_i} / (n + Σ_{j∈I^+} e^{O_j})，通过加上偏置 n（默认 n=1）恢复原始分布的平滑性。实验比较 MileBench 基准上与 H2O、SnapKV、ReCo、LOOK-M 系列方法的性能（29 个多模态子任务），以及不同 cache budget（10%/20%/30%/60%/100%）下的效率（解码延迟 + GPU 内存）。

- 硬件平台是什么，配置是什么。
  LLaVA-v1.5-7b 实验：NVIDIA RTX 4090 GPU，flash-attn-2.4.3post1。LLaVA-v1.5-13b 实验：NVIDIA A100 GPU，flash-attn-2.6.3。抽样温度 0（确定性生成），最大上下文长度 4096 tokens。MMCoQA/NeedleInAHaystack/GPR1200 数据集 batch_size=1，其余数据集 batch_size=24。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-v1.5-7b（主要）、LLaVA-v1.5-13b、InternVL-v1.5-7B、MobileVLM-V2-3B。
  Benchmark：MileBench（29 个多模态数据集），分为：
  - Temporal Multi-Image Tasks (T1-T4)：Action Localization/Prediction/Sequence, Object Existence/Interaction/Moving Attribute/Shuffle, Egocentric Navigation/Moving Direction, Counterfactual Inference/State Change/Character Order/Scene Transition
  - Semantic Multi-Image Tasks (S1-S5)：Webpage QA/Textbook QA/Complex Multimodal QA, Slide QA/OCR QA/Document QA, Spot-the-Diff/CLEVR-Change, MMCoQA/ALFRED, nuScenes
  - Needle in a Haystack (NH)：Text & Image NeedleInAHaystack
  - Image Retrieval (IR)：GPR1200
  评估指标：各子任务内数据集平均准确率/ROUGE-L。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/TerryPei/CSP。算法 pipeline 如下：

  **Algorithm 1 核心伪代码**：
  ```
  Input: O ∈ R^{H×L_q×L_k} (attention logits), K, V caches, budget T, recent size R, observation window O
  for each decoding iteration:
      if L_k < T: return K, V  // cache 未满，不剪枝
      
      // Step 1: n-Softmax 计算注意力权重（平滑恢复）
      A = n-Softmax(O)  // A_i = e^{O_i} / (n + Σ_{j∈I^+} e^{O_j}), n=1
      
      // Step 2: Cross-Self 分解
      // 从 A 中分出 4 个子矩阵：
      A^{st} = A[:L_t, :L_t]           // text→text self-attention
      A^{sv} = A[L_t:, L_t:]           // visual→visual self-attention
      A^{ct} = A[L_t:, :L_t]           // visual→text cross-attention
      A^{cv} = A[:L_t, L_t:]           // text→visual cross-attention
      
      // Step 3: 分别计算 intra- 和 inter- 重要性分数
      A^s = Σ_{k=1}^{L_t} A^{st}_k ⊕ Σ_{k=1}^{L_v} A^{sv}_k  // 沿 query 轴求和
      A^c = Σ_{k=1}^{L_t} A^{ct}_k ⊕ Σ_{k=1}^{L_v} A^{cv}_k
      
      // Step 4: 独立 top-K 选择
      M^s = TopK(A^s, K^s)  // 从 intra-modality 角度选 top-K^s
      M^c = TopK(A^c, K^c)  // 从 inter-modality 角度选 top-K^c
      
      // Step 5: 取交集 + 拼接 recent tokens
      M = M^s ∧ M^c   // token 必须在两个维度都重要
      K = (K ⊙ M) ⊕ K[-R:]  // element-wise mask + 拼接 recent tokens
      V = (V ⊙ M) ⊕ V[-R:]
  ```

  **张量维度说明**：
  - A ∈ [0,1]^{L×L}, L = L_t + L_v（text + visual token 总数）
  - A^{st} ∈ [0,1]^{L_t×L_t}, A^{sv} ∈ [0,1]^{L_v×L_v}
  - A^{ct} ∈ [0,1]^{L_v×L_t}, A^{cv} ∈ [0,1]^{L_t×L_v}
  - M^s, M^c ∈ {0,1}^L（binary mask），M = M^s ∧ M^c
  - 实际使用 A[-O:, :-R] 剪裁版本（在 observation window + 最近的 token 范围内计算）

  **跨 self 比率选择**：K^s 和 K^c 的比例根据数据集特征调整。大多数数据集使用平衡比例（50% intra + 50% inter）。特殊数据集：EgocentricNavigation 使用 bias=0.5（多为 inter-attention），SlideVQA 使用 bias=1.5（偏 self-attention）。IR（Image Retrieval）任务 cross_ratio=0.9（90% inter-attention）。

  **n-Softmax 关键公式**：
  原始 softmax：A_i = e^{O_i} / Σ_{j∈I^+∪I^-} e^{O_j}
  剪枝后：A_i = e^{O_i} / Σ_{j∈I^+} e^{O_j}（分母变小 → 分数增大 → 分布变尖锐）
  n-Softmax：A_i = e^{O_i} / (n + Σ_{j∈I^+} e^{O_j})（加偏置 n 恢复平滑性），n=1

  **关键性能数据**（LLaVA-v1.5-7b, RTX 4090）：
  | Budget | Decoding Latency | GPU Mem   |
  |--------|-----------------|-----------|
  | 100%   | 26.023 ms/token | 1.571 GiB |
  | 60%    | 24.377 ms/token | 1.207 GiB |
  | 30%    | 21.027 ms/token | 0.523 GiB |
  | 10%    | 16.287 ms/token | 0.208 GiB |
