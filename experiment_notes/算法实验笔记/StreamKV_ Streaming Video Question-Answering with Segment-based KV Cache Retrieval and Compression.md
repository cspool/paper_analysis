## StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 StreamKV，一个 training-free 框架，为 Video-LLMs 提供 KV cache 检索与压缩能力。核心算法包括：(1) **语义分段划分**：基于相邻帧 ViT embedding 的 cosine similarity 检测语义边界，配合 exclusion window（最小段长 m）和 segment merging（最大段长 M），动态划分视频流为语义段；(2) **Summary Vector**：每段内逐 spatial location 平均 frame-level features 得到 summary vector，其对应的 KV block 保留不做压缩，用于保留段级语义信息；(3) **Guidance Prompt 驱动的 KV 压缩**：引入 guidance prompt 捕获段内关键语义元素（salient entities、key events、temporal/causal relationships、contextual cues、factual details），用 guidance prompt 的 query vector 作为 selection criterion 选出每段中最 informative 的 KV blocks 保留；(4) **Unified Layer-Adaptive KV Selection Module**：将压缩和检索统一为 per-layer cosine similarity 排序 + 跨层 adaptive budget allocation 问题。每层计算候选 representative key vectors 与 selection criterion 的 softmax-normalized 相似度并按降序排列；通过 binary search 确定全局 cumulative score threshold p，使得跨层累积达到 total budget N 时自适应分配每层选中数量 K_l。

  实验比较：(1) StreamingBench 上 18 个子任务的 VideoQA 准确率，与 ReKV、Dispider、Flash-VStream、VideoLLM-online 等 Online Video-LLMs 及离线 Video-LLMs、闭源 MLLMs (Gemini1.5/ GPT-4o/ Claude3.5) 对比；(2) 不同压缩率下 (0%-90%) 语义分段 vs 均匀分段的性能对比；(3) 有无 summary vector 的性能对比；(4) 压缩和检索分别使用 uniform/adaptive 策略的四象限消融实验；(5) 检索帧数 (0-32 frames) 对准确率的影响对比 (vs ReKV)；(6) 内存使用和推理延迟对比。

- 硬件平台是什么，配置是什么。
  NVIDIA H20 GPU (96GB 显存)，FP16 精度。处理帧率 0.5 FPS，local window size = 15K tokens。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-OneVision-Qwen2-7B-OV 作为基座模型（视觉编码器 + MLP projector + Qwen2-7B LLM）。数据集/Benchmark：StreamingBench，覆盖 18 个子任务分为三大类——Real-Time Visual Understanding (OP, CR, CS, ATP, EU, TR, PR, SU)、Omni-Source Understanding (ACP, CT, All, ER, SCU, SD, MA)、Contextual Understanding (ACU, MCU, SQA, PO)。评测指标：各类子任务准确率及 Overall 准确率。动态分段参数：m=4, M=64 frames，partitioning threshold=0.99。检索帧数 N_r=8。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/sou1p0wer/StreamKV（AAAI-26 接收，当前为 placeholder 状态，代码尚未发布）。MIT License。

  算法流程：
  ```
  # === 语义分段划分 ===
  # 输入：视频帧序列，ViT 编码器提取每帧 embedding f_t ∈ R^{P²×D}
  
  for each adjacent pair (f_{t-1}, f_t):
      s_t = cosine_similarity(f_{t-1}, f_t)  # Eq.(1)
      if s_t < threshold:  # threshold=0.99, 标记为语义边界
          boundaries.append(t)
  
  # 应用 exclusion window (size m=4) 避免过短段
  # 若段长超过 M=64，合并最相似的相邻帧对 (segment merging)
  # 输出：语义段序列 [S^i], S^i = [f_t^i]_{t=1}^{T_i}, T_i ∈ [m, M]
  
  # 每个段的 summary vector: f_s^i = mean(f_t^i) over t  # 逐空间位置平均
  ```

  ```
  # === Segment-based Sliding-window Encoding ===
  # X^i = concat(S^i, f_s^i)  # 段帧 + summary vector
  # 使用 local window L (过去 KV pairs, size=15K) + 当前段 KV 计算 attention
  O = Attn(W_Q X^i, [L_k, W_K X^i], [L_v, W_V X^i])  # Eq.(2)
  
  # 每帧 m 的 KV block: b_m^i = [(k_{m,p}^i, v_{m,p}^i)]_{p=1}^{P²}
  # representative key: r_m^i = (1/P²) Σ_p k_{m,p}^i ∈ R^{D'}  # Eq.(3)
  # 不区分 attention heads，拼接为 D' 维向量
  ```

  ```
  # === Unified Layer-Adaptive KV Selection Module ===
  # 输入: {R_l, c^l}_{l=1}^L (每层候选 representative keys + selection criterion), 总预算 N

  # Step 1: 计算每层每个候选的 cosine similarity
  Sim_l(j) = cos_sim(r_j^l, c^l)  # j ∈ idx(R_l)

  # Step 2: Softmax 归一化 + 降序排序
  ~Sim_l(j) = exp(Sim_l(j)) / Σ_k exp(Sim_l(k))  # Eq.(6)
  priority_l = sort_descending(~Sim_l)

  # Step 3: Binary Search 确定全局阈值 p（Algorithm 1）
  p_1=0, p_2=1
  while p_2 - p_1 > ε:
      p = (p_1 + p_2) / 2
      for each layer l:
          K_l^p = min{k | Σ_{j=1}^k ~Sim_l(s_l(j)) ≥ p}  # Eq.(7)
      if Σ_l K_l^p == N: return p
      elif Σ_l K_l^p < N: p_1 = p
      else: p_2 = p
  # 输出：自适应分配的 {K_l}_{l=1}^L，逐层取 top-K_l 候选为 I_l
  ```

  ```
  # === KV 压缩 (per segment) ===
  # selection criterion: guidance prompt vector g^l = (1/N_g) Σ_k g_k^l
  # 总预算: N = ⌈(1-θ) × T_i⌉ × L  (θ = compression ratio)

  {I_l^i}_{l=1}^L = SelectKV({R_l^i, g^l}_{l=1}^L, N)  # Eq.(9)
  ~B_l^i = [b_m^{i,l} | m ∈ I_l^i]  # 压缩后 frame-level KV blocks
  ~R_l^i = [r_m^{i,l} | m ∈ I_l^i]  # 对应的 representative keys

  # 更新 KV Bank (含 summary KV block b_s^{i,l}, 不参与压缩)
  B_l ← [B_l, ~B_l^i, b_s^{i,l}]  # Eq.(10)
  R_l ← [R_l, ~R_l^i, r_s^{i,l}]
  ```

  ```
  # === KV 检索 (回答问题) ===
  # selection criterion: question vector q^l = (1/N_q) Σ_k q_k^l
  # 总预算: N = N_r × L  (N_r = 期望每层检索帧数, 论文设为 8)

  {I_l}_{l=1}^L = SelectKV({R_l, q^l}_{l=1}^L, N)  # Eq.(11)
  P_l = [B_l[j] | j ∈ I_l]  # Eq.(12) 检索到的 KV blocks

  # 使用检索到的 KV blocks 作为 context 进行 QA
  O = Attn(W_Q X, [C_k, W_K X], [C_v, W_V X])  # Eq.(13)
  # C_k, C_v 包含: 检索到的 KV caches + question + 已生成 tokens
  # RoPE 策略: encoding 阶段仅应用于 local window; QA 阶段基于 relative positions
  ```
