## Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism (FlexMem)

- 属于算法pipeline的实现是什么？实验比较什么？
  FlexMem 是一种训练无关（training-free）的视觉记忆机制，通过双路径压缩（Dual-Pathway Compression, DPC）对 MLLM 的视觉 KV cache 进行迭代式压缩和记忆管理，将压缩后的 local memory 写入 memory bank，并在问答时通过记忆召回（memory recall）选出最相关片段用于解码解答。实验比较了：(1) 与 VideoRAG 方法（AKS）和视觉压缩方法（AdaRETAKE、Video-RAG、BOLT、Panels、DToMA）在相同 MLLM backbone 下的性能；(2) 与 SOTA Video-MLLM（GPT-4o、Gemini-1.5-Pro、Qwen2.5-VL、Video-XL、TSPO、LongVU 等）的跨模型性能；(3) 在 OVOBench streaming QA 场景下使用 MemIndex 的在线 vs 离线性能对比；(4) 消融实验验证双路径压缩策略、context memory 作用、memory reading 策略以及每 clip 帧数的影响；(5) 在单张 3090 24GB 显存受限条件下与 AKS 和 AdaRETAKE 的可扩展性对比。

- 硬件平台：单张 NVIDIA RTX 3090 GPU (24GB) 为主实验平台；部分对比实验在一张 NVIDIA A800 上运行（Table 1 中 FlexMem* 标记）。所有受限实验均在 24GB 显存上限下完成。

- 模型：LLaVA-Video 7B（13k input tokens）和 LLaVA-OneVision 7B（7k input tokens）作为 base MLLM。数据集和 benchmark：MLVU（多任务长视频理解，含 single-detail/multi-detail/holistic 子类）、LongVideoBench（长上下文多模态推理，视频最长 1 小时）、LVBench（极端长视频理解，平均时长 68.4 分钟）、Video-MME（短视频/中视频/长视频混合）、TimeScope（1 分钟到 8 小时超长时间跨度）、OVOBench（在线流式视频问答，含 EPM/ASI/HLD backward tracing 子任务）。

- 开源情况：已开源，代码地址 https://github.com/city1517/FlexMem 。FlexMem 为 training-free 即插即用方法，直接应用于任意 Video-MLLM 无需额外训练。

- 算法 pipeline 详细解释（基于论文 Section 3 Method）：

  **核心思想**：模拟人类看视频行为——持续观看、形成记忆、问答时召回相关记忆片段。迭代处理视频分片，理论上支持无限长视频。

  **步骤 1 — 视频分片与帧采样**：
  将长视频 V 均匀分片为 N 个 clips {V1, ..., VN}。每 clip 固定 8 帧（消融实验确定），总采样帧数 512（TimeScope/LVBench/MLVU）或 1024（Video-MME/LongVideoBench）。对比 baseline 仅 32 或 64 帧均匀采样。

  **步骤 2 — 首次编码（Clip V1）**：
  ```
  Input: V1 (8 frames → visual encoder → KV caches), optional Tq
  Output: M1 (local memory → M_bank), C1 (context memory → next iter)
  MLLM(V1, <Tq>) → M1, C1
  ```

  **步骤 3 — 迭代编码（Clip Vi, i ≥ 2）**：
  ```
  MLLM(<Ml>, C_{i-ns}, ..., C_{i-1}, Vi, <Tq>) → Mi, Ci
  ```
  输入包括：长期记忆 `<Ml>`（从 M_bank 召回，可选）+ ns 个前序 context memory（确保时序连续性）+ 当前 clip Vi + 可选 Tq。输出 Mi（写入 M_bank）和 Ci（传给下一轮）。

  **步骤 4 — Dual-Pathway Compression（每步核心张量计算）**：

  在 MLLM 每层 l，计算当前 clip Vi 对历史 context C 的 cross-clip attention：
  ```
  A_v^l = softmax(Q_{Vi} · K̂_C^T / √d)  ∈ R^{|Vi| × |C|}
  ```

  **(a) Context Memory 路径** — 选择最能聚合历史信息并传播给后续 token 的 KV（服务于 prefill 阶段）：
  ```
  s_j^l = Σ_{k∈C} a_{jk}^l + Σ_{h∈Vi} a_{hj}^l
  ```
  第一项：token j 从历史 context 聚合的信息量。第二项：token j 对后续 token 的因果传播量。
  ```
  c_i^l = {k_j^l, v_j^l | j ∈ argtopK(s_j^l, α_c·|Vi|)}
  ```
  Ci = {c_i^1, ..., c_i^L}——所有 L 层的 context features，用于下一轮迭代的信息传递。

  **(b) Local Memory 路径** — 选择 clip 内最具显著性的 token（服务于 decoding 阶段）：
  ```
  ŝ_j^l = Σ_{k∈Vi} a_{kj}^l
  ```
  仅考虑 clip 内部影响力。
  ```
  m_i^l = {k_j^l, v_j^l | j ∈ argtopK(ŝ_j^l, α_s·|Vi|)}
  ```
  Mi = {m_i^1, ..., m_i^L}——存储到 M_bank 供最终召回。

  **DPC 设计动机**：prefill 阶段目标是将当前 clip 编码进丰富历史上下文（需要 context aggregation），decoding 阶段目标是基于最显著的视觉证据回答问题（需要 local saliency）。两个阶段对 KV cache 的需求不同，因此用两种不同的重要性度量分别压缩。

  **步骤 5 — Memory Recall（记忆召回）**：

  **方式 A: Encoding-based Reading**（精度高，需 encoding 时传入 Tq）：
  ```
  g_i = Σ_{l=3→L} Σ_{j∈Tq} Σ_{k∈Vi} a_{jk}^l
  Recall(M_bank, Tq) = {Mi | g_i ∈ argmax, 取 top na 连续 clip}
  ```
  仅取第 3 层及以后深层 attention（浅层 attention 分布均匀无区分力）。

  **方式 B: MemIndex（快速索引，独立于 memory encoding）**：
  目标——线性回归拟合 encoding-based relevance：
  ```
  min_σ Σ ||σ(r̂_i) - g_i||², σ(r̂_i) = Σ_{l=3→L} α^l·r̂_i^l
  ```
  - 问题编码：q = Q_{Tq}[-1]（最后一个 token 的 query embedding）
  - 视觉索引：从 Mi 取 top k 个 attention-selected key vectors K_{Vi}^*
  - 在选定 H 层上计算点积 attention：r̂_i = Σ_{l∈H} Σ_j Attention(q, k_j^l)
  - 学习层权重 α^l，选 top K=3 层计算最终 relevance

  MemIndex 独立于 memory encoding，适合多问题或 streaming 场景。

  **步骤 6 — 答案解码**：
  ```
  MLLM(M_i, ..., M_{i+na-1}, Tq) → Y
  ```
  仅用召回的 na 个 memory 片段 + Tq 解码。LLaVA-Video 用 13k tokens，LLaVA-OV 用 7k tokens，远小于 AdaRETAKE 的 40k。

  典型性能：单 3090 上 FlexMem+LLaVA-Video 在 TimeScope 超 baseline 32.2%，LVBench 超 19.7%。512/1024 帧采样 vs baseline 64 帧，五个 benchmark 全面领先 AKS 和 AdaRETAKE。24GB 受限下仅损失 0.5% 性能。
