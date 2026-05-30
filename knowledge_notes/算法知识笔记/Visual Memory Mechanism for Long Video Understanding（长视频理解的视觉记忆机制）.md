## Visual Memory Mechanism for Long Video Understanding（长视频理解的视觉记忆机制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Visual Memory Mechanism（视觉记忆机制）是 FlexMem（CVPR 2026）提出的一种将 MLLM 的长视频理解建模为"人类观看视频"过程的训练无关方法。核心思想：MLLM 不应一次性处理所有视频帧（会导致输入上限和计算爆炸），而应像人类一样——持续观看视频内容、将关键信息压缩为视觉记忆（visual memories）、在问答时召回最相关的记忆片段来生成答案。FlexMem 的视觉记忆机制包含三个核心子模块：(1) **记忆编码（Memory Encoding）**——通过 Dual-Pathway Compression 将每个视频 clip 的视觉 KV cache 压缩为 context memory（用于跨 clip 信息传递）和 local memory（用于最终召回），local memory 写入 Visual Memory Bank；(2) **记忆存储（Memory Storage）**——Visual Memory Bank 持久存储所有 clip 的压缩 visual KV cache，每 clip 内存固定，总内存随 clip 数线性增长；(3) **记忆召回（Memory Recall）**——问答时从 M_bank 中根据 clip-问题相关性选择最相关的 na 个连续 clip 的 memory，仅将这些片段输入 MLLM 解码答案。该机制理论上支持无限长视频处理，结合了 RAG 方法（精确定位关键片段）和视觉压缩方法（保持全局理解）的优势。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FlexMem 的 Visual Memory Mechanism 完整 pipeline（以 LLaVA-Video 7B 为 backbone）：
```
# === Visual Memory Mechanism Pipeline (FlexMem) ===
# 输入: 长视频 V={I1,...,I_T}, 问题 Tq, MLLM backbone

# Step 1: 视频分片
clips = uniform_split(V, clip_size=8)  # N clips
M_bank = []  # Visual Memory Bank

# Step 2: 首次编码
KV_1 = MLLM.forward(clips[0], Tq_opt)   # 可选传入Tq
C_1 = context_compress(KV_1, alpha_c)    # context memory (传递)
M_1 = local_compress(KV_1, alpha_s)      # local memory (存储)
M_bank.append(M_1)

# Step 3: 迭代编码
for i in 2..N:
  ctx = [C_{i-ns}, ..., C_{i-1}]        # 前序context memory
  long_term = optional_recall(M_bank)    # 可选长期记忆
  KV_i = MLLM.forward(long_term + ctx + clips[i], Tq_opt)
  C_i = context_compress(KV_i, alpha_c)
  M_i = local_compress(KV_i, alpha_s)
  M_bank.append(M_i)

# Step 4: Memory Recall
g = [sum_attention(Tq → Vi, layers 3..L) for Vi]  # relevance scores
recalled = M_bank[topK_continuous(g, na)]           # 选na个连续clip

# Step 5: 解码
Y = MLLM.decode(recalled, Tq)  # 仅用召回片段
```
核心特点：(a) encoding 阶段逐 clip 处理，每步固定计算量；(b) 解码 token 数固定（13k / 7k），不随视频长度增长；(c) Tq 在 encoding 阶段可选——传入时可用 encoding-based reading，不传入时用 MemIndex。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlexMem 以 Python 实现，基于 HuggingFace Transformers，作为 LLaVA-Video 和 LLaVA-OneVision 的 plug-and-play 模块。核心超参数：每 clip 8 帧、压缩比 α_c/α_s、context memory 窗口 ns、总采样帧 512 或 1024。在单 RTX 3090 24GB 上可处理 1024+ 帧（vs baseline 64 帧），24GB 受限下仅损失 0.5% 性能。代码开源：https://github.com/city1517/FlexMem。

涉及论文标题：
- Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism
