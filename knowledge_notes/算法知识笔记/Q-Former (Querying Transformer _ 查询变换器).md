## Q-Former (Querying Transformer / 查询变换器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Q-Former 是 BLIP-2 (Li et al., ICML 2023) 提出的轻量级 Transformer 模块，用于在冻结的图像编码器和冻结的 LLM 之间建立桥梁。核心思想是使用一组可学习的固定长度 query vectors（查询向量），通过 cross-attention 从编码器输出中提取/蒸馏最相关的信息，输出固定长度的 soft tokens，再通过线性投影送入 LLM 的语言空间。Q-Former 本质上是一个信息瓶颈（information bottleneck）：它将任意长度的编码器输出压缩为固定数量（如 32 或 64 个）的 condensed tokens，解决了不同模态编码器输出长度不固定、维度过大的问题。

Q-Former 内部结构：N 个标准 Transformer block，每层包含 multi-head self-attention (MSA)、cross-attention (CA) 和 FFN。输入是 learnable query vectors X_Q ∈ ℝ^{M×d}（M 为 query 数量），cross-attention 中 query vectors 作为 query，冻结编码器输出 hidden states 作为 key 和 value。输出是经过 N 层处理后的 refined query vectors，保留了编码器输出中的关键信息。

Uni-MoE 中使用 Audio-QFormer（4 层）和 Speech-QFormer（4 层）分别处理 BEATs 音频编码器输出和 Whisper-small 语音编码器输出，每种配置独立的 learnable query vectors 和线性投影层。

从算法pipeline角度拆解术语：

Q-Former 的单层计算流程（以 Audio-QFormer 为例，式 7-11）：

```
输入: X_Q ∈ ℝ^{AM×d} (AM 个 learnable query vectors, AM 为 query 数)
      h_B = BEATs(audio) ∈ ℝ^{T×d'} (冻结音频编码器输出)

对每层 (共 4 层):
  # Step 1: Self-Attention among query vectors
  h_S = MSA(LN(X_Q)) + X_Q                    # 式(9)
  
  # Step 2: Cross-Attention with encoder output
  #   Query: h_S, Key/Value: h_B
  h_C = CA(LN(h_S), h_B) + h_S                # 式(10)
  
  # Step 3: FFN
  X_Q = MLP(h_C)                               # 式(11)

# 最终: 线性投影到 LLM 空间
A = Linear(X_Q_final)                          # 式(4) 的一部分
```

整个多模态 pipeline 中 Q-Former 的位置：
```
audio → BEATs Encoder (frozen) → h_B → Q-Former (4 layers) → Linear → Audio Tokens → LLM
speech → Whisper-small (frozen) → h_S → Q-Former (4 layers) → Linear → Speech Tokens → LLM
```

与 LLaVA 的对比：LLaVA 使用单个线性投影层连接视觉编码器和 LLM（更简单），而 Q-Former 使用 Transformer 架构的交叉注意力蒸馏（更强的信息提取能力，但参数更多）。Q-Former 适用于需要从长序列编码器输出中压缩信息的场景（如音频、语音），线性投影适用于编码器输出已较紧凑的场景（如 CLIP 的图像特征）。

术语一般如何实现？如何使用？

典型实现基于 BLIP-2 的 Q-Former 架构（HuggingFace Transformers）。在 Uni-MoE 中：(1) 为 Audio 和 Speech 分别初始化独立的 Q-Former（4 层 Transformer），各有独立的 learnable query vectors；(2) 阶段一训练时仅训练 Q-Former 参数和投影层，冻结编码器和 LLM；(3) 学习率 2e-5，AdamW 优化器，cosine scheduler。Q-Former 的训练目标是 cross-entropy generation loss：生成的 text 与 ground truth 之间的交叉熵。

涉及论文标题：
- Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts
