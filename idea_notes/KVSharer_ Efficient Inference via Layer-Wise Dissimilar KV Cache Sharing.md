## KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing

- baseline方法是什么？
  Baseline 是全 KV cache 推理（Full KV Cache）和已有的 intra-layer KV cache 压缩方法（H2O、StreamingLLM、PyramidInfer 等）。全栈执行例子如下：
  - **算法层**：Full KV Cache 为每个 Transformer 层的每个 token 保存完整的 Key 和 Value 向量（2 × n_layers × seq_len × d_head × num_kv_heads），KV cache 在推理时占总内存 >80%。已有的 intra-layer 压缩方法（H2O 基于累积 attention scores 保留 heavy-hitter token、StreamingLLM 仅保留 attention sink + 滑动窗口、PyramidInfer 仅对重要 token 计算 KV）均在单层内通过丢弃 token 实现 KV cache 稀疏化。核心缺陷：(a) 所有压缩都在层内（intra-layer），忽略了层间（layer-wise）压缩的潜力——即是否可以不计算某些层的 KV cache 而直接复用其他层的 KV cache；(b) 已有的极少层间 KV cache 压缩工作（MiniCache、LCKV、CLA、YOCO）都需要额外训练模型，无法作为 plug-and-play 方法直接应用于已训练好的 LLM。
  - **系统框架层**：基于 HuggingFace Transformers 推理 pipeline，KV cache 按 autoregressive 方式逐层存储和更新。Intra-layer 压缩方法在 prefill 后/decoding 中执行 KV cache eviction。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 FlashAttention 或 PyTorch 原生 attention 实现。KV cache eviction 在 GPU 上执行 TopK + index gather 操作。
  - **硬件架构层**：NVIDIA A100 80GB GPU。Llama2-13B-Chat 在 512+2048 序列长度下 Full KV cache 内存占用 51639 MB。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  KVSharer 通过一个反直觉的发现——共享不相似的 KV cache 比共享相似的 KV cache 能更好地保持模型性能——提出全新的层间 KV cache 共享策略，全栈执行例子如下：
  - **算法层（核心创新）**：
    (a) **反直觉发现**：传统参数共享/注意力共享方法都基于替换相似的值（相似度越高越好），但 KVSharer 首次发现在 KV cache 的上下文中，共享不相似的层间 KV cache 反而能有效保持模型性能。消融实验（Figure 6）证明按 dissimilarity（欧氏距离降序）共享的 PPL 远低于按 similarity（升序）共享，差距达近 2 倍以上。
    (b) **Strategy Searching 启发式搜索**：在校准数据集（30 句 Wikipedia，每句 64 tokens）上计算任意两层之间 KV cache 的欧氏距离（分别 flatten keys 和 values 为 1D 向量后取平均），按距离降序排列。依次尝试用靠近输入端的层的 KV cache 替换靠近输出端的层的 KV cache，若替换后模型最后一层 hidden state 与原始模型的余弦相似度超过阈值 T=0.5 则保留该替换。不可逆方向设计（仅输出端被输入端替换）是因为靠近输入的层更敏感。
    (c) **非 task-specific**：一次搜索的策略可通用于所有下游任务。
    (d) **正交兼容性**：作为层间压缩方法，与现有 intra-layer 压缩方法（H2O、PyramidInfer）正交兼容，可叠加使用进一步降低内存。
  - **系统框架层**：基于 PyTorch/HuggingFace Transformers 推理 pipeline，即插即用——在 forward pass 中根据共享策略 Z 将被替换层的 KV cache 从前层直接拷贝，不修改模型权重。与 FlashAttention 兼容。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：无自定义 kernel。KV cache 拷贝操作为 GPU 上简单的张量赋值（memcpy），无额外计算开销。prefill 阶段无加速，generation 阶段因跳过部分层的 KV cache 计算实现加速。
  - **硬件架构层**：4× NVIDIA A100 80GB GPU。Llama2-13B-Chat 在 25% 压缩率 + 512+2048 序列下内存从 51639 MB 降至 37049 MB（72%），generation 速度从 18.2 tokens/s 提升至 30.0 tokens/s（1.65× 加速）。叠加 PyramidInfer 后内存降至 30141 MB（58%），generation 速度达 34.1 tokens/s（1.87× 加速）。

  **对比 baseline 的关键差异**：
  - Baseline (intra-layer compression) 仅在单层内丢弃 token → KVSharer 在层间共享 KV cache，完全不计算被替换层的 KV cache，从源头减少 KV cache 计算量
  - Baseline (MiniCache/LCKV/CLA) 需要额外训练 → KVSharer 是 plug-and-play 方法，一次搜索（~60 秒，4×A100）即可应用，无需任何训练
  - Baseline 的相似度共享直觉（替换相似的）→ KVSharer 反直觉地共享不相似的 KV cache，并通过消融实验（Figure 6）证明了 dissimilarity-based 策略显著优于 similarity-based 策略
  - Baseline 层间方法不可与 intra-layer 方法兼容 → KVSharer 与 H2O/PyramidInfer 叠加使用，同时享受层间 + 层内压缩的双重收益
  - 随机共享的 PPL 在 Llama2-13B 上高达 51.41（vs KVSharer 的 9.11），benchmark 下降约 30%，证明并非任意共享都有效——KVSharer 的搜索策略是关键
  - 25% 压缩率下模型性能保持 >90%（Llama2-7B 97.9%、InternLM2-7B 97.0%、InternLM2-20B 97.4%），12.5% 压缩率下 Llama2-7B 甚至提升至 113.6%
