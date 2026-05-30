## SeerAttention-R: Sparse Attention Adaptation for Long Reasoning

- baseline方法是什么？
  Baseline 是标准的 Full Attention（dense decoding），即在长序列推理模型（如 Qwen3, DeepSeek-R1-Distill）的自回归解码阶段，每个新 token 都需要与完整的 KV cache 计算 attention。全栈执行例子（沿一个 decode token 的路径）：

  算法层：Qwen3/DeepSeek-R1 模型使用标准 Multi-Head Attention with GQA。每个 decode step，Q（单 token, multi-head）与完整 K, V（seq_len 长度）计算 scaled dot-product attention，再 softmax，输出 O。复杂度 O(seq_len × d_head) per token，KV cache 大小 O(seq_len × num_kv_heads × d_head)。

  系统框架层：论文未明确说明（baseline 使用标准 PyTorch/HuggingFace Transformers 推理，未涉及 vLLM/SGLang 等 serving 框架的具体配置）。

  编译框架层：论文未明确说明。

  Kernel调度层：FlashAttention-3 (FA3) 的 flash decoding kernel。Kernel 采用 GQA-aware 的 split-KV 策略：沿 (batch, heads_kv, num_splits) 三维 grid launch，每个 SM 负责部分 KV sequence 的 attention 计算，最后 reduce partial results。此为 I/O-bound kernel。

  硬件架构层：NVIDIA H100 GPU，利用 Tensor Core（wgmma 指令）加速矩阵乘法。HBM 带宽为瓶颈。

  Baseline 的核心痛点：
  1. 长推理场景（AIME 平均 11k-18k tokens，max 32k），decode 阶段每 token attention 计算量与 seq_len 成正比，总体生成成本 O(n²)。
  2. 现有训练无关（training-free）稀疏方法（如 Quest）在大 block size（≥64）时准确率显著下降，无法利用粗粒度稀疏块带来的硬件效率优势。
  3. 训练无关方法依赖人为启发式规则，缺乏对 attention 稀疏模式的精确学习。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SeerAttention-R 提出基于自蒸馏 Attention Gate (AttnGate) 的 post-training 块稀疏注意力框架。核心思路：在原始模型注意力层插入轻量级 AttnGate module（可学习 gate），通过自蒸馏训练让 AttnGate 预测哪些 KV blocks 对当前 query 最重要，推理时只计算被选中的 blocks。全栈执行例子（沿一个 decode token 的路径）：

  算法层：
  - AttnGate: Q 分支通过线性层将 GQA group 内 query heads 聚合为 KV-head 数，K 分支通过 Max/Min/Avg 三种 pooling 压缩序列维度，再经线性层和 RoPE 后计算块级 attention 分数 S = softmax(Q_gate @ K_gate^T / sqrt(d_gate))。
  - Shared sparsity: 同一 GQA group 内所有 query heads 共享相同的 sparsity 选择（与 NSA、SAAP 一致），提升硬件效率。
  - 训练（自蒸馏）：只训练 AttnGate 参数（冻结原始模型权重）。修改版 FA2 kernel 同时生成 full attention output 和 1D column-wise maxpooled ground truth，用 KL divergence 训练 AttnGate。仅需 0.4B tokens（OpenR1-MATH-220K），训练效率极高。

  系统框架层：K Compression Cache：为 AttnGate 的 K 分支维护压缩后 K 表示的 cache，每次生成 block_size 个新 token 才更新一次。block_size=64 时 K Compression Cache 仅占原始 KV cache 的 1/128 (<1%)。推理时仅需加载被选中 KV blocks，可结合 KV cache offloading（将完整 KV cache 放 CPU，按需 fetch 选中 blocks 到 GPU）。

  编译框架层：论文未明确说明。

  Kernel调度层：Block Sparse Flash Decoding Kernel（TileLang + Triton 实现）。3D grid launch (batch, heads_kv, num_splits)，仅遍历 selected_block_indices。num_splits 按 max_selected_blocks 分割而非 total_blocks，解决 sparsity 带来的 SM 负载不均衡问题。TileLang 自动应用 tiling、warp specialization、pipelining、tensorization 等优化。在 H100 上，bs=16, seqlen=128k, 90% sparsity 时达 8.6× 加速 vs FA3。

  硬件架构层：NVIDIA H100 GPU，利用 wgmma 指令和 TileLang 自动优化。论文未涉及 RTL 或芯片级修改。

  Baseline 缺陷 → 方法对应的具体设计选择：

  | Baseline 缺陷 | SeerAttention-R 设计 |
  |---|---|
  | Training-free 方法在大 block size 下准确率崩溃 | 自蒸馏 AttnGate 学习精确的 block-level 稀疏模式，block_size=64/128 仍保持 near-lossless 准确率 |
  | 每 token 需计算完整 attention (O(n)) | K Compression Cache + AttnGate 预测选中的 blocks，只计算 O(k) attention（k=token budget） |
  | Training-free 方法无法利用 GQA 共享 sparsity | AttnGate Q 分支聚合 GQA group 内 heads，实现 group 内共享 sparsity |
  | 长序列 KV cache 内存压力大 | K Compression Cache 仅占 <1% KV cache，支持 KV cache offloading |

  Oracle sparsity 实验验证：Qwen3-14B 在 block_size=64、2k token budget 时即可达到 near-lossless 准确率，说明 attention 本身具有内在稀疏性。
