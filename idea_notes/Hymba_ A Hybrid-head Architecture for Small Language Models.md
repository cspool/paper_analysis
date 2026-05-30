## Hymba: A Hybrid-head Architecture for Small Language Models

- baseline方法是什么？
  Baseline 包含三类架构：(1) **纯 Transformer**（Llama 架构）：全部 L 层使用 global causal self-attention，KV cache = O(L×N×d)，内存随层数和序列长度线性增长；(2) **纯 SSM**（Mamba/Mamba2）：所有层使用 state space model，O(1) 常量 cache，但 recall 能力弱（Mamba 300M recall accuracy 仅 19.23% vs Transformer 39.98%）；(3) **Sequential Hybrid**（Samba/Jamba/Zamba）：交替堆叠 Mamba 层和 Attention 层（如 Mamba-FFN-Attn-FFN 重复），但两种层独立处理输入，缺乏协同，信息瓶颈时后续层难以补偿。

  **纯 Transformer (Llama) 全栈执行例子**（Llama3-1B, 8K context, A100）：
  - **算法pipeline**：序列 X → Embedding → L 层 decoder：每层 Masked MHA（QKV 投影 → QK^T/√d → causal mask → softmax → ×V → output proj）→ SwiGLU FFN → 残差连接 → classifier。KV cache = L × N × 2 × d_head × h_kv × 2bytes（FP16），8K 下 Llama3-1B cache ~262MB。Recall accuracy 75.95%（SWDE），commonsense avg 52.82%。Throughput 721.1 tok/s at 8K/bs128（300M scale）。
  - **系统框架**：HuggingFace Transformers + PyTorch。lm-evaluation-harness 评估。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：PyTorch CUDA kernel（标准 FlashAttention 优化），无自定义 kernel。
  - **硬件架构**：NVIDIA A100 GPU，无自定义硬件。

  **Baseline 的核心缺陷**：
  1. **Transformer KV cache 内存爆炸**：KV cache = O(L×N×d)，长序列推理时 HBM 容量成为瓶颈。Llama-1B 8K cache=262MB，而 Hymba 同等规模仅需 79MB。
  2. **SSM recall 能力严重不足**：Mamba 的常量大小 state 无法精确存储和检索历史信息。Mamba 300M recall acc 仅 19.23%（vs Transformer 39.98%），SQuAD-C 仅 36.43%（vs Transformer 75.95%）。Attention sink 问题严重：>50% attention 聚焦于 BOS token（Figure 7）。
  3. **Sequential hybrid 缺乏协同**：Samba 式交替堆叠导致 Mamba 层和 Attention 层独立处理输入，当某一层类型不适合当前 token 的处理需求时，信息瓶颈无法被后续层充分补偿。Samba 1B avg 52.83%，与纯 Llama3 的 52.82% 几乎持平，recall 甚至下降（SWDE 30.00% vs Llama3 75.95%）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Hymba hybrid-head 并行架构**，通过四个核心设计逐一解决 baseline 缺陷：

  **(1) Hybrid-Head 并行融合解决 sequential hybrid 缺乏协同（解决缺陷 3）**
  Baseline Samba 将 Mamba 和 Attention 交替堆叠，每层只执行一种操作。Hymba 在同一层内并行放置 attention heads 和 SSM heads：
  `Y = W_out_proj( β₁·norm(M_attn·X̃) + β₂·norm(M_ssm·X̃) )`
  两者同时处理相同输入，SSM 提供全局上下文摘要（fading memory），Attention 提供高分辨率局部召回（snapshot memory），输出经可学习 per-channel 重缩放 β₁, β₂ 后融合。ERF 分析证明 parallel 结构的有效感受野比 sequential 大一个数量级（Fig. 11），cache size 相当。300M scale 下 hybrid-head avg accuracy 45.19%（+1.12% over sequential 44.07%），recall 49.90%（+4.74% over sequential 45.16%）。

  **(2) KV Cache 优化解决 Transformer 内存瓶颈（解决缺陷 1）**
  - **SSM 摘要全局上下文 → 仅 3 层 global attention**：SSM heads 已经 summarize 了全局 context，因此可以大胆地用 local SWA 替代绝大多数 global attention。仅保留首/中/末 3 层为 global attention 即可恢复 recall 能力。对比实验：全 SWA 时 recall 从 49.90% 骤降至 29.78%；恢复 3 层 global attention 后 recall 回升至 48.79%。
  - **Cross-layer KV sharing**：相邻两层共享同一 KV cache（每 2 层一组），节省参数和 cache。同时将节省的参数重新分配到其他组件，提升 commonsense accuracy +0.60%。
  - **结果**：8K cache size 从 414.7MB（纯 Transformer）降至 39.4MB（10.5× reduction），throughput 从 721.1 tok/s 提升至 2756.5 tok/s（3.8×）。

  **(3) Meta Tokens 解决 Attention Sink 和 Recall 不足（解决缺陷 2 的 recall 部分）**
  Baseline Transformer 中 >50% attention 聚焦于 BOS token（"forced-to-attend"），浪费 attention 预算。Hymba 引入 128 个 learnable meta tokens 前置到输入：
  `X̃ = [r₁, r₂, ..., r₁₂₈, x₁, x₂, ..., x_n]`
  Meta tokens 的作用：(a) 作为 attention sink 的"吸收器"，吸收原本会浪费在 BOS 上的 attention，使后续 token 能关注有意义的信息；(b) 作为 learned cache initialization（推理时离线预计算 K/V/SSM 状态）；(c) 封装压缩的世界知识（不同 domain 的 prompt 激活不同的 meta tokens，Fig. 5）。引入 meta tokens 后：300M recall 从 48.04% 提升至 51.79%（+3.75%），attention map entropy 整体下降（Fig. 15），说明 attention 更集中于信息量大的 token。

  **(4) Attention Map 解耦增强表达能力（辅助解决缺陷 2,3）**
  Hymba 的 attention map 由三部分贡献组成：meta tokens + sliding window attention + SSM（Fig. 6）。相比 Transformer 中 'BOS' 和 'Self' 占比过高的失衡分布，Hymba 的 'Cross' attention（token 间信息交互）比例更高，分布更均衡（Fig. 7）。这意味着 hybrid-head 设计有效解耦了不同类型的信息处理：SSM 关注当前 token（Self），Attention 关注跨 token 关系（Cross），Meta tokens 吸收 attention sink 释放 attention 预算。

  **论文方法全栈执行例子**（Hymba-1.5B, 8K context, A100）：
  - **算法pipeline**：输入 X → prepend 128 meta tokens → X̃ → 32 层 hybrid-head block：每层并行执行 {Sliding Window Attention（仅 3 层为 global）+ Mamba SSM} → β 归一化重缩放融合 → SwiGLU FFN → 残差连接。KV cache：仅 global attention 层存储 + 每 2 层共享（cross-layer sharing）。SSM state：recurrent h_i（常量大小）。Prefill 时 meta tokens 的 K/V/SSM 状态从预计算值加载。8K cache=79MB（vs Llama3-3B 918MB），throughput=664 tok/s（vs Llama3-3B 191 tok/s）。Avg accuracy 61.06%（vs Llama3-3B 59.74%）。
  - **系统框架**：PyTorch + HuggingFace Transformers。训练：128×A100，WSD scheduler。后训练：LMFlow toolkit（FFT → DPO）。HuggingFace 发布 Hymba-1.5B-Base/Instruct。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：PyTorch CUDA kernel（标准 Mamba selective scan kernel + FlashAttention），无自定义 kernel。
  - **硬件架构**：NVIDIA A100 GPU，无自定义硬件。

  关键设计动机映射：
  - Sequential hybrid 缺乏协同 → Hybrid-head 并行融合（同层 attention + SSM，统一对称公式 Eq.3）
  - Transformer KV cache 内存 O(LND) → SSM 摘要全局 + 仅 3 层 global attention + cross-layer KV sharing → O(N + CL)
  - SSM recall 能力弱 → Attention heads 提供高分辨率 snapshot memory 补充 SSM 的 fading memory
  - Attention sink 浪费 >50% attention → Meta tokens 吸收 sink + 作为 learned cache initialization
  - Attention 分布失衡（BOS/Self 主导） → Hybrid-head 解耦：Meta tokens 吸收 BOS，SSM 处理 Self，Attention 处理 Cross
