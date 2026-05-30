## Representation_Shift__Unifying_Token_Compression_with_FlashAttention

- baseline方法是什么？
  Baseline 是基于 attention map 的 token 剪枝方法（如 EViT, BAT, Zero-TPrune, vid-TLDR, DynamicViT, AdaViT），在 Vision/Video Transformer 推理时通过 attention scores 评估 token 重要性并剪除低重要性 token。核心依赖 self-attention 计算过程中产生的 attention map 作为 token 重要性的代理信号，例如 EViT/BAT 使用 class token 对 key tokens 的 attention scores（s = Softmax(q_cls K^T/√C)），vid-TLDR 使用 averaged attention across all query vectors（s = (1/N) Σ A_i）。

  Baseline（以 EViT with DeiT-S 为例）全栈执行例子：
  - 算法层：图像 224×224 → 14×14=196 patches + 1 class token → DeiT-S 12层。第3层：每 token 对其余 tokens 的全连接 self-attention → Attention Map A ∈ R^(197×197) → 取 class token row A_cls = A[0, 1:] ∈ R^196 → Softmax → top-K 选保留 tokens → 丢弃其余 → 剩余层处理缩减后的 tokens。问题：(1) attention map 在早期层不可靠，class token 注意力分布尚未收敛（Figure 3/6 显示 early layer attention 近乎随机）；(2) FlashAttention 不暴露 intermediate attention maps（为减少 HBM I/O），使 attention-based 方法完全不兼容。
  - 系统框架层：PyTorch + HuggingFace Transformers。标准 self-attention（非 FlashAttention），因需要访问 attention map。
  - 编译框架层：论文未明确说明。
  - kernel调度层：标准 PyTorch attention（nn.MultiheadAttention）或手动实现，不使用 FlashAttention fused kernel。attention map 必须显式存储在 HBM 中供剪枝使用。
  - 硬件架构层：单 GPU（训练和推理相同），未使用 FlashAttention 加速。

  Baseline 的核心缺陷：
  1. **Attention map 依赖与 FlashAttention 不兼容**：FlashAttention 将 attention 计算融合为单 kernel，避免构建完整 attention map 及写入 HBM，从而大幅加速（DeiT-S 1.5×, UMT-B 2.7× speedup）。但 attention-based token pruning 需要 attention map 来确定 token 重要性，两者不可兼得。这意味着在享受 FlashAttention 加速的同时无法进一步通过 token pruning 降低计算量。
  2. **Early layer attention 信号质量差**：Transformer 前几层 attention map 不可靠（Figure 3 显示 early attention 近乎随机，Figure 6 定性对比 attention vs rep shift 在 L=1 层的差异）。早期剪枝基于 unreliable 信号会错误丢弃重要 token。
  3. **需要额外训练或参数**：DynamicViT、AdaViT、A-ViT 等方法引入额外可学习网络预测 token 重要性，需要 re-training/fine-tuning，不适用于 training-free 场景。
  4. **架构局限**：attention-based scoring 仅适用于 Transformer（需要 self-attention 机制），无法扩展到 CNN 和 SSM 等其他架构。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：Representation Shift——一种训练无关（training-free）、模型无关（model-agnostic）的 token 重要性度量。核心公式：s = Δx = ||F(x) - x||₂，其中 F(·) 为层的变换函数（选定 MLP 层），Δx 量化每个 token 经过该层后的表示变化量。直观理解：对任务关键的 token 会被网络强调（大 representation shift），冗余 token 几乎不变（小 shift）。无需 attention map，与 FlashAttention 完全兼容。

  **(a) 缺陷1：Attention map 与 FlashAttention 不兼容 → 不需要 attention map 的重要性度量**
  Representation shift 计算 token 在 MLP 前后的 L2 距离，完全独立于 attention 机制。由于不需要 attention map，可以在所有层使用 FlashAttention 的同时在特定层（早期层）应用 token pruning。UMT-B 从 32 vid/s（Base, standard attention）加速到 175 vid/s（+FlashAttention + rep shift pruning），相比之下 attention-based pruning（standard attention）仅 57 vid/s。FlashAttention 本身提供约 2.7× speedup，token pruning 在 FlashAttention 基础上再加速约 2×。

  **(b) 缺陷2：Early layer attention 信号质量差 → MLP-based representation shift 在 early layer 更可靠**
  Figure 3/5a 消融实验显示：(i) 基于 Attention 的 representation shift 不如 MLP-based——因 attention 层进行跨 token 信息交换，transformation 更扩散（diffuse），而 MLP 逐 token 独立操作，产生更具判别性的 representation shift；(ii) Figure 6 定性对比中，rep shift 在第1层即成功检测前景物体（"handles foreground object well"），而 attention map 在早期层近乎随机。L2 距离在深度上一致优于 L1 和 cosine（Figure 5b）。

  **(c) 缺陷3：需要额外训练 → 完全 training-free**
  Representation shift 仅需一次前向传播计算 token 的 L2 差，无需任何额外参数或训练。直接应用于预训练模型。Table 5, 6, 7 的实验均无额外训练（CNN 的 fine-tuning 是为了适应 resolution change，非学习 importance scoring）。

  **(d) 缺陷4：仅适用 Transformer → 扩展到 CNN 和 SSM**
  Representation shift 的 "模型无关" 特性使其可计算任何层的输入输出差。CNN（ResNet）：在各 stage 后计算 feature map 变化，通过行/列级剪枝减少分辨率；SSM（Vision Mamba）：替换 ToP-ViM 的激活值基分数为 rep shift。Table 6/7 展示了 CNN/SSM 上 real throughput gain。

  对比 baseline 的全栈执行例子（Representation Shift + UMT-L + FlashAttention, video-text retrieval）：
  - **算法层**：视频 12 frames × 224² → 2352 tokens → Layer 0：FlashAttention（fused SRAM kernel，不暴露 A）→ MLP(LN(x')) → Δ = ||MLP(LN(x')) - x'||₂ → Top-80% → prune 20% → ×3 layers progressive prune → 1204 tokens remaining → 后续 9 层正常 Transformer（FlashAttention）→ text and video embeddings similarity → R@K retrieval。全程无 attention map 依赖。吞吐量 66 vid/s（vs Base 12 vid/s = 5.5×），FLOPs 从 984.6G 降至 478.5G。
  - **系统框架层**：PyTorch + FlashAttention fused kernel（通过 `scaled_dot_product_attention` 或 flash-attn 库）。修改：在指定层（drop_layers）的 MLP 后插入 rep shift 计算 + token pruning 模块；其余层不变。无 Serving 框架修改。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch eager 模式。
  - **kernel调度层**：FlashAttention fused kernel 用作标准 self-attention 后端。Representation shift 仅增加 L2 norm 计算（O(N × C)），开销可忽略（< 1% of total FLOPs）。无自定义 kernel 修改。关键的 kernel 兼容性：FlashAttention 的 SRAM-resident 计算不产生 attention map，而 rep shift 不需要 attention map，两者正交兼容。
  - **硬件架构层**：单 NVIDIA RTX A6000 GPU。FlashAttention 减少 HBM I/O 实现 2.7× speedup，rep shift-based pruning 减少 token 数实现额外 2× speedup，两者叠加总 speedup 5.5×（UMT-L）。

  核心洞察：representation shift 的成功源于一个经验观察——"网络中信息被放大的 token 对任务更重要"。MLP 的逐 token 独立变换使得这一信号的 distinguishability 最优。L2 距离简单但比 cosine（angular）和 L1（robust but less discriminative in deeper layers）更一致。这一发现使 token pruning 首次实现了与 FlashAttention 的 superposition——两种正交加速技术的组合产生乘法级 speedup（1.5× FlashAttention × 2× pruning ≈ 3-5.5×）。
