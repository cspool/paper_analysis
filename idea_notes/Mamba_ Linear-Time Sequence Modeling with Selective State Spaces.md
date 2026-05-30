## Mamba: Linear-Time Sequence Modeling with Selective State Spaces

- baseline方法是什么？
  - **Transformer（GPT3/LLaMa 风格）**：基于 multi-head self-attention (MHA) 的序列模型。attention 机制通过 QK^T 计算所有 token-pair 的相似度得分，经 softmax 归一化后对 V 加权求和。核心优势是内容感知（content-aware）推理能力强——每个 token 可以"关注"上下文中任意位置的 token。但存在两个根本性缺陷：i) **二次复杂度**：训练 FLOPs = O(BL²D)，推理时需存储 KV cache（每 token 约 2·n_layers·D 个浮点数），自回归生成每步需重读整个 cache，导致 O(L) 时间/步；ii) **有限上下文窗口**：无法建模窗口外的信息，长序列性能受限。
  - **LTI（Linear Time-Invariant）SSM（S4, H3, Hyena 等）**：基于结构化状态空间模型的序列模型。参数 (Δ, A, B, C) 在时间上固定不变，可通过卷积模式（FFT，O(L log L)）做并行训练，或循环模式（O(1)/步）做自回归推理。优点是线性/近线性缩放于序列长度。核心缺陷是**无法进行内容感知推理**（lack of content-based reasoning）：模型动态（Ā, B̄）对所有 token 相同，无法根据当前 token 内容"选择"传播或遗忘哪些信息。这在 Selective Copying（需根据内容决定记住哪些 token）和 Induction Heads（需根据上下文检索相关信息）等任务上暴露为致命弱点。
  - 全栈执行例子（以 Transformer baseline 为例）：
    - **算法层**：输入 token ID → embedding → 逐层 multi-head attention（Q=XW_Q, K=XW_K, V=XW_V → A=softmax(QK^T/√d_k) → O=AV → OW_O）→ FFN/SwiGLU MLP → residual + LayerNorm → LM head → softmax → 采样/argmax 输出下一个 token
    - **系统框架层**：基于 PyTorch + HuggingFace Transformers（GPT3/LLaMa 实现），推理时维护 KV cache 结构（每层存 K, V ∈ R^{B×n_heads×L×d_head}），自回归生成时每步追加新 token 的 K,V 到 cache
    - **编译框架层**：使用 FlashAttention-2 CUDA kernel（tiling + recomputation 将完整 QK^T 矩阵限制在 SRAM 内计算），torch.compile 做图优化
    - **kernel调度层**：FlashAttention-2 将 Q,K,V 分 tile 加载到 SRAM，在线 softmax rescaling 避免将中间 attention matrix 写回 HBM
    - **硬件架构层**：NVIDIA A100 GPU (80GB HBM, 108 SM, 40MB L2 cache, SRAM per SM 192KB)
  - 全栈执行例子（以 LTI SSM — S4/H3 baseline 为例）：
    - **算法层**：输入 x ∈ R^{B×L×D} → 逐通道应用 SSM: 预计算卷积核 K̄ = (CB̄, CĀB̄, ..., CĀ^{L-1}B̄) → y = x ∗ K̄（FFT 加速卷积）或 h_t = Āh_{t-1} + B̄x_t, y_t = Ch_t（循环模式）。参数 (Ā,B̄) 对所有 t 相同
    - **系统框架层**：H3 架构（SSM sandwich：gate → shift-SSM(局部卷积) → SSM → gate），模块间 interleave MLP blocks。PyTorch + FFT convolutions
    - **编译框架层**：论文未明确说明（依赖 PyTorch 默认 FFT/conv 实现）
    - **kernel调度层**：标准卷积 kernel（FFT-based: FFT(x) × FFT(K̄) → IFFT），O(L log L) FLOPs，但无法解决 Selective Copying 任务
    - **硬件架构层**：同 Transformer — NVIDIA A100 GPU
  - Baseline 核心缺陷总结：
    - Transformer: 二次复杂度 → 长序列训练/推理成本高；KV cache → 推理内存线性增长、batch size 受限
    - LTI SSM: 缺乏选择性 → 无法在序列维度上做内容感知的"聚焦/忽略"决策 → 在离散信息密集型数据（文本、DNA）上效果差

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Mamba 方法**通过三项核心创新逐一解决：
    1. **选择机制（Selection Mechanism, S6）**：将 SSM 参数 (Δ, B, C) 从静态改为输入 x 的函数，使模型可以"理解输入内容"后决定传播还是遗忘信息。具体而言：Δ_t = softplus(Parameter + Linear_1(x_t)) 控制"关注当前 vs 保持历史"（大 Δ ≈ 关注当前输入并 reset 状态 → 选择机制；小 Δ ≈ 忽略当前并保持 → 过滤无关信息）。B_t = Linear_N(x_t) 和 C_t = Linear_N(x_t) 提供输入到隐藏状态和隐藏状态到输出的细粒度内容调制。Theorem 1 证明选择机制是经典 RNN gating 的泛化
    2. **硬件感知并行扫描算法**：选择性 SSM 不再是 time-invariant，丢失了卷积形式（FFT）的可用性。若用朴素循环，需物化大小为 (B,L,D,N) 的中间状态 h（比输入大 N=16 倍）。Mamba 通过 kernel fusion（将离散化+扫描+输出计算融合在 SRAM）+ parallel scan（Blelloch 算法，O(L) work O(log L) depth）+ recomputation（反向时重计算 h 而非从 HBM 读取）解决了这一问题，IO 减少 O(N) 倍，实测比 naive scan 快 20–40×
    3. **简化的 Mamba 架构**：将 H3 的 SSM 块和标准 MLP 块合并为同质化单一模块（gate → Conv1d → SiLU → Selective SSM → × gated SiLU → output projection），无需 attention 甚至无需 MLP 块。每个块有 3ED² 参数（E=2 固定），两个 Mamba 块 ≈ 一个 Transformer 块（12D²）
  - 论文方法全栈执行例子：
    - **算法层**：
      输入 token ID → embedding → 逐层 Mamba block →
        x → RMSNorm → Linear (投影到 gate + main 两分支, 2ED 维) →
        gate分支: SiLU → 作为 multiplicative gate
        main分支: Conv1d (kernel=4) → SiLU → 选择性 SSM (S6):
          Δ = softplus(Linear_R(x) + bias) [R=64, 输入投影→D维broadcast]
          B = Linear_N(x), C = Linear_N(x)  [N=16]
          Ā = exp(Δ ⊙ A), B̄ = Δ ⊙ B  [ZOH discretization, fused in SRAM]
          h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t  [parallel scan, fused in SRAM]
          y_ssm_t = C_t ⊙ h_t
        → y_ssm × gate → Linear (投影回 D 维) → residual → RMSNorm →
      → 最后层输出 → LM head → softmax → 采样
    - **系统框架层**：PyTorch + 自定义 CUDA kernel (fused selective scan) + 标准 HuggingFace-style 训练 pipeline（AdamW, BF16, gradient clip）。自回归推理无需 KV cache——每步仅将新的 (h_t, x_t) 送入循环更新，O(1) 时间和 O(DN) 内存/步
    - **编译框架层**：论文未明确说明（使用 CUDA 直接实现 scan kernel，未修改编译器框架）
    - **kernel调度层**：
      Fused Selective Scan kernel 执行流程（per chunk in SRAM）：
        Load: Δ(BLD), A(DN), B(BLN), C(BLN) from HBM [共 O(BLD)]
        SRAM: discretize(Δ, A, B) → Ā, B̄ (BLDN) → parallel scan → h (BLDN) → y = C⊙h (BLD)
        Write: y (BLD) to HBM
      反向: 重新加载输入 O(BLD) → 重计算 h → 计算梯度 → 写回 O(BLD)
      总 HBM IO ≈ 2BLD（vs naive 的 3BLDN，N=16 时节省 16×）
    - **硬件架构层**：NVIDIA A100 GPU。利用 GPU 内存层级（HBM → L2 cache → SM shared memory/SRAM → register），将扫描完全限定在 SRAM 执行避免 HBM 往返。当 L 超过 SRAM 容量时分 chunk 处理（chunk 间通过 HBM 传递 scan state）
  - 关键设计动机映射：
    - Transformer O(L²) 复杂度 → 选择性 SSM 的 O(L) 训练 + O(1) 推理（无需 KV cache）
    - LTI SSM 缺乏内容感知 → 选择机制（Δ, B, C 输入依赖）实现上下文相关的信息过滤/记忆 → 解决 Selective Copying 和 Induction Heads
    - 选择机制破坏卷积可用性 → 硬件感知 fused parallel scan（kernel fusion + recomputation）克服效率瓶颈
    - H3/Transformer 异构架构复杂 → Mamba 同质化简化（H3 + MLP 合一），无 attention、无 MLP 块的极简设计
    - LTI SSM 长上下文不改善（甚至恶化）→ 选择机制天然支持过滤无关上下文 → DNA/音频 1M 长度下性能单调提升
