## AdaSplash: Adaptive Sparse Flash Attention

- baseline方法是什么？
  Baseline 是标准的基于 softmax 的稠密注意力机制，使用 FlashAttention-2 进行硬件优化的 tiling 和 recomputation。具体而言：
  - attention 概率通过 softmax(s_i) = exp(s_i)/Σ_j exp(s_j) 计算，对所有 token 分配非零概率
  - FlashAttention-2 通过 block-wise tiling 将 Q,K,V 分块加载到 SRAM，online softmax 计算避免 materialize 完整 S ∈ R^{n×n} 和 P ∈ R^{n×n}
  - 反向 pass 利用存储的 O 和 online softmax 的 lse (log-sum-exp) 计算梯度
  - 稠密 attention 的缺陷：(a) 对所有 token 分配非零概率导致 attention 分散 (dispersion)，尤其是长上下文场景下小概率累积会稀释重要 token 的贡献；(b) 无法利用 attention 权重的自然稀疏性（实验表明 ~3% entries 覆盖 96% attention mass）来进一步减少计算

  全栈执行例子（Baseline / FlashAttention-2 softmax attention）：
  - 算法pipeline：QK^T/√d → softmax (dense, 每行和为 1) → PV。softmax 强制输出稠密概率分布，FlashAttention-2 通过 tiling 保证 O(n) memory 但无法减少 FLOPs（始终 O(n²) 的计算复杂度）
  - 系统框架：CUDA/Triton kernel，集成到 PyTorch (torch.nn.functional.scaled_dot_product_attention)，HuggingFace Transformers
  - 编译框架：torch.compile 可用但不适用于 attention 的复杂 memory access pattern；FlashAttention-2 使用手工 CUDA kernel
  - kernel调度：FlashAttention-2 kernel 分块加载 Q,K,V → SRAM compute S → online softmax → rescale O → write back。前向仅需 1 次 K,V 加载，反向用 recomputation 避免 store S
  - 硬件架构：Nvidia H100 (80GB) / RTX A6000 (48GB)，利用 GPU 层级内存（HBM → SRAM）

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ADASPLASH 用 α-entmax 替代 softmax 作为 attention 的概率变换，结合 Hybrid Halley-Bisection 算法和自定义 Triton kernel，实现自适应稀疏注意力在训练时的实际加速。

  **对应解决 Baseline 缺陷的三项核心设计**：

  1. **α-entmax 替代 softmax → 解决 attention 分散和无法利用稀疏性问题**：
     α-entmax 通过参数 α > 1 产生真正稀疏的概率分布（α=1 退化为 softmax, α=1.5 约 95% sparsity, α=2 即 sparsemax 约 99% sparsity）。由 [(α-1)s - τ]_+^{1/(α-1)} 公式可知，score 低于 τ/(α-1) 的 token 获得精确零概率——不仅消除了小概率残差的干扰（解决 dispersion），还创造了可利用的稀疏性。

  2. **Hybrid Halley-Bisection 算法 → 解决 α-entmax 计算本身太慢的问题**：
     α-entmax 需通过迭代求解 τ（f(τ) 的根），传统 bisection 需 23 次迭代且每次需完整遍历 S 导致大量 HBM 读写。Halley-bisection 利用 f 的二阶导数实现 cubic convergence rate，仅需 3 次迭代到 machine precision；且 block 版本在 SRAM 中累积 f/f'/f''，不 materialize S。Fail-safe 机制保证即使 Halley 发散也回退到 bisection，确保最坏情况下仍收敛。结果是 15× 加速（2.38ms vs 36.67ms at n=8192）和 1.75× 内存节省。

  3. **Sparsity-aware Triton kernel (block masking + lookup tables) → 真正利用稀疏性减少计算**：
     FlashAttention-2 虽然可以用 block-sparse 变体，但 mask 必须预先定义，而 α-entmax 的稀疏模式是数据依赖的（dynamic）。ADASPLASH 在 Halley-bisection 最后迭代中动态检测哪些 Q,K block pair 产生非零 P，构造 binary block mask M 和 pointer-increment lookup tables（K_j, Q_i）。后续前向和反向 pass 通过 lookup tables 跳过 null blocks 的 HBM 加载和 GEMM 计算，实现真正的稀疏加速。当稀疏度足够高时，ADASPLASH 的 wall-clock time 可超越 FlashAttention-2（后者 runtime 对稀疏度无反应，始终执行 full computation）。

  全栈执行例子（ADASPLASH α-entmax attention, α=1.5, Triton kernel on H100）：
  - 算法pipeline：QK^T/√d → α-entmax（Halley-bisection 求 τ, 仅 3 迭代）→ 稀疏 P = [(α-1)S-τ]_+^{1/(α-1)}（预测 ~95% zeros）→ PV；训练时 α 从 1.0 线性 anneal 到 1.5（over 1B tokens）确保 dense→sparse 平滑过渡
  - 系统框架：Triton kernel 替代 torch.nn.functional.scaled_dot_product_attention，PyTorch + HuggingFace Transformers（在 attention 层替换 fa2 → adasplash）；训练用 fp16/bf16 mixed precision, AdamW optimizer
  - 编译框架：论文未明确说明（Triton 自身是 JIT-compiled 到 GPU）
  - kernel调度：前向：(1) Halley-bisection block kernel → τ (3 passes over K, 仅需此额外开销) → (2) 构造 M 和 lookup tables → (3) 仅对 M_{ij}=1 的 blocks 计算 O_i += P_i^{(j)} V_j。反向：(1) dK/dV kernel 用 K_j lookup 仅迭代有效 Q_i；(2) dQ kernel 用 Q_i lookup 仅迭代有效 K_j。利用 α-entmax 的稀疏 Jacobian (Diag(u) - uu^T/||u||_1, u = p^{2-α}) 替代 softmax 的稠密 Jacobian
  - 硬件架构：Nvidia H100 (80GB) / RTX A6000 (48GB)；Triton kernel 利用 SRAM (on-chip) 做 block-wise computation；block mask M 为 binary 值，跨 attention 层可共享 memory

  **关键 trade-off**：前向 pass 比 FlashAttention-2 多 ~2 次 K 加载（用于 τ 计算），故在低稀疏度下慢于 FA2；但随着序列变长/稀疏度增加，跳过 null blocks 的收益超过额外 τ 计算开销，最终超越 FA2（Figure 1）。内存复杂度在启用 block masking 时变为 O(n + T_r×T_c)（额外 mask 存储），但仍远小于完整的 O(n²)。

  **实验验证的核心结论**：
  - GPT-2 (124M, 1024 ctx, H100): ADASPLASH 1.03 s/step vs FA2 0.98 s/step — 仅慢 5%，但 Torch bisection (7.78 s/step) 和 sorting (3.61 s/step) 不可用
  - ModernBERT (149M, 8192 ctx, A6000): ADASPLASH 1.53s，超越 Halley-bisection without masking (1.61s)，碾压 Torch bisection (4.99s)
  - 下游任务精度无显著损失：GLUE avg RoBERTa α=1.5 → 83.9 (vs softmax 83.9); ModernBERT α=1.5 → 83.5 (vs softmax 83.7); BEIR nDCG@10 多项超越 dense counterpart
  - GPT-2 α=1.5 validation loss 3.263 (vs softmax 3.283)，HellaSwag 30.6 (vs 30.4)
