## MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 MixLoRA，一种基于 LoRA 的 MoE 参数高效微调方法。核心实现：(a) **MoE 构建**：从预训练 dense 模型的 FFN 层构造稀疏 MoE——每个 expert = 共享的冻结 FFN 权重 + 独立 LoRA 适配器（作为 expert 的更新参数存储），替代传统将 LoRA 直接作为 expert 的方式；(b) **Top-K Router**：线性层 + Softmax + KeepTop-2，为每个 token 选择最合适的 2 个 LoRA expert；(c) **负载均衡**：受 Switch Transformers 启发的 auxiliary load balance loss，L_aux = a·N·Σ F_i·P_i，a=1e-2；(d) **Attention 层 LoRA**：额外在 self-attention 的 q,k,v,o 投影上添加独立 LoRA 适配器（非 MoE），提升性能；(e) **MixDoRA**：用 DoRA 替代 LoRA 作为 expert 基础单元的变体。
  - **性能优化**：
    - **(I) 计算复杂度降低**：共享 FFN 的 W1 和 W3 计算结果跨 expert 复用，先将输入送入 W1/W3 做线性投影，再按路由权重切片分发给各 expert 的 LoRA 计算，减少约 30% token 计算延迟。
    - **(II) 多模型高吞吐**：受 m-LoRA 启发，多个 MixLoRA 模型的 multi-task 输入 pack 为单 batch，共享预训练权重，per-model peak GPU memory 降低约 45%。
  - 实验比较：(a) 单任务学习——MixLoRA/MixDoRA vs LoRA/DoRA（r=80），8 个 commonsense reasoning 数据集 accuracy；(b) 多任务学习——混合 ARC/BoolQ/OBQA/PIQA 训练后分别评估，对比 single-task→multi-task 性能退化；(c) 消融：auxiliary loss coefficient a、LoRA rank r、expert load distribution；(d) 计算效率——token compute latency (µs) 和 peak GPU memory (GB)，对比 LoRA/DoRA/vanilla MixLoRA/optimized MixLoRA，含单模型和多模型（×2）场景。

- 硬件平台是什么，配置是什么。
  - 7B 模型：24GB 显存 GPU（RTX 3090, RTX A5000, RTX 4090）。
  - 8B/13B 模型：48GB 显存 GPU（RTX A6000）。
  - 软件栈：Python 3.10, Ubuntu 22.04, x86-64 CPU。
  - 训练精度：half precision（FP16/BF16，论文未细分说明）。
  - 训练超参：cutoff length=512, lr=2e-4, AdamW optimizer, batch size=16, accumulation steps=8, dropout=0.05, epochs=2。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Gemma 2B, LLaMA-2 7B, LLaMA-2 13B, LLaMA-3 8B。
  - MixLoRA 配置：r=16, alpha=32, 8 experts, top-2 router, LoRA 应用于 q,k,v,o（attention）+ w1,w2,w3（FFN expert）。
  - Baseline LoRA/DoRA 配置：r=80, alpha=160, LoRA 应用于 q,k,v,o + w1,w2,w3（控制等量可训练参数）。
  - 数据集（均从 HuggingFace DATASETS 下载）：
    - ARC-e (2250 train / 2380 test), ARC-c (1120 train / 1170 test) — 科学问答
    - BoolQ (9427 train / 3270 test) — 文本分类
    - OpenBookQA (4957 train / 500 test) — 科学事实问答
    - PIQA (16100 train / 1840 test) — 物理交互推理
    - SIQA (33410 train / 1954 test) — 社交交互推理
    - HellaSwag (39905 train / 10042 test) — 句子补全
    - WinoGrande (9248 train / 1267 test) — 填空
  - 评估指标：Accuracy（所有数据集）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/TUDB-Labs/MixLoRA
  - 算法 pipeline 伪代码（MixLoRA 单层 forward，基于 §3.2 公式 5-7 和 Algorithm 1 / Appendix A.7）：

```
输入: hidden states h^{l-1} ∈ R^{B×N×D}  (B=batch, N=seq_len, D=hidden_dim)
      pretrained FFN weights W1,W2,W3 (共享、冻结)
      K 个 LoRA expert: {A_i^{W1}, B_i^{W1}, A_i^{W2}, B_i^{W2}, A_i^{W3}, B_i^{W3}}_{i=1..K}
      每层 router: W_r^l ∈ R^{D×K}
输出: h^l ∈ R^{B×N×D}

// 1. Attention (标准 MSA + LoRA on Q,K,V,O)
z^l = MSA(LN(h^{l-1})) + h^{l-1}
// MSA 中使用 LoRA 修正 Q,K,V,O: W' = W + B·A

// 2. MixLoRA MoE FFN (替代原 FFN)
x = LN(z^l)                              // [B, N, D]

// 2a. Router 计算 (per token)
r = W_r^l · x                            // [B, N, K] logits
r' = KeepTop-2(Softmax(r))               // [B, N, K], 仅 top-2 位置非零

// 2b. [优化] 共享计算：先对全输入做 W1/W3
h_W1 = x · W1^T                           // [B, N, D']  D'=intermediate_dim
h_W3 = x · W3^T                           // [B, N, D']

// 2c. 可选：多模型 batch 模式（Multi-MixLoRA）
// 将来自 M 个 MixLoRA 模型的输入 pack 为一个 batch，共享 W1/W3 计算

// 2d. 逐 expert 计算
h^l = 0                                   // 初始化为零
for k in {1..K}:
    // Expert k 的 LoRA 增量
    h_W1_k = h_W1 + x · (A_k^{W1})^T · (B_k^{W1})^T   // [B, N, D']  W1+LoRA
    h_W3_k = h_W3 + x · (A_k^{W3})^T · (B_k^{W3})^T   // [B, N, D']  W3+LoRA
    // SwiGLU activation
    h_gate = SiLU(h_W1_k) ⊙ h_W3_k                     // [B, N, D']
    // W2 + LoRA
    h_out_k = h_gate · W2^T + h_gate · (A_k^{W2})^T · (B_k^{W2})^T  // [B, N, D]
    // Router 加权累加
    h^l += h_out_k ⊙ r'[:, :, k:k+1]                   // 按 token 的路由权重

// 3. Residual connection
h^l = h^l + z^l

// Training Loss:
L_total = L_CE + a · N · Σ_{i=1}^{N} F_i · P_i
// F_i = 被路由到 expert i 的 token 比例
// P_i = router 分配给 expert i 的概率均值
// a = 1e-2, N = 8 (expert 数)
```

  - **性能优化要点**：
    - 朴素 MixLoRA：每 expert 独立执行 W1·x, SiLU, W2, W3·x 全流程 → 输入序列长时开销大。
    - 优化后：先对全输入计算共享的 W1·x 和 W3·x，再按 expert 切片分发；W2 因依赖 W1/W3 输出无法共享。
    - 多模型模式：M 个 MixLoRA 模型的输入 batch 合并，共享同一份预训练权重，各模型 router 独立路由各自 tokens。训练时 peak GPU memory 从 15.1GB 降至 8.8GB（LLaMA-2 7B + 2 models），推理时从 13.7GB 降至 7.2GB。
  - 单 token 计算延迟（LLaMA-2 7B, µs）：LoRA 245.3, DoRA 659.4, MixLoRA 535.2, †MixLoRA 462.5（优化后降低约 30%）。
