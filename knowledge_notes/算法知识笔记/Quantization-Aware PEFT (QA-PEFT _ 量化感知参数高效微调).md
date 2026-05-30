## Quantization-Aware PEFT (QA-PEFT / 量化感知参数高效微调)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization-Aware PEFT (QA-PEFT) 是将模型量化与参数高效微调（PEFT）结合的范式：先对预训练 LLM 权重做低比特量化（如 4-bit/3-bit/2-bit），然后在量化模型上通过 PEFT 适配器（如 LoRA、稀疏适配器等）进行下游任务微调。与传统 PEFT 不同的是，QA-PEFT 在初始化阶段需要显式补偿量化误差：将适配器初始化为近似量化误差 ΔW_Q = W_0 - W_Q 的某种形式（低秩近似或稀疏近似），使初始输出接近全精度模型，再通过微调进一步恢复精度。核心目标函数为最小化层输出误差：min ||ΔW_Q X - ΔW_adapter X||_F^2，其中 X 为校准集激活。与标准的 PEFT（适配器从随机/零初始化）不同，QA-PEFT 的"量化感知初始化"在 sub-4-bit 场景下尤为关键，因为仅靠微调无法完全恢复极端量化带来的精度损失。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QWHA 中的完整 QA-PEFT pipeline：

```
# Phase 1: Quantization
W_Q = GPTQ_MagR(W_0)  # 4/3/2-bit 量化，group_size=64
ΔW_Q = W_0 - W_Q       # 量化误差

# Phase 2: Calibration (收集激活统计)
calib_set = WikiText2.sample(128 sequences × 2048 tokens)
for X in calib_set:
    accumulate X@X^T  # 外积累积
XX^T = avg(X@X^T)
U, Σ, V^T = SVD(XX^T)
R = U @ sqrt(Σ)        # Hessian 平方根
B = H^{-1} @ R         # WHT 预投影

# Phase 3: Adapter Initialization (minimize ||ΔW_Q·R - F·B||_F^2)
p_i = AdaAlloc(ΔW_Q, X, p)  # 通道级参数分配
for each channel i:
    v = (ΔW_Q)_{i,:} @ R
    E_i = TopK(|v @ B^{-1}|, p_i)   # 选最大系数位置
    c_i = v @ B'^T @ inv(B' @ B'^T) # Refinement
F = Scatter(c, E)

# Phase 4: Fine-tuning (仅更新稀疏矩阵 F 中的非零值 c)
for epoch in 1..3:
    for X, y in Alpaca:
        ΔW = F @ H^{-1}    # 前向：WHT 展开
        Y = (W_Q + α·ΔW) @ X
        loss = CE(Y, y)
        c -= lr * ∂loss/∂c  # 仅更新 c，E 和 H 固定

# Phase 5: Inference
Y = W_Q @ X + α·(F @ (H^{-1} @ X))
```

QA-PEFT 与 PTQ (Post-Training Quantization) 的关键区别：PTQ 量化后不做微调，直接评估；QA-PEFT 利用少量训练数据通过适配器补偿量化误差并适应下游任务。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QA-PEFT 的演进路线：(1) QLoRA (Dettmers et al., 2023) — 首次将 4-bit NF4 量化与 LoRA 结合，但 LoRA 从零初始化；(2) LoftQ (Li et al., 2024) — 迭代 SVD 分解量化误差初始化 LoRA；(3) LQ-LoRA (Guo et al., 2024) — 低秩加量化矩阵分解；(4) CLoQ (Deng et al., 2025) — 校准的 LoRA 初始化，最小化层输出误差；(5) QWHA — 首次将 FT-based adapter (WHA) 引入 QA-PEFT，用 WHT 替代低秩结构，实现 full-rank 适配器 + 量化感知初始化（AdaAlloc + Refinement）。所有方法均使用 GPTQ 作为底层量化方案，适配器应用于所有线性层（Q/K/V/O/Gate/Up/Down projections）。校准集通常使用 WikiText-2（128-256 条序列），因其与微调数据的独立性。

涉及论文标题：
- QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning
- RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models

在 RoSTE 中，QA-PEFT 并非其直接方法（RoSTE 不使用 PEFT adapters 如 LoRA），但 QA-PEFT 是其重要的 baseline 对比对象。RoSTE 对比了 QLoRA（4-bit NF4 + LoRA）作为 QA-PEFT baseline：在 Pythia 6.9B W4A4KV4 下 QLoRA (r=64) ROUGE Avg=20.20 vs RoSTE 23.66（Table 7），在 Llama 3.1 8B 实验中 QLoRA 未直接对比但 STE 方法可视为 QLoRA 的无 adapter 变体。RoSTE vs QA-PEFT 的核心区别：(1) QA-PEFT 在量化后通过适配器补偿量化误差，权重保持冻结；(2) RoSTE 直接在量化约束下训练所有权重（full QAT），不使用额外适配器参数量，而是通过自适应旋转矩阵优化量化配置。
