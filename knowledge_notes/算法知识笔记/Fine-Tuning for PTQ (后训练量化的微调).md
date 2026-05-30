## Fine-Tuning for PTQ (后训练量化的微调)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fine-Tuning for PTQ 是 AQLM (Egiazarian et al., 2024) 引入、QuIP# 进一步改进的一种 PTQ 与 QAT 的混合方法。纯 PTQ 仅基于校准数据优化单层量化参数（scale、rounding），忽略层间交互导致的激活误差累积。Fine-Tuning for PTQ 在量化后对未量化参数进行小规模微调（"adapting"而非"training from scratch"），以恢复层间保真度。QuIP# 的微调分两阶段：(1) Per-Transformer-Block 微调——在每个 Block 内，冻结已量化的线性层权重，Adam 优化未量化层（后续的线性层、layernorm）和 sign vectors S_U, S_V，最小化 Block 输出 MSE；(2) 端到端微调——所有 Block 量化后，优化 layernorms、所有 S_U/S_V、LM head，最小化 CrossEntropy(量化模型输出, 全精度模型 logits)。关键创新：(a) sign vectors 以 FP16 存储（非 bitvector），允许梯度优化——这让 incoherence processing 可以"学习"如何最好地旋转权重以匹配 E8P 码书形状；(b) 微调数据量极小——256 条 RedPajama 序列训练 + 128 验证，5 epochs；(c) 计算成本低——70B 模型约 50 GPU-hours（对比 LLM-QAT 960 GPU-hours 仅生成训练数据）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QuIP# Fine-Tuning 流程（Algorithm 5）：
```
# 输入: 未量化模型 M, Dev set D, 量化顺序 O
# === Phase 1: Per-Block Fine-Tuning ===
X = M.embedding(D)                # token embeddings
for each Decoder Block in M:
    Y = Block(X)                  # FP32 block 输出 (教师)
    X_train, Y_train, X_valid, Y_valid = split(X, Y)
    for each Linear Layer L in Block (按 O 的顺序):
        L_hat = QuIP#-NoFT(L)     # 量化当前层
        L.weight.requires_grad = False  # 冻结量化权重
        # 优化 Block 内参数以最小化 MSE
        optimizer = Adam([Block.unquantized_params, L.S_U, L.S_V], lr=5e-5)
        for epoch in range(5):
            loss = MSE(Block(X_train), Y_train)
            loss.backward(); optimizer.step()
        # 用验证集早停选择最优参数
    X = Y                          # 下一 Block 的输入

# === Phase 2: End-to-End Fine-Tuning ===
# 所有 Block 量化完成，仅 layernorms + S_U + S_V + LM head 可训练
D_train, C_train, D_valid, C_valid = split(D, full_model_logits)
optimizer = Adam([layernorms, all_S_U, all_S_V, lm_head], lr=5e-5)
for epoch in range(5):
    loss = CrossEntropy(M(D_train), C_train)
    loss.backward(); optimizer.step()
```
微调效果：2-bit 模型受益最大（Llama 2 7B: 8.22→6.19 WikiText2），3/4-bit 也有改善。消融显示 FT 改善独立于 E8P 和 RHT 的改善（三组件叠加增益）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 每个 Block 内各层量化顺序 O 可影响最终质量，论文未深入讨论顺序优化（likely sequential by layer position）；(2) 对 2-bit 模型，S_U/S_V 的学习率提高到 5e-4（vs 5e-5），因为在极低比特下 sign vector 的旋转自由度更关键；(3) 70B 端到端微调时序列长度从 4096 降到 3072 以避免 OOM；(4) 开源：https://github.com/Cornell-RelaxML/quip-sharp 提供完整微调脚本；(5) 微调仅需小规模通用文本（RedPajama），无需任务特定数据，保持 PTQ 的数据高效优势。

涉及论文标题：
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks
