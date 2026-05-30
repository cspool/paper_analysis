## QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QuIP# 是一种 weight-only 后训练量化（PTQ）方法，由三大技术组件构成：(1) **Randomized Hadamard Transform (RHT) 非相干处理**：用随机符号向量 S_U, S_V 和 Hadamard 矩阵对权重矩阵 W 和 Hessian H 做双边共轭变换——Ŵ = Had(S_U · Had(S_V · W^T)^T)，Ĥ = Had(S_V · Had(S_V · H)^T)——使权重趋于高斯分布（亚高斯），消除离群值，实现 μ-incoherent 保证，复杂度 O(n log n)（相比 QuIP Kronecker 方法的 O(n√n)）；(2) **BlockLDLQ + E8P 格基码书向量量化**：基于 g-block LDL 分解 H = L^T D L，对非相干化后的权重矩阵按 g=8 列分块自适应舍入——Ŵ_k = Q(W_k + (W_{:(k-1)} - Ŵ_{:(k-1)})A_k)——其中 Q 为 2-bit E8P 向量量化器。E8P 基于 E8 格（8 维最高密度球填充，kissing number 最优），通过符号翻转对称性将 2^16 条码字压缩为 2^8 条源码书（仅 1KiB），支持快速推理。高比特下使用 Residual Vector Quantization (RVQ) 逐残差量化（如 4-bit = 2×2-bit E8P, 3-bit = 2-bit E8P + 1-bit E8）；(3) **层间微调**：先在各 Transformer Block 内微调未量化层补偿已量化层（MSE loss），再端到端微调所有未量化参数（layernorms、S_U、S_V、LM head），优化 CrossEntropy loss。sign vectors 以 FP16 存储。<br>算法伪代码：QuIP#(W, H) → IP-RHT: Ŵ,Ĥ,S_U,S_V ← Had(S_U·Had(S_V·W^T)^T) → BlockLDLQ: Ŵ ← Q_blocks(W+(W-Ŵ)(L^T-I)) using E8P codebook → FineTune: Adam optimize S_U,S_V,layernorms per block then end-to-end.<br>推理伪代码（Algorithm 2）：y ← Had(S_V ⊙ x) → y ← decompress_multiply(Ŵ, C, y) → y ← Had(S_U ⊙ y) → output y，其中 decompress_multiply 用 E8P CUDA kernel 从压缩码书解码权重并与激活做 MMA。
  - 实验比较：(a) Llama 1 (7B/13B/30B/65B) 和 Llama 2 (7B/13B/70B) 在 2/3/4 bit 下与 OmniQuant、AWQ、QuIP、AQLM 的 Wikitext2 和 C4 困惑度对比；(b) Llama 2 在 ARC-C/ARC-E/PIQA/WinoGrande 上的 Zeroshot 精度对比；(c) Llama 1/2 的 bit scaling 行为（3-bit 超越理论无损 4-bit 线）；(d) 消融实验：RHT vs Kronecker（QuIP vs QuIP# no FT & no E8）、E8P vs 半整数格 vs D4 格 vs K-Means、有/无微调；(e) Mixtral 8x7B（MoE）和 Falcon 180B 上的泛化性验证；(f) 推理吞吐：RTX 4090 上 2/4-bit Llama 模型生成速度（tok/s）及峰值显存带宽利用率，与 AQLM/FP16 对比。

- 硬件平台是什么，配置是什么。
  - 量化实验：NVIDIA A100 GPU（多卡节点），Llama 2 70B 无微调 <10 GPU-hours，含微调约 100 GPU-hours（不含 Hessian 生成）；Hessian 生成：RedPajama 1T 数据集 6144 条序列 × 模型原生上下文长度（Llama 1=2048, Llama 2=4096）。
  - 推理性能测试：NVIDIA RTX 4090（1TB/s 峰值显存带宽），FlashAttention 库 Llama 实现；A6000 上 QuIP# 吞吐约为 QuIP 的 2 倍。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama 1 (7B, 13B, 30B, 65B)、Llama 2 (7B, 13B, 70B)、Mixtral 8x7B (MoE)、Falcon 180B
  - 数据集/Benchmark：(a) 困惑度：Wikitext2、C4（OPTQ 采样函数）；(b) Zeroshot：LM Eval Harness（Gao et al., 2023）评测 ARC-Challenge、ARC-Easy、PIQA、WinoGrande、BoolQ；(c) 校准/微调数据：RedPajama 1T 数据集，256 训练序列 + 128 验证序列；
  - 量化位宽：2-bit (E8P), 3-bit (E8P 2-bit + E8 1-bit RVQ), 4-bit (2× E8P 2-bit RVQ)
  - 超参数：E8P scale ρ=0.9；微调：Adam optimizer, lr=5×10^-5（权重）/ 5×10^-4（sign vectors for 2-bit），batch size 8（block内）/ 1（端到端），5 epochs（160 steps），序列长度 = 模型原生上下文（70B 端到端用 3072 避免 OOM）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/Cornell-RelaxML/quip-sharp；预量化模型：https://huggingface.co/relaxml
  - 张量计算流程（以 Llama 2 7B 2-bit QuIP# 为例）：
    1. **Hessian 生成**：从 RedPajama 采样 6144 条序列，对每个线性层计算代理 Hessian H = E_x[xx^T]。
    2. **IP-RHT 非相干处理**（Algorithm 3）：采样随机符号向量 S_V ∼ U{±1}^n, S_U ∼ U{±1}^m → Ŵ ← Had(diag(S_U) · Had(diag(S_V) · W^T)^T)，其中 Had 为 Fast Walsh-Hadamard Transform（O(n log n)，仅 ±1 运算无浮点乘法）→ Ĥ ← Had(diag(S_V) · Had(diag(S_V) · H)^T)。输出 Ŵ, Ĥ, S_U, S_V。对非 2 的幂次维度：分解 n = p × q（p 为最大 2 的幂次，q 已知 Hadamard 矩阵存在），使用 V = H_p ⊗ H_q。
    3. **BlockLDLQ 自适应舍入**（Section 4.1）：对 Ĥ 做 g-block LDL 分解 Ĥ = L^T D L → 设置 U = L^T - I → 按 g=8 列分块迭代舍入 Ŵ_k = Q_E8P(Ŵ_k + (Ŵ_{:(k-1)} - Ŵ̂_{:(k-1)}) · A_k)，其中 A_k 为 U 的第 k 个 8 列块，Q_E8P 为 E8P 2-bit 向量量化。
    4. **E8P 解码**（Section 4.2）：每个 16-bit 码字编码一个 8 维向量——8 bits 查源码书 S（256 个 |D̂_8| 绝对值条目），7 bits 控制 7 个符号翻转（第 8 个符号由奇偶性推断），1 bit 控制 ±1/4 偏移。解码：c = S[code[0:8]] → 符号翻转 parity 恢复 → v ∈ E8 + 1/4。
    5. **RVQ 高比特扩展**（Section 4.3）：4-bit 量化 = 两次 2-bit E8P 残差量化——δ_1 = Q_E8P(Ŵ) · s_1, δ_2 = Q_E8P((Ŵ - δ_1)/s_2) · s_2, Ŵ̂ = δ_1 + δ_2。
    6. **层间微调**（Algorithm 5）：对每个 Decoder Block D ∈ M：Y ← D(X) → 对每层 L ∈ D 按顺序量化 → 冻结 L 的量化权重 → Adam 优化 D 以最小化 MSE(D(X_train), Y_train) → X ← Y。全部 Block 完成后，端到端微调剩余参数（layernorms, S_U, S_V, LM head），最小化 CrossEntropy(M(D_train), C_train)，使用验证集早停。
    7. **推理**（Algorithm 2）：输入激活 x → y ← Had(S_V ⊙ x)（FWHT）→ y ← E8P_decode_matvec(Ŵ, C, y)（CUDA kernel, MMA Tensor Core 指令）→ y ← Had(S_U ⊙ y) → 输出。
