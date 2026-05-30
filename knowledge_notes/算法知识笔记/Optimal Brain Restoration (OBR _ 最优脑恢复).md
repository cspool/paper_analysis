## Optimal Brain Restoration (OBR / 最优脑恢复)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
OBR (Optimal Brain Restoration) 是 ETH Zurich 提出的训练无关统一框架，用于实现 LLM 的联合量化和稀疏化（joint quantization and sparsification），被 ICLR 2026 接收（https://arxiv.org/abs/2509.11177）。核心思想：在剪枝和量化之间通过 Group Error Compensation 计算最优补偿，调和两者对权重分布的冲突需求——量化偏好紧凑数值范围（减少量化误差），剪枝偏好高方差分布（暴露天然稀疏性）。OBR 将权重元素分为 retain set R（鲁棒的）和 eviction set E（易受压缩影响的），通过 Hessian 矩阵作为"桥梁"将 E 的压缩误差转移到 R：Δw_R^* = −H_{RR}^{-1} H_{RE} e_E（闭式解）。使 OBR 成为首个实现 W4A4KV4 + 50% 稀疏度且无需重训练的 LLM 压缩方法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
OBR Pipeline = prune-then-quantize + 两阶段 Group Error Compensation：

```
输入: Hadamard-rotated W ∈ R^{C_out×C_in}, H = 2XX^T, 分区比 α
输出: INT4 + sparse Ŵ

// Step 0: 二阶目标近似
H_full ≈ I ⊗ H  // Kronecker 分解 + 行解耦
→ C_out 个独立子问题: min ½ Δw_i H Δw_i^T

// Step 1: 剪枝（使用 WANDA mask）
M = prune(W),  W^{prune} = W ⊙ M

// Step 2: Group Error Compensation（逐行）
for c = 1 ... C_out:
    // Phase 1: OBR for Pruning
    R₁ = {i | M_{c,i}=1},  E₁ = {j | M_{c,j}=0}
    Δw_{R₁}^{prune} = -H_{R₁R₁}^{-1} H_{R₁E₁} W_{c,E₁}
    w̄ = W_{c,R₁}^{prune} + Δw_{R₁}^{prune}

    // Phase 2: OBR for Quantization
    e^{quant} = w̄ - quantize(w̄)
    E₂ = first α×|R₁| elts,  R₂ = rest (1-α)×|R₁|
    Δw_{R₂}^{quant} = -H_{R₂R₂}^{-1} H_{R₂E₂} e_{E₂}^{quant}

    ΔW_{c,R₁}^{OBR} += Δw_{R₁}^{prune}
    ΔW_{c,R₂}^{OBR} += Δw_{R₂}^{quant}

// Step 3: 量化输出
Ŵ = quantize(W^{prune} + ΔW^{OBR})  // RTN 或 GPTQ
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源: https://github.com/csguoh/OBR (Python + PyTorch + CUTLASS)。128 WikiText2 样本校准计算 H=2XX^T，WANDA 生成 mask，OBR 逐行闭式补偿（α=50% 默认）。预校准模型: https://huggingface.co/HangGuo/OBR。7B 约 2h，70B 约 36h（单 A100）。兼容 QuaRot/SpinQuant/FlatQuant 旋转 + WANDA/SparseGPT/magnitude 剪枝。一次压缩，无限次低代价推理。

涉及论文标题：
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
