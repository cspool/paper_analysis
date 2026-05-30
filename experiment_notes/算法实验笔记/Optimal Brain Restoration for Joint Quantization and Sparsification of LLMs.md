## Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 **OBR (Optimal Brain Restoration)**——一个训练无关（training-free）的统一框架，通过 Group Error Compensation 在剪枝和量化之间计算最优补偿，调和两者对权重分布的冲突需求。完整流程为：(1) Hadamard Rotation 平滑权重 outliers → (2) 剪枝生成稀疏 mask → (3) **OBR for Pruning**：将剪枝损失的信息从 pruned elements (eviction set E₁) 通过 Hessian 传播补偿到 unpruned elements (retain set R₁)，即 Δw_{R₁}^{prune} = -H_{R₁R₁}^{-1} H_{R₁E₁} w_{E₁} → (4) **OBR for Quantization**：将 unpruned elements 按比例 α 分为 eviction set E₂（前 α 比例）和 retain set R₂，补偿量化误差，即 Δw_{R₂}^{quant} = -H_{R₂R₂}^{-1} H_{R₂E₂} (w̄_{E₂} - quant(w̄_{E₂})) → (5) RTN/GPTQ 量化得到最终 W4A4KV4 + 50% sparse 权重。
  - 实验比较：OBR_RTN 和 OBR_GPTQ vs. **QuaRot (quant-only)** W3A4KV4 baseline、**QuaRot+WANDA** (naive combination)、**SparseGPT+GPTQ** (strong joint pruning+quantization baseline)。指标为 WikiText2 perplexity 和 PIQA/BoolQ/HellaSwag/ARC-easy/ARC-challenge/WinoGrande 零样本准确率。额外比较：(a) 不同 bit-width (W4A8KV8, W4A16KV16)；(b) SpinQuant 和 FlatQuant 旋转矩阵；(c) 2:4/4:8 半结构化稀疏；(d) BitNet-2B-4T 对比；(e) 纯剪枝/纯量化单任务扩展；(f) 不同 calibration 数据集 (C4)。INT4 2:4 sparse GEMM kernel 在实际 GPU 上的 latency/FLOPs/TOPS。

- 硬件平台是什么，配置是什么。
  - NVIDIA A100-SXM4-80GB GPU（模型压缩 calibration）。INT4 2:4 sparse GEMM kernel 效率测试在同一 A100 GPU 上进行（利用 Ampere 架构的 native INT4 sparse GEMM 支持）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：**Llama2** (7B/13B/70B)、**Llama3** (8B/70B)、**Qwen2.5-Instruct** (7B/32B)、**Qwen2.5-Instruct** (1.5B/3B，BitNet 对比)。
  - Calibration 数据集：128 samples from **WikiText-2**，sequence length 2048（默认）；也测试 **C4** calibration。
  - 评估数据集/bench：**WikiText-2** test set (perplexity)；**PIQA、BoolQ、HellaSwag、ARC-Easy、ARC-Challenge、WinoGrande**（零样本常识推理，使用 lm-eval-harness）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：GitHub **https://github.com/csguoh/OBR**，HuggingFace **https://huggingface.co/HangGuo/OBR**。LICENSE：QuaRot backbone Apache 2.0，SpinQuant backbone CC-BY-NC 4.0，FlatQuant backbone MIT。
  - 软件环境：Python 3.10, PyTorch, HuggingFace Transformers (Qwen2.5 需 4.45.0), fast-hadamard-transform, CUTLASS (kernel)。
  - OBR 算法 Pipeline 伪代码：

```
输入: Hadamard-rotated 权重矩阵 W ∈ ℝ^{C_out × C_in}, 
      Hessian 近似 H = 2XX^T ∈ ℝ^{C_in × C_in}, 分区比例 α
输出: 低比特稀疏权重 Ŵ ∈ ℤ^{C_out × C_in}

// Step 1: 剪枝
M ∈ {0,1}^{C_out × C_in} = prune(W)    // 使用 WANDA/SparseGPT/magnitude mask
W^{prune} ← W ⊙ M

// Step 2: OBR 误差补偿
ΔW^{OBR} ← 0
for c = 1 ... C_out do                    // 逐行处理 (row-wise decoupling)
    // === OBR for Pruning ===
    R₁ ← {i | M_{c,i} = 1}              // unpruned indices
    E₁ ← {j | M_{c,j} = 0}              // pruned indices
    b₁ ← H_{R₁E₁} · W_{c,E₁}^T          // 从 Hessian 提取子矩阵×剪枝权重
    Δw_{R₁}^{prune} ← -H_{R₁R₁}^{-1} · b₁  // 闭式解: 补偿到 unpruned slots
    w̄ ← W_{c,R₁}^{prune} + Δw_{R₁}^{prune} // 补偿后的稀疏权重

    // === OBR for Quantization ===
    e^{quant} ← w̄ - quantize(w̄)          // 量化误差向量
    t ← ⌊α · |R₁|⌋                        // 按照比例 α 切分
    E₂ ← {r₁, ..., r_t}                  // 前 α 比例 → eviction set
    R₂ ← {r_{t+1}, ..., r_{|R₁|}}        // 剩余 1-α → retain set
    b₂ ← H_{R₂E₂} · e_{E₂}^{quant}^T
    Δw_{R₂}^{quant} ← -H_{R₂R₂}^{-1} · b₂  // 闭式解: 补偿量化误差

    // 合并补偿
    ΔW_{c,R₁}^{OBR} += Δw_{R₁}^{prune}
    ΔW_{c,R₂}^{OBR} += Δw_{R₂}^{quant}
end for

// Step 3: 量化
W^{quant} ← W^{prune} + ΔW^{OBR}
Ŵ ← quantize(W^{quant})                  // RTN 或 GPTQ quantizer
```

  - 关键张量计算与直觉：
    - **二阶 Hessian 目标**：min E[ΔL] ≈ ½ vec(ΔW) H_full vec(ΔW)^T，H_full ≈ I ⊗ H 逐行解耦后变为 C_out 个独立子问题 min ½ Σ_i Δw_i H Δw_i^T。
    - **Group Error Compensation 闭式解**：将 Δw 分为 retain set R 和 eviction set E，令 e_E 为 E 上的压缩误差，则 min_{Δw_R} ½[Δw_R e_E] [H_{RR} H_{RE}; H_{ER} H_{EE}] [Δw_R^T e_E^T] 的闭式解为 Δw_R^* = -H_{RR}^{-1} H_{RE} e_E。Hessian 作为"桥梁"将误差从 E 传播到 R。
    - **计算复杂度**：需要逐行求解线性系统 H_{RR}^{-1} b，对 7B 模型约需 2 小时（单 A100）。
  - 使用示例（QuaRot + Llama2-7B）：
    ```bash
    cd ./QuaRot
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --rotate \
      --a_bits 4 --v_bits 4 --k_bits 4 --w_bits 4 --w_clip --ppl_eval
    ```

 One-Line Revolution for Generative AI Model Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 **LPCD (Layer-Projected Coordinate Descent)**——一种将 layer-wise PTQ 扩展到任意子模块量化的统一框架。核心流程为两步交替：(1) **Relaxation Step**：在固定其他 block 的条件下，对当前 block 做无约束连续优化（求闭式解或梯度下降近似），得到松弛后的全精度权重；(2) **Projection Step**：用标准 layer-wise PTQ 投影器（RTN 的 Π^(d) 或 GPTQ 的 Π^(a)）将松弛解投影回量化域。LPCD 统一了已有的 QEP（单步 weight-side LPCD）和 LoaQ（单步 augmented submodule LPCD），并自然扩展到三个 Transformer 子模块：**QK Module**（grouped-query attention 的 Q/K 投影）、**VO Module**（Value-Output 聚合）、**Up-Down Module**（MLP 的 Up/Down 投影）。QEP 扩展还包括 activation quantization、KV-cache quantization、orthogonal rotation matrices 和 LoRA-based error compensation。
  - 实验比较：LPCD-based submodule quantization vs. **QEP** 和 **LoaQ** 两种 error compensation baseline，分别在基础 quantizer **RTN** 和 **GPTQ** 上叠加。量化位宽：INT4、INT3、INT2（per-channel weight quantization）。指标为 WikiText-2 perplexity (PPL) 和 ARC-Easy/PIQA 零样本平均准确率。模型包括 LLaMA2-7B/13B、LLaMA3-8B、Qwen3-8B/14B。Figure 1 展示各层 output MSE 对比。

- 硬件平台是什么，配置是什么。
  - NVIDIA H100 GPU（TSUBAME 4.0 超级计算机）。单卡运行量化流程。

- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaMA2-7B、LLaMA2-13B、LLaMA3-8B、Qwen3-8B、Qwen3-14B（HuggingFace Transformers 实现，含 Qwen3-4B 和 LLaMA3.2-1B 用于超参网格搜索）。
  - 数据集/bench：**WikiText-2**（perplexity 评估）；**ARC-Easy** 和 **PIQA**（零样本准确率，使用 lm-eval-harness）；**C4** 和 **WikiText-2**（calibration 数据，最终使用 2048 tokens / 256 sequences 随机采样自 WikiText-2 以减轻过拟合）。
  - 量化配置：INT4/INT3/INT2 per-channel weight-only quantization；最后 2 层跳过量化（因激活 outliers 频率高）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未明确提供开源链接。
  - 软件环境：Python 3.12.11, PyTorch 2.4.0, HuggingFace Transformers 4.55.3。
  - 算法pipeline 核心伪代码（以 VO Module 为例，一轮 LPCD 更新 W_V 和 W_O）：

    ```
    # 输入: 已量化的残差流 R̂, 全精度 R
    #       全精度注意力权重 S^(h), 量化注意力权重 Ŝ^(h)
    #       量化校准特征 X̂, 全精度特征 X
    #       上一轮量化权重 Ŵ_V^(g), Ŵ_O

    # === Step 0: 初始化 ===
    # 先用 LoaQ 量化每个子模块作为初始点

    # === Step 1: Value Relaxation (固定 Ŵ_O, 更新 Ŵ_V) ===
    # 对每个 group g:
    #   Y = concat_h(S^(h) X W_V^(g)) W_O + R    # 全精度目标
    #   Ŷ_{¬g} = 不含 group g 贡献的量化输出
    #   y* = vec(Y - Ŷ_{¬g})
    #   Z_V^(g) = Σ_{h∈H_g} (Ŵ_O^(h)T ⊗ Ŝ^(h) X̂)
    #
    #   闭式解（内存不可行时用梯度下降近似）:
    #   vec(W̄_V^(g)) = (Z_V^(g)T Z_V^(g))^{-1} Z_V^(g)T y*

    # === Step 2: Value Projection ===
    #   Ŵ_V^(g) = Π_Q^(w)(W̄_V^(g))   # 用 RTN 或 GPTQ 投影

    # === Step 3: Output Relaxation (固定 Ŵ_V, 更新 Ŵ_O) ===
    #   Ĥ = concat_h(Ŝ^(h) (X̂ Ŵ_V^(g(h))))
    #   W̄_O = (ĤT Ĥ)^{-1} ĤT (Y - R̂)   # 闭式可解（ĤTĤ 规模可控）

    # === Step 4: Output Projection ===
    #   Ŵ_O = Π_Q^(w)(W̄_O)
    ```

  - 对 QK Module 和 Up-Down Module 的 Relaxation Step 因设计矩阵过大（如 QK 的 Z_Q ∈ R^{T² × (D_model d_k)}，Up 的 Z_U ∈ R^{T D_model × (D_model D_up)}），不显式构造矩阵，改为梯度下降近似求解（Adam, bs=8, 40 epochs, cosine LR 起始 1e-5）。
  - 对 Down Step 和 O-Step，因设计矩阵规模可控（仅依赖 head dim 或 D_up），直接用闭式解。
  - LPCD 在 LoaQ 量化结果之上运行（LoaQ 作为 LPCD 的初始化），先 LoaQ 后 LPCD。超参 α ∈ [0,1] step 0.1, β ∈ [0,1] step 0.05 通过小模型（Qwen3-0.5B, LLaMA3.2-1B）网格搜索确定后迁移到大模型。
