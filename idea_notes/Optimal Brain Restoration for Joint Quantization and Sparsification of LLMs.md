## Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs

- baseline方法是什么？
  - **QuaRot (quant-only)**：单独使用量化，通过 Hadamard rotation 平滑权重 outliers 实现 W4A4KV4 量化。缺陷：sub-4bit 时性能急剧下降（如 Llama2-7B W3A4KV4 perplexity 达 132.97），且单独压缩方法逼近极限。
  - **QuaRot+WANDA (naive joint)**：直接将 QuaRot 量化（经过 Hadamard rotation）与 WANDA 剪枝组合，即对 rotated weights 直接施加剪枝 mask 后再量化。缺陷：Hadamard rotation 使权重分布平坦（quantization-friendly），但这与剪枝的需求（需要 weight magnitudes 差异大以呈现天然稀疏性）冲突，导致性能灾难性下降（Llama2-7B perplexity 达 5868.24）。
  - **SparseGPT+GPTQ (strong baseline)**：使用 SparseGPT 剪枝 + GPTQ 量化的组合。缺陷：虽然较 naive combination 有改善，但未专门调和量化（偏好窄范围）与剪枝（偏好高方差）对权重分布的冲突需求，在 W4A4KV4 + 50% sparsity 激进压缩下性能依然不足。
  - 全栈执行例子（以 QuaRot+WANDA baseline 为例）：
    - **算法层**：加载 FP16 Llama2-7B → Hadamard rotation R 作用于 W 和 X (将异常值扩散打平) → WANDA 用激活统计 (|W|·||X||₂) 做重要性得分 → 对 rotated W 施加 50% 非结构化剪枝 mask → RTN 量化 W 到 INT4 → 激活/X 也经 rotation 后量化到 INT4 → KV cache INT4 量化。输出 W4A4KV4 + 50% sparse 模型。
    - **系统框架层**：PyTorch + HuggingFace Transformers, block-wise GPU 加载。校准用 128 WikiText2 样本。
    - **kernel调度层** (OBR baseline 特有)：NVIDIA CUTLASS INT4 2:4 sparse GEMM kernel，利用 Ampere/Hopper 的 Sparse Tensor Cores 实现硬件加速。INT4 权重（packed）+ 2:4 结构化稀疏 mask，FP16 激活经反量化后与 sparse 权重通过 `mma.sp.sync` 指令在 Tensor Core 上执行 MMA。
    - **硬件架构层**：NVIDIA A100 GPU (Ampere 架构)，第三代 Tensor Cores 原生支持 2:4 结构化稀疏和 INT4/INT8 混合精度计算。
  - Baseline 核心缺陷：**量化与剪枝的权重分布需求冲突**——量化希望紧凑的数值范围以减少量化误差，剪枝希望大幅值差异以暴露可剪权重。Hadamard rotation 虽利于量化但破坏了剪枝所需的 distributional disparity。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **OBR (Optimal Brain Restoration)**：在剪枝后、量化前插入 Group Error Compensation，通过 Hessian 矩阵作为"桥梁"将压缩误差从敏感 group 转移到鲁棒 group，从而调和量化与剪枝的冲突。核心公式化：minΔw_R ½[Δw_R e_E] [H_{RR} H_{RE}; H_{ER} H_{EE}] [Δw_R^T e_E^T]，闭式解 Δw_R^* = -H_{RR}^{-1} H_{RE} e_E。
  - **如何解决 Baseline 缺陷**：
    - **调和权重分布冲突 (核心贡献)**：不改变 Hadamard rotation（维持其量化友好性），也不改变剪枝 mask，而是在二者之间通过 Error Compensation 重新分配信息。剪枝损失的权重信息（e_E^{prune} = w_E）通过 Hessian 子矩阵 H_{RE} 传播到 unpruned 权重中进行补偿，使 unpruned 权重在量化前就已经"吸收"了剪枝损失的知识。类似地，量化误差也通过第二轮 OBR 补偿到更多元素中。
    - **二阶 Hessian 目标**：用 min E[ΔL] ≈ ½ vec(ΔW) H_full vec(ΔW)^T 建模权重扰动对 downstream task 的影响。通过 Kronecker 分解 H_full ≈ G ⊗ H 和 row-wise decoupling (G≈I) 将 C_out×C_in 维问题分解为 C_out 个独立的 C_in 维子问题，使原本 O((C_out·C_in)²) 不可行的问题变为可解。
    - **无需额外训练 (training-free)**：OBR 是纯后训练方法，只需一次 calibration 数据前向传播收集 Hessian 统计信息，然后通过闭式解完成补偿。比 QAT (quantization-aware training) 方法更实用。
    - **兼容多种剪枝方法和量化器**：OBR 将剪枝 mask 和 quantizer 视为给定输入（"黑盒"），因此兼容 WANDA、SparseGPT、magnitude-based、甚至 Random 剪枝；也兼容 RTN 和 GPTQ 量化器。
  - 论文方法全栈执行例子（以 OBR via QuaRot + WANDA + RTN 为例）：
    - **算法层**：(0) 128 WikiText2 样本 × 2048 seq_len 前向传播，收集每层激活 X，计算 Hessian H = 2XX^T。(1) Hadamard rotation R 作用于 W 和 X 打平 outliers。(2) WANDA 根据 |W_rot|·||X_rot||₂ 计算重要性得分，生成 50% 非结构化剪枝 mask M。(3) **OBR for Pruning**：对每行，R₁=unpruned, E₁=pruned, 计算 Δw_{R₁}^{prune} = -H_{R₁R₁}^{-1}H_{R₁E₁}W_{c,E₁}, 补偿到 w_{R₁}。(4) **OBR for Quantization**：w̄ = w_{R₁}+Δw^{prune}, 计算量化误差 e^{quant}=w̄-⌊w̄⌉, 按 α=50% 划分 E₂ 和 R₂, 计算 Δw_{R₂}^{quant} = -H_{R₂R₂}^{-1}H_{R₂E₂}e_{E₂}^{quant}。(5) 合并补偿 ΔW^{OBR} = ΔW^{prune} + ΔW^{quant}, W^{quant} = W^{prune} + ΔW^{OBR}, 对 W^{quant} 做 RTN 量化到 INT4。(6) 对 activation X 和 KV cache 也量化到 INT4。输出 W4A4KV4 + 50% sparse Ŵ。
    - **系统框架层**：PyTorch + HuggingFace Transformers, block-wise loading, single A100 GPU。
    - **kernel调度层**：NVIDIA CUTLASS INT4 2:4 sparse GEMM kernel。权重以 packed INT4 + 2:4 metadata 存储（50% 稀疏将访存带宽减少 2×），激活 INT4 → FP16 反量化后进入 Sparse Tensor Core，通过 `mma.sp.sync` 指令利用硬件跳过零值。在 seq_len=4096 时达到 5.9× (vs FP16-dense) 和 1.4× (vs INT4-dense) 加速。
    - **硬件架构层**：NVIDIA A100 GPU (Ampere)，Tensor Cores 原生支持 2:4 sparse MMA 和 INT4 推理。

 One-Line Revolution for Generative AI Model Compression

- baseline方法是什么？
  - **Layer-wise PTQ (RTN / GPTQ)**：将每个 linear layer 独立量化为最小二乘问题。RTN 直接逐权重量化（Eq.1: Ŵ = argmin ||Ŵ - W||²），GPTQ 使用 activation-aware 目标（Eq.2: Ŵ = argmin ||X̂(Ŵ - W)||²）逐行贪心量化并用 Hessian 补偿残差。缺陷：局限于单个 linear layer，无法建模跨层/跨子模块的误差传播。
  - **QEP**：在 layer-wise PTQ 基础上引入误差传播修正（Eq.3: Ŵ_QEP = argmin ||X̂Ŵ - XW||²），等价于将量化目标从 W 改为修正后的 W* = (I + α Ĥ^{-1}C)W。缺陷：仍仅限于单个 linear layer，不处理 attention、MLP、残差连接等更复杂子模块。
  - **LoaQ**：将 QEP 扩展到残差路径（Eq.4: 最小化 ||(R̂ + X̂Ŵ) - (R + XW)||²），通过 W*(α,β) 同时修正线性层误差和残差路径误差。缺陷：限于特定的 attention+MLP+残差+RMSNorm 子模块组合，不提供任意子模块的统一处理，且为单步修正。
  - 全栈执行例子（以 GPTQ+QEP baseline 量化一层 Q 投影为例）：
    - **算法层**：加载 FP16 LLaMA → 校准数据前向收集每层输入 X → 计算 Hessian H = XᵀX 和误差传播矩阵 C = X̂ᵀ(X - X̂) → 计算修正目标 W* = (I + αĤ^{-1}C)W → 对 W* 逐行 GPTQ 贪心量化（取整 → 更新 Hessian 残差补偿到剩余列 → 直到所有列量化完毕）→ 输出 INT-k 权重。
    - **系统框架层**：PyTorch + HuggingFace Transformers, block-wise GPU 加载。推理时标准量化 matmul（反量化×激活）。
    - **编译框架/kernel调度/硬件架构层**：论文未明确说明（依赖 PyTorch 默认 CUDA kernel 和 H100 TensorCores）。
  - Baseline 核心缺陷：layer-wise 独立优化忽视子模块内部的非线性交互（如 softmax、残差连接、门控激活），导致 sub-4bit（尤其是 INT3/INT2）时量化误差急剧累积；QEP/LoaQ 通过单步修正改善误差传播，但修正能力受限于特定模块和固定公式。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **LPCD (Layer-Projected Coordinate Descent)**：将 PTQ 重新定义为对任意子模块 blocks 的离散优化问题（Eq.5: min_{M̂_1,...,M̂_R} L(M̂_1,...,M̂_R)），通过交替坐标下降求解：
    1. **Relaxation Step**（Eq.6）：固定其他 blocks，对当前 block 的连续变量 U 求解无约束优化 M̄_r = argmin_U L_r(U)。当 L_r 为严格凸二次函数时直接求闭式解；否则用梯度下降近似。
    2. **Projection Step**：用标准 layer-wise PTQ 投影器 Π_Q（RTN 的 Π^(d) 或 GPTQ 的 Π^(a)）将 M̄_r 投影回量化域 M̂_r = Π_Q(M̄_r)。
    3. 交替更新所有 blocks，一轮完成后可多轮迭代。
  - **如何解决 Baseline 缺陷**：
    - **从单层到子模块**：LPCD 的 block 变量 M_r 可以是任意 Transformer 子模块的权重、激活或 KV cache。对 QK Module（grouped-query attention 的 Q/K 投影），最小化 masked attention score 的 Frobenius 误差 ||M ⊙ (Ŝ - S)||²；对 VO Module（value-output 聚合），最小化残差流输出误差 ||Ω̂ + R̂ - (Ω + R)||²；对 Up-Down Module（MLP 含 SiLU 门控），最小化 ||F̂ + R̂ - (F + R)||²。这直接建模了子模块内部的非线性交互。
    - **统一 QEP/LoaQ 为特例**：Proposition 4.1 证明 QEP 是两-block (Ŵ, X̂) LPCD 的单步更新；Remark 4.2 证明 LoaQ 是三-block (Ŵ, X̂, R̂) LPCD 的单步更新。LPCD 通过增加迭代次数和扩展 block 集，显著超越单步修正。
    - **避免 STE 不稳定性**：LPCD 的 Relaxation Step 求解严格连续优化（闭式解或梯度下降），Projection Step 复用成熟的 layer-wise PTQ，无需引入 pseudo-gradient（STE），避免了 QAT 中常见的不稳定问题。
    - **正交于基础 quantizer**：LPCD 的 Projection Step 可插拔 RTN 或 GPTQ，因此其收益独立于底层量器选择。实验表明 RTN+LPCD 在部分设置下已超越 GPTQ+QEP。
  - 论文方法全栈执行例子（以 VO Module 的 LPCD 为例）：
    - **算法层**：前向传播收集校准数据的全精度 S^(h), V^(g), Ω, R 和量化版的 Ŝ^(h), V̂^(g), R̂ → 固定 Ŵ_O，对每个 group g 的 Ŵ_V^(g) 做 Relaxation（求解最小化 ||Y - Ŷ_{¬g} - Ŷ_g||² 的线性最小二乘，设计矩阵过大时用 Adam 梯度下降近似）→ Projection（RTN/GPTQ 量化 W̄_V^(g)）→ 固定 Ŵ_V，对 Ŵ_O 做 Relaxation（Ŵ_O = (ĤᵀĤ)⁻¹Ĥᵀ(Y - R̂)，闭式解可行）→ Projection → 完成一轮 VO Module LPCD。同流程应用于 QK 和 Up-Down 模块。
    - **系统框架层**：PyTorch + HuggingFace Transformers，与 layer-wise PTQ 兼容的 block-wise 内存管理。量化流程先运行 LoaQ 作为初始化，再在 LoaQ 结果上运行 LPCD。
    - **编译框架/kernel调度/硬件架构层**：论文未明确说明（仅使用 PyTorch 默认 CUDA kernel + H100 GPU）。
  - LPCD 还将 QEP 风格修正扩展到 activation quantization（Eq. 16-17: X̄ = XWŴᵀ(ŴŴᵀ)⁻¹ → 投影）、KV-cache quantization（Eq. 18-22: Key cache 对齐 pre-softmax logits，Value cache 对齐 post-softmax outputs）、正交旋转矩阵（Eq. 23-27: 通过 LPCD 在固定 X̂, Ŵ 下优化旋转矩阵 R，闭环式解 + 正交 Procrustes 投影）、以及 LoRA 误差补偿（weighted low-rank projection onto E = BA）。
