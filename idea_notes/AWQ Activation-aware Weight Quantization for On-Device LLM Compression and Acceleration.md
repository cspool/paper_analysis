## AWQ Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

- baseline方法是什么？
  Baseline 是 **GPTQ**（Frantar et al., 2022），当时 LLM weight-only 后训练量化的 SOTA 方法。GPTQ 流程：(1) 将权重量化问题建模为逐列（column-by-column）的二阶误差补偿；(2) 使用 Hessian 矩阵的逆来更新未量化权重，补偿已量化列引入的误差；(3) 通过对校准集做 block-wise reconstruction 降低量化误差。GPTQ 的核心缺陷：(i) **校准集过拟合**：reconstruction 过程对校准集分布敏感，当校准集（如 PubMed）与评估集（如 Enron）分布不同时，perplexity 恶化 2.3-4.9；(ii) **需要大量校准数据**：需要 192+ 条序列才能达到好的量化效果；(iii) **需要 trick**：对 LLaMA-7B 和 OPT-66B 需要 reordering trick 才能正常工作；(iv) calibration set 过拟合会扭曲预训练学到的一般性特征，影响 LLM 在 OOD 领域和多模态任务上的泛化能力。

  Baseline（GPTQ）全栈执行例子（LLaMA-7B INT3-g128）：
  - 算法pipeline：加载 FP16 权重 → 逐层对权重矩阵做 block-wise reconstruction：校准集前向传播缓存 layer input → 求 Hessian H=2XX^T → 计算 H^{-1} → 逐列量化：量化第 i 列 → 用 H^{-1} 更新剩余列以补偿第 i 列误差（OBS 算法）→ 重复至所有列量化完毕 → 输出 INT3 量化权重。对部分模型需 reordering（按 Hessian 对角线降序排列列，量化后恢复原序）。
  - 系统框架：AutoGPTQ（https://github.com/PanQiWei/AutoGPTQ）/ GPTQ-for-LLaMA，基于 PyTorch + HuggingFace Transformers。校准集 128-192 条 sequences from C4/WikiText。
  - 编译框架：使用 Triton 编写 INT4 reordered 量化 kernel（GPTQ-for-LLaMA）。
  - kernel调度：Triton kernel：INT4 反量化（通过 scale+zero_point 或对称量化 scale）→ 与 FP16 激活执行 GEMM/GEMV。对于 reordered 量化需额外的索引重排。
  - 硬件架构：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **AWQ（Activation-aware Weight Quantization）**，通过以下核心设计解决 GPTQ 的缺陷：

  **(1) 激活感知的显著权重识别替代 Hessian 二阶误差补偿**：GPTQ 使用 Hessian 指导误差补偿，依赖校准集分布导致过拟合。AWQ 发现只需识别 0.1%-1% 的显著（salient）权重通道即可大幅降低量化误差，且识别方式非常简洁——看**激活分布**（per-channel 平均激活幅度）而非权重分布。这避免了 GPTQ 的逐列 reconstruction 过程，不需要反向传播或回归，从根本上消除了过拟合问题。

  **(2) Per-channel scaling 替代混合精度**：直接保留显著权重为 FP16 虽然有效但会引入混合精度，硬件实现低效。AWQ 通过数学推导（Eq. 2-3）证明：对显著权重通道乘以 s > 1，并将对应的激活除以 s（等效变换），可以降低显著权重的相对量化误差（误差比例 `Δ'/Δ · 1/s < 1`）。scale s 通过单超参 α 的网格搜索（`s = s_X^α`）自动确定，仅需 20 步搜索即可找到平衡显著与非显著通道误差的最优 α。

  **(3) 极简校准集需求**：AWQ 仅需从校准集计算 per-channel 平均激活幅度 `mean(|X[c]|)`，而非做复杂的 reconstruction。因此 AWQ 仅需 16 条序列（GPTQ 需要 192 条，节省 10×），且对校准集分布不敏感——当校准集和评估集分布不同时 perplexity 仅恶化 0.5-0.6（GPTQ 恶化 2.3-4.9）。

  **(4) 首次实现多模态 LLM 低比特量化**：由于 AWQ 不过拟合校准集，可直接应用于 OpenFlamingo-9B、LLaVA-13B、VILA-7B/13B 等视觉语言模型（仅量化语言部分），为领域首次。INT4-g128 下 COCO Captioning 32-shot CIDEr 仅下降 1.17（RTN 下降 4.57，GPTQ 下降 6.72）。

  **(5) TinyChat 推理系统将理论压缩转化为实际加速**：针对 W4A16 量化中存储精度（INT4）与计算精度（FP16）不一致的挑战，TinyChat 设计 on-the-fly dequantization kernel（反量化与 GEMM/GEMV 融合在寄存器完成）、SIMD-aware weight packing（ARM NEON 上 32 个 4-bit 权重仅需 3 条 SIMD 指令解包）、kernel fusion（LayerNorm/Attention/QKV 投影融合），在 4090/Orin/4070 上实现 3.2-3.3× 加速比。

  论文方法全栈执行例子（LLaMA-7B INT4-g128）：
  - 算法pipeline：加载 FP16 权重 → 16 条 Pile 校准集前向传播收集 per-channel 激活幅度 `s_X = mean(|X|)` → 网格搜索 α ∈ [0,1]（20 步），每步：`s = s_X^α → W_scaled = W·diag(s) → INT4-g128 group-wise 量化 W_scaled → 用 `diag(s)^{-1}·X` 评估输出 MSE → 选最优 α → 最终量化 W 并融合 `diag(s)^{-1}` 入前一层。无需 Hessian、无需 reconstruction、无需反向传播。量化后 PPL 5.60（FP16 5.47，GPTQ 5.69）。
  - 系统框架（TinyChat）：PyTorch 前端 + CUDA/PTX 后端 → 加载 AWQ INT4-g128 量化权重 → 推理时：LayerNorm（fused kernel）→ QKV projection（fused, on-the-fly dequantization + RoPE on-the-fly）→ Attention（fused, KV cache 更新在 kernel 内完成）→ Output projection（dequantization GEMV）→ MLP（gate/up/down，fused dequantization）→ 残差连接 → 生成 token。RTX 4090 上从 HF FP16 52 tokens/s 加速至 TinyChat W4A16 ~194 tokens/s（3.7×）。
  - 编译框架：论文未明确说明（使用 PyTorch eager mode + 自定义 CUDA/PTX kernel，未修改 compiler framework）。
  - kernel调度：TinyChat CUDA kernel——INT4 packed 权重（每 2 个 4-bit 占 1 byte）从 DRAM 读取 → 寄存器内 shift + AND 解包 → 乘以 group-wise Δ（FP16）→ 乘以 per-channel s → 与 FP16 激活做 FMA → 输出存回寄存器供下一操作使用。SIMD-aware packing 在 ARM NEON 上提供额外 1.2× 加速。Kernel fusion 将每次推理的 kernel launch 从数十次减少到数次（每个 Transformer Block 约 3-4 次）。
  - 硬件架构：TinyChat 在 NVIDIA Jetson Orin（15W，8GB，移动 GPU）上部署 Llama-2-70B（awq量化后），并在 Raspberry Pi 4B 上部署 Llama-7B（0.7 tokens/s）。
