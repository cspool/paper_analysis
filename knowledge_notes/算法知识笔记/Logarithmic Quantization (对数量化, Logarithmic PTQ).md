## Logarithmic Quantization (对数量化, Logarithmic PTQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Logarithmic Quantization（对数量化）是一种非线性后训练量化方法，其量化级别呈指数间隔排列（在对数域中对应均匀步长）。与线性量化（量化级别均匀分布）不同，对数量化在数值较小时提供密集的量化级别，数值较大时提供稀疏的量化级别，天然匹配深度神经网络中常见的钟形（bell-shaped）和长尾权重分布。LOGART 论文系统比较了三种对数量化变体：
- **Log2**：量化级别为 2 的幂次值 {..., 2^{-3}, 2^{-2}, 2^{-1}, 2^0, 2^1, ...}。硬件优势是可完全用移位器替代乘法器（乘以 2^Q 等价于左移 Q 位），但量化台阶粗糙，大值附近精度不足。
- **Log√2**：量化级别为 √2 的幂次值，粒度更细。代价是乘法涉及 √2 不能直接用移位器实现，需 LUT 或 shift-add 近似。
- **DLog (Dynamic Log)**：混合基方案，大值用 base-√2（细粒度），小值用 base-2（硬件友好），通过对数域阈值 t 分割两个区域。

LOGART 的对数量化核心公式（基-2）：
- Quant: Q_W = clamp(⌊-log_2(|W|/s)⌋, 0, 2^{N-1}-1)，其中 s = 2^{⌊log_2(max(|W|))⌉}
- Dequant: Ŵ = s · sign(W) ⊙ 2^{-Q_W}

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LOGART 论文中对数量化在 PTQ pipeline 中的位置：
```
# 对数量化推理流程 (Log2, weight-only)
输入: W ∈ R^{d_out × d_in}  (FP16 weight), X ∈ R^{d_in × seq}  (激活)
# Step 1: 计算 scale
s = 2^{round(log_2(max(|W|)))}  # per-channel 或 per-tensor
# Step 2: 量化 weight
Q_W = clamp(floor(-log_2(|W| / s)), 0, 2^{N-1} - 1)  # N-bit 整数码字
# Step 3: 推理 (使用移位器替代乘法器)
# 因为 Ŵ_{ij} = s · sign(W_{ij}) · 2^{-Q_W_{ij}}
# 所以 ŴX 的第 i 行 = Σ_j s · sign(W_{ij}) · X_j >> Q_W_{ij}  (右移)
output = shift_accumulate(X, Q_W, s)  # 无乘法器
```
对数量化在超低比特（3-4bit）下通常优于线性量化的原因是其非均匀量化级别能更好地保留大幅值权重的精度，而这些权重对模型输出影响更大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
对数量化可直接在 PyTorch 中实现：用 `torch.log2()` 转换到对数域 → `torch.floor()` 或 `torch.round()` 量化 → `torch.pow(2, ...)` 反量化。硬件实现上，Log2 量化推理时乘法器被移位器替代，Log√2 需额外的 shift-add 近似或 LUT 处理。LOGART 开源代码（https://github.com/logart-lab/logart）提供了完整的对数量化实现，支持 Calibration-based PTQ，可与 HuggingFace Transformers 模型集成。现有对数量化方法包括 LogNet (Lee et al. 2017), FQ-ViT (Lin et al. 2022), SLogII/Xu et al. (2023), 以及 LOGART。

I&S-ViT 在此基础上提出 SULQ (Shift-Uniform-Log2 Quantizer)：在 log2 变换前引入 shift bias η（X_q = UQ(-log₂(X+η), b)），后接均匀量化器。这解决了标准 log2 量化器在 post-Softmax 激活上的"量化低效"问题——大量值被 clamp 到远端量化级别。SULQ 仅增加一次 round 和两次加法，推理时仍通过 bit-shift 执行。

涉及论文标题：
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

---
