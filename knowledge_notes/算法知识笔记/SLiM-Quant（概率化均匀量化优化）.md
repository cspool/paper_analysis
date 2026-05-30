## SLiM-Quant（概率化均匀量化优化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SLiM-Quant 是 SLiM 论文（Mozaffari et al., ICML 2025）提出的概率化均匀量化优化方法。核心思路是将对称均匀量化中 non-convex 的 MSE 最小化问题（argmin_α ||round(clip(W/α))·2^{q-1} - W||²）通过概率化重表述转化为可求解的积分形式。将权重元素视为从概率密度函数 f(·) 中采样，则量化误差期望为 E_Q(α) = ∫ f(x)|Q^{-1}(Q(x)) - x|² dx，拆分为量化误差 E_quant(α) = ∫_0^α f_abs(x)|α·round(x/α)·2^{1-q} - x|² dx 和裁剪误差 E_clip(α) = ∫_α^∞ f_abs(x)|α - x|² dx。由于实际权重分布不符合任何标准 PDF，SLiM-Quant 在权重直方图上做数值积分，采用多网格策略（低分辨率 10 样本均匀扫描 + 在最低误差区域高分辨率细化搜索）高效找到全局最优 scaling factor α*。相比 Grid Search 或 AbsMax（对 outlier 敏感），SLiM-Quant 在保持 uniform quantization 的硬件友好性（单 scale per tensor）的同时，达到 group quantization 级别的精度。

SLiM-Quant 的激活感知变体 SLiM-Quant^O 进一步定义联合显著性 saliency = |diag(x_mean) × W|（x_mean 为校准集激活平均绝对值），对 top 1% 最高显著性通道做 scale up 权重 × s + scale down 对应激活 ÷ s 的等效变换，降低输出误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SLiM-Quant 算法（Algorithm 1 from paper）
# 输入: 权重绝对值PDF f_abs, 步长 η_low/η_high, 权重 W, 位宽 q

def EstimateError(α):
    # 量化误差: 绝对值小于α的元素
    E_quant = ∫_0^α f_abs(x) |α × round(x/α) × 2^{1-q} - x|² dx
    # 裁剪误差: 绝对值大于α的元素
    E_clip = ∫_α^∞ f_abs(x) |α - x|² dx
    return E_quant + E_clip

# Phase 1: 低分辨率扫描 (10 samples in [0, max(W)])
E = {}
for α in linspace(0, max(|W|), 10):
    E[α] = EstimateError(α)
α_low = argmin(E)

# Phase 2: 高分辨率细化
for α in linspace(α_low - η_low, α_low + η_low, resolution=η_high):
    E[α] = EstimateError(α)
α* = argmin(E)

# 量化输出
W_quant = round(clip(W/α*)) × 2^{q-1}
```

关键参数：直方图 bin 数 = max(512, min(d_in×d_out/1000, 20000))，保证概率密度近似精度。多网格策略使计算开销极小（与 Wanda 相当的压缩时间，见表 21）。

SLiM-Quant^O 额外步骤：
```
x_mean = mean(calibration_activations, dim=batch)
saliency = |diag(x_mean) × W|  # per-channel 显著性
top1pct = top_k(saliency, k=0.01*d_in)
W[top1pct] *= s      # scale up weights
x[top1pct] /= s      # scale down activations (等效)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SLiM-Quant 已集成到 SLiM 开源代码库（https://github.com/Mohammad-Mozaffari/slim / https://github.com/Paramathic/slim）。使用方式：(1) 对每个 Linear 层权重独立计算最优 α*（逐层执行，仅需一层权重在 GPU 内存中）；(2) 直方图构建和数值积分均为纯 CPU/GPU 操作；(3) 与 Wanda、SparseGPT 等剪枝方法及 SLiM-LoRA 低秩适配无缝衔接。SLiM-Quant^W（仅权重误差最小化）推理无额外开销；SLiM-Quant^O 约 1% 激活通道的 on-the-fly scaling 引入轻微不规则内存访问，是精度 vs 开销的权衡。论文实测 SLiM-Quant^W 与 SLiM-Quant^O 准确率差距微小（约 0.1%）。

涉及论文标题：
- SLiM One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

---
