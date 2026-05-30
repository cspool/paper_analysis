## Ternary Adaptation (TA, 三元适配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ternary Adaptation（TA，三元适配）是 LoTA-QAF 提出的一种面向量化 LLM 微调的适配器结构。与标准 LoRA 使用 FP16 可训练适配器不同，TA 将适配器权重约束为三值离散空间 {-1, 0, 1}：两个低秩矩阵 A_T ∈ {-1,0,1}^{D_in×r} 和 B_T ∈ {-1,0,1}^{r×D_out}（r ≪ min(D_in, D_out)）。A_T 通过 Kaiming normal 初始化后经阈值 0.75·mean(|A_T|) 三值化；B_T 初始化为零。乘积 ΔW = A_T B_T 形成辅助矩阵，元素为整数 ∈ [-r, r]。通过阈值 ω ∈ (0, r)，将 ΔW 映射为三元调整矩阵 Ŵ ∈ {-1,0,1}^{D_in×D_out}：Ŵ_ij = sign(ΔW_ij) · I_{|ΔW_ij|>ω}。由于 Ŵ 与量化权重 W_int 同属整数域，可直接相加 W'_int = W_int + Ŵ 实现量化网格内（in-grid）权重的直接调整，无需反量化-再量化。同时计算偏移因子 μ = mean(ΔW - ω·Ŵ)，用于更新零点因子 z' = z + s·μ。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LoTA-QAF 中 TA 的完整前向流程（以 4-bit 量化 linear 层为例）：
```
# 输入: x (激活), W_int (量化权重, shape D_in×D_out, 值域[0,15])
#      s, z (量化参数), A_T (D_in×r, 三值), B_T (r×D_out, 三值)
#      ω (阈值, 如 0.75r)

# 1. 计算辅助矩阵
ΔW = A_T @ B_T          # D_in×D_out, 元素 ∈ [-r, r]

# 2. 生成三元调整矩阵
Ŵ = zeros_like(ΔW)
Ŵ[ΔW > ω] = 1
Ŵ[ΔW < -ω] = -1

# 3. 应用边界检查后调整量化权重
W_int_new = W_int + Ŵ
W_int_new = clamp(W_int_new, 0, 2^N - 1)

# 4. 计算偏移因子
W̃ = ΔW - ω * Ŵ           # 残差
μ = mean(W̃)               # per-tensor / per-group / per-channel

# 5. 前向输出
z_new = z + s * μ
y = (s * W_int_new + z_new)^T @ x
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
由于 PyTorch 不支持原生 ternary/int2 数据类型，TA 使用 bfloat16 模拟三值 {-1,0,1}。辅助矩阵 ΔW 的形成和 Ŵ 映射通过 Triton 自定义 kernel 实现（kernel fusion 融合 Ŵ 生成和边界检查为单一 GPU kernel）。在 LoTA-QAF 开源代码（github.com/KingdalfGoodman/LoTA-QAF）中，TA 实现于 LoTA/layer.py 的 CustomLoraLinear 类。推理时，微调完成的适配器通过 lota_merge.py 合并到量化权重中，推理仅需标准量化 kernel（TritonV2QuantLinear/TorchQuantLinear），无额外适配器开销。超参数：rank r 通常 64（8B/14B）或 32（32B/70B），ω = 0.75r（通用）或 0.875r（ViGGO 等特殊任务）。

涉及论文标题：
- LoTA-QAF: Lossless Ternary Adaptation for Quantization-Aware Fine-Tuning
