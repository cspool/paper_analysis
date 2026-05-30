## LoTA-QAF: Lossless Ternary Adaptation for Quantization-Aware Fine-Tuning

- baseline方法是什么？
  - baseline 是 GPTQ+LoRA（类 QLoRA 方案）：先对预训练 LLM 权重 W 进行 GPTQ 非对称分组量化（W_q = s·W_int + z），然后在冻结的量化权重上训练 16-bit LoRA 适配器 A、B（ΔW_16 = A B）。推理时通过 y = (W_q + ΔW_16)^T x 计算，但量化权重（如 4-bit）和 16-bit 适配器之间的数据类型不匹配导致计算效率损失。若将 16-bit 适配器合并入量化权重（W'_int = round((W_q + ΔW_16 - z)/s)），适配器精度会被截断/量化，重新引入量化误差，导致微调精度退化。QA-LoRA 实现了无损合并，但其适配器仅能调整分组量化的零点因子 z，无法直接修改量化权重 W_int。
  - baseline 全栈执行例子（GPTQ+LoRA）：
    - 算法pipeline：预训练 FP16 权重 → GPTQ 逐列量化（二阶 Hessian 信息补偿误差）→ 冻结 W_int，s，z → 初始化 FP16 LoRA 适配器 A∈R^{D_in×r}, B∈R^{r×D_out} → 前向 y = (s·W_int + z + (α/r)AB)^T x → 反向更新 A, B（FP16 精度）
    - 系统框架：论文未明确说明具体 Serving 框架修改
    - 编译框架：论文未明确说明
    - kernel调度：推理时 4-bit 权重需反量化到 FP16 与 LoRA 适配器相加，kernel 为 TritonV2QuantLinear/TorchQuantLinear
    - 硬件架构：在 NVIDIA A800 GPU 上运行，无专用硬件设计

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - LoTA-QAF 通过三个核心设计解决 baseline 缺陷：
    1. **三元适配器（TA）对齐量化网格**：将适配器约束为三值 {-1,0,1}，乘积 ΔW = A_T B_T 为整数矩阵，通过阈值 ω 生成三元调整矩阵 Ŵ ∈ {-1,0,1}。因为 Ŵ 与 W_int 同属整数域，可以直接相加调整量化权重（W'_int = W_int + Ŵ），无需反量化后再量化。这解决了 baseline 中 FP16 适配器无法直接融入量化权重的问题。
    2. **无损合并机制**：微调后通过 W'_int = W_int + Ŵ（带边界检查防溢出）和 z' = z + s·μ 将适配器完全吸收。合并前后前向计算等价（y = (s·W'_int + z')^T x），消除了 baseline 中合并时因适配器精度截断导致的精度损失。
    3. **t-SignSGD 优化器**：针对三值离散约束空间设计，通过 sign(g_t) 和动态百分位阈值 σ_t（top-5%→0.01%）选择性地翻转三值权重，不需学习率缩放，filter 噪声梯度，天然适配 {-1,0,1} 的离散更新空间。
  - 论文方法全栈执行例子：
    - 算法pipeline：GPTQ 量化预训练权重得到 W_int, s, z → 初始化三值适配器 A_T ∈ {-1,0,1}^{D_in×r}（Kaiming normal + 0.75·mean(|A|) 阈值三值化）和 B_T = 0 → 前向：ΔW = A_T B_T, Ŵ_ij = sign(ΔW_ij)·I_{|ΔW_ij|>ω}, W'_int = clamp(W_int + Ŵ, 0, 2^N-1), μ = mean(ΔW - ω·Ŵ), z' = z + s·μ, y = (s·W'_int + z')^T x → 反向（t-SignSGD）：g_t = ∇_{A_T} L, σ_t = top-k% threshold（线性衰减）, A_{T,t+1} = clip(A_{T,t} - sign(g_t)·I_{|g_t|>max(τ,σ_t)}, -1, 1) → 微调完成后合并 W_int ← W'_int, z ← z'，推理时无需适配器计算
    - 系统框架：论文未明确说明
    - 编译框架：使用 Triton 实现自定义 kernel（融合 Ŵ 生成和边界检查为单一 GPU kernel），三元数据类型用 bfloat16 模拟（因 PyTorch 不支持原生 int2/ternary dtype）
    - kernel调度：推理时使用与 baseline 相同的 TritonV2QuantLinear（4/2-bit）或 TorchQuantLinear（3-bit）kernel，但无需适配器开销（合并后），吞吐较 LoRA 提升 1.7x-2.0x
    - 硬件架构：NVIDIA A800 GPU，无专用硬件
