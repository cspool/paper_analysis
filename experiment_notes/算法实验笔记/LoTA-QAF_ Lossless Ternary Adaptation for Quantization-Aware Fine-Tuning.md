## LoTA-QAF: Lossless Ternary Adaptation for Quantization-Aware Fine-Tuning

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：LoTA-QAF 是一种面向量化 LLM 的量化感知微调（QAF）方法，包含三个核心组件：i) 三元适配器（Ternary Adaptation, TA），训练 `A_T ∈ {-1,0,1}^{D_in×r}` 和 `B_T ∈ {-1,0,1}^{r×D_out}`，通过辅助矩阵 `ΔW = A_T B_T` 和阈值 ω 生成三元矩阵 `Ŵ ∈ {-1,0,1}^{D_in×D_out}`（`Ŵ_ij = sign(ΔW_ij) · I_{|ΔW_ij|>ω}`），直接在量化网格内调整量化权重 `W_int`；ii) TA-based lossless merging 机制，通过 `W'_int = W_int + Ŵ` 和 `z' = z + sμ`（μ 为偏移因子）将适配器无损合并到量化权重和零点因子中；iii) t-SignSGD 优化器，使用基于符号梯度的更新和动态百分位阈值 σ_t 选择性地更新三元适配器权重（`A_{T,t+1} = clip(A_{T,t} - sign(g_t) · I_{|g_t|>max(τ,σ_t)}, -1, 1)`）。
  - 实验比较两种微调范式：i) performance-recovery（Alpaca 数据集微调后在 MMLU 5-shot 上评估，对比 GPTQ 量化基线、GPTQ+LoRA（4+16bit）和 QA-LoRA）；ii) task-specific（在 GSM8K、SQL generation、ViGGO 三个任务上微调评估）。

- 硬件平台是什么，配置是什么。
  - 所有实验在一张 NVIDIA A800 GPU 上运行。
  - 推理效率测试使用 Llama 3.1 8B 模型，4-bit/2-bit 用 TritonV2QuantLinear kernel，3-bit 用 TorchQuantLinear kernel。Batch size 8-128，最大推理长度 512 tokens。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama 3.1 8B、Qwen 2.5 14B、Qwen 2.5 32B、Llama 3.3 70B。
  - 量化方式：GPTQ 非对称量化，Llama 8B 和 Qwen 14B 用 group size 64，Qwen 32B 和 Llama 70B 用 group size 128。校准数据使用 C4 数据集 1024 样本。
  - 数据集：Alpaca（performance-recovery 微调）；GSM8K（7.47k 训练/1.32k 测试）、SQL generation（30k 训练/1 测试）、ViGGO（5.1k 训练/1.08k 测试）（task-specific 微调）。
  - Benchmark：MMLU（5-shot，含 Humanities/STEM/Social/Other 四类）；GSM8K（0-shot）、SQL（0-shot）、ViGGO（0-shot）；使用 lm-eval 框架评估 MMLU，使用 HALO 的自定义评估框架评估 task-specific 任务。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码在 github.com/KingdalfGoodman/LoTA-QAF，MIT 协议，包含 LoTA_QAF_main.py（训练/评估）、LoTA/layer.py（CustomLoraLinear 实现三元适配）、LoTA/adapter.py（LTA 推理加载）、LoTA/lota_merge.py（合并逻辑）、t_signSGD.py（优化器）等。
  - 算法 pipeline（张量计算）：
    1. 量化预训练权重：`W_q = s · clamp(round((W - z)/s), 0, 2^N-1) + z`，得到 `W_int`、`s`、`z`
    2. 初始化三元适配器：`A_T` 用 Kaiming normal 初始化后通过阈值 0.75·mean(|A_T|) 三值化为 `{-1,0,1}`；`B_T` 初始化为零
    3. 前向传播：`ΔW = A_T @ B_T`（整数矩阵，元素 ∈ [-r, r]）→ `Ŵ_ij = sign(ΔW_ij) · I_{|ΔW_ij|>ω}` → `W'_int = W_int + Ŵ`（含边界检查防止溢出 [0, 2^N-1]）→ `μ = mean(ΔW - ω·Ŵ)` → `z' = z + s·μ` → `y = (s·W'_int + z')^T · x`
    4. 反向传播（t-SignSGD）：计算梯度 g_t = ∇_{A_T} L → 确定动态阈值 σ_t（top-5% 梯度幅值，线性衰减至 0.01%）→ `A_{T,t+1} = clip(A_{T,t} - sign(g_t) · I_{|g_t|>max(τ,σ_t)}, -1, 1)`
    5. 推理时无损合并：直接使用 `W'_int` 和 `z'` 替代 `W_int` 和 `z`，无需额外适配器计算，保持低比特推理效率（相比 LoRA 的 16-bit 适配器，LoTA 合并后速度提升 1.7x-2.0x）
  - 超参数：rank r=64 (8B/14B) 或 32 (32B/70B)；ω = 0.75r (Alpaca/GSM8K/SQL) 或 0.875r (ViGGO)；σ_t 初始 top-5%，前 80% 训练线性衰减至 0.1%，后 20% 固定 0.01%；优化器 paged AdamW，max grad norm 0.3，batch size 64，source length 1024，target length 256
