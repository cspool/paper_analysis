## Accurate LoRA-Finetuning Quantization of LLMs via Information Retention

- baseline方法是什么？
  Baseline 是 QLoRA (Dettmers et al., 2023)，即 LoRA-finetuning quantization 标准范式：(1) PTQ 阶段使用 NormalFloat (NF) quantization 将 LLM 权重量化到 k-bit；(2) 在量化后的 LLM 上额外附加 LoRA 低秩适配器（rank r=64）进行参数高效微调。量化过程使用对称量化，scale factor s = absmax(w)，无 calibration constant（零点为 0）。同时比较的 baseline 还包括 QA-LoRA（integer 量化 + 量化感知 LoRA）、QLoRA w/ GPTQ（GPTQ 量化）和 PEQA（无 LoRA 的量化感知微调）。

  Baseline 全栈执行例子（QLoRA, 4-bit LLaMA-7B, Alpaca 微调, MMLU 评估）：
  - 算法pipeline：加载 FP16 LLaMA-7B 预训练权重 → 按 block_size=64 分块 → NormalFloat 4-bit 量化 ŵ = NF4(w/absmax(w)) → double quant scale s₁^FP8, s₂^FP16 → 附加 LoRA（r=64, α=16）适配所有 linear 层 → 在 Alpaca 52K 数据上 AdamW 微调 10000 steps → MMLU 5-shot 评估。此 baseline 在信息层面存在两个缺陷：(a) NF 量化采用零点固定为零的对称量化，导致量化权重信息熵最大化受限、与原始权重互信息不足；(b) LoRA 的两个低秩矩阵 ℓ₁, ℓ₂ 仅做矩阵乘法变换，变换形式同质化，且 ℓ₂ 只能使用 ℓ₁ 的中间表示而无法直接利用原始输入 x。
  - 系统框架：基于 HuggingFace Transformers + PEFT 库（LoRA 实现）→ 使用 QLoRA 官方代码库 → PyTorch 训练。
  - 编译框架：论文未明确说明（标准 PyTorch eager mode）。
  - kernel调度：论文未明确说明（使用标准 PyTorch FP16/BF16 矩阵乘法 kernel）。
  - 硬件架构：NVIDIA Tesla A100 GPU，无自定义硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 IR-QLoRA，从统一的信息视角出发解决两个信息丢失问题：

  **(1) ICQ (Information Calibration Quantization) 解决量化信息丢失问题**：
  Baseline QLoRA 中对称量化 ŵ=NFk(w/s) 的零点固定为零，量化权重熵 H(ŵ) 不能最大化，导致与原始权重的互信息不足。ICQ 引入 calibration constant τ，将量化变为 ŵ=NFk((w-τ)/s)，并通过最大化量化权重的信息熵 H(ŵ) = -ΣP(q_i)log₂P(q_i) 来搜索最优 τ*。搜索以 median(w) 为初始值（符合正态分布对称性假设），在 [τ₀-0.1σ, τ₀+0.1σ] 区间内均匀采样 200 个候选，选最大熵对应的 τ*。ICQ 将 4-bit LLaMA-7B 的权重熵从 3.67 提升到 3.74，无需 LoRA 微调即可使 MMLU 提升 0.5%。

  **(2) IEC (Information Elastic Connection) 解决 LoRA 表征能力不足问题**：
  Baseline LoRA 的 ℓ₂ 矩阵只能使用 ℓ₁ 的低秩变换结果，无法访问原始输入 x。IEC 通过两个 parameter-free 操作解决：(a) U₁ 中对输入 x 按 (r/h) 比例分组平均后加到 ℓ₁ 输出，使 ℓ₁ 能融合原始输入信息；(b) U₂ 中对中间表示 x' 重复拼接 (o/r) 次后加到 ℓ₂ 输出，使 ℓ₂ 能直接利用多样化表示。IEC 仅引入 2 个 per-layer learnable scalars (β₁, β₂)，且在推理时可通过矩阵数学合并消除额外开销。

  论文方法全栈执行例子（IR-QLoRA, 4-bit LLaMA-7B, Alpaca, MMLU）：
  - 算法pipeline：加载 FP16 LLaMA-7B → **ICQ**: 按 block_size=64 对每块权重 search τ* → ŵ=NF4((w-τ*)/absmax(w-τ*)) → double quant τ* 和 s → **IEC 微调**: 在 Alpaca 上训练 LoRA+β₁,β₂ 10000 steps → 推理时 IEC 合并入 LoRA → MMLU 5-shot 评估得 40.8%（vs QLoRA 38.4%，提升 1.4%）。
  - 系统框架：基于 QLoRA 官方代码修改 → HuggingFace Transformers + PEFT → PyTorch 训练。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。
  - 硬件架构：NVIDIA Tesla A100 GPU。ICQ 搜索仅增加 0.46%（7B）/ 0.31%（13B）训练时间，IEC 无额外训练时间。存储方面 ICQ 增加 2.04% 参数（7B：2.34GB→2.39GB），IEC 仅增加 2 个 per-layer 标量。

  关键设计动机映射：
  - Baseline 对称量化零点固定 → ICQ 引入可搜索 calibration constant τ，通过熵最大化释放量化器的信息保留灵活性。
  - Baseline LoRA ℓ₂ 无法访问原始输入 → IEC U₁ 的分组平均连接使 ℓ₁ 输出融合原始信息。
  - Baseline LoRA 变换形式同质 → IEC U₂ 的重复拼接引入参数无关的多样化变换。
