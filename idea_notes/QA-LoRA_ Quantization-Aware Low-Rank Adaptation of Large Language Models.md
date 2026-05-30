## QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models

- baseline方法是什么？
  - Baseline 方法：QLoRA（Dettmers et al. 2023a）——将预训练权重从 FP16 量化为 NF4 精度，在 NF4 精度上添加 LoRA 适配器 (A, B) 进行微调。微调后将 LoRA 权重 s·AB 加回量化权重 W̃，使最终模型恢复为 FP16 精度。若想获得量化推理模型，需对合并后的 FP16 模型做 PTQ（如 GPTQ），导致不可控的精度损失。QLoRA 仅解决了微调阶段的显存节省问题，推理阶段仍需 FP16 或承受 PTQ 精度损失。
  - 全栈执行例子（Baseline: QLoRA on LLaMA-7B, Alpaca, INT4 inference）：
    - **算法pipeline**：预训练权重 W_{FP16} 经由 bitsandbytes 量化为 NF4 格式 → 添加 LoRA A∈R^{D_in×r}, B∈R^{r×D_out} 随机初始化 → 加载 Alpaca 52K 数据，以 frozen NF4 W 和可训练 AB 做 LM 交叉熵损失微调 → 微调后合并 W' = W̃_{NF4} + s·AB → 反量化回 FP16 → 需要时再做 GPTQ 后训练量化（INT4），此步骤 PTQ 产生的量化误差无法通过微调补偿，在低比特（INT3/INT2）下尤其严重（如 LLaMA-7B INT2: MMLU 5-shot 仅 25.0-25.8%）。
    - **系统框架**：HuggingFace Transformers + PEFT + bitsandbytes，LoRA rank r（论文未明确说明具体值），Tesla V100 GPU。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：NF4 缺乏 CUDA 算子优化，INT4 有 CUDA 优化的矩阵乘法算子。NF4 微调速度慢于 INT4。
    - **硬件架构**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QA-LoRA 引入分组操作（group-wise operators），通过两个核心设计同时解决 QLoRA 的两个缺陷：
    1. **分组量化增加量化自由度**：将每列权重 W_{:,j} 划分为 L 组，每组 g = D_in/L 个元素独立计算 α_{l,j} 和 β_{l,j}，量化参数从 D_out 对增至 L×D_out 对，显著降低量化误差。
    2. **分组聚合减少适应自由度**：对输入 x 按组求和聚合 A(x)（维度从 D_in 降为 L），LoRA 矩阵 A 从 D_in×D_int 缩小为 L×D_int（L << D_in），行向量在组内共享。这使得 s·AB 的每列 c_{i,j} 在组内为常数，满足合并后仍可表示为 INT 量化格式的数学条件。
    3. **合并推理保持 INT**：微调后只需更新零点矩阵 β'_{l,j} = β_{l,j} - s·(Σ b_{mid,j}·a_{l,mid})/α_{l,j}，无需反量化到 FP16 也无需 PTQ，推理直接使用 INT 格式。
  - 全栈执行例子（QA-LoRA on LLaMA-7B, Alpaca, INT4）：
    - **算法pipeline**：预训练权重 W 通过 GPTQ 以组大小 g=32 进行 INT4 分组不对称量化（act-order=false, true-sequential=true）→ 每组 g 个元素有独立 (α_{l,j}, β_{l,j}) → 初始化 LoRA A∈R^{L×r}, B∈R^{r×D_out}，L = D_in/g = 4096/32 = 128（相比 baseline 的 D_in=4096 减少 32×）→ 前向传播 y = W̃x + s·(A(x)·g)·A^T B^T, 其中 A(x) 将每组 g 个元素求和降维至 L → 微调后合并，仅更新 β' = β - s·(BA)⊘α → 合并后权重 W' 保持 INT4 格式，推理执行 INT4 矩阵乘法（有 CUDA 算子加速），无需 PTQ。INT4 下 MMLU 5-shot 39.4%（与 QLoRA FP16 的 38.4% 相当甚至更优），INT2 下 27.5%（远超 QLoRA w/ GPTQ 的 25.0%）。
    - **系统框架**：HuggingFace Transformers + PEFT，Tensor V100 GPU（7B 单卡 / 65B 双卡），微调时间仅为 QLoRA 的约 35-55%（7B: 21.5h vs 40.0h；65B: 100.5h vs 284.5h）。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：利用 CUDA 优化的 INT4 矩阵乘法算子（vs QLoRA 的 NF4 无优化算子），推理阶段 INT4 GEMM 比 QLoRA FP16 推理快 50% 以上。
    - **硬件架构**：论文未明确说明。

- baseline方法是什么？
  - **Baseline 方法**：标准后训练量化（PTQ：AWQ、AQLM）和量化感知训练/微调（QAT：LLM-QAT、QLoRA），使用 INT4/INT8 位宽，搭配不同风险等级的校准数据集（Risk-I: UltraChat benign、Risk-II: AOA indirectly harmful、Risk-III: AdvBench directly harmful）进行量化。量化目标以效用（utility）为中心，不专门考虑安全。
  - **全栈执行例子（Baseline: QLoRA INT4 + Risk-I benign calibration on Llama-2-7B-Chat）**：
    - **算法pipeline**：QLoRA 将全精度 Llama-2-7B-Chat 权重 W∈R^{d_in×d_out} 量化到 NF4/FP4 精度得到 Q⁰，再通过 LoRA 低秩适配矩阵 (A∈R^{d_in×r}, B∈R^{r×d_out}) 在 benign 数据集（UltraChat）上微调，损失为因果语言模型标准交叉熵 L_LM = -E log p(y|x)，只优化 A、B 而冻结 Q⁰。量化过程中权重被整体修改以最小化效用损失，但安全相关的权重子空间未得到特殊保护。ASR 从 0.3%（FP16）飙升至 42.3%（INT4 Risk-I），MT-Bench 从 6.65 降至 6.40。
    - **系统框架**：HuggingFace Transformers + bitsandbytes 库实现 4-bit 量化，PyTorch 训练循环。4× A100 40GB GPU 上进行微调，batch inference 输出。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：论文未明确说明。使用标准 PyTorch CUDA kernel，未引入自定义量化 kernel。
    - **硬件架构**：NVIDIA A100 40GB GPU × 4。无定制硬件。
  - **Baseline 缺陷**：(i) 所有以效用为中心的量化方法都会损害安全能力——AWQ（PTQ w/o FT）使 ASR 从 0.3% 升至 42.4%（INT4），QLoRA（QAT w/ LoRA）在 benign 数据集上 ASR 即已达 42.3%，harmful 数据集下更飙升至 85.3%；(ii) 低 bit-width（INT4 vs INT8）导致更严重的安全退化——3-bit 和 2-bit 下 ASR 可达 67.3% 和 82.0%；(iii) 校准数据集若包含有害样本（Risk-II/III），安全风险急剧放大——AQLM 在 Risk-III 上 ASR=77.4%，远超 Risk-I 的 18.5%；(iv) 现有的安全对齐方法（SFT/DPO 全量微调）虽能恢复安全，但计算开销大——SFT 需 8.4 GPU-hours，DPO 需 9.6 GPU-hours。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法**：Q-resafe——量化感知的安全修补框架，通过三个核心设计有针对性地恢复量化 LLM 的安全能力，同时保持效用的最小损失：(1) 利用预量化全精度 LLM 的强安全能力作为教师，自动构建 DPO 偏好数据集（y_w 来自全精度模型，y_l 来自量化模型），实现安全知识的蒸馏迁移；(2) 使用 SNIP 分数周期性地识别仅 top-τ% 的安全关键权重（基于连接敏感性 |W_ij · ∇_{Q_ij} L|），而非更新全部权重；(3) 在 LoRA 低秩结构约束下，仅对安全关键权重进行选择性 DPO 更新，其余权重量化后保持不变，避免破坏量化模型的效用。
  - **全栈执行例子（Q-resafe on AWQ INT4 Llama-2-7B-Chat, τ=0.6, r=128, K=1000, benign UltraChat）**：
    - **算法pipeline**：
      1. 数据集构造：对 UltraChat 的每个 prompt x，分别用全精度 Llama-2-7B-Chat（ASR=0.3%）和 AWQ INT4 量化版（ASR=42.4%）生成响应 y_w 和 y_l，构建 200k DPO 三元组。
      2. 周期性安全权重识别（每 K=1000 步）：对当前权重 Q^t 的每层计算 SNIP score = E_x|W_ij · ∇_{Q_ij}(-log p(y|x))|，排序取前 60% 生成 M_Q。该步骤确保随训练进行，安全关键权重子集随模型更新而动态调整。
      3. 选择性 DPO 更新：L_DPO = -log σ(β[log(π_Q(y_w|x)/π_Q⁰(y_w|x)) - log(π_Q(y_l|x)/π_Q⁰(y_l|x))])，仅对 M_A、M_B 掩码位置的 LoRA 参数做 SGD 更新 A^{t+1} = M_A ⊙ (A^t - η∇_A L) + (1-M_A) ⊙ A^t，其余保持零初始化不变。
      4. LoRA 更新后重新量化为 INT4：Q^{t+1} = Q⁰ + Quant(A^{t+1}B^{t+1})，确保修补后模型仍为 INT4 精度，保持内存效率（model size 从 12.6GB FP16 降至 ~2.8-3.5GB）。
      5. 结果：ASR 从 42.4% 降至 1.8%（τ=0.6），MT-Bench 从 6.40 升至 7.14（甚至超过 FP baseline 6.65），仅需 1.2 GPU-hours（vs SFT 8.4h / DPO 9.6h）。在 Risk-III（直接有害数据集）上 ASR 仅 13.6% vs baseline QLoRA 的 85.3%。
    - **系统框架**：PyTorch + HuggingFace Transformers + bitsandbytes。训练使用 4× A100 40GB GPU。推理时量化模型直接加载使用，无需额外全精度模型。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：论文未明确说明。Q-resafe 在算法层面操作，不涉及自定义 kernel。
    - **硬件架构**：NVIDIA A100 40GB GPU × 4。无定制硬件。
  - **Baseline 缺陷 → 方法设计映射**：
    - (i) 效用为中心的量化损害安全 → 方法引入预量化模型的安全知识蒸馏（y_w vs y_l 偏好对），将安全明确作为优化目标而非副作用。
    - (ii) 低 bit-width 加剧安全退化 → 方法只更新极小部分权重（τ=0.6 时仅 60% 权重参与但通过 LoRA 低秩分解实际参数量极少），不扰动大量已校准的量化权重，在 INT4 下即可将 ASR 从 42.4% 恢复到 1.8%，在 2-bit 下仍能维持 ASR=12.4%（vs QLoRA 82.0%）。
    - (iii) 有害校准数据放大风险 → 方法使用 benign 校准集构建 D_patch（y_w 来自安全的全精度模型），即使校准数据包含有害样本也不直接用于权重优化。
    - (iv) 全量安全微调开销大 → 方法通过 SNIP 识别 + LoRA 低秩更新将 GPU-hours 从 8.4h(SFT)/9.6h(DPO) 压缩至 1.2h（~7-8× 加速），且模型大小保持在量化水平（2.8-3.5GB vs FP16 12.6GB）。
