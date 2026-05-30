## PB-LLM Partially Binarized Large Language Models

- baseline方法是什么？
  Baseline 是将已有的网络二值化方法（BNN, XNOR, Bi-Real, ReCU, FDA）直接应用于 LLM 量化的方案，以及传统的 uniform quantization 方法（RTN）。全栈执行例子：
  - **算法pipeline**：已有的 binarization 方法（如 XNOR-Net）使用 sign 函数将所有权重二值化为 ±1，乘以 channel-wise scaling factor（L1 norm 平均）；或 RTN 直接 round-to-nearest 量化到目标 bit-width。但这些方法在 LLM 上完全崩溃——BNN/XOR/Bi-Real/ReCU/FDA 二值化后的 OPT-1.3B 在 7 个零样本常识推理任务上的平均准确率（0.30-0.32）低于随机猜测（0.36）。原因是 LLM 中存在少量对模型容量至关重要的 salient weights（显著权重），全部二值化会导致这些关键权重的信息完全丢失。已有的 LLM 量化方法（如 GPTQ）在 4-bit 以下也出现显著的性能退化。
  - **系统框架**：论文未明确说明。baseline 使用标准 PyTorch 训练流程，无特殊的分布式或 serving 框架修改。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：论文未明确说明。理论上一值化权重可将 FP 乘法替换为 bitwise XNOR+Bitcount 操作，但论文主要关注 memory 压缩（memory-bound LLM inference）而非 compute kernel 加速。
  - **硬件架构**：论文未明确说明。使用标准 GPU 训练和推理。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  PB-LLM 提出**部分二值化**（Partially-Binarized LLM）策略，核心思想是识别并保留少量 salient weights 在高位宽，其余权重二值化。这源自一个关键发现：LLM 中存在少量显著权重对容量至关重要，全部二值化会完全丢失这些信息。PB-LLM 在 PTQ 和 QAT 两种框架下分别实现了这一思路。

  **具体设计如何解决 Baseline 缺陷**：

  1. **Salient Weight Detection + 保留（解决全部二值化崩溃问题）**：Baseline 的所有权重一视同仁地二值化，导致 LLM 完全崩溃（< random guess）。PB-LLM 通过 magnitude（QAT）或 Hessian metric v_i = w_i^2/[H^{-1}]_{ii}^2（PTQ）检测权重矩阵中的 salient weights，保留 5%-30% 为高比特（如 INT8），剩余才二值化为 ±1。图 6 显示即使 50% salient（等效 ~5-bit）且无训练的 OPT-1.3B 仍有 PPL ~20（非崩溃），证明了 salient weights 对 LLM 容量的关键性。

  2. **PB-GPTQ 解决 PTQ 中的二值化误差传播问题**：Baseline RTN 直接逐列二值化/量化，量化误差在列间累积导致最终输出严重偏离。PB-GPTQ 将 GPTQ 的 Hessian 引导误差补偿扩展到部分二值化：每量化一列后，将该列的量化误差通过 Hessian 矩阵加权补偿到剩余未量化列，使得后续列的量化可以在"误差已校正"的权重基础上进行。Table 1 显示 PB-GPTQ 相比 RTN 在 10% salient 时将 PPL 从 4889 降至 895（Magnitude），从 7508 降至 165（Hessian）。

  3. **Salient Weights Frozen 解决训练困难问题**：LLM QAT（如 LLM-QAT）即使只做 4-bit 量化也需要 100K iterations。PB-LLM 的 QAT 通过冻结 salient weights（不参与梯度更新），仅优化 binary weights 的 FP latent，将训练迭代数从 100K 降至 1-10K（图 7 上半部分：30% salient，10K iters 即可恢复性能）。图 5 训练曲线显示冻结 2% salient 权重就能显著加速收敛。

  4. **Optimal Scaling Factor 闭式解解决手工/搜索 scaling factor 次优问题**：Baseline XNOR-Net 的 L1 norm scaling 和 AWQ 的 grid search 分别有近似误差和搜索成本问题。PB-LLM 从 L2 误差最小化出发解析推导 α* = ||w_F||_1/n（当 w̄_B = sign(w_F) 时），无需任何搜索，且在 column-wise 粒度上做到最优。反直觉的是，仅凭 Salient Frozen + Optimal Scaling 两个机制直接应用于未训练的 LLM 就能维持一定语言能力（图 6）。

  **论文方法全栈执行例子（LLaMA-7B QAT，10% salient，等效 ~1.7 bit）**：
  - **算法pipeline**：加载 LLaMA-7B FP16 checkpoint → 对每个 Linear 层按 |W| 排序选 top-10% salient weights → freeze salient weights（INT8 MinMax quantize）→ 剩余 90% 权重：正向 sign(W_F^{unsal}) 二值化 + α* = mean(|W_F^{unsal}|) column-wise scaling → STE 反向传播更新 FP latent W_F^{unsal} → AdamW, lr=2e-5, cosine decay, 10K iters, 每个 GPU batch=1 → 训练数据 RedPajama-simple-1B → 最终得到 partially-binarized LLaMA-7B → 推理时存储 W^{sal}(INT8) + sign(W_F^{unsal})(binary) + α* scaling factors + bitmap index → 总存储 ≤ 1 * 0.9 + 8 * 0.1 + 1 ≈ 2.7 bit/weight。
  - **系统框架**：PyTorch + HuggingFace Transformers → 标准 GPU 训练（论文未明确 GPU 型号）→ `model.generate()` 推理。与标准 LLM 推理流程一致，仅权重矩阵从 FP16 替换为 mixed-precision（INT8 salient + binary unsalient + scaling factors）。
  - **编译框架**：论文未明确说明。部分二值化矩阵可受益于 bitwise XNOR+Bitcount kernel 加速（理论 64x vs FP multiply），但论文未实现此优化。
  - **kernel调度**：论文未明确说明。论文主要聚焦 memory 压缩（binary weights 在显存中占用极少），不涉及自定义 kernel 实现。
  - **硬件架构**：论文未明确说明具体 GPU 型号。推理时 GPU 需将 binary weights 和 INT8 salient weights 反量化为 FP16 进行矩阵乘法（或未来利用 bitwise 操作加速）。

  **核心优势**：将 LLM 量化推至接近 1-bit（部分二值化）+ PTQ/QAT 双框架灵活选择 + Salient Freeze + Optimal Scaling 闭式解双机制加速训练 + 解析最优而非搜索 scaling factor + 训练效率远超已有 LLM QAT 方法。
