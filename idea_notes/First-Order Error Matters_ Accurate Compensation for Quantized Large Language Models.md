## First-Order Error Matters: Accurate Compensation for Quantized Large Language Models

- baseline方法是什么？
  Baseline 是 **GPTQ**（Frantar et al. 2022），一种基于 OBS→OBC 理论演进的经典 PTQ 方法。GPTQ 全栈执行例子（Llama3-8B, W3A16, group_size=128, A800 GPU）：

  - **算法pipeline**：校准数据 X（128 samples, seq_len=2048）→ 对每层权重 W：计算 H = XX^T → Cholesky 分解 H^{-1}=LL^T，保留上三角 T=L^T → 按 block（B列）逐列量化：(a) Q_{:,j} ← quant(W_{:,j})（RTN 量化）；(b) 补偿误差 δw = −(w_q − ŵ_q)/T_{qq} · T_{q,q:}（仅二阶项）；(c) lazy update 批量更新后续列。核心假设：全精度模型已收敛到局部最优，一阶梯度 ≈ 0，可省略。
  - **系统框架**：自实现 PTQ 脚本（PyTorch），无 Serving 框架修改。量化后导出为 GPTQ 格式，部署至 vLLM 推理。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：标准 PyTorch CUDA kernel（torch.matmul、torch.nn.functional.linear），无自定义 kernel。vLLM 端使用其内置的 W4A16 GEMM kernel。
  - **硬件架构**：NVIDIA A800-80GB GPU，无自定义硬件。

  **Baseline 的核心缺陷**：
  1. **一阶项被错误忽略**：GPTQ 沿袭 OBD/OBS 假设"模型已充分优化→一阶梯度为0"，但逐列量化过程中，先量化列的补偿项 δw 持续更新后续列，导致 latent weights W 与原始 full-precision weights 𝕎 产生累积偏差。此偏差在后续列的损失函数 Taylor 展开中引入不可忽略的一阶梯度 g = ∂E/∂w，GPTQ 的纯二阶近似式 δw = −(w_q − ŵ_q)/T_{qq} · T_{q,q:} 在存在非零梯度下不再是理论最优解。
  2. **GPTAQ 的高开销替代方案不理想**：GPTAQ 尝试通过非对称校准改善量化，但引入了显著额外计算（Llama3-8B 量化时间从 825.50s 增至 1112.20s，+34.7%），且精度提升不稳定。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **FOEM（First-Order Enhanced Method）**，在 GPTQ 补偿框架中显式引入一阶梯度项：

  **(1) 保留一阶项重新推导最优补偿**
  从完整 Taylor 展开 δE = gδw^T + ½δwHδw^T 出发，构建带约束 Lagrangian（约束条件：e_q δw^T + w_q − ŵ_q = 0），求导得理论最优：
  δw = −(w_q − ŵ_q − gH^{-1}e_q^T)/[H^{-1}]_{qq} · [H^{-1}]_{q,:} − gH^{-1}
  对比 GPTQ 的 δw = −(w_q − ŵ_q)/T_{qq} · T_{q,q:}，多了梯度相关的分子修正项和整体梯度项。

  **(2) 梯度近似消除计算开销**
  直接反向传播求 g 开销巨大。FOEM 利用 Taylor 展开近似：
  g(W) ≈ g(𝕎) + (W − 𝕎)H ≈ (W − 𝕎)H（因 g(𝕎)≈0，全精度模型已训练到最优）
  引入稳定化因子 β=0.1：g ≈ β(W − 𝕎)H
  将近似代入理论解后，H 和 H^{-1} 在代数运算中**自动消去**，最终补偿项：
  δw = −((w_q − ŵ_q) − β(w_q − 𝕎e_q^T))/T_{qq} · T_{q,q:} − β(W − 𝕎)
  仅需 T（Cholesky 因子）和权重差分运算，无矩阵乘法，无 Hessian 显式求逆。

  **(3) 全栈执行对比（Llama3-8B, W3A16, A800）**
  - **算法pipeline**：流程同 GPTQ，但每列补偿时额外：(a) 计算权重偏差 W_{:,j} − 𝕎_{:,j}；(b) 分子中减去 β(W_{:,j} − 𝕎_{:,j})；(c) 补偿完当前列后，全局减去 β(W_{:,j} − 𝕎_{:,j})。这些差分运算 O(n) 量级，量化时间 828.90s vs GPTQ 825.50s（仅 +0.4%），远优于 GPTAQ 的 1112.20s（+34.7%）。
  - **系统框架**：同 GPTQ，量化为 GPTQ 格式后部署 vLLM。W4A16 推理：input tokens/s 从 FP16 的 184.11 → 250.26（+36%），output tokens/s 从 470.11 → 616.01（+31%）。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：标准 PyTorch CUDA kernel + vLLM 内置量化 kernel，无自定义 kernel。
  - **硬件架构**：NVIDIA A800-80GB GPU，无自定义硬件。

  **关键结果**：
  - W3A16 Llama3-8B：WikiText2 PPL 从 GPTQ 9.86 → FOEM 8.32（↓15.6%），MMLU 从 GPTAQ 53.8% → FOEM 56.1%
  - W4A4KV4 Llama3-8B：WikiText2 PPL 从 GPTQ 8.55 → FOEM 8.35（↓0.20）
  - 跨架构泛化：Mamba-1.4B（SSM）W3A16 PPL 从 GPTAQ 14.10 → FOEM 13.91
